"""Pipeline stages against an in-memory DB, with a fake model and a fake chunker.

No checkpoint or tokenizer download is needed, except for the one test that
checks chunking against chunks.parquet, which skips if that file is missing.
"""

import json
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import pytest

import src.models.predict as predict
import src.pipeline as pipeline
from src.db import store
from src.risk.scorer import load_rules

CATEGORIES = ["Non-Compete", "Governing Law"]
RISKY = "Seller shall not compete worldwide. This agreement is governed by Delaware law."
QUIET = "This agreement is governed by Delaware law. Notices go to the addresses above."


class FakeModel(NamedTuple):
    categories: list
    threshold: float


def fake_score_texts(texts, loaded, batch_size=8):
    return pd.DataFrame({
        "Non-Compete": [0.9 if "compete" in t else 0.05 for t in texts],
        "Governing Law": [0.8 if "governed" in t else 0.05 for t in texts],
    })[loaded.categories]


def fake_chunk_contract(text):
    """Split on sentence ends; offsets index `text` like the real chunker's."""
    if not text.strip():
        return []                                   # as the real chunker does
    chunks, start = [], 0
    for i, piece in enumerate(text.split(". ")):
        end = start + len(piece)
        chunks.append({"chunk_index": i, "start_char": start, "end_char": end,
                       "n_tokens": len(piece.split()), "text": text[start:end]})
        start = end + 2
    return chunks


@pytest.fixture
def conn():
    connection = store.connect(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def run_dir(tmp_path):
    directory = tmp_path / "fake-run"
    directory.mkdir()
    (directory / "run_metadata.json").write_text(json.dumps(
        {"model_name": "fake", "threshold": 0.5, "categories": CATEGORIES}))
    (directory / "classifier.pt").write_bytes(b"fake")
    return directory


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(pipeline, "chunk_contract", fake_chunk_contract)
    monkeypatch.setattr(predict, "score_texts", fake_score_texts)


@pytest.fixture
def model():
    return FakeModel(CATEGORIES, 0.5)


def contract(contract_id, text, split="test"):
    return {"contract_id": contract_id, "title": contract_id, "split": split,
            "source": "test", "text": text}


def quiet(*args, **kwargs):
    pass


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_ingest_writes_chunks_with_raw_and_clean_text(conn):
    pipeline.ingest(conn, [contract("risky", RISKY)], log=quiet)
    chunks = store.read_chunks(conn)
    assert len(chunks) == 2
    for row in chunks.itertuples():
        assert RISKY[row.start_char:row.end_char] == row.text
        assert len(row.text_light) == len(row.text)


def test_ingest_skips_known_contract_unless_forced(conn, run_dir, model):
    pipeline.run(conn, [contract("risky", RISKY)], run_dir, loaded=model, log=quiet)
    pipeline.ingest(conn, [contract("risky", RISKY)], log=quiet)
    assert count(conn, "prediction") > 0              # skip kept the predictions
    pipeline.ingest(conn, [contract("risky", RISKY)], force=True, log=quiet)
    assert count(conn, "prediction") == 0             # re-chunking invalidates them


def test_ingest_leaves_out_empty_contracts(conn):
    ids = pipeline.ingest(conn, [contract("empty", "   "), contract("risky", RISKY)], log=quiet)
    assert ids == ["risky"]


def test_run_end_to_end(conn, run_dir, model):
    run_id, triage = pipeline.run(
        conn, [contract("risky", RISKY), contract("quiet", QUIET)],
        run_dir, loaded=model, log=quiet)

    assert run_id.startswith("fake-run-")
    assert count(conn, "prediction") == 4 * len(CATEGORIES)   # every chunk x category

    register = store.read_clause_register(conn)
    assert set(register["category"]) == {"Non-Compete", "Governing Law"}
    assert register["run_id"].eq(run_id).all()
    assert register["rules_version"].eq(load_rules()["version"]).all()
    assert register["start_char"].notna().all() and register["end_char"].notna().all()

    assert triage["contract_id"].tolist() == ["risky", "quiet"]
    assert triage["review_rank"].tolist() == [1, 2]
    assert triage["run_id"].eq(run_id).all()


def test_evidence_offsets_point_at_the_source_text(conn, run_dir, model):
    pipeline.run(conn, [contract("risky", RISKY)], run_dir, loaded=model, log=quiet)
    for row in store.read_clause_register(conn).itertuples():
        assert RISKY[row.start_char:row.end_char] == row.evidence_text


def test_infer_does_not_load_model_when_nothing_to_do(conn, run_dir, model, monkeypatch):
    pipeline.run(conn, [contract("risky", RISKY)], run_dir, loaded=model, log=quiet)

    def fail(*args, **kwargs):
        raise AssertionError("model should not load")
    monkeypatch.setattr(predict, "load_model", fail)
    pipeline.infer(conn, run_dir, log=quiet)


def test_stages_run_separately(conn, run_dir, model):
    pipeline.ingest(conn, [contract("risky", RISKY)], log=quiet)
    run_id = pipeline.infer(conn, run_dir, loaded=model, log=quiet)
    assert count(conn, "clause_register") == 0
    triage = pipeline.score(conn, run_id, log=quiet)
    assert triage["contract_id"].tolist() == ["risky"]


def test_score_refuses_contracts_without_predictions(conn, run_dir, model):
    run_id, _ = pipeline.run(conn, [contract("risky", RISKY)], run_dir, loaded=model, log=quiet)
    with pytest.raises(ValueError, match="run infer first"):
        pipeline.score(conn, run_id, contract_ids=["never-ingested"], log=quiet)


def test_rescoring_one_contract_reranks_all(conn, run_dir, model):
    run_id, _ = pipeline.run(
        conn, [contract("risky", RISKY), contract("quiet", QUIET)], run_dir,
        loaded=model, log=quiet)
    triage = pipeline.score(conn, run_id, contract_ids=["quiet"], log=quiet)
    assert triage["contract_id"].tolist() == ["risky", "quiet"]


def test_contracts_are_ranked_within_their_split(conn, run_dir, model):
    """A train contract the model has seen must not push test contracts down."""
    _, triage = pipeline.run(
        conn, [contract("test-quiet", QUIET), contract("test-risky", RISKY),
               contract("train-risky", RISKY + " Seller shall not compete.", split="train"),
               contract("outside", "Nothing to see", split="external")],
        run_dir, loaded=model, log=quiet)

    ranks = triage.set_index("contract_id")[["split", "review_rank"]].to_dict("index")
    assert ranks == {
        "outside": {"split": "external", "review_rank": 1},     # no clauses at all
        "test-quiet": {"split": "test", "review_rank": 2},
        "test-risky": {"split": "test", "review_rank": 1},
        "train-risky": {"split": "train", "review_rank": 1},
    }


def test_new_rules_version_keeps_old_scores(conn, run_dir, model):
    rules = load_rules()
    run_id, _ = pipeline.run(conn, [contract("risky", RISKY)], run_dir, rules,
                             loaded=model, log=quiet)
    pipeline.score(conn, run_id, {**rules, "version": "9.9.9-test"}, log=quiet)
    versions = conn.execute(
        "SELECT DISTINCT rules_version FROM triage_score ORDER BY 1").fetchall()
    assert {v[0] for v in versions} == {rules["version"], "9.9.9-test"}


def test_long_predictions_shape():
    long = pipeline.long_predictions(
        ["a__0", "a__1"], pd.DataFrame({"X": [0.2, 0.7], "Y": [0.5, 0.1]}), threshold=0.5)
    assert len(long) == 4
    assert long.set_index(["chunk_id", "category"])["predicted"].to_dict() == {
        ("a__0", "X"): False, ("a__1", "X"): True, ("a__0", "Y"): True, ("a__1", "Y"): False}


def test_cuad_specs():
    test_docs = pipeline._cuad_documents("test")
    assert [c["source"] for c in pipeline.cuad_contracts("test:1-2")] == [
        "cuad:test:1", "cuad:test:2"]
    first = next(pipeline.cuad_contracts("test:0"))
    assert first["contract_id"] == test_docs[0]["title"]
    assert first["split"] == "test"


CHUNKS_PARQUET = Path(pipeline.ROOT) / "data" / "processed" / "chunks.parquet"


@pytest.mark.skipif(not CHUNKS_PARQUET.exists(), reason="chunks.parquet not built")
def test_chunking_matches_training_data(monkeypatch):
    """The pipeline must chunk a contract exactly as build_dataset.py did."""
    from src.ingest.chunker import chunk_contract
    monkeypatch.setattr(pipeline, "chunk_contract", chunk_contract)

    first = next(pipeline.cuad_contracts("test:0"))
    ours = pipeline.chunk_frame(first["contract_id"], first["text"])
    theirs = pd.read_parquet(CHUNKS_PARQUET, filters=[("contract_id", "==", first["contract_id"])])
    columns = ["chunk_id", "start_char", "end_char", "text", "text_light"]
    pd.testing.assert_frame_equal(
        ours[columns].reset_index(drop=True),
        theirs.sort_values("chunk_index")[columns].reset_index(drop=True),
        check_dtype=False)
