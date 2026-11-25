import json

import pandas as pd
import pytest

from src.db import store
from src.db.runs import model_run_info, register_model_run, run_id_for
from src.db.schema import TABLES
from src.risk.evidence import extract_evidence
from src.risk.scorer import load_rules, score_clauses
from src.risk.triage import triage_contracts

TEXT = "Party A shall not compete worldwide. Either party may terminate on notice."


@pytest.fixture
def conn():
    connection = store.connect(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def rules():
    return load_rules()


def chunks_for(contract_id):
    return pd.DataFrame({
        "chunk_id": [f"{contract_id}__0", f"{contract_id}__1"],
        "chunk_index": [0, 1],
        "start_char": [0, 37],
        "end_char": [36, len(TEXT)],
        "n_tokens": [8, 9],
        "text": [TEXT[0:36], TEXT[37:]],
        "text_light": [TEXT[0:36], TEXT[37:]],
    })


def predictions_for(contract_id, non_compete=0.9):
    return pd.DataFrame({
        "chunk_id": [f"{contract_id}__0", f"{contract_id}__1"],
        "category": ["Non-Compete", "Non-Compete"],
        "probability": [non_compete, 0.1],
        "predicted": [non_compete >= 0.5, False],
    })


def add_contract(conn, contract_id, run_id="run-a", non_compete=0.9):
    store.upsert_contract(conn, contract_id, title=contract_id, split="test", n_chars=len(TEXT))
    store.replace_chunks(conn, contract_id, chunks_for(contract_id))
    store.replace_predictions(conn, contract_id, run_id, predictions_for(contract_id, non_compete))


def add_run(conn, run_id="run-a"):
    store.upsert_model_run(conn, {"run_id": run_id, "model_name": "m", "threshold": 0.5,
                                  "metadata_json": {"threshold": 0.5}})


def score(conn, contract_ids, run_id, rules):
    evidence = extract_evidence(store.read_predictions(conn, run_id, contract_ids),
                                store.read_chunks(conn, contract_ids))
    return score_clauses(evidence, rules, run_id=run_id)


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_connect_creates_every_table(conn):
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(TABLES) <= names


def test_connect_creates_file_and_parent(tmp_path):
    path = tmp_path / "nested" / "results.db"
    store.connect(path).close()
    assert path.exists()


def test_upsert_contract_twice_keeps_one_row(conn):
    store.upsert_contract(conn, "c1", title="old")
    store.upsert_contract(conn, "c1", title="new")
    contracts = store.list_contracts(conn)
    assert contracts["title"].tolist() == ["new"]


def test_rerunning_ingest_and_infer_does_not_duplicate(conn):
    add_run(conn)
    add_contract(conn, "c1")
    add_contract(conn, "c1")
    assert count(conn, "chunk") == 2
    assert count(conn, "prediction") == 2


def test_read_predictions_matches_evidence_input(conn):
    add_run(conn)
    add_contract(conn, "c1")
    frame = store.read_predictions(conn, "run-a")
    assert list(frame.columns) == ["contract_id", "chunk_id", "category", "probability",
                                   "predicted_label"]
    assert frame["predicted_label"].dtype == bool
    assert frame["predicted_label"].tolist() == [True, False]


def test_reingest_cascades_to_predictions_and_clauses(conn, rules):
    add_run(conn)
    add_contract(conn, "c1")
    clauses = score(conn, ["c1"], "run-a", rules)
    store.replace_clause_register(conn, ["c1"], "run-a", rules["version"], clauses)
    assert count(conn, "clause_register") == 1

    store.replace_chunks(conn, "c1", chunks_for("c1"))
    assert count(conn, "prediction") == 0
    assert count(conn, "clause_register") == 0


def test_clause_rows_carry_run_rules_and_offsets(conn, rules):
    add_run(conn)
    add_contract(conn, "c1")
    clauses = score(conn, ["c1"], "run-a", rules)
    store.replace_clause_register(conn, ["c1"], "run-a", rules["version"], clauses)

    stored = store.read_clause_register(conn)
    assert len(stored) == 1
    row = stored.iloc[0]
    assert row["run_id"] == "run-a"
    assert row["rules_version"] == rules["version"]
    assert (row["start_char"], row["end_char"]) == (0, 36)
    assert TEXT[row["start_char"]:row["end_char"]] == row["evidence_text"]
    assert set(json.loads(row["signal_scores"])) == set(rules["signals"])


def test_new_rules_version_adds_rows_instead_of_overwriting(conn, rules):
    add_run(conn)
    add_contract(conn, "c1")
    clauses = score(conn, ["c1"], "run-a", rules)
    store.replace_clause_register(conn, ["c1"], "run-a", rules["version"], clauses)

    newer = {**rules, "version": "9.9.9-test"}
    store.replace_clause_register(conn, ["c1"], "run-a", "9.9.9-test",
                                  score(conn, ["c1"], "run-a", newer))
    store.replace_clause_register(conn, ["c1"], "run-a", "9.9.9-test",
                                  score(conn, ["c1"], "run-a", newer))
    versions = store.read_clause_register(conn)["rules_version"].value_counts().to_dict()
    assert versions == {rules["version"]: 1, "9.9.9-test": 1}


def test_rescoring_clears_clauses_that_are_no_longer_detected(conn, rules):
    add_run(conn)
    add_contract(conn, "c1")
    store.replace_clause_register(conn, ["c1"], "run-a", rules["version"],
                                  score(conn, ["c1"], "run-a", rules))
    store.replace_predictions(conn, "c1", "run-a", predictions_for("c1", non_compete=0.2))
    store.replace_clause_register(conn, ["c1"], "run-a", rules["version"],
                                  score(conn, ["c1"], "run-a", rules))
    assert count(conn, "clause_register") == 0


def test_clause_rows_must_match_the_replaced_scope(conn, rules):
    add_run(conn)
    add_contract(conn, "c1")
    clauses = score(conn, ["c1"], "run-a", rules)
    with pytest.raises(ValueError):
        store.replace_clause_register(conn, ["c2"], "run-a", rules["version"], clauses)
    with pytest.raises(ValueError):
        store.replace_clause_register(conn, ["c1"], "run-a", "other-version", clauses)


def test_triage_ranks_contracts_ingested_one_at_a_time(conn, rules):
    add_run(conn)
    add_contract(conn, "risky")
    add_contract(conn, "quiet", non_compete=0.1)
    ids = ["quiet", "risky"]
    clauses = score(conn, ids, "run-a", rules)
    store.replace_clause_register(conn, ids, "run-a", rules["version"], clauses)

    triage = triage_contracts(store.read_clause_register(conn, "run-a", rules["version"]),
                              rules, contract_ids=ids)
    store.replace_triage(conn, "run-a", rules["version"], triage)
    store.replace_triage(conn, "run-a", rules["version"], triage)

    stored = store.read_triage(conn, "run-a", rules["version"])
    assert stored["contract_id"].tolist() == ["risky", "quiet"]
    assert stored["review_rank"].tolist() == [1, 2]
    assert stored["run_id"].eq("run-a").all()
    assert stored["rules_version"].eq(rules["version"]).all()


def test_predictions_need_a_registered_run(conn):
    store.upsert_contract(conn, "c1")
    store.replace_chunks(conn, "c1", chunks_for("c1"))
    with pytest.raises(Exception):
        store.replace_predictions(conn, "c1", "unregistered", predictions_for("c1"))


# --- run identity -------------------------------------------------------------

@pytest.fixture
def run_dir(tmp_path):
    directory = tmp_path / "distilroberta-cuad"
    directory.mkdir()
    (directory / "run_metadata.json").write_text(json.dumps(
        {"model_name": "distilroberta-base", "threshold": 0.4, "categories": ["Non-Compete"]}))
    (directory / "classifier.pt").write_bytes(b"head-v1")
    pd.DataFrame({"category": ["Non-Compete"], "precision": [0.8], "recall": [0.6],
                  "f1": [0.69], "support": [10], "predicted_positive": [7]}
                 ).to_csv(directory / "test_category_metrics.csv", index=False)
    return directory


def test_run_id_is_stable_and_named_after_the_folder(run_dir):
    assert run_id_for(run_dir) == run_id_for(run_dir)
    assert run_id_for(run_dir).startswith("distilroberta-cuad-")


def test_retrained_head_gets_a_new_run_id(run_dir):
    before = run_id_for(run_dir)
    (run_dir / "classifier.pt").write_bytes(b"head-v2")
    assert run_id_for(run_dir) != before


def test_model_run_info_reads_threshold(run_dir):
    info = model_run_info(run_dir)
    assert info["threshold"] == 0.4
    assert info["model_name"] == "distilroberta-base"


def test_register_model_run_stores_run_and_metrics(conn, run_dir):
    run_id = register_model_run(conn, run_dir)
    register_model_run(conn, run_dir)
    run = store.read_model_run(conn, run_id)
    assert run["threshold"] == 0.4
    assert json.loads(run["metadata_json"])["categories"] == ["Non-Compete"]
    metrics = conn.execute("SELECT category, precision_score, support FROM category_metric"
                           ).fetchall()
    assert metrics == [("Non-Compete", 0.8, 10)]
