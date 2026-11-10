import pandas as pd
import pytest

from src.risk.evidence import EVIDENCE_COLUMNS, extract_evidence
from src.risk.scorer import load_rules, score_clauses

CONTRACT = "Party A shall not compete worldwide. Either party may terminate on notice."


@pytest.fixture
def chunks():
    return pd.DataFrame({
        "contract_id": ["c1", "c1"],
        "chunk_id": ["c1__0", "c1__1"],
        "start_char": [0, 37],
        "end_char": [36, len(CONTRACT)],
        "text": [CONTRACT[0:36], CONTRACT[37:]],
    })


def chunk_predictions(*rows):
    return pd.DataFrame(rows, columns=["contract_id", "chunk_id", "category", "probability", "predicted_label"])


def test_keeps_highest_probability_chunk(chunks):
    evidence = extract_evidence(chunk_predictions(
        ("c1", "c1__0", "Non-Compete", 0.9, True),
        ("c1", "c1__1", "Non-Compete", 0.6, True),
    ), chunks)
    assert len(evidence) == 1
    assert evidence.loc[0, "chunk_id"] == "c1__0"
    assert evidence.loc[0, "probability"] == 0.9


def test_offsets_point_at_evidence_text(chunks):
    evidence = extract_evidence(chunk_predictions(("c1", "c1__1", "Termination For Convenience", 0.8, True)), chunks)
    row = evidence.loc[0]
    assert CONTRACT[row["start_char"]:row["end_char"]] == row["evidence_text"]


def test_skips_negative_predictions(chunks):
    evidence = extract_evidence(chunk_predictions(
        ("c1", "c1__0", "Non-Compete", 0.9, True),
        ("c1", "c1__1", "Exclusivity", 0.3, False),
    ), chunks)
    assert evidence["category"].tolist() == ["Non-Compete"]


def test_no_positive_predictions(chunks):
    evidence = extract_evidence(chunk_predictions(("c1", "c1__0", "Non-Compete", 0.1, False)), chunks)
    assert evidence.empty
    assert list(evidence.columns) == EVIDENCE_COLUMNS


def test_unknown_chunk_raises(chunks):
    with pytest.raises(ValueError, match="not in the chunks table"):
        extract_evidence(chunk_predictions(("c1", "c1__9", "Non-Compete", 0.9, True)), chunks)


def test_evidence_feeds_scorer(chunks):
    evidence = extract_evidence(chunk_predictions(("c1", "c1__0", "Non-Compete", 0.9, True)), chunks)
    register = score_clauses(evidence, load_rules(), "run")
    row = register.loc[0]
    assert row["chunk_id"] == "c1__0"
    assert row["evidence_text"] == CONTRACT[0:36]
    assert (row["start_char"], row["end_char"]) == (0, 36)
    assert '"restrictive_keywords": 1.0' in row["signal_scores"]  # "worldwide"
