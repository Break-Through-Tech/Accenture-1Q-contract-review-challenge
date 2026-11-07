import json

import pandas as pd
import pytest

from src.risk.scorer import (
    CLAUSE_REGISTER_COLUMNS,
    gold_predictions,
    load_rules,
    risk_level,
    score_clauses,
)
from src.risk.triage import TRIAGE_SCORE_COLUMNS, triage_contracts


@pytest.fixture(scope="module")
def rules():
    return load_rules()


def predictions(*rows):
    return pd.DataFrame(
        rows, columns=["contract_id", "category", "probability", "predicted_label", "evidence_text"]
    )


def test_rules_cover_every_signal(rules):
    assert set(rules["signals"]) == {
        "category_tier", "model_confidence", "restrictive_keywords", "missing_exception",
    }


def test_risk_level_boundaries(rules):
    high, medium = rules["thresholds"]["high"], rules["thresholds"]["medium"]
    assert risk_level(high, rules) == "High"
    assert risk_level(medium, rules) == "Medium"
    assert risk_level(medium - 0.01, rules) == "Low"


def test_high_tier_alone_is_not_high(rules):
    register = score_clauses(predictions(("c1", "Ip Ownership Assignment", 1.0, True, "")), rules, "run")
    assert register.loc[0, "risk_level"] == "Medium"


def test_high_tier_with_red_flag_is_high(rules):
    register = score_clauses(predictions(("c1", "Ip Ownership Assignment", 1.0, True, "Licensee hereby assigns all rights.")), rules, "run")
    assert register.loc[0, "risk_level"] == "High"


def test_low_confidence_lowers_risk(rules):
    sure = score_clauses(predictions(("c1", "Ip Ownership Assignment", 1.0, True, "")), rules, "run")
    unsure = score_clauses(predictions(("c1", "Ip Ownership Assignment", 0.5, True, "")), rules, "run")
    assert unsure.loc[0, "risk_score"] < sure.loc[0, "risk_score"]


def test_keyword_raises_risk(rules):
    plain = score_clauses(predictions(("c1", "Termination For Convenience", 1.0, True, "Either party may terminate.")), rules, "run")
    flagged = score_clauses(predictions(("c1", "Termination For Convenience", 1.0, True, "Company may terminate in its SOLE DISCRETION.")), rules, "run")
    assert json.loads(flagged.loc[0, "signal_scores"])["restrictive_keywords"] == 1.0
    assert flagged.loc[0, "risk_score"] > plain.loc[0, "risk_score"]


def test_metadata_ignores_keywords(rules):
    register = score_clauses(predictions(("c1", "Expiration Date", 1.0, True, "may be extended at any time")), rules, "run")
    assert json.loads(register.loc[0, "signal_scores"])["restrictive_keywords"] == 0.0
    assert register.loc[0, "risk_level"] == "Low"


def test_protective_exception_in_same_contract(rules):
    alone = score_clauses(predictions(("c1", "Non-Compete", 1.0, True, "")), rules, "run")
    protected = score_clauses(predictions(
        ("c1", "Non-Compete", 1.0, True, ""),
        ("c1", "Competitive Restriction Exception", 1.0, True, ""),
    ), rules, "run")
    assert json.loads(alone.loc[0, "signal_scores"])["missing_exception"] == 1.0
    assert json.loads(protected.loc[0, "signal_scores"])["missing_exception"] == 0.0


def test_exception_in_other_contract_does_not_protect(rules):
    register = score_clauses(predictions(
        ("c1", "Non-Compete", 1.0, True, ""),
        ("c2", "Competitive Restriction Exception", 1.0, True, ""),
    ), rules, "run")
    assert json.loads(register.loc[0, "signal_scores"])["missing_exception"] == 1.0


def test_unknown_category_uses_default_tier(rules):
    register = score_clauses(predictions(("c1", "Most Favored Nation", 1.0, True, "")), rules, "run")
    default = rules["tier_values"][rules["category_tiers"]["default"]]
    assert json.loads(register.loc[0, "signal_scores"])["category_tier"] == default


def test_negative_predictions_are_not_scored(rules):
    register = score_clauses(predictions(
        ("c1", "Non-Compete", 0.9, True, ""),
        ("c1", "Exclusivity", 0.2, False, ""),
    ), rules, "run")
    assert register["category"].tolist() == ["Non-Compete"]


def test_register_schema_and_stamps(rules):
    register = score_clauses(predictions(("c1", "Audit Rights", 1.0, True, "")), rules, "run-7")
    assert list(register.columns) == CLAUSE_REGISTER_COLUMNS
    assert register.loc[0, "rules_version"] == rules["version"]
    assert register.loc[0, "run_id"] == "run-7"


def test_scoring_is_deterministic(rules):
    preds = predictions(
        ("c1", "Non-Compete", 0.8, True, "worldwide"),
        ("c2", "Insurance", 0.9, True, ""),
    )
    pd.testing.assert_frame_equal(score_clauses(preds, rules, "run"), score_clauses(preds, rules, "run"))


def test_no_detected_clauses(rules):
    register = score_clauses(predictions(("c1", "Non-Compete", 0.1, False, "")), rules, "run")
    assert register.empty
    assert list(register.columns) == CLAUSE_REGISTER_COLUMNS

    triage = triage_contracts(register, rules, contract_ids=["c1"])
    assert triage.to_dict("records") == [{
        "contract_id": "c1", "triage_score": 0.0, "review_rank": 1,
        "n_high": 0, "n_medium": 0, "rules_version": rules["version"],
    }]


def test_empty_input_without_contract_ids(rules):
    register = score_clauses(predictions(), rules, "run")
    triage = triage_contracts(register, rules)
    assert triage.empty
    assert list(triage.columns) == TRIAGE_SCORE_COLUMNS


def test_triage_formula_and_ranking(rules):
    register = pd.DataFrame({
        "contract_id": ["a", "a", "b"] + ["c"] * 7,
        "risk_level": ["High", "High", "Medium"] + ["Medium"] * 7,
    })
    triage = triage_contracts(register, rules, contract_ids=["a", "b", "c", "d"]).set_index("contract_id")
    assert triage.loc["a", "triage_score"] == 6.0  # 2 High * 3
    assert triage.loc["c", "triage_score"] == 7.0  # 7 Medium * 1
    assert triage.loc["d", "triage_score"] == 0.0
    # "a" has 2 High clauses, so it ranks first even though "c" scores more.
    assert triage["review_rank"].sort_values().index.tolist() == ["a", "c", "b", "d"]


def test_one_high_clause_is_not_forced_to_top(rules):
    register = pd.DataFrame({
        "contract_id": ["a"] + ["b"] * 4,
        "risk_level": ["High"] + ["Medium"] * 4,
    })
    triage = triage_contracts(register, rules).set_index("contract_id")
    assert triage["review_rank"].sort_values().index.tolist() == ["b", "a"]


def test_gold_predictions_from_cuad_rows():
    rows = pd.DataFrame({
        "contract_id": ["c1", "c1"],
        "category": ["Non-Compete", "Insurance"],
        "present": [True, False],
        "answers": [[{"text": "shall not compete"}, {"text": "worldwide"}], []],
    })
    preds = gold_predictions(rows)
    assert preds.to_dict("records") == [{
        "contract_id": "c1", "category": "Non-Compete", "probability": 1.0,
        "predicted_label": True, "evidence_text": "shall not compete worldwide",
    }]
