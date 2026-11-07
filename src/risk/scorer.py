"""Rule-based clause risk scoring driven by ``configs/risk_rules.yaml``."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd
import yaml

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "risk_rules.yaml"
LEVELS = ("Low", "Medium", "High")
CLAUSE_REGISTER_COLUMNS = [
    "clause_id",
    "contract_id",
    "chunk_id",
    "category",
    "evidence_text",
    "signal_scores",
    "risk_score",
    "risk_level",
    "rules_version",
    "run_id",
]


def load_rules(path: Union[str, Path] = DEFAULT_RULES_PATH) -> Dict:
    """Load and sanity-check the risk rules config."""
    with Path(path).open(encoding="utf-8") as f:
        rules = yaml.safe_load(f)
    total_weight = sum(signal["weight"] for signal in rules["signals"].values())
    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(f"signal weights must sum to 1.0; got {total_weight}")
    thresholds = rules["thresholds"]
    if not 0 <= thresholds["medium"] <= thresholds["high"] <= 1:
        raise ValueError("thresholds must satisfy 0 <= medium <= high <= 1")
    return rules


def risk_level(score: float, rules: Dict) -> str:
    """Map a 0-1 risk score to Low / Medium / High."""
    if score >= rules["thresholds"]["high"]:
        return "High"
    if score >= rules["thresholds"]["medium"]:
        return "Medium"
    return "Low"


def clause_signals(
    category: str,
    probability: float,
    evidence_text: Optional[str],
    contract_categories: set,
    rules: Dict,
) -> Dict[str, float]:
    """Score one detected clause on each of the four signals, each in 0-1."""
    tiers = rules["category_tiers"]
    tier = tiers.get(category, tiers["default"])

    text = (evidence_text or "").lower()
    keyword_rules = rules["keywords"]
    keywords = keyword_rules["global"] + keyword_rules["by_category"].get(category, [])
    has_keyword = category not in keyword_rules.get("skip_categories", []) and any(
        keyword.lower() in text for keyword in keywords
    )

    exception = rules["protective_exceptions"].get(category)
    missing_exception = exception is not None and exception not in contract_categories

    return {
        "category_tier": float(rules["tier_values"][tier]),
        "model_confidence": float(probability),
        "restrictive_keywords": float(has_keyword),
        "missing_exception": float(missing_exception),
    }


def score_clauses(predictions: pd.DataFrame, rules: Dict, run_id: str) -> pd.DataFrame:
    """Turn positive predictions into a clause register.

    ``predictions`` needs ``contract_id``, ``category``, ``probability`` and
    ``predicted_label``, as in the trainer's prediction tables. ``chunk_id`` and
    ``evidence_text`` are optional; without evidence text the keyword signal
    scores 0. Only rows with ``predicted_label`` set are scored.
    """
    detected = predictions[predictions["predicted_label"].astype(bool)].copy()
    for column in ("chunk_id", "evidence_text"):
        if column not in detected:
            detected[column] = None
    if detected.empty:
        return pd.DataFrame(columns=CLAUSE_REGISTER_COLUMNS)

    # Signal 4 looks at the whole contract, not just this clause's chunk.
    categories_by_contract = detected.groupby("contract_id")["category"].agg(set).to_dict()
    weights = {name: signal["weight"] for name, signal in rules["signals"].items()}

    rows = []
    for row in detected.itertuples(index=False):
        signals = clause_signals(
            row.category,
            row.probability,
            row.evidence_text,
            categories_by_contract[row.contract_id],
            rules,
        )
        score = sum(weights[name] * value for name, value in signals.items())
        clause_parts = [row.contract_id, row.chunk_id, row.category]
        rows.append({
            "clause_id": "::".join(str(part) for part in clause_parts if part is not None),
            "contract_id": row.contract_id,
            "chunk_id": row.chunk_id,
            "category": row.category,
            "evidence_text": row.evidence_text,
            "signal_scores": json.dumps(signals),
            "risk_score": round(score, 6),
            "risk_level": risk_level(score, rules),
            "rules_version": rules["version"],
            "run_id": run_id,
        })
    return pd.DataFrame(rows, columns=CLAUSE_REGISTER_COLUMNS)


def gold_predictions(contract_categories: pd.DataFrame) -> pd.DataFrame:
    """Build a perfect-model prediction table from CUAD's human labels.

    Lets the risk engine be built and tested before the classifier is final:
    every annotated category gets probability 1.0, and its answer spans become
    the evidence text.
    """
    present = contract_categories[contract_categories["present"].astype(bool)]
    evidence = present["answers"].apply(lambda answers: " ".join(a["text"] for a in answers))
    return pd.DataFrame({
        "contract_id": present["contract_id"].to_numpy(),
        "category": present["category"].to_numpy(),
        "probability": 1.0,
        "predicted_label": True,
        "evidence_text": evidence.to_numpy(),
    })
