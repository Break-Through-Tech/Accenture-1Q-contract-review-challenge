"""Roll clause risk levels up to a contract triage score and review rank."""

from typing import Dict, Iterable, Optional

import pandas as pd

TRIAGE_SCORE_COLUMNS = [
    "contract_id",
    "triage_score",
    "review_rank",
    "n_high",
    "n_medium",
    "rules_version",
]


def triage_contracts(
    clause_register: pd.DataFrame,
    rules: Dict,
    contract_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Score and rank contracts from most to least critical.

    ``triage_score`` is the sum of ``triage.level_points`` over a contract's
    clauses. Contracts with at least ``always_review_if_high_count`` High
    clauses rank ahead of the rest regardless of score. Pass ``contract_ids``
    to include contracts with no detected clauses; they score 0.
    """
    triage = rules["triage"]
    points = {level.capitalize(): value for level, value in triage["level_points"].items()}

    counts = pd.crosstab(clause_register["contract_id"], clause_register["risk_level"])
    counts = counts.reindex(columns=["High", "Medium", "Low"], fill_value=0)
    if contract_ids is not None:
        counts = counts.reindex(counts.index.union(pd.Index(list(contract_ids))), fill_value=0)
    if counts.empty:
        return pd.DataFrame(columns=TRIAGE_SCORE_COLUMNS)

    result = pd.DataFrame({
        "contract_id": counts.index.astype(str),
        "triage_score": sum(counts[level] * points.get(level, 0) for level in counts.columns).astype(float),
        "n_high": counts["High"].astype(int),
        "n_medium": counts["Medium"].astype(int),
    }).reset_index(drop=True)

    result["_must_review"] = result["n_high"] >= triage["always_review_if_high_count"]
    result = result.sort_values(
        ["_must_review", "triage_score", "n_high", "contract_id"],
        ascending=[False, False, False, True],
    ).drop(columns="_must_review")
    result["review_rank"] = range(1, len(result) + 1)
    result["rules_version"] = rules["version"]
    return result[TRIAGE_SCORE_COLUMNS].reset_index(drop=True)
