"""Metrics for multi-label chunk classification with nullable labels."""

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    known_mask: np.ndarray,
    categories: Sequence[str],
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute per-category precision/recall/F1/support and macro averages.

    Missing targets are excluded through ``known_mask``. Macro metrics average
    over the selected category scope; zero-support categories receive zero
    scores and remain visible in the table.
    """
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    known_mask = np.asarray(known_mask, dtype=bool)
    expected_shape = (targets.shape[0], len(categories))
    if targets.shape != expected_shape or probabilities.shape != expected_shape or known_mask.shape != expected_shape:
        raise ValueError(
            f"targets, probabilities, and known_mask must all have shape {expected_shape}; "
            f"got {targets.shape}, {probabilities.shape}, {known_mask.shape}"
        )
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    predictions = probabilities >= threshold
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for index, category in enumerate(categories):
        valid = known_mask[:, index]
        truth = targets[valid, index].astype(bool)
        predicted = predictions[valid, index]
        tp = int(np.logical_and(truth, predicted).sum())
        fp = int(np.logical_and(~truth, predicted).sum())
        fn = int(np.logical_and(truth, ~predicted).sum())
        support = int(truth.sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "category": category,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_positive": int(predicted.sum()),
        })
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    metrics = pd.DataFrame(rows)
    micro_precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    micro_recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall else 0.0
    )
    summary = {
        "threshold": float(threshold),
        "n_categories": int(len(categories)),
        "macro_precision": float(metrics["precision"].mean()),
        "macro_recall": float(metrics["recall"].mean()),
        "macro_f1": float(metrics["f1"].mean()),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
    }
    return metrics, summary
