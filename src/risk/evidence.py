"""Attach supporting contract text to each detected clause."""

import pandas as pd

EVIDENCE_COLUMNS = [
    "contract_id",
    "chunk_id",
    "category",
    "probability",
    "predicted_label",
    "start_char",
    "end_char",
    "evidence_text",
]


def extract_evidence(chunk_predictions: pd.DataFrame, chunks: pd.DataFrame) -> pd.DataFrame:
    """Pick one evidence chunk per detected contract/category.

    ``chunk_predictions`` follows the trainer's ``test_chunk_predictions``
    table (``contract_id``, ``chunk_id``, ``category``, ``probability``,
    ``predicted_label``). ``chunks`` is ``chunks.parquet`` (``chunk_id``,
    ``start_char``, ``end_char``, ``text``).

    A clause in the 128-token overlap is flagged in two chunks, so each
    contract/category keeps only its highest-probability positive chunk. That
    matches the trainer's max pooling: the evidence probability equals the
    contract-level probability. The evidence is the whole chunk (about a page),
    so keyword matches can come from neighbouring clauses in the same chunk.
    """
    positives = chunk_predictions[chunk_predictions["predicted_label"].astype(bool)]
    if positives.empty:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)

    best = (
        positives.sort_values(["probability", "chunk_id"], ascending=[False, True])
        .drop_duplicates(["contract_id", "category"])
    )
    chunk_text = chunks[["chunk_id", "start_char", "end_char", "text"]].rename(columns={"text": "evidence_text"})
    evidence = best.merge(chunk_text, on="chunk_id", how="left", validate="many_to_one")

    missing = evidence["evidence_text"].isna()
    if missing.any():
        examples = evidence.loc[missing, "chunk_id"].head(3).tolist()
        raise ValueError(f"{int(missing.sum())} predicted chunks are not in the chunks table, e.g. {examples}")

    return (
        evidence[EVIDENCE_COLUMNS]
        .sort_values(["contract_id", "start_char", "category"])
        .reset_index(drop=True)
    )
