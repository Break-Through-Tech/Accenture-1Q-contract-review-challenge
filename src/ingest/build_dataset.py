"""Build the training inputs the trainer reads: chunk text and chunk labels.

Run from the repository root:

    python -m src.ingest.build_dataset

Reads
    data/cuad/train_separate_questions.json   official train split (408 contracts)
    data/cuad/test.json                       official test split (102 contracts)
    data/processed/contract_categories.jsonl  annotated spans per contract/category

Writes
    data/processed/chunks.parquet             chunk offsets, raw text, cleaned text
    data/processed/chunk_labels.parquet       one nullable-boolean column per category
    data/processed/label_audit/               span and contract/category audit tables

Order: clean, then chunk, then label.

The light cleaner blanks extraction artifacts by replacing them with spaces, which
keeps character offsets valid. But a run of spaces costs one token per space, so
blanking a long docket banner makes the text *more* expensive to tokenize, not less.
Chunking the raw text and cleaning afterwards pushed 26% of chunks over the model's
512-token limit (worst case 510 -> 1304). Chunking the cleaned text instead means the
chunker measures what the model actually sees.

Both texts are kept. `text` is the raw slice, for evidence and display; `text_light`
is the cleaned slice, which is what the model reads. Cleaning never changes character
length, so one set of offsets indexes both, and span offsets stay valid.
"""

import json
from pathlib import Path

import pandas as pd

from src.clean.light import clean_transformer_text
from src.ingest.chunker import chunk_contract
from src.ingest.label_chunks import label_chunks

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "cuad"
PROCESSED = ROOT / "data" / "processed"

SPLITS = {"train": "train_separate_questions.json", "test": "test.json"}


def build_chunks() -> pd.DataFrame:
    """Clean each contract, chunk the cleaned text, and keep both texts."""
    rows = []
    for split, filename in SPLITS.items():
        with (RAW / filename).open(encoding="utf-8") as f:
            data = json.load(f)
        for doc in data["data"]:
            raw = doc["paragraphs"][0]["context"]
            cleaned = clean_transformer_text(raw)
            for chunk in chunk_contract(cleaned):
                start, end = chunk["start_char"], chunk["end_char"]
                rows.append({
                    "split": split,
                    "contract_id": doc["title"],
                    "chunk_id": f"{doc['title']}__{chunk['chunk_index']}",
                    "chunk_index": chunk["chunk_index"],
                    "start_char": start,
                    "end_char": end,
                    "n_tokens": chunk["n_tokens"],   # measured on the cleaned text
                    "text": raw[start:end],
                    "text_light": chunk["text"],
                })
    return pd.DataFrame(rows)


def build(rule: str = "any_overlap") -> None:
    chunks = build_chunks()
    print(f"chunks: {len(chunks):,}")
    print(chunks.groupby("split").size().to_string())
    print(f"max tokens in any chunk (cleaned): {chunks.n_tokens.max()}")

    categories = pd.read_json(PROCESSED / "contract_categories.jsonl", lines=True)
    print(f"\ncategories: {len(categories):,} contract/category rows, "
          f"{categories['category'].nunique()} distinct")

    result = label_chunks(chunks, categories, rule=rule)
    labels = result.chunk_labels
    category_columns = [c for c in labels.columns if c not in set(chunks.columns)]

    # The trainer trains on categories with at least one positive in train.
    train_rows = labels[labels["split"] == "train"]
    trainable = [c for c in category_columns
                 if train_rows[c].fillna(False).astype(bool).any()]
    print(f"\nlabel columns:        {len(category_columns)}")
    print(f"positive in train:    {len(trainable)}   <- trainer's num_labels")
    print(f"no positive in train: {sorted(set(category_columns) - set(trainable))}")

    print(f"\ncontract/category pairs by status:\n"
          f"{result.pair_status['status'].value_counts().to_string()}")
    print(f"\nspans by placement:\n{result.span_audit['status'].value_counts().to_string()}")

    unknown = labels[category_columns].isna().sum().sum()
    total = len(labels) * len(category_columns)
    print(f"\nunknown labels: {unknown:,} of {total:,} ({100 * unknown / total:.1f}%)")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(PROCESSED / "chunks.parquet", index=False)
    labels.to_parquet(PROCESSED / "chunk_labels.parquet", index=False)
    audit_dir = PROCESSED / "label_audit"
    audit_dir.mkdir(exist_ok=True)
    result.span_audit.to_parquet(audit_dir / "span_audit.parquet", index=False)
    result.pair_status.to_parquet(audit_dir / "pair_status.parquet", index=False)
    print(f"\nwrote chunks.parquet, chunk_labels.parquet and label_audit/ to {PROCESSED}")


if __name__ == "__main__":
    build()
