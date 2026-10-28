"""End-to-end CUAD preparation helpers for the transformer pipeline."""

from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.clean.light import add_light_cleaned_text
from src.ingest.chunker import chunk_contract
from src.ingest.label_chunks import label_chunks


# Team decision: retain these two categories despite their train prevalence
# falling below the 10% cutoff. The seven other below-cutoff categories are
# outside the transformer's 34-category output scope.
KEEP_BELOW_CUTOFF = {"No-Solicit Of Customers", "Joint Ip Ownership"}


def _category_from_question(question: str) -> str:
    parts = question.split('"')
    if len(parts) < 2:
        raise ValueError(f"Could not extract category from question: {question!r}")
    return parts[1]


def _transform_split(dataset: dict, split_name: str) -> List[dict]:
    records = []
    for doc in dataset["data"]:
        paragraph = doc["paragraphs"][0]
        grouped_answers = {}
        for qa in paragraph["qas"]:
            category = _category_from_question(qa["question"])
            answers = grouped_answers.setdefault(category, [])
            for answer in qa["answers"]:
                start = answer.get("answer_start")
                answers.append({
                    "text": answer["text"],
                    "answer_start": start,
                    "answer_end": start + len(answer["text"]) if start is not None else None,
                })

        for category, answers in grouped_answers.items():
            records.append({
                "split": split_name,
                "contract_id": doc["title"],
                "contract_title": doc["title"],
                "category": category,
                "present": bool(answers),
                "answers": answers,
            })
    return records


def prepare_contract_categories(
    train_data: dict,
    test_data: dict,
    cutoff: float = 0.10,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Reshape CUAD annotations and apply the team-approved train category cutoff.

    Test annotations are retained for the full 41-category set. Training rows
    for seven below-cutoff categories are removed; the two explicitly approved
    rare categories remain. The transformer later selects the 34 categories
    that have positive training labels.
    """
    records = _transform_split(train_data, "train") + _transform_split(test_data, "test")
    categories = pd.DataFrame(records)
    train_rows = categories["split"] == "train"
    positive_rates = categories.loc[train_rows].groupby("category")["present"].mean()
    below_cutoff = positive_rates[positive_rates < cutoff].index.tolist()
    excluded_categories = sorted(set(below_cutoff) - KEEP_BELOW_CUTOFF)
    retained_rare = sorted(set(below_cutoff) & KEEP_BELOW_CUTOFF)

    categories = categories[
        ~((categories["split"] == "train") & categories["category"].isin(excluded_categories))
    ].copy()
    summary = {
        "cutoff": cutoff,
        "original_category_count": int(positive_rates.size),
        "excluded_categories": excluded_categories,
        "retained_below_cutoff": retained_rare,
        "train_category_count": int(categories.loc[categories["split"] == "train", "category"].nunique()),
        "test_category_count": int(categories.loc[categories["split"] == "test", "category"].nunique()),
    }
    return categories, summary


def build_chunks(train_data: dict, test_data: dict, tokenizer=None) -> pd.DataFrame:
    """Chunk original contract text without cleaning, preserving source offsets."""
    records = []
    for split_name, dataset in (("train", train_data), ("test", test_data)):
        for doc in dataset["data"]:
            for chunk in chunk_contract(doc["paragraphs"][0]["context"], tokenizer=tokenizer):
                records.append({
                    "split": split_name,
                    "contract_id": doc["title"],
                    "chunk_id": f"{doc['title']}__{chunk['chunk_index']}",
                    **chunk,
                })
    return pd.DataFrame(records)


def prepare_transformer_data(
    train_data: dict,
    test_data: dict,
    cutoff: float = 0.10,
    tokenizer=None,
    label_rule: str = "any_overlap",
):
    """Prepare contract labels, chunks, chunk labels, and cleaned model text.

    The contract text is chunked and span labels are mapped before text cleaning.
    The returned chunk frame contains both original ``text`` and ``text_light``.
    """
    categories, category_summary = prepare_contract_categories(train_data, test_data, cutoff)
    chunks = build_chunks(train_data, test_data, tokenizer=tokenizer)
    label_result = label_chunks(chunks, categories, rule=label_rule)
    model_chunks = add_light_cleaned_text(chunks, text_column="text", output_column="text_light")
    return {
        "categories": categories,
        "category_summary": category_summary,
        "chunks": model_chunks,
        "chunk_labels": label_result.chunk_labels,
        "span_overlaps": label_result.span_overlaps,
        "span_audit": label_result.span_audit,
        "pair_status": label_result.pair_status,
    }
