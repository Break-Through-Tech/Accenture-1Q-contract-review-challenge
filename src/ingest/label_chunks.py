"""Label chunks with CUAD categories by intersecting annotated spans with chunk offsets.

Inputs
    chunks       one row per chunk, as written by Prepare_contract_category_data.ipynb
                 (contract_id, chunk_id, chunk_index, start_char, end_char, n_tokens,
                 and optionally text).
    categories   one row per (contract, category), from contract_categories.jsonl
                 (split, contract_id, category, present, answers). Each answer has
                 `text` and `answer_start`.

Alignment assumptions (checked in notebooks/Chunk_labels_examples.ipynb)
    1. Every offset is a Python string index into the contract's original `context`
       (paragraphs[0].context in the CUAD JSON). The chunker received that same string,
       with no cleaning or normalization, so spans and chunks share one coordinate system.
    2. Intervals are half-open. A span covers [answer_start, answer_start + len(text)) and a
       chunk covers [start_char, end_char). A span that only touches a chunk edge does not
       overlap it.
    3. Chunk offsets increase strictly in both start_char and end_char within a contract,
       so the chunks a span overlaps are a consecutive run (found with a binary search).
    4. Chunks leave out only whitespace (leading, trailing, or between chunks). Spans have
       no leading or trailing whitespace, so a valid span always overlaps a chunk.
    5. When chunk text is available, the span text must equal the chunk text over the
       overlap. A mismatch means the offsets do not point at the annotated words.

Labelling rules (`rule=`)
    any_overlap      (default) A chunk is positive for a category if it overlaps any of the
                     category's spans by at least one character. A span cut by a chunk
                     boundary marks every chunk it touches, and a span inside the overlap
                     between two chunks marks both. A chunk that holds only a small piece of
                     a span that is whole in a neighbouring chunk is positive too.
    contained_first  A chunk is positive if it fully contains one of the category's spans.
                     A span that no chunk fully contains (a boundary cuts it) marks every
                     chunk it overlaps instead, so every span still produces a positive.
                     Chunks that hold only a small tail of a span that is whole in a
                     neighbour stay negative.
    With either rule each chunk is labelled on its own, so a span inside the overlap between
    two chunks marks both. Merge chunk predictions back to contract level before scoring.

Why labels are True, False, or missing
    True     a span the rule accepts marks this chunk.
    False    the contract has no span for this category (status "absent"), or none of its
             spans mark this chunk. This is a genuine negative.
    <NA>     the label cannot be trusted or does not exist, so it must not be used as a
             negative: the category was dropped from this split, or the contract/category
             has a span that is unplaced or misaligned. Such a span marks nothing, and the
             False labels for that contract/category are masked as <NA>; True labels from
             the same category's good spans are kept.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

RULES = ("any_overlap", "contained_first")

ABSENT, PLACED, UNPLACED, MISALIGNED = "absent", "placed", "unplaced", "misaligned"
META_COLUMNS = ["split", "contract_id", "chunk_id", "chunk_index", "start_char", "end_char", "n_tokens"]
OVERLAP_COLUMNS = ["split", "contract_id", "category", "span_index", "chunk_row", "chunk_index",
                   "overlap_start", "overlap_end", "overlap_chars", "span_chars", "share_of_span",
                   "fully_contained", "text_matches", "marks_chunk"]


class ChunkLabels(NamedTuple):
    chunk_labels: pd.DataFrame   # one row per chunk, one nullable-boolean column per category
    span_overlaps: pd.DataFrame  # one row per (span, chunk) intersection
    span_audit: pd.DataFrame     # one row per span: where it landed and whether it can be trusted
    pair_status: pd.DataFrame    # one row per (contract, category): absent / placed / unplaced / misaligned


def label_chunks(chunks, categories, rule="any_overlap"):
    """Derive chunk labels from span overlap. See the module docstring for the rules."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")

    chunks = chunks.reset_index(drop=True)
    has_text = "text" in chunks.columns
    chunk_text = chunks["text"].to_numpy() if has_text else None

    # Chunk positions per contract, in chunk order, with their offsets.
    by_contract = {}
    for contract_id, group in chunks.groupby("contract_id", sort=False):
        group = group.sort_values("chunk_index")
        starts, ends = group["start_char"].to_numpy(), group["end_char"].to_numpy()
        if (np.diff(starts) <= 0).any() or (np.diff(ends) <= 0).any():
            raise ValueError(f"chunk offsets do not increase strictly in {contract_id!r}")
        by_contract[contract_id] = (group.index.to_numpy(), starts, ends)

    missing = set(categories["contract_id"]) - set(by_contract)
    if missing:
        raise ValueError(f"{len(missing)} contracts have no chunks, e.g. {sorted(missing)[0]!r}")

    overlap_rows, span_rows, pair_rows = [], [], []

    for row in categories.itertuples(index=False):
        positions, starts, ends = by_contract[row.contract_id]
        statuses = []

        for span_index, answer in enumerate(row.answers):
            s = int(answer["answer_start"])
            e = s + len(answer["text"])
            span_text = answer["text"]

            # Chunks with end > s and start < e: a consecutive run, because both arrays increase.
            first = int(np.searchsorted(ends, s, side="right"))
            last = int(np.searchsorted(starts, e, side="left")) - 1
            touched = range(first, last + 1) if e > s else range(0)

            span_overlaps = []
            for i in touched:
                lo, hi = max(s, starts[i]), min(e, ends[i])
                text_ok = None
                if has_text:
                    text_ok = chunk_text[positions[i]][lo - starts[i]:hi - starts[i]] == span_text[lo - s:hi - s]
                span_overlaps.append({
                    "split": row.split, "contract_id": row.contract_id, "category": row.category,
                    "span_index": span_index, "chunk_row": int(positions[i]),
                    "chunk_index": i, "overlap_start": int(lo), "overlap_end": int(hi),
                    "overlap_chars": int(hi - lo), "span_chars": e - s,
                    "share_of_span": (hi - lo) / (e - s),
                    "fully_contained": bool(starts[i] <= s and e <= ends[i]),
                    "text_matches": text_ok,
                })

            n_containing = sum(o["fully_contained"] for o in span_overlaps)

            if e <= s:
                status = MISALIGNED                      # empty span: no offsets to trust
            elif not span_overlaps:
                status = UNPLACED                        # valid span that lands in no chunk
            elif has_text and not all(o["text_matches"] for o in span_overlaps):
                status = MISALIGNED                      # offsets point at different words
            else:
                status = PLACED
            statuses.append(status)

            # Only a trusted span marks chunks: a misaligned one points at the wrong words.
            for o in span_overlaps:
                o["marks_chunk"] = status == PLACED and (
                    o["fully_contained"] or n_containing == 0 if rule == "contained_first" else True)
            overlap_rows.extend(span_overlaps)

            if not span_overlaps:
                kind = "outside_chunks"
            elif n_containing == 0:
                kind = "crossing"                        # every chunk it touches holds only part of it
            elif n_containing == 1:
                kind = "contained"
            else:
                kind = "in_overlap"                      # whole in two or more chunks
            span_rows.append({
                "split": row.split, "contract_id": row.contract_id, "category": row.category,
                "span_index": span_index, "answer_start": s, "answer_end": e, "span_chars": e - s,
                "n_chunks": len(span_overlaps), "n_containing": n_containing,
                "kind": kind, "status": status,
            })

        if not statuses:
            pair_status = ABSENT
        elif MISALIGNED in statuses:
            pair_status = MISALIGNED
        elif UNPLACED in statuses:
            pair_status = UNPLACED
        else:
            pair_status = PLACED
        pair_rows.append({
            "split": row.split, "contract_id": row.contract_id, "category": row.category,
            "n_spans": len(statuses), "n_placed": statuses.count(PLACED),
            "n_unplaced": statuses.count(UNPLACED), "n_misaligned": statuses.count(MISALIGNED),
            "status": pair_status,
        })

    span_overlaps = pd.DataFrame(overlap_rows, columns=OVERLAP_COLUMNS).astype(
        {"fully_contained": "bool", "marks_chunk": "bool", "text_matches": "boolean"})
    span_audit = pd.DataFrame(span_rows)
    pair_status = pd.DataFrame(pair_rows)

    # Wide label matrix: 1 = True, 0 = False, -1 = missing.
    all_categories = sorted(categories["category"].unique())
    col = {c: k for k, c in enumerate(all_categories)}
    matrix = np.zeros((len(chunks), len(all_categories)), dtype=np.int8)

    for split, in_split in categories.groupby("split")["category"].agg(set).items():
        rows = np.flatnonzero((chunks["split"] == split).to_numpy())
        for c in set(all_categories) - in_split:         # category dropped from this split
            matrix[rows, col[c]] = -1

    for r in pair_status[pair_status["status"].isin([UNPLACED, MISALIGNED])].itertuples(index=False):
        matrix[by_contract[r.contract_id][0], col[r.category]] = -1   # mask the negatives

    marked = span_overlaps[span_overlaps["marks_chunk"]]
    matrix[marked["chunk_row"].to_numpy(dtype=int), marked["category"].map(col).to_numpy(dtype=int)] = 1

    labels = pd.DataFrame({
        c: pd.arrays.BooleanArray(matrix[:, k] == 1, matrix[:, k] < 0) for c, k in col.items()
    })
    chunk_labels = pd.concat([chunks[META_COLUMNS], labels], axis=1)

    n_marked = marked.groupby(["contract_id", "category"]).size().rename("n_positive_chunks")
    pair_status = pair_status.join(n_marked, on=["contract_id", "category"])
    pair_status["n_positive_chunks"] = pair_status["n_positive_chunks"].fillna(0).astype(int)
    pair_status["n_chunks"] = pair_status["contract_id"].map({k: len(v[0]) for k, v in by_contract.items()})

    return ChunkLabels(chunk_labels, span_overlaps.drop(columns="chunk_row"), span_audit, pair_status)
