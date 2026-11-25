"""Attach supporting contract text to each detected clause, and pick the part of
that text a reviewer should read."""

import re

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


# --- snippets -------------------------------------------------------------------
# An evidence chunk is ~2000 characters and the clause can sit anywhere in it, so
# showing its first N characters usually shows the wrong paragraph. These pick the
# sentences most likely to be the clause itself.

# Words that appear in category names but carry no signal when matching text.
# "non"/"no" would match "non-public", "nonexclusive" etc. and pull the wrong sentence.
_STOPWORDS = {"of", "on", "to", "or", "and", "the", "a", "for", "in", "not", "non", "no"}
_SENTENCE_END = re.compile(r"[.;:](?=\s)")


def _flat(text):
    return " ".join(str(text or "").split())


def category_terms(category, rules):
    """Terms that suggest text is actually about this category.

    The category's own name, plus the restrictive keywords the risk rules define
    for it. Both are matched case-insensitively.
    """
    words = [w.strip("/-") for w in category.replace("-", " ").replace("/", " ").split()]
    terms = {w.lower() for w in words if len(w) > 2 and w.lower() not in _STOPWORDS}
    terms |= {k.lower() for k in rules.get("keywords", {})
              .get("by_category", {}).get(category, [])}
    return terms


def _sentence_spans(text):
    """(start, end) of each sentence in `text`, trimmed of surrounding whitespace."""
    spans, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.end()))
        start = match.end()
    spans.append((start, len(text)))
    trimmed = []
    for s, e in spans:
        piece = text[s:e]
        lead = len(piece) - len(piece.lstrip())
        tail = len(piece) - len(piece.rstrip())
        if e - tail > s + lead:
            trimmed.append((s + lead, e - tail))
    return trimmed


def snippet_span(text, category, rules, width=240):
    """Character span (start, end) within `text` of the clause's likely wording.

    Scores each sentence by how many of the category's terms it contains, then
    grows outwards from the best one until the snippet is about `width`
    characters. Falls back to the start of the text when nothing matches.
    Returns None for empty text.
    """
    text = str(text or "")
    spans = _sentence_spans(text)
    if not spans:
        return None

    terms = category_terms(category, rules)
    scores = [sum(1 for t in terms if t in _flat(text[s:e]).lower()) for s, e in spans]
    best = max(range(len(spans)), key=lambda i: scores[i]) if max(scores) > 0 else 0

    def length(a, b):
        return len(_flat(text[spans[a][0]:spans[b][1]]))

    start = end = best
    # Grow forward first: a clause heading is followed by its terms, not preceded.
    while length(start, end) < width and (start > 0 or end < len(spans) - 1):
        if end < len(spans) - 1:
            end += 1
        else:
            start -= 1
    return spans[start][0], spans[end][1]


def best_excerpt(text, category, rules, width=240):
    """The snippet from `snippet_span` as one line of display text, with "..."
    where it cuts into the surrounding text."""
    span = snippet_span(text, category, rules, width)
    if span is None:
        return ""
    text = str(text)
    excerpt = _flat(text[span[0]:span[1]])
    prefix = "..." if text[:span[0]].strip() else ""
    suffix = "..." if text[span[1]:].strip() or len(excerpt) > width + 60 else ""
    return f"{prefix}{excerpt[:width + 60]}{suffix}"
