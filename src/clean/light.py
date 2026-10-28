"""Conservative cleaning for CUAD text fed to the transformer.

Run this after answer spans have been mapped to chunks. Known extraction and
source-metadata artifacts are blanked with spaces rather than deleted, so the
cleaned string has exactly the same character length as its source. Keep the
original text for evidence and display; this function only prepares model input.
"""

import re


_ARTIFACT_PATTERNS = (
    # Court docket banners inserted during PDF extraction, e.g.:
    # "Case 18-10248-MFW Doc 632-1 Filed 04/18/18 Page 2 of 60"
    re.compile(
        r"(?im)^[ \t]*Case\s+\d{1,2}-\d{4,6}-[ \t]*[A-Z0-9-]+\s+Doc\b[^\r\n]*"
    ),
    # Page counters commonly embedded in the extracted contract body.
    re.compile(r"(?i)\bPage\s+\d+\s+of\s+\d+\b"),
    # CUAD/source attribution lines are dataset metadata, not contract clauses.
    re.compile(r"(?im)^[ \t]*Source:[^\r\n]*"),
)

_INVISIBLE_ARTIFACTS = ("\u200b", "\ufeff")  # zero-width space and BOM


def _blank_matches(pattern: re.Pattern, text: str) -> str:
    """Replace matched artifact characters with spaces without changing length."""
    return pattern.sub(lambda match: " " * len(match.group(0)), text)


def clean_transformer_text(text: str) -> str:
    """Remove known CUAD extraction artifacts while preserving legal language.

    The transformation is character-length preserving: docket headers, page
    counters, source attribution lines, zero-width spaces, and BOM characters
    are replaced by ordinary spaces. It does not lowercase text, remove
    stopwords, stem or lemmatize words, normalize punctuation, or collapse
    whitespace. Those broader changes could alter legal wording or offsets.

    Args:
        text: Original chunk text, before model tokenization.

    Returns:
        Cleaned text with the same number of Python characters as ``text``.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    cleaned = text
    for artifact in _ARTIFACT_PATTERNS:
        cleaned = _blank_matches(artifact, cleaned)
    for invisible in _INVISIBLE_ARTIFACTS:
        cleaned = cleaned.replace(invisible, " ")

    # This invariant keeps all source character offsets usable after cleaning.
    if len(cleaned) != len(text):  # defensive guard if a future rule changes length
        raise RuntimeError("transformer cleaner changed text length; source offsets would shift")
    return cleaned


def add_light_cleaned_text(chunks, text_column: str = "text", output_column: str = "text_light"):
    """Return a copy of a chunk table with a cleaned transformer-text column.

    The original ``text`` column and all chunk metadata/labels remain unchanged.
    Call this after the span-to-label mapper has run.
    """
    if text_column not in chunks.columns:
        raise KeyError(f"chunk table has no {text_column!r} column")
    cleaned_chunks = chunks.copy()
    cleaned_chunks[output_column] = cleaned_chunks[text_column].map(clean_transformer_text)
    return cleaned_chunks
