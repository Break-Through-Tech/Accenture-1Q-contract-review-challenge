"""Split a contract into overlapping token windows that keep their character offsets.

Chunk size and overlap were decided in #22 (see notebooks/Chunking_strategy.ipynb):
510 tokens of contract text per chunk, which is DistilRoBERTa's 512-token limit minus
the 2 special tokens it adds, and 128 tokens of overlap between neighbouring chunks.
"""

from functools import lru_cache

from transformers import AutoTokenizer

TOKENIZER_NAME = "distilroberta-base"
MAX_TOKENS = 510    # 512 model limit minus <s> and </s>. Do not change to 512.
OVERLAP = 128


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load the tokenizer once and reuse it."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_tokens(text, tokenizer=None):
    """Number of tokens in `text`, not counting the model's special tokens."""
    tokenizer = tokenizer or get_tokenizer()
    return len(tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])


def chunk_contract(text, max_tokens=MAX_TOKENS, overlap=OVERLAP, tokenizer=None):
    """Split `text` into overlapping chunks of at most `max_tokens` tokens.

    Returns a list of dicts in document order:
        chunk_index  0, 1, 2, ...
        start_char   where the chunk starts in `text`
        end_char     where it ends (exclusive), so text[start_char:end_char] is the chunk
        n_tokens     tokens in the chunk text when tokenized on its own
        text         the chunk text

    A window can start or end partway through a word, and that piece can tokenize into
    more tokens on its own than it did inside the full contract. Any chunk that would
    go over `max_tokens` is trimmed from the end until it fits, so the model never
    silently cuts off the end of a chunk.
    """
    if not 0 <= overlap < max_tokens:
        raise ValueError(f"overlap must be at least 0 and less than max_tokens, got {overlap}")
    if not text.strip():
        return []

    tokenizer = tokenizer or get_tokenizer()
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        verbose=False)["offset_mapping"]

    chunks = []
    start = 0                                           # token index where this chunk begins
    while True:
        end = min(start + max_tokens, len(offsets))     # token index, exclusive
        start_char = offsets[start][0]

        # Trim from the end until the chunk text fits on its own (see docstring).
        while True:
            end_char = offsets[end - 1][1]
            n_tokens = count_tokens(text[start_char:end_char], tokenizer)
            if n_tokens <= max_tokens or end - start == 1:
                break
            end -= 1

        chunks.append({
            "chunk_index": len(chunks),
            "start_char": start_char,
            "end_char": end_char,
            "n_tokens": n_tokens,
            "text": text[start_char:end_char],
        })

        if end == len(offsets):
            return chunks
        # The next chunk starts (max_tokens - overlap) tokens later, but never after this
        # chunk's end, so trimming can't leave a gap between chunks.
        start = min(start + max_tokens - overlap, end)
