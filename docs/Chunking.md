# Chunking contracts for clause-category detection

## What chunking means

Chunking divides a long contract into smaller, ordered text sections. Each section is called a **chunk**. Instead of asking a classifier to process an entire contract at once, the pipeline can process and label each chunk individually.

For example, a contract might be represented as chunk 1, chunk 2, chunk 3, and so on. Each chunk should retain its contract ID, chunk ID, and start and end character offsets in the original contract.

## Why this project needs chunks

Contracts can be much longer than a model can accept as one input. Even when a model can accept a long input, relevant clauses may be easier to locate and classify when the input is focused on a smaller section.

Chunking also makes the modeling task concrete: the input is a section of contract text, and the target is the set of clause categories found in that section. Both the TF-IDF baseline and the planned transformer can use these same chunk-level examples.

## How annotated answers become chunk labels

CUAD provides annotated answer text and character offsets within the original contract. The chunker must preserve the relationship between chunk text and those original offsets. For each answer span, the span-to-label mapper determines which chunk or chunks contain the answer and assigns the corresponding category label.

A chunk can have more than one category label if it contains answers from multiple categories. A chunk with no mapped answer is typically labeled negative for the categories it does not contain. The mapping rule must be explicit and consistent, since it defines the model's training targets.

If an answer span crosses a chunk boundary, the pipeline needs a consistent rule for assigning it—for example, assigning the category to every chunk that overlaps the span. If chunks overlap, an answer may appear in more than one chunk; that is acceptable as long as the behavior is deliberate and evaluation accounts for duplicate evidence where needed.

## Chunk size and overlap

Chunk size sets how much text goes into each section. Smaller chunks give more focused examples but can split a clause from its surrounding context. Larger chunks preserve more context but create longer inputs and may combine unrelated clauses.

Overlap repeats a small amount of text from the end of one chunk at the start of the next. It can help keep a clause together when it falls near a boundary, but it also duplicates text and can create duplicate predictions. The project should use the chunking function selected by the team and record its size, overlap, and boundary behavior so results can be reproduced.

The correct settings depend on the downstream models and CUAD annotations. They should be checked by inspecting examples where an answer is near a boundary or spans a boundary.

## Where chunking fits in the pipeline

The preprocessing flow is:

1. Load the original contract text and CUAD answer spans.
2. Split each contract into chunks and record each chunk's offsets.
3. Map answer spans to chunk-level category labels using the original offsets.
4. Apply the appropriate text cleaning for each model. Do this after mapping if cleaning can alter character positions.
5. Save one row per chunk, with its text, IDs, offsets, and category labels.
6. Train and evaluate the TF-IDF baseline and transformer using the same chunk and label definitions.

The existing `contract_categories.jsonl` is at the contract/category level; it is not the chunked model-training dataset. A chunking step should produce a separate chunk-level artifact, as the architecture proposes with `chunks.parquet`.

## Preventing data leakage

Keep all chunks from a contract in the same official train or test split. Never randomly split chunks after chunking: that could put chunks from one contract in both sets, allowing the model to see nearly identical contract wording during training and testing.

Fit any learned preprocessing, such as the TF-IDF vocabulary and IDF weights, on training data only. Apply the fitted preprocessing to validation and test chunks without refitting it.

## Quality checks

Before training, verify that:

- Each chunk points to the correct contract and has valid start and end offsets.
- Chunk text matches the corresponding section of the original contract.
- Every answer span is mapped according to the documented overlap and boundary rules.
- Multi-label chunks retain all applicable categories.
- Contract-level train/test membership is preserved for every chunk.
- The resulting category list agrees with the training cutoff and evaluation scope.

Chunking choices affect both models' examples and results. Record the chunking configuration alongside the processed data and model run so the experiment can be reproduced.
