# Light cleaner for transformer inputs

The transformer receives contract chunks after chunking and answer-span labeling. CUAD's extracted text includes some non-contract material, such as page counters, source attribution lines, occasional court docket banners, and a small number of zero-width spaces. These can add noise to model inputs.

The cleaner is implemented as [`clean_transformer_text`](../src/clean/light.py), with [`add_light_cleaned_text`](../src/clean/light.py) for adding a `text_light` column to a chunk table. It replaces those known artifacts with ordinary spaces. Each replacement preserves the number of source characters, so the cleaned text remains aligned with the original chunk offsets. Keep the unmodified `text` for evidence display and audit; use `text_light` as the transformer's input.

## Cleaning rules

- Blank court docket header lines beginning with a case number and `Doc`, such as `Case 18-10248-MFW Doc 632-1 Filed 04/18/18 Page 2 of 60`.
- Blank embedded `Page N of M` counters.
- Blank `Source: ...` attribution lines added by the dataset extraction.
- Replace zero-width spaces and byte-order marks with ordinary spaces.
- Preserve the string's character length so answer and chunk offsets remain valid.

The cleaner does **not** remove stop words, stem or lemmatize, lowercase, rewrite punctuation, or collapse whitespace. It is intentionally conservative because words, punctuation, and context can carry legal meaning.

## Examples

All examples below preserve the original character count; removed characters become spaces.

| Before | After |
|---|---|
| `Case 18-10248-MFW Doc 632-1 Filed 04/18/18 Page 2 of 60\nAGENCY AGREEMENT` | spaces for the docket line, then `\nAGENCY AGREEMENT` |
| `This continues on Page 2 of 60 under the agreement.` | `This continues on [12 spaces] under the agreement.` |
| `Source: PF HOSPITALITY GROUP INC., 10-12G, 9/23/2015\nThe parties agree...` | spaces for the source line, then `\nThe parties agree...` |
| `above\u200bstated warranty` | `above stated warranty` |

## Use in preprocessing

Apply the cleaner to each chunk after the span-to-label mapper has used original offsets. Store both `text` (original) and `text_light` (cleaned) in the processed chunk artifact. The model reads `text_light`; source offsets and original text remain available for evidence extraction. Since cleaning is length-preserving, offsets point to the same character positions in both strings, though the characters at removed-artifact positions are spaces in `text_light`.
