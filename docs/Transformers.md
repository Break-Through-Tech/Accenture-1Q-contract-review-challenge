# Transformers for clause-category detection

## What a transformer is

A transformer is a neural-network architecture for processing sequences such as words or pieces of words. It looks at how the parts of a sequence relate to one another, which helps it use context when representing text.

For example, the word “party” can mean different things in different passages. A transformer builds a representation of that word using the surrounding text, rather than treating it as an isolated count. This contextual processing can help with legal language, where the same idea may be expressed in different ways.

Transformers are commonly pretrained on large collections of text. Pretraining gives the model general language patterns before it is adapted to a particular task. **Fine-tuning** continues training that pretrained model on labeled examples for the project task.

## Planned model in this project

The architecture proposes fine-tuning **DistilRoBERTa** to detect clause categories in contract chunks. DistilRoBERTa is a smaller version of RoBERTa, designed to retain useful language representations while being lighter than the full model.

The input to the classifier is a contract chunk. A tokenizer converts its text into model-specific token IDs. The model processes those tokens and produces one score for each category. Since one chunk can contain clauses from multiple categories, the output is multi-label: each category is predicted independently as present or absent using a selected threshold.

The training targets come from the span-to-label mapper: annotated CUAD answer spans are aligned to chunks, then converted into category labels. The transformer learns from many chunk-and-label examples to produce category scores for new chunks.

## Why use a transformer?

The main purpose is to test whether contextual language understanding improves clause detection beyond the TF-IDF baseline. TF-IDF relies on word and phrase patterns; a transformer can use surrounding words and may recognize related wording even when it does not exactly match a phrase seen during training.

The transformer provides the more capable modeling approach in the planned comparison. TF-IDF provides a faster, simpler baseline. Evaluating both helps show whether the added compute and complexity provide a measurable benefit for this task.

## How it fits into the workflow

1. Chunk contracts and preserve their original character offsets.
2. Map answer spans to chunk-level category labels.
3. Apply the light cleaning intended for the transformer. Preserve the original text and offsets for audit and evidence use.
4. Tokenize chunks using the tokenizer paired with DistilRoBERTa.
5. Fine-tune the pretrained model on labeled training chunks.
6. Use a validation set to make training decisions and choose prediction thresholds; evaluate final performance on held-out test contracts.
7. Save the checkpoint, tokenizer, category list, thresholds, data version, and evaluation results for reproducibility.

All chunks from a contract must remain within that contract's official train or test split. This avoids leakage between training and evaluation.

## What training involves

During fine-tuning, the model adjusts its learned parameters to reduce the difference between its predicted category scores and the chunk's known labels. A multi-label loss function compares each category's score with its target label. The model is trained over repeated batches of chunks; settings such as learning rate, batch size, number of epochs, and imbalance handling affect the result and should be recorded.

The architecture plans to use Google Colab with a GPU for fine-tuning. GPUs make this training practical, although they do not remove the need to monitor memory, runtime, and model quality.

## Evaluation

Because category frequencies are imbalanced, accuracy alone can hide poor performance on less common categories. Report per-category precision, recall, and F1, along with macro-F1, which gives each evaluated category equal weight. Choose thresholds using training/validation data rather than tuning them against the held-out test set.

The current preparation notebook applies a 10% positive-rate cutoff to training categories. The model's output category list and evaluation scope must reflect that choice. Categories removed from training have no positive examples for the model to learn from; the team should decide whether to exclude them from the intended model scope or change the cutoff before final evaluation.

## Limitations and trade-offs

- Fine-tuning requires more compute and setup than the TF-IDF baseline.
- A transformer can still miss clauses, especially when the training examples are limited, labels are noisy, or the relevant language differs from the training data.
- Its predictions are less directly inspectable than TF-IDF word weights, so the system should retain the clause text and model outputs used downstream.
- Chunk boundaries and span-to-label rules affect the examples used for training, just as they do for TF-IDF.
- A higher model score does not by itself establish that the resulting risk decision is correct; risk scoring and review remain separate downstream steps in the architecture.

The transformer is a proposed classifier component, not a complete contract-review system. It predicts clause categories; evidence handling, risk rules, and contract-level triage are separate pipeline components.
