# Transformers for clause-category detection

## What a transformer is

A transformer is a neural-network architecture for processing sequences such as words or pieces of words. It looks at how the parts of a sequence relate to one another, which helps it use context when representing text.

For example, the word “party” can mean different things in different passages. A transformer builds a representation of that word using the surrounding text, rather than treating it as an isolated count. This contextual processing can help with legal language, where the same idea may be expressed in different ways.

Transformers are commonly pretrained on large collections of text. Pretraining gives the model general language patterns before it is adapted to a particular task. **Fine-tuning** continues training that pretrained model on labeled examples for the project task.

## Model structure in this project

The architecture uses **DistilRoBERTa** to detect clause categories in contract chunks. DistilRoBERTa is a smaller version of RoBERTa, designed to retain useful language representations while being lighter than the full model.

The input to the classifier is a contract chunk. A tokenizer converts its text into model-specific token IDs. The current chunking strategy creates windows of up to 510 text tokens with 128 tokens of overlap; DistilRoBERTa adds two special tokens, keeping the encoded input within its 512-token limit. The model processes those tokens and produces one score for each of the 34 modeled categories. Since one chunk can contain clauses from multiple categories, the output is multi-label: each category is predicted independently as present or absent using a selected threshold.

The training targets come from the span-to-label mapper: annotated CUAD answer spans are aligned to chunks, then converted into category labels. The transformer learns from many chunk-and-label examples to produce category scores for new chunks. The initial implementation is in [`src/models/transformer.py`](../src/models/transformer.py), with starting settings in [`configs/transformer.yaml`](../configs/transformer.yaml).

The model uses the final hidden representation for the first (classification) token, followed by dropout and a linear layer with one output per selected category. It trains with binary cross-entropy-with-logits, which supports multiple independent labels. The label artifact can contain missing values for categories excluded from training or annotations that cannot be trusted; a label mask excludes those values from the loss instead of treating them as negative examples.

## Why use a transformer?

The main purpose is contextual clause detection using the pretrained DistilRoBERTa model. The current architecture decision is to proceed with the transformer and use the naive predict-absent baseline from EDA as the comparison point; the earlier TF-IDF model comparison is no longer part of the plan.

The transformer produces a score for each chunk/category pair. Later pipeline stages can combine predictions from overlapping chunks, retain supporting text, and pass detected clauses to the separate risk-scoring layer.

## How it fits into the workflow

1. Chunk contracts and preserve their original character offsets.
2. Map answer spans to chunk-level category labels.
3. Apply the light cleaning intended for the transformer after span alignment. Keep the original chunk text and offsets for audit and evidence use.
4. Tokenize chunks using the tokenizer paired with DistilRoBERTa.
5. Fine-tune the pretrained model on labeled training chunks.
6. Evaluate once on the official held-out test contracts using the configured 0.5 threshold. Do not tune thresholds against test predictions; threshold tuning would require a separate validation procedure that preserves contract-level splits.
7. Save the checkpoint, tokenizer, category list, thresholds, data version, and evaluation results for reproducibility.

All chunks from a contract must remain within that contract's official train or test split. This avoids leakage between training and evaluation.

## What training involves

During fine-tuning, the model adjusts its learned parameters to reduce the difference between its predicted category scores and the chunk's known labels. A multi-label loss function compares each category's score with its target label. The model is trained over repeated batches of chunks; settings such as learning rate, batch size, number of epochs, and imbalance handling affect the result and should be recorded.

The architecture plans to use Google Colab with a GPU for fine-tuning. GPUs make this training practical, although they do not remove the need to monitor memory, runtime, and model quality. The end-to-end orchestration notebook is [`notebooks/Transformer_pipeline.ipynb`](../notebooks/Transformer_pipeline.ipynb); the reusable training loop and metrics are in [`src/models/trainer.py`](../src/models/trainer.py) and [`src/eval/multilabel.py`](../src/eval/multilabel.py).

## Evaluation

Because category frequencies are imbalanced, accuracy alone can hide poor performance on less common categories. Report per-category precision, recall, and F1, along with macro-F1, which gives each evaluated category equal weight. The evaluation merges chunk probabilities back to one contract/category score using the maximum chunk probability, so overlap does not count a clause twice. The initial flow uses a fixed 0.5 threshold; any later threshold tuning must use validation data, never the held-out test set.

The chunk-label artifact retains nullable columns for categories that have been excluded from training. The model selects the 34 categories with positive training labels and masks missing targets in the loss. Test metrics use that same resolved category list; the other seven categories remain outside the model's scope.

## Limitations and trade-offs

- Fine-tuning requires more compute and setup than a simple keyword or always-absent baseline.
- A transformer can still miss clauses, especially when the training examples are limited, labels are noisy, or the relevant language differs from the training data.
- Its predictions are less directly inspectable than a linear model's word weights, so the system should retain the clause text and model outputs used downstream.
- Chunk boundaries and span-to-label rules determine the examples used for training and can affect model performance.
- A higher model score does not by itself establish that the resulting risk decision is correct; risk scoring and review remain separate downstream steps in the architecture.

The transformer is the clause-classification component, not a complete contract-review system. It predicts the 34 modeled categories; evidence handling, risk rules, and contract-level triage are separate pipeline components.
