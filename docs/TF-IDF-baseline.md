# TF-IDF baseline for clause-category detection

## Purpose

The TF-IDF model is a simple baseline for predicting which CUAD clause categories appear in a contract chunk. It gives the team a fast, understandable result to compare with the planned DistilRoBERTa model.

The comparison answers a practical question: does fine-tuning a transformer improve category detection enough to justify its extra compute and complexity? The baseline also helps reveal whether the labels and chunking pipeline are working before the team spends time tuning a larger model.

This document describes the planned approach in [the architecture](ARCHITECTURE.md); it does not mean that the model has already been implemented or trained.

## How TF-IDF represents text

TF-IDF stands for **term frequency–inverse document frequency**. It converts each chunk into numeric features based on the words and short phrases it contains.

- **Term frequency (TF)** gives weight to terms that occur in a chunk.
- **Inverse document frequency (IDF)** reduces the weight of terms that occur in many chunks and gives more weight to terms that are less common across the training set.
- Multiplying these produces a feature value for each term in each chunk.

For example, general words such as “the” tend to be common across chunks and carry little distinguishing weight. A phrase such as “governing law” may be more useful for identifying the Governing Law category. The architecture proposes using one-word and two-word features (1–2 grams).

TF-IDF does not understand a phrase the way a language model does. It represents patterns in the observed words and phrases, so wording that differs substantially from the training examples may be harder for it to recognize.

## The classifier: one-vs-rest logistic regression

TF-IDF creates the numeric text features; logistic regression learns how those features relate to category labels. Because a chunk can contain multiple clause types, this is a **multi-label** task. The proposed one-vs-rest (OvR) setup trains a binary classifier for each category: that category versus all other chunks. Each classifier returns a score that can be interpreted as the model's confidence for that category, then a threshold turns the score into a predicted label.

The input is one cleaned chunk. The target is the set of category labels assigned to that chunk by the span-to-label mapper. This is why chunking and answer-span alignment must happen before training.

## Why use it alongside a transformer?

The baseline is useful because it is:

- **Quick to train.** It provides an early result without requiring GPU fine-tuning.
- **A reference point.** It shows how well word and phrase patterns work before adding a pretrained transformer.
- **Useful for debugging.** Poor baseline results can expose label, chunking, or data-split problems.
- **Easier to inspect.** The learned word and phrase weights can show which text patterns influenced a category prediction.

The baseline is not expected to be best at understanding legal language. A transformer can use broader context and may recognize similar meanings expressed with different wording. Comparing both models quantifies whether that additional capability helps on this dataset.

See [Transformers for clause-category detection](Transformers.md) for an explanation of the planned DistilRoBERTa model.

## Planned workflow

For a description of the chunking step and span alignment, see [Chunking contracts for clause-category detection](Chunking.md).

1. Create contract chunks while preserving character offsets.
2. Map annotated answer spans to chunk-level category labels.
3. Apply the heavier text cleaning proposed for TF-IDF. Keep labels and offsets established before cleaning, since cleaning can change character positions.
4. Fit the TF-IDF vocabulary and IDF weights using training chunks only.
5. Train the OvR logistic-regression classifiers on those TF-IDF features and multi-label targets.
6. Use the fitted vectorizer to transform validation and test chunks; do not fit it on those splits.
7. Select prediction thresholds using training/validation data, then report per-category precision, recall, and F1, along with macro-F1 on the held-out test contracts.

Keep all chunks from a contract in that contract's official split. This prevents chunks from the same contract appearing in both training and test data.

## Category cutoff and evaluation scope

The current preparation notebook applies a 10% positive-rate cutoff to the training split. Categories below that cutoff are removed from training. A model cannot learn to predict categories for which it has no positive training examples, so the classifier's output labels and reported evaluation scope must match the categories retained for training. If the intended system should cover all CUAD categories, the team should reconsider that cutoff or otherwise define how excluded categories will be handled before final evaluation.

## Limitations

- TF-IDF relies on word and phrase overlap; it has limited ability to capture meaning from context.
- The CUAD labels are imbalanced, so a model can perform poorly on rare categories even if its overall accuracy looks high. Macro-F1 and per-category results are more informative than accuracy alone.
- Prediction scores need category-specific threshold evaluation; a default threshold may not work equally well for every category.
- Chunking and span-to-label rules affect the training targets. Errors in those steps can limit both the TF-IDF and transformer models.

## Expected artifacts

The trained baseline should be saved with its fitted TF-IDF vectorizer and logistic-regression classifiers. Record the training data version, category list, preprocessing choices, thresholds, and evaluation metrics so the run can be reproduced and compared with the transformer.
