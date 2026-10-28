"""DistilRoBERTa multi-label classifier structure for CUAD chunks.

Chunk labels may contain missing values for categories excluded from training or
for untrusted contract/category annotations. ``label_mask`` prevents those
unknown targets from being treated as negatives during the loss calculation.
"""

from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


DEFAULT_MODEL_NAME = "distilroberta-base"
DEFAULT_MAX_LENGTH = 512


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    """Load the tokenizer paired with the selected pretrained backbone."""
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_chunks(texts, tokenizer, max_length: int = DEFAULT_MAX_LENGTH) -> Dict[str, list]:
    """Tokenize chunks without silently truncating annotated text.

    ``chunk_contract`` creates text windows of at most 510 tokens; DistilRoBERTa
    adds two special tokens, giving a maximum encoded length of 512. A violation
    is raised here instead of dropping text and potentially its positive label.
    Padding is left to the training batch collator.
    """
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    over_limit = [
        index for index, ids in enumerate(encoded["input_ids"])
        if len(ids) > max_length
    ]
    if over_limit:
        raise ValueError(
            f"{len(over_limit)} chunk(s) exceed max_length={max_length}; "
            f"first offending batch index: {over_limit[0]}. Recheck chunking; "
            "do not silently truncate labeled evidence."
        )
    return encoded


def select_train_positive_categories(label_frame, split_column="split", train_split="train"):
    """Return category columns with at least one observed positive train label.

    Metadata columns are excluded automatically. Nullable labels are respected:
    an all-missing category is outside the training scope, not an all-negative
    category.
    """
    metadata = {
        "split", "contract_id", "chunk_id", "chunk_index", "start_char",
        "end_char", "n_tokens",
    }
    candidates = [column for column in label_frame.columns if column not in metadata]
    train_rows = label_frame[label_frame[split_column] == train_split]
    return [
        category for category in candidates
        if train_rows[category].fillna(False).astype(bool).any()
    ]


def prepare_label_arrays(label_frame, categories):
    """Convert nullable category columns into float targets and a known-label mask.

    Returns arrays shaped ``(n_chunks, n_categories)``. Unknown labels are set
    to zero in ``targets`` only as a placeholder; ``label_mask`` excludes them
    from the loss.
    """
    values = label_frame[categories].to_numpy(dtype=np.float32, na_value=np.nan)
    label_mask = np.isfinite(values)
    targets = np.where(label_mask, values, 0.0).astype(np.float32)
    return targets, label_mask.astype(bool)


class MultiLabelTransformer(nn.Module):
    """Pretrained transformer encoder with a multi-label classification head."""

    def __init__(
        self,
        num_labels: int,
        model_name: str = DEFAULT_MODEL_NAME,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_labels < 1:
            raise ValueError("num_labels must be positive")

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return raw category logits and, when targets are given, masked BCE loss."""
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # RoBERTa-family encoders prepend a classification token; use its final
        # contextual representation as the chunk representation.
        chunk_vector = encoder_output.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(chunk_vector))
        output = {"logits": logits}

        if labels is not None:
            labels = labels.to(dtype=logits.dtype)
            if label_mask is None:
                label_mask = torch.isfinite(labels)
            else:
                label_mask = label_mask.to(dtype=torch.bool)
            safe_labels = torch.nan_to_num(labels, nan=0.0)
            element_loss = F.binary_cross_entropy_with_logits(
                logits, safe_labels, reduction="none"
            )
            weights = label_mask.to(dtype=element_loss.dtype)
            loss = (element_loss * weights).sum() / weights.sum().clamp_min(1.0)
            output["loss"] = loss

        return output


def predict_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert model logits to independent category probabilities."""
    return torch.sigmoid(logits)
