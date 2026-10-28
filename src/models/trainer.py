"""Training and official-test evaluation for the CUAD DistilRoBERTa model."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.eval.multilabel import multilabel_metrics
from src.models.transformer import (
    DEFAULT_MODEL_NAME,
    MultiLabelTransformer,
    prepare_label_arrays,
    select_train_positive_categories,
    tokenize_chunks,
)


class _ChunkDataset(Dataset):
    def __init__(self, texts, targets, known_mask):
        self.texts = list(texts)
        self.targets = targets
        self.known_mask = known_mask

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.targets[index], self.known_mask[index]


def _device_for_run(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _join_chunk_text_and_labels(chunks: pd.DataFrame, labels: pd.DataFrame, text_column: str):
    keys = ["split", "contract_id", "chunk_id"]
    required_chunk_columns = set(keys + [text_column])
    missing = required_chunk_columns - set(chunks.columns)
    if missing:
        raise ValueError(f"chunk file is missing required columns: {sorted(missing)}")
    missing = set(keys) - set(labels.columns)
    if missing:
        raise ValueError(f"label file is missing join keys: {sorted(missing)}")
    if chunks["chunk_id"].duplicated().any() or labels["chunk_id"].duplicated().any():
        raise ValueError("chunk_id must be unique in both parquet files")

    joined = chunks[keys + [text_column]].merge(
        labels, on=keys, how="inner", validate="one_to_one"
    )
    if len(joined) != len(chunks) or len(joined) != len(labels):
        raise ValueError(
            f"chunk/label join lost rows: chunks={len(chunks)}, labels={len(labels)}, joined={len(joined)}"
        )
    split_counts = joined.groupby("contract_id")["split"].nunique()
    if (split_counts != 1).any():
        raise ValueError("a contract appears in more than one split")
    return joined


def _collate_chunks(batch, tokenizer, max_length: int):
    texts, targets, masks = zip(*batch)
    tokenized = tokenize_chunks(texts, tokenizer, max_length=max_length)
    padded = tokenizer.pad(tokenized, padding=True, return_tensors="pt")
    padded["labels"] = torch.as_tensor(np.stack(targets), dtype=torch.float32)
    padded["label_mask"] = torch.as_tensor(np.stack(masks), dtype=torch.bool)
    return padded


def _positive_weights(targets, known_mask, maximum: float) -> torch.Tensor:
    mask = known_mask.astype(np.float32)
    positives = (targets * mask).sum(axis=0)
    negatives = ((1.0 - targets) * mask).sum(axis=0)
    if np.any(positives == 0):
        raise ValueError("positive-weight strategy requires at least one positive per category")
    return torch.tensor(np.clip(negatives / positives, 1.0, maximum), dtype=torch.float32)


def train_and_evaluate(config_path="configs/transformer.yaml", device="auto") -> Dict[str, object]:
    """Train on the official CUAD train contracts and evaluate once on official test.

    Thresholds are not tuned on the test set. This initial flow uses the
    configured fixed threshold; threshold tuning is a separate decision.
    """
    config_path = Path(config_path)
    with config_path.open() as f:
        config = yaml.safe_load(f)

    seed = int(config["training"]["seed"])
    _set_seed(seed)
    device_obj = _device_for_run(device)
    data_root = config_path.resolve().parents[1]
    chunks_path = data_root / config["data"]["chunks_path"]
    labels_path = data_root / config["data"]["labels_path"]
    chunks = pd.read_parquet(chunks_path)
    labels = pd.read_parquet(labels_path)
    text_column = config["data"]["text_column"]

    categories = select_train_positive_categories(
        labels,
        split_column=config["data"]["split_column"],
        train_split="train",
    )
    expected_num_labels = config["model"].get("expected_num_labels")
    if expected_num_labels is not None and len(categories) != int(expected_num_labels):
        raise ValueError(
            f"expected {expected_num_labels} train-supported categories, found {len(categories)}: {categories}"
        )
    if not categories:
        raise ValueError("no categories have positive training labels")

    # Keep only selected output labels plus identifiers before joining text.
    joined = _join_chunk_text_and_labels(
        chunks,
        labels[["split", "contract_id", "chunk_id"] + categories],
        text_column,
    )
    train_frame = joined[joined["split"] == "train"].reset_index(drop=True)
    test_frame = joined[joined["split"] == "test"].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError("both official train and test splits must be present")

    train_targets, train_known = prepare_label_arrays(train_frame, categories)
    test_targets, test_known = prepare_label_arrays(test_frame, categories)
    if not test_known.any():
        raise ValueError("test split has no known labels for the selected categories")

    tokenizer = AutoTokenizer.from_pretrained(config["model"].get("name", DEFAULT_MODEL_NAME))
    max_length = int(config["model"]["max_length"])
    batch_size = int(config["training"]["batch_size"])
    train_dataset = _ChunkDataset(train_frame[text_column], train_targets, train_known)
    test_dataset = _ChunkDataset(test_frame[text_column], test_targets, test_known)
    generator = torch.Generator().manual_seed(seed)
    collator = lambda batch: _collate_chunks(batch, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator,
        num_workers=0, generator=generator,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator,
        num_workers=0,
    )

    model_name = config["model"].get("name", DEFAULT_MODEL_NAME)
    model = MultiLabelTransformer(
        num_labels=len(categories),
        model_name=model_name,
        dropout=float(config["model"].get("dropout", 0.1)),
    ).to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    imbalance_strategy = config["training"].get("imbalance_strategy", "none")
    if imbalance_strategy == "none":
        pos_weight = None
    elif imbalance_strategy == "pos_weight":
        pos_weight = _positive_weights(
            train_targets, train_known,
            maximum=float(config["training"].get("pos_weight_max", 20.0)),
        ).to(device_obj)
    else:
        raise ValueError(f"unsupported imbalance_strategy: {imbalance_strategy!r}")

    history: List[dict] = []
    epochs = int(config["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_batches = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device_obj) for key, value in batch.items()}
            labels_batch = batch.pop("labels")
            mask_batch = batch.pop("label_mask")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                **batch,
                labels=labels_batch,
                label_mask=mask_batch,
                pos_weight=pos_weight,
            )
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(outputs["loss"].detach().cpu())
            total_batches += 1
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_batches, 1),
        }
        history.append(epoch_record)
        print(f"epoch {epoch}/{epochs} - train loss: {epoch_record['train_loss']:.4f}")

    # Evaluate only after training; do not tune the threshold on these test predictions.
    model.eval()
    probability_batches = []
    target_batches = []
    mask_batches = []
    with torch.inference_mode():
        for batch in test_loader:
            batch = {key: value.to(device_obj) for key, value in batch.items()}
            labels_batch = batch.pop("labels")
            mask_batch = batch.pop("label_mask")
            logits = model(**batch)["logits"]
            probability_batches.append(torch.sigmoid(logits).cpu().numpy())
            target_batches.append(labels_batch.cpu().numpy())
            mask_batches.append(mask_batch.cpu().numpy())
    probabilities = np.concatenate(probability_batches)
    targets = np.concatenate(target_batches)
    known_mask = np.concatenate(mask_batches)
    threshold = float(config["model"].get("decision_threshold", 0.5))
    test_chunk_predictions = pd.DataFrame({
        "split": np.repeat(test_frame["split"].to_numpy(), len(categories)),
        "contract_id": np.repeat(test_frame["contract_id"].to_numpy(), len(categories)),
        "chunk_id": np.repeat(test_frame["chunk_id"].to_numpy(), len(categories)),
        "category": np.tile(np.asarray(categories, dtype=object), len(test_frame)),
        "gold_label": np.where(known_mask, targets, np.nan).reshape(-1),
        "label_known": known_mask.reshape(-1),
        "probability": probabilities.reshape(-1),
        "predicted_label": (probabilities >= threshold).reshape(-1),
    })

    # A clause can mark multiple overlapping chunks. Collapse predictions to
    # one probability per contract/category before scoring to avoid counting a
    # duplicated clause more than once. Max pooling represents "present in any
    # chunk"; contract gold labels are rebuilt from the span-derived chunk labels.
    contract_gold = test_frame.groupby("contract_id", sort=False)[categories].max()
    chunk_probability_frame = pd.DataFrame(probabilities, columns=categories)
    chunk_probability_frame.insert(0, "contract_id", test_frame["contract_id"].to_numpy())
    contract_prob_frame = (
        chunk_probability_frame.groupby("contract_id", sort=False)[categories]
        .max()
        .reindex(contract_gold.index)
    )
    contract_targets, contract_known = prepare_label_arrays(contract_gold.reset_index(), categories)
    contract_probabilities = contract_prob_frame.to_numpy(dtype=np.float32)
    per_category, summary = multilabel_metrics(
        contract_targets,
        contract_probabilities,
        contract_known,
        categories,
        threshold=threshold,
    )
    contract_gold_values = np.where(contract_known, contract_targets, np.nan)
    test_contract_predictions = pd.DataFrame({
        "split": "test",
        "contract_id": np.repeat(contract_gold.index.to_numpy(), len(categories)),
        "category": np.tile(np.asarray(categories, dtype=object), len(contract_gold)),
        "gold_label": contract_gold_values.reshape(-1),
        "label_known": contract_known.reshape(-1),
        "probability": contract_probabilities.reshape(-1),
        "predicted_label": (contract_probabilities >= threshold).reshape(-1),
    })

    output_dir = data_root / config["artifacts"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = output_dir / "encoder"
    tokenizer_dir = output_dir / "tokenizer"
    model.encoder.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    torch.save(model.classifier.state_dict(), output_dir / "classifier.pt")
    per_category.to_csv(output_dir / "test_category_metrics.csv", index=False)
    test_chunk_predictions.to_parquet(output_dir / "test_chunk_predictions.parquet", index=False)
    test_contract_predictions.to_parquet(output_dir / "test_contract_predictions.parquet", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    metadata = {
        "model_name": model_name,
        "categories": categories,
        "num_labels": len(categories),
        "max_length": max_length,
        "threshold": threshold,
        "evaluation_unit": "contract",
        "chunk_probability_aggregation": "maximum probability across a contract's chunks",
        "seed": seed,
        "training": config["training"],
        "official_split": {"train_contracts": 408, "test_contracts": 102},
        "test_summary": summary,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("test summary:", json.dumps(summary, indent=2))
    return {
        "model": model,
        "categories": categories,
        "history": pd.DataFrame(history),
        "per_category_metrics": per_category,
        "test_chunk_predictions": test_chunk_predictions,
        "test_contract_predictions": test_contract_predictions,
        "summary": summary,
        "output_dir": output_dir,
        "device": str(device_obj),
    }
