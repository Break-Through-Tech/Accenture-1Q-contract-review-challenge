"""Cross-validated threshold tuning for the CUAD classifier (#7).

The problem this solves: the shipped model trained on all 408 train contracts, so its
confidence on those is inflated and cannot be used to choose a threshold. The 102 test
contracts are unseen, but tuning on them would make the reported score a best-of-many
rather than an honest measurement.

K-fold cross-validation produces *out-of-fold* probabilities: every train contract is
scored by a model that never saw it. Thresholds tuned on those are honest, and unlike a
single holdout every category keeps all of its positive examples for tuning, which
matters for the rare ones (No-Solicit Of Customers has only 27 positive contracts).

Folds split by contract, never by chunk, so no contract's chunks straddle a fold.

    python -m src.models.cross_validate                 # 5 folds, writes out-of-fold probs
    python -m src.models.cross_validate --tune-only     # re-tune from saved probs, no GPU
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.trainer import (
    _ChunkDataset,
    _collate_chunks,
    _device_for_run,
    _join_chunk_text_and_labels,
    _positive_weights,
    _set_seed,
)
from src.models.transformer import (
    DEFAULT_MODEL_NAME,
    MultiLabelTransformer,
    prepare_label_arrays,
    select_train_positive_categories,
)


def _load(config_path):
    """Config, the train frame joined to its labels, and the category list."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parents[1]

    chunks = pd.read_parquet(root / config["data"]["chunks_path"])
    labels = pd.read_parquet(root / config["data"]["labels_path"])
    categories = select_train_positive_categories(
        labels, split_column=config["data"]["split_column"], train_split="train")

    joined = _join_chunk_text_and_labels(
        chunks, labels[["split", "contract_id", "chunk_id"] + categories],
        config["data"]["text_column"])
    train_frame = joined[joined["split"] == "train"].reset_index(drop=True)
    return config, root, train_frame, categories


def _fit_fold(frame, categories, config, device, seed):
    """Train one model on `frame`. Returns the trained model and its tokenizer."""
    targets, known = prepare_label_arrays(frame, categories)
    model_name = config["model"].get("name", DEFAULT_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = int(config["model"]["max_length"])

    loader = DataLoader(
        _ChunkDataset(frame[config["data"]["text_column"]], targets, known),
        batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda b: _collate_chunks(b, tokenizer, max_length),
    )

    model = MultiLabelTransformer(
        num_labels=len(categories), model_name=model_name,
        dropout=float(config["model"].get("dropout", 0.1))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)))

    strategy = config["training"].get("imbalance_strategy", "none")
    pos_weight = None
    if strategy == "pos_weight":
        pos_weight = _positive_weights(
            targets, known,
            maximum=float(config["training"].get("pos_weight_max", 20.0))).to(device)
    elif strategy != "none":
        raise ValueError(f"unsupported imbalance_strategy: {strategy!r}")

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels_batch = batch.pop("labels")
            mask_batch = batch.pop("label_mask")
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch, labels=labels_batch, label_mask=mask_batch,
                        pos_weight=pos_weight)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += float(out["loss"].detach().cpu())
            batches += 1
        print(f"    epoch {epoch + 1}: loss {total / max(batches, 1):.4f}", flush=True)
    return model, tokenizer


def _predict(model, tokenizer, frame, categories, config, device):
    """Chunk-level probabilities for `frame`, shaped (n_chunks, n_categories)."""
    targets, known = prepare_label_arrays(frame, categories)
    max_length = int(config["model"]["max_length"])
    loader = DataLoader(
        _ChunkDataset(frame[config["data"]["text_column"]], targets, known),
        batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0,
        collate_fn=lambda b: _collate_chunks(b, tokenizer, max_length),
    )
    model.eval()
    out = []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("labels")
            batch.pop("label_mask")
            out.append(torch.sigmoid(model(**batch)["logits"]).cpu().numpy())
    return np.concatenate(out)


def _to_contract_level(frame, probabilities, categories):
    """Max-pool chunk probabilities per contract, alongside the contract's gold label.

    Matches how `trainer.py` scores the test set: a clause inside a chunk overlap
    appears in two chunks and must not be counted twice.
    """
    probs = pd.DataFrame(probabilities, columns=categories)
    probs.insert(0, "contract_id", frame["contract_id"].to_numpy())
    pooled = probs.groupby("contract_id", sort=False)[categories].max()
    gold = frame.groupby("contract_id", sort=False)[categories].max().reindex(pooled.index)

    gold_values, known = prepare_label_arrays(gold.reset_index(), categories)
    return pd.DataFrame({
        "contract_id": np.repeat(pooled.index.to_numpy(), len(categories)),
        "category": np.tile(np.asarray(categories, dtype=object), len(pooled)),
        "probability": pooled.to_numpy(dtype=np.float32).reshape(-1),
        "gold_label": np.where(known, gold_values, np.nan).reshape(-1),
        "label_known": known.reshape(-1),
    })


def run_cross_validation(config_path="configs/transformer.yaml", n_folds=5, device="auto"):
    """Train `n_folds` models and collect out-of-fold contract-level probabilities."""
    config, root, train_frame, categories = _load(config_path)
    seed = int(config["training"]["seed"])
    _set_seed(seed)
    device_obj = _device_for_run(device)

    contracts = np.array(sorted(train_frame["contract_id"].unique()))
    np.random.default_rng(seed).shuffle(contracts)
    folds = np.array_split(contracts, n_folds)

    print(f"{len(contracts)} train contracts, {n_folds} folds, "
          f"{len(categories)} categories, device {device_obj}")

    out_of_fold = []
    for i, held_out in enumerate(folds, start=1):
        is_held = train_frame["contract_id"].isin(held_out)
        fit_frame = train_frame[~is_held].reset_index(drop=True)
        held_frame = train_frame[is_held].reset_index(drop=True)
        print(f"\n  fold {i}/{n_folds}: train {len(fit_frame):,} chunks, "
              f"score {len(held_frame):,} chunks ({len(held_out)} contracts)", flush=True)

        model, tokenizer = _fit_fold(fit_frame, categories, config, device_obj, seed + i)
        probabilities = _predict(model, tokenizer, held_frame, categories, config, device_obj)
        fold_predictions = _to_contract_level(held_frame, probabilities, categories)
        fold_predictions.insert(0, "fold", i)
        out_of_fold.append(fold_predictions)

        del model
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    predictions = pd.concat(out_of_fold, ignore_index=True)
    output_dir = root / config["artifacts"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "oof_predictions.parquet"
    predictions.to_parquet(path, index=False)
    print(f"\nwrote {len(predictions):,} out-of-fold rows to {path}")
    return predictions


def _f1(predicted, gold):
    tp = (predicted & gold).sum()
    fp = (predicted & ~gold).sum()
    fn = (~predicted & gold).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def tune_thresholds(predictions, grid=None):
    """Pick a global threshold and per-category thresholds from out-of-fold probabilities."""
    grid = np.arange(0.02, 0.99, 0.01) if grid is None else grid
    known = predictions[predictions["label_known"]]

    sweep = pd.DataFrame([
        {"threshold": round(float(t), 2),
         "macro_f1": float(np.mean([_f1(g.probability >= t, g.gold_label == 1)
                                    for _, g in known.groupby("category")]))}
        for t in grid
    ])
    global_threshold = float(sweep.loc[sweep.macro_f1.idxmax(), "threshold"])

    rows = []
    for category, g in known.groupby("category"):
        gold = g.gold_label == 1
        scores = [_f1(g.probability >= t, gold) for t in grid]
        best = int(np.argmax(scores))
        rows.append({
            "category": category,
            "threshold": round(float(grid[best]), 2),
            "oof_f1": round(scores[best], 4),
            "oof_f1_at_half": round(_f1(g.probability >= 0.5, gold), 4),
            "support": int(gold.sum()),
        })
    per_category = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)
    return sweep, global_threshold, per_category


def _main():
    parser = argparse.ArgumentParser(description="Cross-validated threshold tuning")
    parser.add_argument("--config", default="configs/transformer.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backup-to", default=None, metavar="DIR",
                        help="also copy results here when finished, e.g. a mounted "
                             "Drive folder. Colab wipes its disk after about 90 minutes "
                             "idle, so an unattended run should set this.")
    parser.add_argument("--tune-only", action="store_true",
                        help="re-tune from saved out-of-fold probabilities (no GPU needed)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    root = Path(args.config).resolve().parents[1]
    output_dir = root / config["artifacts"]["output_dir"]

    if args.tune_only:
        predictions = pd.read_parquet(output_dir / "oof_predictions.parquet")
    else:
        predictions = run_cross_validation(args.config, args.folds, args.device)

    sweep, global_threshold, per_category = tune_thresholds(predictions)

    coarse = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    print("\nGLOBAL THRESHOLD SWEEP (out-of-fold)")
    print(sweep[sweep.threshold.isin(coarse)]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nbest global threshold: {global_threshold}")
    print(f"  macro-F1 at 0.5:   {float(sweep[sweep.threshold == 0.5].macro_f1.iloc[0]):.4f}")
    print(f"  macro-F1 at best:  {float(sweep.macro_f1.max()):.4f}")
    print(f"  per-category:      {per_category.oof_f1.mean():.4f}")

    per_category.to_csv(output_dir / "tuned_thresholds.csv", index=False)
    (output_dir / "thresholds.json").write_text(json.dumps({
        "global_threshold": global_threshold,
        "per_category": dict(zip(per_category.category, per_category.threshold)),
        "tuned_on": "out-of-fold cross-validation over the official train split",
        "note": "Test data was not used to select these thresholds.",
    }, indent=2))
    print(f"\nwrote tuned_thresholds.csv and thresholds.json to {output_dir}")

    if args.backup_to:
        backup = Path(args.backup_to) / output_dir.name
        backup.mkdir(parents=True, exist_ok=True)
        for name in ("oof_predictions.parquet", "tuned_thresholds.csv", "thresholds.json"):
            source = output_dir / name
            if source.exists():
                shutil.copy2(source, backup / name)
        print(f"backed up to {backup}")


if __name__ == "__main__":
    _main()
