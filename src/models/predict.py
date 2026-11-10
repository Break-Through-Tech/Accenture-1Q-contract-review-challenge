"""Run a trained model over a contract and report which clause categories it finds.

    from src.models.predict import load_model, predict_contract

    model = load_model("models/distilroberta-cuad")
    found = predict_contract(open("some_contract.txt").read(), model)

Or from the command line:

    python -m src.models.predict path/to/contract.txt

The contract is cleaned and chunked the same way as in training, every chunk is
scored, and the chunk probabilities are pooled to one probability per category by
taking the maximum. That matches how `trainer.py` scores the test set: a clause
sitting inside a chunk overlap appears in two chunks and must not count twice.
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from transformers import AutoTokenizer

from src.clean.light import clean_transformer_text
from src.ingest.chunker import chunk_contract
from src.models.transformer import MultiLabelTransformer, predict_probabilities


class LoadedModel(NamedTuple):
    model: MultiLabelTransformer
    tokenizer: object
    categories: list
    threshold: float
    device: torch.device


def load_model(run_dir, device="auto") -> LoadedModel:
    """Load a trained run: the encoder, the classifier head, and its category list."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    categories = metadata["categories"]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Build the same architecture, then load the fine-tuned weights over it.
    model = MultiLabelTransformer(num_labels=len(categories),
                                  model_name=metadata["model_name"])
    model.encoder = type(model.encoder).from_pretrained(run_dir / "encoder")
    model.classifier.load_state_dict(
        torch.load(run_dir / "classifier.pt", map_location="cpu"))
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
    return LoadedModel(model, tokenizer, categories,
                       float(metadata["threshold"]), device)


def predict_chunks(text, loaded: LoadedModel, batch_size: int = 8) -> pd.DataFrame:
    """Score every chunk of `text`. One row per chunk, one column per category."""
    cleaned = clean_transformer_text(text)      # clean first, then chunk, as in training
    chunks = chunk_contract(cleaned)
    if not chunks:
        return pd.DataFrame(columns=["chunk_index", "start_char", "end_char"] + loaded.categories)

    rows = []
    with torch.inference_mode():
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            encoded = loaded.tokenizer([c["text"] for c in batch], padding=True,
                                       return_tensors="pt", verbose=False)
            encoded = {k: v.to(loaded.device) for k, v in encoded.items()}
            logits = loaded.model(**encoded)["logits"]
            rows.append(predict_probabilities(logits).cpu().numpy())

    probabilities = pd.DataFrame(sum((list(r) for r in rows), []), columns=loaded.categories)
    meta = pd.DataFrame([{"chunk_index": c["chunk_index"],
                          "start_char": c["start_char"],
                          "end_char": c["end_char"]} for c in chunks])
    return pd.concat([meta, probabilities], axis=1)


def predict_contract(text, loaded: LoadedModel, threshold=None,
                     only_found: bool = True) -> pd.DataFrame:
    """Which clause categories are in this contract?

    Returns one row per category with its probability, whether it clears the
    threshold, and the chunk that scored highest, so a reviewer can go read it.
    """
    threshold = loaded.threshold if threshold is None else threshold
    chunk_scores = predict_chunks(text, loaded)
    if chunk_scores.empty:
        return chunk_scores

    best_row = chunk_scores[loaded.categories].idxmax()
    result = pd.DataFrame({
        "category": loaded.categories,
        "probability": chunk_scores[loaded.categories].max().to_numpy(),
        "best_chunk": chunk_scores.loc[best_row, "chunk_index"].to_numpy(),
        "start_char": chunk_scores.loc[best_row, "start_char"].to_numpy(),
        "end_char": chunk_scores.loc[best_row, "end_char"].to_numpy(),
    })
    result["found"] = result["probability"] >= threshold
    result = result.sort_values("probability", ascending=False).reset_index(drop=True)
    return result[result["found"]] if only_found else result


def _main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    contract_path = Path(sys.argv[1])
    run_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models/distilroberta-cuad")

    text = contract_path.read_text(encoding="utf-8", errors="replace")
    loaded = load_model(run_dir)
    found = predict_contract(text, loaded, only_found=False)

    print(f"{contract_path.name}: {len(text):,} characters, "
          f"threshold {loaded.threshold}, device {loaded.device}\n")
    hits = found[found["found"]]
    print(f"clauses found: {len(hits)} of {len(found)}\n")
    print(found.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    _main()
