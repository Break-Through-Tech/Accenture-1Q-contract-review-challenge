"""End-to-end demo: one contract in, a risk-ranked clause register out.

    python -m src.demo path/to/contract.txt
    python -m src.demo --sample 3                  # use a CUAD test contract
    python -m src.demo contract.txt --threshold 0.25

Chains the pieces that already exist:

    contract text
      -> src.models.predict      which of the 34 clause categories are present
      -> src.risk.scorer         Low / Medium / High per detected clause
      -> src.risk.triage         one score and rank for the whole contract

The evidence text for each clause is the chunk the model was most confident about,
with its character offsets, so a reviewer can go read the source wording.
"""

import argparse
import json
import textwrap
from pathlib import Path

import pandas as pd

from src.models.predict import load_model, predict_chunks
from src.risk.evidence import best_excerpt
from src.risk.scorer import load_rules, score_clauses
from src.risk.triage import triage_contracts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "models" / "distilroberta-cuad"


def analyse(text, contract_id, run_dir=DEFAULT_RUN, threshold=None, rules_path=None,
            model=None):
    """Run the full pipeline over one contract. Returns (clauses, triage, predictions)."""
    model = load_model(run_dir) if model is None else model
    threshold = model.threshold if threshold is None else threshold

    # Score every chunk, then keep each category's most confident chunk as its evidence.
    chunk_scores = predict_chunks(text, model)
    if chunk_scores.empty:
        raise ValueError("contract produced no chunks")

    best = chunk_scores[model.categories].idxmax()
    predictions = pd.DataFrame({
        "contract_id": contract_id,
        "category": model.categories,
        "probability": chunk_scores[model.categories].max().to_numpy(),
        "chunk_id": [f"{contract_id}__{i}" for i in chunk_scores.loc[best, "chunk_index"]],
        "start_char": chunk_scores.loc[best, "start_char"].to_numpy(),
        "end_char": chunk_scores.loc[best, "end_char"].to_numpy(),
    })
    predictions["predicted_label"] = predictions["probability"] >= threshold
    predictions["evidence_text"] = [
        text[int(s):int(e)] for s, e in zip(predictions.start_char, predictions.end_char)
    ]

    rules = load_rules(rules_path) if rules_path else load_rules()
    clauses = score_clauses(predictions, rules, run_id="demo")
    triage = triage_contracts(clauses, rules, contract_ids=[contract_id])
    return clauses.sort_values("risk_score", ascending=False), triage, predictions, rules


def _print_report(clauses, triage, predictions, text, contract_id, threshold,
                  rules=None, width=96):
    print("=" * width)
    print(f"CONTRACT: {contract_id[:80]}")
    print(f"{len(text):,} characters | threshold {threshold}")
    print("=" * width)

    if not triage.empty:
        row = triage.iloc[0]
        print(f"\nTRIAGE SCORE: {row.triage_score}   "
              f"({row.n_high} High, {row.n_medium} Medium clauses)")

    found = predictions[predictions.predicted_label]
    print(f"\nCLAUSES DETECTED: {len(found)} of {len(predictions)} categories\n")

    if clauses.empty:
        print("  none above the threshold - try a lower --threshold")
        return

    for row in clauses.itertuples(index=False):
        probability = float(
            predictions.loc[predictions.category == row.category, "probability"].iloc[0])
        print(f"[{row.risk_level.upper():<6}] {row.category}")
        print(f"          risk {row.risk_score:.3f} | model confidence {probability:.3f} "
              f"| chars {row.start_char}-{row.end_char}")
        evidence = (best_excerpt(row.evidence_text, row.category, rules) if rules
                    else " ".join(str(row.evidence_text or "").split())[:240])
        for line in textwrap.wrap(evidence, width - 10) or [""]:
            print(f"          {line}")
        print()


def _main():
    parser = argparse.ArgumentParser(description="Analyse one contract end to end")
    parser.add_argument("contract", nargs="?", help="path to a .txt contract")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="use CUAD test contract N instead of a file")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--rules", default=None)
    parser.add_argument("--save", metavar="DIR", help="write the clause register as CSV")
    args = parser.parse_args()

    if args.sample is not None:
        data = json.loads((ROOT / "data/cuad/test.json").read_text(encoding="utf-8"))
        document = data["data"][args.sample]
        text, contract_id = document["paragraphs"][0]["context"], document["title"]
    elif args.contract:
        path = Path(args.contract)
        text, contract_id = path.read_text(encoding="utf-8", errors="replace"), path.stem
    else:
        parser.error("give a contract path or --sample N")

    # Load once: the encoder is ~330 MB.
    model = load_model(Path(args.run_dir))
    threshold = args.threshold if args.threshold is not None else model.threshold
    clauses, triage, predictions, rules = analyse(
        text, contract_id, Path(args.run_dir), args.threshold, args.rules, model=model)
    _print_report(clauses, triage, predictions, text, contract_id, threshold, rules)

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        clauses.to_csv(out / f"{contract_id[:60]}_clauses.csv", index=False)
        triage.to_csv(out / f"{contract_id[:60]}_triage.csv", index=False)
        print(f"saved to {out}")


if __name__ == "__main__":
    _main()
