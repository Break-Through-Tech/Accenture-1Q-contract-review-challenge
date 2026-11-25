"""End-to-end pipeline: contracts in, clause register and triage scores in the results DB.

Three stages, each runnable on its own. Each one reads what the previous stage
wrote to ``results/cuad_results.db``, so a failure late in the chain never means
redoing the expensive part.

    ingest   contract text      -> contract, chunk          (clean, then chunk, as in training)
    infer    chunk              -> model_run, prediction    (needs the model checkpoint)
    score    prediction, chunk  -> clause_register, triage_score   (needs risk_rules.yaml only)

Run from the repository root:

    python -m src.pipeline run --cuad test:0 --cuad train:0      # all three stages
    python -m src.pipeline run --file path/to/contract.txt
    python -m src.pipeline ingest --cuad test:all                # just chunk the test split
    python -m src.pipeline infer                                 # predict whatever has no predictions yet
    python -m src.pipeline score                                 # re-score everything, e.g. after editing the rules
    python -m src.pipeline status                                # what is in the DB

A stage skips contracts it has already done for the same model run (and, for
scoring, the same rules version is simply replaced). Pass ``--force`` to redo work.

Every clause and triage row is stamped with ``run_id`` (which checkpoint flagged
it, see ``src/db/runs.py``) and ``rules_version`` (which version of
``configs/risk_rules.yaml`` scored it).
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.clean.light import clean_transformer_text
from src.db import store
from src.db.runs import register_model_run, resolve_run_id
from src.ingest.chunker import chunk_contract
from src.risk.evidence import extract_evidence
from src.risk.scorer import DEFAULT_RULES_PATH, load_rules, score_clauses
from src.risk.triage import triage_contracts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "models" / "distilroberta-cuad"
CUAD_FILES = {
    "train": ROOT / "data" / "cuad" / "train_separate_questions.json",
    "test": ROOT / "data" / "cuad" / "test.json",
}


# --- sources -------------------------------------------------------------------

def _cuad_documents(split, _cache={}):
    if split not in _cache:
        if split not in CUAD_FILES:
            raise ValueError(f"unknown CUAD split {split!r}; use train or test")
        data = json.loads(CUAD_FILES[split].read_text(encoding="utf-8"))
        _cache[split] = data["data"]
    return _cache[split]


def cuad_contracts(spec):
    """Contracts for ``SPLIT:INDEX``, ``SPLIT:START-END`` (inclusive) or ``SPLIT:all``.

    The contract id is the CUAD title, the same id used in chunks.parquet.
    """
    split, _, which = spec.partition(":")
    documents = _cuad_documents(split)
    if which == "all":
        indices = range(len(documents))
    elif "-" in which:
        start, end = (int(part) for part in which.split("-"))
        indices = range(start, end + 1)
    else:
        indices = [int(which)]
    for index in indices:
        document = documents[index]
        yield {
            "contract_id": document["title"],
            "title": document["title"],
            "split": split,
            "source": f"cuad:{split}:{index}",
            "text": document["paragraphs"][0]["context"],
        }


def file_contract(path):
    path = Path(path)
    return {
        "contract_id": path.stem,
        "title": path.stem,
        "split": "external",
        "source": str(path),
        "text": path.read_text(encoding="utf-8", errors="replace"),
    }


def gather_sources(files=(), cuad_specs=()):
    contracts = [file_contract(f) for f in files]
    for spec in cuad_specs:
        contracts.extend(cuad_contracts(spec))
    return contracts


# --- stage 1: ingest -----------------------------------------------------------

def chunk_frame(contract_id, text) -> pd.DataFrame:
    """Clean, then chunk, exactly as build_dataset.py and predict.py do.

    The cleaner never changes length, so the offsets index the raw text too.
    """
    cleaned = clean_transformer_text(text)
    chunks = chunk_contract(cleaned)
    return pd.DataFrame({
        "chunk_id": [f"{contract_id}__{c['chunk_index']}" for c in chunks],
        "chunk_index": [c["chunk_index"] for c in chunks],
        "start_char": [c["start_char"] for c in chunks],
        "end_char": [c["end_char"] for c in chunks],
        "n_tokens": [c["n_tokens"] for c in chunks],
        "text": [text[c["start_char"]:c["end_char"]] for c in chunks],
        "text_light": [c["text"] for c in chunks],
    })


def _already_ingested(conn, contract_id, n_chars):
    row = conn.execute(
        """SELECT c.n_chars, COUNT(k.chunk_id) FROM contract c
           LEFT JOIN chunk k ON k.contract_id = c.contract_id
           WHERE c.contract_id = ? GROUP BY c.contract_id""", (contract_id,)).fetchone()
    return row is not None and row[0] == n_chars and row[1] > 0


def ingest(conn, contracts, force=False, log=print):
    """Write contracts and their chunks. Returns the ids of the contracts that are
    now in the DB (contracts with no text are left out).

    A contract already in the DB with the same length is skipped, because
    re-chunking it would delete its predictions and clause rows.
    """
    ids = []
    for contract in contracts:
        contract_id, text = contract["contract_id"], contract["text"]
        if not force and _already_ingested(conn, contract_id, len(text)):
            log(f"  ingest  skip  {contract_id[:70]} (already in DB)")
            ids.append(contract_id)
            continue
        chunks = chunk_frame(contract_id, text)
        if chunks.empty:
            log(f"  ingest  WARN  {contract_id[:70]} has no text, skipped")
            continue
        ids.append(contract_id)
        store.upsert_contract(conn, contract_id, title=contract.get("title"),
                              split=contract.get("split"), n_chars=len(text),
                              source=contract.get("source"))
        store.replace_chunks(conn, contract_id, chunks)
        log(f"  ingest  {len(chunks):>4} chunks  {contract_id[:70]}")
    return ids


# --- stage 2: infer ------------------------------------------------------------

def long_predictions(chunk_ids, probabilities: pd.DataFrame, threshold) -> pd.DataFrame:
    """Wide (chunk x category) probabilities -> one row per chunk and category."""
    wide = probabilities.copy()
    wide.insert(0, "chunk_id", list(chunk_ids))
    long = wide.melt(id_vars="chunk_id", var_name="category", value_name="probability")
    long["predicted"] = long["probability"] >= threshold
    return long


def infer(conn, run_dir=DEFAULT_RUN_DIR, contract_ids=None, force=False, batch_size=8,
          loaded=None, log=print):
    """Predict every chunk of each contract that has no predictions for this run.

    ``contract_ids=None`` means every contract in the DB. The model is loaded only
    if there is something to predict. Returns the run_id.
    """
    run_id = register_model_run(conn, run_dir)
    if contract_ids is None:
        contract_ids = store.list_contracts(conn)["contract_id"].tolist()
    done = set() if force else store.contracts_with_predictions(conn, run_id)
    todo = [c for c in contract_ids if c not in done]
    for contract_id in sorted(set(contract_ids) & done):
        log(f"  infer   skip  {contract_id[:70]} (already predicted by {run_id})")
    if not todo:
        return run_id

    if loaded is None:
        from transformers.utils import logging as hf_logging
        from src.models.predict import load_model   # torch import is slow; only when needed
        # Building the model loads distilroberta-base before our fine-tuned weights
        # replace it; its "lm_head UNEXPECTED" load report is expected noise.
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        loaded = load_model(run_dir)
    from src.models.predict import score_texts

    for contract_id in todo:
        chunks = store.read_chunks(conn, [contract_id])
        if chunks.empty:
            log(f"  infer   WARN  {contract_id[:70]} has no chunks; run ingest first")
            continue
        started = time.perf_counter()
        probabilities = score_texts(chunks["text_light"].tolist(), loaded, batch_size)
        predictions = long_predictions(chunks["chunk_id"], probabilities, loaded.threshold)
        store.replace_predictions(conn, contract_id, run_id, predictions)
        n_found = predictions.groupby("category")["predicted"].any().sum()
        log(f"  infer   {len(chunks):>4} chunks  {n_found:>2} categories  "
            f"{time.perf_counter() - started:5.1f}s  {contract_id[:60]}")
    return run_id


# --- stage 3: score ------------------------------------------------------------

def score(conn, run_id, rules=None, contract_ids=None, log=print):
    """Build the clause register and re-rank triage for one model run.

    ``contract_ids=None`` scores every contract with predictions for ``run_id``.
    Triage is always recomputed over all of them, because review_rank is relative
    to the whole set. Contracts are ranked within their split (test, train,
    external), so a train contract the model has already seen never outranks
    test contracts. Returns the triage table with a ``split`` column.
    """
    rules = rules or load_rules()
    version = rules["version"]
    predicted = store.contracts_with_predictions(conn, run_id)
    if contract_ids is None:
        contract_ids = sorted(predicted)
    missing = [c for c in contract_ids if c not in predicted]
    if missing:
        raise ValueError(f"no predictions from {run_id} for {len(missing)} contract(s), "
                         f"e.g. {missing[0]!r}; run infer first")

    if contract_ids:
        evidence = extract_evidence(store.read_predictions(conn, run_id, contract_ids),
                                    store.read_chunks(conn, contract_ids))
        clauses = score_clauses(evidence, rules, run_id=run_id)
        store.replace_clause_register(conn, contract_ids, run_id, version, clauses)
        log(f"  score   {len(clauses)} clauses across {len(contract_ids)} contract(s), "
            f"rules {version}")

    register = store.read_clause_register(conn, run_id, version)
    splits = store.contract_splits(conn, predicted)        # contract_id -> split
    per_split = []
    for ids in splits.groupby(splits).groups.values():
        ids = sorted(ids)
        per_split.append(triage_contracts(register[register["contract_id"].isin(ids)],
                                          rules, contract_ids=ids))
    triage = pd.concat(per_split, ignore_index=True)
    store.replace_triage(conn, run_id, version, triage)
    return store.read_triage(conn, run_id, version)


# --- run all -------------------------------------------------------------------

def run(conn, contracts, run_dir=DEFAULT_RUN_DIR, rules=None, force=False, batch_size=8,
        loaded=None, log=print):
    contract_ids = ingest(conn, contracts, force=force, log=log)
    run_id = infer(conn, run_dir, contract_ids, force=force, batch_size=batch_size,
                   loaded=loaded, log=log)
    triage = score(conn, run_id, rules, contract_ids=contract_ids, log=log)
    return run_id, triage


# --- CLI -----------------------------------------------------------------------

def _print_triage(triage, contract_ids=None, limit=20):
    """One ranked table per split; ranks are only comparable within a split."""
    for split, group in triage.groupby("split", sort=True):
        shown = group if contract_ids is None else group[group["contract_id"].isin(contract_ids)]
        if shown.empty:
            continue
        print(f"\n  {split}: rank out of {len(group)}")
        print("  rank  score  high  med  contract")
        for row in shown.head(limit).itertuples(index=False):
            print(f"  {row.review_rank:>4}  {row.triage_score:>5.1f}  {row.n_high:>4}  "
                  f"{row.n_medium:>3}  {row.contract_id[:70]}")
        if len(shown) > limit:
            print(f"  ... {len(shown) - limit} more")


def _status(conn):
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("contract", "chunk", "model_run", "prediction", "clause_register",
                        "triage_score")}
    print("rows: " + ", ".join(f"{t} {n:,}" for t, n in counts.items()))
    for row in conn.execute(
            """SELECT t.run_id, t.rules_version, COALESCE(c.split, 'unknown'),
                      COUNT(*), SUM(t.n_high), SUM(t.n_medium)
               FROM triage_score t LEFT JOIN contract c ON c.contract_id = t.contract_id
               GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"""):
        print(f"  {row[0]}  rules {row[1]}  {row[2]}: {row[3]} contracts, "
              f"{row[4]} High, {row[5]} Medium clauses")


def _parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Contract review pipeline: ingest -> infer -> score, into the results DB.")
    parser.add_argument("--db", default=str(store.DEFAULT_DB_PATH), help="results DB path")
    commands = parser.add_subparsers(dest="command", required=True)

    def sources(p):
        p.add_argument("--file", action="append", default=[], metavar="PATH",
                       help="a .txt contract (repeatable)")
        p.add_argument("--cuad", action="append", default=[], metavar="SPLIT:N",
                       help="CUAD contract: test:3, train:0-9, test:all (repeatable)")

    def model(p):
        p.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="trained run folder")
        p.add_argument("--batch-size", type=int, default=8)

    def rules(p):
        p.add_argument("--rules", default=str(DEFAULT_RULES_PATH), help="risk rules YAML")

    def contracts(p):
        p.add_argument("--contract", action="append", metavar="ID",
                       help="limit to this contract id (repeatable; default: all)")

    def reports(p):
        p.add_argument("--report", action="store_true",
                       help="also write TXT + JSON reports for these contracts (src/report.py)")
        p.add_argument("--report-dir", default=None, help="report folder (default results/reports)")

    p = commands.add_parser("run", help="ingest, infer and score the given contracts")
    sources(p), model(p), rules(p), reports(p)
    p.add_argument("--force", action="store_true", help="redo stages already done")

    p = commands.add_parser("ingest", help="clean and chunk contracts into the DB")
    sources(p)
    p.add_argument("--force", action="store_true", help="re-chunk contracts already in the DB")

    p = commands.add_parser("infer", help="predict chunks that have no predictions yet")
    model(p), contracts(p)
    p.add_argument("--force", action="store_true", help="re-predict contracts already done")

    p = commands.add_parser("score", help="build the clause register and triage from predictions")
    rules(p), contracts(p), reports(p)
    p.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR),
                   help="checkpoint whose predictions to score (only hashed, not loaded)")
    p.add_argument("--run-id", help="score this run id instead of --run-dir")

    commands.add_parser("status", help="row counts and triage summary")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    conn = store.connect(args.db)
    try:
        if args.command == "status":
            _status(conn)
            return

        if args.command in ("run", "ingest"):
            if not (args.file or args.cuad):
                raise SystemExit("give at least one --file or --cuad")
            missing = [f for f in args.file if not Path(f).is_file()]
            if missing:
                raise SystemExit(f"contract file not found: {', '.join(missing)}")
            contracts = gather_sources(args.file, args.cuad)

        if args.command == "ingest":
            ingest(conn, contracts, force=args.force)
        elif args.command == "infer":
            run_id = infer(conn, Path(args.run_dir), args.contract, force=args.force,
                           batch_size=args.batch_size)
            print(f"\nrun_id {run_id}")
        elif args.command in ("score", "run"):
            rules = load_rules(args.rules)
            if args.command == "score":
                try:
                    run_id = resolve_run_id(conn, args.run_id, args.run_dir)
                except ValueError as error:
                    raise SystemExit(str(error))
                triage = score(conn, run_id, rules, args.contract)
                processed = args.contract or triage["contract_id"].tolist()
            else:
                run_id, triage = run(conn, contracts, Path(args.run_dir), rules,
                                     force=args.force, batch_size=args.batch_size)
                processed = [c["contract_id"] for c in contracts
                             if c["contract_id"] in set(triage["contract_id"])]
            print(f"\nrun_id {run_id}")
            _print_triage(triage, processed)
            if args.report:
                from src.report import DEFAULT_OUT_DIR, write_reports
                out_dir = args.report_dir or DEFAULT_OUT_DIR
                print()
                written = write_reports(conn, processed, run_id, rules, out_dir)
                print(f"{len(written)} report file(s) in {out_dir}")
        print(f"db {args.db}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
