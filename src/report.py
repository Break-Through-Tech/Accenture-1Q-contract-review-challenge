"""Per-contract review reports (TXT and JSON) built from the results DB.

    python -m src.report --contract "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
    python -m src.report --split test               # every scored test contract
    python -m src.report --all --format json
    python -m src.pipeline run --file c.txt --report   # straight after the pipeline

Each report has:

* a header saying which contract, model run and rules version produced it
* a summary line (a fixed template for now; #56 replaces it with an LLM overview
  and keeps the template as the fallback)
* FLAGGED FOR REVIEW: the clauses and contract-level issues a person should look
  at, each with the reason it was flagged
* every detected clause, highest risk first, with its signals, the restrictive
  keywords that matched, money amounts normalized to one value ("$1M" and
  "$1,000,000" are the same amount, see src/risk/amounts.py), its location, and
  the snippet most likely to be the clause wording
* what the model does not check (categories outside its scope, weak categories)

Locations are character offsets into the contract text. Page and line numbers
need the cleaner to keep page markers (#55) and file inputs that keep pages (#53);
until then the ``page`` and ``line`` fields are null.

Reports quote contract text, so they are written to ``results/reports/``, which
is gitignored.
"""

import argparse
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from src.db import store
from src.db.runs import resolve_run_id
from src.risk.amounts import find_amounts, group_amounts
from src.risk.evidence import best_excerpt, snippet_span
from src.risk.scorer import DEFAULT_RULES_PATH, load_rules

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "reports"
DEFAULT_RUN_DIR = ROOT / "models" / "distilroberta-cuad"
REPORT_VERSION = "1"

# A detection whose probability is within this much of the threshold is flagged:
# the model was nearly undecided about whether the clause is there at all.
BORDERLINE_MARGIN = 0.10
LEVEL_ORDER = {"High": 0, "Medium": 1, "Low": 2}


# --- per-clause details ------------------------------------------------------------

def matched_keywords(category, text, rules):
    """The restrictive keywords from the rules that appear in `text` (the same
    matching the scorer's keyword signal uses)."""
    keyword_rules = rules["keywords"]
    if category in keyword_rules.get("skip_categories", []):
        return []
    keywords = keyword_rules["global"] + keyword_rules["by_category"].get(category, [])
    lowered = (text or "").lower()
    return [k for k in keywords if k.lower() in lowered]


def clause_review_reasons(clause, rules, threshold):
    """Why a detected clause needs a person to look at it. Empty if it doesn't.

    Low-risk clauses are never flagged: a shaky detection of a routine clause is
    not worth a reviewer's time.
    """
    if clause["risk_level"] == "Low":
        return []
    signals = clause["signals"]
    reasons = []
    if clause["risk_level"] == "High":
        reasons.append({
            "code": "high_risk",
            "reason": f"High risk (score {clause['risk_score']:.2f}); read before signing.",
        })
    exception = rules["protective_exceptions"].get(clause["category"])
    if signals.get("missing_exception") and exception:
        reasons.append({
            "code": "missing_protection",
            "reason": f"No '{exception}' clause was found anywhere in the contract to limit it.",
        })
    if clause["probability"] < threshold + BORDERLINE_MARGIN:
        reasons.append({
            "code": "borderline_detection",
            "reason": (f"Model confidence {clause['probability']:.2f} is close to the "
                       f"{threshold:.2f} threshold; confirm the clause is really there."),
        })
    if clause["category"] in rules.get("caveats", {}).get("weak_detection", []):
        reasons.append({
            "code": "weak_category",
            "reason": "The model is unreliable on this category (test F1 below 0.55); "
                      "check the wording yourself.",
        })
    return reasons


def _amounts(evidence, chunk_start):
    """Money amounts in the evidence chunk, one entry per normalized value, so
    "$1M" and "$1,000,000" show up as the same amount (#52)."""
    found = find_amounts(evidence)
    groups = group_amounts(found)
    for group in groups:
        group["occurrences"] = [
            {"text": a.text,
             "start_char": None if chunk_start is None else int(chunk_start) + a.start,
             "end_char": None if chunk_start is None else int(chunk_start) + a.end}
            for a in found if (a.currency, a.value) == (group["currency"], group["value"])]
    return groups


def _clause_entry(row, rules, threshold):
    signals = json.loads(row["signal_scores"]) if row["signal_scores"] else {}
    evidence = row["evidence_text"] or ""
    start, end = row["start_char"], row["end_char"]
    span = snippet_span(evidence, row["category"], rules)
    snippet_start = snippet_end = None
    if span is not None and start is not None:
        snippet_start, snippet_end = int(start) + span[0], int(start) + span[1]

    clause = {
        "clause_id": row["clause_id"],
        "category": row["category"],
        "risk_level": row["risk_level"],
        "risk_score": float(row["risk_score"]),
        "probability": float(signals.get("model_confidence", float("nan"))),
        "signals": signals,
        "matched_keywords": matched_keywords(row["category"], evidence, rules),
        "location": {
            "chunk_id": row["chunk_id"],
            "chunk_start_char": None if start is None else int(start),
            "chunk_end_char": None if end is None else int(end),
            "snippet_start_char": snippet_start,
            "snippet_end_char": snippet_end,
            "page": None,      # #54 / #55
            "line": None,
        },
        "snippet": best_excerpt(evidence, row["category"], rules),
        "amounts": _amounts(evidence, start),
        "evidence_text": evidence,
    }
    clause["review_reasons"] = clause_review_reasons(clause, rules, threshold)
    return clause


# --- whole report ------------------------------------------------------------------

def build_report(conn, contract_id, run_id, rules):
    """Everything in one contract's report, as a JSON-ready dict."""
    version = rules["version"]
    contract = store.read_contract(conn, contract_id)
    if contract is None:
        raise ValueError(f"contract {contract_id!r} is not in the DB")
    run = store.read_model_run(conn, run_id)
    if run is None:
        raise ValueError(f"model run {run_id!r} is not in the DB")

    triage_all = store.read_triage(conn, run_id, version)
    mine = triage_all[triage_all["contract_id"] == contract_id]
    if mine.empty:
        raise ValueError(f"{contract_id!r} has no triage score for {run_id} / rules {version}; "
                         "run `python -m src.pipeline score` first")
    triage = mine.iloc[0]
    split_size = int((triage_all["split"] == triage["split"]).sum())

    threshold = float(run["threshold"])
    register = store.read_clause_register(conn, run_id, version, [contract_id])
    clauses = [_clause_entry(row, rules, threshold)
               for row in register.to_dict("records")]
    clauses.sort(key=lambda c: (LEVEL_ORDER.get(c["risk_level"], 3), -c["risk_score"]))

    flags = []
    if not clauses:
        flags.append({"code": "nothing_detected", "category": None, "clause_id": None,
                      "reason": "No clause categories were detected. The text may be "
                                "unreadable or not a contract; review it manually."})
    must_review = rules["triage"]["always_review_if_high_count"]
    if int(triage["n_high"]) >= must_review:
        flags.append({"code": "must_review_contract", "category": None, "clause_id": None,
                      "reason": f"{int(triage['n_high'])} High-risk clauses (at least "
                                f"{must_review} puts a contract at the top of the queue)."})
    for clause in clauses:
        for reason in clause["review_reasons"]:
            flags.append({"category": clause["category"], "clause_id": clause["clause_id"],
                          "risk_level": clause["risk_level"], **reason})

    counts = {level: sum(c["risk_level"] == level for c in clauses) for level in LEVEL_ORDER}
    flagged_clauses = sum(bool(c["review_reasons"]) for c in clauses)
    summary = (
        f"Review rank {int(triage['review_rank'])} of {split_size} {triage['split']} contracts "
        f"(triage score {float(triage['triage_score']):g}). "
        f"{counts['High']} High, {counts['Medium']} Medium and {counts['Low']} Low risk "
        f"clauses detected; {flagged_clauses} flagged for review."
    )

    caveats = rules.get("caveats", {})
    return {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contract": {k: contract[k] for k in ("contract_id", "title", "split", "source", "n_chars")},
        "model_run": {"run_id": run_id, "model_name": run["model_name"],
                      "threshold": threshold, "git_commit": run["git_commit"]},
        "rules_version": version,
        "triage": {
            "triage_score": float(triage["triage_score"]),
            "review_rank": int(triage["review_rank"]),
            "ranked_among": split_size,
            "split": triage["split"],
            "n_high": counts["High"],
            "n_medium": counts["Medium"],
            "n_low": counts["Low"],
        },
        "summary": summary,
        "summary_source": "template",       # #56 sets "llm" when the overview model writes it
        "review_flags": flags,
        "clauses": clauses,
        "not_checked": {
            "outside_model_scope": caveats.get("outside_model_scope", []),
            "weak_detection": caveats.get("weak_detection", []),
            "page_numbers": "Not available yet: the cleaner removes page numbers (#55).",
        },
    }


# --- rendering ---------------------------------------------------------------------

def _wrap(text, indent, width=96):
    return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent) \
        if text else indent.rstrip()


def render_text(report, width=96):
    c, run, t = report["contract"], report["model_run"], report["triage"]
    out = ["=" * width, "CONTRACT REVIEW REPORT", "=" * width,
           f"Contract:   {c['title'] or c['contract_id']}",
           f"Source:     {c['source']}  |  split {c['split']}  |  {c['n_chars']:,} characters",
           f"Model run:  {run['run_id']}  (threshold {run['threshold']:.2f})",
           f"Rules:      {report['rules_version']}",
           f"Generated:  {report['generated_at']}",
           "", "SUMMARY", _wrap(report["summary"], "  ", width)]

    # One entry per contract-level flag and per flagged clause, reasons as bullets.
    entries = [("[CONTRACT]", [f["reason"]]) for f in report["review_flags"]
               if not f.get("clause_id")]
    entries += [(f"[{c['risk_level'].upper()}] {c['category']}",
                 [r["reason"] for r in c["review_reasons"]])
                for c in report["clauses"] if c["review_reasons"]]
    out += ["", f"FLAGGED FOR REVIEW ({len(entries)})"]
    if not entries:
        out.append("  Nothing flagged.")
    for i, (label, reasons) in enumerate(entries, 1):
        out.append(f"  {i:>2}. {label}")
        for reason in reasons:
            out.append(textwrap.fill(reason, width, initial_indent="      - ",
                                     subsequent_indent="        "))

    out += ["", f"CLAUSES DETECTED ({len(report['clauses'])}, highest risk first)"]
    for clause in report["clauses"]:
        loc, s = clause["location"], clause["signals"]
        where = "unknown"
        if loc["snippet_start_char"] is not None:
            where = f"chars {loc['snippet_start_char']:,}-{loc['snippet_end_char']:,}"
        red_flags = []
        if clause["matched_keywords"]:
            # Matched anywhere in the ~page-long evidence chunk, not only the snippet.
            red_flags.append("keywords in evidence chunk: "
                             + ", ".join(f'"{k}"' for k in clause["matched_keywords"]))
        if s.get("missing_exception"):
            red_flags.append("no protective counterpart")
        out += [
            "",
            f"  [{clause['risk_level'].upper():<6}] {clause['category']}"
            + ("   << flagged" if clause["review_reasons"] else ""),
            f"           risk {clause['risk_score']:.3f} | confidence {clause['probability']:.3f}"
            f" | {where} | page n/a",
            f"           red flags: {'; '.join(red_flags) if red_flags else 'none'}",
        ]
        if clause["amounts"]:
            amounts = "; ".join(
                f"{g['normalized']} (written " + ", ".join(f'"{w}"' for w in g["written_as"]) + ")"
                for g in clause["amounts"])
            out.append(textwrap.fill(f"amounts in evidence chunk: {amounts}", width,
                                     initial_indent="           ",
                                     subsequent_indent="             "))
        out.append(_wrap(f'"{clause["snippet"]}"', "           ", width))

    nc = report["not_checked"]
    out += ["", "NOT CHECKED",
            _wrap("The model does not look for these CUAD categories: "
                  + ", ".join(nc["outside_model_scope"]) + ".", "  ", width),
            _wrap("Detection is weak for: " + ", ".join(nc["weak_detection"])
                  + ". A missing clause in these categories is not proof it is absent.",
                  "  ", width),
            _wrap(nc["page_numbers"], "  ", width),
            "=" * width]
    return "\n".join(out) + "\n"


def report_filename(contract_id):
    return re.sub(r"[^\w.-]+", "_", contract_id).strip("_")[:150] or "contract"


def write_reports(conn, contract_ids, run_id, rules, out_dir=DEFAULT_OUT_DIR,
                  formats=("txt", "json"), log=print):
    """Write one report per contract. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = {}
    for contract_id in contract_ids:
        other = stems.setdefault(report_filename(contract_id).lower(), contract_id)
        if other != contract_id:
            raise ValueError(f"report filename clash: {other!r} and {contract_id!r}")
    written = []
    for contract_id in contract_ids:
        report = build_report(conn, contract_id, run_id, rules)
        # Append, never with_suffix: names like "EX-10.3-WEB_SITE" contain dots.
        stem = report_filename(contract_id)
        if "txt" in formats:
            path = out_dir / f"{stem}.txt"
            path.write_text(render_text(report), encoding="utf-8")
            written.append(path)
        if "json" in formats:
            path = out_dir / f"{stem}.json"
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            written.append(path)
        log(f"  report  {len(report['review_flags']):>2} flags  {contract_id[:70]}")
    return written


# --- CLI -----------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.report",
                                     description="Write TXT/JSON review reports from the results DB.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--contract", action="append", metavar="ID", help="contract id (repeatable)")
    which.add_argument("--split", help="every scored contract in this split (test, train, external)")
    which.add_argument("--all", action="store_true", help="every scored contract")
    parser.add_argument("--format", choices=["txt", "json", "both"], default="both")
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output folder")
    parser.add_argument("--db", default=str(store.DEFAULT_DB_PATH))
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)

    conn = store.connect(args.db)
    try:
        rules = load_rules(args.rules)
        try:
            run_id = resolve_run_id(conn, args.run_id, args.run_dir)
        except ValueError as error:
            raise SystemExit(str(error))
        triage = store.read_triage(conn, run_id, rules["version"])
        if args.contract:
            contract_ids = args.contract
        elif args.split:
            contract_ids = triage.loc[triage["split"] == args.split, "contract_id"].tolist()
        else:
            contract_ids = triage["contract_id"].tolist()
        if not contract_ids:
            raise SystemExit(f"no scored contracts for {run_id} / rules {rules['version']}")
        formats = ("txt", "json") if args.format == "both" else (args.format,)
        try:
            written = write_reports(conn, contract_ids, run_id, rules, args.out, formats)
        except ValueError as error:
            raise SystemExit(str(error))
        print(f"\n{len(written)} file(s) in {args.out}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
