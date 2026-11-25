import json

import pytest

import src.models.predict as predict
import src.pipeline as pipeline
from src.report import build_report, render_text, report_filename, write_reports
from src.risk.evidence import best_excerpt, snippet_span
from src.risk.scorer import load_rules

# Fakes and fixtures shared with the pipeline tests (pytest puts tests/ on sys.path).
from test_pipeline import (  # noqa: F401  (fixtures are used by name)
    CATEGORIES, QUIET, RISKY, FakeModel, conn, contract, fakes, model, quiet, run_dir,
)


@pytest.fixture
def rules():
    return load_rules()


def scored(conn, run_dir, model, *contracts):
    run_id, _ = pipeline.run(conn, list(contracts), run_dir, loaded=model, log=quiet)
    return run_id


# --- snippets ---------------------------------------------------------------------

def test_snippet_span_indexes_the_text_and_finds_the_clause(rules):
    text = ("Intro text here. The Distributor shall not compete with the Company worldwide; "
            "this survives.\n\n Payment terms: net 30.")
    start, end = snippet_span(text, "Non-Compete", rules, width=40)
    assert text[start:end] == "The Distributor shall not compete with the Company worldwide;"


def test_snippet_ignores_non_in_category_names(rules):
    text = ("The Distributor may not trade on non-public information. "
            "Distributor shall not compete with Company.")
    start, end = snippet_span(text, "Non-Compete", rules, width=10)
    assert "compete" in text[start:end]
    assert "non-public" not in text[start:end]


def test_best_excerpt_handles_empty_and_unpunctuated_text(rules):
    assert best_excerpt("", "Non-Compete", rules) == ""
    assert best_excerpt("no punctuation at all", "Zzz", rules, width=5) == "no punctuation at all"


# --- report content -----------------------------------------------------------------

def test_report_flags_high_risk_and_missing_protection(conn, run_dir, model, rules):
    run_id = scored(conn, run_dir, model, contract("risky", RISKY))
    report = build_report(conn, "risky", run_id, rules)

    by_category = {c["category"]: c for c in report["clauses"]}
    codes = {r["code"] for r in by_category["Non-Compete"]["review_reasons"]}
    assert codes == {"high_risk", "missing_protection"}
    assert by_category["Governing Law"]["review_reasons"] == []      # Low is never flagged
    assert by_category["Non-Compete"]["matched_keywords"] == ["worldwide"]
    assert report["summary_source"] == "template"
    assert report["model_run"]["run_id"] == run_id
    assert report["rules_version"] == rules["version"]


def test_report_snippet_offsets_point_into_the_contract(conn, run_dir, model, rules):
    run_id = scored(conn, run_dir, model, contract("risky", RISKY))
    for clause in build_report(conn, "risky", run_id, rules)["clauses"]:
        loc = clause["location"]
        source = RISKY[loc["snippet_start_char"]:loc["snippet_end_char"]]
        assert " ".join(source.split()) in clause["snippet"]
        assert loc["page"] is None and loc["line"] is None


def test_borderline_detection_is_flagged(conn, tmp_path, rules):
    # The report reads the threshold from run_metadata.json, as load_model does.
    run_dir = tmp_path / "strict-run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(json.dumps(
        {"model_name": "fake", "threshold": 0.85, "categories": CATEGORIES}))
    (run_dir / "classifier.pt").write_bytes(b"strict")
    shaky = FakeModel(CATEGORIES, 0.85)          # Non-Compete scores 0.9: just over
    run_id = scored(conn, run_dir, shaky, contract("risky", RISKY))
    report = build_report(conn, "risky", run_id, rules)
    non_compete = next(c for c in report["clauses"] if c["category"] == "Non-Compete")
    assert "borderline_detection" in {r["code"] for r in non_compete["review_reasons"]}


def test_contract_with_nothing_detected_is_flagged(conn, run_dir, model, rules):
    run_id = scored(conn, run_dir, model, contract("blank", "Lorem ipsum dolor sit amet."))
    report = build_report(conn, "blank", run_id, rules)
    assert report["clauses"] == []
    assert [f["code"] for f in report["review_flags"]] == ["nothing_detected"]
    assert "Nothing flagged" not in render_text(report)


def test_report_lists_amounts_normalized(conn, run_dir, model, rules):
    text = "Seller shall not compete worldwide for $1M, being $1,000,000. Governed by law."
    run_id = scored(conn, run_dir, model, contract("paid", text))
    report = build_report(conn, "paid", run_id, rules)
    non_compete = next(c for c in report["clauses"] if c["category"] == "Non-Compete")
    [amount] = non_compete["amounts"]
    assert (amount["normalized"], amount["written_as"]) == ("USD 1,000,000", ["$1M", "$1,000,000"])
    for occurrence in amount["occurrences"]:
        assert text[occurrence["start_char"]:occurrence["end_char"]] == occurrence["text"]
    assert 'USD 1,000,000 (written "$1M", "$1,000,000")' in render_text(report)


def test_report_rank_is_within_split(conn, run_dir, model, rules):
    run_id = scored(conn, run_dir, model, contract("risky", RISKY),
                    contract("quiet", QUIET), contract("seen", RISKY, split="train"))
    triage = build_report(conn, "seen", run_id, rules)["triage"]
    assert (triage["review_rank"], triage["ranked_among"], triage["split"]) == (1, 1, "train")


def test_report_needs_scores(conn, rules):
    with pytest.raises(ValueError, match="not in the DB"):
        build_report(conn, "missing", "run", rules)


def test_text_report_groups_reasons_per_clause(conn, run_dir, model, rules):
    run_id = scored(conn, run_dir, model, contract("risky", RISKY))
    text = render_text(build_report(conn, "risky", run_id, rules))
    flagged = text.split("FLAGGED FOR REVIEW")[1].split("CLAUSES DETECTED")[0]
    assert flagged.count("Non-Compete") == 1
    assert "read before signing" in flagged and "Competitive Restriction Exception" in flagged


# --- files ----------------------------------------------------------------------------

def test_filenames_keep_dots_and_stay_distinct(conn, run_dir, model, rules, tmp_path):
    ids = ["ACME-EX-10.3-WEB SITE AGREEMENT", "ACME-EX-10.3-OTHER AGREEMENT"]
    run_id = scored(conn, run_dir, model, *(contract(i, RISKY) for i in ids))
    written = write_reports(conn, ids, run_id, rules, tmp_path, log=quiet)
    names = sorted(p.name for p in written)
    assert names == ["ACME-EX-10.3-OTHER_AGREEMENT.json", "ACME-EX-10.3-OTHER_AGREEMENT.txt",
                     "ACME-EX-10.3-WEB_SITE_AGREEMENT.json", "ACME-EX-10.3-WEB_SITE_AGREEMENT.txt"]
    data = json.loads((tmp_path / "ACME-EX-10.3-OTHER_AGREEMENT.json").read_text(encoding="utf-8"))
    assert data["contract"]["contract_id"] == ids[1]


def test_filename_clash_is_an_error(conn, run_dir, model, rules, tmp_path):
    ids = ["a/b", "a?b"]
    assert report_filename(ids[0]) == report_filename(ids[1])
    run_id = scored(conn, run_dir, model, *(contract(i, RISKY) for i in ids))
    with pytest.raises(ValueError, match="clash"):
        write_reports(conn, ids, run_id, rules, tmp_path, log=quiet)


def test_cli_run_with_report(tmp_path, run_dir, model, monkeypatch, capsys):
    contract_file = tmp_path / "acme.txt"
    contract_file.write_text(RISKY, encoding="utf-8")
    monkeypatch.setattr(predict, "load_model", lambda run_dir: model)
    pipeline.main(["--db", str(tmp_path / "r.db"), "run", "--file", str(contract_file),
                   "--run-dir", str(run_dir), "--report", "--report-dir", str(tmp_path / "out")])
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["acme.json", "acme.txt"]
    assert "2 report file(s)" in capsys.readouterr().out
