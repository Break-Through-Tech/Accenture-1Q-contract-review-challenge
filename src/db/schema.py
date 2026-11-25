"""Table definitions for ``results/cuad_results.db``.

Follows docs/ARCHITECTURE.md section 3, with four changes #46 needs:

* ``chunk`` keeps the raw ``text`` next to ``text_light``. Evidence shown to a
  reviewer should be the original wording; the light cleaner never changes
  length, so one pair of offsets indexes both.
* ``clause_register`` carries ``start_char`` / ``end_char`` of its evidence, and
  its key includes ``run_id`` and ``rules_version``, so re-scoring with new rules
  adds rows instead of overwriting the old ones.
* ``triage_score`` carries ``run_id`` and is keyed on
  ``(contract_id, run_id, rules_version)`` for the same reason.
* ``model_run`` records the decision ``threshold``, since ``prediction.predicted``
  depends on it.
"""

TABLES = [
    "contract",
    "chunk",
    "model_run",
    "prediction",
    "category_metric",
    "clause_register",
    "advisor_label",
    "triage_score",
]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS contract (
    contract_id  TEXT PRIMARY KEY,
    title        TEXT,
    split        TEXT,              -- train | test | external
    n_chars      INTEGER,
    source       TEXT               -- file path or CUAD split/index it came from
);

CREATE TABLE IF NOT EXISTS chunk (
    chunk_id     TEXT PRIMARY KEY,
    contract_id  TEXT NOT NULL REFERENCES contract(contract_id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    start_char   INTEGER NOT NULL,
    end_char     INTEGER NOT NULL,
    n_tokens     INTEGER,
    text         TEXT,
    text_light   TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_contract ON chunk(contract_id);

CREATE TABLE IF NOT EXISTS model_run (
    run_id        TEXT PRIMARY KEY,
    model_name    TEXT,
    artifact_uri  TEXT,
    git_commit    TEXT,             -- commit of the pipeline code that registered the run
    created_at    TEXT,
    threshold     REAL,
    metadata_json TEXT              -- run_metadata.json as saved by the trainer
);

CREATE TABLE IF NOT EXISTS prediction (
    chunk_id     TEXT NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    run_id       TEXT NOT NULL REFERENCES model_run(run_id),
    category     TEXT NOT NULL,
    probability  REAL NOT NULL,
    predicted    INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, run_id, category)
);
CREATE INDEX IF NOT EXISTS idx_prediction_run ON prediction(run_id);

CREATE TABLE IF NOT EXISTS category_metric (
    run_id           TEXT NOT NULL REFERENCES model_run(run_id),
    category         TEXT NOT NULL,
    precision_score  REAL,
    recall_score     REAL,
    f1               REAL,
    support          INTEGER,
    PRIMARY KEY (run_id, category)
);

CREATE TABLE IF NOT EXISTS clause_register (
    clause_id      TEXT NOT NULL,
    contract_id    TEXT NOT NULL REFERENCES contract(contract_id) ON DELETE CASCADE,
    chunk_id       TEXT REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    category       TEXT NOT NULL,
    evidence_text  TEXT,
    signal_scores  TEXT,            -- JSON, one entry per signal
    risk_score     REAL,
    risk_level     TEXT,            -- Low | Medium | High
    rules_version  TEXT NOT NULL,
    run_id         TEXT NOT NULL REFERENCES model_run(run_id),
    start_char     INTEGER,
    end_char       INTEGER,
    PRIMARY KEY (clause_id, run_id, rules_version)
);
CREATE INDEX IF NOT EXISTS idx_clause_contract ON clause_register(contract_id);

CREATE TABLE IF NOT EXISTS advisor_label (
    clause_id       TEXT PRIMARY KEY,
    advisor_rank    INTEGER,
    advisor_bucket  TEXT            -- Low | Medium | High
);

CREATE TABLE IF NOT EXISTS triage_score (
    contract_id    TEXT NOT NULL REFERENCES contract(contract_id) ON DELETE CASCADE,
    run_id         TEXT NOT NULL REFERENCES model_run(run_id),
    rules_version  TEXT NOT NULL,
    triage_score   REAL,
    review_rank    INTEGER,
    n_high         INTEGER,
    n_medium       INTEGER,
    PRIMARY KEY (contract_id, run_id, rules_version)
);
"""
