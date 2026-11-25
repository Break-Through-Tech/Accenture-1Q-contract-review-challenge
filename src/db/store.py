"""Read and write helpers for the results DB.

    from src.db.store import connect
    conn = connect()                      # results/cuad_results.db
    conn = connect(":memory:")            # throwaway, for tests

Writes are "replace" operations scoped to what a pipeline stage produced, so
re-running a stage never duplicates rows:

* ``replace_chunks``          all chunks of one contract (and, by cascade, their
                              predictions and clause rows)
* ``replace_predictions``     one contract under one model run
* ``replace_clause_register`` some contracts under one (run_id, rules_version)
* ``replace_triage``          every contract under one (run_id, rules_version),
                              because review_rank is relative to the whole set
"""

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.db.schema import SCHEMA_SQL

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "results" / "cuad_results.db"


def connect(path=DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the results DB, creating the file and tables if needed."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    return conn


# --- helpers ------------------------------------------------------------------

def _python_value(value):
    """numpy / pandas scalars -> plain Python, missing -> None, for sqlite3."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def _insert(conn, table, frame: pd.DataFrame, columns) -> int:
    if frame.empty:
        return 0
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{table}: missing columns {missing}")
    rows = [tuple(_python_value(v) for v in row)
            for row in frame[list(columns)].itertuples(index=False, name=None)]
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})", rows)
    return len(rows)


def _in_clause(values) -> str:
    return ", ".join("?" for _ in values)


# --- writes -------------------------------------------------------------------

CONTRACT_COLUMNS = ["contract_id", "title", "split", "n_chars", "source"]
CHUNK_COLUMNS = ["chunk_id", "contract_id", "chunk_index", "start_char", "end_char",
                 "n_tokens", "text", "text_light"]
MODEL_RUN_COLUMNS = ["run_id", "model_name", "artifact_uri", "git_commit", "created_at",
                     "threshold", "metadata_json"]
PREDICTION_COLUMNS = ["chunk_id", "run_id", "category", "probability", "predicted"]
CATEGORY_METRIC_COLUMNS = ["run_id", "category", "precision_score", "recall_score",
                           "f1", "support"]
CLAUSE_COLUMNS = ["clause_id", "contract_id", "chunk_id", "category", "evidence_text",
                  "signal_scores", "risk_score", "risk_level", "rules_version", "run_id",
                  "start_char", "end_char"]
TRIAGE_COLUMNS = ["contract_id", "run_id", "rules_version", "triage_score", "review_rank",
                  "n_high", "n_medium"]


def upsert_contract(conn, contract_id, title=None, split=None, n_chars=None, source=None):
    """Insert or update one contract row without touching its chunks."""
    with conn:
        conn.execute(
            """INSERT INTO contract (contract_id, title, split, n_chars, source)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(contract_id) DO UPDATE SET
                   title = excluded.title, split = excluded.split,
                   n_chars = excluded.n_chars, source = excluded.source""",
            (contract_id, title, split, _python_value(n_chars), source))


def replace_chunks(conn, contract_id, chunks: pd.DataFrame) -> int:
    """Replace a contract's chunks. Cascades to its predictions and clause rows,
    since those were computed from the old chunks."""
    frame = chunks.assign(contract_id=contract_id)
    with conn:
        conn.execute("DELETE FROM chunk WHERE contract_id = ?", (contract_id,))
        return _insert(conn, "chunk", frame, CHUNK_COLUMNS)


def upsert_model_run(conn, run: dict, category_metrics: Optional[pd.DataFrame] = None):
    """Record a model run and, optionally, its per-category test metrics."""
    metadata = run.get("metadata_json")
    if isinstance(metadata, dict):
        run = {**run, "metadata_json": json.dumps(metadata)}
    with conn:
        conn.execute(
            f"""INSERT INTO model_run ({', '.join(MODEL_RUN_COLUMNS)})
                VALUES ({_in_clause(MODEL_RUN_COLUMNS)})
                ON CONFLICT(run_id) DO UPDATE SET
                    {', '.join(f'{c} = excluded.{c}' for c in MODEL_RUN_COLUMNS[1:])}""",
            tuple(_python_value(run.get(c)) for c in MODEL_RUN_COLUMNS))
        if category_metrics is not None:
            conn.execute("DELETE FROM category_metric WHERE run_id = ?", (run["run_id"],))
            _insert(conn, "category_metric",
                    category_metrics.assign(run_id=run["run_id"]), CATEGORY_METRIC_COLUMNS)


def replace_predictions(conn, contract_id, run_id, predictions: pd.DataFrame) -> int:
    """Replace one contract's chunk predictions under one model run.

    ``predictions`` is long format: ``chunk_id``, ``category``, ``probability``,
    ``predicted``.
    """
    frame = predictions.assign(run_id=run_id)
    frame["predicted"] = frame["predicted"].astype(bool).astype(int)
    with conn:
        conn.execute(
            """DELETE FROM prediction WHERE run_id = ? AND chunk_id IN
               (SELECT chunk_id FROM chunk WHERE contract_id = ?)""",
            (run_id, contract_id))
        return _insert(conn, "prediction", frame, PREDICTION_COLUMNS)


def replace_clause_register(conn, contract_ids: Iterable[str], run_id, rules_version,
                            clauses: pd.DataFrame) -> int:
    """Replace the clause rows of ``contract_ids`` for one (run_id, rules_version).

    Contracts with no detected clauses still have their old rows cleared.
    """
    contract_ids = list(contract_ids)
    if not clauses.empty:
        stray = set(clauses["contract_id"]) - set(contract_ids)
        if stray:
            raise ValueError(f"clauses for contracts not being replaced: {sorted(stray)[:3]}")
        wrong = clauses[(clauses["run_id"] != run_id) | (clauses["rules_version"] != rules_version)]
        if not wrong.empty:
            raise ValueError("clause rows carry a different run_id or rules_version")
    with conn:
        if contract_ids:
            conn.execute(
                f"""DELETE FROM clause_register WHERE run_id = ? AND rules_version = ?
                    AND contract_id IN ({_in_clause(contract_ids)})""",
                (run_id, rules_version, *contract_ids))
        return _insert(conn, "clause_register", clauses, CLAUSE_COLUMNS)


def replace_triage(conn, run_id, rules_version, triage: pd.DataFrame) -> int:
    """Replace every triage row for one (run_id, rules_version)."""
    frame = triage.assign(run_id=run_id)
    if not frame.empty and (frame["rules_version"] != rules_version).any():
        raise ValueError("triage rows carry a different rules_version")
    with conn:
        conn.execute("DELETE FROM triage_score WHERE run_id = ? AND rules_version = ?",
                     (run_id, rules_version))
        return _insert(conn, "triage_score", frame, TRIAGE_COLUMNS)


# --- reads --------------------------------------------------------------------

def _where_contracts(contract_ids, column="contract_id"):
    if contract_ids is None:
        return "", ()
    contract_ids = list(contract_ids)
    if not contract_ids:
        return " AND 0", ()
    return f" AND {column} IN ({_in_clause(contract_ids)})", tuple(contract_ids)


def list_contracts(conn) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT {', '.join(CONTRACT_COLUMNS)} FROM contract "
                             "ORDER BY contract_id", conn)


def read_contract(conn, contract_id) -> Optional[dict]:
    row = conn.execute(f"SELECT {', '.join(CONTRACT_COLUMNS)} FROM contract "
                       "WHERE contract_id = ?", (contract_id,)).fetchone()
    return dict(zip(CONTRACT_COLUMNS, row)) if row else None


def read_chunks(conn, contract_ids=None) -> pd.DataFrame:
    where, params = _where_contracts(contract_ids)
    return pd.read_sql_query(
        f"SELECT {', '.join(CHUNK_COLUMNS)} FROM chunk WHERE 1=1{where} "
        "ORDER BY contract_id, chunk_index", conn, params=params)


def read_model_run(conn, run_id) -> Optional[dict]:
    row = conn.execute(f"SELECT {', '.join(MODEL_RUN_COLUMNS)} FROM model_run "
                       "WHERE run_id = ?", (run_id,)).fetchone()
    return dict(zip(MODEL_RUN_COLUMNS, row)) if row else None


def read_predictions(conn, run_id, contract_ids=None) -> pd.DataFrame:
    """Chunk predictions in the shape ``src.risk.evidence.extract_evidence`` takes."""
    where, params = _where_contracts(contract_ids, "c.contract_id")
    frame = pd.read_sql_query(
        f"""SELECT c.contract_id, p.chunk_id, p.category, p.probability,
                   p.predicted AS predicted_label
            FROM prediction p JOIN chunk c ON c.chunk_id = p.chunk_id
            WHERE p.run_id = ?{where}
            ORDER BY c.contract_id, c.chunk_index, p.category""",
        conn, params=(run_id, *params))
    frame["predicted_label"] = frame["predicted_label"].astype(bool)
    return frame


def contracts_with_predictions(conn, run_id) -> set:
    rows = conn.execute(
        """SELECT DISTINCT c.contract_id FROM prediction p
           JOIN chunk c ON c.chunk_id = p.chunk_id WHERE p.run_id = ?""", (run_id,))
    return {r[0] for r in rows}


def read_clause_register(conn, run_id=None, rules_version=None, contract_ids=None) -> pd.DataFrame:
    where, params = _where_contracts(contract_ids)
    filters, extra = [], []
    if run_id is not None:
        filters.append(" AND run_id = ?")
        extra.append(run_id)
    if rules_version is not None:
        filters.append(" AND rules_version = ?")
        extra.append(rules_version)
    return pd.read_sql_query(
        f"SELECT {', '.join(CLAUSE_COLUMNS)} FROM clause_register WHERE 1=1"
        f"{''.join(filters)}{where} ORDER BY contract_id, risk_score DESC",
        conn, params=(*extra, *params))


def contract_splits(conn, contract_ids) -> pd.Series:
    """contract_id -> split. Contracts with no recorded split map to "unknown"."""
    contract_ids = sorted(contract_ids)
    if not contract_ids:
        return pd.Series(dtype=object)
    rows = conn.execute(
        f"""SELECT contract_id, COALESCE(split, 'unknown') FROM contract
            WHERE contract_id IN ({_in_clause(contract_ids)})""", contract_ids).fetchall()
    found = dict(rows)
    return pd.Series({c: found.get(c, "unknown") for c in contract_ids}, dtype=object)


def read_triage(conn, run_id, rules_version) -> pd.DataFrame:
    """Triage rows plus each contract's ``split``. The pipeline ranks contracts
    within their split, so review_rank is only comparable inside one split."""
    columns = ", ".join(f"t.{c}" for c in TRIAGE_COLUMNS)
    return pd.read_sql_query(
        f"""SELECT {columns}, COALESCE(c.split, 'unknown') AS split
            FROM triage_score t LEFT JOIN contract c ON c.contract_id = t.contract_id
            WHERE t.run_id = ? AND t.rules_version = ?
            ORDER BY split, t.review_rank""",
        conn, params=(run_id, rules_version))
