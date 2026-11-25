"""Give a trained model run a stable ``run_id`` and record it in the results DB.

The trainer's ``run_metadata.json`` has no run id, commit, or timestamp, so the
id is derived from the files themselves:

    run_id = <run folder name>-<first 8 hex of sha256(run_metadata.json + classifier.pt)>

The same checkpoint always maps to the same id, on any machine. A retrained
model gets a new classifier head (and new test metrics), so it gets a new id.
The ~330 MB encoder is not hashed; the head and metadata change with it.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.db.store import upsert_model_run

ROOT = Path(__file__).resolve().parents[2]
HASHED_FILES = ("run_metadata.json", "classifier.pt")


def _git_commit():
    """HEAD of the repo running the pipeline, suffixed ``-dirty`` if there are
    uncommitted changes to tracked files. None outside a git checkout."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if dirty else commit


def run_id_for(run_dir) -> str:
    run_dir = Path(run_dir)
    digest = hashlib.sha256()
    for name in HASHED_FILES:
        digest.update((run_dir / name).read_bytes())
    return f"{run_dir.name}-{digest.hexdigest()[:8]}"


def model_run_info(run_dir) -> dict:
    """The ``model_run`` row for a trained run directory."""
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    created = datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc)
    return {
        "run_id": run_id_for(run_dir),
        "model_name": metadata.get("model_name"),
        "artifact_uri": str(run_dir.resolve()),
        "git_commit": _git_commit(),
        "created_at": created.isoformat(timespec="seconds"),
        "threshold": float(metadata["threshold"]),
        "metadata_json": metadata,
    }


def _category_metrics(run_dir):
    path = Path(run_dir) / "test_category_metrics.csv"
    if not path.exists():
        return None
    metrics = pd.read_csv(path)
    return metrics.rename(columns={"precision": "precision_score", "recall": "recall_score"})


def register_model_run(conn, run_dir) -> str:
    """Record the run (and its test metrics, if saved) and return its ``run_id``."""
    info = model_run_info(run_dir)
    upsert_model_run(conn, info, _category_metrics(run_dir))
    return info["run_id"]


def resolve_run_id(conn, run_id=None, run_dir=None) -> str:
    """Which model run to read results for.

    An explicit ``run_id`` wins. Otherwise the checkpoint at ``run_dir`` is hashed
    (not loaded) and registered. Failing that, the DB's only run is used.
    """
    if run_id:
        return run_id
    if run_dir is not None and (Path(run_dir) / "run_metadata.json").exists():
        return register_model_run(conn, run_dir)
    runs = [r[0] for r in conn.execute("SELECT run_id FROM model_run ORDER BY run_id")]
    if len(runs) == 1:
        return runs[0]
    raise ValueError(f"no checkpoint at {run_dir}; pass a run id (known: {runs})")
