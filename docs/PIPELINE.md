# Running the pipeline

One command takes a contract to a risk-scored clause register, a contract triage
score, and a review report. Everything lands in one SQLite file,
`results/cuad_results.db`, with every row traceable to the model run and the
rules version that produced it.

```
contract text ──ingest──▶ contract, chunk
                  │           (clean, then chunk: same code as training)
                  ▼
              ──infer───▶ model_run, prediction
                  │           (DistilRoBERTa, every chunk × 34 categories)
                  ▼
              ──score───▶ clause_register, triage_score
                  │           (risk engine + configs/risk_rules.yaml)
                  ▼
              ──report──▶ results/reports/<contract>.txt / .json
```

Each stage reads what the previous one wrote to the DB, so each runs on its own:
editing the risk rules means re-running `score` (seconds, no GPU), not inference.

---

## 1. Setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate                   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch with GPU support by default, a ~2.5 GB download.
For inference on a laptop the CPU build is enough and much smaller:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Model checkpoint.** Checkpoints are not in git. Download the trained run folder
from the shared Drive and put it at `models/distilroberta-cuad/`, so that this file
exists: `models/distilroberta-cuad/run_metadata.json`. (`/models/` is gitignored.)
Another location works with `--run-dir PATH`.

The first model load also downloads the `distilroberta-base` base weights (~330 MB)
from Hugging Face, so it needs internet once.

---

## 2. Run it

```powershell
python -m src.pipeline run --cuad test:0 --cuad train:0      # one contract from each split
python -m src.pipeline run --file path\to\contract.txt --report
python -m src.pipeline run --cuad test:all --report          # the whole test split
```

Contracts come from `--file PATH` (a `.txt` file; its name becomes the contract
id) or `--cuad SPLIT:N`, where `N` is an index (`test:3`), a range (`train:0-9`),
or `all`. Both options repeat.

The run prints each contract's review rank and writes everything to the DB.
On CPU, inference takes about 0.25 s per chunk; the 102 test contracts take about
13 minutes. Re-running skips the work already done.

### Stages one at a time

| Command | Does | Needs the model? |
|---|---|---|
| `python -m src.pipeline ingest --cuad test:all` | Cleans and chunks contracts into the DB | No |
| `python -m src.pipeline infer` | Predicts every ingested contract that has no predictions for this checkpoint yet | Yes |
| `python -m src.pipeline score` | Rebuilds the clause register and triage for every predicted contract | No |
| `python -m src.report --split test` | Writes TXT + JSON reports (see section 4) | No |
| `python -m src.pipeline status` | Row counts per table, High/Medium counts per split | No |

Useful options: `--force` redoes work a stage would skip; `--contract ID` limits
`infer`/`score` to some contracts; `--rules PATH` scores with another rules file;
`--run-id ID` scores or reports an older run; `--db PATH` uses another DB file
(handy for experiments, or for keeping train contracts out of the main DB).

### On Colab

Clone the repo, `pip install -r requirements.txt`, mount Drive, and point the
pipeline at Drive so the checkpoint and the DB survive the session:

```python
!python -m src.pipeline --db /content/drive/MyDrive/cuad/cuad_results.db \
    run --cuad test:all --run-dir /content/drive/MyDrive/cuad/distilroberta-cuad
```

A GPU runtime makes `infer` several times faster; nothing else needs one.

---

## 3. What ends up in the DB

The schema is in `src/db/schema.py` and described in `docs/ARCHITECTURE.md` §3.

| Table | One row per | Written by |
|---|---|---|
| `contract` | contract (id, title, split, length, where it came from) | ingest |
| `chunk` | chunk, with offsets, raw `text` and cleaned `text_light` | ingest |
| `model_run` | checkpoint (run id, threshold, metadata, git commit) | infer |
| `category_metric` | category × run, from the checkpoint's saved test metrics | infer |
| `prediction` | chunk × category × run (probability, predicted yes/no) | infer |
| `clause_register` | detected clause × run × rules version | score |
| `triage_score` | contract × run × rules version | score |

**Auditability.** Every clause and triage row carries:

* `run_id`, the checkpoint that flagged it: the run folder name plus a hash of its
  `run_metadata.json` and `classifier.pt`. Same checkpoint, same id, on any
  machine. A retrained model gets a new id, and its predictions sit next to the old
  ones instead of replacing them.
* `rules_version` from `configs/risk_rules.yaml`. Re-scoring after a version bump
  adds rows. Scores from earlier versions stay in the DB for comparison.
* `start_char` / `end_char` of the evidence chunk, which index the original
  contract text. `contract_text[start_char:end_char] == evidence_text`.

**Ranks are per split.** `review_rank` ranks a contract against the other
contracts of the same split (`test`, `train`, or `external` for `--file`
contracts) under the same run and rules version. A train contract the model has
already seen therefore never pushes test contracts down the queue. Always filter
or group by split when comparing ranks.

Example queries (`sqlite3 results/cuad_results.db`, or `pd.read_sql_query`):

```sql
-- The review queue for the test split
SELECT t.review_rank, t.triage_score, t.n_high, t.n_medium, t.contract_id
FROM triage_score t JOIN contract c USING (contract_id)
WHERE c.split = 'test' AND t.rules_version = '0.4.0-draft'
ORDER BY t.review_rank;

-- Every High clause, with where to read it
SELECT contract_id, category, risk_score, start_char, end_char
FROM clause_register WHERE risk_level = 'High' ORDER BY risk_score DESC;

-- How a rules change moved clause levels
SELECT a.category, a.risk_level AS before, b.risk_level AS after, COUNT(*)
FROM clause_register a JOIN clause_register b
  ON a.clause_id = b.clause_id AND a.run_id = b.run_id
WHERE a.rules_version = '0.4.0-draft' AND b.rules_version = '<new version>'
GROUP BY 1, 2, 3 HAVING before <> after;
```

---

## 4. Reports

`python -m src.report` (or `--report` on `run`/`score`) writes one TXT and one
JSON file per contract to `results/reports/`. Reports quote contract text, so
that folder is gitignored, like `contracts/`.

Each report has:

* **Header**: contract, model run and threshold, rules version, generation time.
* **Summary**: review rank within the split, clause counts by risk level. This
  is a fixed template for now; #56 adds an LLM-written overview and keeps the
  template as its fallback (`summary_source` in the JSON says which).
* **Flagged for review**, each with the reason:

  | Flag | When |
  |---|---|
  | `must_review_contract` | the contract has at least `triage.always_review_if_high_count` High clauses |
  | `high_risk` | a clause scored High |
  | `missing_protection` | a clause's protective counterpart (`protective_exceptions` in the rules) is not detected anywhere in the contract |
  | `borderline_detection` | a Medium/High clause's model confidence is within 0.10 of the threshold, so the clause may not really be there |
  | `weak_category` | a Medium/High clause is in a category the model is unreliable on (`caveats.weak_detection`) |
  | `nothing_detected` | no clause categories detected at all, so the text may be unreadable |

  Low-risk clauses are never flagged.
* **Every detected clause**, highest risk first, with its risk score, model
  confidence, the restrictive keywords that matched, money amounts (below), the
  snippet most likely to be the clause wording, and its location.
* **Not checked**: the 7 CUAD categories outside the model's scope, and the
  weak categories, where a missing clause is not proof of absence.

**Locations.** `snippet_start_char` / `snippet_end_char` locate the snippet in
the contract text; `chunk_start_char` / `chunk_end_char` locate the whole evidence
chunk. `page` and `line` are `null` until the cleaner keeps page markers (#55) and
file inputs keep pages (#53, #54).

**Money amounts.** Amounts in a clause's evidence are normalized to one value with
`src/risk/amounts.py`, so `$1M`, `$1,000,000`, `1000000 $` and `1.000.000,00 USD`
all read as `USD 1,000,000` and are listed together. It handles K / M / MM / B and
thousand / million / billion, the currency before or after the number, and US or
EU separators. Only amounts with a currency marker are picked up, so section
numbers, dates and share counts are left alone. The contract text itself is never
rewritten, because every offset depends on it, and amounts do not feed the risk
score.

---

## 5. Tests

```powershell
python -m pytest
```

The tests use a fake model and a temporary in-memory DB, so they need neither the
checkpoint nor a GPU. One test checks that the pipeline chunks a CUAD contract
exactly as `data/processed/chunks.parquet` does; it skips if that file has not
been built.

---

## 6. Known limitations

* **Threshold.** The decision threshold comes from the checkpoint's
  `run_metadata.json`, so every row under one `run_id` used the same threshold.
  There is deliberately no command-line override.
* **Evidence is a chunk, not a clause.** A chunk is about a page. The keyword
  signal and the amounts look at the whole chunk, so they can come from a
  neighbouring clause. The report's snippet narrows it down for the reader, but
  scoring still uses the full chunk.
* **Train contracts.** The model was trained on them, so their detections look
  better than they would on unseen contracts. Never report numbers from them.
* **Base weights.** Building the model downloads `distilroberta-base` before the
  fine-tuned weights replace it (the "lm_head UNEXPECTED" load report in
  `src.demo` output is this, and it is harmless).
