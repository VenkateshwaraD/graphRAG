# CI/CD — GitLab pipeline

The pipeline is a **quality gate**, not a deployment pipeline: it
proves lint + the offline test suite on every push/MR, with an optional
deep retrieval-eval job for seeded environments.

Author: venkateshwara doijode

## Pipeline file

`.gitlab-ci.yml` — two jobs in one `test` stage:

```mermaid
flowchart LR
    P[push / merge request] --> Q[quality-gate<br/>blocking]
    S[schedule / manual on MR] --> R[retrieval-eval<br/>non-blocking on MRs]

    subgraph QJ["quality-gate (image: python:3.12)"]
        B1["python -m venv .venv<br/>pip install -e .[dev]"] --> B2["eval_quality.py"]
        B2 --> B3["ruff check ."]
        B2 --> B4["pytest -q (52 offline tests)"]
        B3 & B4 --> B5{gate PASS/FAIL}
    end
    Q --> QJ

    subgraph RJ["retrieval-eval (manual / scheduled)"]
        R1["eval_quality.py --with-retrieval"] --> R2["labeled questions<br/>recall/coverage floors"]
        R2 --> R3{needs: built index<br/>+ real embeddings}
    end
    R --> RJ
```

## What runs where

| Job | Trigger | Blocking? | What it does |
|---|---|---|---|
| `quality-gate` | push to `main`/`master`, every MR | yes | `scripts/eval_quality.py` → `ruff check .` + `pytest -q` |
| `retrieval-eval` | pipeline schedules; manual on MRs | no (`allow_failure`) | `--with-retrieval` → hybrid recall/coverage floors on the labeled set |

`retrieval-eval` is manual because it needs `data/graphrag/output`
artifacts + sentence-transformers + a reachable LLM gateway — none of
which an ephemeral runner has. Run it on a self-hosted runner with
`.env` and seeded `data/` present, or on a nightly schedule.

## Local dry run (identical steps to the pipeline)

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/eval_quality.py            # == quality-gate job
.venv/bin/python scripts/eval_quality.py --with-retrieval  # == retrieval-eval job
```

## Gate internals — `scripts/eval_quality.py`

```mermaid
flowchart TB
    E[eval_quality.py] --> R["ruff check ."]
    E --> T["pytest -q"]
    E --> O{"--with-retrieval?"}
    O -->|no| X[exit 0/1]
    O -->|yes| G["load vector store + docs<br/>hybrid.retrieve on LABELED set"]
    G --> F["recall ≥ --min-recall (0.5)<br/>coverage ≥ --min-coverage (0.6)"]
    F --> X
    G -.empty store / no index.-> SK[skipped → pass]
    SK --> X
```

Any failed sub-check flips the exit code — the job log prints
`:: label: PASS|FAIL` lines per check so failures are grep-able.

## Notes

- Self-hosted runners: tag this project to a runner with
  `python:3.12` Docker access. For `retrieval-eval`, the runner needs
  the repo's `data/` state and `LLM_*`/`EMBEDDING_*` env (CI/CD
  variables).
- There is no deploy stage — deployment is `docker compose up`
  locally; add a `deploy` stage here if the project gains a target
  environment.
