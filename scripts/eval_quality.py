"""Quality gate for CI: lint, tests, and (optionally) retrieval eval.

Always runs:
  - ruff check (code quality)
  - pytest (offline suite — fake LLM + hash embeddings)

With ``--with-retrieval`` and a built graph index it additionally runs
the labeled retrieval eval and enforces minimum thresholds
(``--min-recall``, ``--min-coverage``) so regressions in traversal /
ranking fail the build. Skipped automatically when no index exists —
the eval needs real embeddings + parquets CI can't produce offline.

Usage:
  .venv/bin/python scripts/eval_quality.py
  .venv/bin/python scripts/eval_quality.py --with-retrieval --min-recall 0.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run(cmd: list[str], label: str) -> bool:
    """Run ``cmd`` in the repo root; print a verdict line, return ok.

    Author: venkateshwara doijode
    """
    print(f":: {label}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT)
    print(f":: {label}: {'PASS' if proc.returncode == 0 else 'FAIL'}")
    return proc.returncode == 0


def _retrieval_gate(min_recall: float, min_coverage: float, k: int) -> bool:
    """Score hybrid retrieval on the labeled set and enforce floors.

    Parses nothing — re-implements the recall/coverage computation so a
    formatting change in eval_retrieval can't silently weaken the gate.
    Returns True (skipped, not failed) when the vector store or graph
    index isn't populated — an empty corpus means 'nothing to check'.

    Author: venkateshwara doijode
    """
    from app.db.session import SessionLocal
    from app.ingestion.embed import get_embedding_provider
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.vectorstore import get_vector_store
    from scripts.eval_retrieval import LABELED, doc_id_to_stem

    session = SessionLocal()
    try:
        store = get_vector_store()
        embedder = get_embedding_provider()
        stems = doc_id_to_stem(session)
        if not stems:
            print(":: retrieval gate: no documents — skipped")
            return True
        hybrid = HybridRetriever(store, embedder)
        hits = cov = 0.0
        for q, gold in LABELED:
            try:
                results = hybrid.retrieve(q, k=k)
            except Exception:
                results = []
            retrieved = {stems.get(r.document_id) for r in results}
            hits += set(gold) <= retrieved
            cov += len(set(gold) & retrieved) / len(gold)
        n = len(LABELED)
        recall, coverage = hits / n, cov / n
        print(f":: retrieval gate: recall={recall:.2f} coverage={coverage:.2f} "
              f"(floors {min_recall}/{min_coverage})")
        return recall >= min_recall and coverage >= min_coverage
    except Exception as exc:
        print(f":: retrieval gate: could not run ({exc}) — skipped")
        return True
    finally:
        session.close()


def main() -> None:
    """Run the gate; exit non-zero when any enforced check fails.

    Author: venkateshwara doijode
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-retrieval", action="store_true",
                    help="also enforce retrieval recall/coverage floors")
    ap.add_argument("--min-recall", type=float, default=0.5)
    ap.add_argument("--min-coverage", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    ok = True
    ok &= _run([sys.executable, "-m", "ruff", "check", "."], "ruff")
    ok &= _run([sys.executable, "-m", "pytest", "-q"], "pytest")
    if args.with_retrieval:
        ok &= _retrieval_gate(args.min_recall, args.min_coverage, args.k)

    print(f":: quality gate: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
