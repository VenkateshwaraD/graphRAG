"""Benchmark: pure vector vs FTS vs hybrid vs GraphRAG traversal.

Retrieval-level comparison over the shared labeled set — for each
strategy it reports recall@k (full gold evidence coverage required),
gold coverage, and MRR, plus mean latency. This is the honest
side-by-side that demonstrates what each component contributes:
vector alone misses lexical/multi-hop evidence, hybrid improves recall,
traverse wins on multi-hop questions.

Usage: .venv/bin/python scripts/eval_benchmark.py [--k 5]
       .venv/bin/python scripts/eval_benchmark.py --api http://localhost:8000
           (end-to-end: also times /chat forced-mode answers)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_retrieval import LABELED, doc_id_to_stem  # noqa: E402


def _bench_retrievers(k: int) -> None:
    """Score vector / FTS / hybrid / traverse on the labeled questions.

    Author: venkateshwara doijode
    """
    from app.config import get_settings
    from app.db.session import SessionLocal
    from app.ingestion.embed import get_embedding_provider
    from app.retrieval.artifacts import load_artifacts
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.traverse import traverse
    from app.retrieval.vectorstore import get_vector_store

    session = SessionLocal()
    embedder = get_embedding_provider()
    store = get_vector_store()
    artifacts = load_artifacts()
    stems = doc_id_to_stem(session)
    hybrid = HybridRetriever(store, embedder)

    strategies = {
        "vector": lambda q: store.vector_search(
            embedder.embed_query(q), k=k, embedding_model=embedder.model_id
        ),
        "fts": lambda q: store.fts_search(q, k=k),
        "hybrid": lambda q: hybrid.retrieve(q, k=k),
        "traverse": lambda q: traverse(
            q, artifacts, embedder,
            max_hops=get_settings().traverse_k_hops, max_items=k,
        ),
    }
    if not artifacts.indexed:
        del strategies["traverse"]
        print("traverse: no graph index — skipped (run /ingest first)")

    print(f"{'strategy':<10} {'recall@k':>9} {'coverage':>9} {'MRR':>6} {'ms/q':>7}")
    for name, fn in strategies.items():
        hits = cov = rr = 0.0
        ms = 0.0
        for q, gold in LABELED:
            t0 = time.perf_counter()
            results = fn(q)
            ms += (time.perf_counter() - t0) * 1000
            retrieved = {stems.get(r.document_id) for r in results}
            hits += set(gold) <= retrieved
            cov += len(set(gold) & retrieved) / len(gold)
            rank = next(
                (i + 1 for i, r in enumerate(results)
                 if stems.get(r.document_id) in gold),
                0,
            )
            rr += 1.0 / rank if rank else 0.0
        n = len(LABELED)
        print(
            f"{name:<10} {hits / n:>9.2f} {cov / n:>9.2f} "
            f"{rr / n:>6.2f} {ms / n:>7.0f}"
        )
    session.close()


def _bench_api(api: str) -> None:
    """Time forced-mode /chat answers end-to-end (requires live API).

    Author: venkateshwara doijode
    """
    import os

    import httpx

    headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
    print(f"\n{'mode':<10} {'ok':>3} {'cites':>6} {'warn':>5} {'ms':>7}")
    for mode in ("hybrid", "traverse", "local", "global"):
        ok = cites = warns = 0
        ms = 0.0
        for q, _ in LABELED[:4]:
            t0 = time.perf_counter()
            try:
                r = httpx.post(
                    f"{api}/chat", json={"question": q, "mode": mode},
                    timeout=600, headers=headers,
                )
                ms += (time.perf_counter() - t0) * 1000
                if not r.is_success:
                    continue
                d = r.json()
                ok += 1
                cites += len(d.get("citations") or [])
                warns += bool(d.get("warning"))
            except Exception:
                continue
        n = min(4, len(LABELED))
        print(f"{mode:<10} {ok:>3} {cites / n:>6.1f} {warns:>5} {ms / n:>7.0f}")


def main() -> None:
    """Run the retrieval benchmark; ``--api`` adds the live-API leg.

    Author: venkateshwara doijode
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--api", default=None)
    args = ap.parse_args()
    _bench_retrievers(args.k)
    if args.api:
        _bench_api(args.api)


if __name__ == "__main__":
    main()
