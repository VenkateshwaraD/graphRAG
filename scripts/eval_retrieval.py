"""Retrieval eval: labeled questions -> recall@k / MRR per retriever.

Labeled set maps a question to the document stem(s) that contain its
answer. Runs hybrid + traverse through the real pipeline (graph routes
require an index build first).

Usage: .venv/bin/python scripts/eval_retrieval.py [--k 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# question -> list of seed_corpus stems containing needed evidence
LABELED = [
    ("What is the late return fine in the library circulation policy?", ["policies"]),
    ("Where is the main office of the department that teaches CS-501?",
     ["courses", "faculty", "campus"]),
    ("Who leads the division that oversees the office administering library fines?",
     ["policies"]),
    ("Which professor chairs the office administering the Energy Systems Certificate?",
     ["courses", "faculty"]),
    ("Where do North Campus shuttles stop?", ["campus"]),
    ("Who is the faculty advisor of the Undergraduate Programming Club?", ["faculty"]),
]


def doc_id_to_stem(session) -> dict:
    """Map Document.id -> filename stem (the eval's label key).

    Author: venkateshwara doijode
    """
    from app.db.models import Document

    return {d.id: Path(d.filename).stem for d in session.query(Document).all()}


def main() -> None:
    """Run the labeled questions through hybrid + traverse and score them.

    For each retriever: a "hit" is any returned chunk whose document stem
    is in the gold set; MRR uses the rank of the first gold hit. Prints
    per-question hits plus aggregate recall@k and MRR. Traverse is
    skipped when no graph index exists yet.

    Author: venkateshwara doijode
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

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

    for name in ("hybrid", "traverse"):
        hits_total = 0
        cov_total = 0.0
        rr_total = 0.0
        path_qs = path_hit = 0
        for q, gold in LABELED:
            if name == "hybrid":
                results = HybridRetriever(store, embedder).retrieve(q, k=args.k)
            else:
                if not artifacts.indexed:
                    print("traverse: no graph index — skipped (run /ingest first)")
                    break
                results = traverse(q, artifacts, embedder,
                                   max_hops=get_settings().traverse_k_hops,
                                   max_items=args.k)
            retrieved = {stems.get(r.document_id) for r in results}
            # a hit requires the FULL gold evidence set — retrieving 1
            # of 3 needed docs must not count as success
            hit = set(gold) <= retrieved
            coverage = len(set(gold) & retrieved) / len(gold)
            hits_total += hit
            cov_total += coverage
            rank = next(
                (i + 1 for i, r in enumerate(results) if stems.get(r.document_id) in gold),
                0,
            )
            rr_total += 1.0 / rank if rank else 0.0
            # path accuracy: on multi-hop questions, did a GOLD chunk
            # arrive via a real multi-entity hop path (not a duplicate-
            # ridden or trivial one)? Measures traversal correctness,
            # not just doc recall.
            if len(gold) > 1:
                path_qs += 1
                gold_paths = [
                    r.meta.get("hop_path")
                    for r in results
                    if stems.get(r.document_id) in gold and r.meta
                ]
                ok_path = any(
                    hp and len(set(hp)) == len(hp) and len(hp) >= 2
                    for hp in gold_paths
                )
                path_hit += bool(ok_path)
            print(f"  [{name}] hit={hit} cov={coverage:.2f} q={q[:60]!r} got={sorted(retrieved)}")
        n = len(LABELED)
        extra = (
            f"  path_accuracy={path_hit/path_qs:.2f} ({path_hit}/{path_qs} multi-hop)"
            if path_qs else ""
        )
        print(
            f"{name}: recall@{args.k}={hits_total/n:.2f}  "
            f"gold_coverage={cov_total/n:.2f}  MRR={rr_total/n:.2f}{extra}"
        )

    session.close()


if __name__ == "__main__":
    main()
