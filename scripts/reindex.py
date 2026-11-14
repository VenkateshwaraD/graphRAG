"""Re-embed all document chunks with the CURRENT configured embedding model.

Required after changing EMBEDDING_MODEL — existing chunks keep the old
model's vectors and are excluded from vector search (embedding_model
filter) until re-embedded. GraphRAG artifacts must be rebuilt separately
via POST /ingest.

Usage: .venv/bin/python scripts/reindex.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db.models import DocumentChunk
from app.db.session import SessionLocal
from app.ingestion.embed import get_embedding_provider
from app.retrieval.vectorstore import get_vector_store


def main() -> None:
    """Re-embed every chunk belonging to a document with stale vectors.

    A document is "stale" when any of its chunks was embedded by a model
    different from the active one. All chunks of affected docs are
    re-embedded (not just the stale rows) because ``delete_document``
    removes every LanceDB row for that doc — re-adding only stale chunks
    would lose rows if a previous reindex died mid-run. Batches of 64
    chunks per embed call; FTS index rebuilt at the end.

    Author: venkateshwara doijode
    """
    embedder = get_embedding_provider()
    store = get_vector_store()
    session = SessionLocal()
    try:
        chunks = session.query(DocumentChunk).all()
        stale_ids = {c.document_id for c in chunks if c.embedding_model != embedder.model_id}
        # re-embed ALL chunks of any doc with stale vectors — delete_document
        # removes every row for that doc, so re-adding only stale chunks
        # would lose rows if a previous reindex died mid-run
        stale = [c for c in chunks if c.document_id in stale_ids]
        print(f"{len(chunks)} chunks total, {len(stale_ids)} docs stale for {embedder.model_id}")
        for doc_id in stale_ids:
            store.delete_document(doc_id)
        for i in range(0, len(stale), 64):
            batch = stale[i : i + 64]
            vecs = embedder.embed_documents([c.text for c in batch])
            rows = []
            for c, v in zip(batch, vecs, strict=False):
                c.embedding_model = embedder.model_id
                rows.append(
                    {
                        "chunk_id": c.id,
                        "document_id": c.document_id,
                        "text": c.text,
                        "vector": v,
                        "embedding_model": embedder.model_id,
                    }
                )
            store.add_chunks(rows)
            session.commit()
        store.ensure_fts_index()
        print("reindex complete")
    finally:
        session.close()


if __name__ == "__main__":
    main()
