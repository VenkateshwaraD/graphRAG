"""LanceDB-backed chunk store: vector similarity + full-text search.

Chosen over Postgres+pgvector here because the environment has no Docker
access — LanceDB is embedded, file-based, and already a GraphRAG dep.
The retriever interface stays identical so Postgres can be swapped back
in behind this module later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import lancedb
import pyarrow as pa

from app.config import get_settings

logger = logging.getLogger(__name__)

TABLE = "chunks"


@dataclass
class RetrievedChunk:
    """One retrieval hit shared by every retriever.

    ``source`` records which signal produced it (vector | fts | rrf |
    traverse); ``meta`` carries retriever-specific extras like the
    traversal ``hop_path``.

    Author: venkateshwara doijode
    """

    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str = ""  # "vector" | "fts" | "rrf"
    meta: dict = field(default_factory=dict)


class LanceChunkStore:
    """Embedded chunk store: one LanceDB table holding text + vector.

    Author: venkateshwara doijode
    """

    def __init__(self, uri: Path, dims: int) -> None:
        """Connect (creating the dir) and remember the vector size.

        Author: venkateshwara doijode
        """
        uri.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(uri))
        self._dims = dims
        self._table = None

    def _schema(self, dims: int) -> pa.Schema:
        """Arrow schema for the chunks table.

        Author: venkateshwara doijode
        """
        return pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dims)),
                pa.field("embedding_model", pa.string()),
            ]
        )

    @property
    def table(self):
        """Lazily open the chunks table, creating it on first use.

        Opens before creating because LanceDB's ``list_tables`` returns a
        paginated object whose membership test was unreliable; a failed
        create surfaces as an exception rather than silent emptiness.

        Author: venkateshwara doijode
        """
        if self._table is None:
            try:
                self._table = self._db.open_table(TABLE)
            except Exception:
                self._table = self._db.create_table(
                    TABLE, schema=self._schema(self._dims)
                )
        return self._table

    def add_chunks(
        self,
        rows: list[dict],
    ) -> None:
        """Append chunk rows matching ``_schema``; no-op on empty input.

        Author: venkateshwara doijode
        """
        if not rows:
            return
        self.table.add(rows)

    def delete_document(self, document_id: str) -> None:
        """Drop every chunk row for one document (delete/re-ingest).

        Failures are logged, not raised — a missing table shouldn't block
        the ORM delete that follows.

        Author: venkateshwara doijode
        """
        try:
            self.table.delete(f"document_id = '{document_id}'")
        except Exception:
            logger.warning("delete failed for document %s", document_id, exc_info=True)

    def ensure_fts_index(self) -> None:
        """Create the Tantivy FTS index once, then update it incrementally.

        A full ``replace=True`` rebuild is O(corpus) — running it after
        every document ingest made bulk uploads quadratic. Instead the
        index is created only when missing and ``optimize()`` indexes new
        rows incrementally. Failures are logged since FTS unavailability
        only degrades (not breaks) hybrid retrieval.

        Author: venkateshwara doijode
        """
        try:
            has_fts = any(
                "fts" in getattr(i, "index_type", "").lower()
                for i in self.table.list_indices()
            )
        except Exception:
            has_fts = False
        try:
            if not has_fts:
                self.table.create_index(
                    "text",
                    config=lancedb.index.FTS(with_position=True),
                    replace=True,
                )
            else:
                self.table.optimize()
        except Exception:
            logger.warning("FTS index update failed", exc_info=True)

    @staticmethod
    def _doc_filter(doc_ids: set[str] | list[str] | None) -> str:
        """WHERE expression restricting results to allowed document ids.

        Author: venkateshwara doijode
        """
        if doc_ids is None:
            return ""
        ids = ",".join(f"'{d}'" for d in sorted(doc_ids))
        return f"document_id IN ({ids})" if ids else "document_id = ''"

    def vector_search(
        self,
        embedding: list[float],
        k: int,
        embedding_model: str,
        doc_ids: set[str] | list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Cosine search over chunks produced by the ACTIVE embedding model only —
        stale embeddings from a previous model are never silently compared.

        ``doc_ids`` scopes results to documents the caller may see (RBAC);
        an empty set matches nothing. Returns up to ``k`` hits scored
        ``1 - distance``; failures log a warning and yield [] so the FTS
        leg can still answer.

        Author: venkateshwara doijode
        """
        try:
            where = f"embedding_model = '{embedding_model}'"
            df = self._doc_filter(doc_ids)
            if df:
                where += f" AND {df}"
            results = (
                self.table.search(embedding)
                .where(where)
                .limit(k)
                .to_list()
            )
        except Exception:
            logger.warning("vector search failed", exc_info=True)
            return []
        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                score=1.0 - float(r.get("_distance", 0.0)),
                source="vector",
            )
            for r in results
        ]

    def fts_search(
        self, query: str, k: int, doc_ids: set[str] | list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Tantivy full-text search returning up to ``k`` BM25-scored hits.

        ``doc_ids`` scopes results to documents the caller may see
        (RBAC). Failures log a warning and yield [] so the vector leg can
        still answer.

        Author: venkateshwara doijode
        """
        try:
            q = self.table.search(query, query_type="fts")
            df = self._doc_filter(doc_ids)
            if df:
                q = q.where(df)
            results = q.limit(k).to_list()
        except Exception:
            logger.warning("FTS search failed", exc_info=True)
            return []
        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                score=float(r.get("_score", 0.0)),
                source="fts",
            )
            for r in results
        ]


@lru_cache
def get_vector_store() -> LanceChunkStore:
    """Process-wide chunk-store singleton built from settings.

    Author: venkateshwara doijode
    """
    s = get_settings()
    return LanceChunkStore(s.lancedb_uri, s.embedding_dims)
