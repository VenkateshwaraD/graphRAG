"""Hybrid retrieval: vector similarity + full-text search merged with RRF."""

from __future__ import annotations

from app.ingestion.embed import EmbeddingProvider
from app.retrieval.vectorstore import LanceChunkStore, RetrievedChunk

_RRF_K = 60


def _rrf_merge(*ranked_lists: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reciprocal-rank fusion over several ranked hit lists.

    Each list contributes ``1/(K + rank)`` per item (K=60 dampens rank
    effects); a chunk present in multiple lists sums its contributions,
    so vector+FTS agreement wins. Returned hits get source ``"rrf"``.

    Author: venkateshwara doijode
    """
    scores: dict[str, float] = {}
    items: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (
                _RRF_K + rank + 1
            )
            items.setdefault(item.chunk_id, item)
    merged = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for chunk_id, score in merged:
        item = items[chunk_id]
        out.append(
            RetrievedChunk(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                text=item.text,
                score=score,
                source="rrf",
            )
        )
    return out


class HybridRetriever:
    """Complementary baseline retriever: LanceDB vector + FTS via RRF.

    Author: venkateshwara doijode
    """

    name = "hybrid"

    def __init__(self, store: LanceChunkStore, embedder: EmbeddingProvider) -> None:
        """Wire the shared store and the shared embedding provider.

        Author: venkateshwara doijode
        """
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        """Answer ``query`` with up to ``k`` fused chunks.

        Embeds the query with the SAME configured provider used at index
        time (the embedding_model filter then matches), runs vector and
        FTS legs in parallel conceptually, and RRF-merges the results.

        Author: venkateshwara doijode
        """
        # Query embedding via the SAME configured provider as indexing.
        q_vec = self._embedder.embed_query(query)
        vector_hits = self._store.vector_search(
            q_vec, k=k, embedding_model=self._embedder.model_id
        )
        fts_hits = self._store.fts_search(query, k=k)
        return _rrf_merge(vector_hits, fts_hits)[:k]
