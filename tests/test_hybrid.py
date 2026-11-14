from app.ingestion.embed import get_embedding_provider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vectorstore import get_vector_store


def _seed(store, embedder):
    """Insert three known chunks + build the FTS index for tests.

    Author: venkateshwara doijode
    """
    texts = [
        "CS-501 Advanced Database Systems is taught by Dr. Priya Raman.",
        "Library late returns incur a fine of two credits per day.",
        "North Campus shuttle buses stop at Turing Hall every fifteen minutes.",
    ]
    vecs = embedder.embed_documents(texts)
    store.add_chunks(
        [
            {
                "chunk_id": f"c{i}",
                "document_id": "d1",
                "text": t,
                "vector": v,
                "embedding_model": embedder.model_id,
            }
            for i, (t, v) in enumerate(zip(texts, vecs, strict=False))
        ]
    )
    store.ensure_fts_index()


def test_vector_search_retrieves_relevant():
    """Seeded query returns the most relevant chunk first.

    Author: venkateshwara doijode
    """
    store = get_vector_store()
    embedder = get_embedding_provider()
    _seed(store, embedder)
    hits = HybridRetriever(store, embedder).retrieve("who teaches advanced databases")
    assert hits
    assert "Priya Raman" in hits[0].text


def test_fts_search_matches_keywords():
    """Keyword FTS finds the chunk containing the query terms.

    Author: venkateshwara doijode
    """
    store = get_vector_store()
    embedder = get_embedding_provider()
    _seed(store, embedder)
    hits = store.fts_search("shuttle Turing Hall", k=3)
    assert hits and "shuttle" in hits[0].text.lower()


def test_stale_embedding_model_excluded():
    """Chunks embedded by a different model must NOT appear in vector results.

    Author: venkateshwara doijode
    """
    store = get_vector_store()
    embedder = get_embedding_provider()
    stale_vec = embedder.embed_documents(["secret text"])[0]
    store.add_chunks(
        [
            {
                "chunk_id": "stale1",
                "document_id": "dx",
                "text": "secret text about databases",
                "vector": stale_vec,
                "embedding_model": "old/model-v0",
            }
        ]
    )
    hits = store.vector_search(
        embedder.embed_query("databases"), k=10, embedding_model=embedder.model_id
    )
    assert all(h.chunk_id != "stale1" for h in hits)
