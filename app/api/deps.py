"""Shared request-scoped services, built once from settings."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.ingestion.embed import get_embedding_provider
from app.rag.graph import build_rag_graph
from app.rag.llm import get_llm, get_router_llm
from app.retrieval.vectorstore import get_vector_store


@lru_cache
def get_graph():
    """Return the compiled LangGraph workflow, built once per process.

    Wires the singleton chunk store + embedding provider with the
    configured router/answer LLMs and traversal/generation tunables.
    Cached so per-request ainvoke is cheap.

    Author: venkateshwara doijode
    """
    s = get_settings()
    return build_rag_graph(
        store=get_vector_store(),
        embedder=get_embedding_provider(),
        router_llm=get_router_llm(),
        answer_llm=get_llm(),
        k_hops=s.traverse_k_hops,
        max_tokens=s.llm_max_tokens,
    )
