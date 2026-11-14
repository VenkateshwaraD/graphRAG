"""Shared request-scoped services, built once from settings."""

from __future__ import annotations

import hmac
import logging
from functools import lru_cache

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from app.config import get_settings
from app.ingestion.embed import get_embedding_provider
from app.rag.graph import build_rag_graph
from app.rag.llm import get_llm, get_router_llm
from app.retrieval.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Enforce the configured ``API_KEY`` on protected routes.

    Mounted as a router-level dependency everywhere except ``health``
    (probes must stay open). When ``api_key`` is unset the app runs
    unauthenticated — a deliberate local-dev escape hatch, warned about
    at startup. Comparison uses ``hmac.compare_digest`` so the header
    can't be probed byte-by-byte through timing.

    Author: venkateshwara doijode
    """
    expected = get_settings().api_key
    if not expected:
        return
    if not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(401, "invalid or missing API key")


def warn_if_auth_disabled() -> None:
    """Log a startup warning when the API runs without authentication.

    Author: venkateshwara doijode
    """
    if not get_settings().api_key:
        logger.warning("API_KEY is not set — all endpoints are unauthenticated")


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
