"""Shared request-scoped services, built once from settings."""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from app.config import get_settings
from app.ingestion.embed import get_embedding_provider
from app.rag.graph import build_rag_graph
from app.rag.llm import get_llm, get_router_llm
from app.retrieval.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class ApiPrincipal:
    """Who the caller is, derived from the presented API key.

    ``groups=None`` means all document groups (admin). ``write`` gates
    mutating endpoints (upload, delete, ingest, review).

    Author: venkateshwara doijode
    """

    key: str | None
    groups: frozenset[str] | None = None
    write: bool = True


def _key_roles() -> dict[str, dict]:
    """Parse ``API_KEYS_JSON`` — {"key": {"groups": [...], "write": bool}}.

    Role map for scoped keys: each key lists the document groups it may
    read and whether it may write. Malformed JSON fails closed (empty
    map -> only the admin API_KEY works).

    Author: venkateshwara doijode
    """
    import json

    raw = get_settings().api_keys_json
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("API_KEYS_JSON is not valid JSON — scoped keys disabled")
        return {}


def require_api_key(api_key: str | None = Security(_api_key_header)) -> ApiPrincipal:
    """Authenticate the caller and return their ``ApiPrincipal``.

    Mounted as a router-level dependency everywhere except ``health``
    (probes must stay open). ``API_KEY`` is the admin key (all groups,
    write). ``API_KEYS_JSON`` adds scoped read/write keys restricted to
    document groups. When neither is set the app runs unauthenticated —
    a deliberate local-dev escape hatch, warned about at startup.
    Comparison uses ``hmac.compare_digest`` so the header can't be
    probed byte-by-byte through timing.

    Author: venkateshwara doijode
    """
    s = get_settings()
    if not s.api_key and not s.api_keys_json:
        return ApiPrincipal(key=None)  # auth disabled: open principal
    admin = s.api_key
    if admin and api_key and hmac.compare_digest(api_key, admin):
        return ApiPrincipal(key=api_key)
    if api_key:
        for key, role in _key_roles().items():
            if hmac.compare_digest(api_key, key):
                groups = role.get("groups") or ["*"]
                return ApiPrincipal(
                    key=key,
                    groups=None if "*" in groups else frozenset(groups),
                    write=bool(role.get("write", False)),
                )
    raise HTTPException(401, "invalid or missing API key")


def require_write(
    principal: ApiPrincipal = Depends(require_api_key),
) -> ApiPrincipal:
    """Gate mutating endpoints to principals with ``write``.

    Author: venkateshwara doijode
    """
    if not principal.write:
        raise HTTPException(403, "this API key is read-only")
    return principal


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
