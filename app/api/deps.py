"""Shared request-scoped services, built once from settings."""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import threading
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Role, User
from app.db.session import get_session
from app.ingestion.embed import get_embedding_provider
from app.rag.graph import build_rag_graph
from app.rag.llm import get_llm, get_router_llm
from app.retrieval.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_MIN_KEY_LEN = 16
_INSECURE_DEFAULTS = {
    "changeme",
    "password",
    "admin",
    "test",
    "test-api-key",
    "api-key",
    "secret",
}


@dataclass(frozen=True)
class ApiPrincipal:
    """Who the caller is, resolved from the presented API key.

    ``user_id`` is None for the break-glass platform admin and for
    auth-disabled dev mode. ``role`` is one of ``Role``'s values;
    ``write``-capable roles may hit mutating endpoints.

    Author: venkateshwara doijode
    """

    key: str | None
    user_id: str | None = None
    tenant_id: str | None = None
    role: str = Role.PLATFORM_ADMIN.value
    department_id: str | None = None

    @property
    def write(self) -> bool:
        """Whether the role may call mutating endpoints.

        Author: venkateshwara doijode
        """
        from app.db.models import WRITE_ROLES

        return self.role in WRITE_ROLES


def generate_api_key() -> tuple[str, str, str]:
    """Mint a high-entropy API key.

    Returns ``(raw, sha256_hex, prefix)`` — the raw key is shown to the
    caller exactly once; only the hash and the non-secret prefix are
    stored.

    Author: venkateshwara doijode
    """
    raw = "uka_" + secrets.token_urlsafe(32)
    return raw, hashlib.sha256(raw.encode()).hexdigest(), raw[:12]


def _hash(raw: str) -> str:
    """sha256 hex digest of a presented key.

    Author: venkateshwara doijode
    """
    return hashlib.sha256(raw.encode()).hexdigest()


_fail_lock = threading.Lock()
_failures: dict[str, deque] = {}


def _client_ip(request: Request | None) -> str:
    """Best-effort client ip for rate-limit bucketing.

    Author: venkateshwara doijode
    """
    return request.client.host if request and request.client else "unknown"


def _throttled(ip: str) -> bool:
    """True when this IP has exhausted its failed-auth budget.

    A sliding one-minute window per IP; the counter only grows on
    failures, so legitimate callers on a shared IP are only blocked
    after repeated bad keys.

    Author: venkateshwara doijode
    """
    import time

    limit = get_settings().auth_fail_limit_per_minute
    now = time.monotonic()
    with _fail_lock:
        attempts = _failures.get(ip)
        if attempts is None:
            return False
        while attempts and now - attempts[0] > 60:
            attempts.popleft()
        return len(attempts) >= limit


def _record_failure(ip: str) -> None:
    """Count one failed authentication for the IP.

    Author: venkateshwara doijode
    """
    import time

    with _fail_lock:
        _failures.setdefault(ip, deque()).append(time.monotonic())


def _audit_denied(session: Session, request: Request | None, api_key: str | None) -> None:
    """Record an auth.denied audit event (prefix only — never the key).

    Author: venkateshwara doijode
    """
    try:
        from app.audit import record_audit

        record_audit(
            session,
            action="auth.denied",
            resource_type="auth",
            decision="deny",
            detail={
                "ip": _client_ip(request),
                "key_prefix": (api_key or "")[:8],
            },
        )
    except Exception:
        logger.debug("audit write failed for denied auth", exc_info=True)


def require_api_key(
    request: Request,
    api_key: str | None = Security(_api_key_header),
    session: Session = Depends(get_session),
) -> ApiPrincipal:
    """Authenticate the caller and return their ``ApiPrincipal``.

    Resolution order: rate-limit check -> env ``API_KEY`` (break-glass
    platform admin) -> user-key lookup by sha256 (``is_active``
    required). Auth-disabled dev mode (no API_KEY + explicit
    ``ALLOW_INSECURE_AUTH``) yields an open platform-admin principal.
    Failed attempts are audit-logged with only the key prefix, and
    counted against a per-IP sliding-window limit that returns 429.

    Author: venkateshwara doijode
    """
    settings = get_settings()
    ip = _client_ip(request)
    if _throttled(ip):
        raise HTTPException(
            429, "too many failed authentication attempts — retry later"
        )
    if not settings.api_key:
        # dev escape hatch — only reachable when startup allowed it
        if settings.allow_insecure_auth:
            return ApiPrincipal(key=None)
        _record_failure(ip)
        raise HTTPException(401, "invalid or missing API key")
    if api_key and hmac.compare_digest(api_key, settings.api_key):
        return ApiPrincipal(key=api_key)
    if api_key:
        user = (
            session.query(User)
            .filter(User.api_key_hash == _hash(api_key))
            .first()
        )
        if user is not None and user.is_active:
            return ApiPrincipal(
                key=api_key,
                user_id=user.id,
                tenant_id=user.tenant_id,
                role=user.role,
                department_id=user.department_id,
            )
    _record_failure(ip)
    _audit_denied(session, request, api_key)
    raise HTTPException(401, "invalid or missing API key")


def require_write(
    principal: ApiPrincipal = Depends(require_api_key),
) -> ApiPrincipal:
    """Gate mutating endpoints to write-capable roles.

    Author: venkateshwara doijode
    """
    if not principal.write:
        raise HTTPException(403, "this role cannot perform write operations")
    return principal


def require_admin(
    principal: ApiPrincipal = Depends(require_api_key),
) -> ApiPrincipal:
    """Gate platform/university administration endpoints.

    Author: venkateshwara doijode
    """
    if principal.role not in (
        Role.PLATFORM_ADMIN.value,
        Role.UNIVERSITY_ADMIN.value,
    ):
        raise HTTPException(403, "administrator role required")
    return principal


def check_bootstrap_auth() -> None:
    """Refuse to start with insecure bootstrap credentials.

    A missing ``API_KEY`` requires an explicit ``ALLOW_INSECURE_AUTH``
    opt-in (local dev only); a set-but-weak ``API_KEY`` is always
    rejected so the break-glass credential can't stay at a default.

    Author: venkateshwara doijode
    """
    settings = get_settings()
    if not settings.api_key:
        if settings.allow_insecure_auth:
            logger.warning(
                "ALLOW_INSECURE_AUTH is set — the API runs unauthenticated"
            )
            return
        raise RuntimeError(
            "API_KEY is not set. Set a strong API_KEY, or set "
            "ALLOW_INSECURE_AUTH=true for local development only."
        )
    if (
        len(settings.api_key) < _MIN_KEY_LEN
        or settings.api_key.lower() in _INSECURE_DEFAULTS
    ):
        raise RuntimeError(
            "API_KEY is too weak (minimum 16 chars, no default values). "
            "Set a strong API_KEY, or ALLOW_INSECURE_AUTH=true for local dev."
        )


@lru_cache
def get_graph():
    """Return the compiled LangGraph workflow, built once per process.

    Wires the singleton chunk store + embedding provider with the
    configured router/answer LLMs and traversal/generation tunables.
    Cached so per-request ainvoke is cheap.

    Author: venkateshwara doijode
    """
    settings = get_settings()
    return build_rag_graph(
        store=get_vector_store(),
        embedder=get_embedding_provider(),
        router_llm=get_router_llm(),
        answer_llm=get_llm(),
        k_hops=settings.traverse_k_hops,
        max_tokens=settings.llm_max_tokens,
    )
