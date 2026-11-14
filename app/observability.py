"""Optional Langfuse tracing for the RAG workflow.

Langfuse is an *optional* dependency — when it isn't installed or the
keys aren't configured every helper degrades to a no-op so the app and
the offline test suite run unchanged. When enabled, a LangChain
``CallbackHandler`` is attached to the LangGraph ``ainvoke`` config,
which captures node runs and LLM calls as spans under one trace.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

_checked = False
_available = False


def langfuse_configured() -> bool:
    """True when langfuse is importable AND the env keys are set.

    Author: venkateshwara doijode
    """
    global _checked, _available
    if not _checked:
        try:
            import langfuse  # noqa: F401

            _available = True
        except ImportError:
            _available = False
        _checked = True
    s = get_settings()
    return bool(
        _available and s.langfuse_public_key and s.langfuse_secret_key
    )


def langfuse_handler(**kwargs: Any) -> Any | None:
    """Return a Langfuse ``CallbackHandler`` for LangGraph, or None.

    Attach via ``graph.ainvoke(state, config={"callbacks": [cb]})`` —
    LangGraph/LangChain then reports each node + LLM call to Langfuse.
    Returns None (never raises) when the package or credentials are
    absent, keeping tracing strictly opt-in.

    Author: venkateshwara doijode
    """
    if not langfuse_configured():
        return None
    s = get_settings()
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
            **kwargs,
        )
    except Exception:
        logger.warning("langfuse handler init failed; tracing disabled")
        return None
