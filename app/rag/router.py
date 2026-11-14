"""Route a question to the right retriever — LLM classification or override."""

from __future__ import annotations

from app.rag.llm import LLMProvider
from app.rag.prompts import ROUTER_SYSTEM

VALID_ROUTES = {"hybrid", "traverse", "local", "global"}


def route_question(
    question: str, router_llm: LLMProvider, override: str | None = None
) -> str:
    """Pick the retrieval strategy for ``question``.

    An explicit ``override`` (API ``mode`` param) wins and is validated —
    unknown values raise ``ValueError`` which the API maps to 400.
    Otherwise the router LLM classifies the question against
    ROUTER_SYSTEM; its reply is normalized to lowercase and scanned for a
    route token so verbose answers still parse. Unparseable output falls
    back to ``hybrid`` (the safest default) rather than erroring.

    Author: venkateshwara doijode
    """
    if override:
        if override not in VALID_ROUTES:
            raise ValueError(f"invalid mode {override!r}; expected one of {VALID_ROUTES}")
        return override
    raw = router_llm.generate(
        [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ],
        max_tokens=8,
    ).strip().lower()
    # tolerate verbose answers — take first matching token
    for route in VALID_ROUTES:
        if route in raw:
            return route
    return "hybrid"
