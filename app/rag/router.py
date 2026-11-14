"""Route a question to the right retriever — LLM classification or override."""

from __future__ import annotations

import re

from app.rag.llm import LLMProvider
from app.rag.prompts import ROUTER_SYSTEM

VALID_ROUTES = {"hybrid", "traverse", "local", "global"}


def route_with_reason(
    question: str, router_llm: LLMProvider, override: str | None = None
) -> tuple[str, str]:
    """Pick the retrieval strategy, also returning the router's raw reply
    as explainability for the choice.

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
        return override, f"mode override: {override}"
    raw = router_llm.generate(
        [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ],
        max_tokens=8,
    ).strip().lower()
    # tolerate verbose answers — first route token by position wins.
    # A regex alternation (not a set scan) keeps parsing deterministic
    # when several tokens appear, and \b blocks substring false hits.
    m = re.search(r"\b(hybrid|traverse|local|global)\b", raw)
    return (m.group(1), raw) if m else ("hybrid", raw)


def route_question(
    question: str, router_llm: LLMProvider, override: str | None = None
) -> str:
    """Route without the rationale — thin wrapper over
    ``route_with_reason`` for callers that only need the token.

    Author: venkateshwara doijode
    """
    return route_with_reason(question, router_llm, override)[0]
