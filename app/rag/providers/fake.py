"""Deterministic offline LLM for tests and CI — no network calls.

Behavior is driven by the last user message:
- containing "ROUTE:" prefix handling is done by router tests via canned map
- echoes a canned answer including any supplied context markers
"""

from __future__ import annotations

from app.rag.llm import ChatMessage


class FakeLLM:
    """Deterministic LLMProvider implementation for offline tests.

    Author: venkateshwara doijode
    """

    name = "fake"

    def __init__(self, *, model: str = "fake-model", canned: str | None = None) -> None:
        """Configure the canned reply and start recording calls.

        ``canned`` forces a fixed response for every prompt (router
        tests); ``calls`` accumulates each message list for assertions.

        Author: venkateshwara doijode
        """
        self.model = model
        self.canned = canned
        self.calls: list[list[ChatMessage]] = []

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return a deterministic reply without any network access.

        Records the messages, returns ``canned`` when configured, replies
        "hybrid" to router-style prompts, and otherwise echoes a fake
        grounded answer mentioning the question text.

        Author: venkateshwara doijode
        """
        self.calls.append(messages)
        if self.canned is not None:
            return self.canned
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        # Router prompts ask for a single route token; default to hybrid.
        if "reply with exactly one of" in last.lower():
            return "hybrid"
        return f"[FAKE ANSWER] grounded on provided context for: {last[:80]}"
