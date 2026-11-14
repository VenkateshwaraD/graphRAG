"""Native Anthropic Messages API provider."""

from __future__ import annotations

import anthropic

from app.rag.llm import ChatMessage
from app.rag.providers.base import with_retries


class AnthropicLLM:
    """Direct ``anthropic`` SDK backend for the Messages API.

    Author: venkateshwara doijode
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        """Build the SDK client; requires a non-empty API key.

        ``base_url`` enables Anthropic-compatible gateways/proxies.

        Author: venkateshwara doijode
        """
        if not api_key:
            raise ValueError("anthropic provider requires an API key")
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key, base_url=base_url, timeout=timeout_s
        )

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Call messages.create with bounded retries; returns reply text.

        The Messages API takes system prompts as a separate field, so
        system-role entries are joined into ``system`` and the rest go
        through as the conversation. Text blocks of the response are
        concatenated (non-text blocks are ignored).

        Author: venkateshwara doijode
        """
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]

        def call() -> str:
            resp = self._client.messages.create(
                model=self.model,
                system=system or None,
                messages=[{"role": m["role"], "content": m["content"]} for m in convo],
                max_tokens=max_tokens or 1024,
                temperature=temperature,
            )
            return "".join(b.text for b in resp.content if b.type == "text")

        return with_retries(call)
