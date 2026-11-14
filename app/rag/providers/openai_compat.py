"""Any OpenAI-compatible chat endpoint: OpenAI, Azure-style gateways, vLLM,
LiteLLM — and the default for this deployment (LLM_BASE_URL gateway)."""

from __future__ import annotations

from openai import OpenAI

from app.rag.llm import ChatMessage
from app.rag.providers.base import with_retries


class OpenAICompatLLM:
    """Chat-completions client for OpenAI-compatible endpoints.

    The configured gateway forwards model names verbatim, so ``model`` is
    passed through unchanged.

    Author: venkateshwara doijode
    """

    name = "openai_compat"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        """Build the SDK client; requires a non-empty API key.

        Author: venkateshwara doijode
        """
        if not api_key:
            raise ValueError("openai_compat provider requires an API key")
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Call chat.completions with bounded retries; returns reply text.

        Empty content is normalized to "".

        Author: venkateshwara doijode
        """
        def call() -> str:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                max_tokens=max_tokens or 1024,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""

        return with_retries(call)
