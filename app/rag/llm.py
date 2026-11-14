"""Provider-agnostic LLM interface.

Everything in the app calls only ``LLMProvider.generate`` — provider SDKs
are imported exclusively inside ``app/rag/providers/``. Selecting a
provider is a config change (``LLM_PROVIDER``); adding one means one new
file implementing the protocol plus a registry entry.
"""

from __future__ import annotations

from typing import Protocol, TypedDict

from app.config import get_settings


class ChatMessage(TypedDict):
    """One chat-completion message.

    Author: venkateshwara doijode
    """

    role: str  # "system" | "user" | "assistant"
    content: str


class LLMProvider(Protocol):
    """Contract every LLM backend must satisfy.

    The whole app (router, answer synthesis) calls only ``generate`` —
    swapping vendors is a config change, not a code change.

    Author: venkateshwara doijode
    """

    name: str
    model: str

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Return the assistant's reply text for a chat transcript.

        Author: venkateshwara doijode
        """
        ...


def get_llm(model: str | None = None) -> LLMProvider:
    """Build the configured LLM provider. Raises at call time, not import.

    Registry: ``openai_compat`` (default — any OpenAI-shaped gateway),
    ``anthropic`` (native Messages API), ``ollama`` (local server),
    ``fake`` (deterministic offline test double).

    Author: venkateshwara doijode
    """
    settings = get_settings()
    provider = settings.llm_provider
    model = model or settings.llm_model

    if provider == "openai_compat":
        from app.rag.providers.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(
            base_url=settings.llm_base_url or None,
            api_key=settings.llm_key(),
            model=model,
            timeout_s=settings.llm_timeout_s,
        )
    if provider == "anthropic":
        from app.rag.providers.anthropic import AnthropicLLM

        return AnthropicLLM(
            api_key=settings.llm_key(),
            model=model,
            base_url=settings.llm_base_url or None,
            timeout_s=settings.llm_timeout_s,
        )
    if provider == "ollama":
        from app.rag.providers.ollama import OllamaLLM

        return OllamaLLM(model=model, timeout_s=settings.llm_timeout_s)
    if provider == "fake":
        from app.rag.providers.fake import FakeLLM

        return FakeLLM(model=model)
    raise ValueError(f"unknown LLM_PROVIDER: {provider!r}")


def get_router_llm() -> LLMProvider:
    """Cheaper/faster model for routing/classification calls.

    Uses ``LLM_ROUTER_MODEL`` (defaults to a haiku-tier model) so route
    decisions don't pay flagship-model latency.

    Author: venkateshwara doijode
    """
    return get_llm(model=get_settings().llm_router_model)
