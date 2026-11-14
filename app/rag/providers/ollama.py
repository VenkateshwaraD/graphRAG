"""Local Ollama provider — chat via its OpenAI-compatible endpoint."""

from __future__ import annotations

import os

from app.rag.providers.openai_compat import OpenAICompatLLM


class OllamaLLM(OpenAICompatLLM):
    """Ollama backend — reuses OpenAICompatLLM against :11434/v1.

    Author: venkateshwara doijode
    """

    name = "ollama"

    def __init__(self, *, model: str, timeout_s: float = 120.0) -> None:
        """Point the compat client at OLLAMA_BASE_URL with a dummy key.

        Ollama ignores auth, so a placeholder key is passed when none is
        configured.

        Author: venkateshwara doijode
        """
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        super().__init__(
            model=model,
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            base_url=base_url,
            timeout_s=timeout_s,
        )
