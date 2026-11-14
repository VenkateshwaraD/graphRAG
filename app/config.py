from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration loaded from .env / environment variables.

    Every tunable of the system lives here: LLM provider + model, the
    shared embedding provider config, GraphRAG paths, storage locations,
    and API bind settings. Values are read once and cached via
    ``get_settings()``.

    Author: venkateshwara doijode
    """

    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: str = "openai_compat"  # openai_compat | anthropic | ollama | fake
    llm_model: str = "global.anthropic.claude-sonnet-4-6"
    llm_router_model: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 1024

    # Embeddings — one provider shared by indexing and query paths
    # sentence_transformers | openai | voyage | hash (tests)
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dims: int = 384
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    # Model-required prefixes (e.g. E5 models need "query: "/"passage: ").
    # Applied inside the provider so indexing and query paths stay consistent.
    embedding_query_prefix: str = ""
    embedding_doc_prefix: str = ""

    # GraphRAG
    graphrag_root: Path = ROOT / "data" / "graphrag"
    graphrag_max_gleanings: int = 1
    graphrag_community_level: int = 0
    traverse_k_hops: int = 2

    # Storage
    database_url: str = f"sqlite:///{ROOT / 'data' / 'app.db'}"
    lancedb_uri: Path = ROOT / "data" / "lancedb"
    upload_dir: Path = ROOT / "data" / "uploads"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @field_validator("graphrag_root", "lancedb_uri", "upload_dir", mode="after")
    @classmethod
    def _abspath(cls, v: Path) -> Path:
        """Resolve relative env paths against the project ROOT.

        GraphRAG's config loader chdir's the whole process into the graph
        root (set_cwd=True), so any relative path would resolve against a
        different cwd on every subsequent call — the ``data/graphrag``
        path-decay bug this validator exists to prevent.

        Author: venkateshwara doijode
        """
        return v if v.is_absolute() else (ROOT / v).resolve()

    @field_validator("database_url", mode="after")
    @classmethod
    def _abs_sqlite(cls, v: str) -> str:
        """Absolutize a relative sqlite URL the same way (see above).

        Author: venkateshwara doijode
        """
        prefix = "sqlite:///"
        if v.startswith(prefix):
            rest = v[len(prefix):]
            if rest and not rest.startswith("/") and rest != ":memory:":
                return f"{prefix}{ROOT / rest}"
        return v

    def llm_key(self) -> str:
        """Resolve the API key for the configured LLM provider.

        Falls back through common env vars (``CLAUDE_API_KEY``,
        ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``) so deployments that only
        export a vendor var still work. Returns "" when nothing is set —
        providers raise their own clear error at construction.

        Author: venkateshwara doijode
        """
        import os

        return (
            self.llm_api_key
            or os.environ.get("CLAUDE_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Cached with ``lru_cache`` so every module shares one parsed
    configuration; tests override env before first import.

    Author: venkateshwara doijode
    """
    return Settings()
