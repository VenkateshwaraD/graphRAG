"""GraphRAG indexing wrapper.

Generates settings.yaml under GRAPHRAG_ROOT and runs the Microsoft
GraphRAG indexing pipeline (extract entities/relationships -> graph ->
Leiden communities -> community reports -> embeddings) over the staged
input texts, producing parquet artifacts + a LanceDB store.

GraphRAG's internal completion model uses LiteLLM against the configured
OpenAI-compatible endpoint. Its internal *embedding* model is a custom
registered type (``local_st``) that calls our shared EmbeddingProvider
in-process — the gateway key has no embedding models, and this keeps
index and query embeddings on the same model anyway.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings

EMBEDDING_TYPE = "local_st"


def _settings_yaml() -> str:
    """Render the GraphRAG ``settings.yaml`` content from app settings.

    Maps our env config onto GraphRAG's litellm completion model (raw
    model name — the provider prefix confuses gateways that validate
    model ids) and the custom ``local_st`` embedding type. Also pins the
    LanceDB vector size to our embedding dims and constrains entity types
    to a university domain. ``None`` values are dropped so no yaml nulls
    leak into the file.

    Author: venkateshwara doijode
    """
    s = get_settings()
    api_key = s.llm_key() or "dummy"
    # litellm forwards `model` verbatim to the endpoint — use the raw name,
    # the provider prefix only confuses gateways that validate model ids.
    litellm_model = s.llm_model
    cfg = {
        "completion_models": {
            "default_completion_model": {
                "type": "litellm",
                "model_provider": "openai",
                "model": litellm_model,
                "api_base": s.llm_base_url or None,
                "api_key": api_key,
                "auth_method": "api_key",
                "retry": {"type": "exponential_backoff"},
            }
        },
        "embedding_models": {
            "default_embedding_model": {
                "type": EMBEDDING_TYPE,
                "model_provider": "local",
                "model": s.embedding_model,
                "auth_method": "api_key",
                "api_key": "not-needed",
            }
        },
        "input": {"type": "text"},
        "chunking": {"size": 1200, "overlap": 100},
        "input_storage": {"type": "file", "base_dir": "input"},
        "output_storage": {"type": "file", "base_dir": "output"},
        # incremental runs stage their delta + previous-index copy here
        "update_output_storage": {"type": "file", "base_dir": "output.updates"},
        "reporting": {"type": "file", "base_dir": "logs"},
        "cache": {"type": "json", "storage": {"type": "file", "base_dir": "cache"}},
        "vector_store": {
            "type": "lancedb",
            "db_uri": "output/lancedb",
            "vector_size": s.embedding_dims,
        },
        "extract_graph": {
            "max_gleanings": s.graphrag_max_gleanings,
            "entity_types": [
                "person",
                "organization",
                "course",
                "department",
                "policy",
                "location",
                "event",
            ],
        },
    }
    # drop None values (yaml null) to keep config clean
    cfg["completion_models"]["default_completion_model"] = {
        k: v
        for k, v in cfg["completion_models"]["default_completion_model"].items()
        if v is not None
    }
    return yaml.safe_dump(cfg, sort_keys=False)


def ensure_settings() -> Path:
    """Write ``GRAPHRAG_ROOT/settings.yaml`` and create its work dirs.

    Rewrites the file only when content changed so an unchanged config
    doesn't bust GraphRAG's cache. Called at app startup and before each
    index run.

    Author: venkateshwara doijode
    """
    s = get_settings()
    root = Path(s.graphrag_root)
    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(parents=True, exist_ok=True)
    settings_file = root / "settings.yaml"
    content = _settings_yaml()
    if not settings_file.exists() or settings_file.read_text() != content:
        settings_file.write_text(content)
    return settings_file


def register_local_embedding() -> None:
    """Register our EmbeddingProvider as a graphrag_llm embedding type.

    Defines ``LocalSTEmbedding`` (a ``graphrag_llm`` ``LLMEmbedding``
    subclass that delegates to the shared in-process provider) and
    registers it under the ``local_st`` type name used in settings.yaml.
    Idempotent — a duplicate registration is swallowed.

    Author: venkateshwara doijode
    """
    from graphrag_llm.embedding import register_embedding
    from graphrag_llm.embedding.embedding import LLMEmbedding
    from graphrag_llm.types import LLMEmbeddingResponse

    class LocalSTEmbedding(LLMEmbedding):
        def __init__(
            self,
            *,
            model_id: str,
            model_config: Any,
            tokenizer: Any = None,
            cache_key_creator: Any = None,
            **_: Any,
        ) -> None:
            """Store the config graphrag_llm passes at construction.

            Author: venkateshwara doijode
            """
            self._model_id = model_id
            self._model_config = model_config
            self._tokenizer = tokenizer

        @property
        def metrics_store(self):
            """Required abstract member — metrics are unused here.

            Author: venkateshwara doijode
            """
            return None

        @property
        def tokenizer(self):
            """Required abstract member — we don't tokenize for counting.

            Author: venkateshwara doijode
            """
            return self._tokenizer

        def _respond(self, kwargs: dict) -> Any:
            """Embed ``kwargs["input"]`` via the shared provider and wrap
            the result in an OpenAI-shaped ``LLMEmbeddingResponse``.

            Author: venkateshwara doijode
            """
            from app.ingestion.embed import get_embedding_provider

            texts = kwargs.get("input") or []
            provider = get_embedding_provider()
            vecs = provider.embed_documents(list(texts))
            return LLMEmbeddingResponse(
                object="list",
                model=provider.model_id,
                data=[
                    {"object": "embedding", "index": i, "embedding": v}
                    for i, v in enumerate(vecs)
                ],
                usage={"prompt_tokens": 0, "total_tokens": 0},
            )

        def embedding(self, /, **kwargs: Any):
            """Sync entry point graphrag_llm calls for embeddings.

            Author: venkateshwara doijode
            """
            return self._respond(kwargs)

        async def embedding_async(self, /, **kwargs: Any):
            """Async entry point — CPU-bound work needs no real await.

            Author: venkateshwara doijode
            """
            return self._respond(kwargs)

    try:
        register_embedding(
            embedding_type=EMBEDDING_TYPE,
            embedding_initializer=LocalSTEmbedding,
            scope="singleton",
        )
    except Exception:
        pass  # already registered


def _archive_version(out_dir: Path, root: Path) -> None:
    """Snapshot the successful output into ``output.versions/<ts>``.

    Keeps the 5 most recent versions so a bad rebuild or update can be
    rolled back beyond the single ``output.prev`` safety net.

    Author: venkateshwara doijode
    """
    import time

    versions = root / "output.versions"
    versions.mkdir(parents=True, exist_ok=True)
    shutil.copytree(out_dir, versions / time.strftime("%Y%m%d-%H%M%S"))
    for old in sorted(versions.iterdir())[:-5]:
        shutil.rmtree(old, ignore_errors=True)


async def build_graph_index(update: bool = False) -> dict:
    """Run the GraphRAG pipeline over ``GRAPHRAG_ROOT/input``.

    ``update=True`` runs an incremental build: GraphRAG diffs the input
    against the previous index and merges only new/changed content
    (cheaper than a full rebuild). Falls back to a full build when no
    previous output exists. For updates the previous output must stay in
    place — the pipeline reads it during merge — so it's *copied* to
    ``output.prev`` for rollback instead of renamed. For full builds the
    output is renamed aside and restored on failure, so a bad reindex
    never leaves the system without a graph. A successful build is
    archived to ``output.versions/<ts>``. Raises ``RuntimeError`` listing
    failed workflows so the job row can record a real error.

    Author: venkateshwara doijode
    """
    from graphrag.api import build_index
    from graphrag.config.load_config import load_config

    register_local_embedding()
    ensure_settings()
    s = get_settings()
    config = load_config(Path(s.graphrag_root))

    out_dir = Path(s.graphrag_root) / "output"
    prev_dir = Path(s.graphrag_root) / "output.prev"
    has_previous = out_dir.exists() and any(out_dir.glob("*.parquet"))
    is_update = update and has_previous

    if prev_dir.exists():
        shutil.rmtree(prev_dir)
    if is_update:
        # the update pipeline merges INTO output — keep it in place but
        # snapshot it for rollback
        shutil.copytree(out_dir, prev_dir)
    elif out_dir.exists():
        out_dir.rename(prev_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = await build_index(
            config=config, verbose=False, is_update_run=is_update
        )
    except Exception:
        _restore(out_dir, prev_dir)
        raise

    stats: dict[str, Any] = {"workflows": {}, "incremental": is_update}
    for r in results:
        stats["workflows"][r.workflow] = (
            "done" if not r.error else f"error: {r.error}"
        )
    # output storage can flush slightly after results return — retry briefly
    for _ in range(10):
        counts = _artifact_stats(out_dir)
        if counts.get("entities", 0) > 0 or counts.get("text_units", 0) > 0:
            break
        await asyncio.sleep(0.5)
    stats.update(counts)
    errors = [w for w, v in stats["workflows"].items() if v != "done"]
    if errors:
        # keep the old index — the partial new output is worthless
        _restore(out_dir, prev_dir)
        raise RuntimeError(f"graphrag workflows failed: {errors}")
    _archive_version(out_dir, Path(s.graphrag_root))
    if prev_dir.exists():
        shutil.rmtree(prev_dir)
    return stats


def _restore(out_dir: Path, prev_dir: Path) -> None:
    """Roll ``output`` back to the pre-build snapshot after a failure.

    Author: venkateshwara doijode
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    if prev_dir.exists():
        prev_dir.rename(out_dir)


def _artifact_stats(out_dir: Path) -> dict:
    """Count rows in each generated artifact parquet (0 when absent).

    Author: venkateshwara doijode
    """
    import pandas as pd

    stats = {}
    for name in ("entities", "relationships", "text_units", "communities", "community_reports"):
        f = out_dir / f"{name}.parquet"
        stats[name] = len(pd.read_parquet(f)) if f.exists() else 0
    return stats


def artifacts_dir() -> Path:
    """Directory holding the current GraphRAG output parquets.

    Author: venkateshwara doijode
    """
    return Path(get_settings().graphrag_root) / "output"
