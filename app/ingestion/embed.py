"""Configurable embedding provider.

ONE instance is built from settings and shared by BOTH the ingestion
pipeline (``embed_documents``) and retrieval (``embed_query``). The query
path never uses a separate or hardcoded model — changing the model is a
config change + ``scripts/reindex.py``.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import get_settings


class EmbeddingProvider(Protocol):
    """Contract every embedding backend must satisfy.

    ``model_id`` is stored on each chunk so retrieval can exclude vectors
    produced by a different model (stale embeddings are never silently
    compared in the wrong space).

    Author: venkateshwara doijode
    """

    model_id: str
    dims: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document/passage texts for indexing.

        Author: venkateshwara doijode
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text with query-side preprocessing.

        Author: venkateshwara doijode
        """
        ...


class SentenceTransformersProvider:
    """Local sentence-transformers backend — the default.

    Loads one ``SentenceTransformer`` model in-process; supports
    configurable query/doc prefixes for asymmetric models (E5-style).

    Author: venkateshwara doijode
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        doc_prefix: str = "",
        expected_dims: int | None = None,
    ) -> None:
        """Load the HF model and verify its output dimension.

        Raises ``ValueError`` when ``expected_dims`` is configured but the
        model produces a different size — catches EMBEDDING_DIMS typos at
        startup instead of silently corrupting the vector store.

        Author: venkateshwara doijode
        """
        from sentence_transformers import SentenceTransformer

        self.model_id = f"sentence_transformers/{model_name}"
        self._model = SentenceTransformer(model_name)
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        # probe dims via encode — stable across sentence-transformers versions
        self.dims = int(self._model.encode(["dimension probe"]).shape[-1])
        if expected_dims and self.dims != expected_dims:
            raise ValueError(
                f"EMBEDDING_DIMS={expected_dims} but model {model_name} produces "
                f"{self.dims} dims — fix .env"
            )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts into L2-normalized float32 vectors.

        Author: venkateshwara doijode
        """
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages with the configured doc prefix (E5-style).

        Author: venkateshwara doijode
        """
        return self._encode([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed one query with the configured query prefix.

        Author: venkateshwara doijode
        """
        return self._encode([self._query_prefix + text])[0]


class OpenAIEmbeddingProvider:
    """OpenAI-compatible embeddings endpoint (api_base configurable)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None,
        dims: int | None = None,
    ) -> None:
        """Create the embeddings client; probes dims when not configured.

        Author: venkateshwara doijode
        """
        from openai import OpenAI

        if not api_key:
            raise ValueError("openai embedding provider requires an API key")
        self.model_id = f"openai/{model}"
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.dims = dims or self._probe()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """One ``embeddings.create`` call returning raw vectors.

        Author: venkateshwara doijode
        """
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    def _probe(self) -> int:
        """Discover output dimensionality with a one-off request.

        Author: venkateshwara doijode
        """
        return len(self._embed(["dimension probe"])[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages in batches of 128 (endpoint request limit).

        Author: venkateshwara doijode
        """
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            out.extend(self._embed(texts[i : i + 128]))
        return out

    def embed_query(self, text: str) -> list[float]:
        """Embed one query text.

        Author: venkateshwara doijode
        """
        return self._embed([text])[0]


class VoyageEmbeddingProvider:
    """Voyage AI backend — uses its native input_type distinction."""

    def __init__(self, model: str, api_key: str, dims: int | None = None) -> None:
        """Create the voyage client; probes dims when not configured.

        Author: venkateshwara doijode
        """
        import voyageai

        if not api_key:
            raise ValueError("voyage embedding provider requires an API key")
        self.model_id = f"voyage/{model}"
        self._client = voyageai.Client(api_key=api_key)
        self._model = model
        self.dims = dims or self._probe()

    def _probe(self) -> int:
        """Discover output dimensionality with a one-off request.

        Author: venkateshwara doijode
        """
        return len(self.embed_documents(["dimension probe"])[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages with ``input_type="document"``.

        Author: venkateshwara doijode
        """
        return self._client.embed(texts, model=self._model, input_type="document").embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed one query with ``input_type="query"``.

        Author: venkateshwara doijode
        """
        return self._client.embed([text], model=self._model, input_type="query").embeddings[0]


class HashEmbeddingProvider:
    """Deterministic hashing embedder for tests — no model download."""

    def __init__(self, dims: int = 64) -> None:
        """Set the vector size used by the hashing buckets.

        Author: venkateshwara doijode
        """
        self.model_id = "hash/test-embedder"
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        """Hash each token into a bucket and L2-normalize the counts.

        Uses blake2b so results are stable across processes (the builtin
        ``hash()`` is salted per run); retrieval quality is meaningless
        but the shape/consistency lets tests exercise the full pipeline.

        Author: venkateshwara doijode
        """
        vec = np.zeros(self.dims, dtype=np.float32)
        for token in text.lower().split():
            # blake2b digest — stable across processes (unlike builtin hash)
            bucket = int.from_bytes(
                hashlib.blake2b(token.encode(), digest_size=4).digest()
            )
            vec[bucket % self.dims] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Hash-embed each passage independently.

        Author: venkateshwara doijode
        """
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Hash-embed one query (same function as documents).

        Author: venkateshwara doijode
        """
        return self._embed(text)


def build_embedding_provider() -> EmbeddingProvider:
    """Construct the provider selected by ``EMBEDDING_PROVIDER``.

    Supported: ``sentence_transformers``, ``openai`` (any compatible
    embeddings endpoint), ``voyage``, ``hash`` (offline tests). Unknown
    values raise ``ValueError`` at startup.

    Author: venkateshwara doijode
    """
    s = get_settings()
    if s.embedding_provider == "sentence_transformers":
        return SentenceTransformersProvider(
            s.embedding_model,
            query_prefix=s.embedding_query_prefix,
            doc_prefix=s.embedding_doc_prefix,
            expected_dims=s.embedding_dims or None,
        )
    if s.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            s.embedding_model,
            s.embedding_api_key or s.llm_key(),
            s.embedding_base_url or None,
            dims=s.embedding_dims or None,
        )
    if s.embedding_provider == "voyage":
        return VoyageEmbeddingProvider(
            s.embedding_model, s.embedding_api_key, dims=s.embedding_dims or None
        )
    if s.embedding_provider == "hash":
        return HashEmbeddingProvider(dims=s.embedding_dims or 64)
    raise ValueError(f"unknown EMBEDDING_PROVIDER: {s.embedding_provider!r}")


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Return the process-wide shared embedding provider.

    The SAME instance (and therefore the same model + preprocessing)
    serves document indexing and query embedding — the core consistency
    invariant of this system.

    Author: venkateshwara doijode
    """
    return build_embedding_provider()
