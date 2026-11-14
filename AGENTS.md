# Agent notes

## Commands

- Lint: `.venv/bin/ruff check .`
- Tests: `.venv/bin/python -m pytest` (offline; LLM_PROVIDER=fake, EMBEDDING_PROVIDER=hash via tests/conftest.py)
- Run API: `.venv/bin/uvicorn app.main:app --port 8000 --loop asyncio`  ← `--loop asyncio` is mandatory (uvloop breaks graphrag_llm's nest_asyncio patch)
- Seed: `.venv/bin/python scripts/seed.py --api http://localhost:8000`
- Graph index: `POST /ingest` then poll `GET /ingest/{job_id}`
- Eval: `scripts/eval_retrieval.py`, `scripts/eval_qa.py`

## Environment quirks (this machine)

- No Docker socket access → embedded stores: SQLite (`data/app.db`) + LanceDB (`data/lancedb`) + GraphRAG artifacts (`data/graphrag/output/*.parquet`). Postgres+pgvector remains the prod path behind `retrieval/vectorstore.py`.
- LLM gateway (`LLM_BASE_URL`) is OpenAI-compatible; key restricted to `global.anthropic.claude-{haiku-4-5,opus-4-5,sonnet-4-6}`. No embedding models on it → embeddings are local sentence-transformers.
- litellm forwards `model` verbatim — never add an `openai/` prefix in GraphRAG settings.
- Restarting the API: kill with `fuser -k 8000/tcp` (do NOT `pkill -f uvicorn` inside a command that also contains "uvicorn" — it self-matches).

## Invariants

- Same `EmbeddingProvider` for indexing (`embed_documents`) and queries (`embed_query`). Never hardcode a second model.
- Chunks carry `embedding_model`; vector search filters to the active model. Model change → `scripts/reindex.py` + `POST /ingest`.
- GraphRAG internals use our embedder via `register_embedding(type="local_st")` in `app/ingestion/graph_index.py`; `vector_size` in generated settings.yaml = `EMBEDDING_DIMS`.
- Citations map graphrag doc ids → our doc ids via `documents.parquet` title stem (`{doc_id}.txt`).

## Layout

- `app/rag/` — LangGraph workflow (`graph.py`), router, prompts, LLM interface (`llm.py`) + `providers/`
- `app/ingestion/` — extract/chunk/embed/pipeline, `graph_index.py` (settings.yaml gen + build_index), `jobs.py`
- `app/retrieval/` — `vectorstore.py` (LanceDB), `hybrid.py`, `traverse.py` (k-hop BFS), `graph_search.py` (local/global), `artifacts.py`
- `app/api/routers/` — documents, ingestion, chat, sources, graph, health
- `tests/` — offline suite; `seed_corpus/` — multi-hop fixture docs
