# Agent notes

## Commands

- Lint: `.venv/bin/ruff check .`
- Tests: `.venv/bin/python -m pytest` (offline; LLM_PROVIDER=fake, EMBEDDING_PROVIDER=hash via tests/conftest.py)
- Run API: `.venv/bin/uvicorn app.main:app --port 8000 --loop asyncio`  ← `--loop asyncio` is mandatory (uvloop breaks graphrag_llm's nest_asyncio patch)
- Seed: `.venv/bin/python scripts/seed.py --api http://localhost:8000`
- Graph index: `POST /ingest` then poll `GET /ingest/{job_id}`
- Eval: `scripts/eval_retrieval.py`, `scripts/eval_qa.py`

## Environment quirks (this machine)

- `API_KEY` env → `X-API-Key` required on every route except `/healthz`+`/readyz`; unset = auth disabled. Clients (seed.py, eval_qa.py, streamlit_app) read `API_KEY` and send the header; tests set `API_KEY=test-api-key` in conftest.
- `MAX_UPLOAD_BYTES` caps `POST /documents` (413 when exceeded; default 25 MB).

- No Docker socket access → embedded stores: SQLite (`data/app.db`) + LanceDB (`data/lancedb`) + GraphRAG artifacts (`data/graphrag/output/*.parquet`). Postgres+pgvector remains the prod path behind `retrieval/vectorstore.py`.
- LLM gateway (`LLM_BASE_URL`) is OpenAI-compatible; key restricted to `global.anthropic.claude-{haiku-4-5,opus-4-5,sonnet-4-6}`. No embedding models on it → embeddings are local sentence-transformers.
- litellm forwards `model` verbatim — never add an `openai/` prefix in GraphRAG settings.
- Restarting the API: kill with `fuser -k 8000/tcp` (do NOT `pkill -f uvicorn` inside a command that also contains "uvicorn" — it self-matches).

## Invariants

- `traverse()` signature: `traverse(q, artifacts, embedder, *, max_hops=None, max_items=8, router_llm=None)`. `traverse_k_hops` is now a CAP — depth adapts to decomposed anchors (fallback `[question]` → full cap). Best-first multi-path expansion, early stop at `max_items` evidence; scores + `hop_path`/`edge_directions` in `meta`.
- `assess_node` never deepens hops — empty results swap retriever once; non-empty results get one router-LLM relevance verdict, "no" → one alternate-retriever attempt.
- `build_graph_index` preserves previous output (`output.prev`) and restores it on failure — never delete `output/` before a build.
- `process_document` embeds before mutating stores; on failure it clears partial LanceDB rows and the staged `{doc_id}.txt`.
- FTS index is created once, then `optimize()`d incrementally — do NOT `create_index(replace=True)` per ingest.
- `POST /ingest` reservation is atomic in-process via `jobs.try_reserve_job` (threading.Lock); multi-worker needs a DB-level constraint.

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
