# University Knowledge Assistant — GraphRAG

An agentic GraphRAG system for multi-hop questions over university
documents. Upload docs → Microsoft GraphRAG builds a knowledge graph
(entities, relationships, communities, reports) → a LangGraph router
selects a retrieval strategy per question → answers come back with
source citations.

## Architecture

```
Streamlit UI ──> FastAPI ──> LangGraph workflow
                                │
                     route (LLM, or ?mode= override)
                                │
          ┌──────────┬──────────┼──────────┐
          v          v          v          v
       traverse   local      global     hybrid
       best-first GraphRAG   GraphRAG   LanceDB
       multi-path Local      Global     vector+FTS (RRF)
          └──────────┴──────────┴──────────┘
                    generate / cited answer

Ingestion: upload → extract (pdf/docx/txt/md) → chunk → embed
           → SQLite (metadata) + LanceDB (vectors+FTS)
           → stage into data/graphrag/input → POST /ingest runs
             GraphRAG indexing → parquet artifacts + LanceDB
```

- **traverse** — guided multi-hop over the extracted knowledge graph:
  the router LLM decomposes the question into entity anchors, anchors
  seed a best-first expansion keeping several scored candidate paths
  per node (edge weight × description relevance, decayed by depth),
  and expansion stops early once enough evidence is found — depth is
  derived from the question, capped by `TRAVERSE_K_HOPS`. Hops are
  auditable: citations carry the entity path and relationship
  directions.
- **local** — GraphRAG LocalSearch: entity-anchored neighborhoods.
- **global** — GraphRAG GlobalSearch: map-reduce over community reports.
- **hybrid** — chunk-level vector similarity + full-text search, RRF-merged.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env        # set LLM_BASE_URL / LLM_API_KEY / models

# API (note: --loop asyncio is REQUIRED — graphrag_llm patches the loop)
.venv/bin/uvicorn app.main:app --port 8000 --loop asyncio

# seed + index + chat
.venv/bin/python scripts/seed.py --api http://localhost:8000
curl -X POST localhost:8000/ingest            # build the knowledge graph
curl localhost:8000/ingest/<job_id>           # poll until "done"
curl -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "Where is the lab directed by the professor who teaches CS-501?"}'

# UI
.venv/bin/streamlit run streamlit_app/app.py
```

Docker: `docker compose up` (api :8000, ui :8501) — embedded stores, no
external DB needed.

## Configuration (env / .env)

| Var | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai_compat` | `openai_compat` \| `anthropic` \| `ollama` \| `fake` |
| `LLM_MODEL` | claude-sonnet-4-6 | answer/graph-extraction model |
| `LLM_ROUTER_MODEL` | claude-haiku-4-5 | cheap routing model |
| `LLM_BASE_URL` / `LLM_API_KEY` | — | OpenAI-compatible endpoint; key falls back to `CLAUDE_API_KEY`/`OPENAI_API_KEY` |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | `sentence_transformers` \| `openai` \| `voyage` \| `hash` (tests) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | shared by indexing AND query paths |
| `EMBEDDING_DIMS` | 384 | must match model; also sets GraphRAG `vector_size` |
| `TRAVERSE_K_HOPS` | 2 | max hop depth cap for traverse (depth adapts to the question) |
| `API_KEY` | — | `X-API-Key` required on all routes except `/healthz`+`/readyz`; unset = auth off |
| `MAX_UPLOAD_BYTES` | 25 MB | upload size cap → `413` when exceeded |
| `GRAPHRAG_COMMUNITY_LEVEL` | 0 | community level for global search |
| `GRAPHRAG_MAX_GLEANINGS` | 1 | entity-extraction passes |

### Provider-agnostic LLMs

`app/rag/llm.py` is the only interface the app calls. Providers live in
`app/rag/providers/` (anthropic, openai_compat, ollama, fake). Swap by
env only; add a vendor with one file + registry entry. GraphRAG's
internal LLM calls go through LiteLLM to the same endpoint.

### Embedding consistency

One `EmbeddingProvider` instance serves BOTH document indexing
(`embed_documents`) and queries (`embed_query`). Chunks record
`embedding_model`; vector search only compares against the active
model — changing `EMBEDDING_MODEL` requires `scripts/reindex.py`
(re-embeds chunks) **and** a graph rebuild (`POST /ingest`) since
GraphRAG's entity embeddings use the same provider via a custom
`local_st` type registered into `graphrag_llm`.

## API

All routes except `/healthz`/`/readyz` require `X-API-Key: $API_KEY`
when `API_KEY` is configured (unset → auth disabled for local dev).

| Route | Purpose |
|---|---|
| `POST /documents` | upload (multipart) → extract/chunk/embed |
| `GET /documents`, `DELETE /documents/{id}` | list / remove |
| `POST /ingest` | async GraphRAG index job; `GET /ingest/{job}` status |
| `POST /chat` | `{question, mode?, conversation_id?, k?}` → answer + citations |
| `GET /sources/{doc_id}` | source document text (citation drill-down) |
| `GET /graph/{stats,entities,relationships}` | browse the knowledge graph |
| `GET /healthz`, `GET /readyz` | liveness / index-readiness |

## Evaluation

```bash
.venv/bin/python scripts/eval_retrieval.py     # recall@k + MRR per retriever
.venv/bin/python scripts/eval_qa.py            # multi-hop answer correctness (needs API + index)
```

## Tests

```bash
.venv/bin/python -m pytest        # 22 tests, fully offline (fake LLM + hash embedder)
.venv/bin/ruff check .
```

## Notes & limitations

- `--loop asyncio` is required: `graphrag_llm` calls `nest_asyncio2.apply()`
  at import, which cannot patch uvloop.
- GraphRAG indexing is LLM-heavy (extraction + community reports). It
  re-indexes the whole staged corpus; incremental updates are limited.
- `GRAPHRAG_COMMUNITY_LEVEL` must match levels actually produced by
  Leiden — small corpora produce only level 0.
- This build dropped the original multi-tenant/permission-aware
  requirements by decision — see git history / plan docs.
- Auth is a shared `X-API-Key`, not per-user identity — there is no
  authorization layer beyond "has the key".
- `POST /ingest` allows one live job (atomic in-process); a
  multi-worker deployment would need DB-level coordination.
- `POST /ingest` keeps the previous graph output until the new build
  succeeds — a failed reindex restores it automatically.
