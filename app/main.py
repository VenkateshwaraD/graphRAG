import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.api.deps import require_api_key, warn_if_auth_disabled
from app.api.routers import chat, documents, graph, health, ingestion, sources
from app.db.session import init_db
from app.ingestion.graph_index import ensure_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup: create DB tables and generate GraphRAG settings.yaml.

    Author: venkateshwara doijode
    """
    init_db()
    ensure_settings()
    warn_if_auth_disabled()
    yield


app = FastAPI(title="University GraphRAG Assistant", lifespan=lifespan)

protected = [Depends(require_api_key)]

app.include_router(health.router)
app.include_router(documents.router, dependencies=protected)
app.include_router(ingestion.router, dependencies=protected)
app.include_router(chat.router, dependencies=protected)
app.include_router(sources.router, dependencies=protected)
app.include_router(graph.router, dependencies=protected)
