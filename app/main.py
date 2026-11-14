import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

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
    yield


app = FastAPI(title="University GraphRAG Assistant", lifespan=lifespan)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(ingestion.router)
app.include_router(chat.router)
app.include_router(sources.router)
app.include_router(graph.router)
