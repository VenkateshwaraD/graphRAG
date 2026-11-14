import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from app.api.deps import check_bootstrap_auth, require_api_key
from app.api.routers import (
    audit,
    catalog,
    chat,
    documents,
    graph,
    health,
    ingestion,
    sources,
    tenants,
    users,
)
from app.db.session import init_db
from app.ingestion.graph_index import ensure_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup: run migrations, generate GraphRAG settings.yaml, and
    verify bootstrap credentials aren't insecure defaults.

    Author: venkateshwara doijode
    """
    init_db()
    ensure_settings()
    check_bootstrap_auth()
    yield


app = FastAPI(title="University GraphRAG Assistant", lifespan=lifespan)

protected = [Depends(require_api_key)]

app.include_router(health.router)
app.include_router(documents.router, dependencies=protected)
app.include_router(ingestion.router, dependencies=protected)
app.include_router(chat.router, dependencies=protected)
app.include_router(sources.router, dependencies=protected)
app.include_router(graph.router, dependencies=protected)
app.include_router(tenants.router, dependencies=protected)
app.include_router(users.router, dependencies=protected)
app.include_router(catalog.router, dependencies=protected)
app.include_router(audit.router, dependencies=protected)
