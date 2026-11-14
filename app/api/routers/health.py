from fastapi import APIRouter

from app.retrieval.artifacts import load_artifacts

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz():
    """Liveness probe — process is up and serving.

    Author: venkateshwara doijode
    """
    return {"status": "ok"}


@router.get("/readyz")
def readyz():
    """Readiness probe — also reports whether a graph index exists.

    Author: venkateshwara doijode
    """
    arts = load_artifacts()
    return {"status": "ok", "graph_indexed": arts.indexed}
