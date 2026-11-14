from fastapi import APIRouter

from app.retrieval.artifacts import load_artifacts
from app.schemas import EntityOut, GraphStatsOut, RelationshipOut

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/stats", response_model=GraphStatsOut)
def stats():
    """Row counts per artifact frame + whether a graph index exists.

    Author: venkateshwara doijode
    """
    a = load_artifacts()
    return GraphStatsOut(
        entities=len(a.entities),
        relationships=len(a.relationships),
        communities=len(a.communities),
        text_units=len(a.text_units),
        community_reports=len(a.community_reports),
        indexed=a.indexed,
    )


@router.get("/entities", response_model=list[EntityOut])
def entities(limit: int = 100):
    """Browse extracted entities (up to ``limit``) from the built graph.

    Returns [] before any index build. ``title`` is preferred over the
    raw ``name`` column because GraphRAG normalizes titles.

    Author: venkateshwara doijode
    """
    a = load_artifacts()
    if a.entities.empty:
        return []
    out = []
    for _, r in a.entities.head(limit).iterrows():
        out.append(
            EntityOut(
                id=str(r.get("id", "")),
                name=str(r.get("title", r.get("name", ""))),
                type=str(r.get("type", "")) or None,
                description=str(r.get("description", ""))[:500] or None,
                degree=int(r.get("degree", 0) or 0),
            )
        )
    return out


@router.get("/relationships", response_model=list[RelationshipOut])
def relationships(limit: int = 100):
    """Browse extracted relationships (up to ``limit``).

    Returns [] before any index build.

    Author: venkateshwara doijode
    """
    a = load_artifacts()
    if a.relationships.empty:
        return []
    out = []
    for _, r in a.relationships.head(limit).iterrows():
        out.append(
            RelationshipOut(
                source=str(r.get("source", "")),
                target=str(r.get("target", "")),
                description=str(r.get("description", ""))[:500] or None,
                weight=float(r.get("weight", 0) or 0),
            )
        )
    return out
