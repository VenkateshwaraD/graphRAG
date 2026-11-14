from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, require_write
from app.db.models import RelationshipReview
from app.db.session import get_session
from app.retrieval.artifacts import load_artifacts
from app.retrieval.review import record_review, rel_key
from app.schemas import (
    EntityOut,
    GraphStatsOut,
    RelationshipOut,
    ReviewIn,
    ReviewItemOut,
    ReviewOut,
    SubgraphEdgeOut,
    SubgraphNodeOut,
    SubgraphOut,
)

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


@router.get("/subgraph/{entity}", response_model=SubgraphOut)
def subgraph(entity: str, depth: int = 1):
    """Return the ``depth``-hop neighborhood of one entity for
    visualization — nodes plus directed edges with descriptions.

    Matches by case-insensitive title; 404 when the entity is unknown.
    ``depth`` is capped at 3 to keep payloads small.

    Author: venkateshwara doijode
    """
    from app.retrieval.traverse import build_graph

    a = load_artifacts()
    g = build_graph(a)
    node = entity.upper()
    if node not in g:
        raise HTTPException(404, f"entity {entity!r} not in graph")
    seen = {node}
    frontier = {node}
    edges: dict[tuple[str, str], dict] = {}
    for _ in range(min(max(depth, 1), 3)):
        nxt = set()
        for n in frontier:
            for nbr in g.neighbors(n):
                edges[(n, nbr)] = g.edges[n, nbr]
                if nbr not in seen:
                    seen.add(nbr)
                    nxt.add(nbr)
        frontier = nxt
    return SubgraphOut(
        nodes=[
            SubgraphNodeOut(
                name=n,
                description=str(g.nodes[n].get("description", ""))[:300] or None,
            )
            for n in sorted(seen)
        ],
        edges=[
            SubgraphEdgeOut(
                source=u,
                target=v,
                direction=d.get("relation_direction", f"{u}->{v}"),
                description=str(d.get("description", ""))[:300] or None,
                weight=float(d.get("weight", 0) or 0),
            )
            for (u, v), d in sorted(edges.items())
        ],
    )


@router.get("/review", response_model=list[ReviewOut])
def list_reviews(session: Session = Depends(get_session)):
    """List recorded human review verdicts, newest first.

    Author: venkateshwara doijode
    """
    return (
        session.query(RelationshipReview)
        .order_by(RelationshipReview.created_at.desc())
        .limit(200)
        .all()
    )


@router.get("/review/queue", response_model=list[ReviewItemOut])
def review_queue(limit: int = 50, session: Session = Depends(get_session)):
    """Relationships not yet reviewed — the human-in-the-loop worklist.

    Review keys are content hashes, so a rebuilt index only re-queues
    relationships whose description text actually changed.

    Author: venkateshwara doijode
    """
    a = load_artifacts()
    if a.relationships.empty:
        return []
    done = {
        r.rel_hash for r in session.query(RelationshipReview.rel_hash).all()
    }
    out = []
    for _, r in a.relationships.iterrows():
        src, tgt = str(r.get("source", "")), str(r.get("target", ""))
        desc = str(r.get("description", ""))
        key = rel_key(src.upper(), tgt.upper(), desc)
        if key in done:
            continue
        out.append(
            ReviewItemOut(
                rel_hash=key,
                source=src,
                target=tgt,
                description=desc[:500] or None,
                weight=float(r.get("weight", 0) or 0),
            )
        )
        if len(out) >= limit:
            break
    return out


@router.post("/review", response_model=ReviewOut, status_code=201)
def submit_review(
    body: ReviewIn,
    principal: ApiPrincipal = Depends(require_write),
    session: Session = Depends(get_session),
):
    """Record a reviewer verdict (approved|rejected) on a relationship.

    Rejected edges are excluded from traversal on the next /chat call —
    no reindex needed. Posting again on the same relationship updates
    the verdict (undo path for mistakes).

    Author: venkateshwara doijode
    """
    row = record_review(
        session,
        source=body.source.upper(),
        target=body.target.upper(),
        description=body.description,
        verdict=body.verdict,
        note=body.note,
    )
    return row
