from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, require_api_key, require_write
from app.authz import doc_scope, unrestricted
from app.db.models import Document, RelationshipReview
from app.db.session import get_session
from app.retrieval.artifacts import load_artifacts, scope_artifacts, stale_doc_ids
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


def _scoped(session: Session, principal: ApiPrincipal):
    """Artifacts restricted to what the caller's documents can prove —
    deleted docs are pruned identically since their ids are gone.

    Author: venkateshwara doijode
    """
    return scope_artifacts(load_artifacts(), doc_scope(session, principal))


@router.get("/stats", response_model=GraphStatsOut)
def stats(
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Row counts per artifact frame + whether a graph index exists.

    Counts reflect the caller's scoped view. ``stale`` is true when the
    index still contains documents that no longer exist (deleted since
    the last build) — their artifacts never reach responses, but a
    rebuild clears them for real.

    Author: venkateshwara doijode
    """
    arts = _scoped(session, principal)
    live = {row[0] for row in session.query(Document.id)}
    stale = stale_doc_ids(load_artifacts(), live)
    return GraphStatsOut(
        entities=len(arts.entities),
        relationships=len(arts.relationships),
        communities=len(arts.communities),
        text_units=len(arts.text_units),
        community_reports=len(arts.community_reports),
        indexed=arts.indexed,
        stale=bool(stale),
        stale_documents=len(stale),
    )


@router.get("/entities", response_model=list[EntityOut])
def entities(
    limit: int = 100,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Browse extracted entities (up to ``limit``) — only those whose
    supporting text units all come from documents the caller may see.

    Author: venkateshwara doijode
    """
    arts = _scoped(session, principal)
    if arts.entities.empty:
        return []
    out = []
    for _, row in arts.entities.head(limit).iterrows():
        out.append(
            EntityOut(
                id=str(row.get("id", "")),
                name=str(row.get("title", row.get("name", ""))),
                type=str(row.get("type", "")) or None,
                description=str(row.get("description", ""))[:500] or None,
                degree=int(row.get("degree", 0) or 0),
            )
        )
    return out


@router.get("/relationships", response_model=list[RelationshipOut])
def relationships(
    limit: int = 100,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Browse extracted relationships (up to ``limit``) — scoped the
    same way as entities.

    Author: venkateshwara doijode
    """
    arts = _scoped(session, principal)
    if arts.relationships.empty:
        return []
    out = []
    for _, row in arts.relationships.head(limit).iterrows():
        out.append(
            RelationshipOut(
                source=str(row.get("source", "")),
                target=str(row.get("target", "")),
                description=str(row.get("description", ""))[:500] or None,
                weight=float(row.get("weight", 0) or 0),
            )
        )
    return out


@router.get("/subgraph/{entity}", response_model=SubgraphOut)
def subgraph(
    entity: str,
    depth: int = 1,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Return the ``depth``-hop neighborhood of one entity for
    visualization — nodes plus directed edges with descriptions.

    The graph is built from the caller's scoped artifacts: an entity
    provable only from unauthorized documents 404s. ``depth`` is capped
    at 3 to keep payloads small.

    Author: venkateshwara doijode
    """
    from app.retrieval.traverse import build_graph

    arts = _scoped(session, principal)
    net = build_graph(arts)
    node = entity.upper()
    if node not in net:
        raise HTTPException(404, f"entity {entity!r} not in graph")
    seen = {node}
    frontier = {node}
    edges: dict[tuple[str, str], dict] = {}
    for _ in range(min(max(depth, 1), 3)):
        next_frontier = set()
        for current in frontier:
            for neighbor in net.neighbors(current):
                edges[(current, neighbor)] = net.edges[current, neighbor]
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
    return SubgraphOut(
        nodes=[
            SubgraphNodeOut(
                name=name,
                description=str(net.nodes[name].get("description", ""))[:300]
                or None,
            )
            for name in sorted(seen)
        ],
        edges=[
            SubgraphEdgeOut(
                source=src,
                target=tgt,
                direction=edge.get("relation_direction", f"{src}->{tgt}"),
                description=str(edge.get("description", ""))[:300] or None,
                weight=float(edge.get("weight", 0) or 0),
            )
            for (src, tgt), edge in sorted(edges.items())
        ],
    )


@router.get("/review", response_model=list[ReviewOut])
def list_reviews(
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """List recorded human review verdicts the caller may see — verdicts
    on relationships not provable from their documents are hidden.
    Platform administrators see every verdict.

    Author: venkateshwara doijode
    """
    rows = (
        session.query(RelationshipReview)
        .order_by(RelationshipReview.created_at.desc())
        .limit(200)
        .all()
    )
    if unrestricted(principal):
        return rows
    arts = _scoped(session, principal)
    visible_keys = {
        rel_key(
            str(row.get("source", "")).upper(),
            str(row.get("target", "")).upper(),
            str(row.get("description", "")),
        )
        for _, row in arts.relationships.iterrows()
    } if not arts.relationships.empty else set()
    return [review for review in rows if review.rel_hash in visible_keys]


@router.get("/review/queue", response_model=list[ReviewItemOut])
def review_queue(
    limit: int = 50,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Relationships not yet reviewed — the human-in-the-loop worklist.

    Review keys are content hashes, so a rebuilt index only re-queues
    relationships whose description text actually changed. The queue is
    scoped to relationships provable from the caller's documents.

    Author: venkateshwara doijode
    """
    arts = _scoped(session, principal)
    if arts.relationships.empty:
        return []
    done = {
        review.rel_hash
        for review in session.query(RelationshipReview.rel_hash).all()
    }
    out = []
    for _, row in arts.relationships.iterrows():
        src, tgt = str(row.get("source", "")), str(row.get("target", ""))
        desc = str(row.get("description", ""))
        key = rel_key(src.upper(), tgt.upper(), desc)
        if key in done:
            continue
        out.append(
            ReviewItemOut(
                rel_hash=key,
                source=src,
                target=tgt,
                description=desc[:500] or None,
                weight=float(row.get("weight", 0) or 0),
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
