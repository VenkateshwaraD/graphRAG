import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, get_graph, require_api_key
from app.db.models import Document
from app.db.session import get_session
from app.rag.graph import enrich_citations, persist_turn
from app.schemas import ChatRequest, ChatResponse, Citation

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


def _allowed_doc_ids(session: Session, principal: ApiPrincipal) -> set[str] | None:
    """Documents the caller's groups may see; ``None`` = unrestricted.

    Author: venkateshwara doijode
    """
    if principal.groups is None:
        return None
    return {
        d.id
        for d in session.query(Document.id)
        .filter(Document.doc_group.in_(sorted(principal.groups)))
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Answer a question through the agentic LangGraph workflow.

    Runs route -> retrieve -> assess -> generate. ``mode`` forces a
    retriever (invalid values -> 400); ``as_of`` scopes traverse evidence
    to relationships valid by that year; any workflow/provider failure
    becomes a 502 with the error surfaced. Retrieval is restricted to
    documents in the caller's groups (RBAC); out-of-group citations are
    dropped. The turn is persisted for conversation history and
    citations are enriched with document titles.

    Author: venkateshwara doijode
    """
    allowed = _allowed_doc_ids(session, principal)
    from app.observability import langfuse_handler

    cb = langfuse_handler(tags=["chat"])
    config = {"callbacks": [cb]} if cb else {}
    try:
        result = await get_graph().ainvoke(
            {
                "question": req.question,
                "mode": req.mode,
                "k": req.k,
                "as_of": req.as_of,
                "allowed_doc_ids": allowed,
            },
            config=config,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("rag workflow failed")
        raise HTTPException(502, f"answer generation failed: {exc}") from exc

    answer = result.get("answer") or "No answer produced."
    route = result.get("route", "hybrid")
    route_reason = result.get("route_reason")
    citations = enrich_citations(session, result.get("citations") or [])
    if allowed is not None:
        # local/global citations bypass doc-scoped retrieval — drop any
        # pointing at documents outside the caller's groups.
        citations = [
            c for c in citations
            if not c.get("document_id") or c["document_id"] in allowed
        ]
    convo_id = persist_turn(
        session, req.conversation_id, req.question, answer, route, citations,
        route_reason=route_reason,
        metrics=result.get("metrics"),
    )
    return ChatResponse(
        answer=answer,
        route=route,
        route_reason=route_reason,
        warning=result.get("warning"),
        conversation_id=convo_id,
        citations=[Citation(**c) for c in citations],
    )
