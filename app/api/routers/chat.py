import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, get_graph, require_api_key
from app.audit import record_audit
from app.authz import doc_scope, unrestricted
from app.db.session import get_session
from app.rag.graph import enrich_citations, persist_turn
from app.schemas import ChatRequest, ChatResponse, Citation

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


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
    becomes a 502 with the error surfaced. Retrieval and graph context
    are restricted to documents the caller's identity may see — scoped
    callers are rerouted off the GraphRAG composer routes
    (``local``/``global``) which cannot enforce per-document
    authorization, and the substitution is reported via
    ``requested_route``. Out-of-scope citations are dropped and the turn
    persisted for conversation history.

    Author: venkateshwara doijode
    """
    # concrete live doc set the caller may see — never None; the graph
    # also receives allowed_doc_ids (None iff unrestricted) to decide
    # whether the composer routes are permitted at all.
    scope = doc_scope(session, principal)
    allowed = None if unrestricted(principal) else scope
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
                "doc_scope": scope,
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
        # residual guard: engine-produced citations can't point outside
        # the caller's document set
        citations = [
            citation
            for citation in citations
            if not citation.get("document_id")
            or citation["document_id"] in allowed
        ]
    # a document deleted mid-request must not linger as a citation —
    # unresolved document_ids (no live Document row) are dropped
    citations = [
        citation
        for citation in citations
        if not citation.get("document_id") or citation.get("document_title")
    ]
    convo_id = persist_turn(
        session, req.conversation_id, req.question, answer, route, citations,
        route_reason=route_reason,
        metrics=result.get("metrics"),
        owner_id=principal.user_id,
        tenant_id=principal.tenant_id,
    )
    record_audit(
        session,
        action="chat.ask",
        principal=principal,
        resource_type="conversation",
        resource_id=convo_id,
        decision="allow",
        detail={"route": route},
    )
    return ChatResponse(
        answer=answer,
        route=route,
        requested_route=result.get("requested_route"),
        route_reason=route_reason,
        warning=result.get("warning"),
        conversation_id=convo_id,
        citations=[Citation(**citation) for citation in citations],
    )
