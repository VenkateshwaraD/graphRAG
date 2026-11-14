import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_graph
from app.db.session import get_session
from app.rag.graph import enrich_citations, persist_turn
from app.schemas import ChatRequest, ChatResponse, Citation

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, session: Session = Depends(get_session)):
    """Answer a question through the agentic LangGraph workflow.

    Runs route -> retrieve -> assess -> generate. ``mode`` forces a
    retriever (invalid values -> 400); any workflow/provider failure
    becomes a 502 with the error surfaced. The turn is persisted for
    conversation history and citations are enriched with document titles.

    Author: venkateshwara doijode
    """
    try:
        result = await get_graph().ainvoke(
            {"question": req.question, "mode": req.mode, "k": req.k}
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("rag workflow failed")
        raise HTTPException(502, f"answer generation failed: {exc}") from exc

    answer = result.get("answer") or "No answer produced."
    route = result.get("route", "hybrid")
    citations = enrich_citations(session, result.get("citations") or [])
    convo_id = persist_turn(
        session, req.conversation_id, req.question, answer, route, citations
    )
    return ChatResponse(
        answer=answer,
        route=route,
        conversation_id=convo_id,
        citations=[Citation(**c) for c in citations],
    )
