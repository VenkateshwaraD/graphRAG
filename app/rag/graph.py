"""LangGraph agentic RAG workflow.

route -> retrieve -> assess (expand/fallback once) -> generate -> persist

Routes:
  hybrid   -> pgvector-style LanceDB vector + FTS (RRF), answer via LLMProvider
  traverse -> deterministic k-hop BFS over the knowledge graph, answer via LLMProvider
  local    -> GraphRAG LocalSearch (engine composes its own answer)
  global   -> GraphRAG GlobalSearch (engine composes its own answer)
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.db.models import Conversation, Document, Message
from app.ingestion.embed import EmbeddingProvider
from app.rag.llm import LLMProvider
from app.rag.prompts import ANSWER_SYSTEM, ASSESS_SYSTEM, INSUFFICIENT_CONTEXT
from app.rag.router import route_question
from app.retrieval.artifacts import load_artifacts
from app.retrieval.vectorstore import LanceChunkStore, RetrievedChunk

logger = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph workflow.

    Author: venkateshwara doijode
    """

    question: str
    mode: str | None
    k: int
    route: str
    expanded: bool
    chunks: list[RetrievedChunk]
    answer: str
    citations: list[dict]
    error: str | None


def build_rag_graph(
    *,
    store: LanceChunkStore,
    embedder: EmbeddingProvider,
    router_llm: LLMProvider,
    answer_llm: LLMProvider,
    k_hops: int,
    max_tokens: int,
):
    """Compile the route -> retrieve -> assess -> generate LangGraph.

    All dependencies are injected (chunk store, shared embedder, router +
    answer LLMs, traversal depth, generation token cap) so the same graph
    topology runs with any provider config — tests swap in fake/hash
    implementations without touching this code.

    Author: venkateshwara doijode
    """

    async def route_node(state: RAGState) -> dict:
        """Pick the retrieval route (LLM classifier or ``mode`` override).

        Author: venkateshwara doijode
        """
        route = route_question(state["question"], router_llm, state.get("mode"))
        return {"route": route}

    async def retrieve_node(state: RAGState) -> dict:
        """Dispatch to the selected retriever and normalize results.

        ``hybrid`` and ``traverse`` produce ``chunks`` for our own LLM to
        answer over; ``local``/``global`` return the engine-composed
        ``answer`` + ``citations`` directly. Graph routes without a built
        index short-circuit with an insufficient-context answer. Any
        retriever failure degrades to ``hybrid`` rather than 500ing.

        Author: venkateshwara doijode
        """
        route = state["route"]
        if route == "hybrid":
            from app.retrieval.hybrid import HybridRetriever

            chunks = HybridRetriever(store, embedder).retrieve(
                state["question"], k=state["k"]
            )
            return {"chunks": chunks}

        artifacts = load_artifacts()
        if route == "traverse":
            from app.retrieval.traverse import traverse

            try:
                chunks = traverse(
                    state["question"], artifacts, embedder,
                    max_hops=k_hops, max_items=state["k"],
                    router_llm=router_llm,
                )
            except Exception:
                logger.exception("traverse failed; falling back to hybrid")
                from app.retrieval.hybrid import HybridRetriever

                chunks = HybridRetriever(store, embedder).retrieve(
                    state["question"], k=state["k"]
                )
                return {"route": "hybrid", "chunks": chunks, "expanded": True}
            return {"chunks": chunks}

        if not artifacts.indexed:
            return {"error": "no_graph", "chunks": [], "answer": INSUFFICIENT_CONTEXT}

        try:
            if route == "local":
                from app.retrieval.graph_search import local_search

                res = await local_search(state["question"], artifacts)
                return {"answer": res.answer, "citations": res.citations}
            if route == "global":
                from app.retrieval.graph_search import global_search

                res = await global_search(state["question"], artifacts)
                return {"answer": res.answer, "citations": res.citations}
        except Exception:
            logger.exception("%s search failed; falling back to hybrid", route)
            from app.retrieval.hybrid import HybridRetriever

            chunks = HybridRetriever(store, embedder).retrieve(
                state["question"], k=state["k"]
            )
            return {"route": "hybrid", "chunks": chunks, "expanded": True}
        return {"error": f"unknown route {route}"}

    async def assess_node(state: RAGState) -> dict:
        """Quality gate: retry when retrieval failed OR returned irrelevant
        chunks — never more than once.

        Empty traverse -> hybrid fallback; empty hybrid -> traverse once
        when a graph index exists. Non-empty chunks get one cheap
        router-LLM relevance verdict; a "no" triggers a single
        alternate-retriever attempt. Skips entirely when a route errored,
        the engine already answered, or a fallback already ran
        (``expanded`` prevents re-loops). Traversal depth is decided
        inside ``traverse`` — this node does not deepen hops.

        Author: venkateshwara doijode
        """
        if state.get("error") or state.get("answer") or state.get("expanded"):
            return {}
        route = state["route"]
        chunks = state.get("chunks") or []

        def _hybrid() -> list[RetrievedChunk]:
            from app.retrieval.hybrid import HybridRetriever

            return HybridRetriever(store, embedder).retrieve(
                state["question"], k=state["k"]
            )

        def _traverse() -> list[RetrievedChunk]:
            from app.retrieval.traverse import traverse

            return traverse(
                state["question"], load_artifacts(), embedder,
                max_hops=k_hops, max_items=state["k"], router_llm=router_llm,
            )

        if not chunks:
            if route == "traverse":
                return {"route": "hybrid", "chunks": _hybrid(), "expanded": True}
            if load_artifacts().indexed:
                t_chunks = _traverse()
                if t_chunks:
                    return {
                        "route": "traverse",
                        "chunks": t_chunks,
                        "expanded": True,
                    }
            return {}

        preview = "\n".join(f"- {c.text[:200]}" for c in chunks[:4])
        try:
            verdict = router_llm.generate(
                [
                    {"role": "system", "content": ASSESS_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {state['question']}\n"
                            f"Snippets:\n{preview}\n"
                            "Can these snippets answer the question?"
                        ),
                    },
                ],
                max_tokens=4,
            ).strip().lower()
        except Exception:
            return {}
        if not verdict.startswith("no"):
            return {}
        if route == "traverse":
            alt_chunks = _hybrid()
            if alt_chunks:
                return {
                    "route": "hybrid",
                    "chunks": alt_chunks,
                    "expanded": True,
                }
        elif load_artifacts().indexed:
            alt_chunks = _traverse()
            if alt_chunks:
                return {
                    "route": "traverse",
                    "chunks": alt_chunks,
                    "expanded": True,
                }
        return {}

    async def generate_node(state: RAGState) -> dict:
        """Synthesize the final grounded answer + numbered citations.

        Skipped when local/global already answered. Empty chunk lists get
        a fixed insufficient-context reply; otherwise context items are
        numbered, the answer LLM is asked to cite them inline as [n], and
        each chunk becomes a citation carrying its doc, excerpt, and (for
        traverse) the hop path.

        Author: venkateshwara doijode
        """
        if state.get("answer"):  # local/global already answered
            return {}
        chunks = state.get("chunks") or []
        if not chunks:
            return {"answer": INSUFFICIENT_CONTEXT, "citations": []}
        context_block = "\n\n".join(
            f"[{i+1}] {c.text}" for i, c in enumerate(chunks)
        )
        user = (
            f"Context:\n{context_block}\n\nQuestion: {state['question']}\n"
            "Answer with [n] citations."
        )
        answer = answer_llm.generate(
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        citations = [
            {
                "n": i + 1,
                "document_id": c.document_id,
                "text_unit_id": c.chunk_id if c.source == "traverse" else None,
                "excerpt": c.text[:400],
                "hop_path": c.meta.get("hop_path") if c.meta else None,
            }
            for i, c in enumerate(chunks)
        ]
        return {"answer": answer, "citations": citations}

    g = StateGraph(RAGState)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("assess", assess_node)
    g.add_node("generate", generate_node)
    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "assess")
    g.add_edge("assess", "generate")
    g.add_edge("generate", END)
    return g.compile()


def persist_turn(
    session: Session,
    conversation_id: str | None,
    question: str,
    answer: str,
    route: str,
    citations: list[dict],
) -> str:
    """Persist one chat turn and return the conversation id.

    Resumes the given conversation when it exists, otherwise creates one —
    a stale/unknown id silently starts fresh rather than erroring. Stores
    the user message plus the assistant message with route + citations.

    Author: venkateshwara doijode
    """
    convo = session.get(Conversation, conversation_id) if conversation_id else None
    if convo is None:
        convo = Conversation()
        session.add(convo)
        session.flush()
    session.add(Message(conversation_id=convo.id, role="user", content=question))
    session.add(
        Message(
            conversation_id=convo.id,
            role="assistant",
            content=answer,
            citations={"items": citations},
            route=route,
        )
    )
    session.commit()
    return convo.id


def enrich_citations(session: Session, citations: list[dict]) -> list[dict]:
    """Attach ``document_title`` to each citation for UI display.

    One batched query resolves all referenced document ids; unresolvable
    (deleted/graphrag-internal) ids get None rather than errors.

    Author: venkateshwara doijode
    """
    doc_ids = {c["document_id"] for c in citations if c.get("document_id")}
    titles = {
        d.id: d.title
        for d in session.query(Document).filter(Document.id.in_(doc_ids)).all()
    } if doc_ids else {}
    for c in citations:
        c["document_title"] = titles.get(c.get("document_id"))
    return citations
