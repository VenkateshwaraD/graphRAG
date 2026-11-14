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
from app.rag.router import route_with_reason
from app.retrieval.artifacts import load_artifacts
from app.retrieval.vectorstore import LanceChunkStore, RetrievedChunk

logger = logging.getLogger(__name__)


def _validate(
    answer: str, chunks: list[RetrievedChunk], router_llm: LLMProvider
) -> tuple[dict[int, bool], str | None]:
    """Verify claim-to-source support and flag contradictory evidence.

    One router-model call grades each snippet against the answer
    (``n:yes``/``n:no``) and reports conflicting snippet pairs
    (``conflicts: n vs m``). Returns {citation_n: supported} — missing
    entries mean "unknown" — plus a human-readable warning when sources
    disagree. Any LLM/parse failure degrades to empty results, never an
    error.

    Author: venkateshwara doijode
    """
    import re

    from app.rag.prompts import VALIDATE_SYSTEM

    preview = "\n".join(f"[{i+1}] {c.text[:200]}" for i, c in enumerate(chunks))
    try:
        raw = router_llm.generate(
            [
                {"role": "system", "content": VALIDATE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Answer:\n{answer}\n\nSnippets:\n{preview}",
                },
            ],
            max_tokens=128,
        ).lower()
    except Exception:
        return {}, None
    supported = {
        int(n): v == "yes"
        for n, v in re.findall(r"(\d+)\s*[:.\-)]\s*(yes|no)\b", raw)
    }
    conflicts = re.findall(r"(\d+)\s+vs\.?\s+(\d+)", raw)
    warning = (
        "sources disagree: "
        + ", ".join(f"[{a}] vs [{b}]" for a, b in conflicts)
        if conflicts
        else None
    )
    return supported, warning


class RAGState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph workflow.

    Author: venkateshwara doijode
    """

    question: str
    mode: str | None
    k: int
    as_of: int | None
    allowed_doc_ids: set[str] | None
    doc_scope: set[str] | None
    route: str
    requested_route: str
    route_reason: str
    expanded: bool
    chunks: list[RetrievedChunk]
    answer: str
    citations: list[dict]
    warning: str | None
    metrics: dict
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

    def _timed(name, fn):
        """Wrap a node so its wall-clock time lands in state.metrics.

        Author: venkateshwara doijode
        """
        import time

        async def wrapper(state: RAGState) -> dict:
            t0 = time.perf_counter()
            out = await fn(state) or {}
            # merge node's own metrics contributions, then add timing
            metrics = {
                **(state.get("metrics") or {}),
                **(out.get("metrics") or {}),
                f"{name}_ms": round((time.perf_counter() - t0) * 1000),
            }
            return {**out, "metrics": metrics}

        return wrapper

    async def route_node(state: RAGState) -> dict:
        """Pick the retrieval route (LLM classifier or ``mode`` override)
        and keep the router's raw reply for explainability.

        Author: venkateshwara doijode
        """
        route, reason = route_with_reason(
            state["question"], router_llm, state.get("mode")
        )
        # Scoped callers can't run the engine's own composer over
        # shared artifacts — it has no per-document authorization.
        # Reroute to retrievers that enforce the caller's doc scope;
        # requested_route records what the router/mode asked for.
        if state.get("allowed_doc_ids") is not None and route in ("local", "global"):
            rerouted = "traverse" if route == "local" else "hybrid"
            return {
                "route": rerouted,
                "requested_route": route,
                "route_reason": f"{reason}; scoped caller: {route}->{rerouted}",
            }
        return {"route": route, "route_reason": reason}

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
                state["question"], k=state["k"],
                doc_ids=state.get("allowed_doc_ids"),
            )
            return {"chunks": chunks}

        artifacts = load_artifacts()
        # every graph consumer sees only artifacts provable from docs
        # in the caller's scope — deleted docs are pruned the same way
        scope = state.get("doc_scope")
        if scope is not None:
            from app.retrieval.artifacts import scope_artifacts

            artifacts = scope_artifacts(artifacts, scope)
        if route == "traverse":
            from app.retrieval.review import load_rejections
            from app.retrieval.traverse import traverse

            try:
                chunks = traverse(
                    state["question"], artifacts, embedder,
                    max_hops=k_hops, max_items=state["k"],
                    router_llm=router_llm, as_of=state.get("as_of"),
                    rejections=load_rejections(),
                    allowed_doc_ids=state.get("allowed_doc_ids"),
                )
            except Exception:
                logger.exception("traverse failed; falling back to hybrid")
                from app.retrieval.hybrid import HybridRetriever

                chunks = HybridRetriever(store, embedder).retrieve(
                    state["question"], k=state["k"],
                    doc_ids=state.get("allowed_doc_ids"),
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
                state["question"], k=state["k"],
                doc_ids=state.get("allowed_doc_ids"),
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
                state["question"], k=state["k"],
                doc_ids=state.get("allowed_doc_ids"),
            )

        scope = state.get("doc_scope")

        def _arts():
            """Artifacts scoped to the caller's live doc set — the same
            provenance view retrieve_node used.

            Author: venkateshwara doijode
            """
            from app.retrieval.artifacts import scope_artifacts

            arts = load_artifacts()
            return (
                scope_artifacts(arts, scope)
                if scope is not None
                else arts
            )

        def _traverse() -> list[RetrievedChunk]:
            from app.retrieval.review import load_rejections
            from app.retrieval.traverse import traverse

            return traverse(
                state["question"], _arts(), embedder,
                max_hops=k_hops, max_items=state["k"], router_llm=router_llm,
                as_of=state.get("as_of"), rejections=load_rejections(),
                allowed_doc_ids=state.get("allowed_doc_ids"),
            )

        if not chunks:
            if route == "traverse":
                return {"route": "hybrid", "chunks": _hybrid(), "expanded": True}
            if _arts().indexed:
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
        elif _arts().indexed:
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
                "edge_directions": (
                    c.meta.get("edge_directions") if c.meta else None
                ),
                "alt_paths": c.meta.get("alt_paths") if c.meta else None,
                "years": c.meta.get("years") if c.meta else None,
                "score": round(float(c.score), 4),
            }
            for i, c in enumerate(chunks)
        ]
        supported, warning = _validate(answer, chunks, router_llm)
        for i, c in enumerate(citations):
            c["supported"] = supported.get(i + 1)
        metrics = dict(state.get("metrics") or {})
        metrics["n_chunks"] = len(chunks)
        metrics["n_unsupported"] = sum(
            1 for v in supported.values() if v is False
        )
        return {
            "answer": answer,
            "citations": citations,
            "warning": warning,
            "metrics": metrics,
        }

    g = StateGraph(RAGState)
    g.add_node("route", _timed("route", route_node))
    g.add_node("retrieve", _timed("retrieve", retrieve_node))
    g.add_node("assess", _timed("assess", assess_node))
    g.add_node("generate", _timed("generate", generate_node))
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
    route_reason: str | None = None,
    metrics: dict | None = None,
    owner_id: str | None = None,
    tenant_id: str | None = None,
) -> str:
    """Persist one chat turn and return the conversation id.

    Resumes the given conversation when it exists AND belongs to the
    caller — a stale, unknown, or foreign-owned id silently starts
    fresh rather than leaking history or appending to another user's
    thread. Stores the user message plus the assistant message with
    route + rationale + citations + per-node latency/quality metrics.

    Author: venkateshwara doijode
    """
    convo = session.get(Conversation, conversation_id) if conversation_id else None
    if convo is not None:
        # foreign-owned conversations are invisible; ownerless legacy
        # rows are claimed by the first authenticated caller
        if convo.owner_id is not None and convo.owner_id != owner_id:
            convo = None
        elif convo.tenant_id is not None and convo.tenant_id != tenant_id:
            convo = None
    if convo is None:
        convo = Conversation(owner_id=owner_id, tenant_id=tenant_id)
        session.add(convo)
        session.flush()
    elif convo.owner_id is None:
        convo.owner_id = owner_id
        convo.tenant_id = tenant_id
    session.add(Message(conversation_id=convo.id, role="user", content=question))
    session.add(
        Message(
            conversation_id=convo.id,
            role="assistant",
            content=answer,
            citations={"items": citations},
            route=route,
            route_reason=route_reason,
            metrics=metrics,
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
    doc_ids = {
        citation["document_id"]
        for citation in citations
        if citation.get("document_id")
    }
    titles = {
        doc.id: doc.title
        for doc in session.query(Document)
        .filter(Document.id.in_(doc_ids))
        .all()
    } if doc_ids else {}
    for citation in citations:
        citation["document_title"] = titles.get(citation.get("document_id"))
    return citations
