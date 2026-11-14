"""GraphRAG Local / Global search via the graphrag query API.

The search engine composes its own answer through the configured
OpenAI-compatible LLM endpoint; we use its response text directly and
map the returned context (text units) to citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.retrieval.artifacts import GraphArtifacts


@dataclass
class SearchResult:
    """Answer text plus citation dicts extracted from engine context.

    Author: venkateshwara doijode
    """

    answer: str
    citations: list[dict] = field(default_factory=list)  # {text_unit_id, document_id, excerpt}


def _config():
    """Load GraphRAG config from ``GRAPHRAG_ROOT`` (settings.yaml).

    Author: venkateshwara doijode
    """
    from graphrag.config.load_config import load_config

    return load_config(get_settings().graphrag_root)


def _context_citations(context: Any, artifacts: GraphArtifacts) -> list[dict]:
    """Pull text-unit citations out of whatever context structure the
    search engine returned, mapping graphrag doc ids to our documents.

    The context object varies across graphrag versions — may be a dict of
    DataFrames (``sources``/``source``/``text_units`` keys) or a bare
    DataFrame. Each row becomes a numbered citation; text units carry a
    ``document_id`` mapped through ``artifacts.doc_id_map`` back to our
    ``Document.id``.

    Author: venkateshwara doijode
    """
    import pandas as pd

    cites: list[dict] = []
    sources: pd.DataFrame | None = None
    if isinstance(context, dict):
        for key in ("sources", "source", "text_units"):
            if isinstance(context.get(key), pd.DataFrame):
                sources = context[key]
                break
    elif isinstance(context, pd.DataFrame):
        sources = context
    if sources is None:
        return cites
    doc_map = artifacts.doc_id_map
    for index, (_, row) in enumerate(sources.iterrows()):
        doc_id = row.get("document_id")
        if doc_id is None and row.get("document_ids") is not None:
            ids = list(row["document_ids"])
            doc_id = ids[0] if ids else None
        doc_id = doc_map.get(str(doc_id), str(doc_id)) if doc_id else None
        cites.append(
            {
                "n": index + 1,
                "text_unit_id": str(row.get("id", "")),
                "document_id": doc_id,
                "excerpt": str(row.get("text", ""))[:400],
            }
        )
    return cites


async def _run_in_thread(coro_fn) -> Any:
    """GraphRAG query engines internally nest event loops (nest_asyncio),
    which cannot patch uvloop. Run them in a worker thread with a plain
    asyncio loop instead.

    Author: venkateshwara doijode
    """
    import asyncio

    return await asyncio.to_thread(lambda: asyncio.run(coro_fn()))


async def local_search(question: str, artifacts: GraphArtifacts) -> SearchResult:
    """GraphRAG LocalSearch — entity-neighborhood retrieval + LLM answer.

    Registers the shared ``local_st`` embedding type (the engine builds
    entity-description embeddings on the fly), then runs
    ``api.local_search`` over the current artifacts. Returns the engine's
    own composed answer plus context-derived citations.

    Author: venkateshwara doijode
    """
    from graphrag.api import local_search as api_local_search

    from app.ingestion.graph_index import register_local_embedding

    register_local_embedding()
    settings = get_settings()

    async def call():
        return await api_local_search(
            config=_config(),
            entities=artifacts.entities,
            communities=artifacts.communities,
            community_reports=artifacts.community_reports,
            text_units=artifacts.text_units,
            relationships=artifacts.relationships,
            covariates=None,
            community_level=settings.graphrag_community_level,
            response_type="multiple paragraphs",
            query=question,
        )

    response, context = await _run_in_thread(call)
    answer = response if isinstance(response, str) else str(response)
    return SearchResult(answer=answer, citations=_context_citations(context, artifacts))


def _report_citations(context: Any, artifacts: GraphArtifacts) -> list[dict]:
    """Resolve GlobalSearch context reports back to source documents.

    The engine's context carries a ``reports`` DataFrame of the
    community reports it used; each report maps through
    ``community_reports`` -> ``communities.text_unit_ids`` ->
    ``text_units.document_id`` -> ``artifacts.doc_id_map`` to our
    ``Document.id``s. One citation is emitted per unique document,
    excerpted with the report title + summary so the provenance trail
    stays inspectable.

    Author: venkateshwara doijode
    """
    import pandas as pd

    if not isinstance(context, dict):
        return []
    reports = context.get("reports")
    if not isinstance(reports, pd.DataFrame) or reports.empty:
        return []
    cr, comms, tus = (
        artifacts.community_reports,
        artifacts.communities,
        artifacts.text_units,
    )
    if cr.empty or comms.empty or tus.empty:
        return []

    # report id -> its community number
    rep_comm = {}
    for _, row in cr.iterrows():
        rep_comm[str(row.get("id"))] = str(row.get("community"))
    # community number -> its text unit ids
    comm_tus: dict[str, list[str]] = {}
    for _, row in comms.iterrows():
        ids = row.get("text_unit_ids")
        comm_tus[str(row.get("community"))] = (
            [str(item) for item in ids] if ids is not None else []
        )
    # text unit id -> graphrag document id
    tu_doc = {
        str(row.get("id")): str(row.get("document_id"))
        for _, row in tus.iterrows()
        if row.get("document_id") is not None
    }
    doc_map = artifacts.doc_id_map

    cites: list[dict] = []
    seen_docs: set[str] = set()
    for _, rep in reports.iterrows():
        rep_id = str(rep.get("id"))
        title = str(rep.get("title", "") or "")
        summary = str(rep.get("summary", rep.get("full_content", "")) or "")
        for tu_id in comm_tus.get(rep_comm.get(rep_id, ""), []):
            gdoc = tu_doc.get(tu_id)
            if gdoc is None:
                continue
            our_doc = doc_map.get(gdoc, gdoc)
            if our_doc in seen_docs:
                continue
            seen_docs.add(our_doc)
            cites.append(
                {
                    "n": len(cites) + 1,
                    "text_unit_id": None,
                    "document_id": our_doc,
                    "excerpt": f"{title}: {summary}"[:400],
                }
            )
    return cites


async def global_search(question: str, artifacts: GraphArtifacts) -> SearchResult:
    """GraphRAG GlobalSearch — map-reduce over community reports.

    Suits broad thematic questions (needs no embedding model — reports
    are already built). ``community_level`` defaults to 0 because small
    corpora only produce a shallow hierarchy. Citations are the
    underlying source documents resolved through each used report's
    community text units — see ``_report_citations``.

    Author: venkateshwara doijode
    """
    from graphrag.api import global_search as api_global_search

    async def call():
        return await api_global_search(
            config=_config(),
            entities=artifacts.entities,
            communities=artifacts.communities,
            community_reports=artifacts.community_reports,
            community_level=get_settings().graphrag_community_level,
            dynamic_community_selection=False,
            response_type="multiple paragraphs",
            query=question,
        )

    response, context = await _run_in_thread(call)
    answer = response if isinstance(response, str) else str(response)
    return SearchResult(answer=answer, citations=_report_citations(context, artifacts))
