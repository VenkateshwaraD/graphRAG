"""Explicit multi-hop retriever over the GraphRAG knowledge graph.

Entity-link the question -> BFS K hops over the relationship graph ->
collect text units attached to visited nodes/edges with the entity
chain as auditable provenance. Deterministic multi-hop, unlike LLM
neighborhood search.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from app.ingestion.embed import EmbeddingProvider
from app.retrieval.artifacts import GraphArtifacts
from app.retrieval.vectorstore import RetrievedChunk


def build_graph(artifacts: GraphArtifacts) -> nx.Graph:
    """Materialize entities/relationships into an undirected NetworkX graph.

    Nodes are uppercased entity titles (GraphRAG normalizes them anyway);
    edges carry weight, description, and the text_unit_ids needed to fetch
    provenance text. Undirected because traversal follows relationships
    both ways, while the BFS path still records direction of discovery.

    Author: venkateshwara doijode
    """
    g = nx.Graph()
    if artifacts.entities.empty:
        return g
    for _, row in artifacts.entities.iterrows():
        name = str(row.get("title", row.get("name", ""))).upper()
        if name:
            g.add_node(name, description=str(row.get("description", "")))
    for _, row in artifacts.relationships.iterrows():
        src, tgt = str(row.get("source", "")).upper(), str(row.get("target", "")).upper()
        tu_ids = row.get("text_unit_ids")
        if src and tgt:
            g.add_edge(
                src,
                tgt,
                weight=float(row.get("weight", 1.0) or 1.0),
                description=str(row.get("description", "")),
                text_unit_ids=list(tu_ids) if tu_ids is not None else [],
            )
    return g


def link_entities(
    question: str, g: nx.Graph, embedder: EmbeddingProvider, top_n: int = 5
) -> list[str]:
    """Match question to graph entities: exact substring first, then
    embedding similarity over entity names via the shared provider.

    Exact uppercase-substring hits short-circuit (a named entity in the
    question is the strongest signal); otherwise entity names are embedded
    with the shared provider and cosine-scored against the question,
    keeping names above a 0.25 similarity floor.

    Author: venkateshwara doijode
    """
    q_upper = question.upper()
    exact = [n for n in g.nodes if n in q_upper]
    if exact:
        return exact[:top_n]
    names = list(g.nodes)
    if not names:
        return []
    q_vec = np.array(embedder.embed_query(question), dtype=np.float32)
    name_vecs = np.array(embedder.embed_documents(names), dtype=np.float32)
    sims = name_vecs @ q_vec
    idx = np.argsort(-sims)[:top_n]
    return [names[i] for i in idx if sims[i] > 0.25]


def _text_units_for(
    hop_paths: dict[tuple[str, str], list[str]],
    g: nx.Graph,
    artifacts: GraphArtifacts,
) -> list[dict]:
    """Text units attached to traversed edges, each carrying the entity
    path that reached it (first path found wins).

    For every BFS edge in ``hop_paths``, looks up the edge's text_unit_ids
    in the graph and maps each text unit to the path of entities that led
    there — that's the multi-hop provenance surfaced in citations.
    Returns [] when the graph produced no edges or no text units exist.

    Author: venkateshwara doijode
    """
    tu_paths: dict[str, list[str]] = {}
    tu_ids: list[str] = []
    for (u, v), path in hop_paths.items():
        edge = g.edges.get((u, v)) or g.edges.get((v, u))
        if not edge:
            continue
        for tu in edge.get("text_unit_ids", []):
            tu_ids.append(tu)
            tu_paths.setdefault(tu, path)
    if artifacts.text_units.empty or not tu_ids:
        return []
    tu = artifacts.text_units
    rows = tu[tu["id"].isin(set(tu_ids))]
    doc_map = artifacts.doc_id_map
    out = []
    for _, r in rows.iterrows():
        graphrag_doc = str(r.get("document_id", ""))
        out.append(
            {
                "text_unit_id": str(r["id"]),
                "document_id": doc_map.get(graphrag_doc, graphrag_doc),
                "text": str(r.get("text", "")),
                "path": tu_paths.get(str(r["id"]), []),
            }
        )
    return out


def traverse(
    question: str,
    artifacts: GraphArtifacts,
    embedder: EmbeddingProvider,
    k_hops: int = 2,
    max_items: int = 8,
) -> list[RetrievedChunk]:
    """Deterministic multi-hop retrieval, fully auditable.

    Builds the entity graph, links the question to seed entities, BFSs
    ``k_hops`` deep while remembering the entity path per edge, then
    collects each reached text unit with its path as ``hop_path`` meta.
    Returns [] when nothing entity-links — the assess node retries at
    K+1 hops then falls back to hybrid.

    Author: venkateshwara doijode
    """
    g = build_graph(artifacts)
    seeds = link_entities(question, g, embedder)
    if not seeds:
        return []

    # BFS paths from seeds up to k hops — edge -> entity path that reached it
    hop_paths: dict[tuple[str, str], list[str]] = {}
    frontier: list[tuple[str, list[str]]] = [(s, [s]) for s in seeds]
    seen = set(seeds)
    for _ in range(k_hops):
        next_frontier = []
        for node, path in frontier:
            for nbr in g.neighbors(node):
                edge = (node, nbr)
                hop_paths.setdefault(edge, path + [nbr])
                if nbr not in seen:
                    seen.add(nbr)
                    next_frontier.append((nbr, path + [nbr]))
        frontier = next_frontier

    units = _text_units_for(hop_paths, g, artifacts)
    hits = [
        RetrievedChunk(
            chunk_id=u["text_unit_id"],
            document_id=u["document_id"],
            text=u["text"],
            score=1.0,
            source="traverse",
            meta={"hop_path": u["path"], "entities": list(seeds)},
        )
        for u in units[:max_items]
    ]
    return hits
