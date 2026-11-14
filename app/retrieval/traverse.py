"""Guided multi-hop retriever over the GraphRAG knowledge graph.

Query decomposition -> entity linking per anchor -> best-first expansion
keeping several scored candidate paths per node -> early stop once
enough evidence is collected. Deterministic multi-hop retrieval, fully
auditable: every returned text unit carries the entity path that reached
it plus the original relationship direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import networkx as nx
import numpy as np

from app.ingestion.embed import EmbeddingProvider
from app.rag.llm import LLMProvider
from app.rag.prompts import DECOMPOSE_SYSTEM
from app.retrieval.artifacts import GraphArtifacts
from app.retrieval.vectorstore import RetrievedChunk

_MAX_PATHS_PER_NODE = 3
_MAX_FRONTIER = 64
_DEPTH_DECAY = 0.85
_STOPWORDS = frozenset(
    "a an the of in on for to is are was were who what where which when how "
    "does do did that this these those by with from at as and or its it".split()
)


def _terms(text: str) -> frozenset[str]:
    """Content-word tokens of ``text`` for lexical relevance scoring.

    Author: venkateshwara doijode
    """
    return frozenset(
        t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS
    )


def build_graph(artifacts: GraphArtifacts) -> nx.Graph:
    """Materialize entities/relationships into an undirected NetworkX graph.

    Nodes are uppercased entity titles (GraphRAG normalizes them anyway);
    edges carry weight, description, the extracted ``relation_direction``
    (source -> target as extracted), and the text_unit_ids needed to
    fetch provenance text. Undirected because traversal follows
    relationships both ways; the extracted direction is preserved for
    provenance even though it doesn't constrain expansion.

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
                relation_direction=f"{src}->{tgt}",
                text_unit_ids=list(tu_ids) if tu_ids is not None else [],
            )
    return g


def decompose(question: str, router_llm: LLMProvider | None) -> list[str]:
    """Break a multi-hop question into ordered entity anchors.

    One small router-model call lists the entity mentions in traversal
    order (anchor -> intermediate -> target); anchors drive both seed
    linking and the dynamic hop target. Returns ``[question]`` when no
    LLM is supplied (tests) or the output is unusable, so traversal
    always degrades to plain entity linking rather than failing.

    Author: venkateshwara doijode
    """
    if router_llm is None:
        return [question]
    try:
        raw = router_llm.generate(
            [
                {"role": "system", "content": DECOMPOSE_SYSTEM},
                {"role": "user", "content": question},
            ],
            max_tokens=64,
        )
    except Exception:
        return [question]
    anchors = [
        re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
        for line in raw.strip().splitlines()
    ]
    anchors = [a for a in anchors if a]
    return anchors or [question]


def _link_scored(
    text: str, g: nx.Graph, embedder: EmbeddingProvider, top_n: int = 3
) -> list[tuple[str, float]]:
    """Link a phrase to graph entities, returning (node, score) pairs.

    Exact uppercase-substring hits short-circuit at score 1.0; otherwise
    entity names are embedded with the shared provider and cosine-scored
    against the phrase, keeping names above a 0.25 similarity floor.

    Author: venkateshwara doijode
    """
    q_upper = text.upper()
    exact = [n for n in g.nodes if n in q_upper]
    if exact:
        return [(n, 1.0) for n in exact[:top_n]]
    names = list(g.nodes)
    if not names:
        return []
    q_vec = np.array(embedder.embed_query(text), dtype=np.float32)
    name_vecs = np.array(embedder.embed_documents(names), dtype=np.float32)
    sims = name_vecs @ q_vec
    idx = np.argsort(-sims)[:top_n]
    return [(names[i], float(sims[i])) for i in idx if sims[i] > 0.25]


def link_entities(
    question: str, g: nx.Graph, embedder: EmbeddingProvider, top_n: int = 5
) -> list[str]:
    """Match question to graph entities: exact substring first, then
    embedding similarity over entity names via the shared provider.

    Author: venkateshwara doijode
    """
    return [n for n, _ in _link_scored(question, g, embedder, top_n)]


@dataclass
class _Path:
    """One candidate traversal path and its accumulated relevance score.

    ``seed_score`` is the entity-linking confidence of the anchor node;
    ``edge_sum`` accumulates per-edge relevance so ``score`` is the
    seed confidence times the mean edge score, decayed by depth —
    shorter, more relevant chains win.

    Author: venkateshwara doijode
    """

    node: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    seed_score: float
    edge_sum: float = 0.0

    @property
    def score(self) -> float:
        if not self.edges:
            return self.seed_score
        return (
            self.seed_score
            * (self.edge_sum / len(self.edges))
            * (_DEPTH_DECAY ** len(self.edges))
        )


def _edge_score(
    q_terms: frozenset[str], attrs: dict, max_weight: float
) -> float:
    """Relevance of one edge: normalized GraphRAG weight blended with
    lexical overlap between the edge description and the question.

    The weight factor dominates; description overlap differentiates
    "oversees" from "located in" when the question asks about reporting
    lines, without requiring extra embedding calls (and staying
    meaningful under the hash embedder used in tests).

    Author: venkateshwara doijode
    """
    weight = float(attrs.get("weight", 1.0) or 1.0)
    norm_w = weight / max_weight if max_weight > 0 else 1.0
    desc = _terms(str(attrs.get("description", "")))
    rel = len(q_terms & desc) / len(q_terms | desc) if desc else 0.0
    return norm_w * (0.4 + 0.6 * rel)


def _evidence_path(rec: _Path) -> list[str]:
    """Entity names along the path — the auditable evidence chain.

    Author: venkateshwara doijode
    """
    return list(rec.nodes)


def _expand(
    g: nx.Graph,
    seeds: list[tuple[str, float]],
    q_terms: frozenset[str],
    target_hops: int,
    max_items: int,
) -> dict[str, dict]:
    """Best-first expansion collecting text-unit evidence along paths.

    Each hop extends every live path to neighbors not already on it
    (per-path cycle detection); up to ``_MAX_PATHS_PER_NODE`` distinct
    paths may reach the same node so alternative routes compete. Path
    score = seed link score x mean edge score x depth decay. Expansion
    stops early once ``max_items`` evidence units are collected — it
    never grows the frontier just because hops remain.

    Returns text_unit_id -> {path, edge_directions, score} keeping the
    best-scoring path per unit.

    Author: venkateshwara doijode
    """
    max_weight = max(
        (float(d.get("weight", 1.0) or 1.0) for _, _, d in g.edges(data=True)),
        default=1.0,
    )
    frontier: list[_Path] = [
        _Path(node=n, nodes=(n,), edges=(), seed_score=s) for n, s in seeds
    ]
    best_per_node: dict[str, list[float]] = {}
    evidence: dict[str, dict] = {}

    for _ in range(target_hops):
        if not frontier or len(evidence) >= max_items:
            break
        nxt: list[_Path] = []
        for rec in frontier:
            for nbr in g.neighbors(rec.node):
                if nbr in rec.nodes:
                    continue  # cycle within this path
                attrs = g.edges[rec.node, nbr]
                path = _Path(
                    node=nbr,
                    nodes=rec.nodes + (nbr,),
                    edges=rec.edges + ((rec.node, nbr),),
                    seed_score=rec.seed_score,
                    edge_sum=rec.edge_sum
                    + _edge_score(q_terms, attrs, max_weight),
                )
                # cap alternative paths to the same node by score
                seen_scores = best_per_node.setdefault(nbr, [])
                if (
                    len(seen_scores) >= _MAX_PATHS_PER_NODE
                    and path.score <= min(seen_scores)
                ):
                    continue
                seen_scores.append(path.score)
                seen_scores.sort(reverse=True)
                del seen_scores[_MAX_PATHS_PER_NODE:]
                nxt.append(path)
                for tu in attrs.get("text_unit_ids", []):
                    prev = evidence.get(tu)
                    if prev is not None and prev["score"] >= path.score:
                        continue
                    evidence[tu] = {
                        "path": _evidence_path(path),
                        "edge_directions": [
                            g.edges[u, v].get("relation_direction", "")
                            for u, v in path.edges
                        ],
                        "score": path.score,
                    }
        nxt.sort(key=lambda p: p.score, reverse=True)
        frontier = nxt[:_MAX_FRONTIER]
    return evidence


def traverse(
    question: str,
    artifacts: GraphArtifacts,
    embedder: EmbeddingProvider,
    *,
    max_hops: int | None = None,
    max_items: int = 8,
    router_llm: LLMProvider | None = None,
) -> list[RetrievedChunk]:
    """Guided multi-hop retrieval, fully auditable.

    Decomposes the question into entity anchors (``router_llm`` optional),
    links each anchor to seed entities, then expands best-first up to a
    dynamic depth derived from the number of anchors and capped by
    ``max_hops``. Multiple scored candidate paths compete per node;
    expansion stops early once enough evidence is found. Returns text
    units ranked by path score with ``hop_path`` + relationship
    directions as citation provenance — [] when nothing entity-links.

    Author: venkateshwara doijode
    """
    if max_hops is None:
        from app.config import get_settings

        max_hops = get_settings().traverse_k_hops

    g = build_graph(artifacts)
    anchors = decompose(question, router_llm)
    seeds: list[tuple[str, float]] = []
    seen_seeds: set[str] = set()
    for anchor in anchors:
        for node, score in _link_scored(anchor, g, embedder):
            if node not in seen_seeds:
                seen_seeds.add(node)
                seeds.append((node, score))
    if not seeds:
        return []

    # depth follows the question's structure: more anchors -> more hops
    # (>=2), capped by ``max_hops``. Without usable decomposition the
    # needed depth is unknown, so expand to the cap — early stopping and
    # the frontier bound keep that from becoming uncontrolled expansion.
    if anchors == [question]:
        target_hops = max_hops
    else:
        target_hops = min(max(2, len(anchors)), max_hops)
    evidence = _expand(
        g, seeds, _terms(question), target_hops, max_items
    )
    if not evidence:
        return []

    tu_ids = list(evidence)
    tu = artifacts.text_units
    rows = tu[tu["id"].isin(set(tu_ids))] if not tu.empty else tu
    doc_map = artifacts.doc_id_map

    hits = []
    for _, r in rows.iterrows():
        e = evidence[str(r["id"])]
        hits.append(
            RetrievedChunk(
                chunk_id=str(r["id"]),
                document_id=doc_map.get(
                    str(r.get("document_id", "")), str(r.get("document_id", ""))
                ),
                text=str(r.get("text", "")),
                score=float(e["score"]),
                source="traverse",
                meta={
                    "hop_path": e["path"],
                    "edge_directions": e["edge_directions"],
                    "entities": [n for n, _ in seeds],
                    "anchors": anchors,
                },
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:max_items]
