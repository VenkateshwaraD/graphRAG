"""Guided multi-hop retriever over the GraphRAG knowledge graph.

Query decomposition -> entity linking per anchor (exact -> alias ->
fuzzy -> semantic) -> best-first expansion over a directed multigraph
keeping several scored candidate paths per node -> early stop once
enough evidence is collected -> explicit score pruning. Deterministic
multi-hop retrieval, fully auditable: every returned text unit carries
the entity path(s) that reached it, extracted relationship directions
plus the direction each edge was traversed, year hints for temporal
filtering, and the alternate paths that competed for it.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from app.ingestion.embed import EmbeddingProvider
from app.rag.llm import LLMProvider
from app.rag.prompts import DECOMPOSE_SYSTEM
from app.retrieval.artifacts import GraphArtifacts
from app.retrieval.review import rel_key
from app.retrieval.vectorstore import RetrievedChunk

_MAX_PATHS_PER_NODE = 3
_MAX_PATHS_PER_EVIDENCE = 2
_MAX_FRONTIER = 64
_DEPTH_DECAY = 0.85
_PRUNE_RATIO = 0.25
_FUTURE_PENALTY = 0.3
_STALE_PENALTY = 0.7
_FUZZY_FLOOR = 0.8
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_RECENCY_RE = re.compile(
    r"\b(?:current(?:ly)?|now|today|latest|newest|present)\b", re.IGNORECASE
)
_HONORIFICS = ("DR.", "DR ", "PROF.", "PROF ", "PROFESSOR ", "MR.", "MS.")


def _terms(text: str) -> frozenset[str]:
    """Content-word tokens of ``text`` for lexical relevance scoring.

    Author: venkateshwara doijode
    """
    return frozenset(
        t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS
    )


_STOPWORDS = frozenset(
    "a an the of in on for to is are was were who what where which when how "
    "does do did that this these those by with from at as and or its it".split()
)


def _years(text: str) -> set[int]:
    """Extract 4-digit years (1900-2099) mentioned in a text.

    Author: venkateshwara doijode
    """
    return {int(y) for y in _YEAR_RE.findall(text)}


def _aliases(title: str) -> set[str]:
    """Name variants an entity answers to: stripped honorifics,
    pre-parenthetical form, and multiword acronym.

    Lets "DR. PRIYA RAMAN" match "Priya Raman" and "DATA SYSTEMS
    LABORATORY" match "DSL" at linking time without an embedding call.

    Author: venkateshwara doijode
    """
    out = {title}
    base = title.split("(")[0].strip()
    out.add(base)
    for p in _HONORIFICS:
        if base.startswith(p):
            out.add(base[len(p):].strip())
    words = base.split()
    if len(words) >= 2:
        out.add("".join(w[0] for w in words))
    return {a for a in out if a}


def build_graph(
    artifacts: GraphArtifacts, rejections: set[str] | None = None
) -> nx.MultiDiGraph:
    """Materialize entities/relationships into a directed multigraph.

    Each extracted relationship is one directed edge (source -> target)
    carrying weight, description, precomputed description ``desc_years``,
    and its text_unit_ids. Traversal crosses edges in either direction
    while recording whether each hop went forward (with the extracted
    direction) or backward (against it). Nodes carry ``aliases`` for
    entity linking. ``rejections`` is a set of human-review ``rel_key``
    hashes — edges a reviewer marked wrong never enter the graph.

    Author: venkateshwara doijode
    """
    g = nx.MultiDiGraph()
    if artifacts.entities.empty:
        return g
    for _, row in artifacts.entities.iterrows():
        name = str(row.get("title", row.get("name", ""))).upper()
        if name:
            g.add_node(
                name,
                description=str(row.get("description", "")),
                aliases=_aliases(name),
            )
    for _, row in artifacts.relationships.iterrows():
        src, tgt = str(row.get("source", "")).upper(), str(row.get("target", "")).upper()
        description = str(row.get("description", ""))
        tu_ids = row.get("text_unit_ids")
        if src and tgt and rel_key(src, tgt, description) not in (rejections or ()):
            g.add_edge(
                src,
                tgt,
                weight=float(row.get("weight", 1.0) or 1.0),
                description=description,
                relation_direction=f"{src}->{tgt}",
                desc_years=_years(description),
                text_unit_ids=list(tu_ids) if tu_ids is not None else [],
            )
    return g


def _edges_from(g: nx.MultiDiGraph, node: str):
    """Yield (neighbor, direction, attrs) for every edge touching
    ``node`` — outgoing edges are "forward", incoming "backward".

    Traversal follows relationships both ways (extraction direction is
    unreliable for question order) but records which way each edge was
    crossed so citations can show it.

    Author: venkateshwara doijode
    """
    for _, nbr, _key, attrs in g.out_edges(node, keys=True, data=True):
        yield nbr, "forward", attrs
    for nbr, _, _key, attrs in g.in_edges(node, keys=True, data=True):
        yield nbr, "backward", attrs


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


def _fuzzy_matches(text: str, g: nx.MultiDiGraph) -> list[tuple[str, float]]:
    """Fuzzy entity linking: slide a same-length n-gram window over the
    text and keep nodes whose name or alias scores a SequenceMatcher
    ratio >= ``_FUZZY_FLOOR`` — catches typos ("TURING HALL" vs "TURING
    HAL") that neither substring nor embeddings reliably fix.

    Author: venkateshwara doijode
    """
    words = text.upper().split()
    scored: list[tuple[str, float]] = []
    for n in g.nodes:
        variants = {n} | set(g.nodes[n].get("aliases", set()))
        width = min(len(v.split()) for v in variants)
        grams = (
            [" ".join(words[i : i + width]) for i in range(len(words) - width + 1)]
            if len(words) >= width
            else [text.upper()]
        )
        best = 0.0
        for v in variants:
            for gram in grams:
                best = max(
                    best, difflib.SequenceMatcher(None, v, gram).ratio()
                )
        if best >= _FUZZY_FLOOR:
            scored.append((n, best))
    return scored


def _link_scored(
    text: str, g: nx.MultiDiGraph, embedder: EmbeddingProvider, top_n: int = 3
) -> list[tuple[str, float]]:
    """Link a phrase to graph entities, returning (node, score) pairs.

    Resolution ladder: exact substring of node title OR alias (score
    1.0) -> fuzzy n-gram match (SequenceMatcher ratio) -> embedding
    similarity over titles with a 0.25 floor. Higher-confidence matches
    short-circuit lower tiers so a typo-tolerant alias never loses to
    a marginal embedding hit.

    Author: venkateshwara doijode
    """
    q_upper = text.upper()
    # node titles match as substring; aliases (often short acronyms)
    # must match on word boundaries so "TH" doesn't hit inside "THE"
    def _alias_hit(n: str) -> bool:
        return any(
            a and re.search(rf"\b{re.escape(a)}\b", q_upper)
            for a in g.nodes[n].get("aliases", set())
        )

    exact = [n for n in g.nodes if n in q_upper or _alias_hit(n)]
    if exact:
        return [(n, 1.0) for n in exact[:top_n]]
    fuzzy = sorted(_fuzzy_matches(text, g), key=lambda kv: -kv[1])
    if fuzzy:
        return fuzzy[:top_n]
    names = list(g.nodes)
    if not names:
        return []
    q_vec = np.array(embedder.embed_query(text), dtype=np.float32)
    name_vecs = np.array(embedder.embed_documents(names), dtype=np.float32)
    sims = name_vecs @ q_vec
    idx = np.argsort(-sims)[:top_n]
    return [(names[i], float(sims[i])) for i in idx if sims[i] > 0.25]


def link_entities(
    question: str, g: nx.MultiDiGraph, embedder: EmbeddingProvider, top_n: int = 5
) -> list[str]:
    """Match question to graph entities — names only, dropping scores.

    Author: venkateshwara doijode
    """
    return [n for n, _ in _link_scored(question, g, embedder, top_n)]


@dataclass
class _Path:
    """One candidate traversal path and its accumulated relevance score.

    ``seed_score`` is the entity-linking confidence of the anchor node;
    ``edge_sum`` accumulates per-edge relevance so ``score`` is the
    seed confidence times the mean edge score, decayed by depth —
    shorter, more relevant chains win. ``dirs`` records forward /
    backward per crossed edge.

    Author: venkateshwara doijode
    """

    node: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    dirs: tuple[str, ...]
    rel_dirs: tuple[str, ...]
    years: frozenset[int]
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


def _expand(
    g: nx.MultiDiGraph,
    seeds: list[tuple[str, float]],
    q_terms: frozenset[str],
    target_hops: int,
    max_items: int,
) -> dict[str, list[dict[str, Any]]]:
    """Best-first expansion collecting text-unit evidence along paths.

    Each hop extends every live path to neighbors not already on it
    (per-path cycle detection); up to ``_MAX_PATHS_PER_NODE`` distinct
    paths may reach the same node so alternative routes compete. Path
    score = seed link score x mean edge score x depth decay. Expansion
    stops early once ``max_items`` evidence units are collected — it
    never grows the frontier just because hops remain.

    Returns text_unit_id -> list of candidate paths (score, entity path,
    edge directions, traversal directions, edge years), best first,
    capped at ``_MAX_PATHS_PER_EVIDENCE`` so alternate reasoning routes
    stay visible in citations.

    Author: venkateshwara doijode
    """
    max_weight = max(
        (float(d.get("weight", 1.0) or 1.0) for _, _, _, d in g.edges(data=True, keys=True)),
        default=1.0,
    )
    frontier: list[_Path] = [
        _Path(
            node=n, nodes=(n,), edges=(), dirs=(), rel_dirs=(),
            years=frozenset(), seed_score=s,
        )
        for n, s in seeds
    ]
    best_per_node: dict[str, list[float]] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}

    for _ in range(target_hops):
        if not frontier or len(evidence) >= max_items:
            break
        nxt: list[_Path] = []
        for rec in frontier:
            for nbr, direction, attrs in _edges_from(g, rec.node):
                path = _Path(
                    node=nbr,
                    nodes=rec.nodes + (nbr,),
                    edges=rec.edges + ((rec.node, nbr),),
                    dirs=rec.dirs + (direction,),
                    rel_dirs=rec.rel_dirs
                    + (str(attrs.get("relation_direction", "")),),
                    years=rec.years | frozenset(attrs.get("desc_years", set())),
                    seed_score=rec.seed_score,
                    edge_sum=rec.edge_sum
                    + _edge_score(q_terms, attrs, max_weight),
                )
                # cross-edge back to an already-visited node: its text
                # units are real evidence for this path, but the path
                # must not extend further through it (cycle)
                is_cycle = nbr in rec.nodes
                if not is_cycle:
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
                    cands = evidence.setdefault(tu, [])
                    if any(c["path"] == list(path.nodes) for c in cands):
                        continue
                    cands.append(
                        {
                            "path": list(path.nodes),
                            "edge_directions": list(path.rel_dirs),
                            "traverse_dirs": list(path.dirs),
                            "years": sorted(path.years),
                            "score": path.score,
                        }
                    )
                    cands.sort(key=lambda c: c["score"], reverse=True)
                    del cands[_MAX_PATHS_PER_EVIDENCE:]
        nxt.sort(key=lambda p: p.score, reverse=True)
        frontier = nxt[:_MAX_FRONTIER]
    return evidence


def _as_of(question: str, as_of: int | None) -> tuple[int | None, bool]:
    """Resolve the temporal scope: explicit param > year in question >
    recency wording (prefer latest-dated evidence) > none.

    Author: venkateshwara doijode
    """
    if as_of is not None:
        return as_of, False
    years = _years(question)
    if years:
        return max(years), False
    if _RECENCY_RE.search(question):
        return None, True
    return None, False


def _temporal_factor(
    years: set[int], as_of: int | None, prefer_latest: bool, latest: int | None
) -> float:
    """Score multiplier for temporal fit of one evidence unit.

    ``as_of`` scope: only future-dated evidence is penalized — a fact
    stated for a later year than asked is suspect; undated or
    earlier-dated units stay neutral. ``prefer_latest`` (recency
    wording): units missing the newest observed year are mildly
    deprioritized. No scope -> 1.0.

    Author: venkateshwara doijode
    """
    if as_of is not None:
        return _FUTURE_PENALTY if years and all(y > as_of for y in years) else 1.0
    if prefer_latest and years and latest is not None:
        return 1.0 if latest in years else _STALE_PENALTY
    return 1.0


def traverse(
    question: str,
    artifacts: GraphArtifacts,
    embedder: EmbeddingProvider,
    *,
    max_hops: int | None = None,
    max_items: int = 8,
    router_llm: LLMProvider | None = None,
    as_of: int | None = None,
    rejections: set[str] | None = None,
    allowed_doc_ids: set[str] | None = None,
) -> list[RetrievedChunk]:
    """Guided multi-hop retrieval, fully auditable.

    Decomposes the question into entity anchors (``router_llm``
    optional), links each anchor to seed entities via exact -> alias ->
    fuzzy -> embedding resolution, then expands best-first up to a
    dynamic depth derived from the number of anchors and capped by
    ``max_hops``. Multiple scored candidate paths compete per node;
    expansion stops early once enough evidence is found. Returns text
    units ranked by adjusted score with ``hop_path``, ``alt_paths``,
    ``edge_directions``, ``traverse_dirs`` and ``years`` as citation
    provenance — [] when nothing entity-links.

    ``as_of`` (or a year / recency wording detected in the question)
    applies temporal filtering. ``rejections`` excludes
    human-review-rejected relationships at graph build.
    ``allowed_doc_ids`` scopes evidence to documents the caller's role
    may see (None = unrestricted). Evidence below ``_PRUNE_RATIO`` of
    the top score is pruned before ranking.

    Author: venkateshwara doijode
    """
    if max_hops is None:
        from app.config import get_settings

        max_hops = get_settings().traverse_k_hops

    g = build_graph(artifacts, rejections=rejections)
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

    # temporal scope: explicit as_of, else a year in the question, else
    # recency wording
    scope, prefer_latest = _as_of(question, as_of)
    tu_ids = list(evidence)
    tu = artifacts.text_units
    rows = tu[tu["id"].isin(set(tu_ids))] if not tu.empty else tu
    doc_map = artifacts.doc_id_map

    entries = []
    for _, r in rows.iterrows():
        doc_id = doc_map.get(
            str(r.get("document_id", "")), str(r.get("document_id", ""))
        )
        if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
            continue
        cands = evidence[str(r["id"])]
        best = cands[0]
        years = _years(str(r.get("text", "")))
        for c in cands:
            years |= set(c.get("years", set()))
        entries.append(
            {
                "row": r,
                "doc_id": doc_id,
                "best": best,
                "alt_paths": [c["path"] for c in cands[1:]],
                "years": sorted(years),
            }
        )
    if not entries:
        return []
    latest = max((y for e in entries for y in e["years"]), default=None)
    for e in entries:
        e["score"] = e["best"]["score"] * _temporal_factor(
            set(e["years"]), scope, prefer_latest, latest
        )

    # explicit prune: drop evidence far below the best adjusted score
    # before context generation — weak tail paths only add noise
    top = max(e["score"] for e in entries)
    kept = [e for e in entries if e["score"] >= _PRUNE_RATIO * top]
    kept.sort(key=lambda e: e["score"], reverse=True)

    hits = []
    for e in kept[:max_items]:
        r = e["row"]
        hits.append(
            RetrievedChunk(
                chunk_id=str(r["id"]),
                document_id=e["doc_id"],
                text=str(r.get("text", "")),
                score=float(e["score"]),
                source="traverse",
                meta={
                    "hop_path": e["best"]["path"],
                    "edge_directions": e["best"]["edge_directions"],
                    "traverse_dirs": e["best"]["traverse_dirs"],
                    "alt_paths": e["alt_paths"],
                    "years": e["years"],
                    "as_of": scope,
                    "entities": [n for n, _ in seeds],
                    "anchors": anchors,
                },
            )
        )
    return hits
