"""Multi-hop traversal over a fixture knowledge graph — no LLM needed."""

import pandas as pd
import pytest

from app.ingestion.embed import HashEmbeddingProvider
from app.retrieval.artifacts import GraphArtifacts
from app.retrieval.review import rel_key
from app.retrieval.traverse import build_graph, decompose, link_entities, traverse


@pytest.fixture
def artifacts():
    """Build a 3-edge fixture graph mirroring real artifact shapes.

    CS-501 -(taught_by)-> PRIYA RAMAN -(directs)-> DATA SYSTEMS LAB
    -(located_in)-> TURING HALL, each edge carrying one text unit.

    Author: venkateshwara doijode
    """
    entities = pd.DataFrame(
        [
            {"id": "e1", "title": "CS-501", "type": "course", "description": "adv databases"},
            {"id": "e2", "title": "PRIYA RAMAN", "type": "person", "description": "professor"},
            {"id": "e3", "title": "DATA SYSTEMS LAB", "type": "organization", "description": "lab"},
            {"id": "e4", "title": "TURING HALL", "type": "location", "description": "building"},
        ]
    )
    relationships = pd.DataFrame(
        [
            {"source": "CS-501", "target": "PRIYA RAMAN", "weight": 1.0,
             "description": "taught by", "text_unit_ids": ["tu1"]},
            {"source": "PRIYA RAMAN", "target": "DATA SYSTEMS LAB", "weight": 1.0,
             "description": "directs", "text_unit_ids": ["tu2"]},
            {"source": "DATA SYSTEMS LAB", "target": "TURING HALL", "weight": 1.0,
             "description": "located in", "text_unit_ids": ["tu3"]},
        ]
    )
    text_units = pd.DataFrame(
        [
            {"id": "tu1", "text": "CS-501 is taught by Dr. Priya Raman.", "document_id": "d1"},
            {"id": "tu2", "text": "Dr. Raman directs the Data Systems Lab.", "document_id": "d2"},
            {"id": "tu3", "text": "The Data Systems Lab is in Turing Hall.", "document_id": "d3"},
        ]
    )
    return GraphArtifacts(entities=entities, relationships=relationships, text_units=text_units)


@pytest.fixture
def cycle_artifacts():
    """A triangle graph: A-B-C-A, used to prove paths can't revisit nodes.

    Author: venkateshwara doijode
    """
    entities = pd.DataFrame(
        [{"id": "e", "title": t, "type": "x", "description": ""} for t in "ABC"]
    )
    relationships = pd.DataFrame(
        [
            {"source": "A", "target": "B", "weight": 1.0,
             "description": "ab", "text_unit_ids": ["tu_ab"]},
            {"source": "B", "target": "C", "weight": 1.0,
             "description": "bc", "text_unit_ids": ["tu_bc"]},
            {"source": "C", "target": "A", "weight": 1.0,
             "description": "ca", "text_unit_ids": ["tu_ca"]},
        ]
    )
    text_units = pd.DataFrame(
        [
            {"id": "tu_ab", "text": "a-b", "document_id": "d"},
            {"id": "tu_bc", "text": "b-c", "document_id": "d"},
            {"id": "tu_ca", "text": "c-a", "document_id": "d"},
        ]
    )
    return GraphArtifacts(entities=entities, relationships=relationships, text_units=text_units)


def test_build_graph(artifacts):
    """Entities become nodes, relationships become edges, direction kept.

    Author: venkateshwara doijode
    """
    g = build_graph(artifacts)
    assert g.number_of_nodes() == 4
    assert g.has_edge("CS-501", "PRIYA RAMAN")
    attrs = next(iter(g.get_edge_data("CS-501", "PRIYA RAMAN").values()))
    assert attrs["relation_direction"] == "CS-501->PRIYA RAMAN"


def test_link_entities_exact(artifacts):
    """A question naming an entity links to it via substring match.

    Author: venkateshwara doijode
    """
    g = build_graph(artifacts)
    hits = link_entities("Where does CS-501 meet?", g, HashEmbeddingProvider())
    assert "CS-501" in hits


def test_decompose_no_llm():
    """Without an LLM, decomposition falls back to the whole question.

    Author: venkateshwara doijode
    """
    assert decompose("where is turing hall?", None) == ["where is turing hall?"]


def test_traverse_two_hops(artifacts):
    """Deep enough traversal follows the full chain and returns the
    end text unit.

    Author: venkateshwara doijode
    """
    hits = traverse(
        "Where is the lab of the professor who teaches CS-501?",
        artifacts,
        HashEmbeddingProvider(),
        max_hops=3,
    )
    texts = " ".join(h.text for h in hits)
    assert "Turing Hall" in texts
    docs = {h.document_id for h in hits}
    assert "d3" in docs


def test_traverse_respects_hop_limit(artifacts):
    """max_hops=1 stays within one hop — far-end units are not returned.

    Author: venkateshwara doijode
    """
    hits = traverse("Tell me about CS-501", artifacts, HashEmbeddingProvider(), max_hops=1)
    docs = {h.document_id for h in hits}
    assert "d1" in docs and "d3" not in docs


def test_traverse_empty_graph():
    """Empty artifacts -> no seeds -> [] instead of an error.

    Author: venkateshwara doijode
    """
    arts = GraphArtifacts()
    assert traverse("anything", arts, HashEmbeddingProvider()) == []


def test_traverse_no_cycles(cycle_artifacts):
    """Triangle graph: paths can't loop — a repeated node only appears
    as the closing node of a cross-edge.

    A-B-C-A would loop forever without the per-path cycle check; the
    C->A edge still contributes its evidence as a cyclic path tail.

    Author: venkateshwara doijode
    """
    hits = traverse("a", cycle_artifacts, HashEmbeddingProvider(), max_hops=4)
    for h in hits:
        path = h.meta["hop_path"]
        assert len(path[:-1]) == len(set(path[:-1])), path
    # the back-edge's text unit is evidence too
    assert any(h.chunk_id == "tu_ca" for h in hits)


def test_traverse_scores_and_provenance(artifacts):
    """Hits carry a real path score plus hop_path and edge directions.

    Author: venkateshwara doijode
    """
    hits = traverse(
        "CS-501 professor lab hall",
        artifacts,
        HashEmbeddingProvider(),
        max_hops=3,
    )
    assert hits
    scores = [h.score for h in hits]
    assert all(0.0 < s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    far = next(h for h in hits if h.document_id == "d3")
    assert far.meta["hop_path"][:1] == ["CS-501"] or "CS-501" in far.meta["hop_path"]
    assert far.meta["edge_directions"]


def test_traverse_early_stop(artifacts):
    """max_items small -> expansion stops once enough evidence exists,
    so a deep graph doesn't explode the result set.

    Author: venkateshwara doijode
    """
    hits = traverse("CS-501", artifacts, HashEmbeddingProvider(), max_hops=3, max_items=1)
    assert len(hits) <= 1


def _df(rows):
    return pd.DataFrame(rows)


@pytest.fixture
def temporal_artifacts():
    """ACME has two symmetric HQ edges; their text units carry years.

    ACME -[hq]-> OLD HQ (tu_old: Boston, 1999)
    ACME -[hq]-> NEW HQ (tu_new: Austin, 2024)

    Author: venkateshwara doijode
    """
    entities = _df(
        [
            {"id": "e1", "title": "ACME", "type": "org", "description": ""},
            {"id": "e2", "title": "OLD HQ", "type": "loc", "description": ""},
            {"id": "e3", "title": "NEW HQ", "type": "loc", "description": ""},
        ]
    )
    relationships = _df(
        [
            {"source": "ACME", "target": "OLD HQ", "weight": 1.0,
             "description": "headquartered at", "text_unit_ids": ["tu_old"]},
            {"source": "ACME", "target": "NEW HQ", "weight": 1.0,
             "description": "headquartered at", "text_unit_ids": ["tu_new"]},
        ]
    )
    text_units = _df(
        [
            {"id": "tu_old", "text": "Acme was headquartered in Boston in 1999.",
             "document_id": "d"},
            {"id": "tu_new", "text": "In 2024 Acme moved its headquarters to Austin.",
             "document_id": "d"},
        ]
    )
    return GraphArtifacts(entities=entities, relationships=relationships, text_units=text_units)


def test_traverse_as_of_penalizes_future(temporal_artifacts):
    """as_of=2020: the 2024-dated unit is scored below the 1999 one —
    same symmetric edges, so the gap is purely temporal.

    Author: venkateshwara doijode
    """
    hits = traverse(
        "where is ACME hq?", temporal_artifacts, HashEmbeddingProvider(),
        max_hops=2, as_of=2020,
    )
    by_id = {h.chunk_id: h for h in hits}
    assert by_id["tu_old"].score > by_id["tu_new"].score
    assert by_id["tu_new"].meta["years"] == [2024]


def test_traverse_recency_prefers_latest(temporal_artifacts):
    """'currently' -> prefer the latest-dated evidence; the stale unit
    is deprioritized rather than dropped.

    Author: venkateshwara doijode
    """
    hits = traverse(
        "where is ACME hq currently?", temporal_artifacts,
        HashEmbeddingProvider(), max_hops=2,
    )
    by_id = {h.chunk_id: h for h in hits}
    assert by_id["tu_new"].score > by_id["tu_old"].score


def test_traverse_rejected_edge_excluded(artifacts):
    """Human-rejected relationships never enter the traversal graph.

    Author: venkateshwara doijode
    """
    key = rel_key("CS-501", "PRIYA RAMAN", "taught by")
    g = build_graph(artifacts, rejections={key})
    assert not g.has_edge("CS-501", "PRIYA RAMAN")
    hits = traverse(
        "Where is the lab of the professor who teaches CS-501?",
        artifacts, HashEmbeddingProvider(), max_hops=3, rejections={key},
    )
    assert hits == []


@pytest.fixture
def diamond_artifacts():
    """Two disjoint routes to the same evidence edge.

    ALPHA -[ab]-> BETA -[bx]-> XTRA -[xy]-> YPSI
    ALPHA -[ac]-> GAMMA -[cx]-> XTRA

    Author: venkateshwara doijode
    """
    entities = _df(
        [
            {"id": f"e{i}", "title": t, "type": "x", "description": ""}
            for i, t in enumerate(["ALPHA", "BETA", "GAMMA", "XTRA", "YPSI"], 1)
        ]
    )
    relationships = _df(
        [
            {"source": "ALPHA", "target": "BETA", "weight": 1.0,
             "description": "ab", "text_unit_ids": []},
            {"source": "ALPHA", "target": "GAMMA", "weight": 1.0,
             "description": "ac", "text_unit_ids": []},
            {"source": "BETA", "target": "XTRA", "weight": 1.0,
             "description": "bx", "text_unit_ids": []},
            {"source": "GAMMA", "target": "XTRA", "weight": 1.0,
             "description": "cx", "text_unit_ids": []},
            {"source": "XTRA", "target": "YPSI", "weight": 1.0,
             "description": "xy", "text_unit_ids": ["tu_xy"]},
        ]
    )
    text_units = _df(
        [{"id": "tu_xy", "text": "XTRA connects to YPSI.", "document_id": "d"}]
    )
    return GraphArtifacts(entities=entities, relationships=relationships, text_units=text_units)


def test_traverse_alternate_paths(diamond_artifacts):
    """The diamond gives the shared edge two candidate paths; the loser
    surfaces as alt_paths instead of being silently discarded.

    Author: venkateshwara doijode
    """
    hits = traverse("ALPHA", diamond_artifacts, HashEmbeddingProvider(), max_hops=3)
    hit = next(h for h in hits if h.chunk_id == "tu_xy")
    assert hit.meta["hop_path"]
    assert hit.meta["alt_paths"], "expected an alternate path"
    for p in [hit.meta["hop_path"], *hit.meta["alt_paths"]]:
        assert p[0] == "ALPHA" and p[-1] == "YPSI"


@pytest.fixture
def prune_artifacts():
    """ALPHA has one strong edge (tuA) and one junk edge (tuB, w=0.05,
    irrelevant description) — the weak tail must be pruned.

    Author: venkateshwara doijode
    """
    entities = _df(
        [
            {"id": "e1", "title": "ALPHA", "type": "x", "description": ""},
            {"id": "e2", "title": "BETA", "type": "x", "description": ""},
            {"id": "e3", "title": "ZETA", "type": "x", "description": ""},
        ]
    )
    relationships = _df(
        [
            {"source": "ALPHA", "target": "BETA", "weight": 1.0,
             "description": "professor teaches course", "text_unit_ids": ["tuA"]},
            {"source": "ALPHA", "target": "ZETA", "weight": 0.05,
             "description": "miscellaneous link", "text_unit_ids": ["tuB"]},
        ]
    )
    text_units = _df(
        [
            {"id": "tuA", "text": "alpha beta good", "document_id": "d"},
            {"id": "tuB", "text": "alpha zeta weak", "document_id": "d"},
        ]
    )
    return GraphArtifacts(entities=entities, relationships=relationships, text_units=text_units)


def test_traverse_prunes_weak_evidence(prune_artifacts):
    """Evidence far below the top score is cut before ranking — weak
    tail branches only add noise to context generation.

    Author: venkateshwara doijode
    """
    hits = traverse(
        "alpha professor", prune_artifacts, HashEmbeddingProvider(), max_hops=1
    )
    ids = {h.chunk_id for h in hits}
    assert "tuA" in ids and "tuB" not in ids
