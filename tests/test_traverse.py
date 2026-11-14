"""Multi-hop traversal over a fixture knowledge graph — no LLM needed."""

import pandas as pd
import pytest

from app.ingestion.embed import HashEmbeddingProvider
from app.retrieval.artifacts import GraphArtifacts
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
    assert g.edges["CS-501", "PRIYA RAMAN"]["relation_direction"] == "CS-501->PRIYA RAMAN"


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
    """Triangle graph: paths never revisit a node they already contain.

    A-B-C-A would loop forever without the per-path cycle check; the
    C->A edge can still contribute evidence via the A-B-C path.

    Author: venkateshwara doijode
    """
    hits = traverse("a", cycle_artifacts, HashEmbeddingProvider(), max_hops=4)
    for h in hits:
        path = h.meta["hop_path"]
        assert len(path) == len(set(path)), path


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
