"""Multi-hop traversal over a fixture knowledge graph — no LLM needed."""

import pandas as pd
import pytest

from app.ingestion.embed import HashEmbeddingProvider
from app.retrieval.artifacts import GraphArtifacts
from app.retrieval.traverse import build_graph, link_entities, traverse


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


def test_build_graph(artifacts):
    """Entities become nodes and relationships become edges.

    Author: venkateshwara doijode
    """
    g = build_graph(artifacts)
    assert g.number_of_nodes() == 4
    assert g.has_edge("CS-501", "PRIYA RAMAN")


def test_link_entities_exact(artifacts):
    """A question naming an entity links to it via substring match.

    Author: venkateshwara doijode
    """
    g = build_graph(artifacts)
    hits = link_entities("Where does CS-501 meet?", g, HashEmbeddingProvider())
    assert "CS-501" in hits


def test_traverse_two_hops(artifacts):
    """K=3 BFS follows the full chain and returns the end text unit.

    Author: venkateshwara doijode
    """
    hits = traverse(
        "Where is the lab of the professor who teaches CS-501?",
        artifacts,
        HashEmbeddingProvider(),
        k_hops=3,
    )
    texts = " ".join(h.text for h in hits)
    assert "Turing Hall" in texts
    docs = {h.document_id for h in hits}
    assert "d3" in docs


def test_traverse_respects_hop_limit(artifacts):
    """K=1 stays within one hop — far-end text units are not returned.

    Author: venkateshwara doijode
    """
    # 1 hop from CS-501 reaches only PRIYA RAMAN's edge text
    hits = traverse("Tell me about CS-501", artifacts, HashEmbeddingProvider(), k_hops=1)
    docs = {h.document_id for h in hits}
    assert "d1" in docs and "d3" not in docs


def test_traverse_empty_graph():
    """Empty artifacts -> no seeds -> [] instead of an error.

    Author: venkateshwara doijode
    """
    arts = GraphArtifacts()
    assert traverse("anything", arts, HashEmbeddingProvider()) == []
