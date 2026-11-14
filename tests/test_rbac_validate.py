"""RBAC scoped keys, citation validation, doc-scoped retrieval, metrics.

Author: venkateshwara doijode
"""

from __future__ import annotations

import pytest

from app.config import get_settings

SCOPED = {
    "reader-key": {"groups": ["faculty"], "write": False},
    "writer-key": {"groups": ["faculty"], "write": True},
    "other-key": {"groups": ["library"], "write": True},
}


@pytest.fixture
def scoped_keys(monkeypatch):
    """Enable scoped keys alongside the admin test key."""
    import json

    monkeypatch.setattr(
        get_settings(), "api_keys_json", json.dumps(SCOPED)
    )
    yield


def _upload(client, content=b"secret faculty notes", group="faculty"):
    r = client.post(
        "/documents",
        files={"file": ("doc.txt", content)},
        params={"doc_group": group},
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_scoped_key_filters_documents(client, scoped_keys):
    """A faculty key sees faculty docs only; other groups 404/hidden."""
    doc = _upload(client)

    h = {"X-API-Key": "reader-key"}
    ids = {d["id"] for d in client.get("/documents", headers=h).json()}
    assert doc["id"] in ids
    assert client.get(f"/documents/{doc['id']}", headers=h).status_code == 200
    assert client.get(f"/sources/{doc['id']}", headers=h).status_code == 200

    h2 = {"X-API-Key": "other-key"}
    ids = {d["id"] for d in client.get("/documents", headers=h2).json()}
    assert doc["id"] not in ids
    assert client.get(f"/documents/{doc['id']}", headers=h2).status_code == 404
    assert client.get(f"/sources/{doc['id']}", headers=h2).status_code == 404


def test_read_only_key_cannot_write(client, scoped_keys):
    """write=false keys get 403 on every mutating endpoint."""
    h = {"X-API-Key": "reader-key"}
    r = client.post(
        "/documents", files={"file": ("x.txt", b"hi")}, headers=h
    )
    assert r.status_code == 403
    assert client.post("/ingest", headers=h).status_code == 403
    r = client.post(
        "/graph/review",
        json={"source": "A", "target": "B", "description": "x",
              "verdict": "rejected"},
        headers=h,
    )
    assert r.status_code == 403
    # reads still work
    assert client.get("/documents", headers=h).status_code == 200


def test_unknown_key_still_401(client, scoped_keys):
    """Scoped keys don't open the door to arbitrary keys."""
    r = client.get("/documents", headers={"X-API-Key": "bogus"})
    assert r.status_code == 401


def test_admin_key_unrestricted(client, scoped_keys):
    """The API_KEY admin sees all groups and can write."""
    doc = _upload(client, group="library")
    ids = {d["id"] for d in client.get("/documents").json()}
    assert doc["id"] in ids


def test_chat_response_shape(client):
    """Chat answers carry warning + metrics-capable fields."""
    _upload(client, b"The registrar's office is in Turing Hall.",
            group="default")
    r = client.post(
        "/chat", json={"question": "where is the registrar's office?"}
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "warning" in data
    assert "citations" in data
    for c in data["citations"]:
        assert "supported" in c


def test_validate_parses_verdicts():
    """_validate maps n:yes/no and conflicts to supported + warning."""
    from app.rag.graph import _validate
    from app.retrieval.vectorstore import RetrievedChunk

    class LLM:
        def generate(self, msgs, max_tokens=0):
            return "1: yes\n2: no\nconflicts: 1 vs 2"

    chunks = [
        RetrievedChunk(
            chunk_id="a", document_id="d", text="t1", score=1.0,
            source="hybrid", meta={},
        ),
        RetrievedChunk(
            chunk_id="b", document_id="d", text="t2", score=1.0,
            source="hybrid", meta={},
        ),
    ]
    supported, warning = _validate("answer", chunks, LLM())
    assert supported == {1: True, 2: False}
    assert warning == "sources disagree: [1] vs [2]"


def test_validate_degrades_on_llm_failure():
    """A throwing LLM returns empty results, never an exception."""
    from app.rag.graph import _validate

    class BadLLM:
        def generate(self, msgs, max_tokens=0):
            raise RuntimeError("boom")

    supported, warning = _validate("answer", [], BadLLM())
    assert supported == {} and warning is None


def test_traverse_allowed_doc_ids():
    """Doc-scoped traversal keeps only evidence from allowed docs."""
    import pandas as pd

    from app.ingestion.embed import HashEmbeddingProvider
    from app.retrieval.artifacts import GraphArtifacts
    from app.retrieval.traverse import traverse

    entities = pd.DataFrame(
        [{"id": "e1", "title": "CS-501", "type": "course", "description": ""}]
    )
    rels = pd.DataFrame(
        [{"source": "CS-501", "target": "CS-501", "weight": 1.0,
          "description": "self", "text_unit_ids": ["tu1"]}]
    )
    tus = pd.DataFrame(
        [{"id": "tu1", "text": "cs", "document_id": "secret-doc"}]
    )
    arts = GraphArtifacts(
        entities=entities, relationships=rels, text_units=tus
    )
    emb = HashEmbeddingProvider(dims=64)
    got = traverse(
        "CS-501", arts, emb, max_hops=1, max_items=5,
        allowed_doc_ids={"other-doc"},
    )
    assert all(c.document_id == "other-doc" for c in got)
    assert got == []


def test_graph_index_version_and_restore(tmp_path):
    """_archive_version snapshots output, _restore rolls back."""
    from app.ingestion.graph_index import _archive_version, _restore

    out = tmp_path / "output"
    out.mkdir()
    (out / "entities.parquet").write_text("v1")
    _archive_version(out, tmp_path)
    versions = tmp_path / "output.versions"
    snaps = list(versions.iterdir())
    assert len(snaps) == 1 and (snaps[0] / "entities.parquet").exists()

    prev = tmp_path / "output.prev"
    prev.mkdir()
    (prev / "entities.parquet").write_text("old")
    out.joinpath("entities.parquet").write_text("broken")
    _restore(out, prev)
    assert (out / "entities.parquet").read_text() == "old"
    assert not prev.exists()


def test_langfuse_handler_none_without_package():
    """langfuse_handler degrades to None when unconfigured."""
    from app.observability import langfuse_handler

    assert langfuse_handler() is None
