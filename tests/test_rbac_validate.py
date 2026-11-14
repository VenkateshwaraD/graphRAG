"""Authorization acceptance matrix + retrieval/citation unit tests.

Covers the six roles, multi-tenant isolation, grants and revocation,
privilege-escalation guards, scoped rerouting, API-key rotation, and
fail-closed empty scopes — all offline (fake LLM, hash embedder).

Author: venkateshwara doijode
"""

from __future__ import annotations

import pytest  # noqa: F401 — fixtures surface via pytest, not direct refs
from conftest import auth


def _upload(client, content=b"some course material", **params):
    """Upload as the platform-admin test key; returns the doc dict.

    Author: venkateshwara doijode
    """
    response = client.post(
        "/documents",
        files={"file": ("doc.txt", content)},
        params=params,
    )
    assert response.status_code == 201, response.text
    return response.json()


# --- document access matrix --------------------------------------------------


def test_student_accesses_own_course_document(client, world):
    """Enrolled student reads a course-scoped document — ALLOW."""
    doc = _upload(
        client,
        tenant_id=world["tenant_a"],
        course_id=world["course_501"],
        classification="public",
    )
    response = client.get(f"/documents/{doc['id']}", headers=auth(world, "student"))
    assert response.status_code == 200
    ids = {
        listed["id"]
        for listed in client.get("/documents", headers=auth(world, "student")).json()
    }
    assert doc["id"] in ids


def test_student_denied_other_course_document(client, world):
    """A student not enrolled in the document's course — DENY."""
    doc = _upload(
        client,
        tenant_id=world["tenant_a"],
        course_id=world["course_501"],
        classification="public",
    )
    response = client.get(f"/documents/{doc['id']}", headers=auth(world, "outsider"))
    assert response.status_code == 404
    assert client.get(
        f"/sources/{doc['id']}", headers=auth(world, "outsider")
    ).status_code == 404


def test_faculty_denied_other_department(client, world):
    """A department-scoped doc is invisible outside that department."""
    doc = _upload(
        client,
        tenant_id=world["tenant_a"],
        department_id=world["dept_cs"],
        classification="internal",
    )
    assert (
        client.get(f"/documents/{doc['id']}", headers=auth(world, "faculty"))
        .status_code
        == 200
    )
    # outsider is a student of the law department — cross-department deny
    assert (
        client.get(f"/documents/{doc['id']}", headers=auth(world, "outsider"))
        .status_code
        == 404
    )


def test_cross_tenant_access_denied(client, world):
    """Tenant B's admin cannot see tenant A's documents — DENY."""
    doc = _upload(
        client, tenant_id=world["tenant_a"], classification="public"
    )
    response = client.get(f"/documents/{doc['id']}", headers=auth(world, "admin_b"))
    assert response.status_code == 404
    assert client.get(
        f"/sources/{doc['id']}", headers=auth(world, "admin_b")
    ).status_code == 404
    ids = {
        listed["id"]
        for listed in client.get("/documents", headers=auth(world, "admin_b")).json()
    }
    assert doc["id"] not in ids


def test_restricted_grant_lifecycle(client, world):
    """restricted: deny without grant -> allow with grant -> deny after
    revoke (immediately, no caching)."""
    doc = _upload(
        client,
        tenant_id=world["tenant_a"],
        classification="restricted",
    )
    headers = auth(world, "student")
    assert client.get(f"/documents/{doc['id']}", headers=headers).status_code == 404

    grant = client.post(
        f"/documents/{doc['id']}/grants",
        json={"grantee_type": "user", "grantee_id": world["users"]["student"]},
    )
    assert grant.status_code == 201, grant.text
    assert client.get(f"/documents/{doc['id']}", headers=headers).status_code == 200
    assert client.get(f"/sources/{doc['id']}", headers=headers).status_code == 200

    response = client.delete(f"/documents/{doc['id']}/grants/{grant.json()['id']}")
    assert response.status_code == 204
    assert client.get(f"/documents/{doc['id']}", headers=headers).status_code == 404


def test_grant_cannot_cross_tenant(client, world):
    """A grant naming a user from another tenant is rejected at write."""
    doc = _upload(
        client, tenant_id=world["tenant_a"], classification="restricted"
    )
    response = client.post(
        f"/documents/{doc['id']}/grants",
        json={
            "grantee_type": "user",
            "grantee_id": world["users"]["admin_b"],
        },
    )
    assert response.status_code == 400


def test_empty_scope_fails_closed(client, world):
    """A doc nobody-but-owner may see produces an empty scope for
    others — listing, detail, and retrieval all deny."""
    doc = _upload(
        client, tenant_id=world["tenant_a"], classification="restricted"
    )
    headers = auth(world, "guest")
    ids = {
        listed["id"]
        for listed in client.get("/documents", headers=headers).json()
    }
    assert doc["id"] not in ids
    # guests also can't reach internal docs at all
    assert client.get(f"/documents/{doc['id']}", headers=headers).status_code == 404
    response = client.post(
        "/chat", json={"question": "tell me everything"}, headers=headers
    )
    assert response.status_code == 200
    assert all(
        citation.get("document_id") != doc["id"]
        for citation in response.json()["citations"]
    )


# --- privilege escalation ----------------------------------------------------


def test_student_cannot_self_promote(client, world):
    """PATCH /users/{self} with a role change is rejected — and students
    can't reach the admin endpoint at all."""
    uid = world["users"]["student"]
    response = client.patch(
        f"/users/{uid}",
        json={"role": "faculty"},
        headers=auth(world, "student"),
    )
    assert response.status_code == 403


def test_admin_cannot_self_change_role(client, world):
    """Even an admin cannot change their own role/status."""
    uid = world["users"]["admin_a"]
    response = client.patch(
        f"/users/{uid}",
        json={"role": "platform_admin"},
        headers=auth(world, "admin_a"),
    )
    assert response.status_code == 403


def test_uni_admin_role_ceiling(client, world):
    """A university admin cannot create or assign platform_admin, and
    cannot act outside its tenant."""
    response = client.post(
        "/users",
        json={"email": "x@a.edu", "role": "platform_admin"},
        headers=auth(world, "admin_a"),
    )
    assert response.status_code == 403
    # role demotion of a platform admin by a tenant admin is also out
    response = client.post(
        "/users",
        json={
            "email": "sneaky@a.edu",
            "role": "student",
            "tenant_id": world["tenant_b"],
        },
        headers=auth(world, "admin_a"),
    )
    assert response.status_code in (201, 409)  # tenant_id is forced to own tenant
    if response.status_code == 201:
        assert response.json()["tenant_id"] == world["tenant_a"]


def test_student_cannot_enroll_self(client, world):
    """Enrollment writes require hod-or-above — students get 403."""
    response = client.post(
        "/enrollments",
        json={
            "user_id": world["users"]["student"],
            "course_id": world["course_501"],
            "kind": "student",
        },
        headers=auth(world, "student"),
    )
    assert response.status_code == 403


# --- API key lifecycle --------------------------------------------------------


def test_key_rotation_revokes_old_key(client, world):
    """Rotating a key invalidates the old one immediately."""
    headers = auth(world, "guest")
    assert client.get("/documents", headers=headers).status_code == 200
    response = client.post(
        f"/users/{world['users']['guest']}/rotate-key", headers=headers
    )
    assert response.status_code == 200, response.text
    new_key = response.json()["api_key"]
    assert new_key and new_key != world["keys"]["guest"]
    assert client.get("/documents", headers=headers).status_code == 401
    assert (
        client.get("/documents", headers={"X-API-Key": new_key}).status_code
        == 200
    )
    world["keys"]["guest"] = new_key  # keep the fixture consistent


def test_deactivated_user_key_fails(client, world):
    """is_active=false immediately revokes the user's key."""
    uid = world["users"]["faculty"]
    headers = auth(world, "faculty")
    assert client.get("/documents", headers=headers).status_code == 200
    response = client.patch(
        f"/users/{uid}",
        json={"is_active": False},
        headers=auth(world, "admin_a"),
    )
    assert response.status_code == 200
    assert client.get("/documents", headers=headers).status_code == 401
    client.patch(  # restore for other tests
        f"/users/{uid}",
        json={"is_active": True},
        headers=auth(world, "admin_a"),
    )


def test_wrong_and_missing_keys_still_401(client, world):
    """Unknown keys and missing keys stay 401 with user keys in play."""
    assert client.get(
        "/documents", headers={"X-API-Key": "bogus"}
    ).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": ""}).status_code == 401


# --- scoped rerouting / chat ---------------------------------------------------


def test_scoped_caller_rerouted_off_local(client, world):
    """mode=local for a scoped user reroutes and reports the
    substitution; the answer must not use the unrestricted composer."""
    _upload(
        client,
        b"Dr. Priya Raman teaches CS-501 in Turing Hall.",
        tenant_id=world["tenant_a"],
        classification="public",
    )
    response = client.post(
        "/chat",
        json={"question": "who teaches CS-501?", "mode": "local"},
        headers=auth(world, "student"),
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["route"] != "local"
    assert data["requested_route"] == "local"


def test_chat_excludes_out_of_scope_context(client, world):
    """Hybrid retrieval only sees in-scope docs: a tenant-A doc never
    reaches tenant-B's context or citations."""
    doc = _upload(
        client,
        b"Turing Hall hosts the Data Systems Lab.",
        tenant_id=world["tenant_a"],
        classification="public",
    )
    response = client.post(
        "/chat",
        json={"question": "what is in Turing Hall?"},
        headers=auth(world, "admin_b"),
    )
    assert response.status_code == 200
    assert all(
        citation.get("document_id") != doc["id"]
        for citation in response.json()["citations"]
    )


def test_chat_response_shape(client):
    """Chat answers carry warning + metrics-capable fields."""
    _upload(client, b"The registrar's office is in Turing Hall.")
    response = client.post(
        "/chat", json={"question": "where is the registrar's office?"}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "warning" in data and "requested_route" in data
    for citation in data["citations"]:
        assert "supported" in citation


# --- graph artifact scoping (unit) ---------------------------------------------


def test_scope_artifacts_strict_provenance():
    """Entities/relationships/communities survive only when their full
    provenance is authorized — mixed-evidence aggregates are dropped."""
    import pandas as pd

    from app.retrieval.artifacts import GraphArtifacts, scope_artifacts

    docs = pd.DataFrame(
        [
            {"id": "gA", "title": "doc-a.txt"},
            {"id": "gB", "title": "doc-b.txt"},
        ]
    )
    tus = pd.DataFrame(
        [
            {"id": "tu1", "document_id": "gA", "text": "a"},
            {"id": "tu2", "document_id": "gB", "text": "b"},
        ]
    )
    ents = pd.DataFrame(
        [
            {"id": "e1", "title": "A", "text_unit_ids": ["tu1"]},
            {"id": "e2", "title": "MIX", "text_unit_ids": ["tu1", "tu2"]},
        ]
    )
    rels = pd.DataFrame(
        [
            {
                "id": "r1",
                "source": "A",
                "target": "A2",
                "text_unit_ids": ["tu1"],
                "description": "x",
                "weight": 1.0,
            },
            {
                "id": "r2",
                "source": "B",
                "target": "C",
                "text_unit_ids": ["tu1", "tu2"],
                "description": "y",
                "weight": 1.0,
            },
        ]
    )
    comms = pd.DataFrame(
        [
            {
                "id": "c0",
                "community": "0",
                "level": "0",
                "entity_ids": ["e1", "e2"],
                "relationship_ids": ["r1"],
                "text_unit_ids": ["tu1", "tu2"],
            },
            {
                "id": "c1",
                "community": "1",
                "level": "0",
                "entity_ids": ["e1"],
                "relationship_ids": ["r1"],
                "text_unit_ids": ["tu1"],
            },
        ]
    )
    reports = pd.DataFrame(
        [
            {"id": "rep0", "community": "0", "level": "0"},
            {"id": "rep1", "community": "1", "level": "0"},
        ]
    )
    arts = GraphArtifacts(
        entities=ents,
        relationships=rels,
        text_units=tus,
        communities=comms,
        community_reports=reports,
        documents=docs,
    )
    scoped = scope_artifacts(arts, {"doc-a"})
    assert set(scoped.entities["id"]) == {"e1"}  # mixed-provenance e2 dropped
    assert set(scoped.relationships["id"]) == {"r1"}
    # community 0 contains unauthorized e2 -> dropped with its report
    assert set(scoped.communities["community"].astype(str)) == {"1"}
    assert set(scoped.community_reports["community"].astype(str)) == {"1"}
    assert set(scoped.text_units["id"]) == {"tu1"}

    empty = scope_artifacts(arts, set())
    assert empty.entities.empty and empty.text_units.empty


def test_stale_doc_ids_detects_deleted():
    """Docs present in the index but gone from the DB are stale."""
    import pandas as pd

    from app.retrieval.artifacts import GraphArtifacts, stale_doc_ids

    docs = pd.DataFrame(
        [
            {"id": "gA", "title": "doc-a.txt"},
            {"id": "gB", "title": "doc-b.txt"},
        ]
    )
    arts = GraphArtifacts(documents=docs)
    assert stale_doc_ids(arts, {"doc-a"}) == {"doc-b"}
    assert stale_doc_ids(arts, {"doc-a", "doc-b"}) == set()


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
