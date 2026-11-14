"""API smoke tests — fake LLM + hash embedder, fully offline."""


def test_health(client):
    """Health endpoint returns 200 + ok status.

    Author: venkateshwara doijode
    """
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_upload_and_chat_and_source(client):
    """End-to-end: upload -> chunked -> chat cites the doc -> source text.

    Author: venkateshwara doijode
    """
    r = client.post(
        "/documents",
        files={"file": ("notes.txt", b"CS-501 is taught by Dr. Priya Raman in Turing Hall.")},
        params={"title": "notes"},
    )
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["status"] == "chunked"

    r = client.post("/chat", json={"question": "who teaches CS-501?"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["route"] in {"hybrid", "traverse", "local", "global"}
    assert data["route_reason"] is not None
    assert data["conversation_id"]
    assert data["citations"] and data["citations"][0]["document_id"] == doc["id"]
    assert data["citations"][0]["score"] is not None

    r = client.get(f"/sources/{doc['id']}")
    assert r.status_code == 200
    assert "Priya Raman" in r.json()["text"]


def test_upload_bad_type(client):
    """Unsupported file type -> doc marked failed, error not raised.

    Author: venkateshwara doijode
    """
    r = client.post(
        "/documents",
        files={"file": ("x.exe", b"MZ binary")},
    )
    assert r.status_code == 201
    assert r.json()["status"] == "failed"  # extraction fails, recorded not raised


def test_chat_invalid_mode(client):
    """Unknown mode override -> 400 from route validation.

    Author: venkateshwara doijode
    """
    r = client.post("/chat", json={"question": "hi", "mode": "bogus"})
    assert r.status_code == 400


def test_missing_source(client):
    """Unknown document id on /sources -> 404.

    Author: venkateshwara doijode
    """
    assert client.get("/sources/nope").status_code == 404


def test_auth_required(client):
    """Protected routes 401 without / with a wrong API key.

    Author: venkateshwara doijode
    """
    r = client.get("/documents", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    r = client.post("/chat", json={"question": "hi"}, headers={"X-API-Key": ""})
    assert r.status_code == 401


def test_health_open_without_key(client):
    """Probes stay unauthenticated so monitors don't need the key.

    Author: venkateshwara doijode
    """
    del client.headers["X-API-Key"]
    try:
        assert client.get("/healthz").status_code == 200
        assert client.get("/documents").status_code == 401
    finally:
        client.headers["X-API-Key"] = "test-api-key"


def test_upload_size_limit(client, monkeypatch):
    """Uploads beyond MAX_UPLOAD_BYTES -> 413.

    Author: venkateshwara doijode
    """
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_bytes", 16)
    r = client.post(
        "/documents",
        files={"file": ("big.txt", b"x" * 64)},
    )
    assert r.status_code == 413


def test_review_flow(client):
    """Submit a verdict -> it appears in the review list; the queue
    stays empty while no index is built.

    Author: venkateshwara doijode
    """
    r = client.post(
        "/graph/review",
        json={
            "source": "CS-501",
            "target": "PRIYA RAMAN",
            "description": "taught by",
            "verdict": "rejected",
            "note": "wrong extraction",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["verdict"] == "rejected" and body["rel_hash"]

    r = client.get("/graph/review")
    assert r.status_code == 200
    assert any(x["rel_hash"] == body["rel_hash"] for x in r.json())

    # same relationship re-posted updates rather than duplicates
    r = client.post(
        "/graph/review",
        json={
            "source": "CS-501",
            "target": "PRIYA RAMAN",
            "description": "taught by",
            "verdict": "approved",
        },
    )
    assert r.status_code == 201
    assert r.json()["verdict"] == "approved"
    rows = client.get("/graph/review").json()
    assert sum(1 for x in rows if x["rel_hash"] == body["rel_hash"]) == 1


def test_review_queue_and_subgraph_empty_index(client):
    """No index built -> empty review queue; subgraph 404s unknown
    entities rather than erroring.

    Author: venkateshwara doijode
    """
    assert client.get("/graph/review/queue").json() == []
    r = client.get("/graph/subgraph/NOPE")
    assert r.status_code == 404
