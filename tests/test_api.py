"""API smoke tests — fake LLM + hash embedder, fully offline."""


def test_health(client):
    """Health endpoint returns 200 + ok status.

    Author: venkateshwara doijode
    """
    response = client.get("/healthz")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_upload_and_chat_and_source(client):
    """End-to-end: upload -> chunked -> chat cites the doc -> source text.

    Author: venkateshwara doijode
    """
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"CS-501 is taught by Dr. Priya Raman in Turing Hall.")},
        params={"title": "notes"},
    )
    assert response.status_code == 201, response.text
    doc = response.json()
    assert doc["status"] == "chunked"

    response = client.post("/chat", json={"question": "who teaches CS-501?"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["route"] in {"hybrid", "traverse", "local", "global"}
    assert data["route_reason"] is not None
    assert data["conversation_id"]
    assert data["citations"] and data["citations"][0]["document_id"] == doc["id"]
    assert data["citations"][0]["score"] is not None

    response = client.get(f"/sources/{doc['id']}")
    assert response.status_code == 200
    assert "Priya Raman" in response.json()["text"]


def test_upload_bad_type(client):
    """Unsupported file type -> doc marked failed, error not raised.

    Author: venkateshwara doijode
    """
    response = client.post(
        "/documents",
        files={"file": ("x.exe", b"MZ binary")},
    )
    assert response.status_code == 201
    assert response.json()["status"] == "failed"  # extraction fails, recorded not raised


def test_chat_invalid_mode(client):
    """Unknown mode override -> 400 from route validation.

    Author: venkateshwara doijode
    """
    response = client.post("/chat", json={"question": "hi", "mode": "bogus"})
    assert response.status_code == 400


def test_missing_source(client):
    """Unknown document id on /sources -> 404.

    Author: venkateshwara doijode
    """
    assert client.get("/sources/nope").status_code == 404


def test_auth_required(client):
    """Protected routes 401 without / with a wrong API key.

    Author: venkateshwara doijode
    """
    response = client.get("/documents", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401
    response = client.post("/chat", json={"question": "hi"}, headers={"X-API-Key": ""})
    assert response.status_code == 401


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
    response = client.post(
        "/documents",
        files={"file": ("big.txt", b"x" * 64)},
    )
    assert response.status_code == 413


def test_review_flow(client):
    """Submit a verdict -> it appears in the review list; the queue
    stays empty while no index is built.

    Author: venkateshwara doijode
    """
    response = client.post(
        "/graph/review",
        json={
            "source": "CS-501",
            "target": "PRIYA RAMAN",
            "description": "taught by",
            "verdict": "rejected",
            "note": "wrong extraction",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["verdict"] == "rejected" and body["rel_hash"]

    response = client.get("/graph/review")
    assert response.status_code == 200
    assert any(
        row["rel_hash"] == body["rel_hash"] for row in response.json()
    )

    # same relationship re-posted updates rather than duplicates
    response = client.post(
        "/graph/review",
        json={
            "source": "CS-501",
            "target": "PRIYA RAMAN",
            "description": "taught by",
            "verdict": "approved",
        },
    )
    assert response.status_code == 201
    assert response.json()["verdict"] == "approved"
    rows = client.get("/graph/review").json()
    assert sum(1 for row in rows if row["rel_hash"] == body["rel_hash"]) == 1


def test_review_queue_and_subgraph_empty_index(client):
    """No index built -> empty review queue; subgraph 404s unknown
    entities rather than erroring.

    Author: venkateshwara doijode
    """
    assert client.get("/graph/review/queue").json() == []
    response = client.get("/graph/subgraph/NOPE")
    assert response.status_code == 404
