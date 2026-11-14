"""Test configuration — must set env BEFORE any app import."""

from __future__ import annotations

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="ugra_test_")

os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("EMBEDDING_DIMS", "64")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/app.db")
os.environ.setdefault("LANCEDB_URI", f"{_TMP}/lancedb")
os.environ.setdefault("UPLOAD_DIR", f"{_TMP}/uploads")
os.environ.setdefault("GRAPHRAG_ROOT", f"{_TMP}/graphrag")

import pytest  # noqa: E402


@pytest.fixture
def session():
    """Yield a DB session against the temp test database.

    Author: venkateshwara doijode
    """
    from app.db.session import SessionLocal, init_db

    init_db()
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def client():
    """Yield a FastAPI TestClient (runs lifespan: init_db + settings.yaml).

    Author: venkateshwara doijode
    """
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
