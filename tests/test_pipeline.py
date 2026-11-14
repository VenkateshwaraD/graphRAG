"""Ingestion failure-path tests — staged file and vector rows must be
cleaned up when the pipeline fails mid-way."""

from app.db.models import DocStatus, Document
from app.ingestion import pipeline


def _doc(session, storage_path):
    doc = Document(
        title="t",
        filename="t.txt",
        mime="text/plain",
        storage_path=str(storage_path),
        status=DocStatus.UPLOADED.value,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)
    return doc


class _BoomEmbedder:
    model_id = "boom"

    def embed_documents(self, texts):
        raise RuntimeError("embed exploded")

    def embed_query(self, text):
        raise RuntimeError("embed exploded")


def test_failed_ingest_removes_staged_file(session, tmp_path, monkeypatch):
    """A leftover staged .txt for a doc that now fails ingest must be
    removed — otherwise the next graph build indexes a failed document.

    Author: venkateshwara doijode
    """
    from app.config import get_settings

    f = tmp_path / "t.txt"
    f.write_text("hello world " * 50)
    doc = _doc(session, f)

    staged = get_settings().graphrag_root / "input" / f"{doc.id}.txt"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("stale staged text from a previous ingest")

    monkeypatch.setattr(pipeline, "get_embedding_provider", lambda: _BoomEmbedder())
    n = pipeline.process_document(session, doc)
    assert n == 0
    session.refresh(doc)
    assert doc.status == DocStatus.FAILED.value
    assert not staged.exists()


def test_failed_ingest_clears_vector_rows(session, tmp_path, monkeypatch):
    """When the vector-store write explodes partway, the doc must not
    keep partial chunk rows in LanceDB.

    Author: venkateshwara doijode
    """
    f = tmp_path / "t.txt"
    f.write_text("hello world " * 50)
    doc = _doc(session, f)

    store = pipeline.get_vector_store()
    orig_add = store.add_chunks

    def flaky_add(rows):
        orig_add(rows[:1])  # partial write, then die
        raise RuntimeError("lancedb exploded mid-add")

    monkeypatch.setattr(store, "add_chunks", flaky_add)
    n = pipeline.process_document(session, doc)
    assert n == 0
    session.refresh(doc)
    assert doc.status == DocStatus.FAILED.value
    df = store.table.to_pandas()
    assert not (df["document_id"] == doc.id).any()


def test_success_ingest_stages_file(session, tmp_path):
    """Happy path: staged input file exists after a successful ingest.

    Author: venkateshwara doijode
    """
    from app.config import get_settings

    f = tmp_path / "t.txt"
    f.write_text("Turing Hall hosts the Data Systems Lab. " * 5)
    doc = _doc(session, f)
    n = pipeline.process_document(session, doc)
    assert n > 0
    session.refresh(doc)
    assert doc.status == DocStatus.CHUNKED.value
    staged = get_settings().graphrag_root / "input" / f"{doc.id}.txt"
    assert staged.exists()
