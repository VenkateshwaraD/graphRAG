"""Document pipeline: upload file -> extract -> chunk -> embed -> stores.

Writes chunks to both SQLite (metadata) and LanceDB (vectors + FTS),
and stages the extracted text into the GraphRAG input dir so a later
index job can build the knowledge graph.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import DocStatus, Document, DocumentChunk
from app.ingestion.chunk import chunk_text
from app.ingestion.embed import get_embedding_provider
from app.ingestion.extract import extract_text
from app.retrieval.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


def process_document(session: Session, document: Document) -> int:
    """Extract, chunk, embed and persist one uploaded document.

    Steps: extract text (fail when nothing extractable) -> chunk -> embed
    with the shared provider -> replace any previous chunks in both
    stores -> stage a ``{doc_id}.txt`` into the GraphRAG input dir ->
    mark the document ``chunked``. Embeddings are computed before any
    store is mutated so failures up to that point are side-effect free;
    on later failure the rollback also clears partial LanceDB rows and
    the staged input file.

    Author: venkateshwara doijode
    """
    settings = get_settings()
    path = Path(document.storage_path)
    staged = settings.graphrag_root / "input" / f"{document.id}.txt"
    store = get_vector_store()
    try:
        text = extract_text(path)
        if not text.strip():
            raise ValueError("no extractable text")

        # Compute everything before mutating any store — a failure above
        # this line leaves the previous index state fully intact.
        pieces = chunk_text(text)
        embedder = get_embedding_provider()
        vectors = embedder.embed_documents(pieces)

        session.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.id
        ).delete()
        store.delete_document(document.id)

        rows = []
        for i, (piece, vec) in enumerate(zip(pieces, vectors, strict=False)):
            chunk = DocumentChunk(
                document_id=document.id,
                idx=i,
                text=piece,
                embedding_model=embedder.model_id,
            )
            session.add(chunk)
            session.flush()
            rows.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "text": piece,
                    "vector": vec,
                    "embedding_model": embedder.model_id,
                }
            )
        store.add_chunks(rows)
        store.ensure_fts_index()

        # Stage for GraphRAG indexing only after the chunk stores
        # committed, so a failed doc never reaches the graph input dir.
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(text, encoding="utf-8")

        document.status = DocStatus.CHUNKED.value
        document.error = None
        session.commit()
        return len(rows)
    except Exception as exc:
        logger.exception("ingestion failed for document %s", document.id)
        session.rollback()
        # Undo non-transactional side effects: partial LanceDB writes and
        # any staged GraphRAG input file.
        store.delete_document(document.id)
        staged.unlink(missing_ok=True)
        document.status = DocStatus.FAILED.value
        document.error = str(exc)[:1000]
        session.commit()
        return 0


def save_upload(filename: str, data: bytes) -> Path:
    """Persist uploaded bytes under ``UPLOAD_DIR`` and return the path.

    The filename is sanitized to its basename (no path traversal), and
    collisions get a ``_1``, ``_2`` suffix rather than overwriting.

    Author: venkateshwara doijode
    """
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    dest = settings.upload_dir / safe_name
    n = 1
    while dest.exists():
        dest = settings.upload_dir / f"{dest.stem}_{n}{dest.suffix}"
        n += 1
    dest.write_bytes(data)
    return dest


def delete_document(session: Session, document: Document) -> None:
    """Remove a document from every store it touched.

    Deletes its LanceDB chunk rows, the staged GraphRAG input file, the
    stored upload, and the ORM row (chunks cascade).

    Author: venkateshwara doijode
    """
    get_vector_store().delete_document(document.id)
    staged = get_settings().graphrag_root / "input" / f"{document.id}.txt"
    staged.unlink(missing_ok=True)
    Path(document.storage_path).unlink(missing_ok=True)
    session.delete(document)
    session.commit()
