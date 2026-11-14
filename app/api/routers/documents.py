from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.models import DocStatus, Document
from app.db.session import get_session
from app.ingestion.pipeline import delete_document, process_document, save_upload
from app.schemas import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=201)
def upload(
    file: UploadFile,
    title: str | None = None,
    session: Session = Depends(get_session),
):
    """Upload a document and run the full ingest pipeline on it.

    Saves the file, creates the Document row, then extract+chunk+embeds
    synchronously — the returned ``status`` is the final ``chunked`` or
    ``failed`` (with ``error`` detail). Empty uploads get a 400.

    Sync endpoint: process_document is CPU-bound (extract+embed), so it
    runs in FastAPI's threadpool, not on the event loop.

    Author: venkateshwara doijode
    """
    data = file.file.read()
    if not data:
        raise HTTPException(400, "empty file")
    path = save_upload(file.filename or "upload", data)
    doc = Document(
        title=title or file.filename or path.name,
        filename=file.filename or path.name,
        mime=file.content_type or "application/octet-stream",
        storage_path=str(path),
        status=DocStatus.UPLOADED.value,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    process_document(session, doc)
    session.refresh(doc)
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)):
    """List all documents, newest first.

    Author: venkateshwara doijode
    """
    return session.query(Document).order_by(Document.created_at.desc()).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str, session: Session = Depends(get_session)):
    """Fetch one document's metadata row; 404 when unknown.

    Author: venkateshwara doijode
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    return doc


@router.delete("/{doc_id}", status_code=204)
def remove_document(doc_id: str, session: Session = Depends(get_session)):
    """Delete a document everywhere — ORM row, LanceDB chunks, staged
    GraphRAG input, and stored file. 404 when unknown.

    Author: venkateshwara doijode
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    delete_document(session, doc)
