from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, require_api_key, require_write
from app.db.models import DocStatus, Document
from app.db.session import get_session
from app.ingestion.pipeline import delete_document, process_document, save_upload
from app.schemas import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])


def _group_check(doc: Document, principal: ApiPrincipal) -> None:
    """404 a document outside the caller's allowed groups — same shape
    as 'not found' so group membership isn't disclosed.

    Author: venkateshwara doijode
    """
    if principal.groups is not None and doc.doc_group not in principal.groups:
        raise HTTPException(404, "document not found")


@router.post("", response_model=DocumentOut, status_code=201)
def upload(
    file: UploadFile,
    title: str | None = None,
    doc_group: str = "default",
    principal: ApiPrincipal = Depends(require_write),
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
    from app.config import get_settings

    max_bytes = get_settings().max_upload_bytes
    data = file.file.read(max_bytes + 1)
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > max_bytes:
        raise HTTPException(
            413, f"file exceeds max upload size of {max_bytes} bytes"
        )
    path = save_upload(file.filename or "upload", data)
    doc = Document(
        title=title or file.filename or path.name,
        filename=file.filename or path.name,
        mime=file.content_type or "application/octet-stream",
        storage_path=str(path),
        status=DocStatus.UPLOADED.value,
        doc_group=doc_group or "default",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    process_document(session, doc)
    session.refresh(doc)
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """List documents the caller's groups may see, newest first.

    Author: venkateshwara doijode
    """
    q = session.query(Document)
    if principal.groups is not None:
        q = q.filter(Document.doc_group.in_(sorted(principal.groups)))
    return q.order_by(Document.created_at.desc()).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(
    doc_id: str,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Fetch one document's metadata row; 404 when unknown or outside
    the caller's groups.

    Author: venkateshwara doijode
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    _group_check(doc, principal)
    return doc


@router.delete("/{doc_id}", status_code=204)
def remove_document(
    doc_id: str,
    principal: ApiPrincipal = Depends(require_write),
    session: Session = Depends(get_session),
):
    """Delete a document everywhere — ORM row, LanceDB chunks, staged
    GraphRAG input, and stored file. 404 when unknown or outside the
    caller's groups; requires a write-capable key.

    Author: venkateshwara doijode
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    _group_check(doc, principal)
    delete_document(session, doc)
