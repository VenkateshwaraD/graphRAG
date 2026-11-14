from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, require_api_key
from app.db.models import Document
from app.db.session import get_session
from app.ingestion.extract import extract_text
from app.schemas import SourceOut

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/{doc_id}", response_model=SourceOut)
def get_source(
    doc_id: str,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Return the extracted text of a source document (for citation drill-down).

    Re-extracts text from the stored file at read time. 404 when the
    document is unknown or outside the caller's groups; 500 when the
    file can't be re-read (deleted or corrupted after ingest).

    Author: venkateshwara doijode
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    if principal.groups is not None and doc.doc_group not in principal.groups:
        raise HTTPException(404, "document not found")
    try:
        text = extract_text(Path(doc.storage_path))
    except Exception as exc:
        raise HTTPException(500, f"could not read source: {exc}") from exc
    return SourceOut(id=doc.id, title=doc.title, filename=doc.filename, text=text)
