from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, require_api_key
from app.audit import record_audit
from app.authz import can_access_document
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

    Re-extracts text from the stored file at read time and re-checks
    authorization — a revoked grant or deleted document denies access
    even if a citation was issued earlier. 404 when the document is
    unknown or inaccessible; 500 when the file can't be re-read.

    Author: venkateshwara doijode
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    if not can_access_document(session, doc, principal):
        record_audit(
            session,
            action="source.open",
            principal=principal,
            resource_type="document",
            resource_id=doc.id,
            decision="deny",
        )
        raise HTTPException(404, "document not found")
    try:
        text = extract_text(Path(doc.storage_path))
    except Exception as exc:
        raise HTTPException(500, f"could not read source: {exc}") from exc
    record_audit(
        session,
        action="source.open",
        principal=principal,
        resource_type="document",
        resource_id=doc.id,
        decision="allow",
    )
    return SourceOut(id=doc.id, title=doc.title, filename=doc.filename, text=text)
