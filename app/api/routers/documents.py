from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, require_api_key, require_write
from app.audit import record_audit
from app.authz import can_access_document, doc_scope, grantee_is_valid
from app.db.models import (
    Classification,
    Course,
    CourseInstructor,
    Department,
    DocStatus,
    Document,
    DocumentGrant,
    Role,
)
from app.db.session import get_session
from app.ingestion.pipeline import delete_document, process_document, save_upload
from app.schemas import DocumentOut, GrantIn, GrantOut

router = APIRouter(prefix="/documents", tags=["documents"])

_VALID_CLASSIFICATIONS = {cls.value for cls in Classification}
_VALID_GRANTEE_TYPES = {"user", "role", "department", "course"}


def _access_check(session: Session, doc: Document, principal: ApiPrincipal) -> None:
    """404 a document the caller may not access — same shape as
    'not found' so document existence isn't disclosed.

    Author: venkateshwara doijode
    """
    if not can_access_document(session, doc, principal):
        record_audit(
            session,
            action="doc.access",
            principal=principal,
            resource_type="document",
            resource_id=doc.id,
            decision="deny",
        )
        raise HTTPException(404, "document not found")


def _can_manage(doc: Document, principal: ApiPrincipal) -> bool:
    """Owner-or-admin gate for destructive/sharing operations.

    Read access is NOT enough — a student able to read a course
    document must not be able to delete it or alter its grants.
    Platform admins always qualify; university admins qualify only
    inside their own tenant; everyone else must own the document.

    Author: venkateshwara doijode
    """
    if principal.role == Role.PLATFORM_ADMIN.value:
        return True
    if principal.role == Role.UNIVERSITY_ADMIN.value:
        return doc.tenant_id == principal.tenant_id
    return doc.owner_id is not None and doc.owner_id == principal.user_id


def _validate_scope(
    session: Session,
    principal: ApiPrincipal,
    tenant_id: str | None,
    department_id: str | None,
    course_id: str | None,
) -> None:
    """Upload metadata validation: any department/course scope must
    belong to the document's tenant, and non-admin uploaders may only
    scope to their own department or courses they instruct.

    Author: venkateshwara doijode
    """
    admin = principal.role in (
        Role.PLATFORM_ADMIN.value,
        Role.UNIVERSITY_ADMIN.value,
    )
    if department_id is not None:
        dept = session.get(Department, department_id)
        if dept is None or dept.tenant_id != tenant_id:
            raise HTTPException(400, "unknown department")
        if not admin and dept.id != principal.department_id:
            raise HTTPException(403, "cannot scope a document to another department")
    if course_id is not None:
        course = session.get(Course, course_id)
        if course is None or course.tenant_id != tenant_id:
            raise HTTPException(400, "unknown course")
        if not admin:
            teaches = (
                session.query(CourseInstructor)
                .filter(
                    CourseInstructor.user_id == principal.user_id,
                    CourseInstructor.course_id == course_id,
                )
                .first()
            )
            if teaches is None:
                raise HTTPException(
                    403, "cannot scope a document to a course you do not instruct"
                )


@router.post("", response_model=DocumentOut, status_code=201)
def upload(
    file: UploadFile,
    title: str | None = None,
    classification: str = Classification.INTERNAL.value,
    department_id: str | None = None,
    course_id: str | None = None,
    tenant_id: str | None = None,
    principal: ApiPrincipal = Depends(require_write),
    session: Session = Depends(get_session),
):
    """Upload a document and run the full ingest pipeline on it.

    Saves the file, creates the Document row, then extract+chunk+embeds
    synchronously — the returned ``status`` is the final ``chunked`` or
    ``failed`` (with ``error`` detail). Empty uploads get a 400.
    ``classification`` is public|internal|restricted; optional
    department/course scope must be same-tenant (and for non-admins,
    the caller's own department or instructed course).

    Sync endpoint: process_document is CPU-bound (extract+embed), so it
    runs in FastAPI's threadpool, not on the event loop.

    Author: venkateshwara doijode
    """
    from app.config import get_settings

    if classification not in _VALID_CLASSIFICATIONS:
        raise HTTPException(400, f"invalid classification {classification!r}")
    effective_tenant = principal.tenant_id
    if effective_tenant is None:
        # platform admin uploads may target an explicit tenant; without
        # one the document stays platform-admin-only
        if tenant_id is not None:
            from app.db.models import Tenant

            if session.get(Tenant, tenant_id) is None:
                raise HTTPException(400, "unknown tenant")
            effective_tenant = tenant_id
    _validate_scope(session, principal, effective_tenant, department_id, course_id)

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
        tenant_id=effective_tenant,
        owner_id=principal.user_id,
        department_id=department_id,
        course_id=course_id,
        classification=classification,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    process_document(session, doc)
    session.refresh(doc)
    record_audit(
        session,
        action="doc.upload",
        principal=principal,
        resource_type="document",
        resource_id=doc.id,
        decision="allow",
        detail={"classification": classification, "status": doc.status},
    )
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """List documents the caller may access, newest first.

    Author: venkateshwara doijode
    """
    scope = doc_scope(session, principal)
    if not scope:
        return []
    return (
        session.query(Document)
        .filter(Document.id.in_(sorted(scope)))
        .order_by(Document.created_at.desc())
        .all()
    )


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(
    doc_id: str,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Fetch one document's metadata row; 404 when unknown or
    inaccessible to the caller.

    Author: venkateshwara doijode
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    _access_check(session, doc, principal)
    return doc


@router.delete("/{doc_id}", status_code=204)
def remove_document(
    doc_id: str,
    reindex: bool = False,
    principal: ApiPrincipal = Depends(require_write),
    session: Session = Depends(get_session),
):
    """Delete a document everywhere — ORM row, LanceDB chunks, staged
    GraphRAG input, and stored file.

    Deletion is effective immediately: retrieval and graph consumers
    only ever see artifacts provable from live documents, so the doc's
    entities/relationships stop answering even before reindexing.
    ``?reindex=true`` additionally enqueues a GraphRAG rebuild when no
    job is running (check GET /ingest for progress); 404 when the doc
    is unknown or inaccessible; requires a write-capable role.

    Author: venkateshwara doijode
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    _access_check(session, doc, principal)
    if not _can_manage(doc, principal):
        raise HTTPException(403, "only the owner or an admin may delete")
    delete_document(session, doc)
    record_audit(
        session,
        action="doc.delete",
        principal=principal,
        resource_type="document",
        resource_id=doc_id,
        decision="allow",
        detail={"reindex_requested": reindex},
    )
    if reindex:
        from app.ingestion.jobs import try_reserve_job

        try_reserve_job(session)  # None when a build is already running


@router.post("/{doc_id}/grants", response_model=GrantOut, status_code=201)
def add_grant(
    doc_id: str,
    body: GrantIn,
    principal: ApiPrincipal = Depends(require_write),
    session: Session = Depends(get_session),
):
    """Grant access to a document for a user, role, department, or course.

    Only the document owner or a tenant/platform admin may grant, and
    the grantee must resolve inside the document's tenant — grants can
    never cross tenant boundaries. Grants are additive: they open
    access, they cannot deny it.

    Author: venkateshwara doijode
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    if not _can_manage(doc, principal):
        raise HTTPException(403, "only the owner or an admin may grant access")
    if body.grantee_type not in _VALID_GRANTEE_TYPES:
        raise HTTPException(400, f"invalid grantee_type {body.grantee_type!r}")
    if not grantee_is_valid(
        session, doc.tenant_id, body.grantee_type, body.grantee_id
    ):
        raise HTTPException(400, "grantee must exist in the document's tenant")
    grant = DocumentGrant(
        document_id=doc.id,
        grantee_type=body.grantee_type,
        grantee_id=body.grantee_id,
        granted_by=principal.user_id,
    )
    session.add(grant)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(409, "grant already exists") from None
    session.refresh(grant)
    record_audit(
        session,
        action="grant.create",
        principal=principal,
        resource_type="document_grant",
        resource_id=grant.id,
        decision="allow",
        detail={
            "document_id": doc.id,
            "grantee_type": body.grantee_type,
            "grantee_id": body.grantee_id,
        },
    )
    return grant


@router.delete("/{doc_id}/grants/{grant_id}", status_code=204)
def revoke_grant(
    doc_id: str,
    grant_id: str,
    principal: ApiPrincipal = Depends(require_write),
    session: Session = Depends(get_session),
):
    """Revoke one document grant — effective on the next request.

    Author: venkateshwara doijode
    """
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "document not found")
    if not _can_manage(doc, principal):
        raise HTTPException(403, "only the owner or an admin may revoke grants")
    grant = session.get(DocumentGrant, grant_id)
    if grant is None or grant.document_id != doc.id:
        raise HTTPException(404, "grant not found")
    session.delete(grant)
    session.commit()
    record_audit(
        session,
        action="grant.revoke",
        principal=principal,
        resource_type="document_grant",
        resource_id=grant_id,
        decision="allow",
        detail={
            "document_id": doc.id,
            "grantee_type": grant.grantee_type,
            "grantee_id": grant.grantee_id,
        },
    )
