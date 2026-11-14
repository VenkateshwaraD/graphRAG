from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, require_admin
from app.db.models import AuditEvent, Role
from app.db.session import get_session
from app.schemas import AuditEventOut

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=list[AuditEventOut])
def list_audit(
    limit: int = 100,
    action: str | None = None,
    decision: str | None = None,
    principal: ApiPrincipal = Depends(require_admin),
    session: Session = Depends(get_session),
):
    """Search audit events — platform admins see everything,
    university admins only their tenant's events.

    Author: venkateshwara doijode
    """
    query = session.query(AuditEvent)
    if principal.role != Role.PLATFORM_ADMIN.value:
        query = query.filter(AuditEvent.tenant_id == principal.tenant_id)
    if action:
        query = query.filter(AuditEvent.action == action)
    if decision:
        query = query.filter(AuditEvent.decision == decision)
    return (
        query.order_by(AuditEvent.created_at.desc())
        .limit(min(limit, 500))
        .all()
    )
