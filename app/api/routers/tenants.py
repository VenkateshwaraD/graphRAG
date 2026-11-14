from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, require_admin
from app.audit import record_audit
from app.db.models import Role, Tenant
from app.db.session import get_session
from app.schemas import TenantIn, TenantOut

router = APIRouter(prefix="/tenants", tags=["tenants"])


@router.post("", response_model=TenantOut, status_code=201)
def create_tenant(
    body: TenantIn,
    principal: ApiPrincipal = Depends(require_admin),
    session: Session = Depends(get_session),
):
    """Register a university tenant — platform administrators only.

    Author: venkateshwara doijode
    """
    if principal.role != Role.PLATFORM_ADMIN.value:
        raise HTTPException(403, "only platform admins manage tenants")
    if session.query(Tenant).filter_by(slug=body.slug).first():
        raise HTTPException(409, "tenant slug already exists")
    tenant = Tenant(slug=body.slug, name=body.name)
    session.add(tenant)
    session.commit()
    session.refresh(tenant)
    record_audit(
        session,
        action="tenant.create",
        principal=principal,
        resource_type="tenant",
        resource_id=tenant.id,
        decision="allow",
        detail={"slug": tenant.slug},
    )
    return tenant


@router.get("", response_model=list[TenantOut])
def list_tenants(
    principal: ApiPrincipal = Depends(require_admin),
    session: Session = Depends(get_session),
):
    """List tenants — all for platform admins, own tenant for
    university admins.

    Author: venkateshwara doijode
    """
    query = session.query(Tenant)
    if principal.role != Role.PLATFORM_ADMIN.value:
        query = query.filter(Tenant.id == principal.tenant_id)
    return query.order_by(Tenant.created_at).all()
