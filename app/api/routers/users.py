"""User lifecycle: create, list, patch, key rotation.

Anti-escalation matrix (enforced here, not just documented):

- only platform_admin may assign ``role=platform_admin``
- university_admin acts only inside its own tenant and may assign
  roles up to its own level (never platform_admin)
- nobody may change their own role or deactivate themselves
- hod/faculty/student/guest have no user-management rights beyond
  rotating their own key
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import (
    ApiPrincipal,
    generate_api_key,
    require_admin,
    require_api_key,
)
from app.audit import record_audit
from app.db.models import Department, Role, User
from app.db.session import get_session
from app.schemas import RotateKeyOut, UserCreatedOut, UserIn, UserOut, UserPatch

router = APIRouter(prefix="/users", tags=["users"])

_RANK = {role.value: i for i, role in enumerate(Role)}  # lower = privileged
_VALID_ROLES = set(_RANK)


def _is_platform(p: ApiPrincipal) -> bool:
    """True for the platform-admin principal.

    Author: venkateshwara doijode
    """
    return p.role == Role.PLATFORM_ADMIN.value


def _can_assign(actor: ApiPrincipal, role: str) -> bool:
    """Role-assignment rule: platform_admin is the only role restricted
    to platform admins; otherwise the actor may assign roles at or
    below their own level.

    Author: venkateshwara doijode
    """
    if role == Role.PLATFORM_ADMIN.value:
        return _is_platform(actor)
    return _RANK[role] >= _RANK[actor.role]


def _target_visible(actor: ApiPrincipal, target: User) -> bool:
    """A non-platform actor only ever sees users in its own tenant —
    cross-tenant targets are invisible (404), not errors.

    Author: venkateshwara doijode
    """
    if _is_platform(actor):
        return True
    return (
        actor.role == Role.UNIVERSITY_ADMIN.value
        and target.tenant_id == actor.tenant_id
    )


def _validate_department(
    session: Session, department_id: str | None, tenant_id: str | None
) -> None:
    """Department must exist and belong to the user's tenant.

    Author: venkateshwara doijode
    """
    if department_id is None:
        return
    dept = session.get(Department, department_id)
    if dept is None or dept.tenant_id != tenant_id:
        raise HTTPException(400, "unknown department")


@router.post("", response_model=UserCreatedOut, status_code=201)
def create_user(
    body: UserIn,
    principal: ApiPrincipal = Depends(require_admin),
    session: Session = Depends(get_session),
):
    """Create a user and mint their API key (returned exactly once).

    Platform admins choose the tenant (``tenant_id`` may be null only
    for another platform admin); university admins are confined to
    their own tenant. Role assignment follows ``_can_assign``.

    Author: venkateshwara doijode
    """
    if body.role not in _VALID_ROLES:
        raise HTTPException(400, f"invalid role {body.role!r}")
    if not _can_assign(principal, body.role):
        raise HTTPException(403, f"cannot assign role {body.role!r}")
    tenant_id = body.tenant_id if _is_platform(principal) else principal.tenant_id
    if body.role == Role.PLATFORM_ADMIN.value:
        tenant_id = None  # platform admins are tenantless
    elif tenant_id is None:
        raise HTTPException(400, "tenant_id required for this role")
    _validate_department(session, body.department_id, tenant_id)
    if (
        session.query(User)
        .filter(User.tenant_id == tenant_id, User.email == body.email)
        .first()
    ):
        raise HTTPException(409, "email already registered in this tenant")
    raw, hashed, prefix = generate_api_key()
    user = User(
        tenant_id=tenant_id,
        email=body.email,
        display_name=body.display_name,
        role=body.role,
        department_id=body.department_id,
        api_key_hash=hashed,
        key_prefix=prefix,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    record_audit(
        session,
        action="user.create",
        principal=principal,
        resource_type="user",
        resource_id=user.id,
        decision="allow",
        detail={"role": user.role, "key_prefix": prefix},
    )
    return UserCreatedOut(
        **UserOut.model_validate(user).model_dump(), api_key=raw
    )


@router.get("", response_model=list[UserOut])
def list_users(
    principal: ApiPrincipal = Depends(require_admin),
    session: Session = Depends(get_session),
):
    """List users — all for platform admins, own tenant for
    university admins.

    Author: venkateshwara doijode
    """
    query = session.query(User)
    if not _is_platform(principal):
        query = query.filter(User.tenant_id == principal.tenant_id)
    return query.order_by(User.created_at).all()


@router.patch("/{user_id}", response_model=UserOut)
def patch_user(
    user_id: str,
    body: UserPatch,
    principal: ApiPrincipal = Depends(require_admin),
    session: Session = Depends(get_session),
):
    """Change a user's role, department, or active flag.

    Anti-escalation: no self role changes or self-deactivation;
    role assignments follow ``_can_assign``; non-platform actors can
    neither see nor touch users outside their tenant, and a
    university_admin cannot edit a platform_admin.

    Author: venkateshwara doijode
    """
    target = session.get(User, user_id)
    if target is None or not _target_visible(principal, target):
        raise HTTPException(404, "user not found")
    if target.id == principal.user_id and (
        body.role is not None or body.is_active is not None
    ):
        raise HTTPException(403, "cannot change your own role or status")
    if body.role is not None:
        if body.role not in _VALID_ROLES:
            raise HTTPException(400, f"invalid role {body.role!r}")
        if not _can_assign(principal, body.role):
            raise HTTPException(403, f"cannot assign role {body.role!r}")
        # platform admins can't be demoted except by platform admins
        if target.role == Role.PLATFORM_ADMIN.value and not _is_platform(principal):
            raise HTTPException(403, "cannot modify a platform administrator")
        target.role = body.role
        if body.role == Role.PLATFORM_ADMIN.value:
            target.tenant_id = None
    if body.department_id is not None:
        _validate_department(session, body.department_id, target.tenant_id)
        target.department_id = body.department_id
    if body.is_active is not None:
        target.is_active = body.is_active
    session.commit()
    session.refresh(target)
    record_audit(
        session,
        action="user.update",
        principal=principal,
        resource_type="user",
        resource_id=target.id,
        decision="allow",
        detail={
            "role": body.role,
            "department_id": body.department_id,
            "is_active": body.is_active,
        },
    )
    return target


@router.post("/{user_id}/rotate-key", response_model=RotateKeyOut)
def rotate_key(
    user_id: str,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Mint a fresh API key — the old key stops working immediately.

    A user may rotate their own key; otherwise requires platform admin
    or a university admin of the target's tenant.

    Author: venkateshwara doijode
    """
    target = session.get(User, user_id)
    if target is None:
        raise HTTPException(404, "user not found")
    if not (
        target.id == principal.user_id
        or _is_platform(principal)
        or (
            principal.role == Role.UNIVERSITY_ADMIN.value
            and target.tenant_id == principal.tenant_id
        )
    ):
        raise HTTPException(403, "cannot rotate another user's key")
    raw, hashed, prefix = generate_api_key()
    target.api_key_hash = hashed
    target.key_prefix = prefix
    session.commit()
    record_audit(
        session,
        action="key.rotate",
        principal=principal,
        resource_type="user",
        resource_id=target.id,
        decision="allow",
        detail={"key_prefix": prefix},
    )
    return RotateKeyOut(api_key=raw, key_prefix=prefix)
