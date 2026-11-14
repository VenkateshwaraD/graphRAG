"""Authorization service — RBAC + ABAC decisions for documents.

Explicit precedence (first match decides; default deny):

- **R0 Tenant boundary**: mandatory for every later rule; only
  ``platform_admin`` operates across tenants. Cross-tenant grants are
  rejected at write time and ignored at read time.
- **R1 Privileged administration**: platform_admin -> allow;
  university_admin of the document's tenant -> allow.
- **R2 Ownership**: ``doc.owner_id == caller`` -> allow.
- **R3 Explicit grant (additive)**: a same-tenant grant matching the
  caller (user / role / department / enrolled-or-instructed course).
- **R4 Restricted**: ``restricted`` documents -> deny (only R1-R3 open).
- **R5 Scope narrowing**: when ``department_id``/``course_id`` is set it
  overrides the classification's tenant-wide default — course requires
  enrollment/instructorship, department requires membership.
- **R6 Unscoped classification**: ``public`` -> all tenant users;
  ``internal`` -> non-guest tenant users.
- **R7 Default**: deny.

Grants are additive allow-only: there are no deny records, so removing
a grant revokes access immediately.
"""

from __future__ import annotations

import enum

from sqlalchemy.orm import Session

from app.db.models import (
    Classification,
    Course,
    CourseInstructor,
    Department,
    Document,
    DocumentGrant,
    Enrollment,
    Role,
    User,
)


class GranteeType(enum.StrEnum):
    """DocumentGrant grantee kinds.

    Author: venkateshwara doijode
    """

    USER = "user"
    ROLE = "role"
    DEPARTMENT = "department"
    COURSE = "course"


def unrestricted(principal) -> bool:
    """True when the caller may cross tenant boundaries (platform admin
    or auth-disabled dev mode).

    Author: venkateshwara doijode
    """
    return principal.role == Role.PLATFORM_ADMIN.value


def _user_courses(session: Session, user_id: str) -> tuple[set[str], set[str]]:
    """(enrolled_course_ids, instructed_course_ids) for one user.

    Author: venkateshwara doijode
    """
    enrolled = {
        enrollment.course_id
        for enrollment in session.query(Enrollment.course_id).filter(
            Enrollment.user_id == user_id
        )
    }
    instructed = {
        instructor.course_id
        for instructor in session.query(CourseInstructor.course_id).filter(
            CourseInstructor.user_id == user_id
        )
    }
    return enrolled, instructed


def _grant_matches(
    grant: DocumentGrant,
    principal,
    enrolled: set[str],
    instructed: set[str],
) -> bool:
    """Whether one grant applies to this caller.

    Author: venkateshwara doijode
    """
    kind, target = grant.grantee_type, grant.grantee_id
    if kind == GranteeType.USER.value:
        return target == principal.user_id
    if kind == GranteeType.ROLE.value:
        return target == principal.role
    if kind == GranteeType.DEPARTMENT.value:
        return target == principal.department_id
    if kind == GranteeType.COURSE.value:
        return target in enrolled or target in instructed
    return False


def _decide(doc: Document, principal, grants, enrolled, instructed) -> bool:
    """R0-R7 on one document with precomputed caller context.

    Author: venkateshwara doijode
    """
    # R1 privileged administration
    if principal.role == Role.PLATFORM_ADMIN.value:
        return True
    # R0 tenant boundary (platform admin already returned)
    if doc.tenant_id is None or doc.tenant_id != principal.tenant_id:
        return False
    if principal.role == Role.UNIVERSITY_ADMIN.value:
        return True
    # R2 ownership
    if doc.owner_id is not None and doc.owner_id == principal.user_id:
        return True
    # R3 explicit grant (additive, same-tenant only by construction)
    for grant in grants:
        if _grant_matches(grant, principal, enrolled, instructed):
            return True
    # R4 restricted classification
    if doc.classification == Classification.RESTRICTED.value:
        return False
    # R5 scope narrowing overrides the tenant-wide classification default
    if doc.course_id is not None:
        return doc.course_id in enrolled or doc.course_id in instructed
    if doc.department_id is not None:
        return (
            principal.department_id is not None
            and doc.department_id == principal.department_id
        )
    # R6 unscoped classification
    if doc.classification == Classification.PUBLIC.value:
        return True
    if doc.classification == Classification.INTERNAL.value:
        return principal.role != Role.GUEST.value
    # R7 default deny
    return False


def can_access_document(
    session: Session, doc: Document, principal
) -> bool:
    """Single-document access check (sources, get/delete, citation open).

    Author: venkateshwara doijode
    """
    if principal.role == Role.PLATFORM_ADMIN.value:
        return True
    enrolled, instructed = _user_courses(session, principal.user_id)
    grants = (
        session.query(DocumentGrant)
        .filter(DocumentGrant.document_id == doc.id)
        .all()
    )
    return _decide(doc, principal, grants, enrolled, instructed)


def doc_scope(session: Session, principal) -> set[str]:
    """The concrete set of live ``Document.id`` values the caller may use.

    Always a concrete set — never ``None`` meaning "unrestricted" — so
    an empty result means zero access (fail closed). Platform admins
    get every live document id, which is also what makes stale-graph
    pruning work for them: ids come from the live table only.

    Author: venkateshwara doijode
    """
    docs = session.query(Document).all()
    if principal.role == Role.PLATFORM_ADMIN.value:
        return {doc.id for doc in docs}
    enrolled, instructed = _user_courses(session, principal.user_id)
    grants_by_doc: dict[str, list[DocumentGrant]] = {}
    for grant in session.query(DocumentGrant).all():
        grants_by_doc.setdefault(grant.document_id, []).append(grant)
    return {
        doc.id
        for doc in docs
        if _decide(
            doc, principal, grants_by_doc.get(doc.id, []), enrolled, instructed
        )
    }


def grantee_is_valid(
    session: Session, tenant_id: str, grantee_type: str, grantee_id: str
) -> bool:
    """Grant write-time validation: the grantee exists inside the same
    tenant — grants can never be created across tenant boundaries.

    Author: venkateshwara doijode
    """
    if grantee_type == GranteeType.USER.value:
        user = session.get(User, grantee_id)
        return user is not None and user.tenant_id == tenant_id
    if grantee_type == GranteeType.ROLE.value:
        return grantee_id in {role.value for role in Role}
    if grantee_type == GranteeType.DEPARTMENT.value:
        dept = session.get(Department, grantee_id)
        return dept is not None and dept.tenant_id == tenant_id
    if grantee_type == GranteeType.COURSE.value:
        course = session.get(Course, grantee_id)
        return course is not None and course.tenant_id == tenant_id
    return False
