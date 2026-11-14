"""Departments, courses, and enrollment management.

Write access matrix: platform_admin anywhere, university_admin inside
its own tenant, hod only for enrollments/instructors on courses in its
own department. Students can never enroll themselves — enrollment
writes always come from a hod-or-above principal.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, require_api_key
from app.audit import record_audit
from app.db.models import (
    Course,
    CourseInstructor,
    Department,
    Enrollment,
    Role,
    User,
)
from app.db.session import get_session
from app.schemas import (
    CourseIn,
    CourseOut,
    DepartmentIn,
    DepartmentOut,
    EnrollmentIn,
    EnrollmentOut,
)

router = APIRouter(tags=["catalog"])


def _is_platform(p: ApiPrincipal) -> bool:
    """Author: venkateshwara doijode"""
    return p.role == Role.PLATFORM_ADMIN.value


def _tenant_for_write(
    principal: ApiPrincipal, tenant_id: str | None
) -> str:
    """Resolve which tenant a write targets: explicit only for
    platform admins, else the caller's own tenant.

    Author: venkateshwara doijode
    """
    if _is_platform(principal):
        if tenant_id is None:
            raise HTTPException(400, "tenant_id required")
        return tenant_id
    if principal.role == Role.UNIVERSITY_ADMIN.value:
        return principal.tenant_id
    raise HTTPException(403, "administrator role required")


@router.post("/departments", response_model=DepartmentOut, status_code=201)
def create_department(
    body: DepartmentIn,
    tenant_id: str | None = None,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Create a department in the caller's (or specified) tenant.

    Author: venkateshwara doijode
    """
    resolved_tenant = _tenant_for_write(principal, tenant_id)
    dept = Department(tenant_id=resolved_tenant, name=body.name)
    session.add(dept)
    session.commit()
    session.refresh(dept)
    record_audit(
        session, action="department.create", principal=principal,
        resource_type="department", resource_id=dept.id, decision="allow",
    )
    return dept


@router.get("/departments", response_model=list[DepartmentOut])
def list_departments(
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """List departments in the caller's tenant (all for platform).

    Author: venkateshwara doijode
    """
    query = session.query(Department)
    if not _is_platform(principal):
        query = query.filter(Department.tenant_id == principal.tenant_id)
    return query.order_by(Department.name).all()


@router.post("/courses", response_model=CourseOut, status_code=201)
def create_course(
    body: CourseIn,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Create a course under a department (same tenant enforced).

    Author: venkateshwara doijode
    """
    dept = session.get(Department, body.department_id)
    if dept is None:
        raise HTTPException(400, "unknown department")
    tenant_id = _tenant_for_write(principal, dept.tenant_id)
    if dept.tenant_id != tenant_id:
        raise HTTPException(404, "department not found")
    course = Course(
        tenant_id=tenant_id,
        department_id=dept.id,
        code=body.code,
        title=body.title,
    )
    session.add(course)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(409, "course code already exists") from None
    session.refresh(course)
    record_audit(
        session, action="course.create", principal=principal,
        resource_type="course", resource_id=course.id, decision="allow",
    )
    return course


@router.get("/courses", response_model=list[CourseOut])
def list_courses(
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """List courses in the caller's tenant (all for platform).

    Author: venkateshwara doijode
    """
    query = session.query(Course)
    if not _is_platform(principal):
        query = query.filter(Course.tenant_id == principal.tenant_id)
    return query.order_by(Course.code).all()


def _can_manage_enrollment(
    principal: ApiPrincipal, course: Course
) -> bool:
    """Enrollment write gate: platform/uni-admin anywhere in their
    scope; hod only for courses in their own department.

    Author: venkateshwara doijode
    """
    if _is_platform(principal):
        return True
    if course.tenant_id != principal.tenant_id:
        return False
    if principal.role == Role.UNIVERSITY_ADMIN.value:
        return True
    return (
        principal.role == Role.HOD.value
        and course.department_id == principal.department_id
    )


@router.post("/enrollments", response_model=EnrollmentOut, status_code=201)
def enroll(
    body: EnrollmentIn,
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """Enroll a student in — or assign an instructor to — a course.

    ``kind=student`` requires the enrollee's role be ``student``;
    ``kind=instructor`` requires ``faculty`` or ``hod``. HODs manage
    only their own department's courses. Students can never enroll
    themselves — every write needs a hod-or-above principal.

    Author: venkateshwara doijode
    """
    course = session.get(Course, body.course_id)
    if course is None:
        raise HTTPException(404, "course not found")
    if not _can_manage_enrollment(principal, course):
        raise HTTPException(
            403, "enrollment requires a hod-or-above role in scope"
        )
    user = session.get(User, body.user_id)
    if user is None or user.tenant_id != course.tenant_id:
        raise HTTPException(400, "user must exist in the course's tenant")
    if body.kind == "student" and user.role != Role.STUDENT.value:
        raise HTTPException(400, "enrollee must have role student")
    if body.kind == "instructor" and user.role not in (
        Role.FACULTY.value,
        Role.HOD.value,
    ):
        raise HTTPException(400, "instructor must have role faculty or hod")
    model = Enrollment if body.kind == "student" else CourseInstructor
    row = model(user_id=user.id, course_id=course.id)
    session.add(row)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(409, "already assigned") from None
    session.refresh(row)
    record_audit(
        session, action="enrollment.create", principal=principal,
        resource_type="enrollment", resource_id=row.id, decision="allow",
        detail={"kind": body.kind, "course_id": course.id},
    )
    return row


@router.get("/enrollments", response_model=list[EnrollmentOut])
def list_enrollments(
    course_id: str | None = None,
    kind: str = "student",
    principal: ApiPrincipal = Depends(require_api_key),
    session: Session = Depends(get_session),
):
    """List enrollments/instructors for the caller's tenant.

    Author: venkateshwara doijode
    """
    if kind not in ("student", "instructor"):
        raise HTTPException(400, "kind must be student|instructor")
    model = Enrollment if kind == "student" else CourseInstructor
    query = session.query(model).join(Course, model.course_id == Course.id)
    if course_id:
        query = query.filter(model.course_id == course_id)
    if not _is_platform(principal):
        query = query.filter(Course.tenant_id == principal.tenant_id)
    return query.all()
