"""Test configuration — must set env BEFORE any app import."""

from __future__ import annotations

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="ugra_test_")

os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")
os.environ.setdefault("EMBEDDING_DIMS", "64")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/app.db")
os.environ.setdefault("LANCEDB_URI", f"{_TMP}/lancedb")
os.environ.setdefault("UPLOAD_DIR", f"{_TMP}/uploads")
os.environ.setdefault("GRAPHRAG_ROOT", f"{_TMP}/graphrag")
# strong enough to pass the bootstrap credential check
os.environ.setdefault("API_KEY", "test-admin-key-9f2c7d41")
os.environ.setdefault("AUTH_FAIL_LIMIT_PER_MINUTE", "1000000")
TEST_API_KEY = "test-admin-key-9f2c7d41"

import pytest  # noqa: E402


@pytest.fixture
def session():
    """Yield a DB session against the temp test database.

    Author: venkateshwara doijode
    """
    from app.db.session import SessionLocal, init_db

    init_db()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    """Yield a FastAPI TestClient (runs lifespan: init_db + settings.yaml).

    Author: venkateshwara doijode
    """
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        c.headers["X-API-Key"] = TEST_API_KEY
        yield c


@pytest.fixture(scope="session")
def world():
    """Build a two-tenant university world covering all six roles.

    Requires the app DB to exist — every test using it must also pull
    in ``client`` (its lifespan runs the migrations) or ``session``.
    Returns ids and raw API keys so tests can act as each persona:

    - tenant_a / tenant_b: two universities
    - dept_cs / dept_law: departments inside tenant_a
    - course_501: CS-501 under dept_cs
    - users: admin_a (university_admin), hod (dept_cs head), faculty
      (instructs course_501), student (enrolled in course_501),
      outsider (student, no enrollment), guest, admin_b (tenant B's
      university admin)
    - keys: username -> raw API key for the X-API-Key header

    Author: venkateshwara doijode
    """
    from app.api.deps import generate_api_key
    from app.db.models import (
        Course,
        CourseInstructor,
        Department,
        Enrollment,
        Role,
        Tenant,
        User,
    )
    from app.db.session import SessionLocal, init_db

    init_db()
    session = SessionLocal()
    try:
        tenant_a = Tenant(slug="uni-a", name="University A")
        tenant_b = Tenant(slug="uni-b", name="University B")
        session.add_all([tenant_a, tenant_b])
        session.flush()
        dept_cs = Department(tenant_id=tenant_a.id, name="Computer Science")
        dept_law = Department(tenant_id=tenant_a.id, name="Law")
        session.add_all([dept_cs, dept_law])
        session.flush()
        course = Course(
            tenant_id=tenant_a.id, department_id=dept_cs.id,
            code="CS-501", title="Data Systems",
        )
        session.add(course)
        session.flush()

        keys: dict[str, str] = {}

        def mk(name, email, role, tenant=tenant_a, dept=None):
            raw, hashed, prefix = generate_api_key()
            user = User(
                tenant_id=tenant.id if tenant else None,
                email=email,
                display_name=name,
                role=role,
                department_id=dept.id if dept else None,
                api_key_hash=hashed,
                key_prefix=prefix,
            )
            session.add(user)
            session.flush()
            keys[name] = raw
            return user

        admin_a = mk("admin_a", "admin@a.edu", Role.UNIVERSITY_ADMIN.value)
        hod = mk("hod", "hod@a.edu", Role.HOD.value, dept=dept_cs)
        faculty = mk("faculty", "fac@a.edu", Role.FACULTY.value, dept=dept_cs)
        student = mk("student", "stu@a.edu", Role.STUDENT.value, dept=dept_cs)
        outsider = mk("outsider", "out@a.edu", Role.STUDENT.value, dept=dept_law)
        guest = mk("guest", "g@a.edu", Role.GUEST.value)
        admin_b = mk(
            "admin_b", "admin@b.edu", Role.UNIVERSITY_ADMIN.value,
            tenant=tenant_b,
        )

        session.add(Enrollment(user_id=student.id, course_id=course.id))
        session.add(CourseInstructor(user_id=faculty.id, course_id=course.id))
        session.commit()

        return {
            "tenant_a": tenant_a.id,
            "tenant_b": tenant_b.id,
            "dept_cs": dept_cs.id,
            "dept_law": dept_law.id,
            "course_501": course.id,
            "users": {
                "admin_a": admin_a.id,
                "hod": hod.id,
                "faculty": faculty.id,
                "student": student.id,
                "outsider": outsider.id,
                "guest": guest.id,
                "admin_b": admin_b.id,
            },
            "keys": keys,
        }
    finally:
        session.close()


def auth(world: dict, who: str) -> dict[str, str]:
    """X-API-Key header dict for one world persona.

    Author: venkateshwara doijode
    """
    return {"X-API-Key": world["keys"][who]}
