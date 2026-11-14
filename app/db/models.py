import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    """Generate a 32-char hex UUID used as a primary key.

    Author: venkateshwara doijode
    """
    return uuid.uuid4().hex


def _now() -> datetime:
    """Return the current UTC timestamp for ``created_at`` defaults.

    Author: venkateshwara doijode
    """
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models.

    Author: venkateshwara doijode
    """

    pass


# ---------------------------------------------------------------------------
# Multi-tenant identity and access model
# ---------------------------------------------------------------------------


class Role(enum.StrEnum):
    """User roles, ordered most to least privileged.

    Author: venkateshwara doijode
    """

    PLATFORM_ADMIN = "platform_admin"
    UNIVERSITY_ADMIN = "university_admin"
    HOD = "hod"
    FACULTY = "faculty"
    STUDENT = "student"
    GUEST = "guest"


ROLE_RANK = {r: i for i, r in enumerate(Role)}  # lower = more privileged

WRITE_ROLES = frozenset(
    {
        Role.PLATFORM_ADMIN.value,
        Role.UNIVERSITY_ADMIN.value,
        Role.HOD.value,
        Role.FACULTY.value,
    }
)


class Classification(enum.StrEnum):
    """Document visibility classes.

    public   -> all users in the tenant (when unscoped)
    internal -> non-guest users in the tenant (when unscoped)
    restricted -> owner, tenant/platform admins, or explicit grant only

    Author: venkateshwara doijode
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class Tenant(Base):
    """A university — the hard isolation boundary.

    Author: venkateshwara doijode
    """

    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Department(Base):
    """An academic department inside one tenant.

    Author: venkateshwara doijode
    """

    __tablename__ = "departments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Course(Base):
    """A course owned by one department of one tenant.

    Author: venkateshwara doijode
    """

    __tablename__ = "courses"
    __table_args__ = (UniqueConstraint("tenant_id", "code"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), index=True
    )
    department_id: Mapped[str] = mapped_column(
        ForeignKey("departments.id", ondelete="CASCADE"), index=True
    )
    code: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class User(Base):
    """One account inside a tenant (``tenant_id`` NULL = platform admin).

    ``api_key_hash`` is the sha256 of the issued key — the raw key is
    only ever returned at issuance/rotation. ``key_prefix`` is a
    non-secret fragment for identifying keys in admin lists and audits.

    Author: venkateshwara doijode
    """

    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "email"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True, index=True
    )
    email: Mapped[str] = mapped_column(String(320))
    display_name: Mapped[str] = mapped_column(String(256), default="")
    role: Mapped[str] = mapped_column(String(32), default=Role.STUDENT.value)
    department_id: Mapped[str | None] = mapped_column(
        ForeignKey("departments.id", ondelete="SET NULL"), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(default=True)
    api_key_hash: Mapped[str | None] = mapped_column(
        String(64), unique=True, index=True, nullable=True
    )
    key_prefix: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Enrollment(Base):
    """A student registered for a course — gates course-scoped documents.

    Author: venkateshwara doijode
    """

    __tablename__ = "enrollments"
    __table_args__ = (UniqueConstraint("user_id", "course_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    course_id: Mapped[str] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class CourseInstructor(Base):
    """A faculty member assigned to teach a course.

    Author: venkateshwara doijode
    """

    __tablename__ = "course_instructors"
    __table_args__ = (UniqueConstraint("user_id", "course_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    course_id: Mapped[str] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class DocStatus(enum.StrEnum):
    """Lifecycle states of an ingested document.

    UPLOADED -> CHUNKED on success, UPLOADED -> FAILED with ``error`` on
    any extraction/embedding problem.

    Author: venkateshwara doijode
    """

    UPLOADED = "uploaded"
    CHUNKED = "chunked"
    FAILED = "failed"


class Document(Base):
    """An uploaded source document (metadata + lifecycle status + ACL).

    Access rule summary (see ``app.authz``): tenant isolation first,
    then privileged admin -> owner -> explicit grant -> classification
    and department/course scope. ``doc_group`` was replaced by this
    structured metadata.

    Author: venkateshwara doijode
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str | None] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"), nullable=True, index=True
    )
    owner_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    department_id: Mapped[str | None] = mapped_column(
        ForeignKey("departments.id", ondelete="SET NULL"), nullable=True, index=True
    )
    course_id: Mapped[str | None] = mapped_column(
        ForeignKey("courses.id", ondelete="SET NULL"), nullable=True, index=True
    )
    classification: Mapped[str] = mapped_column(
        String(16), default=Classification.INTERNAL.value
    )
    title: Mapped[str] = mapped_column(String(512))
    filename: Mapped[str] = mapped_column(String(512))
    mime: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default=DocStatus.UPLOADED.value)
    storage_path: Mapped[str] = mapped_column(String(1024))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    grants: Mapped[list["DocumentGrant"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentGrant(Base):
    """An explicit additive allow on one document for one grantee.

    ``grantee_type`` is one of ``user`` | ``role`` | ``department`` |
    ``course``; ``grantee_id`` is the corresponding id or role name.
    Grants never cross tenants — enforced at write time.

    Author: venkateshwara doijode
    """

    __tablename__ = "document_grants"
    __table_args__ = (UniqueConstraint("document_id", "grantee_type", "grantee_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    grantee_type: Mapped[str] = mapped_column(String(16))
    grantee_id: Mapped[str] = mapped_column(String(64))
    granted_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    document: Mapped[Document] = relationship(back_populates="grants")


class DocumentChunk(Base):
    """One embedded text chunk of a Document (metadata row; the vector
    itself lives in LanceDB keyed by this row's ``id``).

    ``embedding_model`` records which provider produced the vector so the
    retriever can exclude stale embeddings after a model change.

    Author: venkateshwara doijode
    """

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    idx: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    document: Mapped[Document] = relationship(back_populates="chunks")


class AuditEvent(Base):
    """Security-relevant activity: auth decisions, document access,
    grants, admin mutations, and user queries.

    Stores ids and actions only — never document contents or API keys.

    Author: venkateshwara doijode
    """

    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str] = mapped_column(String(64))
    resource_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Conversation(Base):
    """A chat conversation grouping ordered Messages.

    ``owner_id``/``tenant_id`` scope history to the user who started it;
    a conversation is never resumed across owners or tenants.

    Author: venkateshwara doijode
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str | None] = mapped_column(
        String(32), nullable=True, index=True
    )
    owner_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    """One turn in a conversation; assistant rows also carry the chosen
    retrieval ``route`` and serialized ``citations``.

    Author: venkateshwara doijode
    """

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    route: Mapped[str | None] = mapped_column(String(32), nullable=True)
    route_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class JobStatus(enum.StrEnum):
    """States of a background GraphRAG indexing job.

    Author: venkateshwara doijode
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class RelationshipReview(Base):
    """Human verdict on one extracted relationship.

    Keyed by ``rel_hash`` — a content hash of source+target+description —
    so verdicts survive GraphRAG rebuilds that mint new relationship ids.

    Author: venkateshwara doijode
    """

    __tablename__ = "relationship_reviews"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    rel_hash: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    source: Mapped[str] = mapped_column(String(512))
    target: Mapped[str] = mapped_column(String(512))
    description: Mapped[str] = mapped_column(Text, default="")
    verdict: Mapped[str] = mapped_column(String(16))  # approved | rejected
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class IngestionJob(Base):
    """Bookkeeping row for one GraphRAG index run: status, timestamps,
    per-workflow stats, and error text on failure.

    Author: venkateshwara doijode
    """

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.PENDING.value)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    stats: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
