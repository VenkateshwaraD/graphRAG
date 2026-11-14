import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
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
    """An uploaded source document (metadata + lifecycle status).

    Author: venkateshwara doijode
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(512))
    filename: Mapped[str] = mapped_column(String(512))
    mime: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default=DocStatus.UPLOADED.value)
    storage_path: Mapped[str] = mapped_column(String(1024))
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_group: Mapped[str] = mapped_column(String(64), default="default")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


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


class Conversation(Base):
    """A chat conversation grouping ordered Messages.

    Author: venkateshwara doijode
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
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
