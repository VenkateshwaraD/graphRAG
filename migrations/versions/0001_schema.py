"""Initial multi-tenant schema.

Creates all ORM tables (fresh installs get full FKs/uniques/indexes)
and, on pre-existing databases, adds the new ACL columns on
``documents`` and drops the legacy ``doc_group`` column. Batch mode
rebuilds the table so SQLite also gains the new FK constraints.

Revision: 0001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

from app.db.models import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_NEW_DOC_COLUMNS = ("tenant_id", "owner_id", "department_id", "course_id", "classification")
_NEW_DOC_INDEXES = (
    "ix_documents_tenant_id",
    "ix_documents_owner_id",
    "ix_documents_department_id",
    "ix_documents_course_id",
)
# tables introduced by this change — everything else predates alembic
_NEW_TABLES = (
    "tenants",
    "departments",
    "courses",
    "users",
    "enrollments",
    "course_instructors",
    "document_grants",
    "audit_events",
)


def _cols(table: str = "documents") -> set[str]:
    from sqlalchemy import inspect

    return {c["name"] for c in inspect(op.get_bind()).get_columns(table)}


def _tables() -> set[str]:
    from sqlalchemy import inspect

    return set(inspect(op.get_bind()).get_table_names())


def _indexes() -> set[str]:
    from sqlalchemy import inspect

    try:
        return {i["name"] for i in inspect(op.get_bind()).get_indexes("documents")}
    except Exception:
        return set()


def upgrade() -> None:
    """Create missing tables; on legacy DBs add ACL columns to
    ``documents`` and drop ``doc_group``.

    Author: venkateshwara doijode
    """
    Base.metadata.create_all(op.get_bind())
    cols = _cols()
    idx = _indexes()
    with op.batch_alter_table("documents") as b:
        if "tenant_id" not in cols:
            b.add_column(sa.Column("tenant_id", sa.String(32), nullable=True))
        if "owner_id" not in cols:
            b.add_column(sa.Column("owner_id", sa.String(32), nullable=True))
        if "department_id" not in cols:
            b.add_column(sa.Column("department_id", sa.String(32), nullable=True))
        if "course_id" not in cols:
            b.add_column(sa.Column("course_id", sa.String(32), nullable=True))
        if "classification" not in cols:
            b.add_column(
                sa.Column(
                    "classification",
                    sa.String(16),
                    nullable=False,
                    server_default="internal",
                )
            )
        if "doc_group" in cols:
            b.drop_column("doc_group")
    # conversations gains owner scoping (nullable — legacy rows keep
    # working and get lazily claimed by their next authenticated turn)
    conv_cols = _cols("conversations") if "conversations" in _tables() else set()
    if conv_cols:
        with op.batch_alter_table("conversations") as b:
            if "tenant_id" not in conv_cols:
                b.add_column(sa.Column("tenant_id", sa.String(32), nullable=True))
            if "owner_id" not in conv_cols:
                b.add_column(sa.Column("owner_id", sa.String(32), nullable=True))
        existing_conv_idx = {
            i["name"]
            for i in inspect(op.get_bind()).get_indexes("conversations")
        }
        if "ix_conversations_tenant_id" not in existing_conv_idx:
            op.create_index(
                "ix_conversations_tenant_id", "conversations", ["tenant_id"]
            )
    existing = _indexes()
    for name, col in zip(
        _NEW_DOC_INDEXES,
        ("tenant_id", "owner_id", "department_id", "course_id"),
        strict=False,
    ):
        if name not in existing and name not in idx:
            op.create_index(name, "documents", [col])


def downgrade() -> None:
    """Drop the access-control schema (documents columns + new tables)
    and restore ``doc_group``.

    Author: venkateshwara doijode
    """
    cols = _cols()
    with op.batch_alter_table("documents") as b:
        for name in _NEW_DOC_INDEXES:
            try:
                b.drop_index(name)
            except Exception:
                pass
        for col in _NEW_DOC_COLUMNS:
            if col in cols:
                b.drop_column(col)
        if "doc_group" not in cols:
            b.add_column(
                sa.Column(
                    "doc_group",
                    sa.String(64),
                    nullable=False,
                    server_default="default",
                )
            )
    conv_cols = _cols("conversations") if "conversations" in _tables() else set()
    if conv_cols:
        with op.batch_alter_table("conversations") as b:
            try:
                b.drop_index("ix_conversations_tenant_id")
            except Exception:
                pass
            for col in ("tenant_id", "owner_id"):
                if col in conv_cols:
                    b.drop_column(col)
    for name in _NEW_TABLES:
        op.drop_table(name)
