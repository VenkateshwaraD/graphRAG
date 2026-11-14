"""Backfill the ``local`` tenant onto pre-existing documents.

Every legacy document is assigned to a ``local`` tenant so the tenant
boundary holds — but ``owner_id`` is deliberately left NULL: ownership
is never fabricated for documents whose uploader can't be established.
The assignment is recorded in ``audit_events`` and logged rather than
silent.

Revision: 0002
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.backfill")


def upgrade() -> None:
    """Create the ``local`` tenant (idempotent), assign all orphaned
    documents to it, record the assignment in ``audit_events``, and
    validate every document has a tenant.

    Author: venkateshwara doijode
    """
    bind = op.get_bind()
    local_id = bind.execute(
        sa.text("SELECT id FROM tenants WHERE slug = 'local'")
    ).scalar()
    if local_id is None:
        local_id = uuid.uuid4().hex
        bind.execute(
            sa.text(
                "INSERT INTO tenants (id, slug, name, created_at)"
                " VALUES (:id, 'local', 'Local University', :ts)"
            ),
            {"id": local_id, "ts": datetime.now(UTC)},
        )
    orphan_ids = [
        r[0]
        for r in bind.execute(
            sa.text("SELECT id FROM documents WHERE tenant_id IS NULL")
        )
    ]
    if orphan_ids:
        bind.execute(
            sa.text("UPDATE documents SET tenant_id = :t WHERE tenant_id IS NULL"),
            {"t": local_id},
        )
    bind.execute(
        sa.text(
            "INSERT INTO audit_events"
            " (id, tenant_id, user_id, action, resource_type, resource_id,"
            "  decision, detail, created_at)"
            " VALUES (:id, :t, NULL, 'migration.backfill', 'documents',"
            " NULL, NULL, :detail, :ts)"
        ),
        {
            "id": uuid.uuid4().hex,
            "t": local_id,
            "detail": __import__("json").dumps(
                {"assigned_documents": len(orphan_ids), "document_ids": orphan_ids}
            ),
            "ts": datetime.now(UTC),
        },
    )
    logger.info(
        "tenant backfill: assigned %d document(s) to 'local' (owner left NULL)",
        len(orphan_ids),
    )
    remaining = bind.execute(
        sa.text("SELECT count(*) FROM documents WHERE tenant_id IS NULL")
    ).scalar()
    if remaining:
        raise RuntimeError(f"backfill incomplete: {remaining} documents lack tenant")


def downgrade() -> None:
    """Detach documents from the ``local`` tenant and remove it plus the
    backfill audit row.

    Author: venkateshwara doijode
    """
    bind = op.get_bind()
    local_id = bind.execute(
        sa.text("SELECT id FROM tenants WHERE slug = 'local'")
    ).scalar()
    if local_id is None:
        return
    bind.execute(
        sa.text("UPDATE documents SET tenant_id = NULL WHERE tenant_id = :t"),
        {"t": local_id},
    )
    bind.execute(
        sa.text(
            "DELETE FROM audit_events WHERE action = 'migration.backfill'"
            " AND tenant_id = :t"
        ),
        {"t": local_id},
    )
    bind.execute(sa.text("DELETE FROM tenants WHERE id = :t"), {"t": local_id})
