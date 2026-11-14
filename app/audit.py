"""Audit event recording — ids and actions only.

Never store document contents or API keys; ``detail`` should carry
metadata (ips, prefixes, counts, resource ids) not payloads.
"""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.db.models import AuditEvent

logger = logging.getLogger(__name__)


def record_audit(
    session: Session,
    *,
    action: str,
    principal=None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    decision: str | None = None,
    detail: dict | None = None,
) -> AuditEvent:
    """Write one audit row and commit.

    ``principal`` (optional) supplies tenant_id/user_id; failures are
    logged not raised so auditing never breaks the request path.

    Author: venkateshwara doijode
    """
    try:
        ev = AuditEvent(
            tenant_id=getattr(principal, "tenant_id", None),
            user_id=getattr(principal, "user_id", None),
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            decision=decision,
            detail=detail,
        )
        session.add(ev)
        session.commit()
        return ev
    except Exception:
        logger.warning("audit write failed for action %s", action, exc_info=True)
        session.rollback()
        return AuditEvent(action=action)
