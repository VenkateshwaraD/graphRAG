"""Human-in-the-loop review store for extracted relationships.

Reviewers approve/reject relationships through the API. Decisions are
keyed on a content hash (source + target + normalized description)
rather than GraphRAG's per-build relationship id, so verdicts survive
reindexing — the same extracted fact stays approved/rejected, while a
reworded description registers as new unreviewed content.
"""

from __future__ import annotations

import hashlib
import re

from sqlalchemy.orm import Session


def rel_key(source: str, target: str, description: str) -> str:
    """Stable identity hash for one extracted relationship.

    Normalizes case and whitespace so a rebuild producing identical
    content maps to the same key; GraphRAG's internal ids change every
    build and cannot be used for durable review state.

    Author: venkateshwara doijode
    """
    norm = "|".join(
        re.sub(r"\s+", " ", s.strip().upper())
        for s in (source, target, description)
    )
    return hashlib.sha1(norm.encode()).hexdigest()


def load_rejections() -> set[str]:
    """Return rel_keys of every relationship a reviewer rejected.

    Queried per traverse call — a small indexed SQLite read, cheap at
    this scale; revisit with caching if review volume grows.

    Author: venkateshwara doijode
    """
    from app.db.models import RelationshipReview
    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        return {
            r.rel_hash
            for r in session.query(RelationshipReview.rel_hash)
            .filter(RelationshipReview.verdict == "rejected")
            .all()
        }
    finally:
        session.close()


def record_review(
    session: Session,
    *,
    source: str,
    target: str,
    description: str,
    verdict: str,
    note: str | None,
):
    """Upsert a review verdict keyed by ``rel_key``.

    A repeat verdict on the same relationship updates the row instead
    of duplicating — re-review is how a mistaken rejection gets undone.

    Author: venkateshwara doijode
    """
    from app.db.models import RelationshipReview

    key = rel_key(source, target, description)
    row = session.query(RelationshipReview).filter_by(rel_hash=key).first()
    if row is None:
        row = RelationshipReview(
            rel_hash=key, source=source, target=target, description=description
        )
        session.add(row)
    row.verdict = verdict
    row.note = note
    session.commit()
    session.refresh(row)
    return row
