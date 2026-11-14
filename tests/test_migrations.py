"""Alembic migration tests — upgrade, legacy-schema adoption, backfill
validation, and rollback, each against a throwaway sqlite file.

Migrations resolve their URL from ``get_settings().database_url``,
which is cached per-process — so each scenario runs alembic in a
subprocess with its own ``DATABASE_URL``.

Author: venkateshwara doijode
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)

_ALEMBIC_SNIPPET = """
from alembic.config import Config
from alembic import command
cfg = Config(%r)
command.%s(cfg, %r)
"""


def _alembic(db: Path, action: str, rev: str = "head") -> None:
    """Run ``command.<action>(rev)`` in a subprocess pointed at ``db``.

    Author: venkateshwara doijode
    """
    env = {**os.environ, "DATABASE_URL": f"sqlite:///{db}"}
    env.pop("PROFILE", None)
    result = subprocess.run(
        [str(PY), "-c", _ALEMBIC_SNIPPET % (str(ROOT / "alembic.ini"), action, rev)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"alembic {action} {rev} failed:\n{result.stderr}"
    )


def _cols(db: Path, table: str) -> set[str]:
    """Column names of ``table`` in the sqlite file.

    Author: venkateshwara doijode
    """
    con = sqlite3.connect(db)
    try:
        return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    finally:
        con.close()


def _make_legacy_db(db: Path) -> None:
    """Create a pre-migration schema: documents carries doc_group, none
    of the auth tables exist.

    Author: venkateshwara doijode
    """
    con = sqlite3.connect(db)
    try:
        con.executescript(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY, title TEXT, filename TEXT, mime TEXT,
                status TEXT, storage_path TEXT, error TEXT,
                created_at DATETIME,
                doc_group VARCHAR(64) NOT NULL DEFAULT 'default'
            );
            CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at DATETIME);
            INSERT INTO documents
              (id, title, filename, mime, status, storage_path, created_at, doc_group)
              VALUES ('d1', 'T', 'f.txt', 'text/plain', 'chunked', '/x',
                      '2024-01-01', 'faculty');
            """
        )
        con.commit()
    finally:
        con.close()


def test_fresh_upgrade_creates_schema(tmp_path):
    """A blank database reaches the full multi-tenant schema."""
    db = tmp_path / "fresh.db"
    _alembic(db, "upgrade", "head")
    cols = _cols(db, "documents")
    assert {"tenant_id", "owner_id", "classification"} <= cols
    assert "doc_group" not in cols
    con = sqlite3.connect(db)
    try:
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    finally:
        con.close()
    assert {
        "tenants", "users", "departments", "courses",
        "enrollments", "document_grants", "audit_events",
    } <= tables


def test_legacy_upgrade_backfills_tenant(tmp_path):
    """A legacy doc_group DB gains ACL columns, loses doc_group, and
    every orphan document is assigned a tenant (owner stays NULL)."""
    db = tmp_path / "legacy.db"
    _make_legacy_db(db)
    _alembic(db, "upgrade", "head")
    cols = _cols(db, "documents")
    assert "doc_group" not in cols
    assert {"tenant_id", "owner_id", "classification"} <= cols
    con = sqlite3.connect(db)
    try:
        tenant, owner = con.execute(
            "SELECT tenant_id, owner_id FROM documents WHERE id='d1'"
        ).fetchone()
    finally:
        con.close()
    assert tenant is not None          # backfilled, never left dangling
    assert owner is None               # ownership is never fabricated
    # conversation ownership columns added for legacy rows too
    assert {"tenant_id", "owner_id"} <= _cols(db, "conversations")


def test_downgrade_restores_legacy_shape(tmp_path):
    """Rolling back to base restores doc_group and drops the ACL schema."""
    db = tmp_path / "roll.db"
    _make_legacy_db(db)
    _alembic(db, "upgrade", "head")
    _alembic(db, "downgrade", "base")
    cols = _cols(db, "documents")
    assert "doc_group" in cols
    assert "tenant_id" not in cols
    con = sqlite3.connect(db)
    try:
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    finally:
        con.close()
    assert "tenants" not in tables and "document_grants" not in tables


def test_upgrade_is_idempotent(tmp_path):
    """Re-running upgrade head on a migrated DB is a clean no-op."""
    db = tmp_path / "twice.db"
    _alembic(db, "upgrade", "head")
    _alembic(db, "upgrade", "head")
    assert {"tenant_id", "owner_id"} <= _cols(db, "documents")
