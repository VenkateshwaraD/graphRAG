from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import ROOT, get_settings

settings = get_settings()

# ensure the data dir exists before sqlite opens its file
settings.upload_dir.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {},
)


@event.listens_for(engine, "connect")
def _fk_on(dbapi_conn, _record):
    """Enable SQLite foreign-key enforcement on every new connection.

    SQLite ships with FK checks off; without this, ``ondelete="CASCADE"``
    on chunk/message rows would be silently ignored. No-op for non-sqlite
    backends.

    Author: venkateshwara doijode
    """
    if settings.database_url.startswith("sqlite"):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def _sqlite_path() -> Path | None:
    """Filesystem path of the sqlite database, or None for other backends.

    Author: venkateshwara doijode
    """
    url = settings.database_url
    prefix = "sqlite:///"
    if url.startswith(prefix) and not url.endswith(":memory:"):
        return Path(url[len(prefix):])
    return None


def _backup_sqlite() -> Path | None:
    """Snapshot the sqlite file before migrations run so a failed
    upgrade can be restored manually (``app.db.bak-<timestamp>``).

    Author: venkateshwara doijode
    """
    import shutil
    import time

    path = _sqlite_path()
    if path is None or not path.exists():
        return None
    bak = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, bak)
    return bak


def migrate_db() -> None:
    """Run ``alembic upgrade head`` against the configured database.

    Alembic is the sole schema path — on SQLite a file backup is taken
    first and any migration failure propagates so startup aborts rather
    than running against a half-migrated schema.

    Author: venkateshwara doijode
    """
    from alembic import command
    from alembic.config import Config

    _backup_sqlite()
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    command.upgrade(cfg, "head")


def init_db() -> None:
    """Startup schema hook: run pending Alembic migrations.

    Kept under the historical name so ``app.main`` and tests don't
    change; the implementation is Alembic, not ``create_all``.

    Author: venkateshwara doijode
    """
    migrate_db()


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped SQLAlchemy session.

    The session is always closed when the request finishes, so callers
    never leak connections.

    Author: venkateshwara doijode
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
