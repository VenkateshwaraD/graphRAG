from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

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


def init_db() -> None:
    """Create all ORM tables if they do not already exist.

    Called once from the FastAPI lifespan and by tests; idempotent.

    Author: venkateshwara doijode
    """
    from app.db.models import Base

    Base.metadata.create_all(engine)


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
