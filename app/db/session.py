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
    """Create all ORM tables and add any missing columns.

    ``create_all`` covers new tables; the column sync below upgrades
    existing databases (e.g. adding ``documents.doc_group`` or
    ``messages.metrics``) without a full migration framework — new ORM
    columns are nullable/defaulted by convention, so ``ALTER TABLE
    ADD COLUMN`` is always safe.

    Author: venkateshwara doijode
    """
    from sqlalchemy import inspect, text

    from app.db.models import Base

    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        existing = {
            t: {c["name"] for c in inspect(conn).get_columns(t)}
            for t in Base.metadata.tables
        }
        for table in Base.metadata.tables.values():
            for col in table.columns:
                if col.name not in existing.get(table.name, ()):
                    ddl = str(
                        col.type.compile(dialect=engine.dialect)
                    )
                    stmt = f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}'
                    conn.execute(text(stmt))
                    # Python-side defaults don't apply to existing rows —
                    # backfill simple scalar defaults so filters treat
                    # legacy rows as the default value.
                    if col.default is not None and col.default.is_scalar:
                        conn.execute(
                            text(
                                f"UPDATE {table.name} SET {col.name} = :v "
                                f"WHERE {col.name} IS NULL"
                            ),
                            {"v": col.default.arg},
                        )


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
