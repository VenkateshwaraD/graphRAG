"""Alembic environment — online migrations against DATABASE_URL.

``render_as_batch`` is required for SQLite: dropping ``documents.doc_group``
and adding constraints needs table rebuilds (SQLite ALTER is limited).
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from app.config import get_settings  # noqa: E402
from app.db.models import Base  # noqa: E402

target_metadata = Base.metadata


def _url() -> str:
    # settings DATABASE_URL wins over alembic.ini so tests can point the
    # runner at a temp database via env.
    return get_settings().database_url


def run_migrations_online() -> None:
    """Run migrations against a live engine.

    Author: venkateshwara doijode
    """
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = _url()
    connectable = engine_from_config(
        cfg, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


run_migrations_online()
