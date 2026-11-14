"""Background GraphRAG indexing jobs, tracked in ingestion_jobs."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.db.models import IngestionJob, JobStatus
from app.db.session import SessionLocal


def create_job(session: Session) -> IngestionJob:
    """Insert a PENDING IngestionJob row and return it.

    The caller then passes ``job.id`` to ``launch_index_job`` so the row
    exists before the worker thread starts updating it.

    Author: venkateshwara doijode
    """
    job = IngestionJob(status=JobStatus.PENDING.value)
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


_reserve_lock = threading.Lock()


def try_reserve_job(session: Session, update: bool = False) -> IngestionJob | None:
    """Atomically check for a live job and launch a new one if free.

    The check-then-create runs under a process-wide lock so two
    concurrent POST /ingest requests can't both see "no running job" and
    launch parallel builds over the shared output dir. Returns None when
    a job is already pending/running (caller maps to 409). ``update``
    requests an incremental index merge instead of a full rebuild.
    Guards only within this process — a multi-worker deployment would
    need a DB-level unique constraint instead.

    Author: venkateshwara doijode
    """
    with _reserve_lock:
        running = (
            session.query(IngestionJob)
            .filter(
                IngestionJob.status.in_(
                    [JobStatus.PENDING.value, JobStatus.RUNNING.value]
                )
            )
            .first()
        )
        if running:
            return None
        job = create_job(session)
        launch_index_job(job.id, update=update)
        return job


def _set(session: Session, job_id: str, **fields) -> None:
    """Update fields on a job row; no-op when the row is gone.

    Author: venkateshwara doijode
    """
    job = session.get(IngestionJob, job_id)
    if job is None:
        return
    for k, v in fields.items():
        setattr(job, k, v)
    session.commit()


def run_index_job(job_id: str, update: bool = False) -> None:
    """Execute the GraphRAG build for one job row.

    Runs in a background thread with its own DB session (the request's
    session is request-scoped). Marks the job RUNNING, invokes
    ``build_graph_index`` on a fresh event loop (incremental merge when
    ``update`` is set), refreshes the cached artifact loader so the new
    graph is immediately searchable, then records DONE + workflow stats
    — or FAILED with the error text.

    Author: venkateshwara doijode
    """
    session = SessionLocal()
    try:
        _set(session, job_id, status=JobStatus.RUNNING.value, started_at=datetime.now(UTC))
        from app.ingestion.graph_index import build_graph_index

        stats = asyncio.run(build_graph_index(update=update))
        from app.retrieval.artifacts import load_artifacts

        load_artifacts(refresh=True)
        _set(
            session,
            job_id,
            status=JobStatus.DONE.value,
            finished_at=datetime.now(UTC),
            stats=stats,
        )
    except Exception as exc:
        _set(
            session,
            job_id,
            status=JobStatus.FAILED.value,
            finished_at=datetime.now(UTC),
            error=str(exc)[:2000],
        )
    finally:
        session.close()


def launch_index_job(job_id: str, update: bool = False) -> threading.Thread:
    """Spawn a daemon thread running ``run_index_job`` for ``job_id``.

    Author: venkateshwara doijode
    """
    t = threading.Thread(
        target=run_index_job, args=(job_id,), kwargs={"update": update}, daemon=True
    )
    t.start()
    return t
