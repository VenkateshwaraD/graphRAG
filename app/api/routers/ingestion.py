from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, require_write
from app.db.models import IngestionJob
from app.db.session import get_session
from app.ingestion.jobs import try_reserve_job
from app.schemas import JobOut

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("", response_model=JobOut, status_code=202)
def trigger_index(
    update: bool = False,
    principal: ApiPrincipal = Depends(require_write),
    session: Session = Depends(get_session),
):
    """Kick off a GraphRAG indexing run over all staged input texts.

    At most one job may be pending/running — a second request gets 409 so
    concurrent builds can't corrupt the shared output dir. The
    check-and-launch is atomic in-process (see ``try_reserve_job``).
    ``?update=true`` asks GraphRAG to incrementally merge new inputs
    instead of rebuilding from scratch (previous output is still backed
    up first). Requires a write-capable key. Returns the job row; poll
    GET /ingest/{id} for status.

    Author: venkateshwara doijode
    """
    job = try_reserve_job(session, update=update)
    if job is None:
        raise HTTPException(409, "an index job is already running")
    return job


@router.get("/{job_id}", response_model=JobOut)
def job_status(job_id: str, session: Session = Depends(get_session)):
    """Fetch one job's status/stats/error; 404 when unknown.

    Author: venkateshwara doijode
    """
    job = session.get(IngestionJob, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@router.get("", response_model=list[JobOut])
def list_jobs(session: Session = Depends(get_session)):
    """List the 20 most recent index jobs, newest first.

    Author: venkateshwara doijode
    """
    return session.query(IngestionJob).order_by(IngestionJob.created_at.desc()).limit(20).all()
