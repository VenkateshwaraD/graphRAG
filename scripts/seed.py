"""Seed the assistant with the sample multi-hop corpus.

Usage:
    .venv/bin/python scripts/seed.py [--api http://localhost:8000]

Without --api it writes directly through the pipeline (same DB defaults).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "seed_corpus"


def _bootstrap_demo(api: str, headers: dict) -> str | None:
    """Create a demo tenant + department + course + student user.

    Idempotent-ish: reuses an existing 'demo' tenant when present.
    Returns the tenant id so uploads land inside it (visible to the
    demo users, not just the platform admin).

    Author: venkateshwara doijode
    """
    import httpx

    tenants = httpx.get(f"{api}/tenants", headers=headers, timeout=30)
    tenant_id = None
    if tenants.is_success:
        for tenant in tenants.json():
            if tenant["slug"] == "demo":
                tenant_id = tenant["id"]
    if tenant_id is None:
        created = httpx.post(
            f"{api}/tenants",
            json={"slug": "demo", "name": "Demo University"},
            headers=headers,
            timeout=30,
        )
        if not created.is_success:
            print("tenant bootstrap failed:", created.status_code, created.text)
            return None
        tenant_id = created.json()["id"]

    dept = httpx.post(
        f"{api}/departments",
        json={"name": "Computer Science"},
        params={"tenant_id": tenant_id},
        headers=headers,
        timeout=30,
    )
    dept_id = dept.json()["id"] if dept.is_success else None
    if dept_id:
        httpx.post(
            f"{api}/courses",
            json={
                "code": "CS-501",
                "title": "Data Systems",
                "department_id": dept_id,
            },
            headers=headers,
            timeout=30,
        )
    user = httpx.post(
        f"{api}/users",
        json={
            "email": "student@demo.edu",
            "display_name": "Demo Student",
            "role": "student",
            "tenant_id": tenant_id,
            "department_id": dept_id,
        },
        headers=headers,
        timeout=30,
    )
    if user.is_success:
        print("demo student key:", user.json()["api_key"])
    return tenant_id


def seed_via_api(api: str) -> None:
    """Bootstrap the demo tenant and upload every seed_corpus/*.txt
    through a running API's /documents.

    Sends the API_KEY env var as X-API-Key when the server has auth on.
    Documents land in the demo tenant as public so tenant users can
    retrieve them.

    Author: venkateshwara doijode
    """
    import os

    import httpx

    headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
    tenant_id = _bootstrap_demo(api, headers)
    for src_file in sorted(CORPUS.glob("*.txt")):
        params = {"title": src_file.stem, "classification": "public"}
        if tenant_id:
            params["tenant_id"] = tenant_id
        result = httpx.post(
            f"{api}/documents",
            files={"file": (src_file.name, src_file.read_bytes())},
            params=params,
            headers=headers,
            timeout=300,
        )
        print(
            src_file.name, "->", result.status_code,
            result.json().get("status") if result.is_success else result.text,
        )


def seed_direct() -> None:
    """Ingest every seed_corpus/*.txt in-process through the pipeline.

    Used when no API is running — same code path as an HTTP upload.

    Author: venkateshwara doijode
    """
    from app.db.models import DocStatus, Document, Tenant
    from app.db.session import SessionLocal, init_db
    from app.ingestion.pipeline import process_document

    init_db()
    session = SessionLocal()
    try:
        tenant = session.query(Tenant).filter_by(slug="demo").first()
        if tenant is None:
            tenant = Tenant(slug="demo", name="Demo University")
            session.add(tenant)
            session.commit()
        for src_file in sorted(CORPUS.glob("*.txt")):
            doc = Document(
                title=src_file.stem,
                filename=src_file.name,
                mime="text/plain",
                storage_path=str(src_file),
                status=DocStatus.UPLOADED.value,
                tenant_id=tenant.id,
                classification="public",
            )
            session.add(doc)
            session.commit()
            n_chunks = process_document(session, doc)
            print(src_file.name, "->", doc.status, f"({n_chunks} chunks)")
    finally:
        session.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=None)
    args = ap.parse_args()
    seed_via_api(args.api) if args.api else seed_direct()
