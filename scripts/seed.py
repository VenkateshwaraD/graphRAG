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


def seed_via_api(api: str) -> None:
    """Upload every seed_corpus/*.txt through a running API's /documents.

    Sends the API_KEY env var as X-API-Key when the server has auth on.

    Author: venkateshwara doijode
    """
    import os

    import httpx

    headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
    for f in sorted(CORPUS.glob("*.txt")):
        r = httpx.post(
            f"{api}/documents",
            files={"file": (f.name, f.read_bytes())},
            params={"title": f.stem},
            headers=headers,
            timeout=300,
        )
        print(f.name, "->", r.status_code, r.json().get("status") if r.is_success else r.text)


def seed_direct() -> None:
    """Ingest every seed_corpus/*.txt in-process through the pipeline.

    Used when no API is running — same code path as an HTTP upload.

    Author: venkateshwara doijode
    """
    from app.db.models import DocStatus, Document
    from app.db.session import SessionLocal, init_db
    from app.ingestion.pipeline import process_document

    init_db()
    session = SessionLocal()
    try:
        for f in sorted(CORPUS.glob("*.txt")):
            doc = Document(
                title=f.stem,
                filename=f.name,
                mime="text/plain",
                storage_path=str(f),
                status=DocStatus.UPLOADED.value,
            )
            session.add(doc)
            session.commit()
            n = process_document(session, doc)
            print(f.name, "->", doc.status, f"({n} chunks)")
    finally:
        session.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=None)
    args = ap.parse_args()
    seed_via_api(args.api) if args.api else seed_direct()
