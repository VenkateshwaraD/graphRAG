"""End-to-end QA eval: does the answer contain the expected fact, and
does it cite the right documents? Requires a running API with an index.

Usage: .venv/bin/python scripts/eval_qa.py [--api http://localhost:8000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# question, expected substring (case-insensitive), expected cited doc stems
QA = [
    ("Where is the main office of the department that teaches CS-501?",
     "turing hall", ["campus"]),
    ("Who leads the division that oversees the office administering library fines?",
     "whitfield", ["policies"]),
    ("Which professor chairs the office administering the Energy Systems Certificate?",
     "ferreira", ["faculty"]),
]


def main() -> None:
    """Score multi-hop questions end-to-end through the running API.

    For each QA item: answer_ok checks the expected substring appears in
    the answer (case-insensitive), cite_ok checks at least one citation
    exists / matches a gold document stem. Prints per-question verdicts
    (including which route the agent chose) plus aggregate counts.

    Author: venkateshwara doijode
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()
    import os

    import httpx

    headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
    correct = cited = any_cited = 0
    for q, expected, gold in QA:
        r = httpx.post(f"{args.api}/chat", json={"question": q}, timeout=600, headers=headers)
        if not r.is_success:
            print(f"FAIL http {r.status_code}: {q[:60]}")
            continue
        data = r.json()
        ok = expected in data["answer"].lower()
        cited_docs = {Path(c.get("document_title") or "").stem for c in data["citations"]}
        # cite_ok requires a citation from a GOLD document — merely
        # returning any citation must not count (false-positive fix).
        has_cite = bool(cited_docs & set(gold))
        any_cite = bool(data["citations"])
        correct += ok
        cited += has_cite
        any_cited += any_cite
        print(f"[{data['route']}] answer_ok={ok} cite_ok={has_cite} any_cite={any_cite} :: {q[:70]}")
        print(f"    -> {data['answer'][:200]}")
    n = len(QA)
    print(f"answer correctness: {correct}/{n}  gold citations: {cited}/{n}  any citations: {any_cited}/{n}")


if __name__ == "__main__":
    main()
