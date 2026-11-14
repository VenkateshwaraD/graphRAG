# Test Evidence — 2026-11-14 11:19 UTC (updated post-fix)

## Static checks
```
ruff check . -> All checks passed!
```

## Unit tests
```
......................                                                   [100%]
22 passed in 13.28s
```

## Live API
```
GET /healthz   -> {"status":"ok"}
GET /readyz    -> {"status":"ok","graph_indexed":true}
GET /graph/stats -> {"entities":28,"relationships":47,"communities":3,
                     "text_units":5,"community_reports":3,"indexed":true}
```

## Route smoke (live, real LLM — post-fix)
```
mode=hybrid   -> route=hybrid   cites=5 :: Based on the provided context, CS-501
                 is taught by Dr. Priya Raman [1], who directs the Data Systems
                 Research Lab [2]. That lab is located in Turing Hall ...
mode=traverse -> route=traverse cites=4 :: CS-501 -> Dr. Priya Raman -> Data
                 Systems Research Lab -> Turing Hall (per-hit hop_path)
mode=local    -> route=local    cites=3 :: "## Building That Hosts the Lab Led
                 by the CS-501 Professor ..." (engine-composed answer)
mode=global   -> route=global   cites=0 :: "## Building Location ..." (report
                 map-reduce; no text-unit citations — expected)

-- auto-router --
factoid question    -> route=hybrid    "CS-501 ... taught by Dr. Priya Raman. [1]"
multi-hop question  -> route=traverse  resolves CS-501 -> professor -> lab -> hall
thematic question   -> route=hybrid    thematic answer composed from top chunks

-- edge cases --
invalid mode -> 400
empty question -> 422
missing source -> 404
missing doc -> 404
```

## Bug found & fixed during dry run (path-decay)
```
graphrag_common.load_config(set_cwd=True) chdir's the whole process into
GRAPHRAG_ROOT. With a RELATIVE root (data/graphrag), every call after the
first resolved against the moved cwd -> data/graphrag/data/graphrag/...
-> FileNotFoundError -> local/global silently fell back to hybrid.

Fix: Settings validators absolutize graphrag_root, lancedb_uri, upload_dir
and relative sqlite database_url against the project root at startup.

Verification — consecutive calls all hold their route:
call 1 local  -> route=local   cites=3 (real engine answer)
call 2 global -> route=global  (no fallback)
call 3 global -> route=global  (repeat call still works)
call 4 local  -> route=local   (path no longer decays)
```

## Known non-blocking observations
```
- global route returns cites=0: GlobalSearch answers over community
  reports, which carry no text-unit rows — no doc citations to map.
  Answer text is still generated correctly.
- Router non-determinism: thematic questions sometimes route hybrid
  instead of global (LLM classifier); answer quality unaffected since
  hybrid context also covers the theme.
- Upload ingestion is synchronous per request (CPU-bound in threadpool).
- Deleting a doc removes chunks/staged input, but graph entities persist
  until the next /ingest rebuild.
```
