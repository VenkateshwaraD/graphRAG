# Test Evidence — 2026-11-14 11:19 UTC

## Static checks
```
All checks passed!
```

## Unit tests
```
......................                                                   [100%]
22 passed in 13.28s
```

## Live API
```
{"status":"ok"}
{"status":"ok","graph_indexed":true}
{"entities":28,"relationships":47,"communities":3,"text_units":5,"community_reports":3,"indexed":true}
```

## Route smoke (live, real LLM)
```
route=hybrid cites=5 :: Based on the provided context, CS-501 is taught by Dr. Priya Raman [1], who directs the Data Systems Research Lab [2]. That lab is located i
route=traverse cites=4 :: Based on the context, CS-501 is taught by Dr. Priya Raman [3], who directs the Data Systems Research Lab [4]. The Data Systems Research Lab 
route=hybrid cites=5 :: Based on the provided context, CS-501 is taught by Dr. Priya Raman [1], who directs the Data Systems Research Lab [2]. That lab is located i
route=hybrid cites=5 :: Based on the provided context, CS-501 is taught by Dr. Priya Raman [1], who directs the Data Systems Research Lab [2]. That lab is located i

-- auto-router --
route=hybrid    :: CS-501 Advanced Database Systems is taught by Dr. Priya Raman. [1]
route=traverse  :: Based on the context, CS-501 is taught by Dr. Priya Raman [3], who directs the Data Systems Research
route=hybrid    :: ## Main Themes Across All Documents  The documents collectively cover several interconnected themes 

-- edge cases --
invalid mode -> 400
empty question -> 422
missing source -> 404
missing doc -> 404
```

## Post-fix verification (path-decay bug)
```
Bug: graphrag_common.load_config(set_cwd=True) chdir's the process into
GRAPHRAG_ROOT; a RELATIVE root then resolved against the new cwd on each
later call -> data/graphrag/data/graphrag/... -> local/global silently
fell back to hybrid. Fixed: Settings validators absolutize graphrag_root,
lancedb_uri, upload_dir and relative sqlite database_url at startup.

call 1 local  -> route=local   cites=3 (real engine answer)
call 2 global -> route=global  (no fallback)
call 3 global -> route=global  (repeat call still works)
call 4 local  -> route=local   (path no longer decays)
```
