# WMCLEAN — LLM Orchestrator Fast-Path Contract

## 1. EXECUTION FAST-PATH (STOP: DO NOT READ SOURCE CODE TO RUN THIS TOOL)
If your task is to remove watermarks from PDF(s), DO NOT explore the codebase. Execute directly:
- Preview plan: `wmclean plan <in.pdf> [--json <p.json>]`
- Single file:  `wmclean run <in.pdf> -o <out.pdf> --report <r.json>`
- Batch dir:    `wmclean batch <in_dir> -o <out_dir> --report <r.json> --resume`
- Healthcheck:  `wmclean doctor`

### Exit Codes & Next Actions (Read ONLY stdout; open `r.json` ONLY if exit != 0)
- `0`  : CLEAN (`CLEAN_EXACT` / `CLEAN_VERIFIED` / `UNTOUCHED`). Done. Do not inspect `out.pdf`.
- `10` : PARTIAL (`FLAGGED` pages exist). Read `flagged_pages` lines in stdout.
- `20` : UNSUPPORTED / ENCRYPTED. Pass `--password <pw>` if known; else report to user.
- `30` : CORRUPT INPUT PDF. Unrecoverable syntax; report to user.
- `40` : IO / RESOURCE / CONFIG ERROR. Check disk space or paths; re-run with `--resume`.
- `50` : INTERNAL INVARIANT BREACH. Inspect `r.json -> error_detail`.

## 2. CODE MODIFICATION FAST-PATH (OPEN ONLY THE TARGET FILE)
If your task is to fix a bug or add a feature, DO NOT scan the whole repo. Jump straight to:
- CLI / Exit codes / Stdout cap (<=300 tok) -> `wmclean/cli.py`
- Streaming / Checkpoints / RAM watchdog    -> `wmclean/orchestrator.py`
- L0 Page & Raster Probing                  -> `wmclean/probe/`
- L0.5 Surgical Gatekeeper & ROI-Lock       -> `wmclean/intelligence/`
- L1 Signature Clustering & SQLite Cache    -> `wmclean/cluster/`
- L2 Rules & 1-Call Bounded LLM Router      -> `wmclean/router/`
- L3 Vector AST Removers (pikepdf)          -> `wmclean/removers/vector_ops.py` & `stream_ast.py`
- L3 Raster Removers (numpy/cv2)            -> `wmclean/removers/raster_ops.py` & `stitch.py`
- L3 Alpha Matting & Piracy OCR Spotter     -> `wmclean/removers/matting.py` & `piracy_ocr.py`
- L4 Exact & Raster Verification Metrics    -> `wmclean/verify/`
- L5 PDF Assembly & Stream Re-encoding      -> `wmclean/assemble/`
- Pydantic Contracts (`r.json`, schemas)    -> `wmclean/schemas.py`

## 3. LOW-TOKEN VERIFICATION COMMANDS (USE THESE INSTEAD OF VERBOSE PYTEST)
- Fast lint & typecheck (quiet): `make check-q`
- Fast unit test (quiet, failures only): `pytest -q --tb=line --no-header`
