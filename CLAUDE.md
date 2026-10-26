# WMCLEAN — Agent-Native Forensic Actuator Contract

## 1. EXECUTION FAST-PATH (STOP: DO NOT READ SOURCE CODE TO RUN THIS TOOL)
If your task is to remove watermarks from PDF(s), DO NOT explore the codebase. Execute directly:
- Preview plan: `wmclean plan <in.pdf> [--json <p.json>]`
- Single file:  `wmclean run <in.pdf> -o <out.pdf> --report <r.json> [--steer <dsl>] [--dewarp]`
- Batch dir:    `wmclean batch <in_dir> -o <out_dir> --report <r.json> --resume`
- Dewarp page:  `wmclean dewarp <in.pdf> -o <out.pdf> [--mode auto|skew|gutter|bump]`
- Page inspect: `wmclean peek <in.pdf> -p <page> [--crop-roi <box> -o <crop.png>]`
- Time-travel:  `wmclean steer <in.pdf> -s <dsl> [-o <out.pdf>]`
- Healthcheck:  `wmclean doctor`
- Tool schema:  `wmclean schema`

### Exit Codes & Next Actions (Read ONLY stdout; open `r.json` ONLY if exit != 0)
- `0`  : CLEAN (`CLEAN_EXACT` / `CLEAN_VERIFIED` / `UNTOUCHED`). Done. Do not inspect `out.pdf`.
- `10` : PARTIAL (`FLAGGED` pages exist). Read `flagged_pages` lines in stdout.
- `15` : UNVERIFIED (`UNVERIFIED` pages exist: watermark residual was not successfully removed under verified bounds). Inspect `unverified_pages` lines in stdout.
- `20` : UNSUPPORTED / ENCRYPTED. Pass `--password <pw>` if known; else report to user.
- `30` : CORRUPT INPUT PDF. Unrecoverable syntax; report to user.
- `40` : IO / RESOURCE / CONFIG ERROR. Check disk space or paths; re-run with `--resume`.
- `50` : INTERNAL INVARIANT BREACH. Inspect `r.json -> error_detail`.

## 2. CODE MODIFICATION FAST-PATH (OPEN ONLY THE TARGET FILE)
If your task is to fix a bug or add a feature, DO NOT scan the whole repo. Jump straight to:
- CLI / Exit codes / Stdout cap (<=250 tok) -> `wmclean/cli.py`
- Streaming / Checkpoints / RAM watchdog    -> `wmclean/orchestrator.py`
- L-1 Zero-Crop Dewarp & Deskewing          -> `wmclean/geometry/`
- L0 Page & Raster Probing                  -> `wmclean/probe/`
- L0.5 Surgical Gatekeeper, ROI-Lock, DocShield -> `wmclean/intelligence/`
- L1 Signature Clustering & SQLite Cache    -> `wmclean/cluster/`
- L2 Rules & 1-Call Bounded LLM Router      -> `wmclean/router/`
- L3 Vector AST & Layer Redaction           -> `wmclean/removers/vector_ops.py`, `mupdf_layer_redact.py`, `font_zero.py`
- L3 Zero-Smudge Radiometric Removers       -> `wmclean/removers/density_sub.py` & `pointwise_unmix.py`
- L3 Morphological, Twin-Clone & Skeleton   -> `wmclean/removers/bilingual_clone.py`, `maxtree_filter.py`, `skeleton_graph.py`
- L3 Raster Removers (numpy/cv2)            -> `wmclean/removers/raster_ops.py` & `stitch.py`
- L3 Alpha Matting & Piracy OCR Spotter     -> `wmclean/removers/matting.py` & `piracy_ocr.py`
- L4 Verification & Character BBox Shield   -> `wmclean/verify/text_protect.py` & `engine.py`
- L5 PDF Assembly & Delta Ledger            -> `wmclean/assemble/` & `delta_ledger.py`
- L6 Steering Micro-DSL & Auto-Tuner        -> `wmclean/intelligence/steering_dsl.py`, `auto_tuner.py`, `peek.py`
- Pydantic Contracts & Optional Deps        -> `wmclean/schemas.py` & `wmclean/deps.py`

## 3. LOW-TOKEN VERIFICATION COMMANDS (USE THESE INSTEAD OF VERBOSE PYTEST)
- Fast lint & typecheck (quiet): `make check-q`
- Fast unit test (quiet, failures only): `pytest -q --tb=line --no-header`
- End-to-end smoke test: `make smoke`
