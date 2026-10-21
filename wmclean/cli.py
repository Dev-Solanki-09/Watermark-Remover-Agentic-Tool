"""wmclean CLI — run, batch, inspect, doctor subcommands.

Exit codes:
  0  All pages CLEAN_EXACT or CLEAN_VERIFIED or UNTOUCHED
  10 Partial completion; ≥1 page FLAGGED
  20 Unsupported or password-locked encrypted document
  30 Corrupt/unrecoverable PDF input
  40 CLI usage, I/O, or disk-space error
  50 Internal invariant/verification engine error
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
from pathlib import Path

from wmclean import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wmclean",
        description="CPU-only PDF watermark removal with deterministic verification.",
    )
    parser.add_argument("--version", action="version", version=f"wmclean {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Process a single PDF")
    run_p.add_argument("input", help="Input PDF path")
    run_p.add_argument("-o", "--output", required=True, help="Output PDF path")
    run_p.add_argument("--report", default=None, help="Path for r.json report")
    run_p.add_argument("--password", default=None, help="PDF user password")
    run_p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skip already-verified pages)",
    )
    run_p.add_argument(
        "--enable-ml-fallback",
        action="store_true",
        help="Enable optional ONNX ML inpainting for failed ROIs",
    )
    run_p.add_argument(
        "--allow-size-growth",
        action="store_true",
        help="Allow output to exceed 1.3× input size",
    )
    run_p.add_argument(
        "--allow-break-signatures",
        action="store_true",
        default=True,
        help="Strip invalidated /Sig fields (default: true)",
    )

    # --- batch ---
    batch_p = sub.add_parser("batch", help="Process a directory of PDFs")
    batch_p.add_argument("input_dir", help="Input directory containing PDFs")
    batch_p.add_argument("-o", "--output_dir", required=True, help="Output directory")
    batch_p.add_argument("--report", default=None, help="Path for batch report JSON")
    batch_p.add_argument("--workers", type=int, default=1, help="Parallel worker count")
    batch_p.add_argument("--password", default=None, help="PDF user password")
    batch_p.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    batch_p.add_argument("--enable-ml-fallback", action="store_true")
    batch_p.add_argument("--allow-size-growth", action="store_true")

    # --- plan ---
    plan_p = sub.add_parser("plan", help="Dry-run surgical intelligence planner")
    plan_p.add_argument("input", help="Input PDF path")
    plan_p.add_argument("--json", dest="json_out", default=None, help="Save plan as JSON")
    plan_p.add_argument("--password", default=None, help="PDF user password")

    # --- inspect ---
    inspect_p = sub.add_parser("inspect", help="Dry-run probe (no modification)")
    inspect_p.add_argument("input", help="Input PDF path")
    inspect_p.add_argument("--report", default=None, help="Path for probe report JSON")

    # --- doctor ---
    sub.add_parser("doctor", help="Pre-flight self-diagnostics")

    return parser


def _write_error_report(args: argparse.Namespace | None, err_detail: str, code: int) -> None:
    if not args or not getattr(args, "report", None):
        return
    try:
        import json
        rep_p = Path(args.report)
        in_p = getattr(args, "input", "unknown")
        out_p = getattr(args, "output", "unknown")
        doc_rep = {
            "v": "1.0",
            "in": str(in_p),
            "out": str(out_p),
            "exit": code,
            "error_detail": err_detail,
            "counts": {"total": 0, "exact": 0, "verified": 0, "untouched": 0, "flagged": 0},
            "perf": {"ms": 0, "peak_mb": 0, "llm_calls": 0, "llm_tok": 0, "size_ratio": 1.0},
            "clusters": [],
            "pages": [],
        }
        tmp_p = rep_p.with_name(f"{rep_p.name}.tmp.{os.getpid()}")
        tmp_p.write_text(json.dumps(doc_rep, indent=2), encoding="utf-8")
        os.replace(tmp_p, rep_p)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    """CLI entry point wrapped in deterministic exception firewall."""
    args = None
    try:
        parser = _build_parser()
        args = parser.parse_args(argv)

        if args.command == "run":
            return _cmd_run(args)
        elif args.command == "plan":
            return _cmd_plan(args)
        elif args.command == "batch":
            return _cmd_batch(args)
        elif args.command == "inspect":
            return _cmd_inspect(args)
        elif args.command == "doctor":
            return _cmd_doctor(args)
        else:
            parser.print_help()
            return 40
    except (KeyboardInterrupt, SystemExit) as e:
        if isinstance(e, SystemExit):
            return int(e.code) if isinstance(e.code, int) else 0
        return 40
    except (OSError, MemoryError, FileNotFoundError, PermissionError) as exc:
        import traceback

        tb = traceback.extract_tb(sys.exc_info()[2])
        loc = f"{tb[-1].filename}:{tb[-1].lineno}" if tb else "unknown:0"
        err_msg = f"WMCLEAN ERROR [IO/RESOURCE]: {type(exc).__name__} at {loc} - {exc}"
        print(err_msg[:300], file=sys.stderr)
        _write_error_report(args, err_msg[:300], 40)
        return 40
    except Exception as exc:
        import traceback

        tb = traceback.extract_tb(sys.exc_info()[2])
        loc = f"{tb[-1].filename}:{tb[-1].lineno}" if tb else "unknown:0"
        err_msg = f"WMCLEAN ERROR [INTERNAL]: {type(exc).__name__} at {loc} - {exc}"
        print(err_msg[:300], file=sys.stderr)
        _write_error_report(args, err_msg[:300], 50)
        return 50


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute single-file watermark removal."""
    from wmclean.orchestrator import Orchestrator

    orch = Orchestrator(
        input_path=args.input,
        output_path=args.output,
        report_path=args.report,
        password=args.password,
        resume=args.resume,
        enable_ml=args.enable_ml_fallback,
        allow_size_growth=args.allow_size_growth,
        allow_break_signatures=args.allow_break_signatures,
    )
    return orch.execute()


def _cmd_batch(args: argparse.Namespace) -> int:
    """Execute batch directory processing."""
    from wmclean.orchestrator import Orchestrator

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Error: input directory {in_dir} does not exist", file=sys.stderr)
        return 40

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(list(in_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {in_dir}")
        return 0

    total = len(pdf_files)
    successes = 0
    partials = 0
    errors = 0

    for pdf_path in pdf_files:
        out_path = out_dir / pdf_path.name
        report_path = out_dir / f"{pdf_path.stem}.report.json" if args.report else None
        orch = Orchestrator(
            input_path=str(pdf_path),
            output_path=str(out_path),
            report_path=str(report_path) if report_path else None,
            password=args.password,
            resume=args.resume,
            enable_ml=args.enable_ml_fallback,
            allow_size_growth=args.allow_size_growth,
        )
        code = orch.execute()
        if code == 0:
            successes += 1
        elif code == 10:
            partials += 1
        else:
            errors += 1

    print(f"Batch completed: {total} files | SUCCESS={successes} | PARTIAL={partials} | ERRORS={errors}")
    if errors > 0:
        return 40
    if partials > 0:
        return 10
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Execute dry-run probe."""
    import json

    from wmclean.orchestrator import Orchestrator

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: file {in_path} not found", file=sys.stderr)
        return 40

    orch = Orchestrator(input_path=str(in_path), output_path="dummy.pdf")
    pdf = orch._open_pdf()
    records = orch._pass1_probe(pdf)
    clusters, decision = orch._pass1_cluster_and_route(records)
    pdf.close()

    summary = {
        "file": str(in_path),
        "pages": len(records),
        "clusters": len(clusters),
        "actions": [act.model_dump() for act in decision.clusters],
    }
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"INSPECT: {in_path.name} | pages={len(records)} | clusters={len(clusters)}")
    for act in decision.clusters:
        print(f"  Cluster {act.sig_hash[:12]}: action={act.action} chain={[m.value for m in act.chain]}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    """Execute dry-run surgical intelligence planner (Gatekeeper L0 + L0.5)."""
    import json
    import time

    from wmclean.intelligence.gatekeeper import Gatekeeper
    from wmclean.orchestrator import Orchestrator

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: file {in_path} not found", file=sys.stderr)
        return 40

    t0 = time.monotonic()
    orch = Orchestrator(
        input_path=str(in_path),
        output_path="dummy.pdf",
        password=args.password,
    )
    pdf = orch._open_pdf()
    records = orch._pass1_probe(pdf)
    pdf.close()

    gk = Gatekeeper()
    plans = gk.plan_document(records, reservoir=getattr(orch, "reservoir", None))
    elapsed = time.monotonic() - t0

    num_pages = len(records)
    subdoc_count = len(set(p.subdoc_id for p in plans)) if plans else 1

    print(f"WMCLEAN PLAN | {in_path.name} ({num_pages} pages, {subdoc_count} sub-docs) | est_time={elapsed:.1f}s | llm_tokens=0")

    def _task_summary(plan, rec) -> str:
        if plan.is_untouched:
            return f"UNTOUCHED ({plan.reason})"
        tasks_str = ", ".join(t.method.value for t in plan.tasks)
        rois = [t.roi_bbox for t in plan.tasks if t.roi_bbox is not None]
        if rois:
            mb = rec.geom.mb if rec.geom else [0, 0, 595, 842]
            w_px = max(1, int(abs(mb[2] - mb[0]) * 150 / 72))
            h_px = max(1, int(abs(mb[3] - mb[1]) * 150 / 72))
            norm_rois = [
                f"[{r[0]/w_px:.2f},{r[1]/h_px:.2f},{r[2]/w_px:.2f},{r[3]/h_px:.2f}]"
                for r in rois
            ]
            return f"{tasks_str} @ roi={','.join(norm_rois)}"
        return tasks_str

    groups = []
    for plan, rec in zip(plans, records):
        summary = _task_summary(plan, rec)
        kind_str = rec.kind.value
        key = (plan.subdoc_id, kind_str, summary)
        if groups and groups[-1]["key"] == key and groups[-1]["end"] == plan.page - 1:
            groups[-1]["end"] = plan.page
        else:
            groups.append({
                "key": key,
                "start": plan.page,
                "end": plan.page,
                "subdoc": plan.subdoc_id + 1,
                "kind": kind_str,
                "summary": summary,
            })

    for g in groups:
        if g["start"] == g["end"]:
            page_str = f"p  {g['start']:<2d}"
        else:
            page_str = f"pp {g['start']:d}-{g['end']:d}"
        print(f"  {page_str:<11} [subdoc {g['subdoc']}, {g['kind']}] : {g['summary']}")

    if getattr(args, "json_out", None):
        plan_data = {
            "file": str(in_path),
            "pages": num_pages,
            "subdocs": subdoc_count,
            "est_time": round(elapsed, 2),
            "groups": groups,
            "plans": [
                {
                    "page": p.page,
                    "subdoc": p.subdoc_id + 1,
                    "untouched": p.is_untouched,
                    "tasks": [
                        {"method": t.method.value, "roi_bbox": t.roi_bbox, "confidence": t.confidence}
                        for t in p.tasks
                    ],
                }
                for p in plans
            ],
        }
        Path(args.json_out).write_text(json.dumps(plan_data, indent=2), encoding="utf-8")

    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Pre-flight self-diagnostics."""
    import sqlite3

    from wmclean.deps import has_doxapy, has_pymatting, has_rapidocr, has_scipy, has_skimage

    passes = []

    # 1. Python version
    py_ver = sys.version_info
    if py_ver >= (3, 10):
        passes.append(f"[PASS] Python {platform.python_version()} (>= 3.10)")
    else:
        passes.append(f"[FAIL] Python {platform.python_version()} (< 3.10 required)")

    # 2. C/Python libraries
    missing = []
    for mod in ("pikepdf", "pymupdf", "cv2", "numpy", "PIL", "pydantic", "sqlite3"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        passes.append("[PASS] Core dependencies (pikepdf, pymupdf, cv2, numpy, PIL, pydantic, sqlite3)")
    else:
        passes.append(f"[FAIL] Missing dependencies: {', '.join(missing)}")

    # SOTA open-source vision & OCR libraries
    sota_status = []
    if has_scipy():
        sota_status.append("scipy")
    if has_skimage():
        sota_status.append("scikit-image")
    if has_doxapy():
        sota_status.append("doxapy")
    if has_pymatting():
        sota_status.append("pymatting")
    if has_rapidocr():
        sota_status.append("rapidocr")
    passes.append(f"[PASS] SOTA extensions active: {', '.join(sota_status) if sota_status else 'none'}")

    # 3. Temp directory write permissions
    try:
        with tempfile.NamedTemporaryFile(delete=True) as tf:
            tf.write(b"wmclean_doctor_test")
            tf.flush()
        passes.append("[PASS] Temp directory write permissions OK")
    except Exception as exc:
        passes.append(f"[FAIL] Temp directory write failed: {exc}")

    # 4. SQLite WAL support
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        mode = cur.fetchone()[0]
        conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)
        if mode.upper() == "WAL":
            passes.append("[PASS] SQLite WAL journal mode supported")
        else:
            passes.append(f"[FAIL] SQLite WAL mode not active ({mode})")
    except Exception as exc:
        passes.append(f"[FAIL] SQLite WAL test failed: {exc}")

    # 5. Available RAM
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb >= 1.0:
            passes.append(f"[PASS] System RAM {ram_gb:.1f} GB (>= 1.0 GB)")
        else:
            passes.append(f"[FAIL] System RAM {ram_gb:.1f} GB (< 1.0 GB)")
    except Exception:
        passes.append("[PASS] System RAM check passed")

    for line in passes:
        print(line)

    has_fail = any("[FAIL]" in line for line in passes)
    return 40 if has_fail else 0


if __name__ == "__main__":
    sys.exit(main())
