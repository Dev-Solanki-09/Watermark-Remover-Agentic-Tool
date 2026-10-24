"""Benchmark Execution Harness.

Executes and compares:
1. No-Op Baseline
2. Oracle-Tuned Threshold Baseline
3. wmclean Active Pipeline

Outputs structured Markdown table and prints to console.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import psutil
import pymupdf

from tests.benchmarks.baselines import NoOpBaseline, OracleThresholdBaseline
from tests.benchmarks.evaluator import EvaluationResult, evaluate_page_cleaning
from tests.benchmarks.fixture_generator import BenchmarkFixture, generate_benchmark_suite
from wmclean.orchestrator import Orchestrator


def run_single_fixture_benchmark(
    fixture: BenchmarkFixture,
    tmp_dir: Path,
) -> List[EvaluationResult]:
    """Run all methods on a single benchmark fixture and return evaluation results."""
    results: List[EvaluationResult] = []

    # 1. No-Op Baseline
    noop = NoOpBaseline()
    t0 = time.perf_counter()
    noop_out = noop.clean(fixture.watermarked_bgr)
    t_noop = time.perf_counter() - t0
    results.append(
        evaluate_page_cleaning(
            clean_gt_bgr=fixture.clean_bgr,
            watermarked_bgr=fixture.watermarked_bgr,
            cleaned_bgr=noop_out,
            wm_mask_gt=fixture.wm_mask_gt,
            fg_mask_gt=fixture.fg_mask_gt,
            method_name=noop.name,
            fixture_id=fixture.fixture_id,
            duration_sec=t_noop,
            peak_ram_mb=0.0,
        )
    )

    # 2. Oracle Threshold Baseline
    oracle = OracleThresholdBaseline()
    t0 = time.perf_counter()
    oracle_out = oracle.clean(
        fixture.watermarked_bgr,
        clean_gt_bgr=fixture.clean_bgr,
        wm_mask_gt=fixture.wm_mask_gt,
        fg_mask_gt=fixture.fg_mask_gt,
    )
    t_oracle = time.perf_counter() - t0
    results.append(
        evaluate_page_cleaning(
            clean_gt_bgr=fixture.clean_bgr,
            watermarked_bgr=fixture.watermarked_bgr,
            cleaned_bgr=oracle_out,
            wm_mask_gt=fixture.wm_mask_gt,
            fg_mask_gt=fixture.fg_mask_gt,
            method_name=oracle.name,
            fixture_id=fixture.fixture_id,
            duration_sec=t_oracle,
            peak_ram_mb=0.0,
        )
    )

    # 3. wmclean Active Pipeline
    # Synthesize single-page PDF containing the raster fixture
    in_pdf = tmp_dir / f"{fixture.fixture_id}_in.pdf"
    out_pdf = tmp_dir / f"{fixture.fixture_id}_out.pdf"
    report_json = tmp_dir / f"{fixture.fixture_id}_report.json"

    fh, fw = fixture.watermarked_bgr.shape[:2]
    pt_w = fw * 72.0 / fixture.dpi
    pt_h = fh * 72.0 / fixture.dpi

    doc = pymupdf.open()
    page = doc.new_page(width=pt_w, height=pt_h)
    _, png_bytes = cv2.imencode(".png", fixture.watermarked_bgr)
    page.insert_image(pymupdf.Rect(0, 0, pt_w, pt_h), stream=png_bytes.tobytes())
    doc.save(str(in_pdf))
    doc.close()

    t_wmclean = 0.0
    peak_ram = 0.0
    orch = Orchestrator(str(in_pdf), str(out_pdf), report_path=str(report_json))
    try:
        import threading
        proc = psutil.Process()
        peak_bytes = proc.memory_info().rss
        stop_event = threading.Event()

        def _monitor():
            nonlocal peak_bytes
            while not stop_event.is_set():
                try:
                    rss = proc.memory_info().rss
                    if rss > peak_bytes:
                        peak_bytes = rss
                except Exception:
                    pass
                time.sleep(0.005)

        mon_t = threading.Thread(target=_monitor, daemon=True)
        mon_t.start()
        t0 = time.perf_counter()
        try:
            orch.execute()
        finally:
            stop_event.set()
            mon_t.join(timeout=0.2)
            t_wmclean = time.perf_counter() - t0
            try:
                peak_bytes = max(peak_bytes, proc.memory_info().rss)
            except Exception:
                pass

        peak_ram = peak_bytes / (1024 * 1024)

        # Extract cleaned raster image
        if out_pdf.exists():
            out_doc = pymupdf.open(str(out_pdf))
            out_page = out_doc[0]
            imgs = out_page.get_images()
            if imgs:
                xref = imgs[0][0]
                extracted = out_doc.extract_image(xref)
                decoded = cv2.imdecode(np.frombuffer(extracted["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None:
                    wmclean_bgr = decoded
                else:
                    pix = out_page.get_pixmap(dpi=fixture.dpi)
                    wmclean_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    wmclean_bgr = cv2.cvtColor(wmclean_arr, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_GRAY2BGR)
            else:
                pix = out_page.get_pixmap(dpi=fixture.dpi)
                wmclean_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    wmclean_bgr = cv2.cvtColor(wmclean_arr, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    wmclean_bgr = cv2.cvtColor(wmclean_arr, cv2.COLOR_RGB2BGR)
                else:
                    wmclean_bgr = cv2.cvtColor(wmclean_arr, cv2.COLOR_GRAY2BGR)
            out_doc.close()

            # Align dimensions via slice/pad if off-by-one, never resize/interpolate
            gh, gw = fixture.clean_bgr.shape[:2]
            if wmclean_bgr.shape[:2] != (gh, gw):
                aligned = np.full((gh, gw, 3), 255, dtype=np.uint8)
                ch = min(gh, wmclean_bgr.shape[0])
                cw = min(gw, wmclean_bgr.shape[1])
                aligned[:ch, :cw] = wmclean_bgr[:ch, :cw]
                wmclean_bgr = aligned

            results.append(
                evaluate_page_cleaning(
                    clean_gt_bgr=fixture.clean_bgr,
                    watermarked_bgr=fixture.watermarked_bgr,
                    cleaned_bgr=wmclean_bgr,
                    wm_mask_gt=fixture.wm_mask_gt,
                    fg_mask_gt=fixture.fg_mask_gt,
                    method_name="wmclean (active)",
                    fixture_id=fixture.fixture_id,
                    duration_sec=t_wmclean,
                    peak_ram_mb=peak_ram,
                )
            )
        else:
            # Fallback if processing failed
            results.append(
                EvaluationResult(
                    method_name="wmclean (active)",
                    fixture_id=fixture.fixture_id,
                    residual_paper=999.0,
                    ink_loss_rate=1.0,
                    outside_mask_change_rate=1.0,
                    duration_sec=round(t_wmclean, 3),
                    peak_ram_mb=round(peak_ram, 1),
                    ocr_cer=None,
                    extra={"error": "Output PDF was not produced"},
                )
            )
    except Exception as exc:
        results.append(
            EvaluationResult(
                method_name="wmclean (active)",
                fixture_id=fixture.fixture_id,
                residual_paper=999.0,
                ink_loss_rate=1.0,
                outside_mask_change_rate=1.0,
                duration_sec=round(t_wmclean, 3),
                peak_ram_mb=round(peak_ram, 1),
                ocr_cer=None,
                extra={"error": str(exc)},
            )
        )

    return results


def format_markdown_table(all_results: List[EvaluationResult]) -> str:
    """Format evaluation results into a clear GitHub Markdown comparison table."""
    lines = [
        "| Fixture ID | Method | Paper Residual (MAE) | Ink Loss Rate (%) | Outside Mask Change (%) | Time (s) | RAM (MB) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in all_results:
        ink_loss_pct = f"{r.ink_loss_rate * 100:.2f}%"
        outside_pct = f"{r.outside_mask_change_rate * 100:.2f}%"
        lines.append(
            f"| `{r.fixture_id}` | **{r.method_name}** | {r.residual_paper:.2f} | {ink_loss_pct} | {outside_pct} | {r.duration_sec:.2f}s | {r.peak_ram_mb:.0f} |"
        )
    return "\n".join(lines)


def run_benchmark_suite(quick: bool = True, output_md: Path | None = None) -> List[EvaluationResult]:
    """Execute benchmark suite and optionally write markdown report."""
    fixtures = generate_benchmark_suite(quick=quick)
    all_results: List[EvaluationResult] = []

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for i, fixture in enumerate(fixtures, 1):
            sys.stdout.write(f"[{i}/{len(fixtures)}] Evaluating {fixture.fixture_id} ({fixture.description})...\n")
            sys.stdout.flush()
            fixture_res = run_single_fixture_benchmark(fixture, tmp_dir)
            all_results.extend(fixture_res)

    md_table = format_markdown_table(all_results)
    print("\n" + md_table + "\n")

    if output_md is not None:
        header = (
            "# wmclean Benchmark & Baseline Comparison Table\n\n"
            f"Generated: 2026-10-22 | Quick Suite: {quick} | Total Tests: {len(all_results)}\n\n"
            "### Metric Definitions\n"
            "- **Paper Residual (MAE)**: Mean absolute pixel intensity error on background paper within the watermark zone (0.0 is perfect).\n"
            "- **Ink Loss Rate (%)**: Fraction of ground-truth foreground text and diagram strokes eroded or lost (0.00% is perfect).\n"
            "- **Outside Mask Change (%)**: Fraction of pixels modified outside the ground-truth watermark mask (0.00% is perfect).\n"
            "- **Time (s)**: Wall-clock execution time per page.\n\n"
            "### Results Table\n\n"
        )
        output_md.write_text(header + md_table + "\n", encoding="utf-8")
        print(f"Benchmark results written to: {output_md}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run wmclean benchmark comparison against baselines.")
    parser.add_argument("--full", action="store_true", help="Run full 12-case matrix instead of 6-case quick suite.")
    parser.add_argument("--output", type=Path, default=Path("BENCHMARKS.md"), help="Path to write Markdown summary.")
    args = parser.parse_args()

    run_benchmark_suite(quick=not args.full, output_md=args.output)
