"""Benchmark evaluation metrics and runner.

Measures:
1. Residual error on paper pixels (mean absolute error).
2. Ink loss on foreground strokes (false-negative rate on text/diagrams).
3. Modification outside ground-truth watermark mask.
4. Execution time per page.
5. Peak memory usage (via psutil).
6. Optional OCR word/character fidelity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
import psutil


@dataclass
class EvaluationResult:
    method_name: str
    fixture_id: str
    residual_paper: float        # Mean MAE on paper pixels in watermark zone (lower is better, 0.0 is perfect)
    ink_loss_rate: float         # Fraction of foreground ink pixels lost/eroded (lower is better, 0.0 is perfect)
    outside_mask_change_rate: float # Fraction of pixels altered outside watermark mask (lower is better, 0.0 is perfect)
    duration_sec: float          # Runtime in seconds
    peak_ram_mb: float           # Peak memory during execution
    ocr_cer: Optional[float]     # Character Error Rate (if OCR available)
    extra: Dict[str, Any]


def evaluate_page_cleaning(
    clean_gt_bgr: np.ndarray,
    watermarked_bgr: np.ndarray,
    cleaned_bgr: np.ndarray,
    wm_mask_gt: np.ndarray,
    fg_mask_gt: np.ndarray,
    method_name: str,
    fixture_id: str,
    duration_sec: float = 0.0,
    peak_ram_mb: float = 0.0,
) -> EvaluationResult:
    """Evaluate a single cleaned page image against ground-truth clean and masks."""
    h, w = clean_gt_bgr.shape[:2]

    # Ensure masks are boolean
    wm_mask = wm_mask_gt.astype(bool)
    fg_mask = fg_mask_gt.astype(bool)

    # 1. Residual error on paper pixels (non-foreground pixels where watermark existed)
    paper_wm_pixels = wm_mask & (~fg_mask)
    if np.any(paper_wm_pixels):
        diff = np.abs(cleaned_bgr.astype(np.float32) - clean_gt_bgr.astype(np.float32))
        res_paper = float(np.mean(diff[paper_wm_pixels]))
    else:
        res_paper = 0.0

    # 2. Foreground ink loss:
    # A foreground pixel is considered lost if it was dark in ground truth (< 180)
    # but became significantly lightened (> clean_gt + 30) in the cleaned output.
    if np.any(fg_mask):
        fg_gt_gray = cv2.cvtColor(clean_gt_bgr, cv2.COLOR_BGR2GRAY)
        fg_out_gray = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
        eroded = fg_mask & (fg_out_gray > (fg_gt_gray.astype(np.int16) + 30))
        ink_loss = float(np.count_nonzero(eroded) / np.count_nonzero(fg_mask))
    else:
        ink_loss = 0.0

    # 3. Share of pixels changed outside the watermark mask
    outside_wm = ~wm_mask
    if np.any(outside_wm):
        diff_outside = np.max(np.abs(cleaned_bgr.astype(np.int16) - watermarked_bgr.astype(np.int16)), axis=2)
        # Any change > 4 intensity levels is considered modified
        changed = outside_wm & (diff_outside > 4)
        outside_change_rate = float(np.count_nonzero(changed) / np.count_nonzero(outside_wm))
    else:
        outside_change_rate = 0.0

    # 4. Optional OCR evaluation if rapidocr or pytesseract is installed
    ocr_cer = None
    try:
        from wmclean.deps import is_rapidocr_available
        if is_rapidocr_available():
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR()
            txt_gt, _ = engine(cv2.cvtColor(clean_gt_bgr, cv2.COLOR_BGR2RGB))
            txt_out, _ = engine(cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB))
            str_gt = "".join([line[1] for line in txt_gt]) if txt_gt else ""
            str_out = "".join([line[1] for line in txt_out]) if txt_out else ""
            if len(str_gt) > 0:
                import difflib
                matcher = difflib.SequenceMatcher(None, str_gt, str_out)
                ocr_cer = 1.0 - matcher.ratio()
            else:
                ocr_cer = 0.0
    except Exception:
        ocr_cer = None

    return EvaluationResult(
        method_name=method_name,
        fixture_id=fixture_id,
        residual_paper=round(res_paper, 3),
        ink_loss_rate=round(ink_loss, 4),
        outside_mask_change_rate=round(outside_change_rate, 4),
        duration_sec=round(duration_sec, 3),
        peak_ram_mb=round(peak_ram_mb, 1),
        ocr_cer=round(ocr_cer, 4) if ocr_cer is not None else None,
        extra={},
    )


def profile_execution(clean_fn: Callable[[], np.ndarray]) -> tuple[np.ndarray, float, float]:
    """Execute clean_fn while tracking execution time and peak memory."""
    process = psutil.Process()
    ram_before = process.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()
    result = clean_fn()
    duration = time.perf_counter() - t0

    ram_after = process.memory_info().rss / (1024 * 1024)
    peak_ram = max(ram_before, ram_after)

    return result, duration, peak_ram
