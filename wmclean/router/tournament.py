"""Fast Patch-Level Remover Tournament.

Evaluates candidate removers on a fast 256x256 representative crop to pick the
optimal method by residual drop and foreground edge preservation in < 15ms,
replacing static sequential guesswork with empirical selection.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from wmclean.intelligence.doc_shield import parse_doc_shield
from wmclean.removers.raster_ops import (
    raster_alpha_inv,
    raster_color_unmix,
    raster_template_whiten,
)
from wmclean.removers.two_regime import two_regime_fill
from wmclean.schemas import RemoverMethod, RemoverParams

logger = logging.getLogger(__name__)


def extract_tournament_patch(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    patch_size: int = 256,
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int, int]]:
    """Extract a representative 256x256 patch containing both watermark and text."""
    h, w = page_img.shape[:2]
    ps_h = min(patch_size, h)
    ps_w = min(patch_size, w)

    # If mask is provided, center on highest watermark density
    if mask is not None and np.sum(mask > 0) > 100:
        ys, xs = np.where(mask > 0)
        cy = int(np.median(ys))
        cx = int(np.median(xs))
    else:
        # Default to center of page
        cy, cx = h // 2, w // 2

    y1 = max(0, min(cy - ps_h // 2, h - ps_h))
    x1 = max(0, min(cx - ps_w // 2, w - ps_w))
    y2 = y1 + ps_h
    x2 = x1 + ps_w

    patch_img = page_img[y1:y2, x1:x2].copy()
    patch_mask = mask[y1:y2, x1:x2].copy() if mask is not None else None
    return patch_img, patch_mask, (x1, y1, x2, y2)


def evaluate_patch_quality(
    orig_patch: np.ndarray,
    cleaned_patch: np.ndarray,
    shield_m_author: np.ndarray,
) -> float:
    """Evaluate quality score for a cleaned patch (lower is better).

    Score = Residual on paper + 150.0 * Foreground Ink Loss
    """
    orig_gray = cv2.cvtColor(orig_patch, cv2.COLOR_BGR2GRAY) if orig_patch.ndim == 3 else orig_patch
    clean_gray = cv2.cvtColor(cleaned_patch, cv2.COLOR_BGR2GRAY) if cleaned_patch.ndim == 3 else cleaned_patch

    paper_val = float(np.percentile(orig_gray, 95))
    clean_paper_val = float(np.percentile(clean_gray, 95))

    # Watermark candidate pixels on paper
    eval_paper = (~shield_m_author) & (orig_gray < (paper_val - 10))

    if np.sum(eval_paper) > 20:
        residual = float(np.mean(np.abs(clean_gray[eval_paper].astype(np.float32) - clean_paper_val))) / 255.0
    else:
        residual = 0.0

    # Foreground ink loss: check dark strokes in orig vs clean
    orig_fg = orig_gray < 125
    total_fg = int(np.sum(orig_fg))
    if total_fg > 20:
        lost_fg = orig_fg & (clean_gray > 180)
        ink_loss = float(np.sum(lost_fg)) / total_fg
    else:
        ink_loss = 0.0

    # Cost penalizes ink destruction heavily
    return residual + 150.0 * ink_loss


def run_remover_tournament(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    template: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
) -> Tuple[RemoverMethod, Callable, float]:
    """Run fast tournament on a sample patch to select the best remover in < 15ms.

    Returns:
        (winning_method_enum, winning_callable, best_score)
    """
    t0 = time.perf_counter()
    patch_img, patch_mask, (x1, y1, x2, y2) = extract_tournament_patch(page_img, mask=mask)
    patch_tmpl = template[y1:y2, x1:x2].copy() if template is not None else None

    shield = parse_doc_shield(patch_img, template_mask=patch_mask)
    m_author = shield.m_diagram | shield.m_ink_halo_shield | shield.m_core

    candidates: list[Tuple[RemoverMethod, Callable]] = [
        (RemoverMethod.TWO_REGIME_FILL, lambda img, m, p: two_regime_fill(img, mask=m, params=p, template=patch_tmpl)),
        (RemoverMethod.RASTER_TEMPLATE_WHITEN, lambda img, m, p: raster_template_whiten(img, mask=m, params=p, template=patch_tmpl)),
    ]

    # Add color unmixing if image has significant color
    if patch_img.ndim == 3 and patch_img.shape[2] == 3:
        hsv = cv2.cvtColor(patch_img, cv2.COLOR_BGR2HSV)
        if np.count_nonzero(hsv[:, :, 1] > 35) > 100:
            candidates.append(
                (RemoverMethod.RASTER_COLOR_UNMIX, lambda img, m, p: raster_color_unmix(img, m or np.zeros(img.shape[:2], dtype=np.uint8), p or RemoverParams()))
            )

    # Add alpha inversion if template is provided
    if template is not None:
        candidates.append(
            (RemoverMethod.RASTER_ALPHA_INV, lambda img, m, p: raster_alpha_inv(img, patch_tmpl, m or np.zeros(img.shape[:2], dtype=np.uint8), p or RemoverParams()))
        )

    best_method = candidates[0][0]
    best_func = candidates[0][1]
    best_score = float("inf")

    params = params or RemoverParams()

    for method, func in candidates:
        try:
            cand_cleaned = func(patch_img.copy(), patch_mask, params)
            score = evaluate_patch_quality(patch_img, cand_cleaned, m_author)
            if score < best_score:
                best_score = score
                best_method = method
                best_func = func
        except Exception as e:
            logger.debug("Tournament candidate %s failed: %s", method.value, e)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "Tournament completed in %.1fms: winner=%s (score=%.4f)",
        elapsed_ms,
        best_method.value,
        best_score,
    )
    return best_method, best_func, best_score
