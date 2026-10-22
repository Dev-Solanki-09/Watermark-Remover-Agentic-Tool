"""Pointwise Monotonic S-Curve LUT Unmixing via PchipInterpolator + cv2.LUT.

Builds a strictly monotonic, overshoot-free 256-byte lookup table
that maps mid-grey watermark ink smoothly to paper white while keeping black ink (< 88)
and character serifs completely untouched on the identity curve.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from wmclean.intelligence.doc_shield import parse_doc_shield
from wmclean.schemas import RemoverParams

logger = logging.getLogger(__name__)


def build_monotonic_lut(
    black_point: int = 88,
    wm_floor: int = 150,
    paper_white: int = 255,
) -> np.ndarray:
    """Build a strictly monotonic, overshoot-free 256-byte tone curve LUT."""
    x = [0, black_point, wm_floor, 255]
    y = [0, black_point, paper_white, paper_white]
    pchip = PchipInterpolator(x, y)
    lut = pchip(np.arange(256)).clip(0, 255).astype(np.uint8)
    return lut


def apply_monotonic_lut_unmix(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
    **kwargs,
) -> np.ndarray:
    """Apply pointwise monotonic tone curve to unshielded watermark pixels.

    Because cv2.LUT is strictly pointwise in hardware SIMD, spatial blur is mathematically 0.0.
    """
    is_rgb = page_img.ndim == 3 and page_img.shape[2] >= 3
    shield = parse_doc_shield(page_img)
    norm_gray = shield.norm_gray
    h, w = norm_gray.shape[:2]

    # Target unshielded watermark candidate pixels
    if mask is not None and mask.any():
        wm_cand = (mask > 0) & (~shield.m_ink_halo_shield) & (~shield.m_diagram)
    else:
        wm_cand = (
            (norm_gray >= 110)
            & (norm_gray <= 235)
            & (~shield.m_ink_halo_shield)
            & (~shield.m_diagram)
        )

    if np.sum(wm_cand) < 50:
        return page_img

    wm_samples = norm_gray[wm_cand]
    wm_floor = int(np.percentile(wm_samples, 20))
    wm_floor = max(95, min(220, wm_floor))

    lut = build_monotonic_lut(black_point=88, wm_floor=wm_floor, paper_white=255)
    lut_gray = cv2.LUT(norm_gray, lut)

    out = page_img.copy()
    if is_rgb:
        for c in range(3):
            out[:, :, c] = np.where(wm_cand, lut_gray, out[:, :, c])
    else:
        out = np.where(wm_cand, lut_gray, out)

    # Enforce text-core lock
    out = shield.apply_text_core_lock(out, page_img, lock_halo=True)
    logger.info("apply_monotonic_lut_unmix: mapped %d mid-grey pixels via LUT", int(wm_cand.sum()))
    return out


lut_unmix = apply_monotonic_lut_unmix
