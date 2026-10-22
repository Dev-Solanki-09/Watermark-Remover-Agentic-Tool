"""Pointwise Hue-Selective & Monotonic LUT Unmixing (Zero-Smudge Tone Engine).

Provides:
1. Pointwise Redness / Chroma Unmixing: Lifts pink/red paper to local paper background
   while preserving the exact native optical density (255 - R) of black text strokes.
   Strictly protects green/blue diagram hues!
2. Monotonic S-Curve Tone Curve via PchipInterpolator + cv2.LUT:
   Enforces 0.0 overshoot, 0.0 ringing, and 0.0 spatial blur.
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
    wm_floor: int = 160,
    paper_white: int = 255,
) -> np.ndarray:
    """Build a strictly monotonic, overshoot-free 256-byte LUT using PchipInterpolator."""
    x = [0, black_point, wm_floor, 255]
    y = [0, black_point, paper_white, paper_white]
    pchip = PchipInterpolator(x, y)
    lut = pchip(np.arange(256)).clip(0, 255).astype(np.uint8)
    return lut


def pointwise_hue_and_lut(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
    **kwargs,
) -> np.ndarray:
    """Pointwise radiometric unmixing for coloured logos and mid-grey watermarks.

    Args:
        page_img: (H, W, C) or (H, W) uint8 input image.
        mask: Optional candidate watermark mask.
        params: Optional RemoverParams.

    Returns:
        Cleaned image array with zero spatial blurring.
    """
    is_rgb = page_img.ndim == 3 and page_img.shape[2] >= 3
    shield = parse_doc_shield(page_img)
    bg = shield.bg
    norm_gray = shield.norm_gray
    h, w = norm_gray.shape[:2]

    out_img = page_img.copy()

    # -----------------------------------------------------------------------
    # Part 1: Red/Coloured Logo Pointwise Unmix
    # -----------------------------------------------------------------------
    if is_rgb:
        r = out_img[:, :, 0].astype(np.float32)
        g = out_img[:, :, 1].astype(np.float32)
        b = out_img[:, :, 2].astype(np.float32)

        # Excess redness over green/blue
        redness = r - np.maximum(g, b)

        # Only on pixels with true warm watermark chroma, strictly excluding diagrams
        red_wm_mask = (redness > 10.0) & (r > 120.0) & (g <= r) & (b <= r) & (~shield.m_diagram)

        if mask is not None and mask.any():
            # If mask provided, intersect to keep surgical precision
            red_wm_mask &= mask > 0

        if np.sum(red_wm_mask) > 30:
            bg_f = bg.astype(np.float32)
            # Smooth transition: preserve dark text (r < 95), map red chroma to local paper bg
            t = np.clip((r - 95.0) / 25.0, 0.0, 1.0)
            clean_luma = np.clip((1.0 - t) * r + t * bg_f, 0, 255).astype(np.uint8)

            for c in range(3):
                out_img[:, :, c] = np.where(red_wm_mask, clean_luma, out_img[:, :, c])

            logger.info(
                "pointwise_hue_and_lut: unmixed %d red/pink watermark pixels",
                int(red_wm_mask.sum()),
            )

    # -----------------------------------------------------------------------
    # Part 2: Monotonic S-Curve LUT on Mid-Grey Watermarks
    # -----------------------------------------------------------------------
    # Detect if mid-grey watermark exists in regions outside text halo and diagram
    cur_gray = cv2.cvtColor(out_img, cv2.COLOR_RGB2GRAY) if is_rgb else out_img.copy()
    grey_wm_mask = (
        (cur_gray >= 110) & (cur_gray <= 235) & (~shield.m_ink_halo_shield) & (~shield.m_diagram)
    )
    if mask is not None and mask.any():
        grey_wm_mask &= mask > 0

    if np.sum(grey_wm_mask) > 100:
        # Determine average watermark luma floor
        wm_samples = cur_gray[grey_wm_mask]
        wm_floor = int(np.percentile(wm_samples, 25))
        wm_floor = max(110, min(220, wm_floor))

        lut = build_monotonic_lut(black_point=88, wm_floor=wm_floor, paper_white=255)
        lut_applied = cv2.LUT(cur_gray, lut)

        if is_rgb:
            for c in range(3):
                out_img[:, :, c] = np.where(grey_wm_mask, lut_applied, out_img[:, :, c])
        else:
            out_img = np.where(grey_wm_mask, lut_applied, out_img)

        logger.info(
            "pointwise_hue_and_lut: applied monotonic LUT on %d mid-grey pixels",
            int(grey_wm_mask.sum()),
        )

    # Enforce self-healing text-core lock
    out_img = shield.apply_text_core_lock(out_img, page_img, lock_halo=True)
    return out_img
