"""Tier 3 Strictly Clamped Inpainting for Opaque Zones & Photographic Stamps.

Enforces the strict invariant:
- Inpainting operations (Telea, Navier-Stokes, or Fourier) are strictly confined
  to the dilated watermark mask M_dilated.
- 100% of pixels outside M_dilated remain bitwise identical to the original input.
- Authentic text/diagram cores (DocShield) are locked against erosion.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from wmclean.intelligence.doc_shield import DocShieldResult, parse_doc_shield

logger = logging.getLogger(__name__)


def clamped_inpaint(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "telea",
    dilation_px: int = 2,
    shield_core: bool = True,
    inpaint_radius: int = 3,
) -> np.ndarray:
    """Inpaint opaque watermark zones strictly clamped to the mask.

    Args:
        img: RGB or greyscale image array (H, W, C) or (H, W).
        mask: Boolean or uint8 mask of watermark pixels to inpaint.
              If None, estimates the watermark mask via thresholding/DocShield.
        method: "telea" (Fast Marching) or "ns" (Navier-Stokes fluid dynamics).
        dilation_px: Number of pixels to dilate the watermark mask to capture antialiasing.
        shield_core: If True, preserves authentic text cores from being inpainted.
        inpaint_radius: Inpainting neighborhood radius in pixels.

    Returns:
        Cleaned image array with 100% bitwise pixel invariance outside M_dilated.
    """
    is_rgb = img.ndim == 3 and img.shape[2] >= 3
    orig_copy = img.copy()

    h, w = img.shape[:2]

    # Convert to working RGB format
    if is_rgb:
        work_img = img[:, :, :3].copy()
    else:
        gray = img if img.ndim == 2 else img[:, :, 0]
        work_img = np.stack([gray, gray, gray], axis=2)

    # 1. Determine base watermark mask
    if mask is None:
        # Fallback automatic detection: saturation or intensity difference
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (51, 51), 0)
        diff = cv2.absdiff(gray, blur)
        _, raw_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        base_mask_bool = raw_mask > 0
    else:
        if mask.ndim == 3:
            base_mask_bool = np.any(mask > 0, axis=2)
        else:
            base_mask_bool = mask > 0

    if not np.any(base_mask_bool):
        return orig_copy

    # 2. Compute M_dilated
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    m_dilated = cv2.dilate(base_mask_bool.astype(np.uint8), kernel) > 0

    # 3. Apply DocShield core-locking to protect authentic text/diagrams
    active_inpaint_mask = m_dilated.copy()
    doc_shield: Optional[DocShieldResult] = None
    if shield_core:
        try:
            doc_shield = parse_doc_shield(work_img, template_mask=base_mask_bool.astype(np.uint8))
            # Exclude authentic character cores and diagram strokes outside the watermark from being inpainted
            protected = (doc_shield.m_core | doc_shield.m_diagram) & (~base_mask_bool)
            active_inpaint_mask = active_inpaint_mask & (~protected)
        except Exception as exc:
            logger.debug("DocShield core parsing skipped in clamped_inpaint: %s", exc)

    if not np.any(active_inpaint_mask):
        return orig_copy

    # 4. Perform OpenCV Inpainting
    inpaint_flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    inpainted = cv2.inpaint(
        work_img,
        active_inpaint_mask.astype(np.uint8) * 255,
        inpaintRadius=inpaint_radius,
        flags=inpaint_flag,
    )

    # 5. Composite results and enforce strict bitwise invariant outside M_dilated
    if is_rgb:
        result = orig_copy.copy()
        for c in range(3):
            res_c = result[:, :, c]
            inp_c = inpainted[:, :, c]
            res_c[active_inpaint_mask] = inp_c[active_inpaint_mask]
            result[:, :, c] = res_c
            # Hard invariant: Outside active inpainting, keep 100% bitwise original
            result[~active_inpaint_mask, c] = orig_copy[~active_inpaint_mask, c]
    else:
        gray_inp = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        result = orig_copy.copy()
        result[active_inpaint_mask] = gray_inp[active_inpaint_mask]
        result[~active_inpaint_mask] = orig_copy[~active_inpaint_mask]

    # Re-apply text core lock if shield was available
    if shield_core and doc_shield is not None:
        result = doc_shield.apply_text_core_lock(result, orig_copy, lock_halo=True)

    # Final hard clamp invariant check
    result[~m_dilated] = orig_copy[~m_dilated]

    return result
