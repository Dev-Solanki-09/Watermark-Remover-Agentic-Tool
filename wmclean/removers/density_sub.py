"""Continuous Optical-Density Template Subtraction (Zero-Smudge Radiometric Unmixing).

Converts scanned pages into Beer-Lambert Optical Density space:
  D = -log10(norm_gray / 255.0)
and subtracts the exact 20th-percentile continuous watermark density profile D_wm.

Enforces 0.0 spatial blur, 0.0 halo smudging, and complete preservation of fine
question text and diagram lines.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from wmclean.intelligence.doc_shield import parse_doc_shield
from wmclean.removers.crossing_bridge import compute_crossing_stroke_keep_mask
from wmclean.schemas import RemoverParams

logger = logging.getLogger(__name__)


def compute_optical_density(gray: np.ndarray) -> np.ndarray:
    """Convert 2D uint8 grayscale array to Beer-Lambert optical density (D >= 0.0)."""
    clipped = np.clip(gray.astype(np.float32), 1.0, 255.0)
    return -np.log10(clipped / 255.0)


def density_to_intensity(d: np.ndarray) -> np.ndarray:
    """Convert optical density array back to 2D uint8 intensity [0..255]."""
    clipped_d = np.maximum(d, 0.0)
    intensity = 255.0 * (10.0 ** (-clipped_d))
    return np.clip(intensity, 0, 255).astype(np.uint8)


def density_sub(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
    template: Optional[np.ndarray] = None,
    stack_density: Optional[list[np.ndarray]] = None,
) -> np.ndarray:
    """Perform pointwise continuous optical density subtraction on page_img.

    Args:
        page_img: (H, W, C) or (H, W) uint8 input image.
        mask: Optional binary mask.
        params: Optional RemoverParams.
        template: Optional precomputed watermark grayscale template.
        stack_density: Optional aligned optical-density stack from subdoc pages.

    Returns:
        Cleaned image array with zero spatial blurring.
    """
    is_rgb = page_img.ndim == 3 and page_img.shape[2] >= 3
    if is_rgb:
        rgb = page_img[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = page_img if page_img.ndim == 2 else page_img[:, :, 0]
        rgb = np.stack([gray, gray, gray], axis=-1)

    h, w = gray.shape[:2]

    # Compute DocShield to protect character cores, halos, and diagrams
    shield = parse_doc_shield(page_img)
    norm_gray = shield.norm_gray
    d_page = compute_optical_density(norm_gray)

    # 1. Obtain Continuous Watermark Density Template D_wm
    if stack_density is not None and len(stack_density) >= 2:
        # Use 20th percentile across aligned optical density stack
        stack_arr = np.stack(stack_density, axis=0)
        d_wm = np.percentile(stack_arr, 20, axis=0)
    elif template is not None:
        t_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY) if template.ndim == 3 else template
        if t_gray.shape[:2] != (h, w):
            t_gray = cv2.resize(t_gray, (w, h))
        # Estimate background illumination of template
        t_bg = float(np.percentile(t_gray, 95))
        t_delta = np.maximum(t_bg - t_gray.astype(np.float32), 0.0)
        d_wm = -np.log10(np.clip(1.0 - (t_delta / max(t_bg, 1.0)), 0.01, 1.0))
    elif mask is not None and mask.any():
        # Build local density template from masked area
        m_bool = mask > 0
        d_wm = np.where(m_bool, d_page, 0.0)
    else:
        # Mid-grey candidate density strictly outside diagram and text anti-aliased halos
        mid_grey = (
            (norm_gray >= 80)
            & (norm_gray <= 235)
            & (~shield.m_diagram)
            & (~shield.m_ink_halo_shield)
        )
        d_wm = np.where(mid_grey, d_page, 0.0)

    # 2. Crossing-Stroke Protection Weight
    m_wm_footprint = d_wm > 0.025
    m_crossing_keep = compute_crossing_stroke_keep_mask(
        norm_gray=norm_gray,
        m_core=shield.m_core,
        m_wm_stroke=m_wm_footprint,
    )

    # Scale: 0.0 where crossing keep, text core, or text halo, 1.0 inside watermark
    scale = np.where(
        m_crossing_keep | shield.m_core | shield.m_ink_halo_shield,
        0.0,
        np.where(m_wm_footprint, 1.0, 0.0),
    )

    # 3. Pointwise Subtraction (Zero Blur, Zero Inpainting)
    d_clean = np.maximum(d_page - scale * d_wm, 0.0)
    gray_clean = density_to_intensity(d_clean)

    # Apply background restoration
    if shield.bg is not None:
        bg_float = shield.bg.astype(np.float32)
        gray_clean = np.clip(
            gray_clean.astype(np.float32) * (bg_float / 255.0),
            0,
            255,
        ).astype(np.uint8)

    # 4. Construct Output
    if is_rgb:
        result = rgb.copy()
        # Modulate RGB channels pointwise by ratio of cleaned gray to original gray
        ratio = np.where(
            gray > 0,
            gray_clean.astype(np.float32) / np.maximum(gray.astype(np.float32), 1.0),
            1.0,
        )
        ratio = np.clip(ratio, 0.0, 3.0)
        for c in range(3):
            clean_c = np.clip(rgb[:, :, c].astype(np.float32) * ratio, 0, 255).astype(np.uint8)
            # Only update pixels where watermark was present
            result[:, :, c] = np.where(m_wm_footprint & (~shield.m_diagram), clean_c, rgb[:, :, c])
    else:
        result = np.where(m_wm_footprint & (~shield.m_diagram), gray_clean, gray)

    # 5. Enforce Self-Healing Text-Core & Diagram Lock
    result = shield.apply_text_core_lock(result, page_img, lock_halo=True, wm_mask=m_wm_footprint)

    logger.debug("density_sub: cleaned %d watermark footprint pixels", int(m_wm_footprint.sum()))
    return result
