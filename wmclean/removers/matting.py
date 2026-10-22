"""L3 Sub-Pixel Matting — Closed-form alpha matting and guided filtering.

Eliminates JPEG 4:2:0 colour-bleed halos and anti-aliased fringes
where coloured/grey watermarks touch black text or diagrams.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from wmclean.deps import get_pymatting, has_cv2_contrib
from wmclean.schemas import RemoverParams

logger = logging.getLogger(__name__)


def build_trimap(mask: np.ndarray, grey: np.ndarray) -> np.ndarray:
    """Build a 3-zone trimap (0.0=background, 0.5=unknown transition, 1.0=watermark).

    Strictly preserves dark foreground text strokes (grey < 85) by setting them to 0.0.
    """
    m_bool = mask > 0
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(m_bool.astype(np.uint8), k3, iterations=1) > 0
    eroded = cv2.erode(m_bool.astype(np.uint8), k3, iterations=1) > 0

    trimap = np.full(mask.shape, 0.5, dtype=np.float32)
    # Definite background / outside watermark
    trimap[~dilated] = 0.0
    # Definite watermark core
    trimap[eroded & (grey >= 85)] = 1.0
    # Dark text cores are definite foreground (keep clean, trimap=0.0)
    trimap[grey < 85] = 0.0

    return trimap


def guided_filter_matte(
    page_img: np.ndarray,
    mask: np.ndarray,
    params: Optional[RemoverParams] = None,
) -> np.ndarray:
    """Refine watermark boundary alpha matte and unmix foreground without halos.

    Args:
        page_img: (H, W, 3) or (H, W) uint8 image
        mask: binary watermark mask uint8
        params: optional remover parameters

    Returns:
        Cleaned image with sub-pixel halo suppression.
    """
    if mask is None or mask.sum() == 0:
        return page_img

    result = page_img.copy()
    grey = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY) if page_img.ndim == 3 else page_img.copy()
    h, w = grey.shape[:2]

    trimap = build_trimap(mask, grey)
    alpha = trimap.copy()

    # Refine alpha matte using pymatting or cv2.ximgproc.guidedFilter
    pym_mod = get_pymatting()
    use_pymatting = (pym_mod is not None) and (h <= 512 and w <= 512)

    if use_pymatting and pym_mod is not None:
        try:
            img_norm = page_img.astype(np.float64) / 255.0
            if img_norm.ndim == 2:
                img_norm = np.stack([img_norm] * 3, axis=-1)
            alpha_refined = pym_mod.estimate_alpha_cf(img_norm, trimap.astype(np.float64))
            alpha = np.clip(alpha_refined, 0.0, 1.0).astype(np.float32)
        except Exception as exc:
            logger.debug("pymatting CF failed (%s), falling back to guided filter", exc)
            use_pymatting = False

    if not use_pymatting:
        ximgproc = getattr(cv2, "ximgproc", None)
        if has_cv2_contrib() and ximgproc is not None:
            try:
                guide = page_img.astype(np.float32) / 255.0
                src = trimap.astype(np.float32)
                alpha = ximgproc.guidedFilter(guide=guide, src=src, radius=4, eps=1e-3)
                alpha = np.clip(alpha, 0.0, 1.0)
            except Exception as exc:
                logger.debug("guidedFilter failed (%s), using Gaussian fallback", exc)
                alpha = cv2.GaussianBlur(trimap, (5, 5), 1.0)
        else:
            alpha = cv2.GaussianBlur(trimap, (5, 5), 1.0)

    # Strictly protect dark text cores (<75) and non-watermark diagram hues
    alpha[grey < 75] = 0.0

    if page_img.ndim == 3:
        hsv = cv2.cvtColor(page_img, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        # Protect green diagram hues (H in [30, 95]) and blue (H in [96, 140])
        diagram_hues = (hue >= 30) & (hue <= 140) & (hsv[:, :, 1] > 40)
        alpha[diagram_hues] = 0.0

    # Unmix against estimated paper background
    paper_mask = grey >= 240
    bg_val = float(np.percentile(grey[paper_mask], 95)) if paper_mask.sum() > 100 else 255.0

    # Clean watermark region: I_clean = (I - alpha * W) / (1 - alpha)
    # Clamped to background replacement for stability
    alpha_3d = np.stack([alpha] * page_img.shape[2], axis=-1) if page_img.ndim == 3 else alpha
    img_f = page_img.astype(np.float32)

    # Where alpha > 0.05, whiten proportional to alpha
    cleaned_f = img_f + (bg_val - img_f) * alpha_3d
    result = np.clip(cleaned_f, 0, 255).astype(np.uint8)

    # Keep strictly unchanged where alpha == 0
    unchanged = alpha_3d <= 0.01
    result[unchanged] = page_img[unchanged]

    logger.info("guided_filter_matte: processed %d transition pixels", int((alpha > 0.1).sum()))
    return result


# Alias for API compatibility
closed_form_matte_unmix = guided_filter_matte
