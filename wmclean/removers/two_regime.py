"""L3 Tier 1 Two-Regime Watermark Remover.

Applies:
- Regime 1: Over blank paper pixels (~m_author), fills directly with local paper background color (residual -> 0.0).
- Regime 2: Over authentic ink / text halo, unmixes watermark via linear alpha inversion J = (I - αW) / (1 - α).
- Diagram Lock: Strictly protects diagram strokes and chemical rings from inpainting.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from wmclean.intelligence.doc_shield import parse_doc_shield
from wmclean.schemas import RemoverParams

logger = logging.getLogger(__name__)


def two_regime_fill(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
    template: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Execute Two-Regime watermark removal on page image array.

    Args:
        page_img: Page image (H, W, 3) or (H, W) uint8.
        mask: Optional binary watermark mask (H, W) uint8.
        params: Optional RemoverParams.
        template: Optional watermark template array (H, W) or (H, W, 3).

    Returns:
        Cleaned image (H, W, 3) or (H, W) uint8.
    """
    if page_img is None or page_img.size == 0:
        return page_img

    params = params or RemoverParams()
    h, w = page_img.shape[:2]
    is_color = page_img.ndim == 3 and page_img.shape[2] == 3

    # Greyscale representation for structural analysis
    gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY) if is_color else page_img.copy()

    # 1. Estimate paper background luminance and color
    paper_val = int(np.percentile(gray, 95))
    paper_mask = gray >= max(180, paper_val - 6)

    if is_color:
        if np.any(paper_mask):
            paper_bgr = np.median(page_img[paper_mask], axis=0).astype(np.uint8)
        else:
            paper_bgr = np.array([paper_val, paper_val, paper_val], dtype=np.uint8)
        paper_bgr = np.clip(paper_bgr, max(180, paper_val - 10), 255).astype(np.uint8)
    else:
        paper_bgr = paper_val

    # 2. Parse DocShield protection layers
    shield = parse_doc_shield(page_img, template_mask=mask)
    m_author = shield.m_diagram | shield.m_ink_halo_shield | shield.m_core

    # Also protect thin text strokes identified by thresholding
    fg_low = gray < 135
    m_author = m_author | fg_low

    # 3. Check for chromatic watermark (e.g. red/pink stamp or blue seal)
    if is_color:
        hsv = cv2.cvtColor(page_img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        color_wm = (sat > 30) & (~shield.m_diagram)
        if np.count_nonzero(color_wm) > 300:
            lab = cv2.cvtColor(page_img, cv2.COLOR_BGR2LAB).astype(np.float32)
            # Desaturate watermark color in non-diagram regions
            lab[color_wm, 1] = 128.0
            lab[color_wm, 2] = 128.0
            # Lighten background pixels to paper luminance
            bg_chroma = color_wm & (lab[:, :, 0] >= 110.0)
            lab[bg_chroma, 0] = float(paper_val)
            page_img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
            gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)

    # 4. Identify candidate watermark pixels
    if template is not None:
        t_h, t_w = template.shape[:2]
        tmpl = cv2.resize(template, (w, h)) if (t_h, t_w) != (h, w) else template
        t_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY) if tmpl.ndim == 3 else tmpl
        bg_t = float(np.percentile(t_gray, 95))
        delta_t = np.maximum(0.0, bg_t - t_gray.astype(np.float32))
        wm_cand = delta_t > 12.0
    elif mask is not None and np.sum(mask > 0) > 50:
        wm_cand = mask > 0
    else:
        # Dynamic watermark luminance detection on paper
        wm_cand = (gray >= 140) & (gray <= max(140, paper_val - 8)) & (~m_author)

    # 5. Regime 1: Over blank paper, fill directly with local paper color
    wm_paper = wm_cand & (~m_author)
    result = page_img.copy()

    if is_color:
        for c in range(3):
            result[:, :, c] = np.where(wm_paper, paper_bgr[c], result[:, :, c])
    else:
        result = np.where(wm_paper, paper_bgr, result)

    # 6. Regime 2: Over authentic ink / text halo, unmix watermark if template exists
    if template is not None:
        alpha = min(max(params.alpha_clamp, 0.15), 0.75)
        wm_overlap = wm_cand & m_author & (~shield.m_diagram)
        if np.any(wm_overlap):
            W = tmpl.astype(np.float32)
            img_f = page_img.astype(np.float32)
            denom = max(1.0 - alpha, 0.10)
            J = np.clip((img_f - alpha * W) / denom, 0, 255).astype(np.uint8)
            if is_color:
                for c in range(3):
                    result[:, :, c] = np.where(wm_overlap, J[:, :, c], result[:, :, c])
            else:
                result = np.where(wm_overlap, J, result)

    # 7. Strictly enforce text-core and diagram stroke lock
    result = shield.apply_text_core_lock(result, page_img, lock_halo=True, wm_mask=wm_cand)

    logger.debug(
        "two_regime_fill: processed %d paper watermark pixels, locked %d author ink pixels",
        int(np.sum(wm_paper)),
        int(np.sum(m_author)),
    )
    return result
