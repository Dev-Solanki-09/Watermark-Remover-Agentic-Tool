"""Intra-Page Bilingual Twin-Column Diagram & Formula Cloning (Zero-Smudge Authentic Pixel Restoration).

In two-column bilingual exam papers and textbooks (e.g. English left [0..0.5W], Hindi right [0.5W..W]),
diagrams, chemical structures, physics coordinate graphs, reaction arrows, and math equations (1)-(4)
are printed twice on the exact same page.

When a watermark occludes a diagram in one column, this module uses OpenCV's masked template
matcher (cv2.matchTemplate with mask) to locate the identical twin diagram in the opposite column
and transplants authentic clean author-drawn pixels with 0.0 synthetic hallucination.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from wmclean.intelligence.doc_shield import parse_doc_shield
from wmclean.schemas import RemoverParams

logger = logging.getLogger(__name__)


def detect_twin_columns(gray: np.ndarray) -> Optional[int]:
    """Detect vertical column gutter x-coordinate near page midpoint (0.5W +/- 0.08W)."""
    h, w = gray.shape[:2]
    mid_min = int(w * 0.42)
    mid_max = int(w * 0.58)
    if mid_max <= mid_min:
        return None

    # Column gutter is either a vertical black divider line or a continuous vertical whitespace band
    mid_strip = gray[:, mid_min:mid_max]
    # Check vertical whitespace column (mean brightness > 240)
    col_means = mid_strip.mean(axis=0)
    best_col = mid_min + int(np.argmax(col_means))
    if float(np.max(col_means)) > 230:
        return best_col

    # Check for a thin vertical dividing line (mean brightness < 120)
    if float(np.min(col_means)) < 130:
        return mid_min + int(np.argmin(col_means))

    # Default to exact geometric midpoint if layout is two-column width
    if w >= 800:
        return w // 2
    return None


def bilingual_twin_clone(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
    roi_bbox: Optional[tuple[int, int, int, int]] = None,
    **kwargs,
) -> np.ndarray:
    """Clone clean twin diagrams/formulas across bilingual columns.

    Args:
        page_img: (H, W, C) or (H, W) uint8 input image.
        mask: Binary mask of watermark pixels.
        params: Optional RemoverParams.
        roi_bbox: Optional (x1, y1, x2, y2) of the watermarked diagram region.

    Returns:
        Page image array with watermarked diagram pixels restored from twin column.
    """
    is_rgb = page_img.ndim == 3 and page_img.shape[2] >= 3
    gray = (
        cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY)
        if is_rgb
        else (page_img if page_img.ndim == 2 else page_img[:, :, 0])
    )
    h, w = gray.shape[:2]

    gutter_x = detect_twin_columns(gray)
    if gutter_x is None:
        logger.debug("bilingual_twin_clone: no twin column gutter detected")
        return page_img

    shield = parse_doc_shield(page_img)
    norm_gray = shield.norm_gray

    # Determine watermark mask
    if mask is not None and mask.any():
        wm_mask = mask > 0
    else:
        wm_mask = (norm_gray >= 80) & (norm_gray <= 235) & (~shield.m_core) & (~shield.m_diagram)

    # Determine ROI to clone: if roi_bbox provided, use it; else find candidate diagram ROIs
    rois: list[tuple[int, int, int, int]] = []
    if roi_bbox is not None:
        rx1, ry1, rx2, ry2 = roi_bbox
        # If roi_bbox is a large zone (>50% w and h), search diagram CCs within it
        if (rx2 - rx1) > 0.5 * w and (ry2 - ry1) > 0.5 * h:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                shield.m_diagram.astype(np.uint8), connectivity=8
            )
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > 40:
                    bx = stats[i, cv2.CC_STAT_LEFT]
                    by = stats[i, cv2.CC_STAT_TOP]
                    bw = stats[i, cv2.CC_STAT_WIDTH]
                    bh = stats[i, cv2.CC_STAT_HEIGHT]
                    if bx >= rx1 and by >= ry1 and (bx + bw) <= rx2 and (by + bh) <= ry2:
                        cc_mask = labels == i
                        if (cc_mask & wm_mask).any():
                            rois.append(
                                (
                                    max(0, bx - 15),
                                    max(0, by - 15),
                                    min(w, bx + bw + 15),
                                    min(h, by + bh + 15),
                                )
                            )
        else:
            rois.append((max(0, rx1), max(0, ry1), min(w, rx2), min(h, ry2)))
    else:
        # Find diagram connected components overlapping watermark
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            shield.m_diagram.astype(np.uint8), connectivity=8
        )
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 40:
                cc_mask = labels == i
                if (cc_mask & wm_mask).any():
                    bx = stats[i, cv2.CC_STAT_LEFT]
                    by = stats[i, cv2.CC_STAT_TOP]
                    bw = stats[i, cv2.CC_STAT_WIDTH]
                    bh = stats[i, cv2.CC_STAT_HEIGHT]
                    # Pad by 15 px
                    rois.append(
                        (
                            max(0, bx - 15),
                            max(0, by - 15),
                            min(w, bx + bw + 15),
                            min(h, by + bh + 15),
                        )
                    )

    if not rois:
        return page_img

    out_img = page_img.copy()

    for x1, y1, x2, y2 in rois:
        rw, rh = x2 - x1, y2 - y1
        if rw < 20 or rh < 20:
            continue

        target_crop = gray[y1:y2, x1:x2]
        crop_wm = wm_mask[y1:y2, x1:x2]
        clean_mask = (~crop_wm).astype(np.uint8) * 255

        # If too little clean content to match, skip
        if clean_mask.mean() < 30:
            continue

        # Determine source twin column search zone
        is_left = ((x1 + x2) // 2) < gutter_x
        if is_left:
            # Search right column
            twin_x1 = gutter_x
            twin_x2 = min(w, gutter_x + (gutter_x - 0))
        else:
            # Search left column
            twin_x1 = 0
            twin_x2 = gutter_x

        search_y1 = max(0, y1 - 60)
        search_y2 = min(h, y2 + 60)
        search_strip = gray[search_y1:search_y2, twin_x1:twin_x2]
        sw, sh = search_strip.shape[1], search_strip.shape[0]

        if sw < rw or sh < rh:
            continue

        try:
            # Masked template match via OpenCV
            res = cv2.matchTemplate(search_strip, target_crop, cv2.TM_CCORR_NORMED, mask=clean_mask)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val >= 0.88:
                mx, my = max_loc
                # Absolute coordinates in page of matching twin patch
                abs_twin_x = twin_x1 + mx
                abs_twin_y = search_y1 + my

                twin_crop = page_img[abs_twin_y : abs_twin_y + rh, abs_twin_x : abs_twin_x + rw]
                twin_wm = wm_mask[abs_twin_y : abs_twin_y + rh, abs_twin_x : abs_twin_x + rw]

                # Only transplant pixels where target is watermarked AND twin is clean!
                transplant_pixels = crop_wm & (~twin_wm)

                if transplant_pixels.any():
                    if is_rgb:
                        for c in range(3):
                            out_img[y1:y2, x1:x2, c] = np.where(
                                transplant_pixels,
                                twin_crop[:, :, c],
                                out_img[y1:y2, x1:x2, c],
                            )
                    else:
                        out_img[y1:y2, x1:x2] = np.where(
                            transplant_pixels,
                            twin_crop,
                            out_img[y1:y2, x1:x2],
                        )

                    logger.info(
                        "bilingual_twin_clone: cloned %d authentic twin pixels (corr=%.3f) for ROI [%d,%d,%d,%d]",
                        int(transplant_pixels.sum()),
                        max_val,
                        x1,
                        y1,
                        x2,
                        y2,
                    )
        except Exception as exc:
            logger.debug("bilingual_twin_clone matching error for ROI: %s", exc)

    out_img = shield.apply_text_core_lock(out_img, page_img, lock_halo=True, wm_mask=wm_mask)
    return out_img
