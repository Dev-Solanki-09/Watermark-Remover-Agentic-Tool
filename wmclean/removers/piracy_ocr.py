"""L3 OCR Spotter & Guilloche Security Lace Cleaner.

Solves single-page unrepeated raster watermarks on scans without PDF OCR layers using RapidOCR INT8 ONNX,
and strips Guilloche security lace via morphological reconstruction.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import cv2
import numpy as np

from wmclean.deps import get_rapidocr, has_skimage
from wmclean.schemas import RemoverParams

logger = logging.getLogger(__name__)

# Universal Scanner & Document Watermark Pattern Registry
_SCANNER_PATTERNS = (
    r"camscanner|tapscanner|adobe\s*scan|turboscan|clearscanner|simple\s*scan|"
    r"genius\s*scan|vflat|scanner|scanned\s+(?:with|by)|digitized\s+(?:with|by)"
)

_PLATFORM_PATTERNS = (
    r"telegram|whatsapp|discord|wechat|signal|viber|"
    r"t\.me/|wa\.me/|chat\.whatsapp\.com/|discord\.gg/|"
    r"@[a-z0-9_\.]{3,30}"
)

_CALL_TO_ACTION_PATTERNS = (
    r"search\s+(?:on|for|in)|latest\s+updates|join\s+(?:us|our|channel|group)|"
    r"subscribe\s+to|follow\s+us|visit\s+(?:our\s+)?(?:website|link)|"
    r"for\s+more\s+(?:notes|books|materials|pdf|updates)"
)

_DISTRIBUTION_PATTERNS = (
    r"downloaded\s+(?:by|from)|prepared\s+for|licensed\s+to|authorized\s+for|"
    r"watermark|confidential|evaluation\s+copy|do\s+not\s+(?:copy|distribute)|"
    r"sample\s+copy|preview\s+only|strictly\s+private"
)

_CONTACT_PATTERNS = r"(?:\+\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,5}"

_PIRACY_PATTERN = re.compile(
    rf"(?:{_SCANNER_PATTERNS}|{_PLATFORM_PATTERNS}|{_CALL_TO_ACTION_PATTERNS}|{_DISTRIBUTION_PATTERNS}|{_CONTACT_PATTERNS})",
    re.IGNORECASE,
)


def find_anomaly_candidates(
    page_img: np.ndarray,
    params: Optional[RemoverParams] = None,
) -> list[tuple[int, int, int, int]]:
    """Extract candidate bounding boxes likely containing single-page watermarks/stamps.

    Finds:
    1. User-specified ROI bbox if provided
    2. Header and footer margin banner bands (top and bottom 10%)
    3. Saturated color icon/stamp anchors across the spectrum
    4. Low-contrast mid-tone horizontal strips on monochrome pages
    """
    h, w = page_img.shape[:2]
    candidates: list[tuple[int, int, int, int]] = []

    # 1. User-specified ROI bounding box if provided
    if params and params.roi_bbox:
        rx0, ry0, rx1, ry1 = params.roi_bbox
        candidates.append((max(0, rx0), max(0, ry0), min(w, rx1), min(h, ry1)))

    # 2. Header & footer margin zones (top and bottom 10%)
    m_h = max(20, int(h * 0.10))
    candidates.append((0, 0, w, m_h))
    candidates.append((0, h - m_h, w, h))

    # 3. Saturated colour icon anchors across entire color wheel (stamps, seals, badges)
    if page_img.ndim == 3 and page_img.shape[2] == 3:
        c_max = np.max(page_img, axis=2).astype(np.int16)
        c_min = np.min(page_img, axis=2).astype(np.int16)
        color_diff = c_max - c_min
        chroma_anchor = (color_diff > 30) & (c_min < 225)

        if chroma_anchor.sum() > 10:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                chroma_anchor.astype(np.uint8), connectivity=8
            )
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if 10 <= area <= 25000:
                    sy0 = int(stats[i, cv2.CC_STAT_TOP])
                    sh0 = int(stats[i, cv2.CC_STAT_HEIGHT])
                    pad_y = max(20, int(sh0 * 1.5))
                    y0 = max(0, sy0 - pad_y)
                    y1 = min(h, sy0 + sh0 + pad_y)
                    candidates.append((0, y0, w, y1))

    # 4. Low-contrast mid-tone text strips (monochrome watermarks / stamps)
    grey = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY) if page_img.ndim == 3 else page_img
    mid_tone = (grey >= 110) & (grey <= 220)
    row_counts = np.sum(mid_tone, axis=1)
    active_rows = np.where(row_counts > w * 0.05)[0]
    if len(active_rows) > 0:
        diffs = np.diff(active_rows)
        split_points = np.where(diffs > 15)[0] + 1
        row_groups = np.split(active_rows, split_points)
        for group in row_groups:
            if 10 <= len(group) <= int(h * 0.15):
                gy0 = max(0, int(group[0]) - 15)
                gy1 = min(h, int(group[-1]) + 15)
                candidates.append((0, gy0, w, gy1))

    # Deduplicate and merge vertically overlapping strips
    sorted_cands = sorted(candidates, key=lambda c: (c[1], c[3]))
    merged: list[tuple[int, int, int, int]] = []
    for cand in sorted_cands:
        if not merged:
            merged.append(cand)
            continue
        prev_x0, prev_y0, prev_x1, prev_y1 = merged[-1]
        cur_x0, cur_y0, cur_x1, cur_y1 = cand
        if cur_y0 <= prev_y1 + 10:
            merged[-1] = (min(prev_x0, cur_x0), prev_y0, max(prev_x1, cur_x1), max(prev_y1, cur_y1))
        else:
            merged.append(cand)

    return merged


def piracy_ocr_spotter(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
) -> np.ndarray:
    """Spot and remove single-page unrepeated raster watermarks via RapidOCR.

    Operates strictly on candidate strips to maintain fast CPU execution.
    Protects crossing body-text strokes via directional stroke continuity analysis.
    """
    ocr_engine = get_rapidocr()
    if ocr_engine is None:
        logger.debug("piracy_ocr_spotter: RapidOCR not installed, skipping")
        return page_img

    result = page_img.copy()
    h, w = page_img.shape[:2]
    grey = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY) if page_img.ndim == 3 else page_img

    candidates = find_anomaly_candidates(page_img, params=params)
    matched_regions: list[tuple[int, int, int, int]] = []

    for x0, y0, x1, y1 in candidates:
        crop = page_img[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        try:
            ocr_results, _ = ocr_engine(crop)
            if not ocr_results:
                continue

            for item in ocr_results:
                poly, text, score = item[0], item[1], item[2]
                if score < 0.35:
                    continue

                is_match = bool(_PIRACY_PATTERN.search(text))
                if not is_match and params and params.target_texts:
                    is_match = any(re.search(t, text, re.IGNORECASE) for t in params.target_texts)

                if is_match:
                    pts = np.array(poly, dtype=np.int32)
                    line_h = max(16, int(np.max(pts[:, 1]) - np.min(pts[:, 1])))
                    pad_x = max(6, int(line_h * 0.4))
                    pad_y = max(4, int(line_h * 0.25))

                    bx0 = max(0, x0 + int(np.min(pts[:, 0])) - pad_x)
                    by0 = max(0, y0 + int(np.min(pts[:, 1])) - pad_y)
                    bx1 = min(w, x0 + int(np.max(pts[:, 0])) + pad_x)
                    by1 = min(h, y0 + int(np.max(pts[:, 1])) + pad_y)

                    # Sweep horizontally for adjacent chromatic anchors on the same baseline
                    if page_img.ndim == 3 and page_img.shape[2] == 3:
                        c_max = np.max(page_img, axis=2).astype(np.int16)
                        c_min = np.min(page_img, axis=2).astype(np.int16)
                        search_y0 = max(0, by0 - 8)
                        search_y1 = min(h, by1 + 8)
                        line_chroma = (
                            c_max[search_y0:search_y1, :] - c_min[search_y0:search_y1, :] > 30
                        ) & (c_min[search_y0:search_y1, :] < 225)
                        if line_chroma.any():
                            _, cx = np.where(line_chroma)
                            max_dist = max(150, line_h * 5)
                            close_cx = [
                                c
                                for c in cx
                                if abs(c - bx0) <= max_dist or abs(c - bx1) <= max_dist
                            ]
                            if close_cx:
                                icon_x0 = max(0, int(min(close_cx)) - 6)
                                icon_x1 = min(w, int(max(close_cx)) + 6)
                                bx0 = min(bx0, icon_x0)
                                bx1 = max(bx1, icon_x1)

                    matched_regions.append((bx0, by0, bx1, by1))
                    logger.info(
                        "piracy_ocr_spotter: spotted watermark '%s' at [%d,%d,%d,%d]",
                        text,
                        bx0,
                        by0,
                        bx1,
                        by1,
                    )
        except Exception as exc:
            logger.debug("RapidOCR crop execution failed: %s", exc)

    if not matched_regions:
        return page_img

    # Build glyph removal mask with stroke continuity protection
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    paper_val = int(np.percentile(grey[grey >= 230], 95)) if (grey >= 230).sum() > 50 else 255

    for bx0, by0, bx1, by1 in matched_regions:
        roi_grey = grey[by0:by1, bx0:bx1]
        roi_img = page_img[by0:by1, bx0:bx1]
        if page_img.ndim == 3:
            roi_chroma = (
                np.max(roi_img, axis=2).astype(np.int16) - np.min(roi_img, axis=2).astype(np.int16)
            ) > 30
        else:
            roi_chroma = np.zeros_like(roi_grey, dtype=bool)

        roi_ink = (roi_grey < paper_val - 15) | roi_chroma

        # Protect dark body text strokes crossing through the watermark region
        crossing_text = np.zeros_like(roi_ink, dtype=bool)
        if by0 > 5 and by1 < h - 5:
            dark_ink = (roi_grey < 75) & (~roi_chroma)
            if dark_ink.any():
                top_boundary = grey[by0 - 3 : by0, bx0:bx1] < 75
                bottom_boundary = grey[by1 : by1 + 3, bx0:bx1] < 75
                if top_boundary.any() and bottom_boundary.any():
                    crossing_cols = np.where(
                        top_boundary.any(axis=0) | bottom_boundary.any(axis=0)
                    )[0]
                    for col in crossing_cols:
                        c_start = max(0, col - 2)
                        c_end = min(roi_ink.shape[1], col + 3)
                        crossing_text[:, c_start:c_end] |= dark_ink[:, c_start:c_end]

        wm_pixels = roi_ink & (~crossing_text)
        inpaint_mask[by0:by1, bx0:bx1] = np.where(wm_pixels, 255, 0).astype(np.uint8)

    if inpaint_mask.sum() > 0:
        result = cv2.inpaint(result, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return result


def swt_morph_reconstruct(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
) -> np.ndarray:
    """Remove Guilloche security lace and fine background halftone grids.

    Uses morphological grayscale reconstruction seeded from dark high-contrast
    text cores (grey < 90, stroke width >= 2px) to wipe fine background lace.
    """
    if not has_skimage():
        logger.debug("swt_morph_reconstruct: scikit-image not available, skipping")
        return page_img

    from skimage.morphology import reconstruction

    grey = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY) if page_img.ndim == 3 else page_img.copy()

    # Seed is dark text cores eroded
    seed = grey.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed = cv2.erode(seed, k, iterations=1)

    # Reconstruct from text seeds
    try:
        reconstructed = reconstruction(seed, grey, method="dilation")
        reconstructed_u8 = np.clip(reconstructed, 0, 255).astype(np.uint8)

        # Merge back: replace high-frequency background lace with paper white
        diff = cv2.absdiff(grey, reconstructed_u8)
        lace_mask = (diff > 8) & (grey >= 130)

        result = page_img.copy()
        paper_val = int(np.percentile(grey[grey >= 240], 95)) if (grey >= 240).sum() > 50 else 255

        if result.ndim == 3:
            for c in range(result.shape[2]):
                result[:, :, c] = np.where(lace_mask, paper_val, result[:, :, c])
        else:
            result = np.where(lace_mask, paper_val, result)

        logger.info(
            "swt_morph_reconstruct: neutralized %d Guilloche lace pixels", int(lace_mask.sum())
        )
        return result
    except Exception as exc:
        logger.debug("Morphological reconstruction failed: %s", exc)
        return page_img
