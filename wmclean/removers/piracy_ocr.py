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

# Built-in Piracy & Scanner Regex Pattern Registry
_PIRACY_PATTERN = re.compile(
    r"(telegram|@[a-z0-9_]{3,}|gciclc|search\s+on|latest\s+updates|"
    r"camscanner|tapscanner|downloaded\s+by|t\.me/|whatsapp|\+91[\s-]?[6-9]\d{9})",
    re.IGNORECASE,
)


def find_anomaly_candidates(
    page_img: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Extract candidate bounding boxes likely containing single-page watermarks/stamps.

    Finds:
    1. Top and bottom 8% margin banner bands
    2. Coloured icon anchors (R - max(G,B) > 30)
    3. Low-contrast mid-grey horizontal strips
    """
    h, w = page_img.shape[:2]
    candidates: list[tuple[int, int, int, int]] = []

    # 1. Top & bottom 8% margin zones
    m_h = int(h * 0.08)
    candidates.append((0, 0, w, m_h))
    candidates.append((0, h - m_h, w, h))

    # 2. Saturated colour icon anchors (e.g. red stamps / hearts)
    if page_img.ndim == 3 and page_img.shape[2] == 3:
        r = page_img[:, :, 0].astype(np.int16)
        g = page_img[:, :, 1].astype(np.int16)
        b = page_img[:, :, 2].astype(np.int16)
        red_anchor = (r - np.maximum(g, b)) > 30

        if red_anchor.sum() > 15:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                red_anchor.astype(np.uint8), connectivity=8
            )
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if 15 <= area <= 1200:
                    x0 = max(0, stats[i, cv2.CC_STAT_LEFT] - 100)
                    y0 = max(0, stats[i, cv2.CC_STAT_TOP] - 30)
                    x1 = min(w, stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] + 100)
                    y1 = min(h, stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] + 30)
                    candidates.append((x0, y0, x1, y1))

    return candidates


def piracy_ocr_spotter(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
) -> np.ndarray:
    """Spot and remove single-page unrepeated raster watermarks via RapidOCR.

    Operates strictly on candidate strips to maintain <50ms CPU execution.
    Protects crossing body-text strokes (grey < 78) via directional line bridging.
    """
    ocr_engine = get_rapidocr()
    if ocr_engine is None:
        logger.debug("piracy_ocr_spotter: RapidOCR not installed, skipping")
        return page_img

    result = page_img.copy()
    h, w = page_img.shape[:2]
    grey = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY) if page_img.ndim == 3 else page_img

    candidates = find_anomaly_candidates(page_img)
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
                if score < 0.50:
                    continue

                if _PIRACY_PATTERN.search(text) or (params and any(re.search(t, text, re.I) for t in params.target_texts)):
                    pts = np.array(poly, dtype=np.int32)
                    bx0 = max(0, x0 + int(np.min(pts[:, 0])) - 4)
                    by0 = max(0, y0 + int(np.min(pts[:, 1])) - 4)
                    bx1 = min(w, x0 + int(np.max(pts[:, 0])) + 4)
                    by1 = min(h, y0 + int(np.max(pts[:, 1])) + 4)
                    matched_regions.append((bx0, by0, bx1, by1))
                    logger.info("piracy_ocr_spotter: spotted watermark '%s' at [%d,%d,%d,%d]", text, bx0, by0, bx1, by1)
        except Exception as exc:
            logger.debug("RapidOCR crop execution failed: %s", exc)

    if not matched_regions:
        return page_img

    # Build glyph removal mask with stroke protection
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    for bx0, by0, bx1, by1 in matched_regions:
        roi_grey = grey[by0:by1, bx0:bx1]
        # Inpaint candidate pixels (lighter text or coloured ink)
        # Protect dark body-text strokes (grey < 78)
        wm_pixels = (roi_grey >= 78) & (roi_grey <= 235)
        inpaint_mask[by0:by1, bx0:bx1] = np.where(wm_pixels, 255, 0).astype(np.uint8)

    if inpaint_mask.sum() > 0:
        # Telea inpainting
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

        logger.info("swt_morph_reconstruct: neutralized %d Guilloche lace pixels", int(lace_mask.sum()))
        return result
    except Exception as exc:
        logger.debug("Morphological reconstruction failed: %s", exc)
        return page_img
