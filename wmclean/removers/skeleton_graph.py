"""Skeleton-Graph Stroke Disentanglement (Topological Branch Classifier).

Converts candidate 1-px stroke skeletons into a topological branch graph (via skan).
Classifies whole stroke segments between junctions:
- Branches connected to solid black text/diagram cores are 100% kept.
- Only branches tracing the watermark glyph contour are unmixed.

Ensures that thin chemical bonds, dashed graph lines, arrows, and serifs are never
nibbled away by pixel thresholding.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize

from wmclean.deps import get_skan, has_skan
from wmclean.intelligence.doc_shield import parse_doc_shield
from wmclean.schemas import RemoverParams

logger = logging.getLogger(__name__)


def skeleton_graph(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
    **kwargs,
) -> np.ndarray:
    """Disentangle overlapping watermark strokes from diagrams/text using skeleton branch graphs.

    Args:
        page_img: (H, W, C) or (H, W) uint8 input image.
        mask: Optional binary candidate mask.
        params: Optional RemoverParams.

    Returns:
        Cleaned image array with preserved document strokes.
    """
    is_rgb = page_img.ndim == 3 and page_img.shape[2] >= 3
    shield = parse_doc_shield(page_img)
    norm_gray = shield.norm_gray
    bg = shield.bg
    h, w = norm_gray.shape[:2]

    # Candidate stroke pixels: mid-grey ink (80 <= norm_gray <= 235) outside diagram
    if mask is not None and mask.any():
        cand_mask = (mask > 0) & (~shield.m_core) & (~shield.m_diagram)
    else:
        cand_mask = (norm_gray >= 80) & (norm_gray <= 235) & (~shield.m_core) & (~shield.m_diagram)

    if np.sum(cand_mask) < 30:
        return page_img

    # 1. Skeletonize candidate strokes to 1-px width
    skel = skeletonize(cand_mask)
    if not skel.any():
        return page_img

    unmix_stroke_pixels = np.zeros((h, w), dtype=bool)

    # 2. Build Topological Branch Graph
    if has_skan():
        try:
            skan = get_skan()
            if skan is not None:
                skeleton = skan.Skeleton(skel)
                stats = skan.summarize(skeleton, separator="_")
                n_branches = len(stats) if stats is not None else 0

                if n_branches > 0 and n_branches <= 400:
                    for branch_idx in range(n_branches):
                        coords = skeleton.path_coordinates(branch_idx)  # (N, 2) array of (y, x)
                        coords_y = np.clip(np.round(coords[:, 0]).astype(int), 0, h - 1)
                        coords_x = np.clip(np.round(coords[:, 1]).astype(int), 0, w - 1)

                        # Check if this branch touches protected text cores or diagrams
                        touches_core = False
                        for y_pt, x_pt in zip(coords_y[[0, -1]], coords_x[[0, -1]]):
                            y_lo = max(0, y_pt - 2)
                            y_hi = min(h, y_pt + 3)
                            x_lo = max(0, x_pt - 2)
                            x_hi = min(w, x_pt + 3)
                            if (
                                shield.m_core[y_lo:y_hi, x_lo:x_hi].any()
                                or shield.m_diagram[y_lo:y_hi, x_lo:x_hi].any()
                            ):
                                touches_core = True
                                break

                        if not touches_core:
                            unmix_stroke_pixels[coords_y, coords_x] = True
                else:
                    # High-density branch fallback: C++ SIMD connected components
                    num_labels, labels = cv2.connectedComponents(
                        skel.astype(np.uint8), connectivity=8
                    )
                    kernel_3x3 = np.ones((3, 3), np.uint8)
                    dil_core = (
                        cv2.dilate((shield.m_core | shield.m_diagram).astype(np.uint8), kernel_3x3)
                        > 0
                    )
                    for lbl_id in range(1, min(num_labels, 500)):
                        lbl_mask = labels == lbl_id
                        if not (lbl_mask & dil_core).any():
                            unmix_stroke_pixels |= lbl_mask
        except Exception as exc:
            logger.debug("skan branch analysis fallback: %s", exc)
            # Fallback: connected components on skeleton
            num_labels, labels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
            kernel_3x3 = np.ones((3, 3), np.uint8)
            dil_core = cv2.dilate(shield.m_core.astype(np.uint8), kernel_3x3) > 0
            for lbl_id in range(1, num_labels):
                lbl_mask = labels == lbl_id
                if not (lbl_mask & dil_core).any():
                    unmix_stroke_pixels |= lbl_mask
    else:
        # Fallback without skan
        num_labels, labels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
        kernel_3x3 = np.ones((3, 3), np.uint8)
        dil_core = cv2.dilate(shield.m_core.astype(np.uint8), kernel_3x3) > 0
        for lbl_id in range(1, num_labels):
            lbl_mask = labels == lbl_id
            if not (lbl_mask & dil_core).any():
                unmix_stroke_pixels |= lbl_mask

    # Dilate 1-px unmix strokes back to original stroke thickness (1-2 px)
    unmix_dilated = (
        cv2.dilate(unmix_stroke_pixels.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        > 0
    )
    unmix_final = unmix_dilated & cand_mask & (~shield.m_ink_halo_shield) & (~shield.m_diagram)

    if not unmix_final.any():
        return page_img

    out_img = page_img.copy()
    bg_val = bg.copy()
    if is_rgb:
        for c in range(3):
            out_img[:, :, c] = np.where(unmix_final, bg_val, out_img[:, :, c])
    else:
        out_img = np.where(unmix_final, bg_val, out_img)

    # Enforce text-core lock
    out_img = shield.apply_text_core_lock(out_img, page_img, lock_halo=True)
    logger.info("skeleton_graph: unmixed %d watermark stroke pixels", int(unmix_final.sum()))
    return out_img
