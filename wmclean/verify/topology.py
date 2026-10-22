"""L4 Topological Invariance Evaluator & Skeleton Stitching.

Enforces Betti-number invariance on scientific diagrams, chemical formulas,
and circuit diagrams:
- β0: Number of connected components (stroke continuity)
- β1: Number of 1D closed loops (benzene rings, circuit loops, holes)

Relation:
  χ = β0 - β1  ==>  β1 = β0 - χ
Where χ is the Euler characteristic computed via skimage.measure.euler_number.

If an eraser introduces a microscopic gap in a diagram stroke, skeleton endpoint
stitching connects collinear pairs across the watermark mask to restore β1 invariance.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.measure import euler_number
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


def compute_betti_numbers(binary_mask: np.ndarray) -> Tuple[int, int]:
    """Compute (β0, β1) topological invariants for a binary mask.

    Args:
        binary_mask: 2D uint8 or bool array (True / 255 = foreground stroke).

    Returns:
        (beta_0, beta_1):
        - beta_0: Count of connected components (objects)
        - beta_1: Count of 1D closed loops / holes
    """
    binary = (binary_mask > 0).astype(np.uint8)
    if not np.any(binary):
        return 0, 0

    # 1. β0: Number of 8-connected components
    num_labels, _ = cv2.connectedComponents(binary, connectivity=8)
    beta_0 = max(0, num_labels - 1)  # subtract background label 0

    # 2. χ: Euler characteristic (connectivity 2 for 8-connected foreground)
    chi = int(euler_number(binary, connectivity=2))

    # 3. β1 = β0 - χ (in 2D space)
    beta_1 = max(0, beta_0 - chi)

    return beta_0, beta_1


def verify_topological_invariance(
    orig_bgr: np.ndarray,
    clean_bgr: np.ndarray,
    diagram_mask: Optional[np.ndarray] = None,
) -> Tuple[bool, int, int, str]:
    """Verify that diagram strokes, chemical rings, and circuit loops remain topologically invariant.

    Returns:
        (passed, delta_beta_0, delta_beta_1, message)
    """
    orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY) if orig_bgr.ndim == 3 else orig_bgr
    clean_gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY) if clean_bgr.ndim == 3 else clean_bgr

    if diagram_mask is None:
        from wmclean.intelligence.doc_shield import parse_doc_shield
        shield = parse_doc_shield(orig_bgr)
        diagram_mask = shield.m_diagram

    if not np.any(diagram_mask):
        return True, 0, 0, "No diagram strokes present"

    # Extract binary diagram strokes in original and cleaned images
    orig_diag = (orig_gray < 160) & diagram_mask
    clean_diag = (clean_gray < 160) & diagram_mask

    orig_b0, orig_b1 = compute_betti_numbers(orig_diag)
    clean_b0, clean_b1 = compute_betti_numbers(clean_diag)

    delta_b0 = clean_b0 - orig_b0
    delta_b1 = clean_b1 - orig_b1

    # Invariant: closed loops must NOT be destroyed (delta_b1 >= 0)
    # and strokes must NOT be fragmented into many pieces (delta_b0 <= 2)
    passed = (delta_b1 == 0) and (delta_b0 <= 1)

    if not passed:
        msg = f"Topological breach: Δβ1={delta_b1} (loops altered: orig={orig_b1}, clean={clean_b1}), Δβ0={delta_b0} (fragmentation)"
        logger.warning(msg)
    else:
        msg = f"Topological invariance preserved: β0={clean_b0}, β1={clean_b1} (Δβ0={delta_b0}, Δβ1={delta_b1})"

    return passed, delta_b0, delta_b1, msg


def stitch_broken_skeletons(
    stroke_mask: np.ndarray,
    watermark_mask: np.ndarray,
    max_gap_px: int = 10,
) -> np.ndarray:
    """Locate endpoints in broken skeleton within watermark mask and stitch collinear pairs.

    Args:
        stroke_mask: Binary mask of authentic strokes.
        watermark_mask: Binary mask of watermark area where breaks could occur.
        max_gap_px: Maximum distance in pixels to bridge.

    Returns:
        Stitched binary stroke mask with restored topology.
    """
    binary = stroke_mask > 0
    if not np.any(binary):
        return stroke_mask

    # 1. Thin strokes to 1-pixel skeleton
    skel = skeletonize(binary)

    # 2. Identify endpoints (skeleton pixels with exactly 1 neighbor in 3x3)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    # Center pixel is 10, 1 neighbor adds 1 -> total == 11
    endpoints = (neighbors == 11) & skel & (watermark_mask > 0)

    ey, ex = np.where(endpoints)
    if len(ex) < 2:
        return stroke_mask

    stitched = stroke_mask.copy()
    pts = list(zip(ex, ey))
    used = set()

    # 3. Find closest mutual endpoint pairs within max_gap_px
    for i, (x1, y1) in enumerate(pts):
        if i in used:
            continue
        best_j = -1
        best_dist = float("inf")
        for j, (x2, y2) in enumerate(pts):
            if i == j or j in used:
                continue
            dist = np.hypot(x2 - x1, y2 - y1)
            if dist <= max_gap_px and dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j != -1:
            used.add(i)
            used.add(best_j)
            x2, y2 = pts[best_j]
            cv2.line(stitched, (x1, y1), (x2, y2), 255, thickness=2)
            logger.debug("Stitched skeleton gap: (%d,%d) -> (%d,%d) dist=%.1fpx", x1, y1, x2, y2, best_dist)

    return stitched
