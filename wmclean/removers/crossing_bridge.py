"""Directional Collinear Stroke-Continuation (Structure Tensor Crossing Protection).

At pixels where a dark watermark/stamp intersects a black text letter or diagram line:
uses the 2D Structure Tensor orientation and anisotropy coherence to identify
continuous passing character strokes and preserve their authentic native pixels.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

logger = logging.getLogger(__name__)


def compute_crossing_stroke_keep_mask(
    norm_gray: np.ndarray,
    m_core: np.ndarray,
    m_wm_stroke: np.ndarray,
    sigma: float = 1.2,
    coherence_thresh: float = 0.40,
) -> np.ndarray:
    """Detect text/diagram strokes passing through watermark junctions and mark them to keep.

    Args:
        norm_gray: (H, W) uint8 normalized greyscale image
        m_core: (H, W) bool mask of protected foreground text/diagram cores outside the watermark
        m_wm_stroke: (H, W) bool mask of the watermark stroke footprint
        sigma: Gaussian scale for structure tensor
        coherence_thresh: Anisotropy coherence threshold [0..1]

    Returns:
        (H, W) bool mask of crossing pixels that should be strictly preserved.
    """
    h, w = norm_gray.shape[:2]
    if not m_wm_stroke.any() or not m_core.any():
        return np.zeros((h, w), dtype=bool)

    # 1. Structure Tensor & Eigenvalues on normalized gray
    # l1 >= l2; high coherence indicates a strong oriented line/stroke edge
    a_elems = structure_tensor(norm_gray.astype(np.float32) / 255.0, sigma=sigma, mode="reflect")
    l1, l2 = structure_tensor_eigenvalues(a_elems)
    coherence = np.where(
        (l1 + l2) > 1e-4,
        ((l1 - l2) / (l1 + l2 + 1e-5)) ** 2,
        0.0,
    )

    # 2. Find ink entries at the outer boundary of the watermark stroke
    kernel_3x3 = np.ones((3, 3), np.uint8)
    dilated_wm = cv2.dilate(m_wm_stroke.astype(np.uint8), kernel_3x3, iterations=2) > 0
    r_ext = dilated_wm & (~m_wm_stroke)
    ink_entry = r_ext & (norm_gray < 110) & m_core

    # 3. Propagate ink entries inward across the thin watermark stroke along directional kernels
    # Restricted to dark ink pixels (norm_gray < 95) with coherent stroke alignment
    k_h = np.ones((1, 5), np.uint8)
    k_v = np.ones((5, 1), np.uint8)
    k_d1 = np.eye(5, dtype=np.uint8)
    k_d2 = np.fliplr(k_d1)

    prop = ink_entry.astype(np.uint8)
    prop_inward = (
        cv2.dilate(prop, k_h, iterations=1)
        | cv2.dilate(prop, k_v, iterations=1)
        | cv2.dilate(prop, k_d1, iterations=1)
        | cv2.dilate(prop, k_d2, iterations=1)
    ) > 0

    # Crossing keep: inside watermark, dark, connected to external core, with directional coherence
    m_crossing_keep = (
        m_wm_stroke
        & prop_inward
        & (norm_gray < 95)
        & ((coherence > coherence_thresh) | (norm_gray < 65))
    )

    return m_crossing_keep
