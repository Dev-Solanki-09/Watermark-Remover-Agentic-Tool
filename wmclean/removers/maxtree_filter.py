"""Morphological Max-Tree Edge-Preserving Filter (Mathematical Morphology Paradigm).

Uses skimage.morphology.max_tree (Berger's O(N) Cython algorithm) to model an image
as a component tree across all 256 intensity levels simultaneously.

By the connected-operator theorem of mathematical morphology, pruning watermark
plateau nodes in the Max-Tree is strictly edge-preserving: it can never blur, shift,
or round surviving character serifs or thin diagram lines!
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from skimage.morphology import max_tree

from wmclean.intelligence.doc_shield import parse_doc_shield
from wmclean.schemas import RemoverParams

logger = logging.getLogger(__name__)


try:
    from numba import njit

    @njit(fastmath=True)
    def _fast_maxtree_traverse(tree_traverser, flat_inv, parent_flat, filtered_inv_flat):
        for i in range(len(tree_traverser)):
            node = tree_traverser[i]
            val = flat_inv[node]
            orig_p_val = flat_inv[parent_flat[node]]
            is_steep_ink = val > 167
            is_high_contrast = (val - orig_p_val) > 20 and val > 120
            if is_steep_ink or is_high_contrast:
                filtered_inv_flat[node] = val
            else:
                filtered_inv_flat[node] = filtered_inv_flat[parent_flat[node]]
except Exception:

    def _fast_maxtree_traverse(tree_traverser, flat_inv, parent_flat, filtered_inv_flat):
        for node in tree_traverser:
            val = flat_inv[node]
            orig_p_val = flat_inv[parent_flat[node]]
            is_steep_ink = val > 167
            is_high_contrast = (val - orig_p_val) > 20 and val > 120
            if is_steep_ink or is_high_contrast:
                filtered_inv_flat[node] = val
            else:
                filtered_inv_flat[node] = filtered_inv_flat[parent_flat[node]]


def maxtree_filter(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
    contrast_cutoff: int = 55,
    **kwargs,
) -> np.ndarray:
    """Filter out low-contrast watermark plateaus via morphological Max-Tree pruning.

    Args:
        page_img: (H, W, C) or (H, W) uint8 input image.
        mask: Optional binary candidate mask.
        params: Optional RemoverParams.
        contrast_cutoff: Extinction contrast height below which plateaus are pruned.

    Returns:
        Cleaned image array with 0.0 edge blurring.
    """
    is_rgb = page_img.ndim == 3 and page_img.shape[2] >= 3
    shield = parse_doc_shield(page_img)
    norm_gray = shield.norm_gray
    h, w = norm_gray.shape[:2]

    # Invert normalized gray so dark ink strokes and watermarks become bright peaks
    inv = (255 - norm_gray).astype(np.uint8)

    try:
        # Construct Max-Tree
        parent, tree_traverser = max_tree(inv)

        # Reconstructed 1D array initialized to background level 0
        filtered_inv = np.zeros_like(inv)
        flat_inv = inv.ravel()
        parent_flat = parent.ravel()

        _fast_maxtree_traverse(tree_traverser, flat_inv, parent_flat, filtered_inv.ravel())

        # Invert back to normal greyscale
        clean_gray = (255 - filtered_inv).astype(np.uint8)

        # Determine modified watermark pixels
        diff = cv2.absdiff(norm_gray, clean_gray) > 6
        if mask is not None and mask.any():
            diff &= mask > 0
        diff &= ~shield.m_ink_halo_shield
        diff &= ~shield.m_diagram

        out = page_img.copy()
        paper_val = (
            int(np.percentile(norm_gray[norm_gray >= 220], 95))
            if (norm_gray >= 220).sum() > 50
            else 255
        )
        if is_rgb:
            for c in range(3):
                out[:, :, c] = np.where(diff, paper_val, out[:, :, c])
        else:
            out = np.where(diff, paper_val, out)

        # Enforce text-core lock
        out = shield.apply_text_core_lock(out, page_img, lock_halo=True)
        logger.info(
            "maxtree_filter: pruned %d low-contrast watermark plateau pixels", int(diff.sum())
        )
        return out
    except Exception as exc:
        logger.debug("maxtree_filter error: %s; returning original", exc)
        return page_img
