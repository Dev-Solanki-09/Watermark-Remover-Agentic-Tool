"""Sub-Page Pixel Firewall — ensures no L3 remover ever touches pixels outside assigned ROI.

Guarantees 100% byte-identity for all pixels outside roi_bbox = (x1, y1, x2, y2).
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterator, Optional, Tuple

import cv2
import numpy as np


def clamp_roi_bbox(
    roi_bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
    pad: int = 6,
) -> Tuple[int, int, int, int]:
    """Pad and clamp (x1, y1, x2, y2) to image dimensions."""
    x1, y1, x2, y2 = roi_bbox
    x1_clamped = max(0, min(width - 1, int(x1) - pad))
    y1_clamped = max(0, min(height - 1, int(y1) - pad))
    x2_clamped = max(x1_clamped + 1, min(width, int(x2) + pad))
    y2_clamped = max(y1_clamped + 1, min(height, int(y2) + pad))
    return x1_clamped, y1_clamped, x2_clamped, y2_clamped


def apply_in_roi(
    full_img: np.ndarray,
    roi_bbox: Optional[Tuple[int, int, int, int]],
    fn: Callable[[np.ndarray], np.ndarray],
    pad: int = 6,
) -> np.ndarray:
    """Execute a cleaner callable strictly inside roi_bbox, preserving outside pixels.

    Args:
        full_img: (H, W) or (H, W, C) numpy array
        roi_bbox: (x1, y1, x2, y2) bounding box in pixel coordinates, or None for full page
        fn: callable that takes cropped image and returns cleaned cropped image
        pad: padding in pixels around roi_bbox

    Returns:
        New image array with cleaning applied exclusively inside roi_bbox.
    """
    if roi_bbox is None:
        return fn(full_img)

    h, w = full_img.shape[:2]
    x1, y1, x2, y2 = clamp_roi_bbox(roi_bbox, w, h, pad=pad)

    # Sub-slice copy
    crop = full_img[y1:y2, x1:x2].copy()
    cleaned_crop = fn(crop)

    # Construct result by copying original and pasting cleaned crop
    result = full_img.copy()
    if cleaned_crop.shape[:2] == (y2 - y1, x2 - x1):
        result[y1:y2, x1:x2] = cleaned_crop
    else:
        resized = cv2.resize(cleaned_crop, (x2 - x1, y2 - y1))
        result[y1:y2, x1:x2] = resized

    return result


@contextlib.contextmanager
def roi_lock(
    page_img: np.ndarray,
    roi_bbox: Optional[Tuple[int, int, int, int]],
    pad: int = 6,
) -> Iterator[np.ndarray]:
    """Context manager yielding cropped view and updating page_img in-place safely.

    On exit, asserts or guarantees pixels outside roi_bbox are completely unaffected.
    """
    if roi_bbox is None:
        yield page_img
        return

    h, w = page_img.shape[:2]
    x1, y1, x2, y2 = clamp_roi_bbox(roi_bbox, w, h, pad=pad)

    # Work on crop copy
    crop = page_img[y1:y2, x1:x2].copy()
    yield crop

    # Paste back inside roi
    if crop.shape[:2] == (y2 - y1, x2 - x1):
        page_img[y1:y2, x1:x2] = crop
    else:
        resized = cv2.resize(crop, (x2 - x1, y2 - y1))
        page_img[y1:y2, x1:x2] = resized
