"""Euclidean Chamfer Distance Transform localization for shifted watermarks.

Adapted from rohitrango/automatic-watermark-detection (Dekel et al. CVPR 2017):
Locates shifted or jittered watermark templates across pages under heavy text
occlusion and scan jitter by computing the distance transform of scene edges
and convolving with template edge kernels.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Ensure image is 2D uint8 grayscale."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.shape[2] == 4:
            return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
        return img[:, :, 0]
    raise ValueError(f"Unsupported image dimensions: {img.shape}")


def extract_chamfer_edges(
    img: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
) -> np.ndarray:
    """Extract binary edge map for Chamfer distance matching."""
    gray = _to_gray(img)
    return cv2.Canny(gray, canny_low, canny_high)


def chamfer_align_template(
    scene_img: np.ndarray,
    template: np.ndarray,
    search_bbox: Optional[Union[list[int], tuple[int, int, int, int]]] = None,
    canny_low: int = 50,
    canny_high: int = 150,
) -> tuple[int, int, float]:
    """Locate watermark template within scene using Euclidean Chamfer Distance Transform.

    Args:
        scene_img: Full page or ROI image (grayscale or RGB).
        template: Watermark template image or binary edge map.
        search_bbox: Optional [x0, y0, x1, y1] search window within scene.
        canny_low: Lower hysteresis threshold for Canny.
        canny_high: Higher hysteresis threshold for Canny.

    Returns:
        (x_top_left, y_top_left, min_distance):
            Top-left pixel coordinates in scene_img coordinates,
            and mean Euclidean distance (in pixels) of template edges to scene edges.
    """
    scene_gray = _to_gray(scene_img)
    sh, sw = scene_gray.shape

    # Determine search region
    if search_bbox is not None:
        x0, y0, x1, y1 = search_bbox
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(sw, int(x1)), min(sh, int(y1))
        search_crop = scene_gray[y0:y1, x0:x1]
        offset_x, offset_y = x0, y0
    else:
        search_crop = scene_gray
        offset_x, offset_y = 0, 0

    # Extract template edges
    if template.dtype == bool or (len(np.unique(template)) <= 2 and template.max() > 0):
        t_edges = (template > 0).astype(np.uint8) * 255
    else:
        t_edges = extract_chamfer_edges(template, canny_low, canny_high)

    th, tw = t_edges.shape
    total_edge_pixels = int(np.sum(t_edges > 0))
    if total_edge_pixels == 0:
        logger.warning("Template contains no edges for Chamfer alignment")
        return (offset_x, offset_y, float("inf"))

    # Extract scene edges in search region
    s_edges = cv2.Canny(search_crop, canny_low, canny_high)

    # Compute Euclidean distance transform of scene edge map
    # Distance of each pixel to nearest edge
    dist_map = cv2.distanceTransform(255 - s_edges, cv2.DIST_L2, 3)

    # Normalize template edge kernel so filter2D outputs average Euclidean distance per edge pixel
    kernel = (t_edges > 0).astype(np.float32)
    kernel /= float(total_edge_pixels)

    # Convolve distance transform with template edge kernel
    chamfer_map = cv2.filter2D(dist_map, cv2.CV_32F, kernel, borderType=cv2.BORDER_REPLICATE)

    # Filter2D centers the kernel at (tw // 2, th // 2)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(chamfer_map)
    cx, cy = tw // 2, th // 2

    best_x = offset_x + (min_loc[0] - cx)
    best_y = offset_y + (min_loc[1] - cy)

    return (best_x, best_y, float(min_val))
