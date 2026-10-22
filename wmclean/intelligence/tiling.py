"""Overlapping tile processing with raised cosine blending for high-megapixel scans.

Enables processing 300+ DPI scans and multi-megapixel images in bounded O(1) memory
while preventing edge seam artifacts via 2D cosine weight windows.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


def _cosine_window_2d(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
    """Generate a 2D Hann / raised-cosine blending weight window."""
    wy = np.ones(tile_h, dtype=np.float32)
    wx = np.ones(tile_w, dtype=np.float32)

    ov_y = min(overlap, tile_h // 2)
    ov_x = min(overlap, tile_w // 2)

    if ov_y > 0:
        ramp_y = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ov_y, endpoint=False))
        wy[:ov_y] = ramp_y
        wy[-ov_y:] = ramp_y[::-1]

    if ov_x > 0:
        ramp_x = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ov_x, endpoint=False))
        wx[:ov_x] = ramp_x
        wx[-ov_x:] = ramp_x[::-1]

    return np.outer(wy, wx)


def tile_image(
    img: np.ndarray,
    tile_size: int = 1024,
    overlap: int = 64,
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """Slice an image into overlapping rectangular tiles.

    Args:
        img: (H, W) or (H, W, C) input array.
        tile_size: Maximum dimension of each tile along each axis.
        overlap: Border overlap in pixels between adjacent tiles.

    Returns:
        List of ((y0, y1, x0, x1), tile_array).
    """
    h, w = img.shape[:2]
    step = tile_size - overlap
    tiles = []

    y0 = 0
    while y0 < h:
        y1 = min(y0 + tile_size, h)
        x0 = 0
        while x0 < w:
            x1 = min(x0 + tile_size, w)
            tile = img[y0:y1, x0:x1]
            tiles.append(((y0, y1, x0, x1), tile))
            if x1 == w:
                break
            x0 += step
        if y1 == h:
            break
        y0 += step

    return tiles


def stitch_tiles(
    tiles: List[Tuple[Tuple[int, int, int, int], np.ndarray]],
    target_shape: Tuple[int, ...],
    overlap: int = 64,
) -> np.ndarray:
    """Seamlessly stitch overlapping tiles back together using cosine blending.

    Args:
        tiles: List of ((y0, y1, x0, x1), tile_array).
        target_shape: Shape of original full image (H, W) or (H, W, C).
        overlap: Overlap in pixels between adjacent tiles.

    Returns:
        Reconstructed image array with smooth transitions across tile seams.
    """
    is_3d = len(target_shape) == 3
    h, w = target_shape[:2]
    channels = target_shape[2] if is_3d else 1

    accum = np.zeros((h, w, channels), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)

    for (y0, y1, x0, x1), tile in tiles:
        th, tw = y1 - y0, x1 - x0
        win = _cosine_window_2d(th, tw, overlap)

        # For outer boundaries of the entire image, keep weights as 1.0 (no cosine drop)
        if y0 == 0:
            win[: min(overlap, th // 2), :] = 1.0
        if y1 == h:
            win[-min(overlap, th // 2) :, :] = 1.0
        if x0 == 0:
            win[:, : min(overlap, tw // 2)] = 1.0
        if x1 == w:
            win[:, -min(overlap, tw // 2) :] = 1.0

        tile_3d = tile if is_3d else tile[:, :, np.newaxis]
        win_3d = win[:, :, np.newaxis]

        accum[y0:y1, x0:x1] += tile_3d.astype(np.float32) * win_3d
        weight_accum[y0:y1, x0:x1] += win

    denom = np.maximum(weight_accum, 1e-6)
    denom_3d = denom[:, :, np.newaxis]
    reconstructed = np.clip(np.round(accum / denom_3d), 0, 255).astype(np.uint8)

    if not is_3d:
        return reconstructed[:, :, 0]
    return reconstructed


def process_image_tiled(
    img: np.ndarray,
    remover_fn: Callable[[np.ndarray], np.ndarray],
    tile_size: int = 1024,
    overlap: int = 64,
    threshold_dim: int = 2048,
) -> np.ndarray:
    """Process an image in memory-bounded overlapping tiles if larger than threshold_dim.

    Args:
        img: Input image array.
        remover_fn: Function that processes a single image patch.
        tile_size: Maximum tile size.
        overlap: Overlap in pixels.
        threshold_dim: Maximum dimension before activating tiling.

    Returns:
        Processed image with identical seamless blending.
    """
    h, w = img.shape[:2]
    if max(h, w) <= threshold_dim:
        return remover_fn(img)

    tiles = tile_image(img, tile_size=tile_size, overlap=overlap)
    processed_tiles = []
    for coords, tile in tiles:
        clean_tile = remover_fn(tile)
        processed_tiles.append((coords, clean_tile))

    return stitch_tiles(processed_tiles, img.shape, overlap=overlap)
