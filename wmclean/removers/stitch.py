"""Virtual stitch/slice buffer for tiled/strip scans & hybrid sub-images.

Handles G16 (tiled/strip scans) and G17 (hybrid diagram watermarks) by:
1. Stitching multiple image XObjects into a single virtual canvas
2. Applying raster removers on the stitched canvas
3. Slicing back into original XObject dimensions for re-encoding
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def virtual_stitch(
    tiles: list[tuple[np.ndarray, tuple[int, int]]],
    canvas_size: tuple[int, int],
) -> np.ndarray:
    """Stitch image tiles/strips into a single canvas.

    Args:
        tiles: List of (image_array, (x_offset, y_offset)) tuples
        canvas_size: (width, height) of the full canvas

    Returns:
        Stitched canvas as numpy array
    """
    w, h = canvas_size
    ndim = tiles[0][0].ndim if tiles else 3
    if ndim == 3:
        canvas = np.full((h, w, tiles[0][0].shape[2]), 255, dtype=np.uint8)
    else:
        canvas = np.full((h, w), 255, dtype=np.uint8)

    for tile_img, (x_off, y_off) in tiles:
        th, tw = tile_img.shape[:2]
        # Clip to canvas bounds
        x_end = min(x_off + tw, w)
        y_end = min(y_off + th, h)
        tw_clip = x_end - x_off
        th_clip = y_end - y_off

        if ndim == 3:
            canvas[y_off:y_end, x_off:x_end] = tile_img[:th_clip, :tw_clip]
        else:
            canvas[y_off:y_end, x_off:x_end] = tile_img[:th_clip, :tw_clip]

    logger.info("virtual_stitch: stitched %d tiles into %dx%d canvas", len(tiles), w, h)
    return canvas


def slice_repack(
    canvas: np.ndarray,
    tile_specs: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    """Slice a cleaned canvas back into original tile dimensions.

    Args:
        canvas: Cleaned stitched canvas
        tile_specs: List of (x_offset, y_offset, width, height) tuples

    Returns:
        List of tile images in the same order as tile_specs
    """
    tiles = []
    for x_off, y_off, tw, th in tile_specs:
        if canvas.ndim == 3:
            tile = canvas[y_off : y_off + th, x_off : x_off + tw].copy()
        else:
            tile = canvas[y_off : y_off + th, x_off : x_off + tw].copy()
        tiles.append(tile)

    logger.info("slice_repack: sliced canvas into %d tiles", len(tiles))
    return tiles


def subimage_extract(
    page_img: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Extract a sub-image (diagram bitmap) from a hybrid page.

    Args:
        page_img: Full page image
        bbox: (x0, y0, x1, y1) bounding box of the sub-image

    Returns:
        Extracted sub-image
    """
    x0, y0, x1, y1 = bbox
    return page_img[y0:y1, x0:x1].copy()


def subimage_replace(
    page_img: np.ndarray,
    cleaned_sub: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Replace a sub-image in the page with its cleaned version.

    Args:
        page_img: Full page image
        cleaned_sub: Cleaned sub-image
        bbox: (x0, y0, x1, y1) bounding box

    Returns:
        Page image with replaced sub-image
    """
    x0, y0, x1, y1 = bbox
    result = page_img.copy()
    result[y0:y1, x0:x1] = cleaned_sub
    return result
