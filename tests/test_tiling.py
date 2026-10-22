"""Unit tests for overlapping tile processing and cosine blending."""

import numpy as np

from wmclean.intelligence.tiling import (
    process_image_tiled,
    stitch_tiles,
    tile_image,
)


def test_tiling_lossless_roundtrip():
    # 1200 x 1500 gradient image
    h, w = 1200, 1500
    y, x = np.mgrid[0:h, 0:w]
    img = ((x * 0.1 + y * 0.1) % 256).astype(np.uint8)

    tiles = tile_image(img, tile_size=512, overlap=64)
    assert len(tiles) > 1

    stitched = stitch_tiles(tiles, img.shape, overlap=64)

    # Cosine blending should perfectly reconstruct the original image
    diff = np.abs(img.astype(float) - stitched.astype(float))
    assert np.max(diff) <= 1.0, f"Max difference across seam is {np.max(diff)}"
    assert np.mean(diff) <= 0.05, f"Mean error across image is {np.mean(diff)}"


def test_tiling_color_image_seam_invariance():
    h, w = 1000, 1000
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 120
    img[:, :, 1] = 180
    img[:, :, 2] = 220

    tiles = tile_image(img, tile_size=400, overlap=64)
    stitched = stitch_tiles(tiles, img.shape, overlap=64)

    assert np.array_equal(img, stitched), "Uniform image must be bitwise identical after stitch"


def test_process_image_tiled_high_res():
    # 2400 x 2400 image that triggers tiling
    img = np.full((2400, 2400, 3), 240, dtype=np.uint8)

    # Dummy remover function that subtracts 10 from each pixel
    def dummy_remover(patch: np.ndarray) -> np.ndarray:
        return np.clip(patch.astype(int) - 10, 0, 255).astype(np.uint8)

    result = process_image_tiled(
        img,
        dummy_remover,
        tile_size=1024,
        overlap=64,
        threshold_dim=2000,
    )

    assert result.shape == (2400, 2400, 3)
    # Check that values are 230 throughout without border spikes
    assert np.all(result == 230)
