"""Tests for wmclean.intelligence.roi_lock."""

import numpy as np

from wmclean.intelligence.roi_lock import apply_in_roi, clamp_roi_bbox, roi_lock


def test_clamp_roi_bbox():
    """Test clamping and padding to image dimensions."""
    w, h = 100, 200
    # Normal bbox with pad 6
    x1, y1, x2, y2 = clamp_roi_bbox((10, 20, 50, 60), w, h, pad=6)
    assert x1 == 4
    assert y1 == 14
    assert x2 == 56
    assert y2 == 66

    # Edge cases: outside image bounds
    x1, y1, x2, y2 = clamp_roi_bbox((-10, -5, 120, 250), w, h, pad=6)
    assert x1 == 0
    assert y1 == 0
    assert x2 == 100
    assert y2 == 200


def test_apply_in_roi_preserves_outside_pixels():
    """Test that apply_in_roi guarantees 100% byte identity for outside pixels."""
    # Create test image with known gradient pattern
    np.random.seed(42)
    img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    img_orig = img.copy()

    roi = (50, 50, 100, 100)

    # Invert colours inside ROI
    def invert_cleaner(crop: np.ndarray) -> np.ndarray:
        return 255 - crop

    cleaned = apply_in_roi(img, roi, invert_cleaner, pad=0)

    # Pixels inside ROI must be modified
    assert not np.array_equal(cleaned[50:100, 50:100], img_orig[50:100, 50:100])
    assert np.array_equal(cleaned[50:100, 50:100], 255 - img_orig[50:100, 50:100])

    # Pixels outside ROI must be 100% byte identical
    mask_inside = np.zeros((200, 200), dtype=bool)
    mask_inside[50:100, 50:100] = True
    mask_outside = ~mask_inside

    assert np.array_equal(cleaned[mask_outside], img_orig[mask_outside])


def test_roi_lock_context_manager():
    """Test roi_lock context manager preserves outside pixels."""
    img = np.full((150, 150, 3), 200, dtype=np.uint8)
    roi = (20, 20, 60, 60)

    with roi_lock(img, roi, pad=0) as crop:
        assert crop.shape == (40, 40, 3)
        crop[:] = 0

    # Inside ROI is 0
    assert np.all(img[20:60, 20:60] == 0)

    # Outside ROI is still 200
    mask_inside = np.zeros((150, 150), dtype=bool)
    mask_inside[20:60, 20:60] = True
    assert np.all(img[~mask_inside] == 200)
