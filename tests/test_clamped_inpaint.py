"""Unit tests for Tier 3 strictly clamped inpainting."""

import cv2
import numpy as np

from wmclean.removers.clamped_inpaint import clamped_inpaint


def test_clamped_inpaint_bitwise_outside_invariance():
    # Continuous tone gradient photo
    h, w = 300, 300
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            img[y, x] = [x % 256, y % 256, (x + y) % 256]

    # Add opaque red stamp in center
    mask = np.zeros((h, w), dtype=bool)
    mask[100:150, 100:150] = True
    watermarked = img.copy()
    watermarked[mask] = [0, 0, 255]

    # Inpaint with 2px dilation
    cleaned = clamped_inpaint(watermarked, mask=mask, dilation_px=2, method="telea")

    # Invariant: Outside mask dilated by 2px, pixels MUST BE BITWISE IDENTICAL
    dilated_mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    outside_orig = watermarked[~dilated_mask]
    outside_clean = cleaned[~dilated_mask]

    assert np.array_equal(outside_orig, outside_clean), "Pixels outside M_dilated must be bitwise identical"

    # Verify watermark was removed inside center: red channel is no longer high while blue is low
    center_cleaned = cleaned[120:130, 120:130]
    is_pure_red = (center_cleaned[:, :, 2] > 200) & (center_cleaned[:, :, 0] < 50)
    assert not np.all(is_pure_red)


def test_clamped_inpaint_preserves_text_under_stamp():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    # Authentic text
    cv2.putText(img, "VERIFIED", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Semi-opaque or opaque stamp overlaying part of text
    mask = np.zeros((200, 200), dtype=bool)
    mask[70:120, 60:140] = True
    watermarked = img.copy()
    watermarked[mask & (img[:, :, 0] == 255)] = (180, 180, 250)

    cleaned = clamped_inpaint(watermarked, mask=mask, shield_core=True)

    # Text core should not be erased (y=90, x=50 is 'E')
    assert cleaned[90, 50, 0] < 50, "Text core under stamp must be preserved"
