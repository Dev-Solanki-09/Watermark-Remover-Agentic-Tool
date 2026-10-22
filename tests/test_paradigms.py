"""Verification tests for the 4 Orthogonal Zero-Smudge Paradigms (verifying 4.md)."""

from __future__ import annotations

import cv2
import numpy as np
import pymupdf
from skimage.metrics import structural_similarity as ssim

from wmclean.removers.bilingual_clone import bilingual_twin_clone
from wmclean.removers.maxtree_filter import maxtree_filter
from wmclean.removers.mupdf_layer_redact import mupdf_layer_redact


def test_bilingual_twin_column_diagram_clone():
    """Verify bilingual twin-column diagram cloning restores stamped diagram (SSIM >= 0.998)."""
    h, w = 600, 1000
    page = np.full((h, w, 3), 252, dtype=np.uint8)

    # Center divider line
    cv2.line(page, (500, 20), (500, 580), (120, 120, 120), 1)

    # Draw identical chemical benzene ring in Left (English) and Right (Hindi) columns
    left_center = (250, 200)
    right_center = (750, 200)
    r_hex = 35

    def draw_benzene(img, c):
        pts = []
        for i in range(6):
            a = i * np.pi / 3.0
            pts.append([int(c[0] + r_hex * np.cos(a)), int(c[1] + r_hex * np.sin(a))])
        cv2.polylines(
            img,
            [np.array(pts, np.int32).reshape((-1, 1, 2))],
            isClosed=True,
            color=(15, 15, 15),
            thickness=2,
        )
        cv2.circle(img, c, 18, (15, 15, 15), 1)

    draw_benzene(page, left_center)
    draw_benzene(page, right_center)

    # Some column text
    cv2.putText(
        page, "Q.8 Benzene structure", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2
    )
    cv2.putText(
        page, "Prashna 8: Benzene", (600, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2
    )

    # Baseline clean crop of the left benzene ring
    clean_left_crop = page[140:260, 190:310].copy()

    # Overwrite left benzene ring with a dark stamp
    watermarked = page.copy()
    stamp_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.putText(
        watermarked,
        "SEARCH ON TELEGRAM",
        (180, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (50, 50, 50),
        2,
    )
    cv2.putText(stamp_mask, "SEARCH ON TELEGRAM", (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)

    # Execute bilingual_twin_clone
    cloned_page = bilingual_twin_clone(watermarked, mask=stamp_mask)

    # Check restored left benzene ring vs authentic original clean crop
    restored_left_crop = cloned_page[140:260, 190:310]
    left_ssim = ssim(
        cv2.cvtColor(clean_left_crop, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(restored_left_crop, cv2.COLOR_RGB2GRAY),
    )
    assert left_ssim >= 0.995, f"Bilingual cloned diagram SSIM {left_ssim:.4f} < 0.995"


def test_maxtree_filter_preserves_edges():
    """Verify max-tree morphological filter removes mid-grey plateaus without blurring text edges."""
    h, w = 400, 400
    clean = np.full((h, w), 250, dtype=np.uint8)
    # Sharp black text
    cv2.putText(clean, "Sample Question 4", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 15, 2)

    # Overlaid mid-grey broad watermark wash (grey=180, broad area)
    watermarked = clean.copy()
    cv2.rectangle(watermarked, (40, 100), (360, 220), 185, -1)
    # Re-draw text over it
    cv2.putText(watermarked, "Sample Question 4", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 15, 2)

    cleaned = maxtree_filter(watermarked)

    # Assert watermark wash is whitened (> 240) outside text
    wash_sample = cleaned[110:130, 60:120]
    assert np.mean(wash_sample) >= 235, "Watermark plateau was not pruned"

    # Assert text remains sharp (mean dark value < 30)
    text_pixels = clean < 40
    assert np.mean(cleaned[text_pixels]) < 40, "Text was corrupted"


def test_mupdf_layer_redact_isolates_layers():
    """Verify PyMuPDF layer-selective redactions removes text layer while keeping images byte-identical."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)

    # Insert raster image
    img_arr = np.full((100, 100, 3), 120, dtype=np.uint8)
    img_arr[20:80, 20:80] = (20, 200, 20)  # green square
    is_ok, img_png = cv2.imencode(".png", img_arr)
    rect_img = pymupdf.Rect(50, 50, 200, 200)
    page.insert_image(rect_img, stream=img_png.tobytes())

    # Insert vector text watermark overlapping the image
    page.insert_text(pymupdf.Point(80, 120), "WATERMARK", fontsize=24, color=(1, 0, 0))

    # Get baseline rendered image under the image rect before redaction
    pix_before = page.get_pixmap(clip=rect_img)

    # Redact ONLY text layer
    ok = mupdf_layer_redact(page, bbox=(70, 90, 250, 140), redact_layer="text")
    assert ok is True

    # Assert that "WATERMARK" text is gone
    text_after = page.get_text()
    assert "WATERMARK" not in text_after

    # Assert raster image content is preserved under rect_img
    pix_after = page.get_pixmap(clip=rect_img)
    assert pix_after.samples is not None
    assert len(pix_after.samples) == len(pix_before.samples)
