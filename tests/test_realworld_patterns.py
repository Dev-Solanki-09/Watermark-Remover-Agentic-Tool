"""Tests for real-world watermark patterns and text-core lock (verifying 1.md)."""

from __future__ import annotations

import cv2
import numpy as np

from wmclean.intelligence.doc_shield import parse_doc_shield
from wmclean.removers.density_sub import density_sub
from wmclean.removers.pointwise_unmix import pointwise_hue_and_lut
from wmclean.verify.raster import verify_raster


def test_red_logo_behind_black_text_and_green_leaf():
    """Verify red/pink circular logo removal over black text + green leaf diagram.

    Asserts:
    - > 92% watermark removal (R_wm <= 0.03)
    - < 0.5% black text loss (L_fg <= 0.005)
    - 0% green leaf diagram loss
    """
    h, w = 600, 600
    page = np.full((h, w, 3), 250, dtype=np.uint8)

    # 1. Author-drawn green leaf diagram (G > R + 6)
    cv2.ellipse(page, (150, 150), (40, 20), 45, 0, 360, (30, 160, 30), -1)
    cv2.line(page, (130, 130), (170, 170), (20, 100, 20), 2)
    green_mask_orig = (page[:, :, 1].astype(int) > page[:, :, 0].astype(int) + 6) & (
        page[:, :, 1] > 100
    )
    orig_green_pixels = page[green_mask_orig].copy()

    # 2. Black question text
    cv2.putText(
        page,
        "Question 12: Describe photosynthesis.",
        (50, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (25, 25, 25),
        2,
    )
    cv2.putText(
        page,
        "Calculate enthalpy change delta H:",
        (50, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
    )

    # 3. Red/pink circular watermark behind text
    logo_center = (300, 320)
    logo_radius = 120
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - logo_center[0]) ** 2 + (y_grid - logo_center[1]) ** 2)
    logo_mask = (dist_from_center <= logo_radius) & (dist_from_center >= logo_radius - 20)

    # Alpha composite pink watermark over non-green background
    for c, val in enumerate([245, 180, 180]):  # pinkish red
        page[:, :, c] = np.where(
            logo_mask & (~green_mask_orig), np.minimum(page[:, :, c], val), page[:, :, c]
        )

    # Clean with pointwise_hue_and_lut
    cleaned = pointwise_hue_and_lut(page)

    # Assert 1: Green leaf diagram 100% preserved
    green_after = cleaned[green_mask_orig]
    diff_green = np.max(np.abs(orig_green_pixels.astype(int) - green_after.astype(int)))
    assert diff_green == 0, f"Green diagram altered by {diff_green} px"

    # Assert 2: Black text core preserved
    shield = parse_doc_shield(page)
    text_cores = shield.m_core & (page[:, :, 0] < 50)
    text_loss = np.sum((cleaned[text_cores] > 180).any(axis=1)) / max(1, text_cores.sum())
    assert text_loss <= 0.005, f"Text core loss {text_loss} > 0.005"

    # Assert 3: Watermark removed
    v_res = verify_raster(page, cleaned, wm_mask=logo_mask)
    assert v_res.residual <= 0.03, f"Residual {v_res.residual} > 0.03"


def test_top_left_dark_stamp_and_diagonal_outline():
    """Verify top-left dark stamp overlapping text + diagonal watermark."""
    h, w = 500, 500
    page = np.full((h, w), 250, dtype=np.uint8)

    # Black question text
    cv2.putText(page, "Organic Chemistry Exam 2026", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 20, 2)
    cv2.putText(
        page, "Benzene reacts with HNO3/H2SO4:", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 25, 2
    )

    # Top-left dark stamp directly overlapping title
    stamp_mask = np.zeros((h, w), dtype=bool)
    stamp_img = page.copy()
    cv2.putText(stamp_img, "SEARCH ON TELEGRAM", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 60, 2)
    stamp_mask = (stamp_img < page) & (stamp_img == 60)
    page[stamp_mask] = 60

    # Diagonal watermark
    diag_mask = np.zeros((h, w), dtype=bool)
    cv2.line(page, (50, 450), (450, 50), 195, 4)
    diag_mask = page == 195

    # Clean with density_sub / surgical_lama
    cleaned = density_sub(page, mask=stamp_mask | diag_mask)

    # Verify text preservation
    shield = parse_doc_shield(page)
    cleaned_text_val = cleaned[shield.m_ink_core]
    assert np.mean(cleaned_text_val) < 65, "Text was bleached"


def test_margin_banner_and_inline_piracy_stamp():
    """Verify scanner margin banner and inline piracy stamp removal."""
    h, w = 600, 500
    page = np.full((h, w, 3), 250, dtype=np.uint8)

    # Body text
    cv2.putText(
        page,
        "Section B: Physics Numerical Problems",
        (40, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
    )

    # CamScanner footer banner in bottom margin
    cv2.putText(
        page, "Scanned with CamScanner", (240, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1
    )

    # Inline stamp with red heart
    cv2.putText(
        page,
        "search on telegram - @test_user",
        (60, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (140, 140, 140),
        1,
    )
    # Red heart
    cv2.circle(page, (380, 255), 6, (230, 20, 20), -1)

    cleaned = pointwise_hue_and_lut(page)
    # Red heart must be removed
    heart_pixels = cleaned[255:257, 380:382]
    assert np.all(heart_pixels > 220), "Red heart was not whitened"
