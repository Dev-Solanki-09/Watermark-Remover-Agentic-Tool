"""Unit tests for SOTA open-source extensions:
1. Doxapy C++ SIMD ISauvola foreground text-stroke protection mask (M_fg).
2. Beer-Lambert optical density stain deconvolution (skimage.color.separate_stains).
3. Euclidean Chamfer Distance Transform localization for shifted watermarks.
4. PyMuPDF page.get_texttrace() 1ms vector probing and OCR layer protection.
"""

from __future__ import annotations

import cv2
import numpy as np
import pikepdf
import pymupdf

from wmclean.probe.structural import _extract_text_candidates
from wmclean.removers.chamfer_align import chamfer_align_template, extract_chamfer_edges
from wmclean.removers.stain_deconv import beer_lambert_stain_unmix, estimate_stain_vectors
from wmclean.verify.text_protect import compute_fg_protection_mask, measure_stroke_recall_loss

# ---------------------------------------------------------------------------
# 1. Doxapy C++ SIMD Foreground Text Protection Tests
# ---------------------------------------------------------------------------


def test_doxapy_fg_protection_mask():
    """Verify doxapy ISauvola extracts dark text strokes while excluding faint watermarks."""
    # Synthetic page: white background (255)
    img = np.full((120, 120), 255, dtype=np.uint8)

    # 1. Dark text stroke: "E = mc^2" (pixel luminance ~30)
    img[40:80, 20:30] = 30  # vertical text stroke
    img[55:65, 30:50] = 30  # horizontal stroke

    # 2. Faint watermark tint (luminance ~225)
    img[20:100, 60:110] = 225

    mask = compute_fg_protection_mask(img, window=35, k=0.18, max_fg_luminance=190)

    # Text strokes must be protected (mask == 255)
    assert np.all(mask[40:80, 20:30] == 255)
    assert np.all(mask[55:65, 30:50] == 255)

    # Faint watermark background must NOT be protected (mask == 0)
    assert np.all(mask[20:100, 60:110] == 0)

    # Clean background outside text must be 0
    assert mask[10, 10] == 0


def test_stroke_recall_loss_measurement():
    """Verify measure_stroke_recall_loss detects eroded vs preserved strokes."""
    orig = np.full((80, 80), 255, dtype=np.uint8)
    orig[30:50, 30:50] = 20  # dark text block

    # Cleaned image with text preserved
    clean_good = orig.copy()
    loss_good = measure_stroke_recall_loss(orig, clean_good)
    assert loss_good <= 0.020  # Invariant: stroke loss <= 2%

    # Cleaned image where text was erroneously bleached to white (255)
    clean_bad = orig.copy()
    clean_bad[30:50, 30:50] = 255  # bleached
    loss_bad = measure_stroke_recall_loss(orig, clean_bad)
    assert loss_bad > 0.90  # Severe loss detected


# ---------------------------------------------------------------------------
# 2. Beer-Lambert Optical Density Stain Deconvolution Tests
# ---------------------------------------------------------------------------


def test_beer_lambert_stain_unmix_red_stamp():
    """Verify Beer-Lambert deconvolution separates overlapping red stamp and black text."""
    # Create test RGB patch:
    # - Background: white paper [255, 255, 255]
    # - Foreground: black text toner [30, 30, 30]
    # - Watermark: red stamp [220, 50, 50]
    # - Overlap: optical densities add physically
    h, w = 60, 60
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Black text line at x: 10..20, y: 10..50
    img[10:50, 15:25] = [30, 30, 30]

    # Red stamp circle/square at x: 15..45, y: 25..45
    stamp_mask = np.zeros((h, w), dtype=np.uint8)
    stamp_mask[25:45, 15:45] = 255

    # Simulate physical red ink absorption in non-text regions
    img[25:45, 25:45] = [220, 50, 50]

    # In overlap region [25:45, 15:25], text toner and red ink co-exist
    # Text toner absorbs heavily across all channels; red ink absorbs green & blue
    img[25:45, 15:25] = [25, 8, 8]

    # Apply Beer-Lambert stain deconvolution
    cleaned = beer_lambert_stain_unmix(
        rgb_img=img,
        mask=stamp_mask,
        wm_rgb=(220, 50, 50),
        attenuation=1.0,
        protect_text=True,
    )

    # 1. Unobstructed stamp region [25:45, 25:45] should be whitened back to clean background
    unobstructed_red_after = cleaned[30:40, 30:40]
    assert np.mean(unobstructed_red_after) > 220, (
        f"Red stamp not whitened: mean={np.mean(unobstructed_red_after)}"
    )

    # 2. Text outside stamp [10:20, 15:25] must remain dark and untouched
    text_outside = cleaned[12:18, 18:22]
    assert np.mean(text_outside) < 50, f"Clean text corrupted: mean={np.mean(text_outside)}"

    # 3. Text in overlap region [30:40, 18:22] must remain dark and legible
    text_overlap = cleaned[30:40, 18:22]
    assert np.mean(text_overlap) < 90, f"Overlap text bleached: mean={np.mean(text_overlap)}"


def test_estimate_stain_vectors_orthogonality():
    """Verify estimated stain vectors form an invertible, valid 3x3 basis."""
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    img[10:20, 10:20] = [30, 30, 30]  # black ink
    img[30:40, 30:40] = [50, 60, 220]  # blue stamp

    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[30:40, 30:40] = 255

    v_ink, v_wm, v_res = estimate_stain_vectors(img, mask=mask)
    M = np.stack([v_ink, v_wm, v_res], axis=0)

    # Determinant must be non-zero (invertible matrix)
    det = np.linalg.det(M)
    assert abs(det) > 1e-3, f"Singular stain matrix: det={det}"


# ---------------------------------------------------------------------------
# 3. Chamfer Distance Transform Localization Tests
# ---------------------------------------------------------------------------


def test_chamfer_align_template_clean_and_cluttered():
    """Verify Chamfer distance transform locates shifted watermarks even under text clutter."""
    scene = np.full((160, 160), 255, dtype=np.uint8)

    # Create watermark template: a 30x40 rectangle with a circle inside
    template = np.full((40, 50), 255, dtype=np.uint8)
    cv2.rectangle(template, (5, 5), (45, 35), 0, 2)
    cv2.circle(template, (25, 20), 10, 0, 1)

    t_edges = extract_chamfer_edges(template)

    # Place template at known position (target_x=60, target_y=50)
    target_x, target_y = 60, 50
    scene[target_y : target_y + 40, target_x : target_x + 50] = template

    # Add crossing text clutter
    cv2.putText(scene, "CONFIDENTIAL REPORT 2026", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 2)
    cv2.putText(scene, "SECTION 4.2 WATERMARK TEST", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)

    # Align using Chamfer Distance Transform
    best_x, best_y, min_dist = chamfer_align_template(scene, t_edges)

    # Location must match within 1 pixel
    assert abs(best_x - target_x) <= 1, f"Expected x={target_x}, got {best_x}"
    assert abs(best_y - target_y) <= 1, f"Expected y={target_y}, got {best_y}"
    assert min_dist < 1.0, f"Distance error too high: {min_dist}"


# ---------------------------------------------------------------------------
# 4. PyMuPDF page.get_texttrace() 1ms Probing Tests
# ---------------------------------------------------------------------------


def test_get_texttrace_vector_probing_and_ocr_protection(tmp_path):
    """Verify page.get_texttrace() extracts diagonal watermarks and protects OCR layer."""
    pdf_path = tmp_path / "test_trace.pdf"

    # Create synthetic PDF using pymupdf with:
    # 1. Normal horizontal text
    # 2. Diagonal translucent watermark
    # 3. Invisible OCR text (render_mode=3)
    doc = pymupdf.open()
    page = doc.new_page(width=500, height=500)

    # Normal text
    page.insert_text((50, 80), "Standard Document Paragraph Text", fontsize=12)

    # Diagonal watermark (rotate 90 or 270)
    page.insert_text((150, 400), "DRAFT WATERMARK", fontsize=28, rotate=90, color=(0.7, 0.7, 0.7))

    # Invisible OCR layer (render_mode=3)
    page.insert_text((50, 250), "Scanned OCR Invisible Text Layer", fontsize=11, render_mode=3)

    doc.save(str(pdf_path))
    doc.close()

    # Open with pikepdf + pymupdf for structural probing
    with pikepdf.open(str(pdf_path)) as pike_doc, pymupdf.open(str(pdf_path)) as fitz_doc:
        pike_page = pike_doc.pages[0]
        fitz_page = fitz_doc[0]

        candidates = _extract_text_candidates(
            pike_page, page_num=1, total_pages=1, fitz_page=fitz_page
        )

        # 1. Diagonal watermark must be detected
        wm_cand = [
            c
            for c in candidates
            if "DRAFT" in c.norm_str or "WATERMARK" in c.norm_str or abs(c.rot_deg) > 5.0
        ]
        assert len(wm_cand) >= 1, "Failed to detect diagonal watermark candidate via get_texttrace"
        assert abs(wm_cand[0].rot_deg) > 5.0

        # 2. Invisible OCR text must NOT be extracted as a candidate (protected)
        ocr_cand = [c for c in candidates if "OCR" in c.norm_str or c.tr == 3]
        assert len(ocr_cand) == 0, (
            "Invisible OCR layer was mistakenly flagged as a watermark candidate!"
        )
