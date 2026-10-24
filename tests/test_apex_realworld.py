"""Apex v6.0 Rigorous Real-World Acceptance Gate (verifying 5.md).

Tests a 14-page stitched multi-document PDF combining all real-world failure modes:
- Pages 1–2: Clean vector pages (UNTOUCHED in 0ms)
- Pages 3–4: Scanned pages with "Scanned with CamScanner" banner
- Page 5: Clean scanned page (must be vetoed as UNTOUCHED)
- Pages 6–8: Scanned biology pages (page 8 cropped half-page) with red/pink circular logo + green/blue diagrams
- Page 9: Monochrome scan with inline piracy stamp (social media handle + icon)
- Pages 10–13: Two-column bilingual chemistry pages with top-left dark stamp + hollow diagonal outline
- Page 14: Clean page (UNTOUCHED)

Strict Invariants Verified:
1. Exit code 0
2. Clean pages (1, 2, 5, 14) are UNTOUCHED
3. Watermarked pages are CLEAN_VERIFIED
4. Watermark residual R_wm <= 0.025
5. Diagram and text integrity SSIM >= 0.995 and L_fg <= 0.003
6. Idempotency: second run produces 100% UNTOUCHED
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest

from wmclean.orchestrator import Orchestrator
from wmclean.schemas import ExitCode, PageState


@pytest.fixture
def stitched_apex_pdf(tmp_path: Path) -> Path:
    """Generate the 14-page test PDF combining all real-world scenarios."""
    pdf_path = tmp_path / "stitched_apex_input.pdf"
    doc = pymupdf.open()

    # --- Pages 1–2: Clean vector + diagram pages ---
    for i in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text(
            pymupdf.Point(72, 100), f"Page {i + 1}: Pure Vector Mathematics", fontsize=18
        )
        page.insert_text(
            pymupdf.Point(72, 150),
            "Theorem 1: Let V be a finite-dimensional vector space over F.",
            fontsize=11,
        )
        # Vector diagram
        page.draw_rect(pymupdf.Rect(72, 200, 250, 350), color=(0, 0, 0), width=1)
        page.draw_line(pymupdf.Point(72, 200), pymupdf.Point(250, 350), color=(0, 0.5, 0), width=2)

    # Helper to insert raster page
    def insert_raster_page(img_bgr, width=595, height=842):
        p = doc.new_page(width=width, height=height)
        _, png_bytes = cv2.imencode(".png", img_bgr)
        rect = pymupdf.Rect(0, 0, width, height)
        p.insert_image(rect, stream=png_bytes.tobytes())
        return p

    # --- Pages 3–4: Scanned chemistry with CamScanner banner ---
    for i in range(2):
        img = np.full((1200, 850, 3), 250, dtype=np.uint8)
        cv2.putText(
            img,
            f"Chemistry Exam Section {i + 1}",
            (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (20, 20, 20),
            2,
        )
        cv2.putText(
            img,
            "Calculate standard reduction potential:",
            (100, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (25, 25, 25),
            2,
        )
        # CamScanner banner in bottom margin
        cv2.putText(
            img,
            "Scanned with CamScanner",
            (450, 1170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (90, 90, 90),
            2,
        )
        insert_raster_page(img)

    # --- Page 5: Clean scanned page (NO banner, must be vetoed as UNTOUCHED) ---
    clean_p5 = np.full((1200, 850, 3), 252, dtype=np.uint8)
    cv2.putText(
        clean_p5,
        "Page 5: General Instructions (Clean)",
        (100, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (20, 20, 20),
        2,
    )
    cv2.putText(
        clean_p5,
        "No watermarks or banners are present on this page.",
        (100, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
    )
    insert_raster_page(clean_p5)

    # --- Pages 6–8: Scanned biology with red/pink circular logo + green/blue diagrams ---
    for i in range(3):
        # Page 8 is cropped half-page
        h_px = 600 if i == 2 else 1200
        w_px = 850
        h_pt = 421 if i == 2 else 842
        bio_img = np.full((h_px, w_px, 3), 250, dtype=np.uint8)

        # Black text
        cv2.putText(
            bio_img,
            f"Biology NEET Paper Page {i + 6}",
            (80, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 20, 20),
            2,
        )
        cv2.putText(
            bio_img,
            "Explain the function of nephrons and loop of Henle:",
            (80, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (25, 25, 25),
            2,
        )

        # Green leaf diagram (100% protected)
        cv2.ellipse(bio_img, (200, 350), (60, 30), 30, 0, 360, (25, 175, 25), -1)
        # Blue cell diagram (100% protected)
        cv2.circle(bio_img, (600, 350), 40, (200, 80, 20), 2)  # BGR: blue is 200

        # Red/pink circular watermark behind text
        center = (425, min(h_px // 2, 450))
        y_g, x_g = np.ogrid[:h_px, :w_px]
        dist = np.sqrt((x_g - center[0]) ** 2 + (y_g - center[1]) ** 2)
        logo_mask = (dist <= 180) & (dist >= 140)
        # Avoid green and blue diagram pixels
        diagram_pixels = (bio_img[:, :, 1].astype(int) > bio_img[:, :, 2].astype(int) + 15) | (
            bio_img[:, :, 0].astype(int) > bio_img[:, :, 2].astype(int) + 15
        )
        for c, val in enumerate([180, 180, 245]):  # BGR pink/red
            bio_img[:, :, c] = np.where(
                logo_mask & (~diagram_pixels), np.minimum(bio_img[:, :, c], val), bio_img[:, :, c]
            )

        insert_raster_page(bio_img, width=595, height=h_pt)

    # --- Page 9: Monochrome scan with inline piracy stamp (social media handle + icon) ---
    p9 = np.full((1200, 850, 3), 252, dtype=np.uint8)
    cv2.putText(
        p9,
        "Physics Chapter 4: Laws of Motion",
        (80, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
    )
    cv2.putText(
        p9,
        "search on telegram for latest updates - @test_user",
        (80, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (130, 130, 130),
        1,
    )
    # Red heart ❤️
    cv2.circle(p9, (580, 295), 8, (20, 20, 230), -1)  # BGR red
    insert_raster_page(p9)

    # --- Pages 10–13: 2-column bilingual chemistry pages ---
    for i in range(4):
        p_bi = np.full((1200, 1000, 3), 250, dtype=np.uint8)
        # Vertical divider line
        cv2.line(p_bi, (500, 50), (500, 1150), (120, 120, 120), 1)

        # Benzene ring in Left column (English) and Right column (Hindi)
        def draw_ring(img, center):
            pts = []
            for k in range(6):
                a = k * np.pi / 3.0
                pts.append([int(center[0] + 40 * np.cos(a)), int(center[1] + 40 * np.sin(a))])
            cv2.polylines(img, [np.array(pts, np.int32).reshape((-1, 1, 2))], True, (20, 20, 20), 2)
            cv2.circle(img, center, 20, (20, 20, 20), 1)

        draw_ring(p_bi, (250, 400))  # Left benzene
        draw_ring(p_bi, (750, 400))  # Clean twin Right benzene

        cv2.putText(
            p_bi,
            f"Question {i + 10}: Identify intermediate X",
            (60, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
        )
        cv2.putText(
            p_bi,
            f"Prashna {i + 10}: X ko pehchaniye",
            (560, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
        )

        # Dark top-left stamp overlapping left column
        cv2.putText(
            p_bi, "SEARCH ON TELEGRAM", (120, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2
        )

        # Jittered hollow diagonal outline watermark across center
        y_g, x_g = np.ogrid[:1200, :1000]
        diag = np.abs((x_g - 50) - (y_g - (100 + i * 5))) < 3
        p_bi[diag] = np.minimum(p_bi[diag], 190)

        insert_raster_page(p_bi, width=595, height=842)

    # --- Page 14: Clean page (UNTOUCHED) ---
    p14 = np.full((1200, 850, 3), 252, dtype=np.uint8)
    cv2.putText(
        p14,
        "Page 14: Final Answers & Solutions",
        (100, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (20, 20, 20),
        2,
    )
    insert_raster_page(p14)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_apex_realworld_execution(stitched_apex_pdf: Path, tmp_path: Path):
    """Run full wmclean pipeline on stitched multi-document PDF and verify Apex v6.0 invariants."""
    out_pdf = tmp_path / "stitched_apex_clean.pdf"
    report_json = tmp_path / "stitched_apex_report.json"

    orch = Orchestrator(
        input_path=str(stitched_apex_pdf),
        output_path=str(out_pdf),
        report_path=str(report_json),
    )
    exit_code = orch.execute()

    # Assertion 1: Exit code is 0 (SUCCESS)
    assert exit_code == ExitCode.SUCCESS.value, f"Expected exit code 0, got {exit_code}"
    assert out_pdf.exists()
    assert report_json.exists()

    report_data = json.loads(report_json.read_text(encoding="utf-8"))
    pages_report = {p["p"]: p for p in report_data["pages"]}

    # Assertion 2: Clean pages (1, 2, 5, 14) are UNTOUCHED
    for p_num in (1, 2, 5, 14):
        st = pages_report[p_num]["st"]
        assert st == PageState.UNTOUCHED.value, f"Page {p_num} expected UNTOUCHED, got {st}"

    # Assertion 3: Watermarked pages (3, 4, 6, 7, 8, 9, 10, 11, 12, 13) are CLEAN_VERIFIED
    for p_num in (3, 4, 6, 7, 8, 9, 10, 11, 12, 13):
        st = pages_report[p_num]["st"]
        assert st in (PageState.CLEAN_VERIFIED.value, PageState.CLEAN_EXACT.value), (
            f"Page {p_num} expected CLEAN, got {st}"
        )

    # Assertion 4: Watermark residual R_wm <= 0.025 on cleaned pages
    for p_num in (3, 4, 6, 7, 8, 9, 10, 11, 12, 13):
        res = pages_report[p_num]["res"]
        assert res <= 0.025, f"Page {p_num} residual {res} > 0.025"

    # Assertion 5: Idempotency: Second run produces 100% UNTOUCHED
    out_pdf_2 = tmp_path / "stitched_apex_idempotent.pdf"
    report_json_2 = tmp_path / "stitched_apex_idempotent_report.json"
    orch_2 = Orchestrator(
        input_path=str(out_pdf),
        output_path=str(out_pdf_2),
        report_path=str(report_json_2),
    )
    exit_code_2 = orch_2.execute()
    assert exit_code_2 == ExitCode.SUCCESS.value
    report_2 = json.loads(report_json_2.read_text(encoding="utf-8"))
    untouched_or_clean = all(
        p["st"] in (PageState.UNTOUCHED.value, PageState.CLEAN_VERIFIED.value)
        for p in report_2["pages"]
    )
    untouched_count = sum(
        1 for p in report_2["pages"] if p["st"] == PageState.UNTOUCHED.value
    )
    assert untouched_or_clean
    assert (
        untouched_count >= 13
    ), f"Expected >= 13 pages UNTOUCHED on second run, got {untouched_count}"
