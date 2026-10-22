"""Unit test verifying native 300 DPI scan resolution preservation."""

from pathlib import Path
import cv2
import numpy as np
import pikepdf
import pymupdf
import pytest

from wmclean.verify.engine import _detect_page_dpi


def test_detect_page_dpi_300(tmp_path: Path):
    # A4 at 300 DPI is ~2480 x 3508 pixels
    w_px, h_px = 2480, 3508
    w_pt, h_pt = 595.0, 842.0

    # Create synthetic 300 DPI scan
    scan = np.full((h_px, w_px, 3), 255, dtype=np.uint8)
    cv2.putText(scan, "300 DPI SCAN", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)

    pdf_p = tmp_path / "scan_300dpi.pdf"
    doc = pymupdf.open()
    p = doc.new_page(width=w_pt, height=h_pt)
    _, buf = cv2.imencode(".png", scan)
    p.insert_image(p.rect, stream=buf.tobytes())
    doc.save(str(pdf_p))
    doc.close()

    # Verify _detect_page_dpi
    with pymupdf.open(str(pdf_p)) as verify_doc:
        page = verify_doc[0]
        detected_dpi = _detect_page_dpi(page, default_dpi=150, max_dpi=300)
        assert detected_dpi == 300, f"Expected 300 DPI, detected {detected_dpi}"
