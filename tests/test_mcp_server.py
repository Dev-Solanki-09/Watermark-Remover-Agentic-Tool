"""Unit tests for the official FastMCP server tools."""

from pathlib import Path

import cv2
import pymupdf
import pytest

from tests.benchmarks.fixture_generator import generate_benchmark_suite
from wmclean.mcp_server import benchmark_page, inspect_watermark_layers, remove_watermark_auto


@pytest.fixture
def sample_test_pdf(tmp_path: Path) -> Path:
    """Generate a single-page PDF with a watermark for testing."""
    f = generate_benchmark_suite(quick=True)[0]
    pdf_path = tmp_path / "test_sample.pdf"
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=842)
    _, b = cv2.imencode(".png", f.watermarked_bgr)
    p.insert_image(p.rect, stream=b.tobytes())
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_inspect_watermark_layers(sample_test_pdf: Path):
    """inspect_watermark_layers must return diagnostic info and recommended tier."""
    res = inspect_watermark_layers(str(sample_test_pdf))
    assert "error" not in res
    assert res["total_pages"] == 1
    assert "pages" in res
    p1 = res["pages"][0]
    assert p1["page"] == 1
    assert "recommended_tier" in p1


def test_benchmark_page(sample_test_pdf: Path):
    """benchmark_page must return objective metrics including residual and topological invariance."""
    res = benchmark_page(str(sample_test_pdf), page_number=1)
    assert "error" not in res
    assert "residual_watermark_energy" in res
    assert "topological_betti_invariance" in res
    assert res["residual_watermark_energy"] <= 0.035
    assert res["topological_betti_invariance"]["passed"] is True


def test_remove_watermark_auto(sample_test_pdf: Path, tmp_path: Path):
    """remove_watermark_auto must execute cleaning and return success result."""
    out_path = tmp_path / "cleaned_out.pdf"
    res = remove_watermark_auto(str(sample_test_pdf), output_path=str(out_path))

    assert res["success"] is True
    assert res["exit_code"] == 0
    assert res["status"] == "CLEAN"
    assert out_path.exists()
