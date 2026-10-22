"""Unit tests for Tier 0 pure vector AST stripper."""

from pathlib import Path

import pikepdf
import pytest

from wmclean.removers.ast_stripper import strip_vector_watermarks_ast


@pytest.fixture
def pdf_with_annot_watermark(tmp_path: Path) -> Path:
    """Create a digital PDF with authentic text and a /Watermark annotation."""
    pdf_p = tmp_path / "annot_wm.pdf"
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(595, 842))

    # Add text stream
    stream_content = b"BT /F1 12 Tf 72 712 Td (Authentic Document Body Text) Tj ET"
    page.Contents = pdf.make_stream(stream_content)

    # Add /Watermark annotation
    annot = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name("/Annot"),
            Subtype=pikepdf.Name("/Watermark"),
            Rect=pikepdf.Array([100, 100, 400, 400]),
            Contents=pikepdf.String("CONFIDENTIAL WATERMARK"),
        )
    )
    page.Annots = pdf.make_indirect(pikepdf.Array([annot]))

    pdf.save(str(pdf_p))
    pdf.close()
    return pdf_p


def test_ast_stripper_removes_annot_losslessly(pdf_with_annot_watermark: Path):
    pdf = pikepdf.open(str(pdf_with_annot_watermark))
    page = pdf.pages[0]

    # Pre-check: annotation exists
    assert "/Annots" in page
    assert len(page.Annots) == 1

    # Execute Tier 0
    modified, actions = strip_vector_watermarks_ast(pdf, 0)

    assert modified is True
    assert len(actions) >= 1
    assert "Pruned 1 watermark annotation" in actions[0]

    # Post-check: annotation is completely gone
    assert "/Annots" not in page or len(page.Annots) == 0

    # Invariant: text stream is 100% intact
    text_data = page.Contents.read_bytes()
    assert b"Authentic Document Body Text" in text_data

    pdf.close()
