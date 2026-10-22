"""Idempotence and invariance tests for wmclean."""

import json
from pathlib import Path

import pikepdf
import pytest

from wmclean.orchestrator import Orchestrator
from wmclean.schemas import ExitCode, PageState


@pytest.fixture
def clean_sample_pdf(tmp_path: Path) -> Path:
    """Create a pristine digital document without watermarks."""
    pdf_p = tmp_path / "clean_document.pdf"
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(595, 842))
    page.Contents = pdf.make_stream(b"BT /F1 14 Tf 72 700 Td (Pristine Document Content) Tj ET")
    pdf.save(str(pdf_p))
    pdf.close()
    return pdf_p


def test_idempotence_clean_pdf(clean_sample_pdf: Path, tmp_path: Path):
    """Running wmclean on an already-clean or cleaned document must return UNTOUCHED for 100% of pages."""
    out1 = tmp_path / "pass1.pdf"
    rep1 = tmp_path / "rep1.json"

    # Pass 1
    orch1 = Orchestrator(
        input_path=str(clean_sample_pdf),
        output_path=str(out1),
        report_path=str(rep1),
    )
    code1 = orch1.execute()
    assert code1 == ExitCode.SUCCESS.value

    # Pass 2 on the output of pass 1
    out2 = tmp_path / "pass2.pdf"
    rep2 = tmp_path / "rep2.json"
    orch2 = Orchestrator(
        input_path=str(out1),
        output_path=str(out2),
        report_path=str(rep2),
    )
    code2 = orch2.execute()
    assert code2 == ExitCode.SUCCESS.value

    # Validate report 2: 100% of pages must be UNTOUCHED
    data2 = json.loads(rep2.read_text(encoding="utf-8"))
    pages2 = data2.get("pages", [])
    assert len(pages2) == 1
    assert pages2[0]["st"] == PageState.UNTOUCHED.value
