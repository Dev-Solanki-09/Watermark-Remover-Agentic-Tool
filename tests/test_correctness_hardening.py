"""Hardened correctness and fail-closed security tests."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pikepdf
import pymupdf

from tests.fixture_gen import ALL_FIXTURE_GENERATORS
from wmclean.orchestrator import Orchestrator
from wmclean.schemas import FailureCode, PageKind, PageState, RemoverMethod
from wmclean.verify.engine import VerificationEngine
from wmclean.verify.exact import verify_exact
from wmclean.verify.raster import verify_raster


def test_noop_remover_cannot_pass_verification():
    """A remover that does not remove the watermark must fail verification with high residual."""
    orig = np.full((300, 300, 3), 255, dtype=np.uint8)
    orig[100:200, 100:200] = 120  # watermark stamp

    # No-op remover returns original unmodified
    cleaned = orig.copy()

    res = verify_raster(orig, cleaned)
    assert res.passed is False
    assert res.residual > 0.035


def test_injected_verifier_exception_fails_closed():
    """Any unexpected exception inside verifiers must fail closed (passed=False)."""
    orig = np.full((100, 100, 3), 255, dtype=np.uint8)
    clean = np.full((100, 100, 3), 255, dtype=np.uint8)

    with patch("wmclean.verify.raster._to_grey", side_effect=RuntimeError("Simulated memory fault")):
        res = verify_raster(orig, clean)
        assert res.passed is False
        assert "failed-closed" in (res.reason or "")
        assert res.failure_code == FailureCode.ERR_DIAGRAM_INTEGRITY


def test_legitimate_body_text_containing_keywords_protected():
    """Legitimate body text containing words like 'CONFIDENTIAL' must trigger verification failure if lost."""
    import io

    doc_orig = pymupdf.open()
    page_orig = doc_orig.new_page(width=595, height=842)
    page_orig.insert_text(
        pymupdf.Point(72, 100),
        "Notice: This patient medical history is strictly CONFIDENTIAL under HIPAA regulations.",
        fontsize=12,
    )
    b_orig = doc_orig.tobytes()

    doc_tampered = pymupdf.open()
    page_tampered = doc_tampered.new_page(width=595, height=842)
    # The text was wiped out by an overly aggressive regex/remover
    page_tampered.insert_text(pymupdf.Point(72, 100), "Notice: ", fontsize=12)
    b_tampered = doc_tampered.tobytes()

    with pikepdf.open(io.BytesIO(b_orig)) as p_orig, pikepdf.open(io.BytesIO(b_tampered)) as p_tamp:
        res = verify_exact(p_orig, p_tamp, page_idx=0)
        assert res.passed is False
        assert res.text_diff > 0
        assert "text modified" in (res.reason or "").lower()




def test_transactional_rollback_on_failed_remover():
    """When a remover damages document text, apply_with_rollback must rollback transactionally."""
    pdf = pikepdf.new()
    pdf.add_blank_page(page_size=(595, 842))
    orig_text_stream = b"BT /F1 12 Tf 72 700 Td (Important Legal Clause 4.2) Tj ET"
    pdf.pages[0].Contents = pdf.make_stream(orig_text_stream)

    def destructive_remover(page, params):
        page.Contents = pdf.make_stream(b"BT /F1 12 Tf 72 700 Td () Tj ET")
        return True

    engine = VerificationEngine()
    st, applied, v_res = engine.apply_with_rollback(
        pdf=pdf,
        page_idx=0,
        page_kind=PageKind.VECTOR,
        remover_funcs=[(RemoverMethod.TEXT_OP_STRIP, destructive_remover, {})],
    )

    # Must rollback because verify_exact fails on lost text
    assert st in (PageState.FLAGGED, PageState.UNVERIFIED)
    assert len(applied) == 0
    # Original text must be restored in pdf.pages[0].Contents
    restored_bytes = bytes(pdf.pages[0].Contents.read_bytes())
    assert b"Important Legal Clause 4.2" in restored_bytes


def test_crash_safe_resume_loads_checkpoint(fixtures_dir: Path, tmp_path: Path):
    """Orchestrator(resume=True) must load intermediate checkpoint state and finish cleanly."""
    wm_path, _, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    out_pdf = tmp_path / "out_resume.pdf"
    rep_json = tmp_path / "rep_resume.json"

    ckpt_pdf = out_pdf.with_name(f"{out_pdf.name}.ckpt.pdf")
    state_file = out_pdf.with_name(f"{out_pdf.name}.wmclean_state.json")

    # Seed partial state file and intermediate checkpoint
    with pikepdf.open(str(wm_path)) as p_in:
        p_in.save(str(ckpt_pdf))

    state_data = {
        "completed_pages": [1],
        "page_reports": [
            {
                "p": 1,
                "st": "CLEAN_EXACT",
                "cl": "cl_0",
                "m": ["annot_del"],
                "err": None,
                "res": 0.0,
                "loss": 0.0,
                "ms": 10,
                "crop": None,
            }
        ],
        "llm_calls": 0,
        "llm_tok": 0,
    }
    state_file.write_text(json.dumps(state_data), encoding="utf-8")

    orch = Orchestrator(
        input_path=str(wm_path),
        output_path=str(out_pdf),
        report_path=str(rep_json),
        resume=True,
    )
    code = orch.execute()
    assert code == 0
    assert out_pdf.exists()
    assert not state_file.exists()  # Cleaned up upon successful completion
    assert not ckpt_pdf.exists()  # Cleaned up upon successful completion
