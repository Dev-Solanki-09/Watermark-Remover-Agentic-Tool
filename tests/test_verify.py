"""Unit tests for Phase 3: L4 Verifier & Escalation Engine (exact, raster, rollback)."""

import io
from pathlib import Path

import cv2
import numpy as np
import pikepdf
import pymupdf

from tests.fixture_gen import ALL_FIXTURE_GENERATORS
from wmclean.removers.vector_ops import annot_del
from wmclean.schemas import FailureCode, PageKind, PageState, RemoverMethod, RemoverParams
from wmclean.verify.engine import VerificationEngine
from wmclean.verify.exact import verify_exact
from wmclean.verify.raster import verify_raster


def test_verify_exact_pass_on_clean_removal(fixtures_dir: Path):
    """Verify exact passes when watermark is cleanly removed without touching body text."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as orig_pdf, pikepdf.open(str(wm_path)) as clean_pdf:
        # Clean page 0
        deleted = annot_del(clean_pdf.pages[0])
        assert len(deleted) >= 1

        result = verify_exact(orig_pdf, clean_pdf, 0)
        assert result.passed is True
        assert result.text_diff == 0
        assert result.failure_code is None


def test_verify_exact_rejects_text_modification(fixtures_dir: Path):
    """Verify exact catches and rejects modification of legitimate non-watermark body text."""
    clean_path = fixtures_dir / "clean_vector_verify.pdf"
    from tests.fixture_gen import create_seed_vector

    create_seed_vector(clean_path, pages=1)

    with pikepdf.open(str(clean_path)) as orig_pdf, pikepdf.open(str(clean_path)) as modified_pdf:
        # Intentionally alter content stream to delete a body text operator
        cs = pikepdf.parse_content_stream(modified_pdf.pages[0])
        filtered = []
        dropped = False
        for operands, operator in cs:
            if str(operator) in ("Tj", "TJ") and not dropped:
                dropped = True
                continue  # drop one text operator
            filtered.append((operands, operator))

        modified_pdf.pages[0].Contents = modified_pdf.make_stream(
            pikepdf.unparse_content_stream(filtered)
        )

        result = verify_exact(orig_pdf, modified_pdf, 0)
        assert result.passed is False
        assert result.text_diff > 0
        assert result.failure_code == FailureCode.ERR_VEC_VERIFY_BREACH


def test_verify_raster_pass():
    """Verify raster passes when watermark residual and foreground loss are below limits."""
    # Create white document with black text
    orig = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.putText(orig, "Hello World", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    # Add light watermark
    wm = orig.copy()
    cv2.putText(wm, "WATERMARK", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)

    # Cleaned image has watermark erased (turned white)
    clean = orig.copy()

    result = verify_raster(wm, clean)
    assert result.passed is True
    assert result.residual <= 0.035
    assert result.content_loss <= 0.020


def test_verify_raster_high_residual():
    """Verify raster rejects high watermark residual (R_wm > 0.035)."""
    orig = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.putText(orig, "Hello World", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    # Watermarked image
    wm = orig.copy()
    cv2.putText(wm, "WATERMARK", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)

    # Incomplete cleaning: dark residue remains
    bad_clean = wm.copy()

    mask = np.zeros((300, 300), dtype=bool)
    mask[170:220, 20:250] = True

    result = verify_raster(wm, bad_clean, wm_mask=mask)
    assert result.passed is False
    assert result.residual > 0.035
    assert result.failure_code == FailureCode.ERR_HIGH_RESIDUAL


def test_verify_raster_content_loss():
    """Verify raster rejects severe foreground content loss (L_fg > 0.020)."""
    orig = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.putText(
        orig, "Important Text Lines", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )

    # Over-aggressive cleaning erases legitimate text
    bad_clean = np.full((300, 300, 3), 255, dtype=np.uint8)

    result = verify_raster(orig, bad_clean)
    assert result.passed is False
    assert result.content_loss > 0.020
    assert result.failure_code == FailureCode.ERR_DIAGRAM_INTEGRITY


def test_verify_raster_opaque_occlusion(fixtures_dir: Path):
    """Verify raster flags opaque dark occlusion (FX_G19) with ERR_OPAQUE_OCCLUSION."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G19"](fixtures_dir)

    with pymupdf.open(str(wm_path)) as doc:
        pix = doc[0].get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        result = verify_raster(img, img)
        assert result.passed is False
        assert result.failure_code == FailureCode.ERR_OPAQUE_OCCLUSION


def test_verification_engine_escalation_and_rollback(fixtures_dir: Path):
    """VerificationEngine tries candidate removers, rolls back bad attempts, and accepts valid."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)

    def bad_remover(pdf, page, params):
        # A remover that destroys body text
        page.Contents = pdf.make_stream(b"")
        return 1

    def good_remover(pdf, page, params):
        return annot_del(page, params)

    engine = VerificationEngine()
    with pikepdf.open(str(wm_path)) as pdf:
        # Chain has bad_remover first, then good_remover
        chain = [
            (RemoverMethod.TEXT_OP_STRIP, bad_remover, RemoverParams()),
            (RemoverMethod.ANNOT_DEL, good_remover, RemoverParams()),
        ]

        state, applied, result = engine.apply_with_rollback(
            pdf,
            0,
            PageKind.VECTOR,
            chain,
        )

        # bad_remover must have been rolled back and good_remover succeeded
        assert state == PageState.CLEAN_EXACT
        assert applied == [RemoverMethod.ANNOT_DEL.value]
        assert result.passed is True
        assert pdf.pages[0].get("/Annots") is None
        # Page body text is intact
        buf = io.BytesIO()
        pdf.save(buf)
        with pymupdf.open(stream=buf.getvalue(), filetype="pdf") as doc:
            txt = doc[0].get_text()
            assert "Deterministic Verification in Document Systems" in txt
