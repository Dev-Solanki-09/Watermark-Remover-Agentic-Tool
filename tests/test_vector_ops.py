"""Unit tests for Phase 2: L3 Exact Vector & Structural Removers (FX_G01–FX_G09, FX_M01, FX_M04)."""

import io
from pathlib import Path

import pikepdf
import pymupdf

from tests.fixture_gen import ALL_FIXTURE_GENERATORS
from wmclean.removers.vector_ops import (
    annot_del,
    artifact_strip,
    clip_scope_strip,
    decrypt,
    ocg_strip,
    pattern_sh_strip,
    rev_rollback,
    smask_gs_strip,
    text_op_strip,
    vec_path_strip,
    xobj_do_strip,
)
from wmclean.schemas import RemoverParams


def _get_page_text(pdf: pikepdf.Pdf, page_idx: int) -> str:
    """Helper to extract text from a pikepdf page via PyMuPDF without writing to disk."""
    buf = io.BytesIO()
    pdf.save(buf)
    buf.seek(0)
    with pymupdf.open(stream=buf.getvalue(), filetype="pdf") as doc:
        return doc[page_idx].get_text()


def test_annot_del_g01(fixtures_dir: Path):
    """FX_G01: Delete annotation watermark while preserving all body text."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        text_before = _get_page_text(pdf, 0)
        assert pdf.pages[0].get("/Annots") is not None

        deleted = annot_del(pdf.pages[0])
        assert len(deleted) >= 1
        assert pdf.pages[0].get("/Annots") is None

        text_after = _get_page_text(pdf, 0)
        # Invariant: body text preserved, annotation stamp text removed
        assert "Deterministic Verification in Document Systems" in text_after
        if "APPROVED" in text_before or "CONFIDENTIAL" in text_before:
            assert "APPROVED" not in text_after and "CONFIDENTIAL" not in text_after


def test_ocg_strip_g02(fixtures_dir: Path):
    """FX_G02: Strip OCG marked content watermark layer."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G02"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        text_before = _get_page_text(pdf, 0)
        assert "CONFIDENTIAL OCG" in text_before

        stripped = ocg_strip(pdf, pdf.pages[0])
        assert len(stripped) >= 1

        text_after = _get_page_text(pdf, 0)
        assert "CONFIDENTIAL OCG" not in text_after
        assert "Deterministic Verification in Document Systems" in text_after


def test_artifact_strip_g03(fixtures_dir: Path):
    """FX_G03: Strip marked content /Artifact (/Subtype /Watermark) blocks."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G03"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        text_before = _get_page_text(pdf, 0)
        assert "DRAFT ARTIFACT" in text_before

        count = artifact_strip(pdf, pdf.pages[0])
        assert count >= 1

        text_after = _get_page_text(pdf, 0)
        assert "DRAFT ARTIFACT" not in text_after
        assert "Deterministic Verification in Document Systems" in text_after


def test_text_op_strip_g04(fixtures_dir: Path):
    """FX_G04: Remove diagonal text watermark while preserving standard body text."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G04"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        text_before = _get_page_text(pdf, 0)
        assert "CONFIDENTIAL" in text_before

        count = text_op_strip(pdf, pdf.pages[0])
        assert count >= 1

        text_after = _get_page_text(pdf, 0)
        assert "CONFIDENTIAL" not in text_after
        assert "Deterministic Verification in Document Systems" in text_after


def test_clip_scope_strip_g05(fixtures_dir: Path):
    """FX_G05: Excise clipping-path shaded watermark box."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G05"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        count = clip_scope_strip(pdf, pdf.pages[0])
        assert count >= 1

        text_after = _get_page_text(pdf, 0)
        assert "Deterministic Verification in Document Systems" in text_after


def test_xobj_do_strip_g06(fixtures_dir: Path):
    """FX_G06: Strip Form XObject watermark Do calls and clean resources."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G06"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        res = pdf.pages[0].get("/Resources")
        assert res and "/XObject" in res
        assert "/FmWatermark" in res["/XObject"]

        count = xobj_do_strip(pdf, pdf.pages[0], RemoverParams(target_xobj_names=["/FmWatermark"]))
        assert count >= 1

        res_after = pdf.pages[0].get("/Resources")
        xobj_after = res_after.get("/XObject") if res_after else {}
        assert "/FmWatermark" not in xobj_after

        text_after = _get_page_text(pdf, 0)
        assert "Deterministic Verification in Document Systems" in text_after


def test_vec_path_strip_g07(fixtures_dir: Path):
    """FX_G07: Replace untagged vector path paint operations with 'n'."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G07"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        count = vec_path_strip(pdf, pdf.pages[0])
        assert count >= 1

        text_after = _get_page_text(pdf, 0)
        assert "Deterministic Verification in Document Systems" in text_after


def test_pattern_sh_strip_g08(fixtures_dir: Path):
    """FX_G08: Remove pattern / shading rectangle fill."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G08"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        count = pattern_sh_strip(pdf, pdf.pages[0])
        assert count >= 1

        text_after = _get_page_text(pdf, 0)
        assert "Deterministic Verification in Document Systems" in text_after


def test_smask_gs_strip_g09(fixtures_dir: Path):
    """FX_G09: Strip soft-mask / blend-mode watermark overlay."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G09"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        count = smask_gs_strip(pdf, pdf.pages[0])
        assert count >= 1

        text_after = _get_page_text(pdf, 0)
        assert "Deterministic Verification in Document Systems" in text_after


def test_rev_rollback_m01(fixtures_dir: Path):
    """FX_M01: Rollback incremental revision to restore clean document."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_M01"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        # wm_path has an incremental revision containing an annotation
        assert pdf.pages[0].get("/Annots") is not None

        clean_pdf = rev_rollback(pdf, file_path=wm_path)
        assert clean_pdf is not None
        assert len(clean_pdf.pages) == len(pdf.pages)
        # In the pre-watermark revision, Annots is None!
        assert clean_pdf.pages[0].get("/Annots") is None


def test_decrypt_m04(fixtures_dir: Path):
    """FX_M04: Decrypt document with empty user password."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_M04"](fixtures_dir)

    with pikepdf.open(str(wm_path)) as pdf:
        assert pdf.is_encrypted is True

        dec_pdf = decrypt(pdf)
        assert dec_pdf.is_encrypted is False
        assert len(dec_pdf.pages) == len(pdf.pages)
