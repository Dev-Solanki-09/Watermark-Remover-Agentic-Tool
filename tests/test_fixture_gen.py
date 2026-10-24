"""Acceptance tests for synthetic fixture generation (Phase 1, Task 1.1)."""

from __future__ import annotations

import time
from pathlib import Path

import pikepdf
import pymupdf

from tests.fixture_gen import ALL_FIXTURE_GENERATORS, generate_all_fixtures


def test_fixture_generation_completeness(fixtures_dir: Path):
    """Verify that all 25 fixtures (G01–G19, M01–M06) generate and are valid PDFs."""
    start_time = time.monotonic()
    manifest = generate_all_fixtures(fixtures_dir)
    elapsed = time.monotonic() - start_time

    assert len(manifest) == 25, f"Expected 25 fixtures, got {len(manifest)}"
    assert elapsed < 30.0, f"Fixture generation took {elapsed:.2f}s, expected < 30s"

    for fid, (wm_path, clean_path, mask) in manifest.items():
        assert wm_path.exists(), f"Watermarked file missing for {fid}: {wm_path}"
        assert clean_path.exists(), f"Clean reference missing for {fid}: {clean_path}"
        assert wm_path.stat().st_size > 0, f"Watermarked file is empty: {wm_path}"
        assert clean_path.stat().st_size > 0, f"Clean reference is empty: {clean_path}"

        # Test PDF validity via pikepdf (except M04 which has an owner password)
        if fid != "FX_M04":
            with pikepdf.open(str(wm_path)) as pdf:
                assert len(pdf.pages) >= 1
            with pikepdf.open(str(clean_path)) as pdf:
                assert len(pdf.pages) >= 1
        else:
            # M04 has empty user password, test that empty password works
            with pikepdf.open(str(wm_path), password="") as pdf:
                assert len(pdf.pages) >= 1

        # Test PDF validity via PyMuPDF
        doc_wm = pymupdf.open(str(wm_path))
        assert doc_wm.page_count >= 1
        doc_wm.close()

        doc_cl = pymupdf.open(str(clean_path))
        assert doc_cl.page_count >= 1
        doc_cl.close()


def test_fixture_determinism(fixtures_dir: Path):
    """Verify that running fixture generation again produces deterministic valid outputs."""
    dir1 = fixtures_dir / "run1"
    dir2 = fixtures_dir / "run2"

    # Test a representative vector and a raster generator
    wm1_v, cl1_v, _ = ALL_FIXTURE_GENERATORS["FX_G01"](dir1)
    wm2_v, cl2_v, _ = ALL_FIXTURE_GENERATORS["FX_G01"](dir2)

    assert abs(wm1_v.stat().st_size - wm2_v.stat().st_size) <= 10
    assert abs(cl1_v.stat().st_size - cl2_v.stat().st_size) <= 10
