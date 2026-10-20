"""Acceptance tests for L0 Structural and Raster Probes (Phase 1, Tasks 1.2 & 1.3)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pikepdf
import pymupdf

from tests.fixture_gen import ALL_FIXTURE_GENERATORS
from wmclean.probe.raster_stats import RasterReservoir, probe_raster
from wmclean.probe.structural import ProbeSpillDB, probe_structural


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_probe_structural_vector_carriers(fixtures_dir: Path):
    """Test that L0 structural probe detects 100% of injected vector carriers G01–G09."""
    # FX_G01: Annotations
    wm_g01, _, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    with pikepdf.open(str(wm_g01)) as pdf:
        rec = probe_structural(pdf, 0)
        assert len(rec.annots) >= 1
        assert any(a.sub in ("/Watermark", "/Stamp", "/FreeText") for a in rec.annots)

    # FX_G02: OCGs
    wm_g02, _, _ = ALL_FIXTURE_GENERATORS["FX_G02"](fixtures_dir)
    with pikepdf.open(str(wm_g02)) as pdf:
        rec = probe_structural(pdf, 0)
        assert len(rec.ocgs) >= 1
        assert any(o.name == "WatermarkLayer" for o in rec.ocgs)

    # FX_G03: Marked-content artifacts
    wm_g03, _, _ = ALL_FIXTURE_GENERATORS["FX_G03"](fixtures_dir)
    with pikepdf.open(str(wm_g03)) as pdf:
        rec = probe_structural(pdf, 0)
        assert len(rec.artifacts) >= 1
        assert rec.artifacts[0].sub == "/Watermark"

    # FX_G04: Text candidates
    wm_g04, _, _ = ALL_FIXTURE_GENERATORS["FX_G04"](fixtures_dir)
    doc_fitz = pymupdf.open(str(wm_g04))
    with pikepdf.open(str(wm_g04)) as pdf:
        rec = probe_structural(pdf, 0, fitz_page=doc_fitz[0])
        assert len(rec.text_cand) >= 1
        assert any("CONFIDENTIAL" in tc.norm_str for tc in rec.text_cand)
        tc = [tc for tc in rec.text_cand if "CONFIDENTIAL" in tc.norm_str][0]
        assert abs(tc.rot_deg) > 10.0
    doc_fitz.close()

    # FX_G06: XObjects
    wm_g06, _, _ = ALL_FIXTURE_GENERATORS["FX_G06"](fixtures_dir)
    with pikepdf.open(str(wm_g06)) as pdf:
        rec = probe_structural(pdf, 0)
        assert len(rec.xobj_cand) >= 1
        assert any(x.kind == "FORM" for x in rec.xobj_cand)

    # FX_G07: Untagged vector paths
    wm_g07, _, _ = ALL_FIXTURE_GENERATORS["FX_G07"](fixtures_dir)
    with pikepdf.open(str(wm_g07)) as pdf:
        rec = probe_structural(pdf, 0)
        assert len(rec.path_cand) >= 1

    # FX_G09: Soft-mask / Multiply blend mode ExtGState
    wm_g09, _, _ = ALL_FIXTURE_GENERATORS["FX_G09"](fixtures_dir)
    with pikepdf.open(str(wm_g09)) as pdf:
        rec = probe_structural(pdf, 0)
        assert len(rec.gs_alpha) >= 1
        assert any(g.bm == "/Multiply" or g.ca is not None for g in rec.gs_alpha)


def test_probe_structural_modifiers(fixtures_dir: Path):
    """Test that structural probe accurately detects M01–M03 modifiers."""
    # FX_M01: Incremental revision
    wm_m01, _, _ = ALL_FIXTURE_GENERATORS["FX_M01"](fixtures_dir)
    with pikepdf.open(str(wm_m01)) as pdf:
        rec = probe_structural(pdf, 0)
        assert rec.rev_clean is not None
        assert rec.rev_clean > 0

    # FX_M02: Rotated pages
    wm_m02, _, _ = ALL_FIXTURE_GENERATORS["FX_M02"](fixtures_dir)
    with pikepdf.open(str(wm_m02)) as pdf:
        rec0 = probe_structural(pdf, 0)
        rec1 = probe_structural(pdf, 1)
        assert rec0.geom.rot == 90
        assert rec1.geom.rot == 270

    # FX_M03: CropBox bleed
    wm_m03, _, _ = ALL_FIXTURE_GENERATORS["FX_M03"](fixtures_dir)
    with pikepdf.open(str(wm_m03)) as pdf:
        rec = probe_structural(pdf, 0)
        assert rec.geom.cb != rec.geom.mb


def test_probe_raster_statistics(fixtures_dir: Path):
    """Test that L0 raster probe computes saturation, bitonal, and reservoir template stats."""
    reservoir = RasterReservoir()

    # FX_G10: Light grey alpha-blended fixed raster watermark
    wm_g10, _, _ = ALL_FIXTURE_GENERATORS["FX_G10"](fixtures_dir)
    fitz_doc = pymupdf.open(str(wm_g10))
    with pikepdf.open(str(wm_g10)) as pdf:
        for p in range(len(pdf.pages)):
            rec = probe_structural(pdf, p)
            rec = probe_raster(pdf, p, rec, reservoir=reservoir, fitz_doc=fitz_doc)
            assert rec.ras_stats.has_raster is True
        # After 3 pages, reservoir has median template
        rec_last = rec
        assert rec_last.ras_stats.tmpl_cov > 0.0
    fitz_doc.close()

    # FX_G11: Coloured watermark (high saturation)
    wm_g11, _, _ = ALL_FIXTURE_GENERATORS["FX_G11"](fixtures_dir)
    with pikepdf.open(str(wm_g11)) as pdf:
        rec = probe_structural(pdf, 0)
        rec = probe_raster(pdf, 0, rec)
        assert rec.ras_stats.sat_p95 > 25.0, f"Expected sat_p95 > 25.0, got {rec.ras_stats.sat_p95}"

    # FX_G15: Bitonal 1-bit scan
    wm_g15, _, _ = ALL_FIXTURE_GENERATORS["FX_G15"](fixtures_dir)
    with pikepdf.open(str(wm_g15)) as pdf:
        rec = probe_structural(pdf, 0)
        rec = probe_raster(pdf, 0, rec)
        assert rec.ras_stats.is_bitonal is True

    # FX_G16: Tiled strips scan
    wm_g16, _, _ = ALL_FIXTURE_GENERATORS["FX_G16"](fixtures_dir)
    with pikepdf.open(str(wm_g16)) as pdf:
        rec = probe_structural(pdf, 0)
        rec = probe_raster(pdf, 0, rec)
        assert rec.ras_stats.is_tiled_strips is True
        assert rec.ras_stats.strip_count >= 2


def test_probe_non_mutation_invariant(fixtures_dir: Path):
    """Invariant: Probing a PDF must NEVER modify its bytes or file content."""
    wm_g01, _, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    h_before = _hash_file(wm_g01)

    with pikepdf.open(str(wm_g01)) as pdf:
        for p in range(len(pdf.pages)):
            rec = probe_structural(pdf, p)
            rec = probe_raster(pdf, p, rec)

    h_after = _hash_file(wm_g01)
    assert h_before == h_after, "Probing mutated the input file!"


def test_probe_spill_db():
    """Test SQLite spill DB for large document memory bounding."""
    from wmclean.schemas import PageGeometry, PageKind, ProbeRecord, RasterStats

    spill = ProbeSpillDB()
    try:
        # Create and save 10 records
        for i in range(1, 11):
            rec = ProbeRecord(
                p=i,
                kind=PageKind.VECTOR,
                geom=PageGeometry(mb=[0, 0, 612, 792], cb=[0, 0, 612, 792], rot=0),
                ras_stats=RasterStats(),
            )
            spill.save(rec)

        # Retrieve single record
        rec_5 = spill.get(5)
        assert rec_5 is not None
        assert rec_5.p == 5
        assert rec_5.kind == PageKind.VECTOR

        # Retrieve all records
        all_recs = spill.get_all()
        assert len(all_recs) == 10
        assert [r.p for r in all_recs] == list(range(1, 11))
    finally:
        spill.close()
