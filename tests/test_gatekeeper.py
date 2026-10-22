"""Tests for L0.5 Surgical Intelligence Layer (Gatekeeper)."""

from wmclean.intelligence.gatekeeper import Gatekeeper
from wmclean.schemas import (
    AnnotCandidate,
    PageGeometry,
    PageKind,
    ProbeRecord,
    RasterStats,
    RemoverMethod,
    TextCandidate,
)


def _make_dummy_record(
    p: int,
    kind: PageKind = PageKind.SCAN,
    w: float = 595.0,
    h: float = 842.0,
    sat: float = 0.0,
    tmpl_ncc: float = 0.0,
    annots: list[AnnotCandidate] | None = None,
    text_cands: list[TextCandidate] | None = None,
) -> ProbeRecord:
    """Helper to create dummy ProbeRecord for testing."""
    return ProbeRecord(
        p=p,
        kind=kind,
        geom=PageGeometry(mb=[0.0, 0.0, w, h], cb=[0.0, 0.0, w, h], rot=0),
        annots=annots or [],
        text_cand=text_cands or [],
        ras_stats=RasterStats(
            has_raster=(kind in {PageKind.SCAN, PageKind.SCAN_OCR, PageKind.HYBRID}),
            sat_p95=sat,
            tmpl_ncc=tmpl_ncc,
        ),
    )


def test_segment_subdocuments_change_points():
    """Gate 1: Detects sub-document boundaries when layout, kind, or chroma shifts."""
    gatekeeper = Gatekeeper()

    records = [
        # Subdoc 0: Pages 1-3 (Vector, A4)
        _make_dummy_record(1, kind=PageKind.VECTOR, w=595.0, h=842.0),
        _make_dummy_record(2, kind=PageKind.VECTOR, w=595.0, h=842.0),
        _make_dummy_record(3, kind=PageKind.VECTOR, w=595.0, h=842.0),
        # Subdoc 1: Pages 4-6 (Scan, US Letter, different aspect ratio)
        _make_dummy_record(4, kind=PageKind.SCAN, w=612.0, h=792.0),
        _make_dummy_record(5, kind=PageKind.SCAN, w=612.0, h=792.0),
        _make_dummy_record(6, kind=PageKind.SCAN, w=612.0, h=792.0),
        # Subdoc 2: Pages 7-9 (Scan, saturated red background)
        _make_dummy_record(7, kind=PageKind.SCAN, w=612.0, h=792.0, sat=45.0),
        _make_dummy_record(8, kind=PageKind.SCAN, w=612.0, h=792.0, sat=48.0),
        _make_dummy_record(9, kind=PageKind.SCAN, w=612.0, h=792.0, sat=46.0),
    ]

    subdoc_ids = gatekeeper.segment_subdocuments(records)
    assert len(subdoc_ids) == 9
    assert subdoc_ids[0] == subdoc_ids[1] == subdoc_ids[2] == 0
    assert subdoc_ids[3] == subdoc_ids[4] == subdoc_ids[5] == 1
    assert subdoc_ids[6] == subdoc_ids[7] == subdoc_ids[8] == 2


def test_per_page_presence_veto_clean_outlier():
    """Gate 2: Outlier clean page inside a watermarked cluster must be vetoed (marked UNTOUCHED)."""
    gatekeeper = Gatekeeper()

    # Suppose pages 14-23 have saturated red watermarks, but page 21 is clean (achromatic)
    records = []
    for p in range(14, 24):
        sat = 0.0 if p == 21 else 35.0
        records.append(_make_dummy_record(p, kind=PageKind.SCAN, sat=sat))

    plans = gatekeeper.plan_document(records)

    # Pages 14-20 and 22-23 should have hue unmix tasks, but page 21 should be UNTOUCHED
    for plan in plans:
        if plan.page == 21:
            assert plan.is_untouched, "Page 21 must be UNTOUCHED via negative presence veto"
            assert len(plan.tasks) == 0
        else:
            assert not plan.is_untouched
            methods = [t.method for t in plan.tasks]
            assert RemoverMethod.RASTER_HUE_SELECTIVE_UNMIX in methods


def test_per_page_template_correlation_veto():
    """Gate 2: Pages with tmpl_ncc < 0.35 must be vetoed from template whitening."""
    gatekeeper = Gatekeeper()

    records = []
    # Pages 1-10 have diagonal watermark (tmpl_ncc = 0.70)
    for p in range(1, 11):
        records.append(_make_dummy_record(p, kind=PageKind.SCAN, tmpl_ncc=0.70))
    # Pages 11-25 are clean (tmpl_ncc = 0.05 < 0.35 threshold)
    for p in range(11, 26):
        records.append(_make_dummy_record(p, kind=PageKind.SCAN, tmpl_ncc=0.05))

    plans = gatekeeper.plan_document(records)

    for plan in plans:
        if plan.page <= 10:
            assert not plan.is_untouched
            methods = [t.method for t in plan.tasks]
            assert RemoverMethod.RASTER_TEMPLATE_WHITEN in methods
        else:
            assert plan.is_untouched, f"Page {plan.page} must be UNTOUCHED"
            assert len(plan.tasks) == 0


def test_vector_triggers():
    """Gate 2/3: Vector triggers map deterministically to AST removers."""
    gatekeeper = Gatekeeper()

    # Page with annotation watermark
    annot = AnnotCandidate(xref=12, sub="/Watermark", bb=[100, 100, 200, 200])
    rec_annot = _make_dummy_record(1, kind=PageKind.VECTOR, annots=[annot])
    plan = gatekeeper.plan_page(rec_annot, subdoc_id=0, subdoc_records=[rec_annot])
    assert any(t.method == RemoverMethod.ANNOT_DEL for t in plan.tasks)

    # Page with rotated text candidate
    rot_text = TextCandidate(
        sig="t1",
        norm_str="CONFIDENTIAL",
        rot_deg=45.0,
        bb=[100, 100, 400, 400],
        font="/F1",
        size=36.0,
        color=[0.5],
    )
    rec_text = _make_dummy_record(2, kind=PageKind.VECTOR, text_cands=[rot_text])
    plan_t = gatekeeper.plan_page(rec_text, subdoc_id=0, subdoc_records=[rec_text])
    methods = [t.method for t in plan_t.tasks]
    assert RemoverMethod.AST_TJ_REASSEMBLE in methods
    assert RemoverMethod.TEXT_OP_STRIP in methods
