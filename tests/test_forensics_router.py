"""Unit tests for ForensicAnalyzer, VisionBridge, and IntelligentRouter."""


from wmclean.analysis.forensic_analyzer import ForensicAnalyzer
from wmclean.analysis.vision_bridge import VisionBridge
from wmclean.router.intelligent_router import IntelligentRouter
from wmclean.schemas import (
    DocumentForensicDossier,
    PageGeometry,
    PageKind,
    ProbeRecord,
    RasterStats,
    RemoverMethod,
    SubdocumentForensicProfile,
    WatermarkModality,
    XObjCandidate,
)


def _make_dummy_record(
    p: int,
    kind: PageKind = PageKind.VECTOR,
    xobjs: list[str] | None = None,
    sat: float = 0.0,
    tmpl_ncc: float = 0.0,
    tmpl_cov: float = 0.0,
) -> ProbeRecord:
    geom = PageGeometry(mb=[0, 0, 595, 842], cb=[0, 0, 595, 842], rot=0)
    x_cands = []
    if xobjs:
        for name in xobjs:
            x_cands.append(
                XObjCandidate(
                    name=name,
                    xref=10,
                    kind="IMAGE",
                    bb=[0.0, 0.0, 1122.0, 946.0],
                    doc_freq=0.5,
                )
            )
    is_scan = kind in (PageKind.SCAN, PageKind.SCAN_OCR)
    effective_cov = tmpl_cov if tmpl_cov > 0 else (0.05 if is_scan else 0.0)
    effective_ncc = tmpl_ncc if tmpl_ncc > 0 else (0.40 if is_scan else 0.0)
    rec = ProbeRecord(
        p=p,
        kind=kind,
        geom=geom,
        xobj_cand=x_cands,
        ras_stats=RasterStats(
            has_raster=is_scan,
            sat_p95=sat,
            tmpl_ncc=effective_ncc,
            tmpl_cov=effective_cov,
        ),
    )
    return rec


def test_forensic_analyzer_hybrid_nested_xobject():
    """Verify ForensicAnalyzer detects CONTAINER_FORM_XOBJECT_STAMP for hybrid pages with Form XObjects."""
    records = [
        _make_dummy_record(1, kind=PageKind.HYBRID, xobjs=["/Image2"]),
        _make_dummy_record(2, kind=PageKind.HYBRID, xobjs=["/Image2"]),
    ]
    analyzer = ForensicAnalyzer()
    dossier = analyzer.analyze_document(records)

    assert len(dossier.subdocs) == 1
    prof = dossier.subdocs[0]
    assert prof.detected_modality == WatermarkModality.CONTAINER_FORM_XOBJECT_STAMP
    assert prof.completeness_score >= 0.95
    assert not prof.is_ambiguous
    assert not prof.requires_vision_llm
    assert RemoverMethod.XOBJ_DO_STRIP in prof.recommended_chain
    assert "/Image2" in prof.target_xobj_names


def test_forensic_analyzer_scanned_subdocument():
    """Verify ForensicAnalyzer classifies scanned continuous-tone watermark and calibrates LUT."""
    records = [
        _make_dummy_record(1, kind=PageKind.SCAN),
        _make_dummy_record(2, kind=PageKind.SCAN),
        _make_dummy_record(3, kind=PageKind.SCAN),
    ]
    analyzer = ForensicAnalyzer()
    dossier = analyzer.analyze_document(records)

    assert len(dossier.subdocs) == 1
    prof = dossier.subdocs[0]
    assert prof.detected_modality == WatermarkModality.SCANNED_NEUTRAL_GRAY
    assert prof.completeness_score >= 0.90
    assert not prof.is_ambiguous
    assert prof.lut_params is not None
    assert prof.lut_params["black_point"] == 85
    assert RemoverMethod.POINTWISE_HUE_AND_LUT in prof.recommended_chain


def test_intelligent_router_multi_subdocument():
    """Verify IntelligentRouter seamlessly routes hybrid and scan pages to their specific techniques."""
    records = [
        # Subdoc 1: Hybrid vector pages with stamp
        _make_dummy_record(1, kind=PageKind.HYBRID, xobjs=["/Image2"]),
        _make_dummy_record(2, kind=PageKind.HYBRID, xobjs=["/Image2"]),
        # Subdoc 2: Scanned paper pages with faint gray watermark
        _make_dummy_record(3, kind=PageKind.SCAN),
        _make_dummy_record(4, kind=PageKind.SCAN),
    ]
    router = IntelligentRouter()
    decision, plans = router.route_document(records)

    assert len(plans) == 4
    # Pages 1 & 2 must route to xobj_do_strip
    p1_methods = [t.method for t in plans[0].tasks]
    p2_methods = [t.method for t in plans[1].tasks]
    assert RemoverMethod.XOBJ_DO_STRIP in p1_methods
    assert RemoverMethod.XOBJ_DO_STRIP in p2_methods

    # Pages 3 & 4 must route to pointwise_hue_and_lut
    p3_methods = [t.method for t in plans[2].tasks]
    p4_methods = [t.method for t in plans[3].tasks]
    assert RemoverMethod.POINTWISE_HUE_AND_LUT in p3_methods
    assert RemoverMethod.POINTWISE_HUE_AND_LUT in p4_methods


def test_vision_bridge_gated_execution():
    """Verify VisionBridge is NOT invoked when deterministic information is complete (0 LLM tokens)."""
    dossier = DocumentForensicDossier(
        total_pages=2,
        subdocs=[
            SubdocumentForensicProfile(
                subdoc_id=0,
                page_indices=[0, 1],
                page_kind=PageKind.HYBRID,
                detected_modality=WatermarkModality.CONTAINER_FORM_XOBJECT_STAMP,
                completeness_score=1.0,
                is_ambiguous=False,
                requires_vision_llm=False,
            )
        ],
        overall_completeness=1.0,
        ambiguous_subdoc_count=0,
    )

    call_count = 0

    def mock_llm_caller(prompt, image):
        nonlocal call_count
        call_count += 1
        return {"is_watermark": True, "modality": "XOBJECT"}

    bridge = VisionBridge(llm_caller=mock_llm_caller)
    resolved = bridge.resolve_ambiguities(dossier, pdf_path="dummy.pdf")

    assert call_count == 0, "VisionBridge must NEVER invoke LLM when information is complete"
    assert resolved.overall_completeness == 1.0
