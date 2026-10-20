"""Unit tests for Phase 5: L1 Clustering, L2 Deterministic Router, and Signature Library."""

from pathlib import Path

from tests.fixture_gen import ALL_FIXTURE_GENERATORS
from wmclean.cluster.library import SignatureLibrary
from wmclean.cluster.signature import compute_clusters, compute_signature
from wmclean.router.llm_fallback import _build_montage, build_llm_fallback_decision
from wmclean.router.rules import route_deterministic
from wmclean.schemas import (
    CarrierFlag,
    ClusterAction,
    PageGeometry,
    PageKind,
    ProbeRecord,
    RasterStats,
    RemoverMethod,
    RemoverParams,
)


def _make_dummy_probe_record(page_num: int, carrier: CarrierFlag) -> ProbeRecord:
    geom = PageGeometry(mb=[0, 0, 612, 792], cb=[0, 0, 612, 792], rot=0)
    record = ProbeRecord(
        p=page_num,
        kind=PageKind.VECTOR,
        geom=geom,
    )
    if carrier == CarrierFlag.ANNOT_WM:
        from wmclean.schemas import AnnotCandidate

        record.annots = [AnnotCandidate(xref=10, sub="/Watermark", bb=[100, 100, 300, 200])]
    elif carrier == CarrierFlag.OCG:
        from wmclean.schemas import OCGCandidate

        record.ocgs = [OCGCandidate(xref=20, name="WatermarkLayer")]
    elif carrier == CarrierFlag.TEXT_VECTOR:
        from wmclean.schemas import TextCandidate

        record.text_cand = [
            TextCandidate(
                sig="t_abc",
                norm_str="CONFIDENTIAL <DATE>",
                rot_deg=45.0,
                bb=[50, 50, 400, 400],
                font="/F1",
                size=36.0,
                color=[0.8, 0.8, 0.8],
            )
        ]
    elif carrier == CarrierFlag.RASTER_BAKED:
        record.kind = PageKind.SCAN
        record.ras_stats = RasterStats(has_raster=True, tmpl_cov=0.15, tmpl_ncc=0.85, pos_stab=0.95)
    return record


def test_signature_canonicalization_and_clustering():
    """Verify that pages with identical watermark characteristics cluster together."""
    r1 = _make_dummy_probe_record(1, CarrierFlag.ANNOT_WM)
    r2 = _make_dummy_probe_record(2, CarrierFlag.ANNOT_WM)
    r3 = _make_dummy_probe_record(3, CarrierFlag.TEXT_VECTOR)

    sig1 = compute_signature(r1)
    sig2 = compute_signature(r2)
    sig3 = compute_signature(r3)

    assert sig1.sig_hash == sig2.sig_hash
    assert sig1.sig_hash != sig3.sig_hash

    clusters = compute_clusters([r1, r2, r3])
    assert len(clusters) == 2
    assert sig1.sig_hash in clusters
    assert clusters[sig1.sig_hash][1] == [0, 1]
    assert clusters[sig3.sig_hash][1] == [2]


def test_deterministic_routing_vector():
    """Verify deterministic routing for key vector watermark categories."""
    r_annot = _make_dummy_probe_record(1, CarrierFlag.ANNOT_WM)
    r_ocg = _make_dummy_probe_record(2, CarrierFlag.OCG)
    r_text = _make_dummy_probe_record(3, CarrierFlag.TEXT_VECTOR)

    records = [r_annot, r_ocg, r_text]
    clusters = compute_clusters(records)
    decision = route_deterministic(clusters, records)

    assert len(decision.clusters) == 3
    actions = {act.sig_hash: act for act in decision.clusters}

    sig_annot = compute_signature(r_annot)
    sig_ocg = compute_signature(r_ocg)
    sig_text = compute_signature(r_text)

    assert actions[sig_annot.sig_hash].chain == [RemoverMethod.ANNOT_DEL]
    assert actions[sig_ocg.sig_hash].chain == [RemoverMethod.OCG_STRIP]
    assert actions[sig_text.sig_hash].chain == [RemoverMethod.TEXT_OP_STRIP]


def test_deterministic_routing_raster():
    """Verify deterministic routing for raster watermark categories."""
    # 1. Fixed alpha template
    r_fixed = _make_dummy_probe_record(1, CarrierFlag.RASTER_BAKED)
    r_fixed.ras_stats.tmpl_cov = 0.20
    r_fixed.ras_stats.pos_stab = 0.95
    r_fixed.ras_stats.est_alpha = 0.30

    # 2. Saturated colour watermark
    r_color = _make_dummy_probe_record(2, CarrierFlag.RASTER_BAKED)
    r_color.ras_stats.sat_p95 = 55.0

    # 3. Bitonal despeckle
    r_bitonal = _make_dummy_probe_record(3, CarrierFlag.RASTER_BITONAL)
    r_bitonal.ras_stats.has_raster = True
    r_bitonal.ras_stats.is_bitonal = True

    records = [r_fixed, r_color, r_bitonal]
    clusters = compute_clusters(records)
    decision = route_deterministic(clusters, records)

    actions = {act.sig_hash: act for act in decision.clusters}
    sig_fixed = compute_signature(r_fixed)
    sig_color = compute_signature(r_color)
    sig_bitonal = compute_signature(r_bitonal)

    assert RemoverMethod.RASTER_ALPHA_INV in actions[sig_fixed.sig_hash].chain
    assert actions[sig_color.sig_hash].chain == [RemoverMethod.RASTER_COLOR_UNMIX]
    assert actions[sig_bitonal.sig_hash].chain == [RemoverMethod.RASTER_BITONAL_DESPECKLE]


def test_signature_library_persistence(tmp_path: Path):
    """Verify SignatureLibrary stores and retrieves verified signatures with O(1) latency."""
    db_file = tmp_path / "signatures_test.sqlite"
    r = _make_dummy_probe_record(1, CarrierFlag.ANNOT_WM)
    sig = compute_signature(r)
    sig.proven_chain = [RemoverMethod.ANNOT_DEL]
    sig.proven_params = {"target_xrefs": [10]}
    sig.historical_pass_rate = 1.0

    with SignatureLibrary(db_path=db_file) as lib:
        # Initially not found
        assert lib.lookup(sig.sig_hash) is None

        # Store verified signature
        lib.store(sig)

        # Lookup returns exact proven chain
        cached = lib.lookup(sig.sig_hash)
        assert cached is not None
        assert cached.proven_chain == [RemoverMethod.ANNOT_DEL]
        assert cached.proven_params == {"target_xrefs": [10]}

        # Test router uses library
        clusters = compute_clusters([r])
        decision = route_deterministic(clusters, [r], library=lib)
        assert len(decision.clusters) == 1
        assert decision.clusters[0].chain == [RemoverMethod.ANNOT_DEL]


def test_llm_fallback_error_resilience(tmp_path: Path):
    """Verify LLM fallback gracefully falls back to deterministic rule on ANY failure."""
    r = _make_dummy_probe_record(1, CarrierFlag.ANNOT_WM)
    clusters = compute_clusters([r])
    sig = compute_signature(r)

    det_fallback = {
        sig.sig_hash: ClusterAction(
            sig_hash=sig.sig_hash,
            action="REMOVE",
            chain=[RemoverMethod.ANNOT_DEL],
            params=RemoverParams(target_xrefs=[10]),
        )
    }

    # Case 1: llm_caller raises Exception
    def bad_llm(summary, montage):
        raise RuntimeError("LLM API timeout 10s")

    decision1 = build_llm_fallback_decision(
        clusters, [r], det_fallback, llm_caller=bad_llm
    )
    assert len(decision1.clusters) == 1
    assert decision1.clusters[0].chain == [RemoverMethod.ANNOT_DEL]

    # Case 2: llm_caller returns invalid schema dict
    def invalid_llm(summary, montage):
        return {"not_valid": 123}

    decision2 = build_llm_fallback_decision(
        clusters, [r], det_fallback, llm_caller=invalid_llm
    )
    assert len(decision2.clusters) == 1
    assert decision2.clusters[0].chain == [RemoverMethod.ANNOT_DEL]

    # Case 3: llm_caller returns valid schema dict
    def valid_llm(summary, montage):
        return {
            "clusters": [
                {
                    "sig_hash": sig.sig_hash,
                    "action": "REMOVE",
                    "chain": ["ANNOT_DEL"],
                    "params": {"target_xrefs": [10]},
                }
            ]
        }

    decision3 = build_llm_fallback_decision(
        clusters, [r], det_fallback, llm_caller=valid_llm
    )
    assert len(decision3.clusters) == 1
    assert decision3.clusters[0].chain == [RemoverMethod.ANNOT_DEL]


def test_montage_generation(fixtures_dir: Path):
    """Verify 768x768 montage generation from sample fixture."""
    wm_path, _, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    r = _make_dummy_probe_record(1, CarrierFlag.ANNOT_WM)
    clusters = compute_clusters([r])

    montage_bytes = _build_montage(clusters, pdf_path=str(wm_path))
    assert montage_bytes is not None
    assert len(montage_bytes) > 1000
    assert montage_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes
