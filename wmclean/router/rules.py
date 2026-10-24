"""L2 Deterministic Rule Router — scores candidate method chains and routes
clusters with confidence ≥ 0.85 directly, without any LLM call.

The rule matrix maps carrier flags and raster characteristics to ordered
method chains from the closed RemoverMethod enum.
"""

from __future__ import annotations

from typing import Any, Optional

from wmclean.schemas import (
    CarrierFlag,
    ClusterAction,
    ClusterSignature,
    ProbeRecord,
    RemoverMethod,
    RemoverParams,
    RouterDecision,
)

# ---------------------------------------------------------------------------
# Rule Matrix: carrier_flags → (chain, base_confidence)
# ---------------------------------------------------------------------------

# Each rule is a tuple of (required_flags, chain, base_confidence).
# Rules are evaluated in priority order; the first match ≥ 0.85 wins.
_RULES: list[tuple[set[str], list[RemoverMethod], float]] = [
    # G01: Annotation watermarks — highest confidence, simplest removal
    (
        {CarrierFlag.ANNOT_WM.value},
        [RemoverMethod.ANNOT_DEL],
        0.99,
    ),
    (
        {CarrierFlag.ANNOT_STAMP.value},
        [RemoverMethod.ANNOT_DEL],
        0.99,
    ),
    (
        {CarrierFlag.ANNOT_FREETEXT.value},
        [RemoverMethod.ANNOT_DEL],
        0.95,
    ),
    # G02: OCG layers
    (
        {CarrierFlag.OCG.value},
        [RemoverMethod.OCG_STRIP],
        0.98,
    ),
    # G03: Marked-content artifacts
    (
        {CarrierFlag.ARTIFACT_WM.value},
        [RemoverMethod.ARTIFACT_STRIP],
        0.98,
    ),
    # G04: Vector text watermarks
    (
        {CarrierFlag.TEXT_VECTOR.value},
        [RemoverMethod.TEXT_OP_STRIP],
        0.95,
    ),
    # G05: Clipping path watermarks
    (
        {CarrierFlag.CLIP_PATH.value},
        [RemoverMethod.CLIP_SCOPE_STRIP],
        0.93,
    ),
    # G06: XObject image/form watermarks
    (
        {CarrierFlag.XOBJ_IMAGE.value},
        [RemoverMethod.XOBJ_DO_STRIP],
        0.92,
    ),
    (
        {CarrierFlag.XOBJ_FORM.value},
        [RemoverMethod.XOBJ_DO_STRIP],
        0.92,
    ),
    # G07: Vector path watermarks
    (
        {CarrierFlag.VEC_PATH.value},
        [RemoverMethod.VEC_PATH_STRIP],
        0.90,
    ),
    # G08: Pattern/shading watermarks
    (
        {CarrierFlag.PATTERN.value},
        [RemoverMethod.PATTERN_SH_STRIP],
        0.90,
    ),
    (
        {CarrierFlag.SHADING.value},
        [RemoverMethod.PATTERN_SH_STRIP],
        0.90,
    ),
    # G09: Soft-mask / blend-mode overlays
    (
        {CarrierFlag.SMASK.value},
        [RemoverMethod.SMASK_GS_STRIP],
        0.92,
    ),
    (
        {CarrierFlag.BLEND_MODE.value},
        [RemoverMethod.SMASK_GS_STRIP],
        0.90,
    ),
    # M01: Incremental revision rollback
    (
        {CarrierFlag.REVISION_APPENDABLE.value},
        [RemoverMethod.REV_ROLLBACK],
        0.99,
    ),
    # Raster baked watermarks — confidence depends on raster characteristics
    # G10: Fixed-position template with stable alpha
    (
        {CarrierFlag.RASTER_BAKED.value},
        [RemoverMethod.RASTER_TEMPLATE_WHITEN, RemoverMethod.RASTER_ALPHA_INV],
        0.88,
    ),
    # G15: Bitonal dithered
    (
        {CarrierFlag.RASTER_BITONAL.value},
        [RemoverMethod.RASTER_BITONAL_DESPECKLE],
        0.90,
    ),
]


def _score_cluster(
    sig: ClusterSignature,
) -> tuple[list[RemoverMethod], float]:
    """Score a cluster against the rule matrix.

    Returns the best-matching (chain, confidence) pair.
    """
    carrier_set = {c.value for c in sig.carrier_flags}

    best_chain: list[RemoverMethod] = []
    best_confidence = 0.0

    for required_flags, chain, base_confidence in _RULES:
        if required_flags.issubset(carrier_set):
            confidence = base_confidence
            if sig.raster_bucket.tmpl_stable and RemoverMethod.RASTER_ALPHA_INV in chain:
                confidence = min(confidence + 0.05, 0.99)

            if confidence > best_confidence:
                best_confidence = confidence
                best_chain = list(chain)

    # Refine raster baked watermarks based on specific raster stats
    if CarrierFlag.RASTER_BAKED.value in carrier_set:
        if sig.color_bucket == "COLOURED" or sig.raster_bucket.chroma_sep:
            best_chain = [RemoverMethod.RASTER_COLOR_UNMIX]
            best_confidence = 0.92
        elif sig.raster_bucket.periodic_fft:
            best_chain = [RemoverMethod.RASTER_FFT_NOTCH]
            best_confidence = 0.90
        elif sig.raster_bucket.tmpl_stable or sig.color_bucket == "LIGHT_GREY_ALPHA":
            best_chain = [RemoverMethod.RASTER_TEMPLATE_WHITEN, RemoverMethod.RASTER_ALPHA_INV]
            best_confidence = 0.90
        elif not best_chain:
            best_chain = [RemoverMethod.RASTER_LUMA_SEP]
            best_confidence = 0.88

    # Compound carriers: vector watermark + raster watermark
    vec_carriers = {
        CarrierFlag.TEXT_VECTOR.value,
        CarrierFlag.ANNOT_WM.value,
        CarrierFlag.ANNOT_STAMP.value,
        CarrierFlag.OCG.value,
        CarrierFlag.ARTIFACT_WM.value,
        CarrierFlag.XOBJ_IMAGE.value,
        CarrierFlag.XOBJ_FORM.value,
        CarrierFlag.VEC_PATH.value,
        CarrierFlag.CLIP_PATH.value,
        CarrierFlag.SMASK.value,
    }
    ras_carriers = {
        CarrierFlag.RASTER_BAKED.value,
        CarrierFlag.RASTER_BITONAL.value,
    }

    if (carrier_set & vec_carriers) and (carrier_set & ras_carriers):
        compound_chain = []
        for req_flags, chain, conf in _RULES:
            if req_flags.issubset(carrier_set & vec_carriers):
                for m in chain:
                    if m not in compound_chain:
                        compound_chain.append(m)
        if CarrierFlag.RASTER_BITONAL.value in carrier_set:
            if RemoverMethod.RASTER_BITONAL_DESPECKLE not in compound_chain:
                compound_chain.append(RemoverMethod.RASTER_BITONAL_DESPECKLE)
        elif CarrierFlag.RASTER_BAKED.value in carrier_set:
            if sig.color_bucket == "COLOURED" or sig.raster_bucket.chroma_sep:
                if RemoverMethod.RASTER_COLOR_UNMIX not in compound_chain:
                    compound_chain.append(RemoverMethod.RASTER_COLOR_UNMIX)
            elif sig.raster_bucket.tmpl_stable:
                if RemoverMethod.RASTER_ALPHA_INV not in compound_chain:
                    compound_chain.append(RemoverMethod.RASTER_ALPHA_INV)
            else:
                if RemoverMethod.RASTER_LUMA_SEP not in compound_chain:
                    compound_chain.append(RemoverMethod.RASTER_LUMA_SEP)

        if compound_chain:
            best_chain = compound_chain
            best_confidence = 0.89

    return best_chain, best_confidence


def _build_params(sig: ClusterSignature, records: list[ProbeRecord]) -> RemoverParams:
    """Build RemoverParams from cluster signature and probe records."""
    params = RemoverParams()

    for record in records:
        for a in record.annots:
            if a.xref not in params.target_xrefs:
                params.target_xrefs.append(a.xref)
        for tc in record.text_cand:
            if tc.sig not in params.target_text_sigs:
                params.target_text_sigs.append(tc.sig)
            if tc.norm_str not in params.target_texts:
                params.target_texts.append(tc.norm_str)
        for pc in record.path_cand:
            if (
                getattr(pc, "is_light", False)
                or (pc.ca is not None and pc.ca < 0.85)
                or pc.is_clip
                or (pc.doc_freq >= 0.70)
            ):
                if pc.hash not in params.target_path_hashes:
                    params.target_path_hashes.append(pc.hash)
        for xc in record.xobj_cand:
            is_wm_xobj = (
                (xc.doc_freq >= 0.20)
                or (len(records) <= 2)
                or any(kw in xc.name.lower() for kw in ("watermark", "stamp", "draft", "wm"))
            )
            if is_wm_xobj:
                if xc.name not in params.target_xobj_names:
                    params.target_xobj_names.append(xc.name)
        for ocg in record.ocgs:
            if ocg.xref not in params.target_ocg_xrefs:
                params.target_ocg_xrefs.append(ocg.xref)
        for gs in record.gs_alpha:
            if gs.name not in params.target_gs_names:
                params.target_gs_names.append(gs.name)

    # Set raster parameters from first record's stats
    if records:
        rs = records[0].ras_stats
        if rs.est_alpha > 0:
            params.alpha_clamp = min(max(rs.est_alpha + 0.1, 0.1), 0.92)

    return params


def route_deterministic(
    clusters: dict[str, tuple[ClusterSignature, list[int]]],
    probe_records: list[ProbeRecord],
    library: Optional[Any] = None,
) -> RouterDecision:
    """Route all clusters deterministically using the rule matrix or library.

    Returns a RouterDecision with one ClusterAction per cluster.
    """
    actions = []

    for sig_hash, (sig, page_indices) in clusters.items():
        # Check library cache first
        if library is not None:
            cached = library.lookup(sig_hash)
            if cached is not None and cached.proven_chain:
                params = _build_params(sig, [probe_records[i] for i in page_indices])
                if cached.proven_params:
                    for k, v in cached.proven_params.items():
                        if hasattr(params, k):
                            setattr(params, k, v)
                actions.append(
                    ClusterAction(
                        sig_hash=sig_hash,
                        action="REMOVE",
                        chain=cached.proven_chain,
                        params=params,
                    )
                )
                continue

        chain, confidence = _score_cluster(sig)

        if not chain:
            actions.append(
                ClusterAction(
                    sig_hash=sig_hash,
                    action="SKIP",
                    chain=[],
                    params=RemoverParams(),
                )
            )
            continue

        cluster_records = [probe_records[i] for i in page_indices]
        params = _build_params(sig, cluster_records)

        actions.append(
            ClusterAction(
                sig_hash=sig_hash,
                action="REMOVE",
                chain=chain,
                params=params,
            )
        )

    return RouterDecision(clusters=actions)
