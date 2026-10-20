"""L1 Signature computation and page clustering.

Pages with identical structural/raster watermark characteristics share a
`sig_hash` so L2 routes the entire cluster at once.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from wmclean.schemas import (
    CarrierFlag,
    ClusterSignature,
    PageKind,
    ProbeRecord,
    RasterBucket,
)


def _canonicalize_features(record: ProbeRecord) -> dict:
    """Canonicalize a ProbeRecord into a hashable feature dictionary.

    Dynamic per-page values (page number, specific xrefs, exact bbox coordinates)
    are excluded or quantized so that pages with the same watermark type produce
    identical feature dicts.
    """
    features: dict[str, Any] = {
        "kind": record.kind.value,
        "geom_aspect": round(
            (record.geom.cb[2] - record.geom.cb[0])
            / max(record.geom.cb[3] - record.geom.cb[1], 1.0),
            2,
        ),
        "rot": record.geom.rot,
    }

    # Carrier flags
    carriers = []
    if record.annots:
        for a in record.annots:
            if a.sub == "/Watermark":
                carriers.append(CarrierFlag.ANNOT_WM.value)
            elif a.sub == "/Stamp":
                carriers.append(CarrierFlag.ANNOT_STAMP.value)
            elif a.sub == "/FreeText":
                carriers.append(CarrierFlag.ANNOT_FREETEXT.value)
    if record.ocgs:
        carriers.append(CarrierFlag.OCG.value)
    if record.artifacts:
        carriers.append(CarrierFlag.ARTIFACT_WM.value)
    if record.text_cand:
        carriers.append(CarrierFlag.TEXT_VECTOR.value)
    if record.xobj_cand:
        for x in record.xobj_cand:
            if x.kind == "IMAGE":
                carriers.append(CarrierFlag.XOBJ_IMAGE.value)
            elif x.kind == "FORM":
                carriers.append(CarrierFlag.XOBJ_FORM.value)
    if record.path_cand:
        carriers.append(CarrierFlag.VEC_PATH.value)
        if any(pc.is_clip for pc in record.path_cand):
            carriers.append(CarrierFlag.CLIP_PATH.value)
    if any(g.smask for g in record.gs_alpha):
        carriers.append(CarrierFlag.SMASK.value)
    if any(g.bm and g.bm not in ("/Normal", "") for g in record.gs_alpha):
        carriers.append(CarrierFlag.BLEND_MODE.value)
    if record.ras_stats.has_raster and (
        record.ras_stats.tmpl_cov > 0.01
        or record.ras_stats.sat_p95 > 30
        or record.ras_stats.est_alpha > 0
        or record.kind in (PageKind.SCAN, PageKind.SCAN_OCR)
    ):
        carriers.append(CarrierFlag.RASTER_BAKED.value)
    if record.ras_stats.is_bitonal:
        carriers.append(CarrierFlag.RASTER_BITONAL.value)
    if record.rev_clean is not None:
        carriers.append(CarrierFlag.REVISION_APPENDABLE.value)

    features["carriers"] = sorted(set(carriers))

    # Text pattern (normalized — same pattern across pages)
    if record.text_cand:
        patterns = sorted(set(tc.norm_str for tc in record.text_cand))
        features["text_patterns"] = patterns
        features["text_rot"] = int(record.text_cand[0].rot_deg // 15) * 15  # quantize to 15°

    # Path topology hashes
    if record.path_cand:
        features["path_hashes"] = sorted(set(pc.hash for pc in record.path_cand))

    # Raster bucket
    rs = record.ras_stats
    features["raster"] = {
        "bitonal": rs.is_bitonal,
        "periodic_fft": False,  # determined later by FFT analysis
        "chroma_sep": rs.sat_p95 > 30,  # high saturation suggests colour watermark
        "tmpl_stable": rs.pos_stab > 0.8,
    }

    # Color bucket
    if rs.is_bitonal:
        features["color_bucket"] = "BITONAL"
    elif rs.sat_p95 > 30:
        features["color_bucket"] = "COLOURED"
    elif rs.est_alpha > 0 and rs.est_alpha < 0.5:
        features["color_bucket"] = "LIGHT_GREY_ALPHA"
    else:
        features["color_bucket"] = "UNKNOWN"

    return features


def compute_signature(record: ProbeRecord) -> ClusterSignature:
    """Compute a ClusterSignature for a single page's ProbeRecord."""
    features = _canonicalize_features(record)

    # Deterministic JSON serialization for hashing
    canon_json = json.dumps(features, sort_keys=True, separators=(",", ":"))
    sig_hash = "sha256:" + hashlib.sha256(canon_json.encode()).hexdigest()

    carrier_flags = [CarrierFlag(c) for c in features.get("carriers", [])]

    return ClusterSignature(
        sig_hash=sig_hash,
        page_kind=record.kind,
        carrier_flags=carrier_flags,
        quant_geom_aspect=features["geom_aspect"],
        norm_text_pattern=(
            features.get("text_patterns", [None])[0]
            if features.get("text_patterns")
            else None
        ),
        rot_bucket=features.get("text_rot", record.geom.rot),
        color_bucket=features.get("color_bucket", "UNKNOWN"),
        raster_bucket=RasterBucket(**features.get("raster", {})),
    )


def compute_clusters(
    probe_records: list[ProbeRecord],
) -> dict[str, tuple[ClusterSignature, list[int]]]:
    """Group pages by identical signature hash.

    Returns a dict mapping sig_hash → (ClusterSignature, [page_indices]).
    Page indices are 0-based.
    """
    clusters: dict[str, tuple[ClusterSignature, list[int]]] = {}

    for i, record in enumerate(probe_records):
        sig = compute_signature(record)
        if sig.sig_hash in clusters:
            clusters[sig.sig_hash][1].append(i)
        else:
            clusters[sig.sig_hash] = (sig, [i])

    return clusters
