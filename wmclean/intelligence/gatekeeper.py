"""L0.5 Surgical Intelligence Layer — Gatekeeper.

Deterministic 4-Gate dispatcher that sits between L0 Probe and L3 Removers:
1. Gate 1: Sub-Document Change-Point Segmentation
2. Gate 2: Per-Page Positive Trigger & Negative Veto Matrix
3. Gate 3: Method-Specific Pre-Condition & Dispatch Planning
4. Gate 4: Sub-Page ROI Bounding-Box Lock (ROIGate)

Ensures zero LLM tokens and 0 ms spent on clean pages/regions.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

from wmclean.schemas import CarrierFlag, PageKind, ProbeRecord, RemoverMethod, RemoverParams

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MethodTask:
    """A single remover action bound to a specific sub-page ROI."""

    method: RemoverMethod
    roi_bbox: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) in pixel space
    mask_ref: Optional[str] = None
    confidence: float = 0.90
    params: Optional[RemoverParams] = None


@dataclasses.dataclass
class PageDispatchPlan:
    """Per-page surgical execution plan."""

    page: int  # 1-indexed page number
    subdoc_id: int
    tasks: list[MethodTask] = dataclasses.field(default_factory=list)
    reason: str = ""

    @property
    def is_untouched(self) -> bool:
        """True if no removers are scheduled for this page."""
        return len(self.tasks) == 0


class Gatekeeper:
    """L0.5 Surgical Gatekeeper."""

    def __init__(self, target_dpi: int = 150):
        self.target_dpi = target_dpi

    def segment_subdocuments(self, probe_records: list[ProbeRecord]) -> list[int]:
        """Gate 1: Detect sub-document change-points across pages 1..N.

        Stitched 'Franken-PDFs' combine distinct documents. A boundary is triggered when:
        a) CropBox/MediaBox aspect ratio shifts > 3%
        b) Page base kind shifts (VECTOR vs SCAN vs HYBRID)
        c) Dominant background chroma (sat_p95) jumps > 15
        d) MediaBox dimensions jump > 25 pt
        """
        if not probe_records:
            return []

        subdoc_ids: list[int] = [0]
        current_subdoc = 0

        for i in range(1, len(probe_records)):
            prev = probe_records[i - 1]
            curr = probe_records[i]

            prev_mb = prev.geom.mb if prev.geom else [0, 0, 595, 842]
            curr_mb = curr.geom.mb if curr.geom else [0, 0, 595, 842]

            prev_w = abs(prev_mb[2] - prev_mb[0])
            prev_h = abs(prev_mb[3] - prev_mb[1])
            curr_w = abs(curr_mb[2] - curr_mb[0])
            curr_h = abs(curr_mb[3] - curr_mb[1])

            prev_ar = prev_w / max(prev_h, 1.0)
            curr_ar = curr_w / max(curr_h, 1.0)

            # Check aspect ratio shift > 3%
            ar_shift = abs(curr_ar - prev_ar) / max(prev_ar, 0.01) > 0.03

            # Check page kind shift
            kind_shift = (prev.kind != curr.kind) and not (
                {prev.kind, curr.kind} <= {PageKind.SCAN, PageKind.SCAN_OCR}
            )

            # Check dimensions shift > 25pt
            dim_shift = (abs(curr_w - prev_w) > 25.0) or (abs(curr_h - prev_h) > 25.0)

            # Check saturation jump > 15 (only if sustained for >= 2 pages, not an isolated 1-page pulse)
            prev_sat = prev.ras_stats.sat_p95 if prev.ras_stats else 0.0
            curr_sat = curr.ras_stats.sat_p95 if curr.ras_stats else 0.0
            sat_shift = False
            if abs(curr_sat - prev_sat) > 15.0:
                if i + 1 < len(probe_records):
                    next_rec = probe_records[i + 1]
                    next_sat = next_rec.ras_stats.sat_p95 if next_rec.ras_stats else 0.0
                    # If next page does not revert back to prev_sat, it's a true sustained regime shift
                    if abs(next_sat - prev_sat) > 15.0:
                        sat_shift = True
                else:
                    sat_shift = True

            if ar_shift or kind_shift or dim_shift or sat_shift:
                current_subdoc += 1

            subdoc_ids.append(current_subdoc)

        return subdoc_ids

    def plan_page(
        self,
        record: ProbeRecord,
        subdoc_id: int,
        subdoc_records: list[ProbeRecord],
        reservoir: Optional[object] = None,
    ) -> PageDispatchPlan:
        """Gates 2, 3, 4: Evaluate positive triggers, per-page negative vetoes, and ROI locks."""
        from wmclean.cluster.signature import compute_signature

        tasks: list[MethodTask] = []
        p_num = record.p
        mb = record.geom.mb if record.geom else [0, 0, 595, 842]
        w_pt = abs(mb[2] - mb[0])
        h_pt = abs(mb[3] - mb[1])
        w_px = int(w_pt * self.target_dpi / 72.0)
        h_px = int(h_pt * self.target_dpi / 72.0)

        sig = compute_signature(record)
        carrier_set = {c.value if hasattr(c, "value") else c for c in sig.carrier_flags}

        # -------------------------------------------------------------------
        # 1. Vector AST Triggers (Gate 2/3)
        # -------------------------------------------------------------------
        if CarrierFlag.ANNOT_WM in carrier_set or CarrierFlag.ANNOT_STAMP in carrier_set or record.annots:
            tasks.append(MethodTask(method=RemoverMethod.ANNOT_DEL, confidence=0.99))

        if CarrierFlag.OCG in carrier_set or record.ocgs:
            tasks.append(MethodTask(method=RemoverMethod.OCG_STRIP, confidence=0.98))

        if CarrierFlag.ARTIFACT_WM in carrier_set or record.artifacts:
            tasks.append(MethodTask(method=RemoverMethod.ARTIFACT_STRIP, confidence=0.95))

        if CarrierFlag.REVISION_APPENDABLE in carrier_set and record.rev_clean is not None:
            tasks.append(MethodTask(method=RemoverMethod.REV_ROLLBACK, confidence=0.99))

        # Vector text watermarks (e.g. rotated text, low alpha, split-TJ arrays)
        if CarrierFlag.TEXT_VECTOR in carrier_set or record.text_cand:
            tasks.append(MethodTask(method=RemoverMethod.AST_TJ_REASSEMBLE, confidence=0.92))
            tasks.append(MethodTask(method=RemoverMethod.TEXT_OP_STRIP, confidence=0.90))

        # Vector image/form XObjects
        if CarrierFlag.XOBJ_IMAGE in carrier_set or CarrierFlag.XOBJ_FORM in carrier_set:
            tasks.append(MethodTask(method=RemoverMethod.XOBJ_DO_STRIP, confidence=0.92))

        # Vector path / clip scope / pattern / smask
        if CarrierFlag.SMASK in carrier_set or CarrierFlag.BLEND_MODE in carrier_set:
            tasks.append(MethodTask(method=RemoverMethod.SMASK_GS_STRIP, confidence=0.92))

        if CarrierFlag.CLIP_PATH in carrier_set:
            tasks.append(MethodTask(method=RemoverMethod.CLIP_SCOPE_STRIP, confidence=0.90))

        if CarrierFlag.PATTERN in carrier_set or CarrierFlag.SHADING in carrier_set:
            tasks.append(MethodTask(method=RemoverMethod.PATTERN_SH_STRIP, confidence=0.90))

        if CarrierFlag.VEC_PATH in carrier_set or record.path_cand:
            tasks.append(MethodTask(method=RemoverMethod.VEC_SPATIAL_BAND_CLUSTER, confidence=0.90))
            tasks.append(MethodTask(method=RemoverMethod.VEC_PATH_STRIP, confidence=0.88))

        # -------------------------------------------------------------------
        # 2. Raster Layer Triggers & Vetoes (Gate 2/3/4)
        # -------------------------------------------------------------------
        ras = record.ras_stats
        if ras and ras.has_raster:
            # Analyze cluster-wide context
            cluster_has_template = any(
                (r.ras_stats and r.ras_stats.tmpl_ncc >= 0.35) for r in subdoc_records
            )
            cluster_has_chroma = any(
                (r.ras_stats and r.ras_stats.sat_p95 >= 15.0) for r in subdoc_records
            )

            # Check Margin Banner Trigger (top/bottom margins, e.g. scanner text)
            if (CarrierFlag.RASTER_BAKED in carrier_set and ras.ocr_spans > 0) or any(
                (x.bb and (x.bb[1] < 50 or x.bb[3] > h_pt - 50)) for x in record.xobj_cand
            ):
                roi_margin = (
                    int(w_px * 0.45),
                    int(h_px * 0.90),
                    int(w_px * 0.995),
                    int(h_px * 0.995),
                )
                tasks.append(
                    MethodTask(
                        method=RemoverMethod.RASTER_MARGIN_BANNER,
                        roi_bbox=roi_margin,
                        confidence=0.89,
                    )
                )

            # Check Hue-Selective Chroma Trigger
            # Trigger: sat_p95 >= 15 with dominant watermark chroma band
            # Per-Page Veto: page is achromatic (sat_p95 < 12)
            if ras.sat_p95 >= 15.0:
                roi_chroma = (
                    int(w_px * 0.05),
                    int(h_px * 0.10),
                    int(w_px * 0.95),
                    int(h_px * 0.90),
                )
                tasks.append(
                    MethodTask(
                        method=RemoverMethod.RASTER_HUE_SELECTIVE_UNMIX,
                        roi_bbox=roi_chroma,
                        confidence=0.88,
                    )
                )

            # Check Bitonal Despeckle Trigger
            if ras.is_bitonal or CarrierFlag.RASTER_BITONAL in carrier_set:
                tasks.append(MethodTask(method=RemoverMethod.RASTER_BITONAL_DESPECKLE, confidence=0.90))

            # Check Template Watermark (e.g. diagonal watermark or repeating stamp)
            # Gate 2 Negative Presence Veto:
            # If the cluster has a template watermark (>= 0.35), but page p correlates poorly (< 0.35),
            # page p is an outlier clean page -> VETO!
            if cluster_has_template:
                if ras.tmpl_ncc >= 0.35:
                    roi_diag = (
                        int(w_px * 0.05),
                        int(h_px * 0.05),
                        int(w_px * 0.95),
                        int(h_px * 0.95),
                    )
                    tasks.append(
                        MethodTask(
                            method=RemoverMethod.RASTER_TEMPLATE_WHITEN,
                            roi_bbox=roi_diag,
                            confidence=0.88,
                        )
                    )
                    tasks.append(
                        MethodTask(
                            method=RemoverMethod.RASTER_ALPHA_INV,
                            roi_bbox=roi_diag,
                            confidence=0.85,
                        )
                    )
                # Else: tmpl_ncc < 0.35 -> VETOED! Do not add template removers!
            elif CarrierFlag.RASTER_BAKED in carrier_set and not ras.is_bitonal and not cluster_has_chroma:
                # Document has baked raster carrier but no cross-page template was established
                # (e.g. short 1-2 page document like FX_G19).
                # Schedule template whiten / alpha inv so L4 verifier can evaluate residual or detect opaque occlusions.
                roi_full = (
                    int(w_px * 0.05),
                    int(h_px * 0.05),
                    int(w_px * 0.95),
                    int(h_px * 0.95),
                )
                tasks.append(
                    MethodTask(
                        method=RemoverMethod.RASTER_TEMPLATE_WHITEN,
                        roi_bbox=roi_full,
                        confidence=0.85,
                    )
                )
                tasks.append(
                    MethodTask(
                        method=RemoverMethod.RASTER_ALPHA_INV,
                        roi_bbox=roi_full,
                        confidence=0.85,
                    )
                )

        reason = "scheduled tasks" if tasks else "no watermark triggers"
        return PageDispatchPlan(page=p_num, subdoc_id=subdoc_id, tasks=tasks, reason=reason)

    def plan_document(
        self,
        probe_records: list[ProbeRecord],
        reservoir: Optional[object] = None,
    ) -> list[PageDispatchPlan]:
        """Generate dispatch plan for entire document across all sub-document segments."""
        if not probe_records:
            return []

        subdoc_ids = self.segment_subdocuments(probe_records)

        # Group records by subdoc_id
        subdoc_groups: dict[int, list[ProbeRecord]] = {}
        for rec, s_id in zip(probe_records, subdoc_ids):
            subdoc_groups.setdefault(s_id, []).append(rec)

        plans: list[PageDispatchPlan] = []
        for rec, s_id in zip(probe_records, subdoc_ids):
            plan = self.plan_page(
                record=rec,
                subdoc_id=s_id,
                subdoc_records=subdoc_groups[s_id],
                reservoir=reservoir,
            )
            plans.append(plan)

        return plans
