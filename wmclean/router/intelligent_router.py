"""Intelligent Router — routes pages directly and seamlessly based on forensic dossier.

Takes the deep forensic profile (subdocuments, modality, parameters, completeness)
and directs every page to its specific required watermark removal technique,
ensuring zero false positives and zero token waste on deterministic pages.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from wmclean.analysis.forensic_analyzer import ForensicAnalyzer
from wmclean.intelligence.gatekeeper import MethodTask, PageDispatchPlan
from wmclean.schemas import (
    ClusterAction,
    DocumentForensicDossier,
    ProbeRecord,
    RemoverMethod,
    RemoverParams,
    RouterDecision,
    SubdocumentForensicProfile,
    WatermarkModality,
)

logger = logging.getLogger(__name__)


class IntelligentRouter:
    """Seamless Intelligent Router for forensic watermark removal."""

    def __init__(self, target_dpi: int = 150):
        self.target_dpi = target_dpi
        self.analyzer = ForensicAnalyzer(target_dpi=target_dpi)

    def route_document(
        self,
        records: list[ProbeRecord],
        dossier: Optional[DocumentForensicDossier] = None,
        pdf_path: Optional[str] = None,
        reservoir: Optional[Any] = None,
        subdoc_templates: Optional[dict[int, Any]] = None,
    ) -> tuple[RouterDecision, list[PageDispatchPlan]]:
        """Route all pages based on the forensic dossier.

        Args:
            records: List of ProbeRecord from L0 probe.
            dossier: Optional precomputed DocumentForensicDossier.
            pdf_path: Optional PDF file path.
            reservoir: Optional RasterReservoir.
            subdoc_templates: Optional precomputed templates.

        Returns:
            tuple[RouterDecision, list[PageDispatchPlan]]:
                Deterministic RouterDecision (for cluster-based flows) and
                per-page PageDispatchPlan list (for seamless execution).
        """
        if dossier is None:
            dossier = self.analyzer.analyze_document(
                records=records,
                pdf_path=pdf_path,
                reservoir=reservoir,
                subdoc_templates=subdoc_templates,
            )

        # Build subdoc lookup by page index
        subdoc_by_page: dict[int, SubdocumentForensicProfile] = {}
        for prof in dossier.subdocs:
            for p_idx in prof.page_indices:
                subdoc_by_page[p_idx] = prof

        plans: list[PageDispatchPlan] = []
        cluster_actions: list[ClusterAction] = []

        for p_idx, rec in enumerate(records):
            prof_opt: Optional[SubdocumentForensicProfile] = subdoc_by_page.get(p_idx)
            plan = self._route_page(rec, p_idx, prof_opt)
            plans.append(plan)

        # Build cluster actions for backwards compatibility
        for prof in dossier.subdocs:
            sig_hash = f"subdoc_{prof.subdoc_id}_{prof.detected_modality.value}"
            if prof.detected_modality == WatermarkModality.CLEAN_UNTOUCHED:
                cluster_actions.append(
                    ClusterAction(
                        sig_hash=sig_hash,
                        action="SKIP",
                        chain=[],
                        params=RemoverParams(),
                    )
                )
            else:
                params = RemoverParams(
                    target_xobj_names=prof.target_xobj_names,
                    target_texts=prof.target_texts,
                    target_path_hashes=prof.target_path_hashes,
                )
                cluster_actions.append(
                    ClusterAction(
                        sig_hash=sig_hash,
                        action="REMOVE",
                        chain=prof.recommended_chain,
                        params=params,
                    )
                )

        return RouterDecision(clusters=cluster_actions), plans

    def _route_page(
        self,
        record: ProbeRecord,
        p_idx: int,
        prof: Optional[SubdocumentForensicProfile],
    ) -> PageDispatchPlan:
        """Create surgical PageDispatchPlan for a single page."""
        p_num = p_idx + 1
        s_id = prof.subdoc_id if prof else 0
        mb = record.geom.mb if record.geom else [0, 0, 595, 842]
        w_pt = abs(mb[2] - mb[0])
        h_pt = abs(mb[3] - mb[1])
        w_px = max(1, int(w_pt * self.target_dpi / 72.0))
        h_px = max(1, int(h_pt * self.target_dpi / 72.0))

        if prof is None or prof.detected_modality == WatermarkModality.CLEAN_UNTOUCHED:
            return PageDispatchPlan(
                page=p_num,
                subdoc_id=s_id,
                tasks=[],
                reason="Negative presence veto: clean page verified",
            )

        tasks: list[MethodTask] = []
        mod = prof.detected_modality

        if mod in (
            WatermarkModality.CONTAINER_FORM_XOBJECT_STAMP,
            WatermarkModality.TOP_LEVEL_XOBJECT_IMAGE,
        ):
            has_xobj = any(xc.name in prof.target_xobj_names for xc in record.xobj_cand)
            if not has_xobj and prof.target_xobj_names:
                return PageDispatchPlan(
                    page=p_num,
                    subdoc_id=s_id,
                    tasks=[],
                    reason="Negative presence veto: target XObject absent on this page",
                )
            params = RemoverParams(target_xobj_names=prof.target_xobj_names)
            tasks.append(
                MethodTask(
                    method=RemoverMethod.XOBJ_DO_STRIP,
                    confidence=0.96,
                    params=params,
                )
            )

        elif mod == WatermarkModality.VECTOR_TEXT_ROTATED:
            cand_texts = [c.norm_str for c in record.text_cand]
            has_text = any(
                any(t.lower() in ct.lower() for ct in cand_texts)
                for t in prof.target_texts
            )
            if not has_text and prof.target_texts:
                return PageDispatchPlan(
                    page=p_num,
                    subdoc_id=s_id,
                    tasks=[],
                    reason="Negative presence veto: target vector text absent on this page",
                )
            params = RemoverParams(target_texts=prof.target_texts)
            tasks.append(
                MethodTask(
                    method=RemoverMethod.TEXT_OP_STRIP,
                    confidence=0.96,
                    params=params,
                )
            )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.AST_TJ_REASSEMBLE,
                    confidence=0.92,
                    params=params,
                )
            )

        elif mod == WatermarkModality.VECTOR_PATH_STROKES:
            has_paths = any(pc.hash in prof.target_path_hashes for pc in record.path_cand)
            if not has_paths and prof.target_path_hashes:
                return PageDispatchPlan(
                    page=p_num,
                    subdoc_id=s_id,
                    tasks=[],
                    reason="Negative presence veto: target vector paths absent on this page",
                )
            params = RemoverParams(target_path_hashes=prof.target_path_hashes)
            tasks.append(
                MethodTask(
                    method=RemoverMethod.VEC_SPATIAL_BAND_CLUSTER,
                    confidence=0.92,
                    params=params,
                )
            )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.VEC_PATH_STRIP,
                    confidence=0.90,
                    params=params,
                )
            )

        elif mod == WatermarkModality.SMASK_ALPHA_OVERLAY:
            if not any(g.smask for g in record.gs_alpha):
                return PageDispatchPlan(
                    page=p_num,
                    subdoc_id=s_id,
                    tasks=[],
                    reason="Negative presence veto: SMask absent on this page",
                )
            tasks.append(MethodTask(method=RemoverMethod.SMASK_GS_STRIP, confidence=0.92))

        elif mod == WatermarkModality.ANNOTATION_STAMP:
            if not record.annots:
                return PageDispatchPlan(
                    page=p_num,
                    subdoc_id=s_id,
                    tasks=[],
                    reason="Negative presence veto: annotations absent on this page",
                )
            tasks.append(MethodTask(method=RemoverMethod.ANNOT_DEL, confidence=0.99))

        elif mod == WatermarkModality.OCG_LAYER:
            if not record.ocgs:
                return PageDispatchPlan(
                    page=p_num,
                    subdoc_id=s_id,
                    tasks=[],
                    reason="Negative presence veto: OCG layers absent on this page",
                )
            tasks.append(MethodTask(method=RemoverMethod.OCG_STRIP, confidence=0.98))

        elif mod == WatermarkModality.SCANNER_MARGIN_BANNER:
            has_margin = any(
                (x.bb and (x.bb[3] - x.bb[1] < 120) and (x.bb[1] < 60 or x.bb[3] > h_pt - 60))
                for x in record.xobj_cand
            ) or any(
                any(kw in tc.norm_str.lower() for kw in ("camscanner", "scanned by", "tapscanner", "adobe scan"))
                for tc in record.text_cand
            )
            if not has_margin:
                return PageDispatchPlan(
                    page=p_num,
                    subdoc_id=s_id,
                    tasks=[],
                    reason="Negative presence veto: margin banner absent on this page",
                )
            roi_margin = (
                int(w_px * 0.40),
                int(h_px * 0.88),
                int(w_px * 0.995),
                int(h_px * 0.995),
            )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.RASTER_MARGIN_BANNER,
                    roi_bbox=roi_margin,
                    confidence=0.90,
                )
            )

        elif mod == WatermarkModality.SCANNED_CHROMATIC_STAMP:
            ras = record.ras_stats
            if ras and ras.sat_p95 < 6.0:
                return PageDispatchPlan(
                    page=p_num,
                    subdoc_id=s_id,
                    tasks=[],
                    reason="Negative presence veto: chromatic stamp absent on this page",
                )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.POINTWISE_HUE_AND_LUT,
                    roi_bbox=None,
                    confidence=0.94,
                )
            )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.RASTER_HUE_SELECTIVE_UNMIX,
                    roi_bbox=None,
                    confidence=0.90,
                )
            )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.PIRACY_OCR_SPOTTER,
                    roi_bbox=None,
                    confidence=0.88,
                )
            )

        elif mod == WatermarkModality.SCANNED_NEUTRAL_GRAY:
            ras = record.ras_stats
            has_neutral_wm = (
                ras
                and (
                    len(prof.page_indices) <= 2
                    or (
                        ras.tmpl_cov >= 0.002
                        and (ras.tmpl_ncc >= 0.25 or ras.est_alpha >= 0.03 or ras.sat_p95 >= 6.0)
                    )
                )
            )
            if not has_neutral_wm:
                return PageDispatchPlan(
                    page=p_num,
                    subdoc_id=s_id,
                    tasks=[],
                    reason="Negative presence veto: neutral watermark absent on this page",
                )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.TWO_REGIME_FILL,
                    roi_bbox=None,
                    confidence=0.98,
                )
            )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.POINTWISE_HUE_AND_LUT,
                    roi_bbox=None,
                    confidence=0.94,
                )
            )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.DENSITY_SUB,
                    roi_bbox=None,
                    confidence=0.88,
                )
            )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.RASTER_TEMPLATE_WHITEN,
                    roi_bbox=None,
                    confidence=0.86,
                )
            )
            tasks.append(
                MethodTask(
                    method=RemoverMethod.RASTER_ALPHA_INV,
                    roi_bbox=None,
                    confidence=0.84,
                )
            )

        return PageDispatchPlan(
            page=p_num,
            subdoc_id=s_id,
            tasks=tasks,
            reason=f"Routed to {mod.value}",
        )
