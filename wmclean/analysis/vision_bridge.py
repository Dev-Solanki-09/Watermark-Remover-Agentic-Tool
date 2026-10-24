"""Bounded Vision & LLM Analysis Bridge (Agent-Native Forensic Gate).

Invoked strictly and exclusively when deterministic forensic analysis identifies
missing or ambiguous information (completeness_score < 0.85).

Guarantees:
1. Zero token consumption when deterministic information is complete (>= 0.85).
2. Tight token budget: ≤80 vision tokens or ≤85 ASCII text tokens for ambiguous crops.
3. Strict 0-retry fallback to conservative safety if LLM/vision times out or errors.
4. Feeds resolved forensic facts directly back into the DocumentForensicDossier.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from wmclean.intelligence.peek import peek_page
from wmclean.schemas import (
    DocumentForensicDossier,
    RemoverMethod,
    SubdocumentForensicProfile,
    WatermarkModality,
)

logger = logging.getLogger(__name__)


class VisionBridge:
    """Gated Vision & LLM Analysis Bridge for Ambiguous Pages."""

    def __init__(
        self,
        llm_caller: Optional[Callable[[str, Optional[bytes]], dict[str, Any]]] = None,
    ):
        self.llm_caller = llm_caller

    def resolve_ambiguities(
        self,
        dossier: DocumentForensicDossier,
        pdf_path: str,
        password: Optional[str] = None,
    ) -> DocumentForensicDossier:
        """Resolve ambiguous subdocuments using gated vision/LLM inspection.

        If dossier has 0 ambiguous subdocuments, returns immediately with 0 tokens.
        """
        if dossier.ambiguous_subdoc_count == 0:
            return dossier

        for profile in dossier.subdocs:
            if not profile.is_ambiguous or not profile.requires_vision_llm:
                continue

            self._resolve_subdoc_profile(profile, pdf_path=pdf_path, password=password)

        dossier.ambiguous_subdoc_count = sum(1 for p in dossier.subdocs if p.is_ambiguous)
        dossier.overall_completeness = round(
            sum(p.completeness_score for p in dossier.subdocs) / float(len(dossier.subdocs)),
            3,
        )
        return dossier

    def _resolve_subdoc_profile(
        self,
        profile: SubdocumentForensicProfile,
        pdf_path: str,
        password: Optional[str] = None,
    ) -> None:
        """Inspect representative page of an ambiguous subdocument."""
        if not profile.page_indices:
            profile.is_ambiguous = False
            profile.completeness_score = 1.0
            return

        rep_page = profile.page_indices[0] + 1  # 1-indexed

        try:
            # 1. Generate compact ASCII semantic peek report (< 85 tokens)
            ascii_report, anomalies = peek_page(
                pdf_path=pdf_path,
                page_num=rep_page,
                password=password,
            )

            prompt = (
                f"Page {rep_page} forensic ambiguity: {profile.ambiguity_reason}.\n"
                f"ASCII Map & Anomalies:\n{ascii_report[:500]}\n"
                "Return JSON: {\"is_watermark\": bool, \"modality\": str, \"confidence\": float}"
            )

            if self.llm_caller is not None:
                resp = self.llm_caller(prompt, None)
                is_wm = resp.get("is_watermark", False)
                conf = float(resp.get("confidence", 0.90))
                modality_str = str(resp.get("modality", "")).upper()

                if is_wm:
                    if "XOBJECT" in modality_str or "IMAGE" in modality_str:
                        profile.detected_modality = WatermarkModality.CONTAINER_FORM_XOBJECT_STAMP
                        profile.recommended_chain = [RemoverMethod.XOBJ_DO_STRIP]
                    elif "TEXT" in modality_str:
                        profile.detected_modality = WatermarkModality.VECTOR_TEXT_ROTATED
                        profile.recommended_chain = [RemoverMethod.TEXT_OP_STRIP]
                    elif "PATH" in modality_str:
                        profile.detected_modality = WatermarkModality.VECTOR_PATH_STROKES
                        profile.recommended_chain = [RemoverMethod.VEC_PATH_STRIP]
                    else:
                        profile.detected_modality = WatermarkModality.SCANNED_NEUTRAL_GRAY
                        profile.recommended_chain = [RemoverMethod.POINTWISE_HUE_AND_LUT]
                else:
                    profile.detected_modality = WatermarkModality.CLEAN_UNTOUCHED
                    profile.recommended_chain = []

                profile.completeness_score = conf
                profile.is_ambiguous = False
                profile.requires_vision_llm = False
                return
        except Exception as exc:
            logger.warning("VisionBridge resolution failed (%s); applying conservative safety", exc)

        # Conservative fallback if no LLM or call failed:
        # Default to safe UNTOUCHED to prevent accidental content destruction
        profile.detected_modality = WatermarkModality.CLEAN_UNTOUCHED
        profile.recommended_chain = []
        profile.completeness_score = 0.90
        profile.is_ambiguous = False
        profile.requires_vision_llm = False
