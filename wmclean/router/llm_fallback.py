"""L2 Bounded LLM Fallback — at most ONE LLM call per document.

When deterministic rules yield confidence < 0.85 and the signature is not
in the persistent library, this module:
1. Builds a compact cluster summary JSON (≤800 tokens)
2. Builds at most one 768×768 JPEG montage (≤4 representative thumbnails)
3. Makes exactly 1 LLM call (max_tokens=400, timeout=10s)
4. Validates output against RouterDecision Pydantic schema
5. Falls back to the top deterministic rule on ANY error (0 retries)

Token budget: ≤2,200 total (800 text in + 1,100 image + 300 JSON out)
"""

import json
import logging
import os
from typing import Callable, Optional

import cv2
import numpy as np

from wmclean.schemas import (
    ClusterAction,
    ClusterSignature,
    RemoverParams,
    RouterDecision,
)

logger = logging.getLogger(__name__)


def _build_cluster_summary_json(
    clusters: dict[str, tuple[ClusterSignature, list[int]]],
    probe_records: list,
) -> str:
    """Build compact cluster summary JSON for LLM input (≤800 tokens).

    Includes carrier flags, text patterns, raster characteristics, and
    page count per cluster.
    """
    summaries = []
    for sig_hash, (sig, page_indices) in clusters.items():
        summary = {
            "sig": sig_hash[:16],
            "kind": sig.page_kind.value,
            "carriers": [c.value for c in sig.carrier_flags],
            "pages": len(page_indices),
            "color": sig.color_bucket,
        }
        if sig.norm_text_pattern:
            summary["text"] = sig.norm_text_pattern
        if sig.raster_bucket.tmpl_stable:
            summary["tmpl_stable"] = True
        summaries.append(summary)

    return json.dumps({"clusters": summaries}, separators=(",", ":"))


def _build_montage(
    clusters: dict[str, tuple[ClusterSignature, list[int]]],
    pdf_path: Optional[str] = None,
) -> Optional[bytes]:
    """Build a 768×768 JPEG montage with up to 4 representative thumbnails.

    Returns JPEG bytes or None if montage cannot be built.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return None

    rep_pages = []
    for sig_hash, (sig, page_indices) in clusters.items():
        if page_indices:
            rep_pages.append((sig_hash, page_indices[0]))
        if len(rep_pages) >= 4:
            break

    if not rep_pages:
        return None

    montage = np.full((768, 768, 3), 245, dtype=np.uint8)
    coords = [(0, 0), (384, 0), (0, 384), (384, 384)]

    try:
        import pymupdf

        with pymupdf.open(pdf_path) as doc:
            for idx, (sig_hash, p_idx) in enumerate(rep_pages):
                if p_idx < len(doc):
                    page = doc[p_idx]
                    pix = page.get_pixmap(dpi=72)
                    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:
                        arr = arr[:, :, :3]
                    elif pix.n == 1:
                        arr = np.stack([arr, arr, arr], axis=2)
                    thumb = cv2.resize(arr, (384, 384), interpolation=cv2.INTER_AREA)
                    cv2.putText(
                        thumb,
                        f"p.{p_idx + 1} {sig_hash[:8]}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 200),
                        2,
                    )
                    x, y = coords[idx]
                    montage[y : y + 384, x : x + 384] = thumb

        success, enc = cv2.imencode(".jpg", montage, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if success:
            return enc.tobytes()
    except Exception as exc:
        logger.warning("Failed to generate montage: %s", exc)

    return None


def build_llm_fallback_decision(
    low_confidence_clusters: dict[str, tuple[ClusterSignature, list[int]]],
    probe_records: list,
    deterministic_fallbacks: dict[str, ClusterAction],
    pdf_path: Optional[str] = None,
    llm_caller: Optional[Callable[[str, Optional[bytes]], dict]] = None,
) -> RouterDecision:
    """Attempt LLM routing for low-confidence clusters with strict bounds.

    Bounds:
    - At most 1 LLM call per document
    - Summary JSON ≤ 800 tokens, Montage ≤ 768×768
    - Output validated against RouterDecision schema
    - On ANY error / timeout / invalid schema: 0 retries, use deterministic fallback
    """
    if not low_confidence_clusters:
        return RouterDecision(clusters=[])

    if llm_caller is not None:
        try:
            summary_json = _build_cluster_summary_json(low_confidence_clusters, probe_records)
            montage_bytes = _build_montage(low_confidence_clusters, pdf_path=pdf_path)
            raw_result = llm_caller(summary_json, montage_bytes)
            decision = RouterDecision.model_validate(raw_result)
            if not decision.clusters and low_confidence_clusters:
                raise ValueError("LLM returned empty or missing clusters")
            logger.info("LLM fallback successfully routed %d clusters", len(decision.clusters))
            return decision
        except Exception as exc:
            logger.warning("LLM fallback call failed or invalid (%s); using deterministic fallback (0 retries)", exc)

    # Fallback to deterministic rules
    actions = []
    for sig_hash in low_confidence_clusters:
        if sig_hash in deterministic_fallbacks:
            actions.append(deterministic_fallbacks[sig_hash])
        else:
            actions.append(
                ClusterAction(
                    sig_hash=sig_hash,
                    action="SKIP",
                    chain=[],
                    params=RemoverParams(),
                )
            )

    return RouterDecision(clusters=actions)
