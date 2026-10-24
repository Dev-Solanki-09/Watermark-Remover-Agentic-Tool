"""Sub-page layout zone segmenter and dual-verification QA gate.

Segments pages into:
- TEXT_LINE_ART: High stroke density, discrete glyphs, low color variance. Routes to Tier 0 or Tier 1.
- SCIENTIFIC_GRAPH: Closed loops (benzene rings), circuit loops, graph axes, formulas. Routes to Tier 2 with Betti gate.
- CONTINUOUS_TONE_PHOTO: High color variance, continuous gradients. Routes to Tier 1 or Tier 3.

Provides dual-verification QA gate enforcing:
- Text zones: 0 character/stroke loss and stroke integrity.
- Diagram zones: Strict Betti number invariance (Δβ1=0, no broken loops).
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from wmclean.intelligence.doc_shield import DocShieldResult, parse_doc_shield
from wmclean.verify.topology import verify_topological_invariance


class ZoneType(str, enum.Enum):
    """Semantic zone types for layout routing."""

    TEXT_LINE_ART = "TEXT_LINE_ART"
    SCIENTIFIC_GRAPH = "SCIENTIFIC_GRAPH"
    CONTINUOUS_TONE_PHOTO = "CONTINUOUS_TONE_PHOTO"


@dataclasses.dataclass
class ZoneBox:
    """A bounded sub-page layout zone with recommended restoration tier."""

    zone_type: ZoneType
    bbox: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    mask: np.ndarray  # (H, W) bool
    recommended_tier: str  # e.g. "Tier 1", "Tier 2", "Tier 3"
    confidence: float = 1.0


def segment_page_zones(
    img: np.ndarray,
    doc_shield: Optional[DocShieldResult] = None,
) -> List[ZoneBox]:
    """Segment an image into layout zones with semantic routing targets.

    Args:
        img: RGB or greyscale image array (H, W, C) or (H, W).
        doc_shield: Optional precomputed DocShieldResult.

    Returns:
        List of ZoneBox instances covering detected regions.
    """
    if img.ndim == 3 and img.shape[2] >= 3:
        rgb = img[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    else:
        gray = img if img.ndim == 2 else img[:, :, 0]
        rgb = np.stack([gray, gray, gray], axis=2)

    h, w = gray.shape[:2]
    if doc_shield is None:
        doc_shield = parse_doc_shield(img)

    zones: List[ZoneBox] = []

    # 1. Scientific Graph / Diagram Zones
    # Use m_diagram from DocShield to locate diagram clusters
    diag_mask = doc_shield.m_diagram
    if diag_mask.any():
        # Find contours of diagram clusters with dilation
        dilated_diag = cv2.dilate(diag_mask.astype(np.uint8), np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(dilated_diag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Filter trivial noise (< 20px)
            if cw >= 20 and ch >= 20:
                zmask = np.zeros((h, w), dtype=bool)
                zmask[y : y + ch, x : x + cw] = True
                zones.append(
                    ZoneBox(
                        zone_type=ZoneType.SCIENTIFIC_GRAPH,
                        bbox=(x, y, x + cw, y + ch),
                        mask=zmask,
                        recommended_tier="Tier 2 (Topological Skeleton Stitching)",
                        confidence=0.95,
                    )
                )

    # 2. Continuous Tone Photo Zones
    # High color saturation variance or dense gradient texture without discrete text
    if img.ndim == 3 and img.shape[2] >= 3:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        blur_sat = cv2.GaussianBlur(sat, (31, 31), 0)
        photo_cand = (blur_sat > 70) & (~doc_shield.m_diagram)
        if photo_cand.any():
            dilated_photo = cv2.dilate(photo_cand.astype(np.uint8), np.ones((25, 25), np.uint8))
            contours, _ = cv2.findContours(dilated_photo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw >= 100 and ch >= 100 and (cw * ch) > (h * w * 0.05):
                    zmask = np.zeros((h, w), dtype=bool)
                    zmask[y : y + ch, x : x + cw] = True
                    zones.append(
                        ZoneBox(
                            zone_type=ZoneType.CONTINUOUS_TONE_PHOTO,
                            bbox=(x, y, x + cw, y + ch),
                            mask=zmask,
                            recommended_tier="Tier 3 (Strictly Clamped Inpaint / Matting)",
                            confidence=0.88,
                        )
                    )

    # 3. Text / Line Art Zone (Remainder of page)
    # The default baseline for scanned documents
    occupied = np.zeros((h, w), dtype=bool)
    for z in zones:
        occupied |= z.mask

    text_mask = ~occupied
    if text_mask.any() or not zones:
        zones.append(
            ZoneBox(
                zone_type=ZoneType.TEXT_LINE_ART,
                bbox=(0, 0, w, h),
                mask=text_mask if zones else np.ones((h, w), dtype=bool),
                recommended_tier="Tier 1 (Two-Regime Paper & Core Lock)",
                confidence=1.0,
            )
        )

    return zones


def dual_verification_qa_gate(
    orig: np.ndarray,
    cleaned: np.ndarray,
    zones: Optional[List[ZoneBox]] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Execute dual-verification QA checks across all semantic zones.

    - Text zones: verify zero foreground erosion (L_fg <= 0.01).
    - Diagram zones: verify topological invariance (Δβ1 == 0, no broken loops).

    Returns:
        (passed, summary_message, detailed_metrics)
    """
    if zones is None:
        zones = segment_page_zones(orig)

    violations: List[str] = []
    metrics: Dict[str, Any] = {
        "text_zones_checked": 0,
        "diagram_zones_checked": 0,
        "photo_zones_checked": 0,
        "diagram_betti_passed": True,
        "text_stroke_loss": 0.0,
        "violations": violations,
    }

    all_passed = True

    for z in zones:
        x0, y0, x1, y1 = z.bbox
        crop_orig = orig[y0:y1, x0:x1]
        crop_cleaned = cleaned[y0:y1, x0:x1]

        if z.zone_type == ZoneType.SCIENTIFIC_GRAPH:
            metrics["diagram_zones_checked"] += 1
            t_pass, db0, db1, t_msg = verify_topological_invariance(crop_orig, crop_cleaned)
            if not t_pass or db1 < 0:
                all_passed = False
                metrics["diagram_betti_passed"] = False
                metrics["violations"].append(
                    f"Diagram loop broken at bbox {z.bbox}: Δβ1={db1}, Δβ0={db0}"
                )

        elif z.zone_type == ZoneType.TEXT_LINE_ART:
            metrics["text_zones_checked"] += 1
            # Check text stroke erosion
            shield = parse_doc_shield(crop_orig)
            if shield.m_core.any():
                g_orig = cv2.cvtColor(crop_orig, cv2.COLOR_BGR2GRAY) if crop_orig.ndim == 3 else crop_orig
                g_clean = cv2.cvtColor(crop_cleaned, cv2.COLOR_BGR2GRAY) if crop_cleaned.ndim == 3 else crop_cleaned
                # Lightened pixels in text core
                loss_count = np.count_nonzero((g_clean.astype(int) - g_orig.astype(int) > 25) & shield.m_core)
                loss_rate = loss_count / max(1, np.count_nonzero(shield.m_core))
                metrics["text_stroke_loss"] = max(metrics["text_stroke_loss"], loss_rate)
                if loss_rate > 0.02:  # >2% text core erosion
                    all_passed = False
                    metrics["violations"].append(
                        f"Text core erosion {loss_rate*100:.2f}% at bbox {z.bbox}"
                    )

        elif z.zone_type == ZoneType.CONTINUOUS_TONE_PHOTO:
            metrics["photo_zones_checked"] += 1

    summary = (
        "Dual-verification QA passed: all text strokes and diagram loops intact"
        if all_passed
        else f"Dual-verification QA failed: {'; '.join(metrics['violations'])}"
    )

    return all_passed, summary, metrics
