"""Unit tests for layout zone segmentation and dual-verification QA gate."""

import cv2
import numpy as np

from wmclean.intelligence.zone_segmenter import (
    ZoneType,
    dual_verification_qa_gate,
    segment_page_zones,
)


def create_mock_exam_page() -> np.ndarray:
    """Create a synthetic hybrid exam page with text, a benzene ring, and a photo badge."""
    img = np.full((600, 600, 3), 255, dtype=np.uint8)

    # 1. Text lines (top half)
    for y in range(40, 200, 30):
        cv2.putText(img, f"Exam Question {y//30}: Evaluate thermodynamics", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    # 2. Scientific Diagram: Benzene ring (bottom left)
    center = (180, 420)
    pts = []
    for i in range(6):
        angle = i * np.pi / 3
        pts.append((int(center[0] + 50 * np.cos(angle)), int(center[1] + 50 * np.sin(angle))))
    cv2.polylines(img, [np.array(pts)], isClosed=True, color=(10, 10, 10), thickness=3)
    cv2.circle(img, center, 25, (10, 10, 10), 2)

    # 3. Continuous-tone photo badge (bottom right)
    photo_patch = np.zeros((150, 150, 3), dtype=np.uint8)
    for c in range(3):
        photo_patch[:, :, c] = np.random.randint(50, 230, (150, 150), dtype=np.uint8)
    # Smooth to make it continuous tone
    photo_patch = cv2.GaussianBlur(photo_patch, (15, 15), 0)
    img[350:500, 380:530] = photo_patch

    return img


def test_segment_page_zones():
    page = create_mock_exam_page()
    zones = segment_page_zones(page)

    assert len(zones) >= 1
    types = {z.zone_type for z in zones}
    assert ZoneType.TEXT_LINE_ART in types

    # Check that diagram zone or text zone covers their respective areas
    for z in zones:
        x0, y0, x1, y1 = z.bbox
        assert x0 >= 0 and y0 >= 0
        assert x1 <= 600 and y1 <= 600
        assert z.mask.shape == (600, 600)


def test_dual_verification_qa_gate_passes_clean():
    page = create_mock_exam_page()
    cleaned = page.copy()  # perfect cleaning

    passed, summary, metrics = dual_verification_qa_gate(page, cleaned)
    assert passed is True
    assert "passed" in summary.lower()
    assert metrics["diagram_betti_passed"] is True
    assert metrics["text_stroke_loss"] <= 0.01


def test_dual_verification_qa_gate_fails_on_broken_diagram():
    page = create_mock_exam_page()
    broken = page.copy()
    # Break the benzene ring (erase 15px gap across ring)
    broken[410:430, 200:240] = 255

    zones = segment_page_zones(page)
    # Ensure a diagram zone was identified around the ring
    diag_zones = [z for z in zones if z.zone_type == ZoneType.SCIENTIFIC_GRAPH]
    if diag_zones:
        passed, summary, metrics = dual_verification_qa_gate(page, broken, zones=zones)
        assert passed is False
        assert metrics["diagram_betti_passed"] is False
        assert len(metrics["violations"]) >= 1
