import cv2
import numpy as np

from wmclean.geometry.probe_geom import probe_page_geometry
from wmclean.schemas import GeometricModality


def test_probe_flat_page():
    img = np.full((600, 400), 255, dtype=np.uint8)
    for y in [100, 200, 300, 400]:
        cv2.line(img, (50, y), (350, y), 0, 2)
    probe = probe_page_geometry(img)
    assert probe.modality == GeometricModality.GEOM_FLAT
    assert probe.is_flat is True
    assert abs(probe.skew_angle_deg) < 0.3


def test_probe_skewed_page():
    img = np.full((600, 400), 255, dtype=np.uint8)
    for y in [100, 200, 300, 400]:
        cv2.line(img, (50, y), (350, y + 20), 0, 2)
    probe = probe_page_geometry(img)
    assert probe.modality == GeometricModality.GEOM_SKEW_RIGID
    assert abs(probe.skew_angle_deg) >= 1.0


def test_probe_gutter_curved_page():
    img = np.full((600, 400), 255, dtype=np.uint8)
    for y in [100, 200, 300, 400]:
        pts = [(x, int(y + 25 * np.exp(-x / 35.0))) for x in range(30, 370, 5)]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], 0, 2)
    probe = probe_page_geometry(img)
    assert probe.modality == GeometricModality.GEOM_GUTTER_CURVE
    assert probe.spine_dip_px >= 3.0
