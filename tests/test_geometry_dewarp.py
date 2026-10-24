import cv2
import numpy as np

from wmclean.geometry.dewarp import (
    dewarp_2d_wavy_mesh,
    dewarp_cylindrical_gutter,
    dewarp_page,
    rotate_expanded_canvas,
)
from wmclean.schemas import GeometricModality


def test_rotate_expanded_canvas_zero_crop():
    h, w = 400, 300
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (280, 380), 0, 2)
    rotated = rotate_expanded_canvas(img, angle_deg=5.0)
    # Canvas MUST be larger than original to guarantee zero clipping
    assert rotated.shape[0] > h
    assert rotated.shape[1] > w
    # Make sure text strokes / lines were preserved inside rotated image
    assert np.count_nonzero(rotated < 100) > 0


def test_dewarp_page_flat_identity():
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Flat Text", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    out, result = dewarp_page(img)
    assert result.modality == GeometricModality.GEOM_FLAT
    assert result.applied is False
    assert np.array_equal(out, img)


def test_dewarp_page_skewed():
    img = np.full((600, 400), 255, dtype=np.uint8)
    for y in [100, 200, 300, 400]:
        cv2.line(img, (50, y), (350, y + 25), 0, 2)
    out, result = dewarp_page(img)
    assert result.applied is True
    assert result.modality == GeometricModality.GEOM_SKEW_RIGID
    assert out.shape[0] >= img.shape[0]
    assert out.shape[1] >= img.shape[1]


def test_dewarp_page_gutter_curved():
    img = np.full((600, 400), 255, dtype=np.uint8)
    for y in [100, 200, 300, 400]:
        pts = [(x, int(y + 25 * np.exp(-x / 35.0))) for x in range(30, 370, 5)]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], 0, 2)
    out, result = dewarp_page(img)
    assert result.applied is True
    assert result.modality in (GeometricModality.GEOM_GUTTER_CURVE, GeometricModality.GEOM_WAVY_BUMP)
    assert out.shape[0] >= img.shape[0]
    assert out.shape[1] >= img.shape[1]


def test_dewarp_cylindrical_gutter_direct():
    img = np.full((400, 300), 255, dtype=np.uint8)
    cv2.line(img, (30, 200), (270, 200), 0, 2)
    rectified = dewarp_cylindrical_gutter(img, spine_dip_px=15.0, side="left")
    assert rectified.shape[0] >= img.shape[0]
    assert rectified.shape[1] >= img.shape[1]


def test_dewarp_2d_wavy_mesh_direct():
    img = np.full((400, 300), 255, dtype=np.uint8)
    cv2.line(img, (30, 200), (270, 200), 0, 2)
    rectified = dewarp_2d_wavy_mesh(img, amplitude_px=10.0, wavelength_px=80.0)
    assert rectified.shape[0] >= img.shape[0]
    assert rectified.shape[1] >= img.shape[1]
