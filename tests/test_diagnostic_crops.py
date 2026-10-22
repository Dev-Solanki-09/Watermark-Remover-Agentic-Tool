"""Unit tests for automatic diagnostic diff crops for vision agents."""

from pathlib import Path

import cv2
import numpy as np

from wmclean.intelligence.peek import generate_diagnostic_diff_crop


def test_generate_diagnostic_diff_crop(tmp_path: Path):
    h, w = 200, 200
    orig = np.full((h, w, 3), 255, dtype=np.uint8)
    # Add authentic text
    cv2.putText(orig, "ORIGINAL", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Cleaned image with watermark residual
    cleaned = orig.copy()
    cv2.putText(cleaned, "DIFF", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    out_file = tmp_path / "test_p1_diff.png"
    crop_res = generate_diagnostic_diff_crop(orig, cleaned, str(out_file), max_dim=400)

    assert Path(crop_res).exists()
    saved = cv2.imread(crop_res)
    assert saved is not None
    # Saved image should be a 3-pane horizontal composite: [orig | diff | cleaned]
    sh, sw = saved.shape[:2]
    assert sw > sh, "Diagnostic diff should be wide horizontal composite"
