import tempfile
from pathlib import Path

import cv2
import numpy as np

from wmclean.mcp_server import dewarp_document_page


def test_dewarp_document_page_mcp_flat():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        p = Path(tf.name)
    try:
        img = np.full((400, 300, 3), 255, dtype=np.uint8)
        cv2.putText(img, "Test Text", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(str(p), img)

        res = dewarp_document_page(file_path=str(p))
        assert "modality" in res
        assert res["modality"] == "GEOM_FLAT"
        assert res["applied"] is False
    finally:
        if p.exists():
            p.unlink()


def test_dewarp_document_page_mcp_skew():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        p = Path(tf.name)
    out_p = p.with_name(f"{p.stem}_rect.png")
    try:
        img = np.full((600, 400, 3), 255, dtype=np.uint8)
        for y in [100, 200, 300, 400]:
            cv2.line(img, (50, y), (350, y + 25), (0, 0, 0), 2)
        cv2.imwrite(str(p), img)

        res = dewarp_document_page(file_path=str(p), output_path=str(out_p), mode="skew")
        assert res["modality"] == "GEOM_SKEW_RIGID"
        assert res["applied"] is True
        assert out_p.exists()
    finally:
        if p.exists():
            p.unlink()
        if out_p.exists():
            out_p.unlink()
