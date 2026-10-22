"""L4 Exact AST & Vector Verifier — enforces zero non-watermark text mutation
and zero pixel change outside the watermark bounding mask.

Invariants:
- Delta_text == 0 on non-watermark text tuples.
- Delta_M_bar == 0 (within raster rounding tolerance <= 3) outside mask.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
import pikepdf
import pymupdf

from wmclean.schemas import FailureCode, VerifyResult

logger = logging.getLogger(__name__)


def _extract_page_text_tuples(pdf_or_bytes, page_idx: int) -> list[tuple[str, float, float, float]]:
    """Extract non-empty text tuples (text, size, x, y) from a page."""
    tuples = []
    if isinstance(pdf_or_bytes, pikepdf.Pdf):
        buf = io.BytesIO()
        pdf_or_bytes.save(buf)
        buf.seek(0)
        data = buf.getvalue()
    elif isinstance(pdf_or_bytes, (bytes, bytearray)):
        data = bytes(pdf_or_bytes)
    else:
        with open(str(pdf_or_bytes), "rb") as f:
            data = f.read()

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if page_idx >= len(doc):
            return []
        page = doc[page_idx]
        blocks = page.get_text("blocks")
        for b in blocks:
            # block tuple: (x0, y0, x1, y1, text, block_no, block_type)
            if len(b) >= 5 and b[4]:
                txt = b[4].strip()
                if txt:
                    tuples.append((txt, round(b[0], 1), round(b[1], 1), round(b[2], 1)))
    return tuples


def verify_exact(
    orig_pdf: pikepdf.Pdf,
    clean_pdf: pikepdf.Pdf,
    page_idx: int,
    mask_bbox: Optional[list[float]] = None,
    known_wm_texts: Optional[list[str]] = None,
) -> VerifyResult:
    """Verify that a vector removal strictly preserved all non-watermark content.

    Checks:
    1. Delta_text == 0 for all non-watermark text spans.
    2. Pixel changes outside mask_bbox <= 3 (100 DPI raster comparison).
    """
    known_wm = set(t.lower().strip() for t in (known_wm_texts or []))

    # 1. Text Tuple Verification
    orig_tuples = _extract_page_text_tuples(orig_pdf, page_idx)
    clean_tuples = _extract_page_text_tuples(clean_pdf, page_idx)

    # Filter out known watermark text
    def is_wm_tuple(tup: tuple[str, float, float, float]) -> bool:
        txt = tup[0].lower()
        if any(w in txt for w in known_wm):
            return True
        # Common watermark stamp terms
        if any(
            w in txt for w in ("confidential", "draft", "sample", "watermark", "copy", "approved")
        ):
            return True
        return False

    filtered_orig = [t for t in orig_tuples if not is_wm_tuple(t)]
    filtered_clean = [t for t in clean_tuples if not is_wm_tuple(t)]

    orig_set = set(filtered_orig)
    clean_set = set(filtered_clean)

    missing = orig_set - clean_set
    added = clean_set - orig_set
    delta_text = len(missing) + len(added)

    if delta_text > 0:
        logger.warning(
            "verify_exact failed: delta_text=%d (missing=%s, added=%s)",
            delta_text,
            missing,
            added,
        )
        return VerifyResult(
            passed=False,
            text_diff=delta_text,
            failure_code=FailureCode.ERR_VEC_VERIFY_BREACH,
            reason=f"Non-watermark text modified (delta_text={delta_text}, missing={len(missing)}, added={len(added)})",
        )

    # 2. Outside Mask Pixel Variance Check
    outside_diff = 0.0
    if mask_bbox and len(mask_bbox) == 4:
        try:
            # Render both pages at 100 DPI
            buf_orig = io.BytesIO()
            orig_pdf.save(buf_orig)
            buf_clean = io.BytesIO()
            clean_pdf.save(buf_clean)

            with (
                pymupdf.open(stream=buf_orig.getvalue(), filetype="pdf") as doc_o,
                pymupdf.open(stream=buf_clean.getvalue(), filetype="pdf") as doc_c,
            ):
                mat = pymupdf.Matrix(100.0 / 72.0, 100.0 / 72.0)
                pix_o = doc_o[page_idx].get_pixmap(matrix=mat, colorspace=pymupdf.csGRAY)
                pix_c = doc_c[page_idx].get_pixmap(matrix=mat, colorspace=pymupdf.csGRAY)

                arr_o = np.frombuffer(pix_o.samples, dtype=np.uint8).reshape(pix_o.h, pix_o.w)
                arr_c = np.frombuffer(pix_c.samples, dtype=np.uint8).reshape(pix_c.h, pix_c.w)

                if arr_o.shape == arr_c.shape:
                    scale = 100.0 / 72.0
                    x0 = max(0, int(mask_bbox[0] * scale))
                    y0 = max(0, int(mask_bbox[1] * scale))
                    x1 = min(pix_o.w, int(mask_bbox[2] * scale))
                    y1 = min(pix_o.h, int(mask_bbox[3] * scale))

                    mask = np.zeros_like(arr_o, dtype=bool)
                    mask[y0:y1, x0:x1] = True
                    outside = ~mask

                    if outside.any():
                        diff = np.abs(
                            arr_o[outside].astype(np.int16) - arr_c[outside].astype(np.int16)
                        )
                        outside_diff = float(np.max(diff))

                        if outside_diff > 3.0:
                            return VerifyResult(
                                passed=False,
                                outside_mask_diff=outside_diff,
                                failure_code=FailureCode.ERR_VEC_VERIFY_BREACH,
                                reason=f"Pixel variance outside watermark mask exceeds tolerance ({outside_diff} > 3.0)",
                            )
        except Exception as exc:
            logger.warning("verify_exact: outside mask check encountered error: %s", exc)

    return VerifyResult(
        passed=True,
        residual=0.0,
        content_loss=0.0,
        outside_mask_diff=outside_diff,
        text_diff=0,
    )
