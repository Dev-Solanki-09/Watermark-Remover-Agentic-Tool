"""L4 Verification & Escalation Engine — applies candidate removers in order,
evaluates verification gates, rolls back upon failure, and escalates to the next
remover until clean or FLAGGED.

Guarantees:
- Zero false positives on text corruption: an unverified page is NEVER written to output.
- Instant rollback via content-stream snapshotting.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import pikepdf

from wmclean.schemas import (
    FailureCode,
    PageKind,
    PageState,
    RemoverMethod,
    RemoverParams,
    VerifyResult,
)
from wmclean.verify.exact import verify_exact
from wmclean.verify.raster import verify_raster

logger = logging.getLogger(__name__)


def _get_page_contents(page: pikepdf.Page) -> bytes:
    """Extract raw bytes from page /Contents (stream or Array of streams)."""
    if "/Contents" not in page:
        return b""
    c = page.Contents
    if isinstance(c, pikepdf.Array):
        return b"\n".join(bytes(s.read_bytes()) for s in c)
    return bytes(c.read_bytes())


class VerificationEngine:
    """Coordinates remover execution and verification gating with automatic rollback."""

    def __init__(self):
        pass

    def apply_with_rollback(
        self,
        pdf: pikepdf.Pdf,
        page_idx: int,
        page_kind: PageKind,
        remover_funcs: list[tuple[RemoverMethod, Callable, RemoverParams]],
        mask_bbox: Optional[list[float]] = None,
        known_wm_texts: Optional[list[str]] = None,
    ) -> tuple[PageState, list[str], VerifyResult]:
        """Apply a sequence of candidate removers with verification gates and rollback.

        Returns (PageState, methods_applied, VerifyResult).
        """
        import io

        page = pdf.pages[page_idx]

        # Take clean initial snapshot of page contents and annotations
        orig_contents_bytes = _get_page_contents(page)
        orig_annots = page.get("/Annots")
        orig_annots_copy = pikepdf.Array(orig_annots) if orig_annots is not None else None

        # Fully isolated baseline PDF for verification comparison
        orig_buf = io.BytesIO()
        pdf.save(orig_buf)
        orig_buf.seek(0)
        orig_pdf_copy = pikepdf.open(orig_buf)

        methods_applied = []
        last_result = VerifyResult(passed=False, reason="No removers executed")

        for method, func, params in remover_funcs:
            logger.debug("Testing remover %s on page %d", method.value, page_idx + 1)
            try:
                if method.value.startswith("raster_"):
                    # Raster remover execution
                    import cv2
                    import numpy as np
                    import pymupdf

                    from wmclean.assemble.encoder import update_page_image

                    # Render baseline page at 150 DPI
                    with pymupdf.open(stream=orig_buf.getvalue(), filetype="pdf") as doc:
                        pix = doc[page_idx].get_pixmap(dpi=150)
                        orig_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                        if pix.n == 4:
                            orig_arr = orig_arr[:, :, :3]
                        elif pix.n == 1:
                            orig_arr = np.stack([orig_arr, orig_arr, orig_arr], axis=2)

                    # Compute mask depending on remover
                    h, w = orig_arr.shape[:2]
                    grey = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2GRAY)
                    if method == RemoverMethod.RASTER_COLOR_UNMIX:
                        hsv = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2HSV)
                        mask = (hsv[:, :, 1] > 30).astype(np.uint8) * 255
                    elif method == RemoverMethod.RASTER_BITONAL_DESPECKLE:
                        mask = np.ones_like(grey, dtype=np.uint8) * 255
                    elif method == RemoverMethod.RASTER_FFT_NOTCH:
                        mask = np.ones_like(grey, dtype=np.uint8) * 255
                    else:
                        mask = ((grey > 160) & (grey < 248)).astype(np.uint8) * 255

                    if method == RemoverMethod.RASTER_ALPHA_INV:
                        tmpl = np.full_like(grey, 190)
                        cleaned_arr = func(orig_arr, tmpl, mask, params)
                    elif method == RemoverMethod.RASTER_BITONAL_DESPECKLE:
                        cleaned_arr = func(grey, mask, params)
                    else:
                        cleaned_arr = func(orig_arr, mask, params)

                    v_result = verify_raster(orig_arr, cleaned_arr, wm_mask=(mask > 0))
                    if v_result.passed:
                        # Find Image XObject name on this page
                        res = page.get("/Resources")
                        xobjs = res.get("/XObject") if res else None
                        img_name = None
                        if xobjs:
                            for name, obj in xobjs.items():
                                resolved = obj.resolve() if hasattr(obj, "resolve") else obj
                                if resolved.get("/Subtype") == pikepdf.Name.Image:
                                    img_name = str(name)
                                    break
                        if img_name:
                            is_bit = (method == RemoverMethod.RASTER_BITONAL_DESPECKLE)
                            update_page_image(pdf, page, img_name, cleaned_arr, is_bitonal=is_bit)
                        methods_applied.append(method.value)
                        logger.info("Page %d passed raster verification using %s", page_idx + 1, method.value)
                        orig_pdf_copy.close()
                        return PageState.CLEAN_VERIFIED, methods_applied, v_result
                    else:
                        logger.warning("Page %d raster remover %s failed verification (%s)", page_idx + 1, method.value, v_result.reason)
                        last_result = v_result
                        self._rollback_page(pdf, page, orig_contents_bytes, orig_annots_copy)
                        continue

                # Vector remover execution
                res = func(pdf, page, params) if "pdf" in func.__code__.co_varnames else func(page, params)

                # Run L4 Verification
                v_result = verify_exact(
                    orig_pdf_copy,
                    pdf,
                    page_idx,
                    mask_bbox=mask_bbox,
                    known_wm_texts=known_wm_texts,
                )

                if v_result.passed:
                    methods_applied.append(method.value)
                    state = PageState.CLEAN_EXACT if page_kind == PageKind.VECTOR else PageState.CLEAN_VERIFIED
                    logger.info("Page %d passed verification using %s", page_idx + 1, method.value)
                    orig_pdf_copy.close()
                    return state, methods_applied, v_result
                else:
                    logger.warning(
                        "Page %d remover %s failed verification (%s, code=%s); rolling back",
                        page_idx + 1,
                        method.value,
                        v_result.reason,
                        v_result.failure_code,
                    )
                    last_result = v_result
                    self._rollback_page(pdf, page, orig_contents_bytes, orig_annots_copy)

            except Exception as exc:
                logger.warning("Error running remover %s on page %d: %s; rolling back", method.value, page_idx + 1, exc)
                last_result = VerifyResult(
                    passed=False,
                    failure_code=FailureCode.ERR_VEC_VERIFY_BREACH,
                    reason=f"Remover exception: {exc}",
                )
                self._rollback_page(pdf, page, orig_contents_bytes, orig_annots_copy)

        orig_pdf_copy.close()
        # If all removers failed verification, leave page completely untouched
        self._rollback_page(pdf, page, orig_contents_bytes, orig_annots_copy)
        logger.warning("Page %d: all removers exhausted or failed verification; FLAGGED", page_idx + 1)
        return PageState.FLAGGED, methods_applied, last_result

    def _rollback_page(
        self,
        pdf: pikepdf.Pdf,
        page: pikepdf.Page,
        orig_contents: bytes,
        orig_annots: Optional[pikepdf.Array],
    ) -> None:
        """Roll back page contents and annotations to snapshot."""
        if orig_contents:
            page.Contents = pdf.make_stream(orig_contents)
        elif "/Contents" in page:
            del page["/Contents"]

        if orig_annots is not None:
            page["/Annots"] = orig_annots
        elif "/Annots" in page:
            del page["/Annots"]
