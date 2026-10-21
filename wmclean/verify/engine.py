"""L4 Verification & Escalation Engine — applies candidate removers in order,
evaluates verification gates, rolls back upon failure, and escalates to the next
remover until clean or FLAGGED.

Guarantees:
- Zero false positives on text corruption: an unverified page is NEVER written to output.
- Instant rollback via content-stream snapshotting.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pikepdf

from wmclean.schemas import (
    FailureCode,
    PageKind,
    PageState,
    RemoverMethod,
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
        remover_funcs: list[Any],
        mask_bbox: Optional[list[float]] = None,
        known_wm_texts: Optional[list[str]] = None,
        template: Optional[np.ndarray] = None,
        roi_bbox: Optional[tuple[int, int, int, int]] = None,
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

        for item in remover_funcs:
            if len(item) == 4:
                method, func, params, item_roi = item
            else:
                method, func, params = item
                item_roi = roi_bbox

            logger.debug("Testing remover %s on page %d", method.value, page_idx + 1)
            try:
                is_raster = (
                    method.value.startswith("raster_")
                    or method in {
                        RemoverMethod.INVERTED_BANNER_STRIP,
                        RemoverMethod.PIRACY_OCR_SPOTTER,
                        RemoverMethod.ILLUM_FLATTEN,
                        RemoverMethod.SWT_MORPH_RECONSTRUCT,
                        RemoverMethod.BEER_LAMBERT_STAIN_UNMIX,
                    }
                )
                if is_raster:
                    # Raster remover execution
                    import cv2
                    import numpy as np
                    import pymupdf

                    from wmclean.assemble.encoder import update_page_image
                    from wmclean.intelligence.roi_lock import apply_in_roi, clamp_roi_bbox

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
                    if method in (
                        RemoverMethod.RASTER_COLOR_UNMIX,
                        RemoverMethod.RASTER_HUE_SELECTIVE_UNMIX,
                        RemoverMethod.BEER_LAMBERT_STAIN_UNMIX,
                    ):
                        hsv = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2HSV)
                        mask = (hsv[:, :, 1] > 25).astype(np.uint8) * 255
                    elif method == RemoverMethod.RASTER_BITONAL_DESPECKLE:
                        mask = np.ones_like(grey, dtype=np.uint8) * 255
                    elif method == RemoverMethod.RASTER_FFT_NOTCH:
                        mask = np.ones_like(grey, dtype=np.uint8) * 255
                    else:
                        mask = ((grey > 130) & (grey < 250)).astype(np.uint8) * 255

                    if method in (
                        RemoverMethod.RASTER_TEMPLATE_WHITEN,
                        RemoverMethod.RASTER_DARK_STAMP_CONSENSUS,
                        RemoverMethod.RASTER_REGISTERED_OUTLINE_CLEAN,
                    ):
                        fg_m = (grey < 120).astype(np.uint8) * 255
                        fg_p = cv2.dilate(fg_m, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
                        if template is not None:
                            t_grey = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY) if template.ndim == 3 else template
                            if t_grey.shape != grey.shape:
                                t_grey = cv2.resize(t_grey, (grey.shape[1], grey.shape[0]))
                            bg_t = float(np.percentile(t_grey, 95))
                            delta_t = np.where((t_grey > 130) & (t_grey < 245), bg_t - t_grey.astype(np.float32), 0.0)
                            t_mask = (delta_t > 15) & (fg_p == 0)
                            if t_mask.sum() > 50:
                                eval_mask = t_mask.astype(np.uint8) * 255
                            else:
                                eval_mask = (((grey > 130) & (grey < 250)) & (fg_p == 0)).astype(np.uint8) * 255
                        else:
                            eval_mask = (((grey > 130) & (grey < 250)) & (fg_p == 0)).astype(np.uint8) * 255
                    else:
                        eval_mask = mask

                    def _run_single_remover(crop_img, crop_mask=None, crop_tmpl=None):
                        c_grey = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY) if crop_img.ndim == 3 else crop_img
                        m = crop_mask if crop_mask is not None else ((c_grey > 130) & (c_grey < 250)).astype(np.uint8) * 255
                        if method == RemoverMethod.RASTER_ALPHA_INV:
                            tmpl_use = crop_tmpl if crop_tmpl is not None else np.full_like(c_grey, 190)
                            return func(crop_img, tmpl_use, m, params)
                        elif method in (
                            RemoverMethod.RASTER_TEMPLATE_WHITEN,
                            RemoverMethod.RASTER_DARK_STAMP_CONSENSUS,
                            RemoverMethod.RASTER_REGISTERED_OUTLINE_CLEAN,
                        ):
                            return func(crop_img, m, params, template=crop_tmpl)
                        elif method == RemoverMethod.RASTER_BITONAL_DESPECKLE:
                            return func(c_grey, m, params)
                        elif method == RemoverMethod.BEER_LAMBERT_STAIN_UNMIX:
                            return func(crop_img, mask=m)
                        else:
                            return func(crop_img, m, params)

                    if item_roi is not None:
                        rx1, ry1, rx2, ry2 = clamp_roi_bbox(item_roi, w, h, pad=6)
                        sub_mask = mask[ry1:ry2, rx1:rx2] if mask is not None else None
                        sub_tmpl = None
                        if template is not None:
                            t_h, t_w = template.shape[:2]
                            t_res = cv2.resize(template, (w, h)) if (t_h, t_w) != (h, w) else template
                            sub_tmpl = t_res[ry1:ry2, rx1:rx2]

                        cleaned_arr = apply_in_roi(
                            orig_arr,
                            item_roi,
                            lambda c: _run_single_remover(c, crop_mask=sub_mask, crop_tmpl=sub_tmpl),
                        )
                    else:
                        cleaned_arr = _run_single_remover(orig_arr, crop_mask=mask, crop_tmpl=template)

                    c_diff = cv2.absdiff(grey, cv2.cvtColor(cleaned_arr, cv2.COLOR_RGB2GRAY) if cleaned_arr.ndim == 3 else cleaned_arr) > 8
                    wm_mask_eval = (eval_mask > 0) | c_diff

                    v_result = verify_raster(orig_arr, cleaned_arr, wm_mask=wm_mask_eval, template=template)
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
