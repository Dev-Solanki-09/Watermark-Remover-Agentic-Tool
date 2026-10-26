"""L4 Verification & Escalation Engine — applies candidate removers in order,
evaluates verification gates, rolls back upon failure, and escalates to the next
remover until clean or FLAGGED.

Enforces:
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
        return b"\n".join(s.read_bytes() for s in c)
    return c.read_bytes()


def _detect_page_dpi(mupdf_page: Any, default_dpi: int = 150, max_dpi: int = 300) -> int:
    """Detect native raster scan DPI to avoid silent downsampling of high-res scans."""
    try:
        w_pt = float(mupdf_page.rect.width)
        if w_pt <= 0:
            return default_dpi
        images = mupdf_page.get_images()
        if not images:
            return default_dpi
        max_img_w = max(int(img_info[2]) for img_info in images if len(img_info) > 2)
        if max_img_w > 0:
            calc_dpi = int(round(max_img_w * 72.0 / w_pt))
            if calc_dpi >= default_dpi:
                return min(calc_dpi, max_dpi)
    except Exception:
        pass
    return default_dpi


def _find_primary_image_xobject(page: pikepdf.Page) -> Optional[str]:
    """Find the primary full-page scan Image XObject to update, avoiding corrupting secondary logos or icons."""
    res = page.get("/Resources")
    xobjs = res.get("/XObject") if res else None
    if not xobjs:
        return None
    img_names = []
    for name, item in xobjs.items():
        if item.get("/Subtype") == pikepdf.Name.Image:
            img_names.append(name)
    if not img_names:
        return None
    if len(img_names) == 1:
        return img_names[0]

    # Calculate page aspect ratio if mediabox exists
    page_ar = 1.0
    try:
        mbox = page.get("/MediaBox")
        if mbox and len(mbox) == 4:
            pw = float(mbox[2]) - float(mbox[0])
            ph = float(mbox[3]) - float(mbox[1])
            if ph > 0:
                page_ar = pw / ph
    except Exception:
        page_ar = 1.0

    best_score = -1.0
    best_name: Optional[str] = None
    for name in img_names:
        raw_obj = xobjs.get(name)
        if raw_obj is not None:
            try:
                res_obj = raw_obj.resolve() if hasattr(raw_obj, "resolve") else raw_obj
                w_obj = int(res_obj.get("/Width", 0))
                h_obj = int(res_obj.get("/Height", 0))
                area = float(w_obj * h_obj)
                if area < 10000:
                    continue  # Ignore tiny icons/glyphs
                img_ar = w_obj / max(1.0, float(h_obj))
                # Prefer images matching page aspect ratio (ratio penalty close to 1)
                ar_diff = abs(img_ar - page_ar) / max(0.1, page_ar)
                score = area / (1.0 + ar_diff * 2.0)
                if score > best_score:
                    best_score = score
                    best_name = name
            except Exception:
                pass
    return best_name or img_names[0]


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
        dewarp: bool = False,
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
        any_modification_attempted = False
        last_result = VerifyResult(passed=False, reason="No removers executed")

        for item in remover_funcs:
            page = pdf.pages[page_idx]
            if len(item) == 4:
                method, func, params, item_roi = item
            else:
                method, func, params = item
                item_roi = roi_bbox

            logger.debug("Testing remover %s on page %d", method.value, page_idx + 1)
            try:
                is_raster = method.value.startswith("raster_") or method in {
                    RemoverMethod.TWO_REGIME_FILL,
                    RemoverMethod.INVERTED_BANNER_STRIP,
                    RemoverMethod.PIRACY_OCR_SPOTTER,
                    RemoverMethod.ILLUM_FLATTEN,
                    RemoverMethod.SWT_MORPH_RECONSTRUCT,
                    RemoverMethod.BEER_LAMBERT_STAIN_UNMIX,
                    RemoverMethod.BILINGUAL_TWIN_CLONE,
                    RemoverMethod.DENSITY_SUB,
                    RemoverMethod.POINTWISE_HUE_AND_LUT,
                    RemoverMethod.MAXTREE_FILTER,
                    RemoverMethod.SKELETON_GRAPH,
                    RemoverMethod.SURGICAL_LAMA,
                }
                if is_raster:
                    # Raster remover execution
                    import cv2
                    import numpy as np
                    import pymupdf

                    from wmclean.assemble.encoder import update_page_image
                    from wmclean.intelligence.doc_shield import parse_doc_shield
                    from wmclean.intelligence.roi_lock import apply_in_roi, clamp_roi_bbox

                    # Render baseline page at native or detected DPI (prevents silent downsampling)
                    with pymupdf.open(stream=orig_buf.getvalue(), filetype="pdf") as doc:
                        p_mupdf = doc[page_idx]
                        render_dpi = _detect_page_dpi(p_mupdf, default_dpi=150, max_dpi=300)
                        pix = p_mupdf.get_pixmap(dpi=render_dpi)
                        orig_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                            pix.h, pix.w, pix.n
                        )
                        if pix.n == 4:
                            orig_arr = orig_arr[:, :, :3]
                        elif pix.n == 1:
                            orig_arr = np.stack([orig_arr, orig_arr, orig_arr], axis=2)

                    if dewarp:
                        try:
                            from wmclean.geometry.dewarp import dewarp_page
                            orig_arr, _ = dewarp_page(orig_arr)
                        except Exception as exc:
                            logger.debug("Dewarp failed on page %d: %s", page_idx + 1, exc)

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
                    elif method == RemoverMethod.POINTWISE_HUE_AND_LUT:
                        hsv = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2HSV)
                        sat_p95 = np.percentile(hsv[:, :, 1], 95)
                        if sat_p95 > 15:
                            mask = (hsv[:, :, 1] > 20).astype(np.uint8) * 255
                        else:
                            mask = ((grey >= 110) & (grey <= 248)).astype(np.uint8) * 255
                    elif method == RemoverMethod.RASTER_BITONAL_DESPECKLE:
                        mask = np.ones_like(grey, dtype=np.uint8) * 255
                    elif method == RemoverMethod.RASTER_FFT_NOTCH:
                        mask = np.ones_like(grey, dtype=np.uint8) * 255
                    else:
                        mask = ((grey > 130) & (grey < 250)).astype(np.uint8) * 255

                    def _run_single_remover(crop_img, crop_mask=None, crop_tmpl=None):
                        c_grey = (
                            cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                            if crop_img.ndim == 3
                            else crop_img
                        )
                        m = (
                            crop_mask
                            if crop_mask is not None
                            else ((c_grey > 130) & (c_grey < 250)).astype(np.uint8) * 255
                        )
                        if method == RemoverMethod.RASTER_ALPHA_INV:
                            tmpl_use = (
                                crop_tmpl if crop_tmpl is not None else np.full_like(c_grey, 190)
                            )
                            return func(crop_img, tmpl_use, m, params)
                        elif method in (
                            RemoverMethod.TWO_REGIME_FILL,
                            RemoverMethod.RASTER_TEMPLATE_WHITEN,
                            RemoverMethod.RASTER_DARK_STAMP_CONSENSUS,
                            RemoverMethod.RASTER_REGISTERED_OUTLINE_CLEAN,
                            RemoverMethod.DENSITY_SUB,
                        ):
                            return func(crop_img, m, params, template=crop_tmpl)
                        elif method == RemoverMethod.RASTER_BITONAL_DESPECKLE:
                            return func(c_grey, m, params)
                        elif method == RemoverMethod.BEER_LAMBERT_STAIN_UNMIX:
                            return func(crop_img, mask=m)
                        elif method in (
                            RemoverMethod.POINTWISE_HUE_AND_LUT,
                            RemoverMethod.MAXTREE_FILTER,
                            RemoverMethod.SKELETON_GRAPH,
                            RemoverMethod.BILINGUAL_TWIN_CLONE,
                            RemoverMethod.SURGICAL_LAMA,
                        ):
                            return func(crop_img, mask=m, params=params)
                        else:
                            return func(crop_img, m, params)

                    if item_roi is not None:
                        rx1, ry1, rx2, ry2 = clamp_roi_bbox(item_roi, w, h, pad=6)
                        sub_mask = mask[ry1:ry2, rx1:rx2] if mask is not None else None
                        sub_tmpl = None
                        if template is not None:
                            t_h, t_w = template.shape[:2]
                            t_res = (
                                cv2.resize(template, (w, h)) if (t_h, t_w) != (h, w) else template
                            )
                            sub_tmpl = t_res[ry1:ry2, rx1:rx2]

                        cleaned_arr = apply_in_roi(
                            orig_arr,
                            item_roi,
                            lambda c: _run_single_remover(
                                c, crop_mask=sub_mask, crop_tmpl=sub_tmpl
                            ),
                        )
                    else:
                        cleaned_arr = _run_single_remover(
                            orig_arr, crop_mask=mask, crop_tmpl=template
                        )

                    # Self-healing Text-Core Lock: enforces body text and diagram cores remain 100% intact
                    shield = parse_doc_shield(orig_arr)
                    c_diff_raw = (
                        cv2.absdiff(
                            grey,
                            cv2.cvtColor(cleaned_arr, cv2.COLOR_RGB2GRAY)
                            if cleaned_arr.ndim == 3
                            else cleaned_arr,
                        )
                        > 8
                    )
                    if not c_diff_raw.any() or int(np.sum(c_diff_raw)) < 250:
                        logger.debug(
                            "Page %d raster remover %s: 0 modifications made (< 250 px change)",
                            page_idx + 1,
                            method.value,
                        )
                        continue

                    target_wm_mask = c_diff_raw
                    cleaned_arr = shield.apply_text_core_lock(
                        cleaned_arr, orig_arr, lock_halo=True, wm_mask=target_wm_mask
                    )
                    c_diff = (
                        cv2.absdiff(
                            grey,
                            cv2.cvtColor(cleaned_arr, cv2.COLOR_RGB2GRAY)
                            if cleaned_arr.ndim == 3
                            else cleaned_arr,
                        )
                        > 8
                    )
                    if not c_diff.any() or int(np.sum(c_diff)) < 250:
                        logger.debug(
                            "Page %d raster remover %s: all changes reverted by shield or sub-threshold",
                            page_idx + 1,
                            method.value,
                        )
                        continue

                    wm_mask_eval = mask if mask is not None else target_wm_mask

                    any_modification_attempted = True
                    v_result = verify_raster(
                        orig_arr,
                        cleaned_arr,
                        wm_mask=wm_mask_eval,
                        template=template,
                        enforce_four_invariants=True,
                    )
                    if v_result.passed:
                        # Find primary full-page scan Image XObject to update (avoids corrupting secondary logos/icons)
                        target_img = _find_primary_image_xobject(page)
                        if target_img:
                            is_bit = method == RemoverMethod.RASTER_BITONAL_DESPECKLE
                            update_page_image(
                                pdf, page, target_img, cleaned_arr, is_bitonal=is_bit
                            )
                        methods_applied.append(method.value)
                        logger.info(
                            "Page %d passed raster verification using %s",
                            page_idx + 1,
                            method.value,
                        )
                        orig_pdf_copy.close()
                        return PageState.CLEAN_VERIFIED, methods_applied, v_result
                    else:
                        logger.warning(
                            "Page %d raster remover %s failed verification (%s)",
                            page_idx + 1,
                            method.value,
                            v_result.reason,
                        )
                        last_result = v_result
                        last_result.diagnostic_images = (orig_arr.copy(), cleaned_arr.copy())
                        self._rollback_page(
                            pdf,
                            page,
                            orig_contents_bytes,
                            orig_annots_copy,
                            page_idx=page_idx,
                            orig_pdf_copy=orig_pdf_copy,
                        )
                        page = pdf.pages[page_idx]
                        continue

                # Vector remover execution
                if method == RemoverMethod.MUPDF_LAYER_REDACT:
                    import pymupdf

                    tmp_buf = io.BytesIO()
                    pdf.save(tmp_buf)
                    tmp_buf.seek(0)
                    with pymupdf.open(stream=tmp_buf.getvalue(), filetype="pdf") as f_doc:
                        f_page = f_doc[page_idx]
                        func(f_page, params=params)
                        out_b = f_doc.tobytes()
                    with pikepdf.open(io.BytesIO(out_b)) as f_pike:
                        if "/Contents" in f_pike.pages[page_idx]:
                            page.Contents = pdf.copy_foreign(f_pike.pages[page_idx].Contents)
                        if "/Annots" in f_pike.pages[page_idx]:
                            page["/Annots"] = pdf.copy_foreign(f_pike.pages[page_idx].Annots)
                        elif "/Annots" in page:
                            del page["/Annots"]
                    any_modification_attempted = True
                else:
                    res = (
                        func(pdf, page, params)
                        if "pdf" in func.__code__.co_varnames
                        else func(page, params)
                    )
                    if res is False or res == 0:
                        logger.debug(
                            "Page %d vector remover %s: 0 modifications made",
                            page_idx + 1,
                            method.value,
                        )
                        continue
                    any_modification_attempted = True

                # Run L4 Verification
                v_result = verify_exact(
                    orig_pdf_copy,
                    pdf,
                    page_idx,
                    mask_bbox=mask_bbox,
                    known_wm_texts=known_wm_texts or (params.target_texts if params else None),
                )

                if v_result.passed:
                    methods_applied.append(method.value)
                    last_result = v_result
                    orig_contents_bytes = _get_page_contents(page)
                    orig_annots = page.get("/Annots")
                    orig_annots_copy = pikepdf.Array(orig_annots) if orig_annots is not None else None
                    logger.info("Page %d vector step passed verification using %s", page_idx + 1, method.value)
                    continue
                else:
                    logger.warning(
                        "Page %d remover %s failed verification (%s, code=%s); rolling back",
                        page_idx + 1,
                        method.value,
                        v_result.reason,
                        v_result.failure_code,
                    )
                    last_result = v_result
                    self._rollback_page(
                        pdf,
                        page,
                        orig_contents_bytes,
                        orig_annots_copy,
                        page_idx=page_idx,
                        orig_pdf_copy=orig_pdf_copy,
                    )
                    page = pdf.pages[page_idx]

            except Exception as exc:
                logger.warning(
                    "Error running remover %s on page %d: %s; rolling back",
                    method.value,
                    page_idx + 1,
                    exc,
                )
                last_result = VerifyResult(
                    passed=False,
                    failure_code=FailureCode.ERR_VEC_VERIFY_BREACH,
                    reason=f"Remover exception: {exc}",
                )
                self._rollback_page(
                    pdf,
                    page,
                    orig_contents_bytes,
                    orig_annots_copy,
                    page_idx=page_idx,
                    orig_pdf_copy=orig_pdf_copy,
                )
                page = pdf.pages[page_idx]

        # Multi-remover cumulative cascade pass for multi-watermarked raster pages
        raster_methods = [
            rf
            for rf in remover_funcs
            if rf[0].value.startswith("raster_")
            or rf[0]
            in {
                RemoverMethod.INVERTED_BANNER_STRIP,
                RemoverMethod.PIRACY_OCR_SPOTTER,
                RemoverMethod.ILLUM_FLATTEN,
                RemoverMethod.SWT_MORPH_RECONSTRUCT,
                RemoverMethod.BEER_LAMBERT_STAIN_UNMIX,
                RemoverMethod.BILINGUAL_TWIN_CLONE,
                RemoverMethod.DENSITY_SUB,
                RemoverMethod.POINTWISE_HUE_AND_LUT,
                RemoverMethod.MAXTREE_FILTER,
                RemoverMethod.SKELETON_GRAPH,
                RemoverMethod.SURGICAL_LAMA,
            }
        ]
        if len(raster_methods) > 1 and not methods_applied:
            import cv2
            import numpy as np
            import pymupdf

            from wmclean.assemble.encoder import update_page_image
            from wmclean.intelligence.doc_shield import parse_doc_shield
            from wmclean.intelligence.roi_lock import apply_in_roi, clamp_roi_bbox

            with pymupdf.open(stream=orig_buf.getvalue(), filetype="pdf") as doc:
                p_mupdf = doc[page_idx]
                render_dpi = _detect_page_dpi(p_mupdf, default_dpi=150, max_dpi=300)
                pix = p_mupdf.get_pixmap(dpi=render_dpi)
                orig_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    orig_arr = orig_arr[:, :, :3]
                elif pix.n == 1:
                    orig_arr = np.stack([orig_arr, orig_arr, orig_arr], axis=2)

            shield = parse_doc_shield(orig_arr)
            cascade_arr = orig_arr.copy()
            cascade_applied = []
            h_c, w_c = cascade_arr.shape[:2]

            for m_item, f_item, p_item, r_item in raster_methods:
                try:
                    c_grey = (
                        cv2.cvtColor(cascade_arr, cv2.COLOR_RGB2GRAY)
                        if cascade_arr.ndim == 3
                        else cascade_arr
                    )
                    m_cand = ((c_grey > 130) & (c_grey < 250)).astype(np.uint8) * 255
                    if r_item is not None:
                        rx1, ry1, rx2, ry2 = clamp_roi_bbox(r_item, w_c, h_c, pad=6)
                        sub_m = m_cand[ry1:ry2, rx1:rx2]
                        sub_t = None
                        if template is not None:
                            sub_t = (
                                template[ry1:ry2, rx1:rx2]
                                if template.shape[:2] == (h_c, w_c)
                                else cv2.resize(template, (w_c, h_c))[ry1:ry2, rx1:rx2]
                            )

                        def _sub_run(crop):
                            if m_item in (
                                RemoverMethod.TWO_REGIME_FILL,
                                RemoverMethod.RASTER_TEMPLATE_WHITEN,
                                RemoverMethod.DENSITY_SUB,
                            ):
                                return f_item(crop, sub_m, p_item, template=sub_t)
                            elif m_item in (
                                RemoverMethod.POINTWISE_HUE_AND_LUT,
                                RemoverMethod.MAXTREE_FILTER,
                                RemoverMethod.SKELETON_GRAPH,
                                RemoverMethod.BILINGUAL_TWIN_CLONE,
                            ):
                                return f_item(crop, mask=sub_m, params=p_item)
                            return f_item(crop, sub_m, p_item)

                        cascade_arr = apply_in_roi(cascade_arr, r_item, _sub_run)
                    else:
                        if m_item in (
                            RemoverMethod.TWO_REGIME_FILL,
                            RemoverMethod.RASTER_TEMPLATE_WHITEN,
                            RemoverMethod.DENSITY_SUB,
                        ):
                            cascade_arr = f_item(cascade_arr, m_cand, p_item, template=template)
                        elif m_item in (
                            RemoverMethod.POINTWISE_HUE_AND_LUT,
                            RemoverMethod.MAXTREE_FILTER,
                            RemoverMethod.SKELETON_GRAPH,
                            RemoverMethod.BILINGUAL_TWIN_CLONE,
                        ):
                            cascade_arr = f_item(cascade_arr, mask=m_cand, params=p_item)
                        else:
                            cascade_arr = f_item(cascade_arr, m_cand, p_item)
                    orig_g = (
                        cv2.cvtColor(orig_arr, cv2.COLOR_RGB2GRAY)
                        if orig_arr.ndim == 3
                        else orig_arr
                    )
                    casc_g = (
                        cv2.cvtColor(cascade_arr, cv2.COLOR_RGB2GRAY)
                        if cascade_arr.ndim == 3
                        else cascade_arr
                    )
                    c_diff_step = cv2.absdiff(orig_g, casc_g) > 8
                    cascade_arr = shield.apply_text_core_lock(
                        cascade_arr, orig_arr, lock_halo=True, wm_mask=c_diff_step
                    )
                    cascade_applied.append(m_item.value)
                except Exception as exc:
                    logger.debug("Cascade step %s error: %s", m_item.value, exc)

            if cascade_applied:
                any_modification_attempted = True
            orig_g = cv2.cvtColor(orig_arr, cv2.COLOR_RGB2GRAY) if orig_arr.ndim == 3 else orig_arr
            casc_g = (
                cv2.cvtColor(cascade_arr, cv2.COLOR_RGB2GRAY)
                if cascade_arr.ndim == 3
                else cascade_arr
            )
            c_diff = cv2.absdiff(orig_g, casc_g) > 8
            casc_mask: Optional[np.ndarray] = c_diff
            v_res = verify_raster(orig_arr, cascade_arr, wm_mask=casc_mask, template=template)
            if v_res.passed:
                target_img = _find_primary_image_xobject(page)
                if target_img:
                    update_page_image(pdf, page, target_img, cascade_arr)
                logger.info(
                    "Page %d passed raster cascade verification with %s",
                    page_idx + 1,
                    cascade_applied,
                )
                orig_pdf_copy.close()
                return PageState.CLEAN_VERIFIED, cascade_applied, v_res
            else:
                last_result = v_res
                last_result.diagnostic_images = (orig_arr.copy(), cascade_arr.copy())

        if methods_applied:
            state = (
                PageState.CLEAN_EXACT
                if page_kind == PageKind.VECTOR
                else PageState.CLEAN_VERIFIED
            )
            orig_pdf_copy.close()
            return state, methods_applied, last_result

        orig_pdf_copy.close()
        # If all removers failed verification, leave page completely untouched
        self._rollback_page(
            pdf,
            page,
            orig_contents_bytes,
            orig_annots_copy,
            page_idx=page_idx,
            orig_pdf_copy=orig_pdf_copy,
        )
        if not any_modification_attempted and not remover_funcs:
            return (
                PageState.UNTOUCHED,
                [],
                VerifyResult(passed=True, residual=0.0, content_loss=0.0),
            )
        if last_result.failure_code == FailureCode.ERR_HIGH_RESIDUAL or (
            last_result.residual and last_result.residual > 0.035
        ):
            logger.warning(
                "Page %d: watermark residual remains uncleaned (R_wm=%.4f); UNVERIFIED",
                page_idx + 1,
                last_result.residual or 0.0,
            )
            return PageState.UNVERIFIED, methods_applied, last_result

        logger.warning(
            "Page %d: all removers exhausted or failed verification; FLAGGED", page_idx + 1
        )
        return PageState.FLAGGED, methods_applied, last_result

    def _rollback_page(
        self,
        pdf: pikepdf.Pdf,
        page: pikepdf.Page,
        orig_contents: bytes,
        orig_annots: Optional[pikepdf.Array],
        page_idx: Optional[int] = None,
        orig_pdf_copy: Optional[pikepdf.Pdf] = None,
    ) -> None:
        """Full transactional rollback: restores the entire page dictionary from snapshot."""
        if orig_pdf_copy is not None and page_idx is not None and page_idx < len(orig_pdf_copy.pages):
            try:
                pdf.pages[page_idx] = pikepdf.Page(pdf.copy_foreign(orig_pdf_copy.pages[page_idx].obj))
                return
            except Exception as exc:
                logger.debug("Transactional copy_foreign rollback fallback: %s", exc)

        if orig_contents:
            page.Contents = pdf.make_stream(orig_contents)
        elif "/Contents" in page:
            del page["/Contents"]

        if orig_annots is not None:
            page["/Annots"] = orig_annots
        elif "/Annots" in page:
            del page["/Annots"]
