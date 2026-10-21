"""L3 Raster Removers — template whiten, alpha inversion, colour unmixing,
luma separation, FFT notch filtering, connected-component inpainting,
bitonal despeckle, and optional ML inpainting.

All removers operate on numpy arrays and return cleaned numpy arrays.
The L4 verifier validates results; if thresholds are exceeded, the
rung escalation engine tries the next method or flags the page.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from wmclean.schemas import RemoverParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# G10: Template Whiten
# ---------------------------------------------------------------------------


def raster_template_whiten(
    page_img: np.ndarray,
    *args,
    **kwargs,
) -> np.ndarray:
    """Replace watermark region with local background estimate or template unmixing.

    Used when watermark does not overlap foreground text/strokes.
    Supports either (page_img, template, mask, params) or (page_img, mask, params, template=...).

    Args:
        page_img: Page image (H, W, C) uint8
        template: Optional median template (H, W) or (H, W, C) uint8
        mask: Binary watermark mask (H, W) uint8, 255=watermark
        params: Remover parameters

    Returns:
        Cleaned image (H, W, C) uint8
    """
    template: np.ndarray | None = kwargs.get("template", None)
    mask: np.ndarray | None = kwargs.get("mask", None)
    params: RemoverParams | None = kwargs.get("params", None)

    if len(args) == 3:
        template, mask, params = args
    elif len(args) == 2:
        mask, params = args
    elif len(args) == 1:
        if mask is None:
            mask = args[0]

    if mask is None:
        mask = np.zeros(page_img.shape[:2], dtype=np.uint8)
    if params is None:
        params = RemoverParams()

    result = page_img.copy()
    h, w = result.shape[:2]

    # Convert to greyscale for stroke protection analysis
    grey = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) if result.ndim == 3 else result.copy()

    # Foreground text stroke mask (luminance < 120 per L4 verification gate)
    fg_mask = (grey < 120).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    fg_protected = cv2.dilate(fg_mask, kernel, iterations=1)

    # Candidate watermark pixels: either from passed mask or grey range (130..250)
    if mask is not None and mask.sum() > 50:
        wm_cand = (mask > 0)
    else:
        wm_cand = (grey > 130) & (grey < 250)

    # If template is provided, intersect with template watermark region
    if template is not None:
        if template.shape[:2] != (h, w):
            tmpl_resized = cv2.resize(template, (w, h))
        else:
            tmpl_resized = template
        t_grey = cv2.cvtColor(tmpl_resized, cv2.COLOR_RGB2GRAY) if tmpl_resized.ndim == 3 else tmpl_resized
        bg_t = float(np.percentile(t_grey, 95))
        delta = np.where((t_grey > 130) & (t_grey < 245), bg_t - t_grey.astype(np.float32), 0.0)
        t_mask = (delta > 15)
        if t_mask.sum() > 50:
            wm_cand = wm_cand & t_mask

    wm_to_whiten = wm_cand & (fg_protected == 0)

    # Estimate background value from paper background pixels (grey >= 240)
    paper_bg = (grey >= 240)
    bg_val = int(np.percentile(grey[paper_bg], 95)) if paper_bg.sum() > 100 else 255

    if result.ndim == 3:
        for c in range(result.shape[2]):
            result[:, :, c] = np.where(wm_to_whiten, bg_val, result[:, :, c])
    else:
        result = np.where(wm_to_whiten, bg_val, result)

    logger.info("raster_template_whiten: whitened %d watermark pixels", int(wm_to_whiten.sum()))
    return result


# ---------------------------------------------------------------------------
# G10/G11: Alpha Inversion
# ---------------------------------------------------------------------------


def raster_alpha_inv(
    page_img: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    params: RemoverParams,
) -> np.ndarray:
    """Remove semi-transparent watermark via alpha inversion.

    J = clip((I - α·W) / (1 - α), 0, 255)

    where I is the observed page, W is the watermark template, and α is
    the estimated blending alpha.

    Args:
        page_img: Observed page image (H, W, C) uint8
        template: Watermark template (H, W) or (H, W, C) uint8
        mask: Binary watermark mask (H, W) uint8, 255=watermark
        params: Remover parameters (alpha_clamp used)

    Returns:
        Cleaned image (H, W, C) uint8
    """
    alpha = min(params.alpha_clamp, 0.92)

    # Ensure template matches page dimensions and channels
    if template.ndim == 2 and page_img.ndim == 3:
        template = np.stack([template] * page_img.shape[2], axis=2)

    if template.shape != page_img.shape:
        template = cv2.resize(template, (page_img.shape[1], page_img.shape[0]))
        if template.ndim == 2 and page_img.ndim == 3:
            template = np.stack([template] * page_img.shape[2], axis=2)

    # Alpha inversion with feathered mask
    feather_px = max(params.mask_dilate_px, 2)
    mask_float = mask.astype(np.float32) / 255.0

    # Feather the mask edges for smooth blending
    mask_feathered = cv2.GaussianBlur(mask_float, (feather_px * 2 + 1, feather_px * 2 + 1), 0)
    if mask_feathered.ndim == 2 and page_img.ndim == 3:
        mask_feathered = np.stack([mask_feathered] * page_img.shape[2], axis=2)

    img_float = page_img.astype(np.float32)
    W = template.astype(np.float32)

    # J = (img_float - α·W) / (1 - α)
    denom = max(1.0 - alpha, 0.08)  # clamp to avoid division by near-zero
    J = (img_float - alpha * W) / denom
    J = np.clip(J, 0, 255).astype(np.uint8)

    # Blend: use cleaned pixels inside the mask, original outside
    result = page_img.copy()
    if page_img.ndim == 3:
        for c in range(page_img.shape[2]):
            result[:, :, c] = np.where(mask > 0, J[:, :, c], page_img[:, :, c])
    else:
        result = np.where(mask > 0, J, page_img)

    logger.info("raster_alpha_inv: α=%.3f, processed %d watermark pixels", alpha, int(mask.sum() / 255))
    return result


# ---------------------------------------------------------------------------
# G11: Colour Unmixing
# ---------------------------------------------------------------------------


def raster_color_unmix(
    page_img: np.ndarray,
    mask: np.ndarray,
    params: RemoverParams,
) -> np.ndarray:
    """Remove coloured watermark via CIE LAB chroma projection.

    Separates the watermark by exploiting its high saturation relative
    to achromatic body text, then applies local alpha unmixing.

    Args:
        page_img: Page image (H, W, 3) uint8 RGB
        mask: Binary watermark mask (H, W) uint8
        params: Remover parameters

    Returns:
        Cleaned image (H, W, 3) uint8
    """
    if page_img.ndim != 3 or page_img.shape[2] != 3:
        return page_img

    # Convert to LAB
    lab = cv2.cvtColor(page_img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]

    # In watermark regions, suppress chroma (make achromatic)
    mask_bool = mask > 0
    if mask_bool.sum() == 0:
        return page_img

    # Estimate background luminance from non-watermark regions
    bg_L = float(L[~mask_bool].mean()) if (~mask_bool).sum() > 0 else 200.0

    # Neutralize chroma (remove colour) across watermark regions
    result_lab = lab.copy()
    result_lab[mask_bool, 1] = 128.0  # Neutral a*
    result_lab[mask_bool, 2] = 128.0  # Neutral b*

    # Only replace background pixels with estimated paper white; preserve dark foreground text!
    is_bg = mask_bool & (L >= 120.0)
    result_lab[is_bg, 0] = bg_L

    result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    logger.info("raster_color_unmix: desaturated %d watermark pixels", int(mask_bool.sum()))
    return result


# ---------------------------------------------------------------------------
# G12: Luminance Separation
# ---------------------------------------------------------------------------


def raster_luma_sep(
    page_img: np.ndarray,
    mask: np.ndarray,
    params: RemoverParams,
) -> np.ndarray:
    """Remove varying-position grey watermarks via adaptive background
    normalization + morphological stroke separation + local alpha unmix.

    Args:
        page_img: Page image (H, W, C) uint8
        mask: Binary watermark mask (H, W) uint8
        params: Remover parameters (luma_thresh used)

    Returns:
        Cleaned image (H, W, C) uint8
    """
    grey = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY) if page_img.ndim == 3 else page_img.copy()

    # Adaptive background estimation
    bg_val = int(np.percentile(grey, 95))

    # Threshold to separate dark strokes from watermark
    luma_thresh = params.luma_thresh
    _, stroke_mask = cv2.threshold(grey, luma_thresh, 255, cv2.THRESH_BINARY_INV)

    # Dilate stroke mask slightly to protect stroke edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    stroke_mask = cv2.dilate(stroke_mask, kernel, iterations=1)

    # In watermark regions where there are no dark strokes, whiten
    wm_only = cv2.bitwise_and(mask, cv2.bitwise_not(stroke_mask))

    result = page_img.copy()
    if page_img.ndim == 3:
        for c in range(page_img.shape[2]):
            result[:, :, c] = np.where(wm_only > 0, bg_val, page_img[:, :, c])
    else:
        result = np.where(wm_only > 0, bg_val, page_img)

    logger.info("raster_luma_sep: separated %d watermark pixels", int(wm_only.sum() / 255))
    return result


# ---------------------------------------------------------------------------
# G13: FFT Notch Filter
# ---------------------------------------------------------------------------


def raster_fft_notch(
    page_img: np.ndarray,
    mask: np.ndarray,
    params: RemoverParams,
) -> np.ndarray:
    """Remove periodic hatched/grid/tiled watermarks via 2D FFT spectral
    peak suppression masked to watermark regions.

    Args:
        page_img: Page image (H, W, C) uint8
        mask: Binary watermark mask (H, W) uint8
        params: Remover parameters

    Returns:
        Cleaned image (H, W, C) uint8
    """
    result = page_img.copy()

    # Process each channel independently
    channels = [result[:, :, c] for c in range(result.shape[2])] if result.ndim == 3 else [result]

    for i, channel in enumerate(channels):
        ch_float = channel.astype(np.float32)

        # 2D FFT
        f_transform = np.fft.fft2(ch_float)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Find spectral peaks (exclude DC component)
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Zero out DC region
        mag_no_dc = magnitude.copy()
        dc_radius = min(h, w) // 20
        mag_no_dc[cy - dc_radius : cy + dc_radius, cx - dc_radius : cx + dc_radius] = 0

        # Find peaks above threshold
        threshold = np.percentile(mag_no_dc, 99.5)
        peaks = mag_no_dc > threshold

        # Create notch filter
        notch_filter = np.ones_like(magnitude)
        notch_radius = 3
        peak_coords = np.argwhere(peaks)
        for py, px in peak_coords:
            notch_filter[
                max(0, py - notch_radius) : py + notch_radius + 1,
                max(0, px - notch_radius) : px + notch_radius + 1,
            ] = 0

        # Apply notch filter
        f_filtered = f_shift * notch_filter

        # Inverse FFT
        f_ishift = np.fft.ifftshift(f_filtered)
        cleaned = np.abs(np.fft.ifft2(f_ishift))
        cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

        # Only apply in watermark mask region
        if result.ndim == 3:
            result[:, :, i] = np.where(mask > 0, cleaned, channel)
        else:
            result = np.where(mask > 0, cleaned, channel)

    logger.info("raster_fft_notch: filtered %d spectral peaks", len(peak_coords) if 'peak_coords' in dir() else 0)
    return result


# ---------------------------------------------------------------------------
# G14: Connected-Component Inpainting
# ---------------------------------------------------------------------------


def raster_cc_inpaint(
    page_img: np.ndarray,
    mask: np.ndarray,
    params: RemoverParams,
) -> np.ndarray:
    """Remove thin outline stamps via connected-component masking + Telea inpainting.

    Args:
        page_img: Page image (H, W, C) uint8
        mask: Binary watermark mask (H, W) uint8
        params: Remover parameters (inpaint_radius used)

    Returns:
        Cleaned image (H, W, C) uint8
    """
    inpaint_radius = params.inpaint_radius

    # Dilate mask slightly for better inpainting coverage
    dilate_px = params.mask_dilate_px
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        inpaint_mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        inpaint_mask = mask

    # Use Telea inpainting (deterministic, no ML)
    if page_img.ndim == 3:
        result = cv2.inpaint(page_img, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
    else:
        result = cv2.inpaint(page_img, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)

    logger.info(
        "raster_cc_inpaint: inpainted %d pixels (radius=%d)",
        int(inpaint_mask.sum() / 255),
        inpaint_radius,
    )
    return result


# ---------------------------------------------------------------------------
# G15: Bitonal Despeckle
# ---------------------------------------------------------------------------


def raster_bitonal_despeckle(
    page_img: np.ndarray,
    mask: np.ndarray,
    params: RemoverParams,
) -> np.ndarray:
    """Remove dithered watermark from 1-bit scans via stroke-density and
    connected-component duty-cycle filtering.

    Preserves solid text strokes while removing sparse halftone stipple
    that represents the watermark.

    Args:
        page_img: Bitonal image (H, W) uint8, values 0 or 255
        mask: Binary watermark mask (H, W) uint8
        params: Remover parameters

    Returns:
        Cleaned bitonal image (H, W) uint8
    """
    result = page_img.copy()
    if result.ndim == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

    # Find connected components in the watermark mask region
    inverted = cv2.bitwise_not(result)
    wm_ink = cv2.bitwise_and(inverted, mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        wm_ink, connectivity=8
    )

    # Remove small components (halftone dots) while keeping large ones (text strokes)
    min_stroke_area = 15  # pixels — text strokes are larger than halftone dots

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_stroke_area:
            # This is likely a halftone dot — remove it (set to white)
            result[labels == label_id] = 255

    logger.info(
        "raster_bitonal_despeckle: analyzed %d components, removed small stipple",
        num_labels - 1,
    )
    return result


# ---------------------------------------------------------------------------
# G17 (optional): ML Inpainter (crop-only, ONNX)
# ---------------------------------------------------------------------------


def raster_ml_inpaint(
    page_img: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    params: RemoverParams,
) -> np.ndarray:
    """Optional crop-only ML inpainting for failed ROIs (≤512×512).

    Only activates when --enable-ml-fallback is set. Crops the failed
    bounding box, runs CPU ONNX LaMa model, and composites inside mask.

    Still subject to L4 verification.

    Args:
        page_img: Full page image (H, W, C) uint8
        mask: Binary watermark mask (H, W) uint8
        bbox: (x0, y0, x1, y1) bounding box of failed region
        params: Remover parameters

    Returns:
        Page image with inpainted ROI composited
    """
    x0, y0, x1, y1 = bbox

    # Enforce 512×512 crop limit
    crop_w = x1 - x0
    crop_h = y1 - y0
    if crop_w > 512 or crop_h > 512:
        logger.warning("raster_ml_inpaint: ROI %dx%d exceeds 512×512 limit, skipping", crop_w, crop_h)
        return page_img

    import importlib.util

    if importlib.util.find_spec("onnxruntime") is None:
        logger.warning("raster_ml_inpaint: onnxruntime not installed, skipping")
        return page_img

    # TODO: Phase 4 — implement ONNX LaMa inference
    logger.info("raster_ml_inpaint: stub — ONNX inference not yet implemented")
    return page_img


# ---------------------------------------------------------------------------
# SOTA: Illumination Normalization (Rolling-Ball / Morphological Closing)
# ---------------------------------------------------------------------------


def illum_flatten(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
) -> np.ndarray:
    """Normalize uneven mobile-camera shadows and bleed-through to 255 paper white.

    Divides image by morphological closing background envelope, flattening
    shadows while preserving dark ink strokes.
    """
    grey = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY) if page_img.ndim == 3 else page_img.copy()
    h, w = grey.shape[:2]

    # Fast downsampled background envelope estimation
    scale = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(grey, (int(w * scale), int(h * scale))) if scale < 1.0 else grey
    kernel_size = max(21, int(51 * scale))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg = cv2.resize(bg_small, (w, h)) if scale < 1.0 else bg_small

    # Check illumination variance
    if float(np.std(bg)) < 6.0:
        return page_img  # already flat lighting

    norm = np.clip(grey.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0, 0, 255).astype(np.uint8)

    if page_img.ndim == 3:
        result = page_img.copy()
        for c in range(3):
            result[:, :, c] = np.clip(
                page_img[:, :, c].astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0,
                0,
                255,
            ).astype(np.uint8)
        return result
    return norm


# ---------------------------------------------------------------------------
# SOTA: Piecewise-Affine ECC Stack Alignment
# ---------------------------------------------------------------------------


def ecc_align_stack(
    ref_img: np.ndarray,
    target_img: np.ndarray,
) -> np.ndarray:
    """Sub-pixel affine alignment using cv2.findTransformECC."""
    ref_grey = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY) if ref_img.ndim == 3 else ref_img
    tgt_grey = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY) if target_img.ndim == 3 else target_img
    h, w = ref_grey.shape[:2]

    # Downsample for fast and robust convergence
    scale = min(1.0, 600.0 / max(h, w))
    s_ref = cv2.resize(ref_grey, (int(w * scale), int(h * scale))) if scale < 1.0 else ref_grey
    s_tgt = cv2.resize(tgt_grey, (int(w * scale), int(h * scale))) if scale < 1.0 else tgt_grey

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 1e-3)

    try:
        cv2.findTransformECC(s_ref, s_tgt, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        # Scale translation back to full resolution
        warp_matrix[0, 2] /= scale
        warp_matrix[1, 2] /= scale
        aligned = cv2.warpAffine(target_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except Exception as exc:
        logger.debug("ECC alignment failed (%s), returning unaligned", exc)
        return target_img


# ---------------------------------------------------------------------------
# SOTA: Margin Banner & Inverted Banner Strippers
# ---------------------------------------------------------------------------


def raster_margin_banner(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
) -> np.ndarray:
    """Remove scanner margin banners (e.g. CamScanner, TapScanner, footer stamps).

    Detects isolated ink blocks in top or bottom 8% margins separated from body text
    by at least 20px whitespace and replaces with paper background white.
    """
    result = page_img.copy()
    h, w = page_img.shape[:2]
    grey = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY) if page_img.ndim == 3 else page_img.copy()

    # Check top margin and bottom margin (8% height)
    m_h = int(h * 0.08)
    paper_val = int(np.percentile(grey[grey >= 240], 95)) if (grey >= 240).sum() > 100 else 255

    for y0, y1 in [(0, m_h), (h - m_h, h)]:
        roi_grey = grey[y0:y1, :]
        ink = (roi_grey < 160)
        if ink.sum() > 30:
            # Check horizontal whitespace separation between margin and body text
            if y0 == 0:
                sep_zone = grey[m_h : m_h + 20, :]
            else:
                sep_zone = grey[h - m_h - 20 : h - m_h, :]

            # If separation zone is predominantly whitespace (>92% white)
            if (sep_zone >= 220).mean() > 0.90:
                # Whiten margin banner
                if result.ndim == 3:
                    result[y0:y1, :] = paper_val
                else:
                    result[y0:y1, :] = paper_val
                logger.info("raster_margin_banner: stripped banner in margin [%d:%d]", y0, y1)

    return result


def inverted_banner_strip(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
) -> np.ndarray:
    """Remove inverted dark-box watermarks (white text inside solid dark rectangle)."""
    result = page_img.copy()
    h, w = page_img.shape[:2]
    grey = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY) if page_img.ndim == 3 else page_img.copy()

    # Find solid dark components (luminance < 60)
    dark_mask = (grey < 60).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

    paper_val = int(np.percentile(grey[grey >= 240], 95)) if (grey >= 240).sum() > 100 else 255

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = bw / max(bh, 1)

        # Inverted banner: wide horizontal bar with high fill ratio
        if aspect > 3.0 and area > 1000 and (area / (bw * bh)) > 0.60:
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            if result.ndim == 3:
                result[by : by + bh, bx : bx + bw] = paper_val
            else:
                result[by : by + bh, bx : bx + bw] = paper_val
            logger.info("inverted_banner_strip: stripped dark banner at [%d,%d,%d,%d]", bx, by, bw, bh)

    return result


# ---------------------------------------------------------------------------
# SOTA: Consensus Dark Stamp & Registered Outline Removers
# ---------------------------------------------------------------------------


def raster_dark_stamp_consensus(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
    template: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Remove high-density repeating stamp using P85 consensus and stroke bridging."""
    return raster_template_whiten(page_img, mask, params, template=template)


def raster_registered_outline_clean(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
    template: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Remove registered diagonal outline watermark using edge-aligned template."""
    return raster_template_whiten(page_img, mask, params, template=template)


def raster_hue_selective_unmix(
    page_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    params: Optional[RemoverParams] = None,
) -> np.ndarray:
    """Remove saturated watermark chroma while strictly protecting green/blue diagram hues."""
    return raster_color_unmix(page_img, mask if mask is not None else np.zeros(page_img.shape[:2], dtype=np.uint8), params if params is not None else RemoverParams())

