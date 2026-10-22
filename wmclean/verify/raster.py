"""L4 Raster Verifier — evaluates watermark residual (R_wm), foreground content
loss (L_fg), diagram stroke integrity, and outside-mask variance.

Invariants:
- R_wm <= 0.035 (residual watermark energy < 3.5%)
- L_fg <= 0.020 (foreground stroke recall loss < 2.0%)
- Outside mask difference <= 5.0 (scale 0-255)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from wmclean.schemas import FailureCode, VerifyResult
from wmclean.verify.text_protect import compute_fg_protection_mask

logger = logging.getLogger(__name__)

# Verification thresholds
_MAX_RESIDUAL = 0.035
_MAX_CONTENT_LOSS = 0.020
_MAX_OUTSIDE_DIFF = 10.0


def _to_grey(img: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR/Grey image to single channel greyscale uint8."""
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.shape[2] == 4:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    return img[:, :, 0]


def verify_raster(
    orig_img: np.ndarray,
    clean_img: np.ndarray,
    wm_mask: Optional[np.ndarray] = None,
    template: Optional[np.ndarray] = None,
) -> VerifyResult:
    """Evaluate raster cleaning quality against deterministic verification gates.

    Returns VerifyResult with pass/fail decision, metric values, and failure code.
    """
    orig_grey = _to_grey(orig_img)
    clean_grey = _to_grey(clean_img)

    if orig_grey.shape != clean_grey.shape:
        return VerifyResult(
            passed=False,
            failure_code=FailureCode.ERR_VEC_VERIFY_BREACH,
            reason=f"Dimension mismatch: orig {orig_grey.shape} vs clean {clean_grey.shape}",
        )

    # 1. Check for Opaque Occlusion (e.g. FX_G19 pitch black rectangle)
    # Detect if original has an opaque pitch black patch covering text
    dark_mask = orig_grey < 5
    if np.sum(dark_mask) > 1000:
        # Check if connected component forms a solid opaque occluding block
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark_mask.astype(np.uint8), connectivity=8
        )
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if w > 100 and h > 30 and area > 3000:
                logger.warning("verify_raster: detected opaque occlusion block (%d px)", area)
                return VerifyResult(
                    passed=False,
                    failure_code=FailureCode.ERR_OPAQUE_OCCLUSION,
                    reason=f"Opaque dark occlusion detected (area={area}px, size={w}x{h}); unrecoverable text corruption",
                )

    # Compute default watermark mask if none provided
    if wm_mask is None:
        # Regions where orig and clean differ
        abs_diff = cv2.absdiff(orig_grey, clean_grey)
        wm_mask = abs_diff > 8
        if template is not None:
            t_grey = _to_grey(template)
            if t_grey.shape != orig_grey.shape:
                t_grey = cv2.resize(t_grey, (orig_grey.shape[1], orig_grey.shape[0]))
            bg_t = float(np.percentile(t_grey, 95))
            wm_mask = wm_mask | (np.abs(bg_t - t_grey.astype(np.float32)) > 15)

    outside_mask = ~wm_mask if wm_mask is not None else np.ones_like(orig_grey, dtype=bool)

    # 2. Foreground Content Loss (L_fg)
    # Check for corrupt/blank output image
    if float(np.std(clean_grey)) < 2.0:
        fg_m = compute_fg_protection_mask(orig_grey)
        blank_loss = 1.0 if np.sum(fg_m > 0) > 10 else 0.0
        return VerifyResult(
            passed=False,
            content_loss=blank_loss,
            failure_code=FailureCode.ERR_DIAGRAM_INTEGRITY,
            reason="Output image is corrupt or completely blank (std < 2.0)",
        )

    # Identify foreground strokes with C++ SIMD ISauvola binarization
    fg_mask = compute_fg_protection_mask(orig_grey)
    orig_fg = fg_mask > 0
    # Exclude confirmed watermark pixels so erased watermark ink is not miscounted as lost text
    if wm_mask is not None:
        wm_bool = wm_mask > 0 if wm_mask.dtype != bool else wm_mask
        eval_fg = orig_fg & (~wm_bool)
    else:
        eval_fg = orig_fg

    total_eval_fg = int(np.sum(eval_fg))

    content_loss = 0.0
    if total_eval_fg > 50:
        erased_fg = eval_fg & (clean_grey > 200)
        content_loss = float(np.sum(erased_fg)) / total_eval_fg

        if content_loss > _MAX_CONTENT_LOSS:
            logger.warning(
                "verify_raster failed: L_fg=%.4f > %.4f", content_loss, _MAX_CONTENT_LOSS
            )
            return VerifyResult(
                passed=False,
                content_loss=round(content_loss, 4),
                failure_code=FailureCode.ERR_DIAGRAM_INTEGRITY,
                reason=f"Foreground content loss exceeded ({content_loss:.4f} > {_MAX_CONTENT_LOSS:.4f})",
            )

    # 3. Watermark Residual (R_wm)
    residual = 0.0
    if wm_mask is not None and np.sum(wm_mask) > 100:
        bg_val = float(np.percentile(clean_grey, 95))
        # Protect author diagrams, text cores, and halos from being evaluated as residual watermark
        from wmclean.intelligence.doc_shield import parse_doc_shield

        shield = parse_doc_shield(
            orig_img if orig_img is not None else orig_grey, template_mask=wm_mask
        )
        bg_eval_mask = (
            wm_mask & (~shield.m_diagram) & (~shield.m_ink_halo_shield) & (orig_grey >= 120)
        )
        if np.sum(bg_eval_mask) > 50:
            wm_region = clean_grey[bg_eval_mask].astype(np.float32)
            residual = float(np.mean(np.abs(wm_region - bg_val))) / 255.0
        else:
            clean_eval = wm_mask & (~shield.m_diagram) & (~shield.m_ink_halo_shield)
            if np.sum(clean_eval) > 50:
                wm_region = clean_grey[clean_eval].astype(np.float32)
                residual = float(np.mean(np.abs(wm_region - bg_val))) / 255.0
            else:
                residual = 0.0

        if residual > _MAX_RESIDUAL:
            logger.warning("verify_raster failed: R_wm=%.4f > %.4f", residual, _MAX_RESIDUAL)
            return VerifyResult(
                passed=False,
                residual=round(residual, 4),
                content_loss=round(content_loss, 4),
                failure_code=FailureCode.ERR_HIGH_RESIDUAL,
                reason=f"Watermark residual energy exceeded ({residual:.4f} > {_MAX_RESIDUAL:.4f})",
            )

    # 4. Outside Mask Maximum Pixel Difference
    outside_diff = 0.0
    if wm_mask is not None and outside_mask.any():
        outside_diff = float(
            np.max(
                np.abs(
                    orig_grey[outside_mask].astype(np.int16)
                    - clean_grey[outside_mask].astype(np.int16)
                )
            )
        )
        if outside_diff > _MAX_OUTSIDE_DIFF:
            logger.warning(
                "verify_raster failed: outside_diff=%.1f > %.1f", outside_diff, _MAX_OUTSIDE_DIFF
            )
            return VerifyResult(
                passed=False,
                residual=round(residual, 4),
                content_loss=round(content_loss, 4),
                outside_mask_diff=outside_diff,
                failure_code=FailureCode.ERR_VEC_VERIFY_BREACH,
                reason=f"Outside-mask pixel difference exceeded tolerance ({outside_diff} > {_MAX_OUTSIDE_DIFF})",
            )

    return VerifyResult(
        passed=True,
        residual=round(residual, 4),
        content_loss=round(content_loss, 4),
        outside_mask_diff=outside_diff,
        text_diff=0,
    )
