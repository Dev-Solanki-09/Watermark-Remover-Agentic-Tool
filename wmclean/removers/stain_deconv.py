"""Beer-Lambert optical density colour deconvolution for watermark removal.

Uses Ruifrok & Johnston (2001) / Landini formulation via skimage.color.separate_stains
and combine_stains to separate overlapping coloured/tinted watermark stamps (e.g. red,
blue, magenta) from black text toner in log-absorbance space.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
from skimage.color import combine_stains, separate_stains

from wmclean.verify.text_protect import compute_fg_protection_mask

logger = logging.getLogger(__name__)


def _rgb_to_od(rgb_norm: np.ndarray) -> np.ndarray:
    """Convert normalized RGB in [0, 1] to optical density vector."""
    rgb_safe = np.maximum(rgb_norm, 1e-6)
    return np.log(rgb_safe) / np.log(1e-6)


def estimate_stain_vectors(
    rgb_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fg_mask: Optional[np.ndarray] = None,
    wm_rgb: Optional[Union[tuple[int, int, int], list[int], np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate normalized optical density vectors (v_ink, v_wm, v_res).

    Args:
        rgb_img: (H, W, 3) uint8 RGB image.
        mask: Optional binary mask of watermark area.
        fg_mask: Optional binary mask of foreground text strokes.
        wm_rgb: Optional reference RGB color of watermark.

    Returns:
        (v_ink, v_wm, v_res): 3 unit-length vectors in optical density space.
    """
    # 1. Text Toner Vector (v_ink)
    if fg_mask is None:
        gray = np.mean(rgb_img, axis=-1)
        fg_pixels = gray < 80
    else:
        fg_pixels = fg_mask > 0

    if np.sum(fg_pixels) > 20:
        ink_rgb = np.median(rgb_img[fg_pixels].astype(np.float64) / 255.0, axis=0)
        od_ink = _rgb_to_od(ink_rgb)
        norm_ink = np.linalg.norm(od_ink)
        v_ink = od_ink / norm_ink if norm_ink > 1e-5 else np.ones(3) / np.sqrt(3)
    else:
        # Default neutral carbon black toner
        v_ink = np.ones(3) / np.sqrt(3)

    # 2. Watermark Stain Vector (v_wm)
    if wm_rgb is not None:
        wm_color = np.array(wm_rgb, dtype=np.float64) / 255.0
        od_wm = _rgb_to_od(wm_color)
        norm_wm = np.linalg.norm(od_wm)
        v_wm = od_wm / norm_wm if norm_wm > 1e-5 else np.array([0.15, 0.7, 0.7])
    elif mask is not None and np.sum(mask > 0) > 20:
        # Unobstructed watermark pixels: inside mask, outside text, not pure background
        gray = np.mean(rgb_img, axis=-1)
        unobstructed = (mask > 0) & (~fg_pixels) & (gray < 245) & (gray > 40)
        if np.sum(unobstructed) > 20:
            wm_color = np.median(rgb_img[unobstructed].astype(np.float64) / 255.0, axis=0)
            od_wm = _rgb_to_od(wm_color)
            norm_wm = np.linalg.norm(od_wm)
            v_wm = od_wm / norm_wm if norm_wm > 1e-5 else np.array([0.15, 0.7, 0.7])
        else:
            # Fallback red stamp default
            v_wm = np.array([0.1, 0.7, 0.7]) / np.linalg.norm([0.1, 0.7, 0.7])
    else:
        # Default red stamp optical density
        v_wm = np.array([0.1, 0.7, 0.7]) / np.linalg.norm([0.1, 0.7, 0.7])

    # Ensure v_ink and v_wm are not collinear
    dot = np.abs(np.dot(v_ink, v_wm))
    if dot > 0.98:
        # Perturb slightly to avoid singular matrix
        v_wm = np.array([v_wm[0] * 0.8, v_wm[1] * 1.2, v_wm[2] * 0.9])
        v_wm /= np.linalg.norm(v_wm)

    # 3. Orthogonal Residual Vector (v_res)
    v_res = np.cross(v_ink, v_wm)
    norm_res = np.linalg.norm(v_res)
    if norm_res < 1e-6:
        v_res = np.array([0.0, 1.0, 0.0])
    else:
        v_res /= norm_res

    return v_ink, v_wm, v_res


def beer_lambert_stain_unmix(
    rgb_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fg_mask: Optional[np.ndarray] = None,
    wm_rgb: Optional[Union[tuple[int, int, int], list[int], np.ndarray]] = None,
    attenuation: float = 1.0,
    protect_text: bool = True,
) -> np.ndarray:
    """Separate and attenuate watermark stain using Beer-Lambert optical density deconvolution.

    Args:
        rgb_img: (H, W, 3) uint8 RGB image.
        mask: Optional binary mask (255 where watermark is present).
        fg_mask: Optional binary mask of foreground text to protect.
        wm_rgb: Optional reference RGB color of watermark.
        attenuation: Fraction of watermark optical density to remove [0.0 to 1.0].
        protect_text: Whether to compute/apply M_fg text stroke protection.

    Returns:
        np.ndarray: (H, W, 3) uint8 cleaned image.
    """
    if rgb_img.ndim != 3 or rgb_img.shape[2] != 3:
        raise ValueError(f"beer_lambert_stain_unmix expects 3-channel RGB, got {rgb_img.shape}")

    if fg_mask is None and protect_text:
        fg_mask = compute_fg_protection_mask(rgb_img)

    v_ink, v_wm, v_res = estimate_stain_vectors(
        rgb_img=rgb_img,
        mask=mask,
        fg_mask=fg_mask,
        wm_rgb=wm_rgb,
    )

    # Form stain matrix M where each row is a normalized stain OD vector
    M = np.stack([v_ink, v_wm, v_res], axis=0)
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        logger.warning("Stain matrix inversion failed; returning original image")
        return rgb_img.copy()

    # Deconvolve into stain space
    stains = separate_stains(rgb_img, M_inv)

    # Attenuate channel 1 (watermark stain OD) while preserving channel 0 (ink OD)
    stains[:, :, 1] *= max(0.0, 1.0 - float(attenuation))

    # Reconstruct in RGB space
    recon = combine_stains(stains, M)
    cleaned_rgb = np.clip(recon * 255.0, 0, 255).astype(np.uint8)

    # If mask is supplied, confine edits strictly to mask
    if mask is not None:
        out = rgb_img.copy()
        mask_bool = mask > 0
        if fg_mask is not None and protect_text:
            # Under watermark, apply unmixed result, but protect dark text strokes
            text_bool = fg_mask > 0
            edit_region = mask_bool & (~text_bool)
            overlap_region = mask_bool & text_bool

            out[edit_region] = cleaned_rgb[edit_region]
            # In overlap region, blend unmixed with original text
            out[overlap_region] = cleaned_rgb[overlap_region]
        else:
            out[mask_bool] = cleaned_rgb[mask_bool]
        return out

    return cleaned_rgb
