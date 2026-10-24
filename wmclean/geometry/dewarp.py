"""Core SOTA geometric dewarping actuator.

Provides dense inverse-coordinate remapping (cv2.remap) and zero-crop expanded canvas
transformations for:
1. Rigid skew rotation with bounding envelope expansion
2. Cylindrical book-gutter spine unrolling (page-dewarp / doc-dewarping model)
3. 2D wavy mesh / local bump disparity field flattening
4. Negative-presence veto bypass for flat pages
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from wmclean.geometry.probe_geom import probe_page_geometry
from wmclean.schemas import DewarpResult, GeometricModality, GeometryProbeRecord

logger = logging.getLogger(__name__)


def rotate_expanded_canvas(
    img: np.ndarray,
    angle_deg: float,
    bg_color: int = 255,
) -> np.ndarray:
    """Rotate an image with zero corner cropping using dynamic canvas expansion.

    Calculates:
        W' = ceil(|W*cos(theta)| + |H*sin(theta)|)
        H' = ceil(|H*cos(theta)| + |W*sin(theta)|)

    Resamples with Lanczos interpolation (cv2.INTER_LANCZOS4) and pads new outer
    canvas borders with clean paper white.

    Args:
        img: Greyscale or RGB image array (H, W) or (H, W, C).
        angle_deg: Rotation angle in degrees (clockwise positive or deskew angle).
        bg_color: Border padding color (default 255 for paper white).

    Returns:
        Expanded rotated image array (H', W') or (H', W', C).
    """
    if abs(angle_deg) < 0.05:
        return img

    h, w = img.shape[:2]
    # For deskew correction, we rotate in reverse of detected skew angle
    rot_angle = -angle_deg

    rad = np.radians(abs(rot_angle))
    cos_a = float(np.cos(rad))
    sin_a = float(np.sin(rad))

    new_w = max(w, int(np.ceil(w * cos_a + h * sin_a)))
    new_h = max(h, int(np.ceil(h * cos_a + w * sin_a)))

    cx = w / 2.0
    cy = h / 2.0

    rot_mat = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
    # Adjust translation offset so entire rotated envelope fits in new dimensions
    rot_mat[0, 2] += (new_w / 2.0) - cx
    rot_mat[1, 2] += (new_h / 2.0) - cy

    border_val = (bg_color, bg_color, bg_color) if img.ndim == 3 else bg_color

    rotated = cv2.warpAffine(
        img,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    return rotated


def dewarp_cylindrical_gutter(
    img: np.ndarray,
    spine_dip_px: float,
    side: str = "left",
    bg_color: int = 255,
) -> np.ndarray:
    """Rectify book-gutter spine curvature via cylindrical inverse unrolling.

    Models the page near the spine binding as a generalized cylindrical surface.
    Constructs a destination-to-source dense coordinate map (map_x, map_y) and expands
    the canvas envelope to guarantee zero border clipping.

    Args:
        img: Input image array (H, W) or (H, W, C).
        spine_dip_px: Peak vertical text displacement near the spine.
        side: Binding side ('left' or 'right').
        bg_color: Outer background padding value (default 255).

    Returns:
        Rectified, unrolled image array.
    """
    h, w = img.shape[:2]
    if abs(spine_dip_px) < 1.0:
        return img

    # Spine decay length scale (roughly 15% of page width)
    tau = max(20.0, float(w) * 0.15)

    # Destination coordinate grid
    grid_y, grid_x = np.indices((h, w), dtype=np.float32)

    # Compute source coordinates based on inverse displacement
    if side == "left":
        dist_from_spine = grid_x
    else:
        dist_from_spine = float(w - 1) - grid_x

    # Monotonic exponential spine curve profile
    curve_offset = float(spine_dip_px) * np.exp(-dist_from_spine / tau)

    # Invert the curve: source Y coordinate had text shifted down/up
    src_y = grid_y + curve_offset

    # Calculate expansion needed to encompass full envelope
    min_y = float(np.min(src_y))
    max_y = float(np.max(src_y))

    pad_top = int(max(0.0, np.ceil(-min_y)))
    pad_bottom = int(max(0.0, np.ceil(max_y - (h - 1))))
    new_h = h + pad_top + pad_bottom
    new_w = w

    # Shift mapping into expanded canvas
    dest_y, dest_x = np.indices((new_h, new_w), dtype=np.float32)
    if side == "left":
        d_spine = dest_x
    else:
        d_spine = float(new_w - 1) - dest_x

    map_y = (dest_y - pad_top) + float(spine_dip_px) * np.exp(-d_spine / tau)
    map_x = dest_x

    border_val = (bg_color, bg_color, bg_color) if img.ndim == 3 else bg_color

    rectified = cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    return rectified


def dewarp_2d_wavy_mesh(
    img: np.ndarray,
    amplitude_px: float = 6.0,
    wavelength_px: float = 120.0,
    bg_color: int = 255,
) -> np.ndarray:
    """Rectify local paper ripples, circular bulges, and wavy distortions.

    Uses dense inverse-coordinate disparity mapping to straighten oscillating baselines.
    Expands the canvas vertically so that waves near top/bottom margins are never clipped.

    Args:
        img: Input image array (H, W) or (H, W, C).
        amplitude_px: Peak ripple amplitude in pixels.
        wavelength_px: Characteristic wavelength of ripples across horizontal axis.
        bg_color: Outer background padding value (default 255).

    Returns:
        Rectified, flattened image array.
    """
    h, w = img.shape[:2]
    if abs(amplitude_px) < 1.0:
        return img

    pad_v = int(np.ceil(abs(amplitude_px) * 1.5))
    new_h = h + 2 * pad_v
    new_w = w

    dest_y, dest_x = np.indices((new_h, new_w), dtype=np.float32)

    # Sinusoidal baseline ripple correction map
    wave = float(amplitude_px) * np.sin(2.0 * np.pi * dest_x / max(20.0, float(wavelength_px)))
    map_y = (dest_y - pad_v) + wave
    map_x = dest_x

    border_val = (bg_color, bg_color, bg_color) if img.ndim == 3 else bg_color

    rectified = cv2.remap(
        img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    return rectified


def dewarp_page(
    img: np.ndarray,
    probe: Optional[GeometryProbeRecord] = None,
) -> Tuple[np.ndarray, DewarpResult]:
    """Execute intelligent, forensic-gated dewarping on a single page image.

    - Flat pages bypass all processing with 0ms overhead and zero byte alteration.
    - Warped pages are rectified with expanded canvases and Lanczos interpolation.

    Args:
        img: Greyscale or RGB image array (H, W) or (H, W, C).
        probe: Optional precomputed GeometryProbeRecord.

    Returns:
        tuple[rectified_image, DewarpResult]
    """
    t0 = time.monotonic()
    h, w = img.shape[:2]

    if probe is None:
        probe = probe_page_geometry(img)

    # 1. Flat Page (Negative Presence Veto)
    if probe.is_flat:
        ms = int((time.monotonic() - t0) * 1000)
        return img, DewarpResult(
            modality=GeometricModality.GEOM_FLAT,
            applied=False,
            original_dims=(w, h),
            rectified_dims=(w, h),
            skew_deg=probe.skew_angle_deg,
            processing_ms=ms,
            summary="Geometric pass untouched (page is flat)",
        )

    rectified = img

    # 2. Gutter Curve Unroll
    if probe.modality == GeometricModality.GEOM_GUTTER_CURVE:
        rectified = dewarp_cylindrical_gutter(img, spine_dip_px=probe.spine_dip_px, side="left")
        # Also correct residual skew if non-trivial
        if abs(probe.skew_angle_deg) >= 0.5:
            rectified = rotate_expanded_canvas(rectified, probe.skew_angle_deg)
        summary = f"Cylindrical spine unroll applied (dip {probe.spine_dip_px:.1f}px)"

    # 3. Wavy Mesh Rectification
    elif probe.modality == GeometricModality.GEOM_WAVY_BUMP:
        rectified = dewarp_2d_wavy_mesh(img, amplitude_px=probe.waviness_std_px)
        if abs(probe.skew_angle_deg) >= 0.5:
            rectified = rotate_expanded_canvas(rectified, probe.skew_angle_deg)
        summary = f"2D wavy mesh dewarping applied (amplitude {probe.waviness_std_px:.1f}px)"

    # 4. Rigid Skew Rotation
    elif probe.modality == GeometricModality.GEOM_SKEW_RIGID:
        rectified = rotate_expanded_canvas(img, probe.skew_angle_deg)
        summary = f"Rigid zero-crop deskew applied: {probe.skew_angle_deg:.2f} deg"

    else:
        summary = "No geometric transformation applied"

    ms = int((time.monotonic() - t0) * 1000)
    out_h, out_w = rectified.shape[:2]

    return rectified, DewarpResult(
        modality=probe.modality,
        applied=True,
        original_dims=(w, h),
        rectified_dims=(out_w, out_h),
        skew_deg=probe.skew_angle_deg,
        processing_ms=ms,
        summary=summary,
    )
