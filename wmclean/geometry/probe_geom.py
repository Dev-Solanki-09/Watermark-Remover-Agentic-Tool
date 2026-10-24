"""Geometric distortion profiler and forensic probe.

Measures angular tilt (skew), spine curvature (gutter dip), and local paper
waviness to classify page geometry into one of four mutually exclusive modalities:
- GEOM_FLAT: No deformation (0ms NO-OP)
- GEOM_SKEW_RIGID: Global rotational tilt
- GEOM_GUTTER_CURVE: Monotonic vertical dip near spine binding
- GEOM_WAVY_BUMP: Non-monotonic multi-frequency wave / circular bulge
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from wmclean.deps import get_deskew, has_deskew
from wmclean.schemas import GeometricModality, GeometryProbeRecord

logger = logging.getLogger(__name__)


def probe_page_geometry(img: np.ndarray) -> GeometryProbeRecord:
    """Analyze page geometry to determine distortion modality and parameters.

    Args:
        img: Greyscale or RGB image array (H, W) or (H, W, C).

    Returns:
        GeometryProbeRecord with classified modality and metrics.
    """
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = img if img.ndim == 2 else img[:, :, 0]

    h, w = gray.shape[:2]
    if h < 20 or w < 20:
        return GeometryProbeRecord(modality=GeometricModality.GEOM_FLAT)

    # 1. Measure Angular Skew via deskew library or Hough Lines
    skew_angle_deg = 0.0
    if has_deskew():
        try:
            deskew_mod = get_deskew()
            if deskew_mod is not None:
                detected_angle = deskew_mod.determine_skew(
                    gray,
                    min_angle=-15.0,
                    max_angle=15.0,
                )
                if detected_angle is not None and abs(float(detected_angle)) < 14.8:
                    skew_angle_deg = float(detected_angle)
        except Exception as exc:
            logger.debug("deskew library failed: %s; falling back to line probe", exc)


    # 2. Extract Text Baselines via Horizontal Morphology
    # Invert so text is white on black background
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        10,
    )

    # Horizontal structuring element to bridge glyphs along text lines
    kernel_len = max(20, int(w * 0.08))
    line_mask = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 5)))

    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valid_lines = []
    min_line_w = max(40, int(w * 0.15))

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw >= min_line_w and ch <= max(45, int(h * 0.15)):
            pts = cnt.reshape(-1, 2)
            # Sort by X
            pts = pts[np.argsort(pts[:, 0])]
            valid_lines.append(pts)

    spine_dips: list[float] = []
    waviness_stds: list[float] = []
    line_skews: list[float] = []

    for pts in valid_lines:
        raw_x = pts[:, 0]
        raw_y = pts[:, 1]
        unique_xs = np.unique(raw_x)
        if len(unique_xs) < 10:
            continue
        xs = unique_xs.astype(float)
        ys = np.array([float(np.mean(raw_y[raw_x == x])) for x in unique_xs])

        # Linear trend fit
        p = np.polyfit(xs, ys, 1)
        slope = p[0]
        deg = float(np.degrees(np.arctan(slope)))
        line_skews.append(deg)

        fitted_ys = np.polyval(p, xs)
        residuals = ys - fitted_ys


        # Check spine dip: compare detrended margin residuals vs center residuals
        left_margin_mask = xs < (w * 0.18)
        right_margin_mask = xs > (w * 0.82)
        center_mask = (xs >= w * 0.35) & (xs <= w * 0.65)

        if np.any(left_margin_mask) and np.any(center_mask):
            left_dip = abs(np.mean(residuals[left_margin_mask]) - np.mean(residuals[center_mask]))
            spine_dips.append(float(left_dip))
        elif np.any(right_margin_mask) and np.any(center_mask):
            right_dip = abs(np.mean(residuals[right_margin_mask]) - np.mean(residuals[center_mask]))
            spine_dips.append(float(right_dip))

        # Check waviness std
        if len(residuals) > 5:
            waviness_stds.append(float(np.std(residuals)))

    # Cross-check deskew angle against detected text baselines
    if line_skews:
        median_skew = float(np.median(line_skews))
        # If Hough angle disagrees substantially with physical text lines (>1.5 deg difference),
        # or if detected text baselines are flat (<0.8 deg), prefer the physical line baseline
        if abs(skew_angle_deg - median_skew) > 1.5 or abs(median_skew) < 0.8:
            skew_angle_deg = median_skew
        elif abs(skew_angle_deg) < 0.1:
            skew_angle_deg = median_skew

    max_spine_dip = float(np.max(spine_dips)) if spine_dips else 0.0
    mean_waviness = float(np.mean(waviness_stds)) if waviness_stds else 0.0

    # 3. Decision Matrix
    # Gutter curve takes priority when spine dip is pronounced
    if max_spine_dip >= 3.0:
        modality = GeometricModality.GEOM_GUTTER_CURVE
    elif mean_waviness >= 2.5:
        modality = GeometricModality.GEOM_WAVY_BUMP
    elif abs(skew_angle_deg) >= 0.8:
        modality = GeometricModality.GEOM_SKEW_RIGID
    else:
        modality = GeometricModality.GEOM_FLAT

    return GeometryProbeRecord(
        modality=modality,
        skew_angle_deg=round(skew_angle_deg, 2),
        spine_dip_px=round(max_spine_dip, 2),
        waviness_std_px=round(mean_waviness, 2),
        confidence=0.95 if modality != GeometricModality.GEOM_FLAT else 1.0,
    )
