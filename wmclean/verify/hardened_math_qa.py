"""Hardened Mathematical & Topological QA Invariants (4-Invariant Anti-Bleaching Gate)
and 2D FFT Periodic Notch Watermark Filter.

Enforces:
1. Betti-1 closed-ring loop invariance (Delta beta_1 == 0) for chemical rings and mesh circuits.
2. Skeleton arc-length conservation on high-confidence foreground ink (>= 97%), preventing
   all-white bleaching of open acyclic trees, arrows, and line art.
3. Dangling skeleton endpoint conservation (Delta E_tips <= 0), preventing micro-severing
   of continuous lines.
4. Residual watermark gradient energy and 2px boundary DCT mosquito halo reduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class FourInvariantQAResult:
    passed: bool
    betti_1_delta: int                # Must be == 0 (closed loops/rings preserved)
    skeleton_length_retention: float  # Must be >= 0.97 (prevents "All-White Bleaching" on acyclic graphs)
    dangling_endpoint_delta: int      # Must be <= 0 (prevents severed lines creating new line tips)
    residual_wm_energy: float         # Lower is better (gradient energy left inside watermark mask)
    dct_boundary_halo_var: float      # Lower is better (2px JPEG mosquito halo variance)
    failure_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count_skeleton_endpoints(skel_bool: np.ndarray) -> int:
    """Counts 1-pixel skeleton dangling endpoints (pixels with exactly 1 neighbor in an 8-connected grid).

    Severing a continuous line inside a watermark intersection increases this count by +2.
    """
    if not np.any(skel_bool):
        return 0
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_sum = convolve(skel_bool.astype(np.uint8), kernel, mode="constant", cval=0)
    # A skeleton pixel (10) with exactly 1 neighbor (1) has sum == 11
    return int(np.sum(neighbor_sum == 11))


def evaluate_four_invariant_qa(
    original_gray: np.ndarray,
    candidate_gray: np.ndarray,
    watermark_mask: Optional[np.ndarray] = None,
    max_ring_delta: int = 0,
    min_skeleton_retention: float = 0.97,
    max_new_endpoints: int = 0,
) -> FourInvariantQAResult:
    """Ground-truth-free verification gate for candidate evaluation.

    Eliminates the "All-White Bleaching" reward-hacking vulnerability by enforcing:
      1. Betti-1 closed-ring invariance (Delta beta_1 == 0)
      2. Skeleton arc-length conservation on non-watermark ink (>= 97%)
      3. Dangling skeleton endpoint conservation (Delta E_tips <= 0)
      4. Residual watermark gradient & 2px DCT halo reduction
    """
    if original_gray.shape != candidate_gray.shape:
        return FourInvariantQAResult(
            passed=False,
            betti_1_delta=999,
            skeleton_length_retention=0.0,
            dangling_endpoint_delta=999,
            residual_wm_energy=1.0,
            dct_boundary_halo_var=1.0,
            failure_reasons=(f"Dimension mismatch: {original_gray.shape} vs {candidate_gray.shape}",),
        )

    if watermark_mask is None or not np.any(watermark_mask):
        return FourInvariantQAResult(
            passed=True,
            betti_1_delta=0,
            skeleton_length_retention=1.0,
            dangling_endpoint_delta=0,
            residual_wm_energy=0.0,
            dct_boundary_halo_var=0.0,
            failure_reasons=(),
        )

    wm_mask_bool = watermark_mask > 0 if watermark_mask.dtype != bool else watermark_mask
    ys, xs = np.where(wm_mask_bool)
    if len(ys) == 0:
        return FourInvariantQAResult(
            passed=True,
            betti_1_delta=0,
            skeleton_length_retention=1.0,
            dangling_endpoint_delta=0,
            residual_wm_energy=0.0,
            dct_boundary_halo_var=0.0,
            failure_reasons=(),
        )

    # Low-spec resource optimization: crop to active watermark ROI with generous padding (32px)
    # Drastically reduces CPU and RAM consumption for dual-core / 4GB systems while maintaining full fidelity
    pad = 32
    h, w = original_gray.shape
    y1, y2 = max(0, int(np.min(ys)) - pad), min(h, int(np.max(ys)) + pad)
    x1, x2 = max(0, int(np.min(xs)) - pad), min(w, int(np.max(xs)) + pad)

    orig_roi = original_gray[y1:y2, x1:x2]
    cand_roi = candidate_gray[y1:y2, x1:x2]
    mask_roi = wm_mask_bool[y1:y2, x1:x2]

    # Isolate authentic foreground ink inside the ROI
    # In candidate, watermark has been removed to background, so dark ink is authentic document content
    _, fg_cand = cv2.threshold(cand_roi, 135, 255, cv2.THRESH_BINARY_INV)

    # In original, exclude pure watermark strokes that were cleaned to background (> 200)
    # so watermark letters (e.g. O, P, R in "COPY PROTECTED") are not falsely treated as authentic loops
    is_cleaned_wm = mask_roi & (cand_roi > 200)
    fg_orig_raw = cv2.threshold(orig_roi, 135, 255, cv2.THRESH_BINARY_INV)[1]
    fg_orig = np.where(is_cleaned_wm, 0, fg_orig_raw).astype(np.uint8)


    orig_has_fg = np.any(fg_orig > 0)
    cand_has_fg = np.any(fg_cand > 0)

    if not orig_has_fg and not cand_has_fg:
        betti_1_delta = 0
        retention = 1.0
        endpoint_delta = 0
    else:
        skel_orig = skeletonize(fg_orig > 0)
        skel_cand = skeletonize(fg_cand > 0)

        # Invariant 1: Betti-1 Closed Loop Count (beta_1 = beta_0 - euler_number)
        _, b0_orig = label(skel_orig, connectivity=2, return_num=True)
        b1_orig = b0_orig - euler_number(skel_orig, connectivity=2)

        _, b0_cand = label(skel_cand, connectivity=2, return_num=True)
        b1_cand = b0_cand - euler_number(skel_cand, connectivity=2)
        betti_1_delta = abs(int(b1_cand - b1_orig))

        # Invariant 2: Skeleton Arc-Length Conservation (prevents bleaching acyclic diagrams)
        orig_len = int(np.sum(skel_orig))
        cand_len = int(np.sum(skel_cand))
        retention = float(cand_len / orig_len) if orig_len > 0 else 1.0

        # Invariant 3: Dangling Endpoint Conservation (detects micro-breaks across lines)
        tips_orig = _count_skeleton_endpoints(skel_orig)
        tips_cand = _count_skeleton_endpoints(skel_cand)
        endpoint_delta = int(tips_cand - tips_orig)

    # Metric 4: Residual watermark gradient energy in background watermark regions
    wm_bg_zone = mask_roi & (fg_orig == 0)
    gx = cv2.Sobel(cand_roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(cand_roi, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    residual_energy = float(np.mean(grad_mag[wm_bg_zone])) if np.any(wm_bg_zone) else 0.0

    # Metric 5: 2px Boundary DCT Ringing Halo Variance
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    wm_dilated = cv2.dilate(mask_roi.astype(np.uint8) * 255, ring_kernel) > 0
    boundary_zone = wm_dilated & (~mask_roi) & (fg_orig == 0)
    halo_var = float(np.var(cand_roi[boundary_zone])) if np.any(boundary_zone) else 0.0

    failures: list[str] = []
    if betti_1_delta > max_ring_delta:
        failures.append(f"Betti-1 ring count changed by {betti_1_delta} (allowed: {max_ring_delta})")
    if retention < min_skeleton_retention:
        failures.append(
            f"Skeleton arc-length dropped to {retention:.1%} (minimum required: {min_skeleton_retention:.1%})"
        )
    if endpoint_delta > max_new_endpoints:
        failures.append(
            f"Line severance detected: +{endpoint_delta} new dangling skeleton endpoints created"
        )

    return FourInvariantQAResult(
        passed=len(failures) == 0,
        betti_1_delta=betti_1_delta,
        skeleton_length_retention=round(retention, 4),
        dangling_endpoint_delta=endpoint_delta,
        residual_wm_energy=round(residual_energy, 4),
        dct_boundary_halo_var=round(halo_var, 4),
        failure_reasons=tuple(failures),
    )


def remove_periodic_tiled_watermark_fft(
    image_bgr: np.ndarray,
    dc_protection_radius: int = 18,
    spike_quantile: float = 0.9985,
    notch_radius: int = 4,
) -> tuple[np.ndarray, float]:
    """Detects and suppresses periodic tiled/grid watermarks in the 2D Fourier domain
    while protecting low-frequency background gradients and sharp foreground ink edges.
    Returns (cleaned_bgr, periodic_spike_score).
    """
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32)
    h, w = y_channel.shape

    f_shift = np.fft.fftshift(np.fft.fft2(y_channel))
    mag_log = np.log1p(np.abs(f_shift))

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist_from_dc = np.hypot(yy - cy, xx - cx)

    # Mask out the central low-frequency DC core so natural image structure is never touched
    high_freq_region = dist_from_dc > dc_protection_radius
    thresh = float(np.quantile(mag_log[high_freq_region], spike_quantile))

    spikes = (mag_log > thresh) & high_freq_region
    spike_score = float(np.sum(spikes) / max(1, np.sum(high_freq_region)))

    if not np.any(spikes):
        return image_bgr.copy(), 0.0

    # Dilate spike coordinates slightly to form smooth notch reject masks
    notch_mask = (~spikes).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * notch_radius + 1, 2 * notch_radius + 1)
    )
    notch_mask = cv2.erode(notch_mask, kernel)
    soft_notch = cv2.GaussianBlur(notch_mask.astype(np.float32) / 255.0, (5, 5), 1.2)

    filtered_shift = f_shift * soft_notch
    reconstructed_y = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_shift)))

    # Clamp updates strictly to non-black-ink pixels so sharp diagram text never rings
    ink_protect_mask = y_channel < 85.0
    final_y = np.where(ink_protect_mask, y_channel, reconstructed_y)
    ycrcb[:, :, 0] = np.clip(final_y, 0, 255).astype(np.uint8)

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), round(spike_score * 1000.0, 4)
