"""Strict Anti-Smudge & Sharpness Verification Test (verifying 2.md & 3.md).

Enforces:
1. Diagram SSIM >= 0.995 (zero smudging or line distortion on physics graphs, benzene rings, green diagrams)
2. Text Anti-Aliased Edge Gradient Correlation >= 0.990 (zero melting/thinning of character serifs)
3. Watermark Residual <= 0.025 (>95% watermark energy removed)
4. cv2.inpaint is NOT called in the default zero-smudge pipeline
"""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from wmclean.removers.density_sub import density_sub
from wmclean.removers.pointwise_unmix import pointwise_hue_and_lut


def test_zero_smudge_and_diagram_integrity():
    """Synthesize physics graph + benzene ring + 10-pt text + green diagram + watermarks."""
    h, w = 700, 700
    clean_original = np.full((h, w, 3), 252, dtype=np.uint8)

    # 1. Physics coordinate graph (thin 1-px axes + dashed grid)
    cv2.line(clean_original, (80, 50), (80, 250), (15, 15, 15), 1)  # y-axis
    cv2.line(clean_original, (80, 250), (280, 250), (15, 15, 15), 1)  # x-axis
    # Dashed line
    for y_step in range(70, 240, 15):
        cv2.line(clean_original, (80, y_step), (85, y_step), (60, 60, 60), 1)
    for x_step in range(100, 270, 20):
        cv2.line(clean_original, (x_step, 245), (x_step, 250), (60, 60, 60), 1)

    # 2. Chemical benzene ring (hexagon with internal circle)
    center = (450, 150)
    r_hex = 40
    pts = []
    for i in range(6):
        angle = i * np.pi / 3.0
        pts.append([int(center[0] + r_hex * np.cos(angle)), int(center[1] + r_hex * np.sin(angle))])
    pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(clean_original, [pts_arr], isClosed=True, color=(20, 20, 20), thickness=2)
    cv2.circle(clean_original, center, 22, (20, 20, 20), 1)

    # 3. 10-pt Black text with 1-px anti-aliased grey edges
    cv2.putText(
        clean_original,
        "Fig 1.2: Kinetic energy distribution and aromaticity.",
        (60, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        clean_original,
        "Answer questions (a) to (d) showing complete working.",
        (60, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )

    # 4. Coloured green diagram
    cv2.rectangle(clean_original, (450, 260), (530, 340), (40, 180, 40), -1)
    cv2.circle(clean_original, (490, 300), 25, (20, 120, 20), 2)

    # Region masks for evaluation
    graph_roi = (slice(40, 260), slice(70, 290))
    benzene_roi = (slice(100, 200), slice(400, 500))
    green_mask = clean_original[:, :, 1].astype(int) > clean_original[:, :, 0].astype(int) + 10

    # -----------------------------------------------------------------------
    # Overwrite with Watermarks:
    # (a) Red circular watermark
    # (b) Hollow diagonal outline watermark
    # (c) Dark top-left repeated stamp
    # -----------------------------------------------------------------------
    watermarked = clean_original.copy()

    # (a) Red circular logo behind text & benzene ring
    y_g, x_g = np.ogrid[:h, :w]
    c_logo = (350, 250)
    d_logo = np.sqrt((x_g - c_logo[0]) ** 2 + (y_g - c_logo[1]) ** 2)
    red_ring = (d_logo <= 180) & (d_logo >= 160)
    for c, val in enumerate([240, 175, 175]):
        watermarked[:, :, c] = np.where(
            red_ring & (~green_mask), np.minimum(watermarked[:, :, c], val), watermarked[:, :, c]
        )

    # (b) Hollow diagonal outline
    diag_pts = np.abs((x_g - 50) - (y_g - 50)) < 4
    diag_wm = diag_pts & (x_g > 100) & (x_g < 600) & (y_g > 100) & (y_g < 600)
    for c in range(3):
        watermarked[:, :, c] = np.where(
            diag_wm, np.minimum(watermarked[:, :, c], 195), watermarked[:, :, c]
        )

    # (c) Dark top-left repeated stamp
    stamp_img = watermarked.copy()
    cv2.putText(
        stamp_img, "SEARCH ON TELEGRAM", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (55, 55, 55), 2
    )
    dark_stamp_mask = stamp_img[:, :, 0] == 55
    watermarked[dark_stamp_mask] = 55

    # -----------------------------------------------------------------------
    # Clean using Zero-Smudge Pipeline
    # Ensure cv2.inpaint is NEVER called!
    # -----------------------------------------------------------------------
    with patch("cv2.inpaint", side_effect=RuntimeError("cv2.inpaint called in zero-smudge path!")):
        # Stage 1: Pointwise Hue & Monotonic LUT Unmixing
        clean_step1 = pointwise_hue_and_lut(watermarked)
        # Stage 2: Continuous Optical-Density Subtraction
        clean_final = density_sub(clean_step1)

    # -----------------------------------------------------------------------
    # Strict Invariant Assertions
    # -----------------------------------------------------------------------
    # Invariant 1: Diagram SSIM >= 0.995 on Physics Graph and Benzene Ring
    orig_graph = cv2.cvtColor(clean_original[graph_roi], cv2.COLOR_RGB2GRAY)
    final_graph = cv2.cvtColor(clean_final[graph_roi], cv2.COLOR_RGB2GRAY)
    graph_ssim = ssim(orig_graph, final_graph)
    assert graph_ssim >= 0.995, f"Physics graph SSIM {graph_ssim:.4f} < 0.995"

    orig_benzene = cv2.cvtColor(clean_original[benzene_roi], cv2.COLOR_RGB2GRAY)
    final_benzene = cv2.cvtColor(clean_final[benzene_roi], cv2.COLOR_RGB2GRAY)
    benzene_ssim = ssim(orig_benzene, final_benzene)
    assert benzene_ssim >= 0.995, f"Benzene ring SSIM {benzene_ssim:.4f} < 0.995"

    # Invariant 2: Green Diagram 100% Preserved
    green_diff = np.max(
        np.abs(clean_original[green_mask].astype(int) - clean_final[green_mask].astype(int))
    )
    assert green_diff == 0, f"Green diagram modified by {green_diff} px"

    # Invariant 3: Text anti-aliased edge gradient correlation >= 0.990
    text_roi = (slice(310, 380), slice(50, 650))
    orig_text = cv2.cvtColor(clean_original[text_roi], cv2.COLOR_RGB2GRAY).astype(np.float32)
    final_text = cv2.cvtColor(clean_final[text_roi], cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_orig_x = cv2.Sobel(orig_text, cv2.CV_32F, 1, 0, ksize=3)
    grad_final_x = cv2.Sobel(final_text, cv2.CV_32F, 1, 0, ksize=3)
    corr = np.corrcoef(grad_orig_x.ravel(), grad_final_x.ravel())[0, 1]
    assert corr >= 0.990, f"Text edge gradient correlation {corr:.4f} < 0.990"

    # Invariant 4: Watermark residual <= 0.025 on unshielded watermark background
    from wmclean.intelligence.doc_shield import parse_doc_shield

    shield = parse_doc_shield(clean_original)
    eval_mask = red_ring & (~shield.m_diagram) & (~shield.m_ink_halo_shield)
    bg_clean = np.percentile(clean_final, 95)
    red_residual = np.mean(np.abs(clean_final[eval_mask].astype(float) - bg_clean)) / 255.0
    assert red_residual <= 0.025, f"Red logo residual {red_residual:.4f} > 0.025"
