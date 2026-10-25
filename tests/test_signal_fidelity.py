"""Signal fidelity tests: PSNR, Betti loop preservation, and skeleton retention on scientific diagrams."""

import cv2
import numpy as np

from tests.benchmarks.fixture_generator import draw_benzene_ring
from wmclean.removers.two_regime import two_regime_fill
from wmclean.verify.hardened_math_qa import evaluate_four_invariant_qa
from wmclean.verify.raster import verify_raster


def test_chemistry_ring_signal_fidelity():
    """Verify chemistry ring diagram watermark removal preserves PSNR >= 35.0, Δβ1 = 0, and skeleton retention >= 97%."""
    # 1. Ground truth clean chemistry diagram
    gt = np.full((240, 240, 3), 255, dtype=np.uint8)
    draw_benzene_ring(gt, (120, 120), radius=50, line_width=2)
    gt_grey = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

    # 2. Add realistic semi-transparent watermark text across the diagram
    watermarked = gt.copy()
    wm_mask = np.zeros((240, 240), dtype=np.uint8)
    cv2.putText(
        wm_mask,
        "SAMPLE",
        (25, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        255,
        2,
        cv2.LINE_AA,
    )

    # Blend watermark only where wm_mask is active (alpha=0.30)
    alpha = 0.30
    for c in range(3):
        watermarked[:, :, c] = np.where(
            wm_mask > 0,
            (1.0 - alpha) * gt[:, :, c] + alpha * 185,
            gt[:, :, c],
        ).astype(np.uint8)


    # 3. Clean watermark using two-regime fill with the watermark mask
    cleaned = two_regime_fill(watermarked, mask=wm_mask)
    clean_grey = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)


    # 4. Measure PSNR
    mse = np.mean((gt_grey.astype(float) - clean_grey.astype(float)) ** 2)
    psnr = 10.0 * np.log10((255.0 ** 2) / mse) if mse > 0 else 100.0

    # Cleaned image must have high PSNR against ground truth
    assert psnr >= 35.0, f"PSNR {psnr:.2f} dB < 35.0 dB threshold"

    # 5. Measure 4-Invariant Topological QA
    qa_res = evaluate_four_invariant_qa(gt_grey, clean_grey, wm_mask)


    # Topological loops must be exactly preserved (no broken benzene rings)
    assert qa_res.betti_1_delta == 0, f"Betti-1 loops altered: {qa_res.betti_1_delta}"

    # Diagram skeleton strokes must have >= 97% retention
    assert qa_res.skeleton_length_retention >= 0.97, (
        f"Skeleton retention {qa_res.skeleton_length_retention * 100:.1f}% < 97%"
    )

    # Dangling endpoints must not introduce breaks (delta <= 2)
    assert qa_res.dangling_endpoint_delta <= 2, f"Dangling tip delta {qa_res.dangling_endpoint_delta} too high"


    # 6. Overall raster verification pass
    v_res = verify_raster(watermarked, cleaned, wm_mask=wm_mask)
    assert v_res.passed is True

