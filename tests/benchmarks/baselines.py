"""Reference baseline methods for benchmark comparison.

Includes:
1. NoOpBaseline: Returns input image completely unmodified.
2. OracleThresholdBaseline: Sweeps luminance thresholds against ground truth
   to find the best possible single-threshold pixel unmixing.
3. SauvolaBaseline: Standard local Sauvola binarization baseline.
"""

from __future__ import annotations

import cv2
import numpy as np


class NoOpBaseline:
    """Identity baseline: leaves input image untouched."""

    name = "No-Op (Identity)"

    def clean(self, img_bgr: np.ndarray, **kwargs) -> np.ndarray:
        return img_bgr.copy()


class OracleThresholdBaseline:
    """Oracle-tuned threshold baseline.

    Uses ground-truth knowledge (if available) or background statistics
    to find the single scalar threshold in [160, 250] that minimizes
    watermark residual while penalizing foreground ink loss.
    """

    name = "Oracle Threshold"

    def clean(
        self,
        img_bgr: np.ndarray,
        clean_gt_bgr: np.ndarray | None = None,
        wm_mask_gt: np.ndarray | None = None,
        fg_mask_gt: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if clean_gt_bgr is None or fg_mask_gt is None:
            # Blind fallback: estimate paper background as 95th percentile,
            # threshold anything above paper - 35 to paper color.
            paper_val = int(np.percentile(gray, 95))
            thresh = max(160, paper_val - 30)
            res = img_bgr.copy()
            res[gray >= thresh] = paper_val
            return res

        # Oracle mode: search optimal threshold T in [160, 254] in steps of 4
        best_t = 240
        best_cost = float("inf")
        paper_bgr = np.median(img_bgr[gray >= 230], axis=0) if np.any(gray >= 230) else np.array([255, 255, 255])
        paper_bgr = np.clip(paper_bgr, 240, 255).astype(np.uint8)

        fg_bool = fg_mask_gt.astype(bool)
        wm_bool = wm_mask_gt.astype(bool) if wm_mask_gt is not None else np.ones((h, w), dtype=bool)

        # Vectorized sweep across candidate thresholds
        candidates = list(range(170, 252, 5))
        for t in candidates:
            # Pixels >= t turned to paper
            whiten_mask = (gray >= t)
            # Residual: where wm was present, diff between whiten and clean_gt
            # Ink loss: where fg_mask_gt was true, count how many got whitened
            ink_lost = np.count_nonzero(whiten_mask & fg_bool)
            ink_loss_ratio = ink_lost / max(1, np.count_nonzero(fg_bool))

            # Paper residual error
            paper_pixels = (~fg_bool) & wm_bool
            if np.any(paper_pixels):
                # Where whitened, diff is paper_bgr vs clean_gt
                # Where not whitened, diff is img_bgr vs clean_gt
                diff = np.where(
                    whiten_mask[:, :, None],
                    np.abs(paper_bgr.astype(np.float32) - clean_gt_bgr.astype(np.float32)),
                    np.abs(img_bgr.astype(np.float32) - clean_gt_bgr.astype(np.float32))
                )
                mean_res = np.mean(diff[paper_pixels])
            else:
                mean_res = 0.0

            # Cost: residual + heavy penalty for destroying real text/diagrams
            cost = mean_res + 250.0 * ink_loss_ratio
            if cost < best_cost:
                best_cost = cost
                best_t = t

        # Apply best threshold
        res = img_bgr.copy()
        whiten = (gray >= best_t)
        res[whiten] = paper_bgr
        return res


class SauvolaBaseline:
    """Standard Sauvola local adaptive binarization baseline."""

    name = "Sauvola Binarization"

    def clean(self, img_bgr: np.ndarray, window_size: int = 25, k: float = 0.2, **kwargs) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.boxFilter(gray, -1, (window_size, window_size), borderType=cv2.BORDER_REFLECT)
        sqr_mean = cv2.boxFilter(gray ** 2, -1, (window_size, window_size), borderType=cv2.BORDER_REFLECT)
        variance = np.maximum(sqr_mean - mean ** 2, 0)
        std = np.sqrt(variance)
        r = 128.0
        threshold = mean * (1.0 + k * (std / r - 1.0))

        fg_mask = gray <= threshold
        res = np.full_like(img_bgr, 255)
        res[fg_mask] = img_bgr[fg_mask]
        return res
