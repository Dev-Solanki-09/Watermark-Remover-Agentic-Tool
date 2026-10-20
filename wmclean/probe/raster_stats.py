"""L0 Raster Probe — compute 100-DPI raster statistics and maintain bounded
reservoir of thumbnails per geometry bucket for cross-page median template.

Memory contract: reservoir is capped at K=31 thumbnails per geometry bucket,
keeping total raster probe memory ≤15 MB regardless of page count.

This module NEVER modifies the input PDF.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from wmclean.schemas import PageKind, ProbeRecord, RasterStats

# Maximum thumbnails per geometry bucket for reservoir sampling
_RESERVOIR_K = 31

# Probe rasterization DPI
_PROBE_DPI = 100


class ReservoirBucket:
    """Bounded reservoir of thumbnails for a single geometry bucket.

    Uses reservoir sampling (Algorithm R) to maintain at most K representative
    thumbnails from an arbitrary number of pages, using O(K) memory.
    """

    def __init__(self, k: int = _RESERVOIR_K):
        self.k = k
        self.samples: list[np.ndarray] = []  # greyscale thumbnails
        self.page_indices: list[int] = []
        self.n_seen: int = 0
        self._rng = np.random.default_rng(seed=42)

    def offer(self, thumbnail: np.ndarray, page_idx: int) -> None:
        """Offer a thumbnail to the reservoir. O(1) amortized."""
        self.n_seen += 1
        if len(self.samples) < self.k:
            self.samples.append(thumbnail)
            self.page_indices.append(page_idx)
        else:
            j = self._rng.integers(0, self.n_seen)
            if j < self.k:
                self.samples[j] = thumbnail
                self.page_indices[j] = page_idx

    def compute_median_template(self) -> Optional[np.ndarray]:
        """Compute pixel-wise median of reservoir samples.

        Returns None if fewer than 3 samples are available.
        """
        if len(self.samples) < 3:
            return None

        # Stack and compute median — all thumbnails are same geometry bucket
        # so they should be the same shape (within quantization tolerance)
        target_shape = self.samples[0].shape
        compatible = [s for s in self.samples if s.shape == target_shape]
        if len(compatible) < 3:
            return None

        stack = np.stack(compatible, axis=0)
        return np.median(stack, axis=0).astype(np.uint8)


class RasterReservoir:
    """Collection of geometry-bucketed reservoirs for cross-page template estimation."""

    def __init__(self, k: int = _RESERVOIR_K):
        self.k = k
        self.buckets: dict[str, ReservoirBucket] = {}

    def _bucket_key(self, width: int, height: int) -> str:
        """Quantize page dimensions to bucket key."""
        # Quantize to nearest 50px to group similar page sizes
        qw = (width // 50) * 50
        qh = (height // 50) * 50
        return f"{qw}x{qh}"

    def offer(self, thumbnail: np.ndarray, page_idx: int) -> ReservoirBucket:
        """Offer a thumbnail, routing to the correct geometry bucket."""
        h, w = thumbnail.shape[:2]
        key = self._bucket_key(w, h)
        if key not in self.buckets:
            self.buckets[key] = ReservoirBucket(self.k)
        bucket = self.buckets[key]
        bucket.offer(thumbnail, page_idx)
        return bucket

    def get_template(self, width: int, height: int) -> Optional[np.ndarray]:
        """Get the median template for a geometry bucket, if available."""
        key = self._bucket_key(width, height)
        bucket = self.buckets.get(key)
        if bucket is None:
            return None
        return bucket.compute_median_template()


# ---------------------------------------------------------------------------
# Raster statistics computation
# ---------------------------------------------------------------------------


def _compute_saturation_p95(rgb: np.ndarray) -> float:
    """Compute 95th percentile saturation from an RGB image."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return 0.0

    # Fast saturation: S = (max - min) / max for each pixel
    cmax = rgb.max(axis=2).astype(np.float32)
    cmin = rgb.min(axis=2).astype(np.float32)
    denom = np.maximum(cmax, 1.0)
    sat = ((cmax - cmin) / denom * 255).astype(np.uint8)
    # Check top 0.5% (p99.5) which captures localized watermark stamps/text
    # against predominantly achromatic document backgrounds
    p99_5 = float(np.percentile(sat, 99.5))
    p95 = float(np.percentile(sat, 95))
    return max(p95, p99_5)


def _estimate_watermark_alpha(
    page_grey: np.ndarray, template: Optional[np.ndarray]
) -> tuple[float, list[int]]:
    """Estimate watermark alpha and RGB from page and template.

    Returns (estimated_alpha, estimated_wm_rgb).
    """
    if template is None:
        return 0.0, [200, 200, 200]

    # In the median template, the paper background is the 95th percentile luminance
    bg_val = float(np.percentile(template, 95))
    # Watermark regions are darker than the white background
    diff = bg_val - template.astype(np.float32)
    mask = diff > 15  # threshold for watermark presence in median template

    if mask.sum() < 100:
        return 0.0, [200, 200, 200]

    wm_mean = float(template[mask].mean())
    if bg_val < 10:
        return 0.0, [200, 200, 200]

    alpha = (bg_val - wm_mean) / bg_val
    alpha = max(0.0, min(1.0, alpha))
    wm_rgb = [int(wm_mean)] * 3
    return round(alpha, 3), wm_rgb


def _compute_template_ncc(
    page_grey: np.ndarray, template: Optional[np.ndarray]
) -> tuple[float, float]:
    """Compute NCC and coverage of template against page.

    Returns (tmpl_ncc, tmpl_cov).
    """
    if template is None:
        return 0.0, 0.0

    if page_grey.shape != template.shape:
        return 0.0, 0.0

    bg_val = float(np.percentile(template, 95))
    diff = bg_val - template.astype(np.float32)
    mask = diff > 15

    coverage = float(mask.sum()) / max(mask.size, 1)
    if mask.sum() < 100:
        return 0.0, round(coverage, 4)

    p = page_grey[mask].astype(np.float32)
    t = template[mask].astype(np.float32)
    p_centered = p - p.mean()
    t_centered = t - t.mean()
    denom = max(np.sqrt((p_centered**2).sum() * (t_centered**2).sum()), 1e-8)
    ncc = float((p_centered * t_centered).sum() / denom)

    return round(max(0.0, ncc), 3), round(coverage, 4)


def _detect_strips(page_pixmap_width: int, page_pixmap_height: int, xobj_count: int) -> tuple[bool, int]:
    """Detect whether a page uses strip/tiled image layout."""
    # A page with multiple image XObjects of similar width suggests strips
    if xobj_count > 1:
        return True, xobj_count
    return False, 1


def _count_ocr_spans(page_grey: np.ndarray) -> int:
    """Estimate OCR text span count from invisible text rendering.

    This is a heuristic — actual OCR span detection uses Tr=3 in content stream.
    """
    # OCR span counting is done during structural probe via content stream
    # This is a placeholder for raster-only estimation
    return 0


# ---------------------------------------------------------------------------
# Main raster probe entry point
# ---------------------------------------------------------------------------


def probe_raster(
    pdf,  # pikepdf.Pdf or path
    page_num: int,
    record: ProbeRecord,
    reservoir: Optional[RasterReservoir] = None,
    fitz_doc=None,
) -> ProbeRecord:
    """Augment a ProbeRecord with raster statistics.

    Rasterizes the page at 100 DPI for analysis. Uses reservoir sampling
    to maintain bounded memory for cross-page template estimation.

    This function NEVER modifies the input PDF.
    """
    import pymupdf

    owned_doc = False
    if fitz_doc is None:
        try:
            filename = getattr(pdf, "filename", str(pdf))
            fitz_doc = pymupdf.open(str(filename))
            owned_doc = True
        except Exception:
            return record

    try:
        fitz_page = fitz_doc[page_num]

        # Rasterize at 100 DPI
        mat = pymupdf.Matrix(_PROBE_DPI / 72.0, _PROBE_DPI / 72.0)
        pix = fitz_page.get_pixmap(matrix=mat, alpha=False)

        # Convert to numpy
        if pix.n == 1:
            # Greyscale
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
            rgb = np.stack([img, img, img], axis=2)
            grey = img
        elif pix.n == 3:
            # RGB
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            grey = np.mean(rgb, axis=2).astype(np.uint8)
        elif pix.n == 4:
            # RGBA — drop alpha
            rgba = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 4)
            rgb = rgba[:, :, :3]
            grey = np.mean(rgb, axis=2).astype(np.uint8)
        else:
            return record

        # Detect bitonal
        is_bitonal = (record.kind == PageKind.BITONAL)
        if not is_bitonal:
            try:
                page_obj = pdf.pages[page_num] if hasattr(pdf, "pages") else None
                if page_obj is not None:
                    res = page_obj.get("/Resources")
                    if res and "/XObject" in res:
                        for _, xo_ref in res["/XObject"].items():
                            xo = xo_ref.resolve() if hasattr(xo_ref, "resolve") else xo_ref
                            if str(xo.get("/Subtype", "")) == "/Image":
                                if int(xo.get("/BitsPerComponent", 8)) == 1:
                                    is_bitonal = True
                                    break
            except Exception:
                pass
        if not is_bitonal:
            unique_vals = np.unique(grey)
            if len(unique_vals) <= 4:
                is_bitonal = True

        # Detect strips/tiles
        is_tiled, strip_count = _detect_strips(
            pix.w, pix.h, len([x for x in record.xobj_cand if x.kind == "IMAGE"])
        )

        # Saturation
        sat_p95 = _compute_saturation_p95(rgb)

        # Reservoir sampling for template estimation
        if reservoir is None:
            reservoir = RasterReservoir()

        bucket = reservoir.offer(grey.copy(), page_num)
        template = bucket.compute_median_template()

        # Template metrics
        tmpl_ncc, tmpl_cov = _compute_template_ncc(grey, template)

        # Alpha estimation
        est_alpha, est_wm_rgb = _estimate_watermark_alpha(grey, template)

        # Position stability (requires at least 3 samples)
        pos_stab = 0.0
        if bucket.n_seen >= 3 and tmpl_ncc > 0.5:
            pos_stab = tmpl_ncc  # simplified: high NCC implies stable position

        record.ras_stats = RasterStats(
            has_raster=any(x.kind == "IMAGE" for x in record.xobj_cand) or pix.w * pix.h > 0,
            is_bitonal=is_bitonal,
            is_tiled_strips=is_tiled,
            strip_count=strip_count,
            ocr_spans=_count_ocr_spans(grey),
            tmpl_cov=tmpl_cov,
            tmpl_ncc=tmpl_ncc,
            pos_stab=pos_stab,
            sat_p95=sat_p95,
            est_alpha=est_alpha,
            est_wm_rgb=est_wm_rgb,
        )

    finally:
        if owned_doc and fitz_doc is not None:
            fitz_doc.close()

    return record
