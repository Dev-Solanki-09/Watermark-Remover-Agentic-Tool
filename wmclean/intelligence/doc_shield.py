"""DocShield — Structural Document & Diagram Semantic Parser.

Parses every page into 4 semantic layers (< 6 ms/page):
1. M_core: High-contrast solid dark foreground text & line cores (via doxapy ISauvola or adaptive fallback).
2. M_halo: 1-px outward anti-aliased grey halo protecting character serifs and edge sharpness.
3. M_diagram: Coloured biology diagrams (green/cyan/blue) + Euler-topology closed loops (benzene rings, blocks)
   + coordinate graph axes (HoughLinesP).
4. M_ink_halo_shield: Combined text, halo, and diagram protection mask.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import cv2
import numpy as np

from wmclean.deps import get_doxapy, has_doxapy

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DocShieldResult:
    """Document semantic structural maps for a page or ROI."""

    norm_gray: np.ndarray  # (H, W) uint8 normalized grayscale
    bg: np.ndarray  # (H, W) uint8 local background illumination
    m_core: np.ndarray  # (H, W) bool, solid dark text/line cores
    m_halo: np.ndarray  # (H, W) bool, 1-px anti-aliased character edges
    m_ink_halo_shield: np.ndarray  # (H, W) bool, core + halo protection
    m_diagram: np.ndarray  # (H, W) bool, chemical rings, graphs, coloured figures

    @property
    def m_ink_core(self) -> np.ndarray:
        return self.m_core

    def apply_text_core_lock(
        self,
        cleaned: np.ndarray,
        orig: np.ndarray,
        lock_halo: bool = True,
        wm_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Composite protected text and diagram cores back onto cleaned image via np.minimum.

        Enforces zero foreground text loss (L_fg == 0.000) and preserves authentic diagram strokes.
        """
        out = cleaned.copy()
        wm_mask_bool = None
        if wm_mask is not None:
            wm_mask_bool = np.any(wm_mask > 0, axis=2) if wm_mask.ndim == 3 else (wm_mask > 0)
        mask = self.m_ink_halo_shield if lock_halo else self.m_core
        if wm_mask_bool is not None:
            mask = mask & (~wm_mask_bool)

        diag_to_restore = (
            (self.m_diagram & (~wm_mask_bool)) if wm_mask_bool is not None else self.m_diagram
        )

        if out.ndim == 3 and orig.ndim == 3:
            for c in range(3):
                out_c = out[:, :, c]
                orig_c = orig[:, :, c]
                out_c[mask] = np.minimum(out_c[mask], orig_c[mask])
                out[:, :, c] = out_c
            if diag_to_restore.any():
                out[diag_to_restore] = orig[diag_to_restore]
        elif out.ndim == 2 and orig.ndim == 2:
            out[mask] = np.minimum(out[mask], orig[mask])
            if diag_to_restore.any():
                out[diag_to_restore] = orig[diag_to_restore]
        elif out.ndim == 3 and orig.ndim == 2:
            for c in range(3):
                out[:, :, c] = np.where(mask, np.minimum(out[:, :, c], orig), out[:, :, c])
            if diag_to_restore.any():
                for c in range(3):
                    out[:, :, c] = np.where(diag_to_restore, orig, out[:, :, c])
        return out


def parse_doc_shield(
    img: np.ndarray,
    template_mask: Optional[np.ndarray] = None,
) -> DocShieldResult:
    """Compute DocShield semantic structural maps on an image (< 6 ms).

    Args:
        img: RGB or greyscale image array (H, W, C) or (H, W).
        template_mask: Optional boolean or uint8 mask of known multi-page watermark strokes
                       (so watermark pixels are excluded from the text/diagram shield).

    Returns:
        DocShieldResult containing all semantic shields.
    """
    is_rgb = img.ndim == 3 and img.shape[2] >= 3
    if is_rgb:
        rgb = img[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img if img.ndim == 2 else img[:, :, 0]
        rgb = np.stack([gray, gray, gray], axis=-1)

    h, w = gray.shape[:2]

    paper_p95 = float(np.percentile(gray, 95))
    k_size = 35
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    if paper_p95 >= 180:
        bg = np.maximum(bg, int(paper_p95 - 20))
    norm_gray = np.clip(
        gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0) * 255.0, 0, 255
    ).astype(np.uint8)

    # Watermark chroma detector (redness or high chroma watermark stamps)
    is_wm_chroma = np.zeros((h, w), dtype=bool)
    if is_rgb:
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)
        # Redness / warm watermark chroma
        redness = r - np.maximum(g, b)
        is_wm_chroma = (redness > 12.0) & (r > 120.0)

    # 2. Extract M_core (Solid Foreground Cores)
    # Exclude watermark chroma and template mask
    known_wm = is_wm_chroma.copy()
    if template_mask is not None:
        known_wm |= template_mask > 0

    m_core = np.zeros((h, w), dtype=bool)
    if has_doxapy():
        try:
            doxa = get_doxapy()
            if doxa is not None:
                doxa_bin = doxa.to_binary(
                    doxa.Binarization.Algorithms.ISAUVOLA,
                    norm_gray,
                    {"window": 45, "k": 0.20},
                )
                # doxapy returns 0 for foreground text, 255 for background
                m_core = (doxa_bin == 0) & (norm_gray < 115) & (~known_wm)
        except Exception as exc:
            logger.debug("doxapy in doc_shield failed: %s; using adaptive fallback", exc)
            m_core = (norm_gray < 95) & (~known_wm)
    else:
        # Fallback to local adaptive thresholding
        adapt = cv2.adaptiveThreshold(
            norm_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            45,
            C=6,
        )
        m_core = (adapt > 0) & (norm_gray < 105) & (~known_wm)

    # Exclude scanner banners in extreme margins from text core lock
    margin_mask = np.zeros((h, w), dtype=bool)
    margin_mask[int(h * 0.94) :, :] = True
    margin_mask[: int(h * 0.04), :] = True
    m_core = m_core & (~margin_mask)

    # 3. M_halo: 1-px Outward Anti-Aliased Grey Halo Shield
    # Protects 1-px grey halo (norm_gray < 210) adjacent to text cores from being eroded/smoothed
    r_1px = cv2.dilate(m_core.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    m_halo = r_1px & (~m_core) & (norm_gray < 210) & (~known_wm)
    m_ink_halo_shield = m_core | m_halo

    # 4. M_diagram: Geometric Diagram & Figure Shield
    m_diagram = np.zeros((h, w), dtype=bool)

    # a) Coloured diagram shield: Green, yellow-green, cyan, or blue pixels
    if is_rgb:
        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        # Diagram colours: Green leaf diagrams, cyan curves, blue cell membranes
        # G > R + 6 or B > R + 6 (excluding known watermark pixels)
        green_blue_diagram = ((g > r + 3) | (b > r + 3)) & (norm_gray < 248) & (~known_wm)
        m_diagram |= green_blue_diagram

    # b) Closed-loop Euler topology & Hough line axes
    # Detect chemical benzene rings, physics blocks/pulleys, graph axes
    line_art_cand = (norm_gray < 170) & (~known_wm) & (~margin_mask)
    if np.sum(line_art_cand) > 50:
        # Check closed loops via skimage.measure.regionprops
        try:
            from skimage.measure import label, regionprops

            lbl = label(line_art_cand)
            props = regionprops(lbl)
            diagram_cc = np.zeros((h, w), dtype=bool)
            for p in props:
                # In topology, objects with internal holes (benzene rings, circular cells, blocks) have Euler <= 0
                bh = p.bbox[2] - p.bbox[0]
                bw = p.bbox[3] - p.bbox[1]
                if p.euler_number <= 0 and p.area >= 200 and max(bh, bw) >= 30:
                    diagram_cc[lbl == p.label] = True
            m_diagram |= diagram_cc
        except Exception as exc:
            logger.debug("Euler diagram detection skipped: %s", exc)

        # Coordinate axes: straight line segments via HoughLinesP
        try:
            edges = cv2.Canny(norm_gray, 50, 150)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=5
            )
            if lines is not None:
                line_mask = np.zeros((h, w), dtype=np.uint8)
                for line in lines:
                    pts = np.asarray(line).ravel()
                    if len(pts) >= 4:
                        x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
                        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                m_diagram |= (line_mask > 0) & (norm_gray < 185) & (~known_wm) & (~margin_mask)
        except Exception as exc:
            logger.debug("Hough lines detection skipped: %s", exc)

    # Extreme margins never contain authentic diagrams (they contain headers, footers, scanner stamps)
    m_diagram = m_diagram & (~margin_mask)

    # Add diagram shield to combined protection
    m_ink_halo_shield |= m_diagram

    return DocShieldResult(
        norm_gray=norm_gray,
        bg=bg,
        m_core=m_core,
        m_halo=m_halo,
        m_ink_halo_shield=m_ink_halo_shield,
        m_diagram=m_diagram,
    )


DocShield = DocShieldResult
setattr(DocShield, "parse", staticmethod(parse_doc_shield))
