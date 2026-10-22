"""Zero-Vision-Token Semantic Page Map & Vision Crop Inspector (wmclean peek).

Generates an 85-token 16x8 ASCII character grid and anomaly table so calling LLMs
have complete spatial, structural, chromatic, and OCR awareness of a page without
burning expensive vision tokens. When vision inspection is desired, outputs a compact
side-by-side Before vs. Preview-Cleaned PNG crop (< 320px) in < 80 vision tokens.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pymupdf

from wmclean.intelligence.doc_shield import parse_doc_shield
from wmclean.intelligence.steering_dsl import NAMED_ZONES

logger = logging.getLogger(__name__)


def peek_page(
    pdf_path: str,
    page_num: int,  # 1-indexed
    password: Optional[str] = None,
    crop_roi: Optional[str] = None,
    output_crop_path: Optional[str] = None,
    target_dpi: int = 150,
) -> tuple[str, list[dict[str, Any]]]:
    """Compute semantic spatial map and anomaly list for a page.

    Args:
        pdf_path: Path to input PDF.
        page_num: 1-indexed page number.
        password: Optional PDF password.
        crop_roi: Optional ROI name ('ROI_1', 'top-left') or normalized '[x1,y1,x2,y2]'.
        output_crop_path: Optional path to save side-by-side Before/After crop image.
        target_dpi: Rendering DPI (default 150).

    Returns:
        tuple[str, list[dict]]: Formatted ASCII report string and list of anomaly dicts.
    """
    doc = pymupdf.open(pdf_path)
    if password:
        doc.authenticate(password)

    page_idx = page_num - 1
    if page_idx < 0 or page_idx >= len(doc):
        doc.close()
        raise ValueError(f"Page {page_num} out of bounds (document has {len(doc)} pages)")

    page = doc[page_idx]
    pix = page.get_pixmap(dpi=target_dpi)
    orig_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        orig_arr = orig_arr[:, :, :3]
    elif pix.n == 1:
        orig_arr = np.stack([orig_arr, orig_arr, orig_arr], axis=2)

    h, w = orig_arr.shape[:2]

    # Compute DocShield
    shield = parse_doc_shield(orig_arr)
    norm_gray = shield.norm_gray
    m_core = shield.m_core
    m_diag = shield.m_diagram

    # Candidate watermark mask: mid-grey or chroma anomaly outside shielded text/diagrams
    is_rgb = orig_arr.ndim == 3 and orig_arr.shape[2] >= 3
    if is_rgb:
        r = orig_arr[:, :, 0].astype(np.float32)
        g = orig_arr[:, :, 1].astype(np.float32)
        b = orig_arr[:, :, 2].astype(np.float32)
        chroma_anomaly = (np.abs(r - g) > 25) | (np.abs(r - b) > 25)
    else:
        chroma_anomaly = np.zeros((h, w), dtype=bool)

    wm_cand = (((norm_gray >= 80) & (norm_gray <= 235)) | chroma_anomaly) & (~m_core) & (~m_diag)

    # Collision mask: watermark candidate directly abutting text or diagram halo
    kernel_3x3 = np.ones((3, 3), np.uint8)
    dil_core = cv2.dilate((m_core | m_diag).astype(np.uint8), kernel_3x3) > 0
    collision = wm_cand & dil_core

    # Build 16x8 ASCII grid
    grid_cols = 16
    grid_rows = 8
    cell_w = max(1, w // grid_cols)
    cell_h = max(1, h // grid_rows)

    grid_lines: list[str] = []
    for r_idx in range(grid_rows):
        y0 = r_idx * cell_h
        y1 = min(h, (r_idx + 1) * cell_h)
        row_chars: list[str] = []
        for c_idx in range(grid_cols):
            x0 = c_idx * cell_w
            x1 = min(w, (c_idx + 1) * cell_w)

            cell_col = collision[y0:y1, x0:x1]
            cell_wm = wm_cand[y0:y1, x0:x1]
            cell_diag = m_diag[y0:y1, x0:x1]
            cell_text = m_core[y0:y1, x0:x1]

            n_pix = cell_col.size
            if cell_col.sum() > n_pix * 0.05:
                row_chars.append("!")
            elif cell_diag.sum() > n_pix * 0.08:
                row_chars.append("D")
            elif cell_wm.sum() > n_pix * 0.06:
                row_chars.append("W")
            elif cell_text.sum() > n_pix * 0.04:
                row_chars.append("T")
            else:
                row_chars.append(".")

        y_pct_0 = r_idx / grid_rows
        y_pct_1 = (r_idx + 1) / grid_rows
        grid_lines.append(f"  {''.join(row_chars)}  y={y_pct_0:.2f}-{y_pct_1:.2f}")

    # Detect Anomaly Regions (ROIs) using connected components
    anomalies: list[dict[str, Any]] = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        wm_cand.astype(np.uint8), connectivity=8
    )

    roi_idx = 1
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 80:  # Minimum anomaly area threshold
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            norm_box = (
                round(bx / w, 2),
                round(by / h, 2),
                round((bx + bw) / w, 2),
                round((by + bh) / h, 2),
            )

            crop_gray = norm_gray[by : by + bh, bx : bx + bw]
            luma_mean = int(np.mean(crop_gray))
            dominant_hue = "grey"
            if is_rgb:
                crop_rgb = orig_arr[by : by + bh, bx : bx + bw]
                r_m = np.mean(crop_rgb[:, :, 0])
                g_m = np.mean(crop_rgb[:, :, 1])
                b_m = np.mean(crop_rgb[:, :, 2])
                if r_m > g_m + 15 and r_m > b_m + 15:
                    dominant_hue = "red"
                elif b_m > r_m + 15:
                    dominant_hue = "blue"

            # Check for text in this bbox via page text
            pt_rect = pymupdf.Rect(
                bx * 72.0 / target_dpi,
                by * 72.0 / target_dpi,
                (bx + bw) * 72.0 / target_dpi,
                (by + bh) * 72.0 / target_dpi,
            )
            text_inside = page.get_text("text", clip=pt_rect).strip().replace("\n", " ")
            if len(text_inside) > 50:
                text_inside = text_inside[:47] + "..."

            has_overlap = bool(collision[by : by + bh, bx : bx + bw].any())

            anomalies.append(
                {
                    "id": f"ROI_{roi_idx}",
                    "bbox": norm_box,
                    "area_px": int(area),
                    "luma": luma_mean,
                    "hue": dominant_hue,
                    "overlap": has_overlap,
                    "text": text_inside,
                    "px_bbox": (bx, by, bx + bw, by + bh),
                }
            )
            roi_idx += 1
            if roi_idx > 8:
                break

    doc.close()

    # Build ASCII output
    out_lines = [
        f"PEEK p{page_num} ({w}x{h} px, target_dpi={target_dpi})",
        "GRID (16x8: .=paper, T=text, D=diagram, W=wm_cand, !=collision):",
    ]
    out_lines.extend(grid_lines)

    if anomalies:
        out_lines.append(f"ANOMALIES ({len(anomalies)}):")
        for a in anomalies:
            desc = f"  {a['id']} [{a['bbox'][0]:.2f},{a['bbox'][1]:.2f},{a['bbox'][2]:.2f},{a['bbox'][3]:.2f}]: {a['hue']}(luma={a['luma']},area={a['area_px']}px)"
            if a["text"]:
                desc += f' | text="{a["text"]}"'
            if a["overlap"]:
                desc += " | text_overlap=true"
            out_lines.append(desc)

        # Suggested steering command
        target_ids = ",".join(a["id"] for a in anomalies[:3])
        out_lines.append(
            f'SUGGESTED STEER: --steer "p{page_num}@{target_ids}:density_sub+pointwise_unmix"'
        )
    else:
        out_lines.append("ANOMALIES (0): Clean page, no watermark anomalies detected.")
        out_lines.append(f'SUGGESTED STEER: --steer "p{page_num}:skip"')

    report_str = "\n".join(out_lines)

    # If crop_roi requested, generate compact side-by-side Before/After preview
    if crop_roi and output_crop_path:
        _export_crop_preview(
            orig_arr,
            anomalies,
            crop_roi,
            output_crop_path,
            shield,
            w,
            h,
        )

    return report_str, anomalies


def _export_crop_preview(
    orig_arr: np.ndarray,
    anomalies: list[dict[str, Any]],
    crop_roi: str,
    output_crop_path: str,
    shield: Any,
    w: int,
    h: int,
) -> None:
    """Export compact side-by-side Before vs. Preview-Cleaned crop image."""
    from PIL import Image

    # Determine pixel bbox
    px_bbox: Optional[tuple[int, int, int, int]] = None
    crop_roi_clean = crop_roi.strip()

    # Check named anomalies (ROI_1, etc.)
    for a in anomalies:
        if a["id"].lower() == crop_roi_clean.lower():
            px_bbox = a["px_bbox"]
            break

    # Check named zones
    if px_bbox is None and crop_roi_clean.lower() in NAMED_ZONES:
        nx1, ny1, nx2, ny2 = NAMED_ZONES[crop_roi_clean.lower()]
        px_bbox = (int(nx1 * w), int(ny1 * h), int(nx2 * w), int(ny2 * h))

    # Check explicit coords
    if px_bbox is None and (crop_roi_clean.startswith("[") or crop_roi_clean.startswith("(")):
        inner = crop_roi_clean[1:-1].strip()
        try:
            coords = [float(c.strip()) for c in inner.split(",")]
            if len(coords) == 4:
                px_bbox = (
                    int(coords[0] * w),
                    int(coords[1] * h),
                    int(coords[2] * w),
                    int(coords[3] * h),
                )
        except Exception:
            pass

    if px_bbox is None:
        px_bbox = (0, 0, min(w, 400), min(h, 400))

    x1, y1, x2, y2 = px_bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return

    crop_before = orig_arr[y1:y2, x1:x2].copy()

    # Create quick preview cleaned patch using doc_shield
    crop_shield = shield.norm_gray[y1:y2, x1:x2]
    crop_core = shield.m_core[y1:y2, x1:x2]
    crop_diag = shield.m_diagram[y1:y2, x1:x2]
    crop_clean = crop_before.copy()

    # Whiten non-shielded watermark pixels in preview
    wm_mask = (crop_shield >= 90) & (crop_shield <= 235) & (~crop_core) & (~crop_diag)
    crop_bg = shield.bg[y1:y2, x1:x2]
    for c in range(crop_clean.shape[2]):
        crop_clean[:, :, c] = np.where(wm_mask, crop_bg, crop_clean[:, :, c])

    # Side-by-side composite with a 2-pixel black divider
    divider = np.zeros((y2 - y1, 2, 3), dtype=np.uint8)
    preview_arr = np.hstack([crop_before, divider, crop_clean])

    # Scale to max 320 px wide for extreme vision token economy (< 80 vision tokens)
    pw_h, pw_w = preview_arr.shape[:2]
    if pw_w > 320:
        scale = 320.0 / pw_w
        preview_arr = cv2.resize(
            preview_arr, (320, max(1, int(pw_h * scale))), interpolation=cv2.INTER_AREA
        )

    out_p = Path(output_crop_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(preview_arr)
    img.save(str(out_p), format="PNG", optimize=True)
    logger.info("Saved compact side-by-side preview crop to %s", output_crop_path)
