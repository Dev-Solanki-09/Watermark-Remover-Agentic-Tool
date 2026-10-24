"""Comprehensive Benchmark Fixture Generator.

Synthesizes high-fidelity academic exam pages (Hindi/Devanagari, Organic Chemistry,
Circuit Physics) and composites realistic watermarks varying opacity, angle, color,
tiling, JPEG compression, and DPI (150, 200, 300).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BenchmarkFixture:
    fixture_id: str
    description: str
    dpi: int
    clean_bgr: np.ndarray
    watermarked_bgr: np.ndarray
    wm_mask_gt: np.ndarray       # Ground-truth boolean mask of watermark
    fg_mask_gt: np.ndarray       # Ground-truth boolean mask of authentic text/diagrams
    metadata: Dict[str, object]


def draw_benzene_ring(img: np.ndarray, center: Tuple[int, int], radius: int = 45, line_width: int = 2) -> None:
    """Draw a clean chemical benzene ring with alternating double bonds and resonance circle."""
    cx, cy = center
    pts = []
    for k in range(6):
        angle = k * np.pi / 3.0
        x = int(cx + radius * np.cos(angle))
        y = int(cy + radius * np.sin(angle))
        pts.append((x, y))

    # Outer hexagon
    for i in range(6):
        cv2.line(img, pts[i], pts[(i + 1) % 6], (20, 20, 20), line_width, cv2.LINE_AA)

    # Inner resonance ring or alternating double bonds
    inner_r = int(radius * 0.65)
    cv2.circle(img, (cx, cy), inner_r, (20, 20, 20), max(1, line_width - 1), cv2.LINE_AA)

    # Substituent bonds
    top_p = pts[4]  # Top vertex
    cv2.line(img, top_p, (top_p[0], top_p[1] - 25), (20, 20, 20), line_width, cv2.LINE_AA)
    cv2.putText(
        img,
        "OH",
        (top_p[0] - 12, top_p[1] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6 * (radius / 45.0),
        (20, 20, 20),
        line_width,
        cv2.LINE_AA,
    )


def draw_circuit_diagram(img: np.ndarray, top_left: Tuple[int, int], scale: float = 1.0) -> None:
    """Draw an electrical RLC circuit diagram with resistor, capacitor, and voltage source."""
    x0, y0 = top_left
    lw = max(1, int(2 * scale))

    # Voltage source (circle with sine wave)
    cv2.circle(img, (x0 + int(40 * scale), y0 + int(80 * scale)), int(25 * scale), (20, 20, 20), lw, cv2.LINE_AA)
    # AC sine wave inside circle
    sine_pts = []
    for deg in range(-15, 16):
        rad = deg / 15.0 * np.pi
        sx = int(x0 + 40 * scale + deg)
        sy = int(y0 + 80 * scale - 12 * scale * np.sin(rad))
        sine_pts.append((sx, sy))
    for i in range(len(sine_pts) - 1):
        cv2.line(img, sine_pts[i], sine_pts[i + 1], (20, 20, 20), lw, cv2.LINE_AA)

    # Wire to resistor
    cv2.line(img, (x0 + int(40 * scale), y0 + int(55 * scale)), (x0 + int(40 * scale), y0 + int(20 * scale)), (20, 20, 20), lw)
    cv2.line(img, (x0 + int(40 * scale), y0 + int(20 * scale)), (x0 + int(110 * scale), y0 + int(20 * scale)), (20, 20, 20), lw)

    # Resistor zigzag
    rz = [(0, 0), (8, -12), (16, 12), (24, -12), (32, 12), (40, -12), (48, 12), (56, 0)]
    for i in range(len(rz) - 1):
        p1 = (x0 + int((110 + rz[i][0]) * scale), y0 + int((20 + rz[i][1]) * scale))
        p2 = (x0 + int((110 + rz[i + 1][0]) * scale), y0 + int((20 + rz[i + 1][1]) * scale))
        cv2.line(img, p1, p2, (20, 20, 20), lw, cv2.LINE_AA)

    # Wire from resistor to capacitor
    cv2.line(img, (x0 + int(166 * scale), y0 + int(20 * scale)), (x0 + int(230 * scale), y0 + int(20 * scale)), (20, 20, 20), lw)

    # Capacitor parallel plates
    cv2.line(img, (x0 + int(230 * scale), y0 + int(5 * scale)), (x0 + int(230 * scale), y0 + int(35 * scale)), (20, 20, 20), lw + 1)
    cv2.line(img, (x0 + int(240 * scale), y0 + int(5 * scale)), (x0 + int(240 * scale), y0 + int(35 * scale)), (20, 20, 20), lw + 1)

    # Return wire
    cv2.line(img, (x0 + int(240 * scale), y0 + int(20 * scale)), (x0 + int(280 * scale), y0 + int(20 * scale)), (20, 20, 20), lw)
    cv2.line(img, (x0 + int(280 * scale), y0 + int(20 * scale)), (x0 + int(280 * scale), y0 + int(140 * scale)), (20, 20, 20), lw)
    cv2.line(img, (x0 + int(280 * scale), y0 + int(140 * scale)), (x0 + int(40 * scale), y0 + int(140 * scale)), (20, 20, 20), lw)
    cv2.line(img, (x0 + int(40 * scale), y0 + int(140 * scale)), (x0 + int(40 * scale), y0 + int(105 * scale)), (20, 20, 20), lw)

    # Ground symbol on bottom rail
    gx = x0 + int(160 * scale)
    gy = y0 + int(140 * scale)
    cv2.line(img, (gx, gy), (gx, gy + int(15 * scale)), (20, 20, 20), lw)
    cv2.line(img, (gx - int(15 * scale), gy + int(15 * scale)), (gx + int(15 * scale), gy + int(15 * scale)), (20, 20, 20), lw)
    cv2.line(img, (gx - int(10 * scale), gy + int(20 * scale)), (gx + int(10 * scale), gy + int(20 * scale)), (20, 20, 20), lw)
    cv2.line(img, (gx - int(5 * scale), gy + int(25 * scale)), (gx + int(5 * scale), gy + int(25 * scale)), (20, 20, 20), lw)


def create_clean_exam_page(kind: str = "chemistry", dpi: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a realistic examination page with text and scientific illustrations.

    Returns:
        (clean_bgr, fg_mask_gt)
    """
    # 595 x 842 pt (A4)
    w_px = int(595 * dpi / 72.0)
    h_px = int(842 * dpi / 72.0)
    scale = dpi / 150.0

    # Base page paper (light off-white scanned paper tone 250-252)
    img = np.full((h_px, w_px, 3), 251, dtype=np.uint8)

    # Draw header banner
    cv2.putText(
        img,
        "NATIONAL ENTRANCE EXAMINATION 2026 — PAPER 1",
        (int(40 * scale), int(45 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * scale,
        (40, 40, 40),
        max(1, int(1.5 * scale)),
        cv2.LINE_AA,
    )
    cv2.line(
        img,
        (int(40 * scale), int(58 * scale)),
        (w_px - int(40 * scale), int(58 * scale)),
        (100, 100, 100),
        max(1, int(1 * scale)),
    )

    # Two column layout
    mid_x = w_px // 2
    cv2.line(
        img,
        (mid_x, int(70 * scale)),
        (mid_x, h_px - int(70 * scale)),
        (180, 180, 180),
        max(1, int(1 * scale)),
    )

    if kind == "chemistry":
        # Left column: English
        cv2.putText(
            img,
            "SECTION A: ORGANIC CHEMISTRY",
            (int(40 * scale), int(90 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale,
            (20, 20, 20),
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Q.1 Identify intermediate X in the reaction:",
            (int(40 * scale), int(120 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 * scale,
            (20, 20, 20),
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )

        draw_benzene_ring(img, (int(140 * scale), int(220 * scale)), radius=int(35 * scale), line_width=max(1, int(2 * scale)))

        # Reaction arrow
        cv2.arrowedLine(
            img,
            (int(200 * scale), int(220 * scale)),
            (int(320 * scale), int(220 * scale)),
            (20, 20, 20),
            max(1, int(2 * scale)),
            tipLength=0.25,
        )
        cv2.putText(
            img,
            "HNO3 / H2SO4",
            (int(215 * scale), int(205 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4 * scale,
            (20, 20, 20),
            max(1, int(1.2 * scale)),
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Product [X]",
            (int(340 * scale), int(225 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale,
            (20, 20, 20),
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )

        # Right column: Hindi / Devanagari translation using PyMuPDF overlay
        cv2.putText(
            img,
            "KHAND A: CARBONIC RASAYAN",
            (mid_x + int(40 * scale), int(90 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale,
            (20, 20, 20),
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Prashna 1: Abhikriya mein X ko pehchaniye:",
            (mid_x + int(40 * scale), int(120 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 * scale,
            (20, 20, 20),
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )
        draw_benzene_ring(img, (mid_x + int(140 * scale), int(220 * scale)), radius=int(35 * scale), line_width=max(1, int(2 * scale)))

    elif kind == "physics":
        # Physics with circuit
        cv2.putText(
            img,
            "SECTION B: PHYSICS (CIRCUITS)",
            (int(40 * scale), int(90 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale,
            (20, 20, 20),
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "Q.2 Find the resonance frequency of the circuit:",
            (int(40 * scale), int(120 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 * scale,
            (20, 20, 20),
            max(1, int(1.5 * scale)),
            cv2.LINE_AA,
        )
        draw_circuit_diagram(img, (int(60 * scale), int(160 * scale)), scale=scale * 0.9)

        # Multiple choice options
        options = ["(A) 50.0 Hz", "(B) 100.0 Hz", "(C) 159.2 Hz", "(D) 314.1 Hz"]
        for idx, opt in enumerate(options):
            cv2.putText(
                img,
                opt,
                (int(50 * scale), int((360 + idx * 35) * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48 * scale,
                (20, 20, 20),
                max(1, int(1.2 * scale)),
                cv2.LINE_AA,
            )

    # Footer
    cv2.line(
        img,
        (int(40 * scale), h_px - int(50 * scale)),
        (w_px - int(40 * scale), h_px - int(50 * scale)),
        (180, 180, 180),
        max(1, int(1 * scale)),
    )
    cv2.putText(
        img,
        "PAGE 1 OF 1 — CONFIDENTIAL",
        (mid_x - int(80 * scale), h_px - int(25 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45 * scale,
        (80, 80, 80),
        max(1, int(1.2 * scale)),
        cv2.LINE_AA,
    )

    # Derive ground-truth foreground mask (ink strokes < 200)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fg_mask_gt = gray < 210

    return img, fg_mask_gt


def composite_watermark(
    clean_bgr: np.ndarray,
    text: str = "SAMPLE WATERMARK",
    opacity: float = 0.35,
    angle_deg: float = -35.0,
    color_bgr: Tuple[int, int, int] = (120, 120, 120),
    tiled: bool = False,
    banner: bool = False,
    jpeg_quality: Optional[int] = 75,
) -> Tuple[np.ndarray, np.ndarray]:
    """Composite a synthetic watermark onto a clean page.

    Returns:
        (watermarked_bgr, wm_mask_gt)
    """
    h, w = clean_bgr.shape[:2]
    np.zeros((h, w, 3), dtype=np.uint8)
    alpha_layer = np.zeros((h, w), dtype=np.float32)

    font_scale = max(1.0, w / 700.0)
    thickness = max(2, int(3 * font_scale))

    if not tiled:
        # Centered diagonal text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cx, cy = w // 2, h // 2

        # Draw onto rotation canvas
        rot_mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        temp_img = np.zeros((h, w), dtype=np.uint8)
        cv2.putText(
            temp_img,
            text,
            (cx - tw // 2, cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            255,
            thickness,
            cv2.LINE_AA,
        )
        rotated_alpha = cv2.warpAffine(temp_img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        alpha_layer = (rotated_alpha.astype(np.float32) / 255.0) * opacity
    else:
        # Tiled repeat pattern across page
        step_x = int(w / 3.0)
        step_y = int(h / 4.0)
        temp_img = np.zeros((h, w), dtype=np.uint8)
        for y in range(step_y // 2, h, step_y):
            for x in range(step_x // 4, w, step_x):
                cv2.putText(
                    temp_img,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.75,
                    255,
                    max(1, thickness - 1),
                    cv2.LINE_AA,
                )
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
        rotated_alpha = cv2.warpAffine(temp_img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
        alpha_layer = (rotated_alpha.astype(np.float32) / 255.0) * opacity

    if banner:
        # Footer banner: e.g. "Scanned with CamScanner"
        banner_text = "Scanned with CamScanner"
        b_scale = max(0.6, font_scale * 0.5)
        (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, b_scale, 2)
        bx = w - bw - int(30 * (w / 1000.0))
        by = h - int(20 * (h / 1400.0))
        banner_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.putText(
            banner_mask,
            banner_text,
            (bx, by),
            cv2.FONT_HERSHEY_SIMPLEX,
            b_scale,
            255,
            2,
            cv2.LINE_AA,
        )
        alpha_layer = np.maximum(alpha_layer, (banner_mask.astype(np.float32) / 255.0) * 0.65)

    # Ground-truth watermark mask: where alpha > 0.05
    wm_mask_gt = alpha_layer > 0.03

    # Alpha blending: I = (1 - alpha) * J + alpha * W
    alpha_3d = alpha_layer[:, :, None]
    wm_color_arr = np.full_like(clean_bgr, color_bgr)
    composite = (1.0 - alpha_3d) * clean_bgr.astype(np.float32) + alpha_3d * wm_color_arr.astype(np.float32)
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    # Optional JPEG degradation to simulate realistic compression artifacts
    if jpeg_quality is not None:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, enc = cv2.imencode(".jpg", composite, encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is not None:
            composite = dec

    return composite, wm_mask_gt.astype(np.uint8)


def generate_benchmark_suite(quick: bool = True) -> List[BenchmarkFixture]:
    """Generate the standardized test suite covering varied scenarios."""
    fixtures: List[BenchmarkFixture] = []

    # Matrix configurations:
    # (id, kind, dpi, text, opacity, angle, color_bgr, tiled, banner, jpeg_q)
    configs = [
        # 1. Standard diagonal gray stamp at 150 DPI
        ("BM_01_diag_gray_150", "chemistry", 150, "CONFIDENTIAL DRAFT", 0.35, -45.0, (140, 140, 140), False, False, 75),
        # 2. Low-opacity subtle watermark at 200 DPI
        ("BM_02_subtle_200", "chemistry", 200, "EXAM PREVIEW ONLY", 0.18, -30.0, (160, 160, 160), False, False, 85),
        # 3. Heavy dark watermark at 200 DPI
        ("BM_03_heavy_dark_200", "physics", 200, "TEST COPY DO NOT COPY", 0.55, -40.0, (80, 80, 80), False, False, 75),
        # 4. Colored stamp (red/pink) overlapping circuits
        ("BM_04_red_stamp_200", "physics", 200, "INSTITUTE VERIFIED", 0.40, 0.0, (50, 50, 220), False, False, 80),
        # 5. Tiled repeating background grid
        ("BM_05_tiled_diag_150", "chemistry", 150, "PIRACY SCAN", 0.25, -35.0, (150, 150, 150), True, False, 70),
        # 6. CamScanner banner in footer margin
        ("BM_06_camscanner_banner_200", "physics", 200, "COPIED", 0.20, -45.0, (160, 160, 160), False, True, 75),
    ]

    if not quick:
        # Full 12-case matrix adding 300 DPI high-res, low-quality JPEG 60, and blue watermarks
        configs.extend([
            ("BM_07_hires_300_chem", "chemistry", 300, "TOP SECRET EXAM", 0.30, -45.0, (130, 130, 130), False, False, 85),
            ("BM_08_hires_300_circuit", "physics", 300, "SAMPLE CIRCUIT", 0.45, -30.0, (100, 100, 100), False, False, 85),
            ("BM_09_lowq_jpeg60_150", "chemistry", 150, "INTERNET COPY", 0.40, -45.0, (120, 120, 120), False, False, 60),
            ("BM_10_blue_stamp_200", "chemistry", 200, "SAMPLE COPY", 0.35, -20.0, (220, 100, 40), False, False, 75),
            ("BM_11_tiled_dense_200", "physics", 200, "@TELEGRAM_CHANNEL", 0.30, -45.0, (140, 140, 140), True, True, 75),
            ("BM_12_very_faint_150", "chemistry", 150, "ORIGINAL DRAFT", 0.15, -45.0, (180, 180, 180), False, False, 85),
        ])

    for cfg in configs:
        f_id, kind, dpi, txt, opac, ang, col, tiled, ban, jq = cfg
        clean, fg_mask = create_clean_exam_page(kind=kind, dpi=dpi)
        wm_img, wm_mask = composite_watermark(
            clean,
            text=txt,
            opacity=opac,
            angle_deg=ang,
            color_bgr=col,
            tiled=tiled,
            banner=ban,
            jpeg_quality=jq,
        )
        fixtures.append(
            BenchmarkFixture(
                fixture_id=f_id,
                description=f"{kind} {dpi}dpi opac={opac} ang={ang} deg banner={ban}",
                dpi=dpi,
                clean_bgr=clean,
                watermarked_bgr=wm_img,
                wm_mask_gt=wm_mask,
                fg_mask_gt=fg_mask,
                metadata={
                    "kind": kind,
                    "dpi": dpi,
                    "text": txt,
                    "opacity": opac,
                    "angle_deg": ang,
                    "tiled": tiled,
                    "banner": ban,
                    "jpeg_quality": jq,
                },
            )
        )

    return fixtures
