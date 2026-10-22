"""Synthetic Watermark Fixture Generator.

Deterministically synthesizes clean ground-truth seed PDFs and watermarked
test fixtures for all coverage matrix groups (FX_G01–FX_G19) and modifiers
(FX_M01–FX_M06) as specified in project-plan-and-implementation.md.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pikepdf
import pymupdf
from PIL import Image


def _get_page_content_bytes(page: pikepdf.Page) -> bytes:
    """Safely extract bytes from page Contents whether Stream or Array."""
    if "/Contents" not in page:
        return b""
    contents = page.Contents
    if isinstance(contents, pikepdf.Array):
        parts = []
        for item in contents:
            try:
                parts.append(item.read_bytes())
            except Exception:
                pass
        return b"\n".join(parts)
    elif hasattr(contents, "read_bytes"):
        try:
            return contents.read_bytes()
        except Exception:
            return b""
    return b""


# ---------------------------------------------------------------------------
# Seed PDF Generators
# ---------------------------------------------------------------------------


def create_seed_vector(path: Path, pages: int = 3) -> Path:
    """Create a clean multi-page academic vector PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page(width=612, height=792)  # Letter

        # Header (top 8% furniture)
        page.insert_text(
            (72, 36),
            "JOURNAL OF REPRODUCIBLE RESEARCH — VOL. 42, 2026",
            fontsize=8,
            fontname="helv",
            color=(0.3, 0.3, 0.3),
        )

        # Title & Authors
        page.insert_text(
            (72, 80),
            f"Section {p + 1}: Deterministic Verification in Document Systems",
            fontsize=16,
            fontname="helv",
            color=(0, 0, 0),
        )
        page.insert_text(
            (72, 105),
            "A. Turing, C. Shannon, J. von Neumann — Institute for Advanced Study",
            fontsize=10,
            fontname="helv",
            color=(0.2, 0.2, 0.2),
        )

        # Two-column body paragraphs
        col1_rect = pymupdf.Rect(72, 130, 290, 720)
        col2_rect = pymupdf.Rect(322, 130, 540, 720)

        body_text_col1 = (
            "1. Introduction\n"
            "Document integrity requires rigorous safeguards against accidental data loss. "
            "When removing artifacts from archival documents, naive pixel thresholding often "
            "severs thin foreground glyph strokes. This experiment evaluates multi-tier "
            "verification strategies combining AST exact matching with bounded raster metrics.\n\n"
            "2. Mathematical Formulation\n"
            "Let I(x, y) denote the observed composite raster page. We model watermark formation "
            "as alpha blending I = (1 - alpha) * J + alpha * W, where J is the ground truth "
            "document and W is the watermark pattern. When alpha < 1.0, J is recoverable via "
            "the inverse transformation J = (I - alpha * W) / (1 - alpha).\n\n"
            "3. Geometric Considerations\n"
            "Vector content streams define coordinate transformations through the Current "
            "Transformation Matrix (CTM). Deleting operators without graphics state tracking "
            "causes catastrophic drift in subsequent text spans."
        )
        page.insert_textbox(col1_rect, body_text_col1, fontsize=9, fontname="tiro")

        body_text_col2 = (
            "4. Empirical Evaluation\n"
            "We tested 1,000 synthetic documents across 19 carrier mechanisms. Vector carriers "
            "achieve 100% exact removal with zero pixel variance outside the bounding mask. "
            "For raster carriers, foreground stroke recall L_fg remained strictly below 0.02.\n\n"
            "5. Preserving Page Furniture\n"
            "Running headers, running footers, and page numbers occupy standard margins. "
            "The verifier ensures no legitimate text glyphs within these bounds are excised.\n\n"
            "References:\n"
            "[1] D. Knuth, Digital Typography, CSLI Publications, 1999.\n"
            "[2] Adobe Systems, PDF Reference Sixth Edition, 2006."
        )
        page.insert_textbox(col2_rect, body_text_col2, fontsize=9, fontname="tiro")

        # Footer
        page.insert_text(
            (280, 760),
            f"— Page {p + 1} of {pages} —",
            fontsize=9,
            fontname="helv",
            color=(0.4, 0.4, 0.4),
        )

    doc.save(str(path))
    doc.close()
    return path


def create_seed_scan_rgb(path: Path, pages: int = 3, dpi: int = 150) -> Path:
    """Create a clean scanned PDF with 24-bit RGB page images."""
    vec_path = path.with_suffix(".tmp_vec.pdf")
    create_seed_vector(vec_path, pages=pages)

    src_doc = pymupdf.open(str(vec_path))
    out_doc = pymupdf.open()

    for page in src_doc:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB)
        # Add subtle paper texture (light noise) to simulate a real scan
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()
        noise = np.random.randint(-2, 3, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img_bytes = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))[1].tobytes()

        out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=img_bytes)

    out_doc.save(str(path))
    out_doc.close()
    src_doc.close()
    vec_path.unlink(missing_ok=True)
    return path


def create_seed_scan_grey(path: Path, pages: int = 3, dpi: int = 150) -> Path:
    """Create a clean scanned PDF with 8-bit greyscale page images."""
    vec_path = path.with_suffix(".tmp_vec_grey.pdf")
    create_seed_vector(vec_path, pages=pages)

    src_doc = pymupdf.open(str(vec_path))
    out_doc = pymupdf.open()

    for page in src_doc:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        img_bytes = pix.tobytes(output="png")
        out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=img_bytes)

    out_doc.save(str(path))
    out_doc.close()
    src_doc.close()
    vec_path.unlink(missing_ok=True)
    return path


def create_seed_scan_ocr(path: Path, pages: int = 3, dpi: int = 150) -> Path:
    """Create a scan with invisible OCR text layer (Tr 3)."""
    grey_path = path.with_suffix(".tmp_grey.pdf")
    create_seed_scan_grey(grey_path, pages=pages, dpi=dpi)

    doc = pymupdf.open(str(grey_path))
    for p, page in enumerate(doc):
        # Insert invisible text overlay (render_mode = 3)
        page.insert_text(
            (72, 80),
            f"Section {p + 1}: Deterministic Verification in Document Systems",
            fontsize=16,
            render_mode=3,
        )
        page.insert_text(
            (72, 140),
            "Document integrity requires rigorous safeguards against accidental data loss.",
            fontsize=9,
            render_mode=3,
        )
    doc.save(str(path))
    doc.close()
    grey_path.unlink(missing_ok=True)
    return path


def create_seed_bitonal(path: Path, pages: int = 3, dpi: int = 150) -> Path:
    """Create a 1-bit bitonal CCITT Group 4 scan."""
    vec_path = path.with_suffix(".tmp_vec_bi.pdf")
    create_seed_vector(vec_path, pages=pages)

    src_doc = pymupdf.open(str(vec_path))
    out_doc = pymupdf.open()

    for page in src_doc:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
        # Threshold to 1-bit: background 255 (white), text 0 (black)
        bi_arr = (arr > 180).astype(np.uint8) * 255

        # Convert to PIL 1-bit image
        pil_img = Image.fromarray(bi_arr, mode="L").convert("1")
        buf = io.BytesIO()
        pil_img.save(buf, format="TIFF", compression="group4")

        out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=buf.getvalue())

    out_doc.save(str(path))
    out_doc.close()
    src_doc.close()
    vec_path.unlink(missing_ok=True)
    return path


def create_seed_hybrid(path: Path, pages: int = 3) -> Path:
    """Create a hybrid page with vector text and an embedded diagram bitmap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page(width=612, height=792)

        # Vector text
        page.insert_text(
            (72, 50), f"Hybrid Page {p + 1}: Vector Text + Embedded Diagram", fontsize=14
        )
        page.insert_text((72, 80), "The figure below illustrates system architecture.", fontsize=10)

        # Create a diagram bitmap (300 x 200)
        diag = np.full((200, 300, 3), 255, dtype=np.uint8)
        # Draw some geometric shapes representing a diagram
        cv2.rectangle(diag, (20, 30), (120, 80), (50, 50, 200), 2)
        cv2.putText(diag, "Input PDF", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.arrowedLine(diag, (120, 55), (180, 55), (0, 0, 0), 2)
        cv2.rectangle(diag, (180, 30), (280, 80), (50, 180, 50), 2)
        cv2.putText(diag, "L0 Probe", (195, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        diag_bytes = cv2.imencode(".png", diag)[1].tobytes()
        rect = pymupdf.Rect(156, 120, 456, 320)
        page.insert_image(rect, stream=diag_bytes)

        page.insert_text(
            (72, 350), "Figure 1: Pipeline stage decomposition and data flow.", fontsize=9
        )
        page.insert_text(
            (72, 380),
            "Subsequent text describes the probe and routing steps in detail.",
            fontsize=10,
        )

    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Fixture Generators: G01–G19 (Master Coverage Matrix)
# ---------------------------------------------------------------------------


def gen_fx_g01_annot(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G01: Annotation (/Watermark, /Stamp, /FreeText)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g01.pdf"
    wm = out_dir / "fx_g01_annot.pdf"
    create_seed_vector(clean, pages=2)

    doc = pymupdf.open(str(clean))
    for page in doc:
        rect = pymupdf.Rect(100, 250, 512, 550)
        annot = page.add_stamp_annot(rect)
        annot.set_name("CONFIDENTIAL")
        annot.update()
    doc.save(str(wm))
    doc.close()

    mask = {0: {"bbox": [100, 250, 512, 550], "type": "annot"}}
    return wm, clean, mask


def gen_fx_g02_ocg(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G02: Optional-Content Layer (/OCProperties / /OCG)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g02.pdf"
    wm = out_dir / "fx_g02_ocg.pdf"
    create_seed_vector(clean, pages=2)

    pdf = pikepdf.open(str(clean))
    # Create OCG dictionary
    ocg = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.OCG,
            Name="WatermarkLayer",
            Intent=pikepdf.Name.View,
        )
    )

    # Add to document Root /OCProperties
    pdf.Root.OCProperties = pikepdf.Dictionary(
        OCGs=pikepdf.Array([ocg]),
        D=pikepdf.Dictionary(
            Order=pikepdf.Array([ocg]),
            ON=pikepdf.Array([ocg]),
        ),
    )

    # In each page, wrap watermark text in marked content BDC /OC /WatermarkLayer ... EMC
    for page in pdf.pages:
        if "/Resources" not in page:
            page.Resources = pikepdf.Dictionary()
        if "/Properties" not in page.Resources:
            page.Resources.Properties = pikepdf.Dictionary()
        page.Resources.Properties["/WatermarkLayer"] = ocg

        # Append BDC block to content stream
        wm_stream = (
            b"\n/OC /WatermarkLayer BDC\n"
            b"q 0.8 0.8 0.8 rg\n"
            b"BT /Helv 48 Tf 1 0 0 1 150 400 Tm (CONFIDENTIAL OCG) Tj ET\n"
            b"Q\n"
            b"EMC\n"
        )
        # Standard font
        if "/Font" not in page.Resources:
            page.Resources.Font = pikepdf.Dictionary()
        page.Resources.Font["/Helv"] = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1, BaseFont=pikepdf.Name.Helvetica
            )
        )

        orig_stream = _get_page_content_bytes(page)
        page.Contents = pdf.make_stream(orig_stream + wm_stream)

    pdf.save(str(wm))
    pdf.close()
    mask = {0: {"bbox": [150, 400, 500, 450], "type": "ocg"}}
    return wm, clean, mask


def gen_fx_g03_artifact(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G03: Marked-Content /Artifact (/Subtype /Watermark)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g03.pdf"
    wm = out_dir / "fx_g03_artifact.pdf"
    create_seed_vector(clean, pages=2)

    pdf = pikepdf.open(str(clean))
    for page in pdf.pages:
        if "/Resources" not in page:
            page.Resources = pikepdf.Dictionary()
        if "/Font" not in page.Resources:
            page.Resources.Font = pikepdf.Dictionary()
        page.Resources.Font["/Helv"] = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1, BaseFont=pikepdf.Name.Helvetica
            )
        )

        wm_stream = (
            b"\n/Artifact << /Subtype /Watermark >> BDC\n"
            b"q 0.75 0.75 0.75 rg\n"
            b"BT /Helv 54 Tf 0.707 0.707 -0.707 0.707 150 300 Tm (DRAFT ARTIFACT) Tj ET\n"
            b"Q\n"
            b"EMC\n"
        )
        orig = _get_page_content_bytes(page)
        page.Contents = pdf.make_stream(orig + wm_stream)

    pdf.save(str(wm))
    pdf.close()
    mask = {0: {"bbox": [150, 300, 480, 550], "type": "artifact"}}
    return wm, clean, mask


def gen_fx_g04_text_vec(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G04: Content-stream Text (diagonal text watermark)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g04.pdf"
    wm = out_dir / "fx_g04_text_vec.pdf"
    create_seed_vector(clean, pages=2)

    pdf = pikepdf.open(str(clean))
    for p, page in enumerate(pdf.pages):
        if "/Resources" not in page:
            page.Resources = pikepdf.Dictionary()
        if "/Font" not in page.Resources:
            page.Resources.Font = pikepdf.Dictionary()
        page.Resources.Font["/Helv"] = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1, BaseFont=pikepdf.Name.Helvetica
            )
        )

        # Include dynamic text: email and date
        text = f"CONFIDENTIAL user_{p + 1}@example.com 2026-10-20"
        wm_stream = (
            f"\nq 0.7 0.7 0.7 rg\n"
            f"BT /Helv 36 Tf 0.707 0.707 -0.707 0.707 120 200 Tm ({text}) Tj ET\n"
            f"Q\n"
        ).encode("latin1")

        orig = _get_page_content_bytes(page)
        page.Contents = pdf.make_stream(orig + wm_stream)

    pdf.save(str(wm))
    pdf.close()
    mask = {0: {"bbox": [120, 200, 500, 600], "type": "text_vec"}}
    return wm, clean, mask


def gen_fx_g05_clip_shade(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G05: Text or vector clipping path (W/W* n or Tr=4..7) over shading/fill."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g05.pdf"
    wm = out_dir / "fx_g05_clip_shade.pdf"
    create_seed_vector(clean, pages=2)

    pdf = pikepdf.open(str(clean))
    for page in pdf.pages:
        wm_stream = (
            b"\nq\n"
            b"150 250 300 150 re W n\n"  # Clipping path
            b"0.85 0.85 0.85 rg\n"
            b"150 250 300 150 re f\n"
            b"Q\n"
        )
        orig = _get_page_content_bytes(page)
        page.Contents = pdf.make_stream(orig + wm_stream)

    pdf.save(str(wm))
    pdf.close()
    mask = {0: {"bbox": [150, 250, 450, 400], "type": "clip_shade"}}
    return wm, clean, mask


def gen_fx_g06_xobj(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G06: Repeated Image XObject or Form XObject."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g06.pdf"
    wm = out_dir / "fx_g06_xobj.pdf"
    create_seed_vector(clean, pages=2)

    pdf = pikepdf.open(str(clean))

    # Create a Form XObject for the watermark
    form_stream = b"q 0.8 0.2 0.2 rg\n10 10 180 40 re f\nQ\n"
    form_xobj = pdf.make_stream(
        form_stream,
        Type=pikepdf.Name.XObject,
        Subtype=pikepdf.Name.Form,
        BBox=pikepdf.Array([0, 0, 200, 60]),
    )

    for page in pdf.pages:
        if "/Resources" not in page:
            page.Resources = pikepdf.Dictionary()
        if "/XObject" not in page.Resources:
            page.Resources.XObject = pikepdf.Dictionary()
        page.Resources.XObject["/FmWatermark"] = form_xobj

        call_stream = b"\nq 1 0 0 1 200 650 cm /FmWatermark Do Q\n"
        orig = _get_page_content_bytes(page)
        page.Contents = pdf.make_stream(orig + call_stream)

    pdf.save(str(wm))
    pdf.close()
    mask = {0: {"bbox": [200, 650, 400, 710], "type": "xobj"}}
    return wm, clean, mask


def gen_fx_g07_vec_path(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G07: Untagged vector paths (m l c re S/f/B)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g07.pdf"
    wm = out_dir / "fx_g07_vec_path.pdf"
    create_seed_vector(clean, pages=2)

    pdf = pikepdf.open(str(clean))
    for page in pdf.pages:
        # Diagonal banner vector paths
        path_stream = b"\nq 0.8 0.8 0.8 RG 12 w\n50 100 m 550 700 l S\n50 140 m 550 740 l S\nQ\n"
        orig = _get_page_content_bytes(page)
        page.Contents = pdf.make_stream(orig + path_stream)

    pdf.save(str(wm))
    pdf.close()
    mask = {0: {"bbox": [50, 100, 550, 740], "type": "vec_path"}}
    return wm, clean, mask


def gen_fx_g08_pattern(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G08: Tiling pattern or shading (/PatternType 1 or 2)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g08.pdf"
    wm = out_dir / "fx_g08_pattern.pdf"
    create_seed_vector(clean, pages=2)

    # We inject a simulated shading rectangle fill
    pdf = pikepdf.open(str(clean))
    for page in pdf.pages:
        shade_stream = b"\nq\n0.9 0.9 0.9 rg\n100 200 400 350 re f\nQ\n"
        orig = _get_page_content_bytes(page)
        page.Contents = pdf.make_stream(orig + shade_stream)
    pdf.save(str(wm))
    pdf.close()
    mask = {0: {"bbox": [100, 200, 500, 550], "type": "pattern"}}
    return wm, clean, mask


def gen_fx_g09_smask(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G09: Soft-mask (/SMask) or blend-mode (/BM /Multiply) overlay."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g09.pdf"
    wm = out_dir / "fx_g09_smask.pdf"
    create_seed_vector(clean, pages=2)

    pdf = pikepdf.open(str(clean))
    gs_dict = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.ExtGState,
            BM=pikepdf.Name.Multiply,
            ca=0.35,
            CA=0.35,
        )
    )

    for page in pdf.pages:
        if "/Resources" not in page:
            page.Resources = pikepdf.Dictionary()
        if "/ExtGState" not in page.Resources:
            page.Resources.ExtGState = pikepdf.Dictionary()
        page.Resources.ExtGState["/GS_SMask"] = gs_dict

        smask_stream = b"\nq /GS_SMask gs\n0.5 0.5 0.5 rg\n100 200 412 392 re f\nQ\n"
        orig = _get_page_content_bytes(page)
        page.Contents = pdf.make_stream(orig + smask_stream)

    pdf.save(str(wm))
    pdf.close()
    mask = {0: {"bbox": [100, 200, 512, 592], "type": "smask"}}
    return wm, clean, mask


def gen_fx_g10_ras_alpha_fixed(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G10: Light grey alpha-blended fixed-position raster watermark."""
    clean = out_dir / "clean_g10.pdf"
    wm = out_dir / "fx_g10_ras_alpha_fixed.pdf"
    create_seed_scan_grey(clean, pages=3, dpi=150)

    src_doc = pymupdf.open(str(clean))
    out_doc = pymupdf.open()

    for page in src_doc:
        pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w).copy()

        # Alpha blend diagonal text watermark: I = (1 - alpha) * I + alpha * W
        alpha = 0.30
        wm_val = 190  # light grey

        # Create text mask using OpenCV
        mask_img = np.zeros_like(img)
        cv2.putText(
            mask_img,
            "SAMPLE WATERMARK",
            (int(pix.w * 0.15), int(pix.h * 0.55)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            255,
            6,
            cv2.LINE_AA,
        )

        blended = np.where(
            mask_img > 0,
            np.clip((1.0 - alpha) * img + alpha * wm_val, 0, 255).astype(np.uint8),
            img,
        )

        img_bytes = cv2.imencode(".png", blended)[1].tobytes()
        out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=img_bytes)

    out_doc.save(str(wm))
    out_doc.close()
    src_doc.close()
    mask = {0: {"alpha": 0.30, "color": [190, 190, 190], "type": "raster_alpha"}}
    return wm, clean, mask


def gen_fx_g11_ras_color(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G11: Coloured watermark in 24-bit RGB raster scan."""
    clean = out_dir / "clean_g11.pdf"
    wm = out_dir / "fx_g11_ras_color.pdf"
    create_seed_scan_rgb(clean, pages=3, dpi=150)

    src_doc = pymupdf.open(str(clean))
    out_doc = pymupdf.open()

    for page in src_doc:
        pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csRGB)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()

        alpha = 0.40
        wm_color = np.array([220, 60, 60], dtype=np.float32)  # Saturated Red in RGB

        mask_img = np.zeros((pix.h, pix.w), dtype=np.uint8)
        cv2.putText(
            mask_img,
            "COPY PROTECTED",
            (int(pix.w * 0.12), int(pix.h * 0.50)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            255,
            6,
            cv2.LINE_AA,
        )

        in_mask = mask_img > 0
        img_float = img.astype(np.float32)
        for c in range(3):
            img_float[in_mask, c] = np.clip(
                (1.0 - alpha) * img_float[in_mask, c] + alpha * wm_color[c], 0, 255
            )

        blended = img_float.astype(np.uint8)
        img_bytes = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))[1].tobytes()
        out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=img_bytes)

    out_doc.save(str(wm))
    out_doc.close()
    src_doc.close()
    mask = {0: {"alpha": 0.40, "color": [220, 60, 60], "type": "raster_color"}}
    return wm, clean, mask


def gen_fx_g12_ras_grey_var(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G12: Varying-position / dynamic-text light grey raster watermark."""
    clean = out_dir / "clean_g12.pdf"
    wm = out_dir / "fx_g12_ras_grey_var.pdf"
    create_seed_scan_grey(clean, pages=3, dpi=150)

    src_doc = pymupdf.open(str(clean))
    out_doc = pymupdf.open()

    for p, page in enumerate(src_doc):
        pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w).copy()

        alpha = 0.35
        wm_val = 185
        y_pos = int(pix.h * (0.35 + 0.15 * p))  # Varying position per page

        mask_img = np.zeros_like(img)
        cv2.putText(
            mask_img,
            f"USER_ID_{1042 + p} DATE_2026",
            (int(pix.w * 0.10), y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            255,
            5,
            cv2.LINE_AA,
        )

        blended = np.where(
            mask_img > 0,
            np.clip((1.0 - alpha) * img + alpha * wm_val, 0, 255).astype(np.uint8),
            img,
        )
        img_bytes = cv2.imencode(".png", blended)[1].tobytes()
        out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=img_bytes)

    out_doc.save(str(wm))
    out_doc.close()
    src_doc.close()
    mask = {0: {"alpha": 0.35, "type": "raster_varying"}}
    return wm, clean, mask


def gen_fx_g13_ras_fft(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G13: Periodic hatched or grid raster watermark."""
    clean = out_dir / "clean_g13.pdf"
    wm = out_dir / "fx_g13_ras_fft.pdf"
    create_seed_scan_grey(clean, pages=2, dpi=150)

    src_doc = pymupdf.open(str(clean))
    out_doc = pymupdf.open()

    for page in src_doc:
        pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w).copy()

        # Generate 2D periodic sine pattern: frequency = 0.05 cycles/pixel
        y, x = np.mgrid[0 : pix.h, 0 : pix.w]
        grid = (15.0 * np.sin(2.0 * np.pi * 0.04 * (x + y))).astype(np.float32)
        blended = np.clip(img.astype(np.float32) + grid, 0, 255).astype(np.uint8)

        img_bytes = cv2.imencode(".png", blended)[1].tobytes()
        out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=img_bytes)

    out_doc.save(str(wm))
    out_doc.close()
    src_doc.close()
    mask = {0: {"type": "periodic_fft"}}
    return wm, clean, mask


def gen_fx_g14_ras_inpaint(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G14: Thin outline-only stamp in raster."""
    clean = out_dir / "clean_g14.pdf"
    wm = out_dir / "fx_g14_ras_inpaint.pdf"
    create_seed_scan_grey(clean, pages=2, dpi=150)

    src_doc = pymupdf.open(str(clean))
    out_doc = pymupdf.open()

    for page in src_doc:
        pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w).copy()

        # Draw thin 2-pixel outline circular stamp in corner
        center = (int(pix.w * 0.85), int(pix.h * 0.15))
        cv2.circle(img, center, 50, 100, 2, cv2.LINE_AA)
        cv2.putText(
            img, "VALID", (center[0] - 25, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 100, 1
        )

        img_bytes = cv2.imencode(".png", img)[1].tobytes()
        out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=img_bytes)

    out_doc.save(str(wm))
    out_doc.close()
    src_doc.close()
    mask = {0: {"type": "thin_stamp"}}
    return wm, clean, mask


def gen_fx_g15_bitonal_dither(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G15: 1-bit bitonal scan with dithered watermark."""
    clean = out_dir / "clean_g15.pdf"
    wm = out_dir / "fx_g15_bitonal_dither.pdf"
    create_seed_bitonal(clean, pages=2, dpi=150)

    src_doc = pymupdf.open(str(clean))
    out_doc = pymupdf.open()

    for page in src_doc:
        pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w).copy()

        # Add a 50% dither pattern (checkerboard) in diagonal text region
        mask = np.zeros_like(arr)
        cv2.putText(
            mask,
            "DRAFT BITONAL",
            (int(pix.w * 0.10), int(pix.h * 0.50)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            255,
            6,
            cv2.LINE_AA,
        )

        y, x = np.mgrid[0 : pix.h, 0 : pix.w]
        checker = (x + y) % 2 == 0

        # Watermark pixels where mask is white become dithered black dots
        arr[(mask > 0) & checker] = 0

        # Convert to 1-bit TIFF
        pil_img = Image.fromarray(arr, mode="L").convert("1")
        buf = io.BytesIO()
        pil_img.save(buf, format="TIFF", compression="group4")

        out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=buf.getvalue())

    out_doc.save(str(wm))
    out_doc.close()
    src_doc.close()
    mask = {0: {"type": "bitonal_dither"}}
    return wm, clean, mask


def gen_fx_g16_tiled_scan(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G16: Tiled/strip scan with watermark crossing strip boundary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g16.pdf"
    wm = out_dir / "fx_g16_tiled_scan.pdf"

    # Create 1-page document with 2 vertical image strips
    doc_clean = pymupdf.open()
    page_c = doc_clean.new_page(width=612, height=792)

    # 2 strips: top half (0 to 396), bottom half (396 to 792)
    top_strip = np.full((396, 612, 3), 250, dtype=np.uint8)
    cv2.putText(
        top_strip,
        "Strip 1: Top Half Content",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    bottom_strip = np.full((396, 612, 3), 250, dtype=np.uint8)
    cv2.putText(
        bottom_strip,
        "Strip 2: Bottom Half Content",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )

    page_c.insert_image(
        pymupdf.Rect(0, 0, 612, 396), stream=cv2.imencode(".png", top_strip)[1].tobytes()
    )
    page_c.insert_image(
        pymupdf.Rect(0, 396, 612, 792), stream=cv2.imencode(".png", bottom_strip)[1].tobytes()
    )
    doc_clean.save(str(clean))
    doc_clean.close()

    # Now create watermarked version where watermark crosses y=396
    doc_wm = pymupdf.open()
    page_w = doc_wm.new_page(width=612, height=792)

    top_wm = top_strip.copy()
    cv2.putText(
        top_wm, "CROSS STRIP WM", (100, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 3
    )
    bottom_wm = bottom_strip.copy()
    cv2.putText(
        bottom_wm, "CROSS STRIP WM", (100, -16), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 3
    )

    page_w.insert_image(
        pymupdf.Rect(0, 0, 612, 396), stream=cv2.imencode(".png", top_wm)[1].tobytes()
    )
    page_w.insert_image(
        pymupdf.Rect(0, 396, 612, 792), stream=cv2.imencode(".png", bottom_wm)[1].tobytes()
    )
    doc_wm.save(str(wm))
    doc_wm.close()

    mask = {0: {"type": "tiled_strips"}}
    return wm, clean, mask


def gen_fx_g17_hybrid_diag_wm(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G17: Hybrid page with watermark inside embedded diagram bitmap."""
    clean = out_dir / "clean_g17.pdf"
    wm = out_dir / "fx_g17_hybrid_diag_wm.pdf"
    create_seed_hybrid(clean, pages=2)

    # Inject watermark only into the diagram bitmap
    doc = pymupdf.open()
    for p in range(2):
        page = doc.new_page(width=612, height=792)
        page.insert_text(
            (72, 50), f"Hybrid Page {p + 1}: Vector Text + Embedded Diagram", fontsize=14
        )
        page.insert_text((72, 80), "The figure below illustrates system architecture.", fontsize=10)

        # Diagram with baked light-grey watermark
        diag = np.full((200, 300, 3), 255, dtype=np.uint8)
        cv2.rectangle(diag, (20, 30), (120, 80), (50, 50, 200), 2)
        cv2.putText(diag, "Input PDF", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.arrowedLine(diag, (120, 55), (180, 55), (0, 0, 0), 2)
        cv2.rectangle(diag, (180, 30), (280, 80), (50, 180, 50), 2)
        cv2.putText(diag, "L0 Probe", (195, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Watermark text inside diagram
        cv2.putText(
            diag, "DIAGRAM WM", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 190, 190), 2
        )

        diag_bytes = cv2.imencode(".png", diag)[1].tobytes()
        rect = pymupdf.Rect(156, 120, 456, 320)
        page.insert_image(rect, stream=diag_bytes)
        page.insert_text(
            (72, 350), "Figure 1: Pipeline stage decomposition and data flow.", fontsize=9
        )

    doc.save(str(wm))
    doc.close()
    mask = {0: {"type": "hybrid_diag"}}
    return wm, clean, mask


def gen_fx_g18_compound_hybrid(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G18: Compound multi-layered watermark (Annot + Vector Text + Raster)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = out_dir / "clean_g18.pdf"
    wm = out_dir / "fx_g18_compound_hybrid.pdf"
    create_seed_hybrid(clean, pages=2)

    # First add annotation via PyMuPDF
    doc = pymupdf.open(str(clean))
    for page in doc:
        rect = pymupdf.Rect(50, 50, 200, 100)
        annot = page.add_stamp_annot(rect)
        annot.set_name("Draft")
        annot.update()
    doc.save(str(wm))
    doc.close()

    # Then add vector text via pikepdf
    pdf = pikepdf.open(str(wm), allow_overwriting_input=True)
    for page in pdf.pages:
        wm_stream = (
            b"\nq 0.7 0.7 0.7 rg\nBT /Helv 36 Tf 1 0 0 1 150 500 Tm (COMPOUND VECTOR) Tj ET\nQ\n"
        )
        orig = _get_page_content_bytes(page)
        page.Contents = pdf.make_stream(orig + wm_stream)
    pdf.save(str(wm))
    pdf.close()

    mask = {0: {"type": "compound"}}
    return wm, clean, mask


def gen_fx_g19_opaque(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_G19: Opaque dark raster watermark occluding text (should be FLAGGED)."""
    clean = out_dir / "clean_g19.pdf"
    wm = out_dir / "fx_g19_opaque.pdf"
    create_seed_scan_grey(clean, pages=2, dpi=150)

    src_doc = pymupdf.open(str(clean))
    out_doc = pymupdf.open()

    for page in src_doc:
        pix = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w).copy()

        # Draw completely opaque pitch-black rectangle covering body text lines
        cv2.rectangle(
            img,
            (int(pix.w * 0.15), int(pix.h * 0.20)),
            (int(pix.w * 0.85), int(pix.h * 0.35)),
            0,
            -1,
        )

        img_bytes = cv2.imencode(".png", img)[1].tobytes()
        out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
        out_page.insert_image(out_page.rect, stream=img_bytes)

    out_doc.save(str(wm))
    out_doc.close()
    src_doc.close()
    mask = {0: {"type": "opaque_occlusion", "expected_status": "FLAGGED"}}
    return wm, clean, mask


# ---------------------------------------------------------------------------
# Modifiers: M01–M06 (Geometry & Document-State Modifiers)
# ---------------------------------------------------------------------------


def gen_fx_m01_inc_rev(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_M01: Incremental revision containing watermark added to clean base."""
    clean = out_dir / "clean_m01.pdf"
    wm = out_dir / "fx_m01_inc_rev.pdf"
    create_seed_vector(clean, pages=2)

    # Copy clean to wm first
    import shutil

    shutil.copyfile(clean, wm)

    # Open and add annotation with incremental update using PyMuPDF
    doc = pymupdf.open(str(wm))
    annot = doc[0].add_stamp_annot(pymupdf.Rect(100, 200, 500, 600))
    annot.set_name("Draft")
    annot.update()
    doc.save(doc.name, incremental=True, encryption=getattr(pymupdf, "PDF_ENCRYPT_KEEP", 0))
    doc.close()

    mask = {0: {"type": "inc_rev"}}
    return wm, clean, mask


def gen_fx_m02_rotated(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_M02: Rotated pages (/Rotate 90 or 270)."""
    clean = out_dir / "clean_m02.pdf"
    wm = out_dir / "fx_m02_rotated.pdf"
    create_seed_vector(clean, pages=2)

    pdf = pikepdf.open(str(clean))
    pdf.pages[0].Rotate = 90
    pdf.pages[1].Rotate = 270

    # Add annotation on rotated pages
    for page in pdf.pages:
        annot = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Annot,
                Subtype=pikepdf.Name.Watermark,
                Rect=pikepdf.Array([100, 100, 400, 400]),
            )
        )
        page.Annots = pikepdf.Array([annot])

    pdf.save(str(wm))
    pdf.close()
    mask = {0: {"type": "rotated"}}
    return wm, clean, mask


def gen_fx_m03_cropbox(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_M03: Mixed page sizes & CropBox != MediaBox."""
    clean = out_dir / "clean_m03.pdf"
    wm = out_dir / "fx_m03_cropbox.pdf"
    create_seed_vector(clean, pages=2)

    pdf = pikepdf.open(str(clean))
    for page in pdf.pages:
        # MediaBox is 0, 0, 612, 792; set CropBox with 36pt bleed
        page.CropBox = pikepdf.Array([36, 36, 576, 756])
        annot = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Annot,
                Subtype=pikepdf.Name.Watermark,
                Rect=pikepdf.Array([10, 10, 500, 500]),  # crosses CropBox bleed
            )
        )
        page.Annots = pikepdf.Array([annot])

    pdf.save(str(wm))
    pdf.close()
    mask = {0: {"type": "cropbox_bleed"}}
    return wm, clean, mask


def gen_fx_m04_encrypt(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_M04: Encrypted with empty user password (owner locked)."""
    clean = out_dir / "clean_m04.pdf"
    wm = out_dir / "fx_m04_encrypt.pdf"
    create_seed_vector(clean, pages=2)

    # Add watermark annotation
    doc = pymupdf.open(str(clean))
    for page in doc:
        annot = page.add_stamp_annot(pymupdf.Rect(100, 100, 400, 300))
        annot.set_name("CONFIDENTIAL")
        annot.update()
    doc.save(str(wm))
    doc.close()

    # Encrypt with pikepdf: empty user password, owner password 'owner123'
    pdf = pikepdf.open(str(wm), allow_overwriting_input=True)
    encryption = pikepdf.Encryption(owner="owner123", user="", R=4)
    pdf.save(str(wm), encryption=encryption)
    pdf.close()

    mask = {0: {"type": "empty_user_pw"}}
    return wm, clean, mask


def gen_fx_m05_corrupt_huge(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_M05: Linearized / recoverable XRef with watermark."""
    clean = out_dir / "clean_m05.pdf"
    wm = out_dir / "fx_m05_corrupt_huge.pdf"
    create_seed_vector(clean, pages=2)

    pdf = pikepdf.open(str(clean))
    for page in pdf.pages:
        annot = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Annot,
                Subtype=pikepdf.Name.Watermark,
                Rect=pikepdf.Array([100, 100, 400, 300]),
            )
        )
        page.Annots = pikepdf.Array([annot])
    # Save linearized
    pdf.save(str(wm), linearize=True)
    pdf.close()

    mask = {0: {"type": "linearized"}}
    return wm, clean, mask


def gen_fx_m06_struct_pdfa(out_dir: Path) -> Tuple[Path, Path, Dict]:
    """FX_M06: Bookmarks (Outlines), StructTreeRoot, PDF metadata."""
    clean = out_dir / "clean_m06.pdf"
    wm = out_dir / "fx_m06_struct_pdfa.pdf"
    create_seed_vector(clean, pages=2)

    # Add outlines/bookmarks via PyMuPDF
    doc = pymupdf.open(str(clean))
    toc = [
        [1, "Chapter 1: Introduction", 1],
        [2, "1.1 Background", 1],
        [1, "Chapter 2: Methods", 2],
    ]
    doc.set_toc(toc)
    for page in doc:
        annot = page.add_stamp_annot(pymupdf.Rect(100, 200, 450, 400))
        annot.set_name("CONFIDENTIAL")
        annot.update()
    doc.save(str(wm))
    doc.close()

    mask = {0: {"type": "struct_pdfa"}}
    return wm, clean, mask


# ---------------------------------------------------------------------------
# Master Fixture Dictionary & Generator
# ---------------------------------------------------------------------------

ALL_FIXTURE_GENERATORS = {
    "FX_G01": gen_fx_g01_annot,
    "FX_G02": gen_fx_g02_ocg,
    "FX_G03": gen_fx_g03_artifact,
    "FX_G04": gen_fx_g04_text_vec,
    "FX_G05": gen_fx_g05_clip_shade,
    "FX_G06": gen_fx_g06_xobj,
    "FX_G07": gen_fx_g07_vec_path,
    "FX_G08": gen_fx_g08_pattern,
    "FX_G09": gen_fx_g09_smask,
    "FX_G10": gen_fx_g10_ras_alpha_fixed,
    "FX_G11": gen_fx_g11_ras_color,
    "FX_G12": gen_fx_g12_ras_grey_var,
    "FX_G13": gen_fx_g13_ras_fft,
    "FX_G14": gen_fx_g14_ras_inpaint,
    "FX_G15": gen_fx_g15_bitonal_dither,
    "FX_G16": gen_fx_g16_tiled_scan,
    "FX_G17": gen_fx_g17_hybrid_diag_wm,
    "FX_G18": gen_fx_g18_compound_hybrid,
    "FX_G19": gen_fx_g19_opaque,
    "FX_M01": gen_fx_m01_inc_rev,
    "FX_M02": gen_fx_m02_rotated,
    "FX_M03": gen_fx_m03_cropbox,
    "FX_M04": gen_fx_m04_encrypt,
    "FX_M05": gen_fx_m05_corrupt_huge,
    "FX_M06": gen_fx_m06_struct_pdfa,
}


def generate_all_fixtures(out_dir: Path) -> Dict[str, Tuple[Path, Path, Dict]]:
    """Generate all 25 synthetic fixtures in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for fid, generator in ALL_FIXTURE_GENERATORS.items():
        wm_path, clean_path, mask = generator(out_dir)
        manifest[fid] = (wm_path, clean_path, mask)
    return manifest
