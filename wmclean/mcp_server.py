"""Official Model Context Protocol (MCP) Server for WMCLEAN.

Exposes semantic forensic actuators and diagnostics for LLMs, autonomous coding
agents, and MCP client IDEs (Claude Code, Cursor, Windsurf, Cline).
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from mcp.server.mcpserver import MCPServer
except Exception:
    import importlib

    try:
        fastmcp_mod = importlib.import_module("mcp.server.fastmcp")
        MCPServer = getattr(fastmcp_mod, "FastMCP")
    except Exception:
        class MCPServer:  # type: ignore[no-redef]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def tool(self, *args: Any, **kwargs: Any) -> Any:
                return lambda f: f

            def run(self, *args: Any, **kwargs: Any) -> None:
                pass

from wmclean.orchestrator import Orchestrator
from wmclean.probe.document import probe_document
from wmclean.schemas import ExitCode

logger = logging.getLogger(__name__)

# Initialize MCP Server with official metadata
mcp = MCPServer(
    name="Watermark-Remover-MCP",
    instructions="Universal forensic watermark, stamp, seal, and background remover for PDFs, scanned documents, and scientific diagrams.",
)


@mcp.tool()
def remove_watermark_auto(
    file_path: str,
    output_path: Optional[str] = None,
    mode: str = "adaptive",
    preserve_scientific_topology: bool = True,
) -> Dict[str, Any]:
    """Universal watermark, stamp, seal, and background overlay remover for PDFs, scanned documents, scientific diagrams, chemical structures, and photographs.

    Automatically routes between lossless PDF AST layer stripping (Tier 0), deterministic
    inverse alpha matting (Tier 1), topological skeleton stitching with Betti-number loop
    preservation (Tier 2), and masked inpainting (Tier 3) with independent fail-closed structural QA verification.

    Args:
        file_path: Absolute or relative path to the input PDF or document image.
        output_path: Optional destination path for the cleaned PDF. If omitted, generates '<file>_cleaned.pdf'.
        mode: Execution mode: 'fast', 'accurate', 'adaptive' (default), or 'hybrid'.
        preserve_scientific_topology: If True, enforces Betti-number loop invariance (Δβ1=0) on chemical rings and circuit diagrams.

    Returns:
        Structured audit dictionary containing exit_code, status, pages_summary, verified_residual, and latency_ms.
    """
    in_p = Path(file_path).resolve()
    if not in_p.exists():
        return {
            "success": False,
            "status": "ERROR_FILE_NOT_FOUND",
            "exit_code": ExitCode.IO_ERROR.value,
            "error": f"Input file not found: {file_path}",
        }

    if output_path:
        out_p = Path(output_path).resolve()
    else:
        out_p = in_p.with_name(f"{in_p.stem}_cleaned{in_p.suffix}")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_rep:
        rep_p = Path(tmp_rep.name)

    try:
        orch = Orchestrator(
            input_path=str(in_p),
            output_path=str(out_p),
            report_path=str(rep_p),
            mode=mode,
        )
        exit_code = orch.execute()

        report_data: Dict[str, Any] = {}
        if rep_p.exists():
            try:
                report_data = json.loads(rep_p.read_text(encoding="utf-8"))
            except Exception:
                pass

        pages = report_data.get("pages", [])
        counts = report_data.get("counts", {})
        # Map both current schema ('exact', 'verified') and legacy keys ('clean_exact', 'clean_verified')
        cleaned_count = (
            counts.get("exact", 0)
            + counts.get("verified", 0)
            + counts.get("clean_exact", 0)
            + counts.get("clean_verified", 0)
        )

        # Average verified residual
        residuals = [p.get("res", 0.0) for p in pages if p.get("res") is not None]
        avg_residual = float(sum(residuals) / len(residuals)) if residuals else 0.0

        return {
            "success": exit_code == ExitCode.SUCCESS.value,
            "status": "CLEAN" if exit_code == 0 else ("UNVERIFIED" if exit_code == 15 else "PARTIAL"),
            "exit_code": exit_code,
            "mode": mode,
            "input_file": str(in_p),
            "output_file": str(out_p) if out_p.exists() else None,
            "total_pages": len(pages),
            "cleaned_pages": cleaned_count,
            "untouched_pages": counts.get("untouched", 0),
            "unverified_pages": counts.get("unverified", 0),
            "flagged_pages": counts.get("flagged", 0),
            "average_watermark_residual": round(avg_residual, 4),
            "details": report_data.get("perf", {}),
        }
    finally:
        if rep_p.exists():
            try:
                rep_p.unlink()
            except Exception:
                pass


@mcp.tool()
def inspect_watermark_layers(file_path: str) -> Dict[str, Any]:
    """Diagnoses whether a watermark is a removable vector PDF layer (XObject/OCG/Artifact), a semi-transparent alpha blend, or an opaque raster stamp across document pages.

    Args:
        file_path: Path to the input PDF or image.

    Returns:
        Structured layer diagnostics with detected carrier types, estimated alpha, OCR piracy spans, and recommended removal tier.
    """
    in_p = Path(file_path).resolve()
    if not in_p.exists():
        return {"error": f"File not found: {file_path}"}

    try:
        records = probe_document(str(in_p))
    except Exception as exc:
        return {"error": f"Failed to probe document structure: {exc}"}

    page_diagnostics = []
    recommended_tiers = []

    for r in records:
        carriers = []
        if r.annots:
            carriers.append("annot")
        if r.ocgs:
            carriers.append("ocg")
        if r.artifacts:
            carriers.append("artifact")
        if r.text_cand:
            carriers.append("vector_text")
        if r.path_cand:
            carriers.append("vector_path")
        if r.xobj_cand:
            carriers.append("xobject")

        has_vector = bool(carriers)
        has_raster = r.ras_stats.has_raster if r.ras_stats else False

        if has_vector and not has_raster:
            tier = "Tier 0 (Bitwise Lossless Vector AST Surgery)"
        elif has_raster and r.ras_stats and r.ras_stats.est_alpha > 0.05:
            tier = "Tier 1 (Deterministic Inverse Alpha Matting)"
        elif r.kind.value == "vector_diagram":
            tier = "Tier 2 (Topological Skeleton Graph Stitching)"
        else:
            tier = "Tier 1 (Two-Regime Paper & Core Lock)"

        recommended_tiers.append(tier)
        page_diagnostics.append({
            "page": r.p,
            "kind": r.kind.value,
            "carriers": carriers,
            "recommended_tier": tier,
            "has_raster_scan": has_raster,
            "template_correlation": r.ras_stats.tmpl_ncc if r.ras_stats else 0.0,
            "estimated_alpha": r.ras_stats.est_alpha if r.ras_stats else 0.0,
            "ocr_piracy_spans": r.ras_stats.ocr_spans if r.ras_stats else 0,
        })

    return {
        "file": str(in_p),
        "total_pages": len(records),
        "dominant_tier": max(set(recommended_tiers), key=recommended_tiers.count) if recommended_tiers else "Tier 0",
        "pages": page_diagnostics,
    }


@mcp.tool()
def benchmark_page(file_path: str, page_number: int = 1) -> Dict[str, Any]:
    """Runs objective forensic verification benchmarks on a specific page, measuring residual watermark energy, authentic stroke retention, outside-mask invariance, and topological loop continuity.

    Args:
        file_path: Path to the PDF document.
        page_number: 1-indexed page number to benchmark.

    Returns:
        Objective metrics dictionary: residual, foreground_loss, outside_mask_diff, and topological_integrity.
    """
    in_p = Path(file_path).resolve()
    if not in_p.exists():
        return {"error": f"File not found: {file_path}"}

    import pymupdf

    from wmclean.verify.raster import verify_raster
    from wmclean.verify.topology import verify_topological_invariance

    with pymupdf.open(str(in_p)) as doc:
        if page_number < 1 or page_number > len(doc):
            return {"error": f"Page number {page_number} out of range (1..{len(doc)})"}
        p = doc[page_number - 1]
        pix = p.get_pixmap(dpi=150)
        import numpy as np
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        elif pix.n == 1:
            img = np.stack([img, img, img], axis=2)

    # Clean with Two-Regime Fill as empirical benchmark
    from wmclean.removers.two_regime import two_regime_fill
    cleaned = two_regime_fill(img)

    v_res = verify_raster(img, cleaned)
    t_pass, db0, db1, t_msg = verify_topological_invariance(img, cleaned)

    return {
        "file": str(in_p),
        "page": page_number,
        "verification_passed": v_res.passed and t_pass,
        "residual_watermark_energy": v_res.residual,
        "foreground_stroke_loss": v_res.content_loss,
        "outside_mask_pixel_diff": v_res.outside_mask_diff,
        "topological_betti_invariance": {
            "passed": t_pass,
            "delta_betti_1_components": db0,
            "delta_beta_1_closed_loops": db1,
            "summary": t_msg,
        },
    }


@mcp.tool()
def inspect_watermark_scorecard(
    file_path: str,
    page_number: int = 1,
    mode: str = "adaptive",
) -> Dict[str, Any]:
    """Generates an objective forensic before-and-after audit scorecard for a specific page.

    Quantifies detected watermark properties before treatment (energy, estimated alpha, carrier types,
    vector counts), and measures post-treatment residual energy, content loss, and topological/spectral
    invariance (Betti loops, skeleton retention, dangling tips, gradient magnitude, DCT halo variance).

    Args:
        file_path: Path to the input PDF or document image.
        page_number: 1-indexed page number to evaluate (default 1).
        mode: Execution mode ('fast', 'accurate', 'adaptive', 'hybrid').

    Returns:
        Structured audit scorecard containing detected_before and detected_after metrics.
    """
    import gc

    in_p = Path(file_path).resolve()
    if not in_p.exists():
        return {"error": f"File not found: {file_path}"}

    import cv2
    import numpy as np

    from wmclean.verify.hardened_math_qa import evaluate_four_invariant_qa
    from wmclean.verify.raster import verify_raster

    ext = in_p.suffix.lower()
    is_pdf = ext == ".pdf"

    detected_before: Dict[str, Any] = {}
    detected_after: Dict[str, Any] = {}

    if not is_pdf:
        img = cv2.imread(str(in_p))
        if img is None:
            return {"error": f"Failed to read image file: {file_path}"}
        orig_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        from wmclean.removers.two_regime import two_regime_fill

        cleaned = two_regime_fill(img)
        clean_grey = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY) if len(cleaned.shape) == 3 else cleaned

        v_res = verify_raster(img, cleaned)
        qa_res = evaluate_four_invariant_qa(orig_grey, clean_grey)

        detected_before = {
            "page_kind": "raster_image",
            "carriers": ["raster_layer"],
            "estimated_alpha": 0.0,
            "watermark_energy": round(float(v_res.residual), 4),
            "ocr_piracy_spans": 0,
            "vector_text_count": 0,
            "vector_path_count": 0,
        }
        detected_after = {
            "verification_passed": v_res.passed,
            "watermark_residual": v_res.residual,
            "foreground_content_loss": v_res.content_loss,
            "outside_mask_pixel_diff": v_res.outside_mask_diff,
            "four_invariant_qa": {
                "passed": qa_res.passed,
                "delta_betti_1_loops": qa_res.betti_1_delta,
                "skeleton_retention_pct": round(qa_res.skeleton_length_retention * 100.0, 2),
                "dangling_tip_delta": qa_res.dangling_endpoint_delta,
                "residual_gradient_magnitude": round(qa_res.residual_wm_energy, 4),
                "dct_halo_variance": round(qa_res.dct_boundary_halo_var, 4),
                "failure_reasons": list(qa_res.failure_reasons),
            },
            "methods_applied": ["two_regime_fill"],
        }
        del img, cleaned, orig_grey, clean_grey
        gc.collect()
        return {
            "file": str(in_p),
            "page": 1,
            "mode": mode,
            "status": "VERIFIED" if v_res.passed else "UNVERIFIED",
            "detected_before": detected_before,
            "detected_after": detected_after,
        }

    # PDF Processing:
    import pymupdf

    with pymupdf.open(str(in_p)) as doc:
        if page_number < 1 or page_number > len(doc):
            return {"error": f"Page number {page_number} out of range (1..{len(doc)})"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        single_in_pdf = tmp_p / "single_in.pdf"
        single_out_pdf = tmp_p / "single_out.pdf"
        single_rep_json = tmp_p / "single_rep.json"

        # Extract only the targeted page to minimize CPU/RAM overhead
        with pymupdf.open(str(in_p)) as doc:
            single_doc = pymupdf.open()
            single_doc.insert_pdf(doc, from_page=page_number - 1, to_page=page_number - 1)
            single_doc.save(str(single_in_pdf))
            single_doc.close()

        # Probe the extracted single page
        try:
            records = probe_document(str(single_in_pdf))
            rec = records[0] if records else None
        except Exception:
            rec = None

        carriers: list[str] = []
        est_alpha = 0.0
        tmpl_ncc = 0.0
        ocr_spans = 0
        vec_texts = 0
        vec_paths = 0
        page_kind = "unknown"

        if rec is not None:
            page_kind = rec.kind.value
            if rec.annots:
                carriers.append("annot")
            if rec.ocgs:
                carriers.append("ocg")
            if rec.artifacts:
                carriers.append("artifact")
            if rec.text_cand:
                carriers.append("vector_text")
                vec_texts = len(rec.text_cand)
            if rec.path_cand:
                carriers.append("vector_path")
                vec_paths = len(rec.path_cand)
            if rec.xobj_cand:
                carriers.append("xobject")
            if rec.ras_stats:
                est_alpha = rec.ras_stats.est_alpha
                tmpl_ncc = rec.ras_stats.tmpl_ncc
                ocr_spans = rec.ras_stats.ocr_spans

        detected_before = {
            "page_kind": page_kind,
            "carriers": carriers,
            "estimated_alpha": round(float(est_alpha), 4),
            "watermark_energy": round(float(tmpl_ncc), 4),
            "ocr_piracy_spans": ocr_spans,
            "vector_text_count": vec_texts,
            "vector_path_count": vec_paths,
        }

        # Run Orchestrator on the single page
        orch = Orchestrator(
            input_path=str(single_in_pdf),
            output_path=str(single_out_pdf),
            report_path=str(single_rep_json),
            mode=mode,
        )
        orch_exit = orch.execute()

        methods_applied: list[str] = []
        page_st = "UNVERIFIED"
        if single_rep_json.exists():
            try:
                rep_data = json.loads(single_rep_json.read_text(encoding="utf-8"))
                pages = rep_data.get("pages", [])
                if pages:
                    p0 = pages[0]
                    methods_applied = p0.get("m", [])
                    page_st = p0.get("st", "UNVERIFIED")
            except Exception:
                pass

        # Render original and treated pages for independent 4-invariant QA verification
        with pymupdf.open(str(single_in_pdf)) as orig_d:
            pix_orig = orig_d[0].get_pixmap(dpi=150)
            orig_img = np.frombuffer(pix_orig.samples, dtype=np.uint8).reshape(pix_orig.h, pix_orig.w, pix_orig.n)
            if pix_orig.n == 4:
                orig_img = orig_img[:, :, :3]
            elif pix_orig.n == 1:
                orig_img = np.stack([orig_img, orig_img, orig_img], axis=2)

        treated_file = single_out_pdf if single_out_pdf.exists() else single_in_pdf
        with pymupdf.open(str(treated_file)) as treat_d:
            pix_treat = treat_d[0].get_pixmap(dpi=150)
            clean_img = np.frombuffer(pix_treat.samples, dtype=np.uint8).reshape(pix_treat.h, pix_treat.w, pix_treat.n)
            if pix_treat.n == 4:
                clean_img = clean_img[:, :, :3]
            elif pix_treat.n == 1:
                clean_img = np.stack([clean_img, clean_img, clean_img], axis=2)

        orig_grey = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        clean_grey = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)

        v_res = verify_raster(orig_img, clean_img)
        qa_res = evaluate_four_invariant_qa(orig_grey, clean_grey)

        detected_after = {
            "verification_passed": v_res.passed and (orch_exit == 0),
            "watermark_residual": v_res.residual,
            "foreground_content_loss": v_res.content_loss,
            "outside_mask_pixel_diff": v_res.outside_mask_diff,
            "four_invariant_qa": {
                "passed": qa_res.passed,
                "delta_betti_1_loops": qa_res.betti_1_delta,
                "skeleton_retention_pct": round(qa_res.skeleton_length_retention * 100.0, 2),
                "dangling_tip_delta": qa_res.dangling_endpoint_delta,
                "residual_gradient_magnitude": round(qa_res.residual_wm_energy, 4),
                "dct_halo_variance": round(qa_res.dct_boundary_halo_var, 4),
                "failure_reasons": list(qa_res.failure_reasons),
            },
            "methods_applied": methods_applied,
        }

        del orig_img, clean_img, orig_grey, clean_grey
        gc.collect()

        return {
            "file": str(in_p),
            "page": page_number,
            "mode": mode,
            "status": page_st,
            "detected_before": detected_before,
            "detected_after": detected_after,
        }



@mcp.tool()
def dewarp_document_page(
    file_path: str,
    output_path: Optional[str] = None,
    page_number: int = 1,
    mode: str = "auto",
) -> Dict[str, Any]:
    """Execute zero-crop geometric dewarping and deskewing on a document page or image.

    Straightens tilted pages, unrolls book-gutter spine curvature, and flattens wavy paper
    bulges while dynamically expanding the canvas to guarantee zero margin cropping.

    Args:
        file_path: Path to the input PDF or image file.
        output_path: Optional output path for the dewarped PDF or image.
        page_number: 1-based page number (for PDFs).
        mode: Dewarping mode ('auto', 'skew', 'gutter', 'bump').

    Returns:
        Structured JSON dictionary containing modality, applied flag, original_dims,
        rectified_dims, and latency_ms.
    """
    in_p = Path(file_path).resolve()
    if not in_p.exists():
        return {"error": f"File not found: {file_path}"}

    import cv2
    import numpy as np

    from wmclean.geometry.dewarp import dewarp_page
    from wmclean.geometry.probe_geom import probe_page_geometry
    from wmclean.schemas import GeometricModality

    ext = in_p.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
        img = cv2.imread(str(in_p))
        if img is None:
            return {"error": f"Failed to read image: {file_path}"}
    else:
        import pymupdf

        with pymupdf.open(str(in_p)) as doc:
            if page_number < 1 or page_number > len(doc):
                return {"error": f"Page number {page_number} out of range (1..{len(doc)})"}
            p = doc[page_number - 1]
            pix = p.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    probe = None
    if mode != "auto":
        forced_modality = {
            "skew": GeometricModality.GEOM_SKEW_RIGID,
            "gutter": GeometricModality.GEOM_GUTTER_CURVE,
            "bump": GeometricModality.GEOM_WAVY_BUMP,
        }.get(mode, GeometricModality.GEOM_FLAT)
        probe = probe_page_geometry(img)
        probe.modality = forced_modality

    rectified, res = dewarp_page(img, probe=probe)

    if output_path:
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), rectified)

    return {
        "file": str(in_p),
        "page": page_number,
        "modality": res.modality.value,
        "applied": res.applied,
        "original_dims": res.original_dims,
        "rectified_dims": res.rectified_dims,
        "skew_deg": res.skew_deg,
        "processing_ms": res.processing_ms,
        "summary": res.summary,
    }


def main():
    """Run FastMCP server over standard I/O transport."""
    mcp.run()



if __name__ == "__main__":
    main()
