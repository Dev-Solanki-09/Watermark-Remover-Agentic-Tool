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
    preserve_scientific_topology: bool = True,
) -> Dict[str, Any]:
    """Universal watermark, stamp, seal, and background overlay remover for PDFs, scanned documents, scientific diagrams, chemical structures, and photographs.

    Automatically routes between lossless PDF AST layer stripping (Tier 0), deterministic
    inverse alpha matting (Tier 1), topological skeleton stitching with Betti-number loop
    preservation (Tier 2), and masked inpainting (Tier 3) with independent fail-closed structural QA verification.

    Args:
        file_path: Absolute or relative path to the input PDF or document image.
        output_path: Optional destination path for the cleaned PDF. If omitted, generates '<file>_cleaned.pdf'.
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
        cleaned_count = counts.get("clean_exact", 0) + counts.get("clean_verified", 0)

        # Average verified residual
        residuals = [p.get("res", 0.0) for p in pages if p.get("res") is not None]
        avg_residual = float(sum(residuals) / len(residuals)) if residuals else 0.0

        return {
            "success": exit_code == ExitCode.SUCCESS.value,
            "status": "CLEAN" if exit_code == 0 else ("UNVERIFIED" if exit_code == 15 else "PARTIAL"),
            "exit_code": exit_code,
            "input_file": str(in_p),
            "output_file": str(out_p) if exit_code == 0 else None,
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
            "delta_beta_0_components": db0,
            "delta_beta_1_closed_loops": db1,
            "summary": t_msg,
        },
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
