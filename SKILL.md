---
name: watermark-remover
description: Fail-closed forensic watermark, stamp, seal, and background remover for PDFs, scanned documents, and scientific diagrams.
---

# Watermark Remover (`wmclean`) — Agent Skill

This skill allows AI coding agents and autonomous workflows to inspect, diagnose, and remove watermarks, stamps, and background overlays from PDF documents and scans without blurring text or breaking scientific diagram topologies.

## When to Use This Skill
- User asks to remove, strip, or clean a watermark from a PDF, document scan, or image.
- Document contains semi-transparent tinted text ("CONFIDENTIAL", "DRAFT", "SAMPLE", student IDs, timestamps).
- Document contains delicate scientific diagrams, circuit schematics, chemical benzene rings, or non-Latin scripts (Devanagari, CJK, Arabic) that must not be degraded.
- Agent needs to inspect whether a PDF contains vector watermark layers (OCG/XObject/Annotations) or raster stamps.

## Execution Fast-Path (CLI)

Run commands directly via shell:
```bash
# 1. Preview and plan without modifying the file (sub-second dry-run)
wmclean plan <input.pdf> [--json plan.json]

# 2. Clean single PDF with audit report and optional dewarping/steering
wmclean run <input.pdf> -o <output.pdf> --report report.json [--dewarp] [--steer <dsl>]

# 3. Batch clean an entire directory with resume checkpointing
wmclean batch <input_dir> -o <output_dir> --report batch_report.json --resume

# 4. Standalone zero-crop geometric dewarping and deskewing
wmclean dewarp <input.pdf> -o <output.pdf> [--mode auto|skew|gutter|bump]

# 5. Semantic zero-vision ASCII page map and anomaly inspector (<85 tokens)
wmclean peek <input.pdf> -p 1 [--crop-roi <box>] [-o crop.png]

# 6. Sub-second interactive time-travel re-steering via SQLite delta ledger
wmclean steer <input.pdf> -s <dsl> [-o <output.pdf>]

# 7. Pre-flight healthcheck for dependencies, SOTA extensions, and RAM
wmclean doctor

# 8. Output machine-readable JSON schema for report.json
wmclean schema
```

## Exit Code Contract & Next Actions
Always inspect process exit codes:
- `0` (CLEAN / UNTOUCHED): Output is verified. Zero content loss ($\Delta_{\text{text}}=0$), foreground stroke protected, residual watermark within bounds.
- `10` (PARTIAL): At least one page flagged for potential occlusion or edge risk. Read `flagged_pages` in stdout or JSON report.
- `15` (UNVERIFIED): Watermark residual or content loss exceeded tolerance bounds. Output was held or reverted. Inspect using `wmclean peek`.
- `20` (UNSUPPORTED / ENCRYPTED): File is password-protected or encrypted XFA. Request password from user or pass `--password <pw>`.
- `30` (CORRUPT): Input file is unrecoverable at PDF syntax level.
- `40` (IO ERROR): File path, permissions, or disk space exhausted.
- `50` (INTERNAL): Invariant breach; inspect report JSON `error_detail`.

## Declarative Steering Micro-DSL (`--steer`)
Direct fine-grained remediation logic across pages and spatial zones without touching code:
```bash
wmclean run exam.pdf -o clean.pdf --steer "p1-5@margins:margin_banner; p7@top-left:bilingual_clone; all:strip_text(match='@username|watermark')"
```
- **Page Selectors**: `all`, `odd`, `even`, `p5`, `p1-10`
- **Spatial Zones**: `@top-left`, `@top-right`, `@bottom-left`, `@bottom-right`, `@margins`, `@header`, `@footer`, `@center`, or normalized `@[x1,y1,x2,y2]`
- **Method Chains**: `bilingual_clone+density_sub`, `hue_unmix(hue=red)`, `density_sub(p=25)`
- **Meta-Actions**: `skip`, `protect_zone`, `restore_original`, `auto_solve(aggressiveness=0.8)`

## Technical Attribution & Credits
For full mathematical formulations, citations, research papers, and open-source software credits across all 20+ synthesized libraries, refer to [CREDITS.md](CREDITS.md).

## MCP Server Integration

To connect as a local MCP server, add to your agent or IDE configuration (`.mcp.json` or `claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "watermark-remover": {
      "command": "wmclean-mcp"
    }
  }
}
```

### Direct Python Agent Invocation

#### Using `smolagents`
```python
from smolagents import Tool
from wmclean.mcp_server import remove_watermark_auto

class WatermarkRemoverTool(Tool):
    name = "remove_watermark"
    description = "Removes watermarks from PDFs while strictly preserving text and diagram topology."
    inputs = {
        "file_path": {"type": "string", "description": "Path to input PDF."},
        "output_path": {"type": "string", "description": "Destination path for cleaned PDF.", "nullable": True}
    }
    output_type = "string"

    def forward(self, file_path: str, output_path: str = None) -> str:
        res = remove_watermark_auto(file_path, output_path=output_path)
        return f"Status: {res['status']}, Exit: {res['exit_code']}, Cleaned: {res.get('cleaned_pages', 0)} pages"
```

#### Using LangChain
```python
from langchain.tools import tool
from wmclean.mcp_server import remove_watermark_auto, inspect_watermark_layers

@tool
def clean_pdf_watermark(file_path: str, output_path: str = None) -> dict:
    """Removes watermarks from a PDF document with fail-closed forensic verification."""
    return remove_watermark_auto(file_path, output_path)

@tool
def inspect_pdf_layers(file_path: str) -> dict:
    """Diagnoses whether watermark is vector, alpha-blended, or raster stamp."""
    return inspect_watermark_layers(file_path)
```
