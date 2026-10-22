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
# 1. Preview and plan without modifying the file
wmclean plan <input.pdf>

# 2. Clean single PDF with audit report
wmclean run <input.pdf> -o <output.pdf> --report report.json

# 3. Batch clean an entire directory with resume capability
wmclean batch <input_dir> -o <output_dir> --resume

# 4. Inspect specific page crop or diff
wmclean peek <input.pdf> -p 1 -o crop.png

# 5. Healthcheck environment and available vision removers
wmclean doctor
```

## Exit Code Contract & Next Actions
Always inspect process exit codes:
- `0` (CLEAN / UNTOUCHED): Output is verified. Zero content loss, text layer intact, residual watermark within bounds.
- `10` (PARTIAL): At least one page flagged for potential edge degradation. Check `flagged_pages` in stdout or JSON report.
- `15` (UNVERIFIED): Watermark residual or content loss exceeded tolerance bounds. Output was held or reverted. Inspect using `wmclean peek`.
- `20` (UNSUPPORTED / ENCRYPTED): File is password-protected. Request password from user or provide `--password <pw>`.
- `30` (CORRUPT): Input file is unrecoverable.
- `40` (IO ERROR): File path or permissions issue.
- `50` (INTERNAL): Invariant breach; inspect report JSON.

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
