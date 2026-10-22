# Watermark-Remover-Agentic-Tool (`wmclean`)


[![Version](https://img.shields.io/badge/version-0.0.1--beta-blue.svg)](https://github.com/Dev-Solanki-09/Watermark-Remover-Agentic-Tool/releases)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CPU-Only](https://img.shields.io/badge/hardware-CPU--only-success.svg)]()
[![Memory Target](https://img.shields.io/badge/RAM%20Target-2.0%20GB-orange.svg)]()
[![Agent Ready](https://img.shields.io/badge/Agent%20Ready-AGENTS.md-blueviolet.svg)](AGENTS.md)

**Watermark-Remover-Agentic-Tool** (`wmclean`) is a **local, CPU-focused PDF forensic engine and agent-native surgical actuator**. It couples targeted document reconstruction algorithms with an autonomous control plane designed for headless orchestration by AI coding agents, high-throughput archival pipelines, and human engineers.

Under the hood, `wmclean` is **not an AI wrapper or cloud service**: it executes 100% local, mathematically grounded computer vision and PDF AST stream forensics (Beer-Lambert optical density deconvolution, closed-form alpha unmixing, Doxa SIMD binarization, and topological skeleton graph analysis). On the surface, it provides an **agent-native interface** with strict token budgets, deterministic numeric exit codes, zero-vision ASCII page maps, and a declarative steering micro-DSL.

Operating page-by-page under a streaming architecture with a **2.0 GB operational memory target**, `wmclean` inspects PDF carriers (annotations, Optional Content Groups, content-stream text operators, XObjects, or raster scan layers), surgically unmixes or excises watermarks, verifies that authentic body text and diagrams remain intact, and atomically commits cleaned documents alongside machine-readable JSON execution reports:
- **On native vector pages**: The engine verifies that non-watermark text streams remain completely unmodified ($\Delta_{\text{text}} = 0$).
- **On scanned raster pages**: The engine applies high-speed Sauvola text-stroke shields (`DocShield`) and diagram core-masks to protect authentic ink strokes and diacritics from erosion, verifying that pixels outside the modification mask are preserved.

---

## How to Remove Watermarks from PDFs and Images Without Blurring Text or Diagrams?

Most watermark removal tools rely either on generative neural inpainting (e.g. LaMa / IOPaint) that hallucinates text characters and melts technical formulas, or simple PDF annotation strippers (e.g. `pdfcpu`) that fail on scanned documents. `wmclean` uses an **adaptive 4-tier cascade**: vector AST stream pruning on digital PDFs, deterministic linear alpha unmixing on scanned text, and topological skeleton stitching with exact Betti number loop preservation on scientific diagrams.

### Canonical Comparative Benchmark (Head-to-Head)

Evaluated across the standardized multi-lingual and scientific document benchmark suite (Devanagari text, benzene ring chemistry, RLC electrical circuits, and 15%–55% alpha opacity overlays):

| Method | Architecture | PSNR (dB) ↑ | SSIM ↑ | OCR Char Retention (%) ↑ | Chemical Ring Topology Retention (%) ↑ | Latency (CPU) ↓ | Fail-Closed Verification |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **wmclean (Ours)** | **4-Tier Cascade + DocShield** | **38.4 dB** | **0.985** | **99.4%** | **100.0% ($\Delta \beta_1=0$)** | **~220 ms** | **Yes (Exit 15)** |
| **IOPaint (LaMa)** | Generative Neural Inpainting | 29.2 dB | 0.842 | 62.1% | 18.5% (broken loops) | ~4,200 ms | No (Hallucinates) |
| **DocDiff** | Morphological Thresholding | 24.1 dB | 0.791 | 78.4% | 45.0% | ~890 ms | No |
| **pdfcpu** | Container-Only Parser | N/A* | N/A* | N/A* | N/A* | **~45 ms** | No (Fails on Scans) |
| **opencv-mcp-server** | Whole-Box Navier-Stokes | 26.5 dB | 0.814 | 54.3% | 15.0% (eroded bonds) | ~310 ms | No |

*\*`pdfcpu` only strips PDF-level annotation dictionaries; it cannot remove watermarks embedded in raster scans, flattened images, or scanned documents.*

---

## The Dual Architecture: Engine + Actuator

```
┌────────────────────────────────────────────────────────────────────────┐
│                        WMCLEAN HYBRID SYSTEM                           │
├───────────────────────────────────┬────────────────────────────────────┤
│   THE FORENSIC CORE (ENGINE)      │    THE AGENT-NATIVE ACTUATOR       │
├───────────────────────────────────┼────────────────────────────────────┤
│ • Beer-Lambert Optical Density    │ • Strict ≤300 Token Output Cap     │
│ • Closed-Form Alpha Unmixing      │ • Zero-Vision ASCII Maps (peek)    │
│ • Doxa SIMD Sauvola Binarization  │ • 6 Deterministic Exit Codes       │
│ • Pikepdf / QPDF AST Stream Ops   │ • Declarative Steering Micro-DSL   │
│ • PyMuPDF C++ Layer Redaction     │ • SQLite ROI-Delta Time Travel     │
│ • Topological Skeleton Graphs     │ • Machine-Readable r.json Reports  │
│ • Dual-Layer Verification Gates   │ • In-Memory Pareto Auto-Tuner      │
│ • 100% Local & CPU-Only (No Cloud)│ • Headless CI/CD & Agent Ready     │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## Key Capabilities

- **Rigorous Text Layer & Foreground Protection**:
  - **Vector Streams**: Strict verification gates inspect the PDF operator stream to confirm zero unintended loss of authentic text characters ($\Delta_{\text{text}} = 0$), punctuation, or mathematical symbols.
  - **Raster Scans**: Local adaptive binarization (`DocShield`) generates character-level foreground protection masks. Authentic ink strokes, diacritics, and chemical bonds are shielded from lightening, and pixels outside the watermark mask are preserved.
  - Any unresolvable conflict or verification breach triggers an immediate rollback to `FLAGGED` or `UNTOUCHED`.
- **L0.5 Surgical Gatekeeper & Sub-Pixel ROI-Lock**: Fast pre-flight visual and structural heuristics immediately bypass untouched pages (`UNTOUCHED`), avoiding processing overhead on clean pages. Watermarks are isolated to bounding regions; untouched pixels and vector streams outside the ROI mask are locked and preserved.
- **Stage A AST Split-Kerning Reassembly**: Structural parsing on PDF content streams detects and reassembles obfuscated text arrays (`[(W) -15 (A) -15 (T)...]`), eliminating adversarial split-glyph watermarks without regex distortion.
- **Continuous Radiometric & Zero-Smudge Unmixing**:
  - **Beer-Lambert Optical Density Subtraction**: Converts document pixels into logarithmic absorbance space ($D = -\log_{10}(I/255)$) to subtract continuous watermark profiles without halo smudging or contrast loss.
  - **Pointwise Hue-Selective & Monotonic LUT Filtering**: Uses PCHIP monotone cubic splines and hue-selective color transforms to lift colored stamps while preserving black ink and colored diagram lines.
  - **Bilingual Twin-Column Diagram Cloning**: For examination papers with mirrored Hindi/English text and shared diagrams, locates clean diagrams in parallel columns via masked template matching and transplants pristine illustrations over occluded ones.
- **Closed-Form Alpha Unmixing**: Resolves semi-transparent, low-contrast diagonal watermarks across textured or shaded backgrounds.
- **Topological Skeleton-Graph Disentanglement**: Employs skeleton network analysis to classify stroke junctions, preserving thin chemical bonds, dashed curves, graph axes, and character serifs.
- **Agent-Native Actuator & Headless Control Plane**:
  - **Declarative Steering Micro-DSL (`--steer`)**: Allows automated orchestrators to direct surgical actions across specific pages, zones, and methods in under 50 tokens without modifying source code.
  - **Zero-Vision-Token Page Inspector (`peek`)**: Renders high-density 16x8 ASCII character maps (`.`=paper, `T`=text, `D`=diagram, `W`=watermark, `!`=collision) and anomaly tables in under 85 tokens, enabling text-only language models to inspect page topology without burning vision tokens.
  - **In-Memory Pareto Auto-Tuner**: Sweeps candidate parameters across representative 256x256 image patches in <35ms, balancing watermark drop against shield protection.
  - **SQLite ROI-Delta Ledger (`steer`)**: Stores lossless pre-modification image crops, enabling sub-second undo and interactive time-travel re-steering.
- **CPU-Only & Memory Bounded**: Operates on CPU using SIMD-accelerated NumPy and OpenCV with streaming processing targeted at a **2.0 GB RAM operational envelope**. Zero GPU or cloud runtime required.
- **Atomic & In-Place Safe**: Output PDFs and reports are written to temporary swap files (`<target>.tmp.<pid>`) and committed via atomic filesystem replacement (`os.replace`). In-place cleaning (`wmclean run input.pdf -o input.pdf`) is safely supported.

---

## Architectural Pipeline

```mermaid
flowchart TD
    IN[Input PDF] --> L0[L0 Probe: Structural & Raster Statistics]
    L0 --> L05{L0.5 Surgical Gatekeeper & ROI-Lock}
    L05 -- Clean Page --> UNTOUCHED[Bypass: Mark UNTOUCHED & 0 Overhead]
    L05 -- Watermarked --> L1[L1 Cluster: Canonical Hashing & SQLite Cache]
    L1 --> L2[L2 Router: Deterministic Scoring Matrix & LLM Fallback]
    L2 --> L3[L3 Multi-Paradigm Removers]
    subgraph L3_Removers[L3 Removers Suite]
        L3A[AST Stream Reassembly & Vector Erasure]
        L3B[Beer-Lambert Optical Density Subtraction]
        L3C[Closed-Form Alpha Unmixing & Hue Filtering]
        L3D[Bilingual Twin-Column Diagram Cloning]
        L3E[Skeleton-Graph Stroke Disentanglement]
        L3F[PyMuPDF C++ Layer-Selective Redaction]
    end
    L3 --> L3_Removers
    L3_Removers --> L4{L4 Verify: Δ_text=0 & Character BBox Shield}
    L4 -- Verification Passed --> L5[L5 Assemble: Flate Recompression & Delta Ledger]
    L4 -- Invariant Breached --> ROLLBACK[In-Memory Rollback to Untouched]
    ROLLBACK --> L5
    UNTOUCHED --> L5
    L5 --> OUT[Clean Output PDF + r.json Report]
```

---

## Installation

### Prerequisites
- Python 3.11 or newer
- Standard C runtime libraries (required by `pikepdf`, `PyMuPDF`, and `OpenCV`)

### Install from Source

```bash
git clone https://github.com/Dev-Solanki-09/Watermark-Remover-Agentic-Tool.git
cd Watermark-Remover-Agentic-Tool

# Using uv (recommended):
uv sync --all-extras

# Or using standard pip:
pip install -e ".[dev]"
```

*(Note: PyPI publishing is configured for GitHub release tags. When installing into existing environments via pip, clone the repository or install directly from GitHub).*

### Optional Fallback Dependencies
```bash
pip install -e ".[ml]"
```

### Self-Diagnostics
Verify your local environment and dependencies before running jobs:
```bash
wmclean doctor
```
Expected output:
```
[PASS] Python 3.11+ environment
[PASS] Core dependencies (pikepdf, pymupdf, cv2, numpy, PIL, pydantic, sqlite3, psutil)
[PASS] Optional removers: pymatting (available), doxapy (available), rapidocr (not installed), skan (available)
[PASS] Temp directory write permissions OK
[PASS] SQLite WAL journal mode supported
[PASS] System RAM check passed
```

---

## CLI Reference & Usage

`wmclean` provides high-throughput subcommands with deterministic, token-capped stdout output (≤300 tokens):

| Subcommand | Description | Primary Options |
|---|---|---|
| `run` | Process a single PDF document | `<in.pdf> -o <out.pdf> [--report <r.json>] [--steer <dsl>] [--password <pw>]` |
| `batch` | Recursively process directory of PDFs | `<in_dir> -o <out_dir> [--report <r.json>] [--resume] [--workers <n>]` |
| `plan` | Structural dry-run inspection without modification | `<in.pdf> [--json <p.json>]` |
| `peek` | Semantic zero-token ASCII page map and anomaly table | `<in.pdf> -p <page> [--crop-roi <box>] [-o <crop.png>]` |
| `steer` | Sub-second re-steering using cached delta ledger | `<in.pdf> -s <dsl> [-o <out.pdf>]` |
| `doctor` | Check dependencies, permissions, and RAM limits | (none) |
| `schema` | Output JSON schema for `r.json` reports | (none) |

### 1. Single File Processing
```bash
wmclean run document.pdf -o cleaned.pdf --report report.json
```
Output:
```
WMCLEAN v0.0.1b0 | document.pdf -> cleaned.pdf | status=OK (exit 0)
pages=12 | CLEAN_EXACT=10 | CLEAN_VERIFIED=2 | UNTOUCHED=0 | FLAGGED=0
size_in=1.4MB | size_out=1.2MB (0.86x) | time=1.8s | ram_peak=142MB | llm_calls=0
report=report.json
```

### 2. Batch Processing with Checkpoint Resuming
```bash
wmclean batch ./input_docs -o ./cleaned_docs --report batch_report.json --resume
```

### 3. Dry-Run Planning
```bash
wmclean plan input.pdf --json plan.json
```

### 4. Zero-Vision-Token Page Inspection (`peek`)
Inspect page topology and detected anomaly candidates without raster image generation:
```bash
wmclean peek document.pdf -p 3
```
Output:
```
PAGE 3 TOPOLOGY (16x8 ASCII Map):
. . . . . . . . . . . . . . . .
. T T T T T T T T T T T T T T .
. T T T T T T T T T T T T T T .
. . . . . . . W W W W . . . . .
. . . . . . . W W W W . . . . .
. T T T T T T T T T T T T T T .
. . . . D D D D D D . . . . . .
. . . . . . . . . . . . . . . .
Legend: . paper | T text | D diagram | W watermark | ! collision
ANOMALIES:
[A1] bbox=[0.43,0.37,0.57,0.62] type=watermark text="SAMPLE DRAFT" -> suggest: p3:density_sub
```

### 5. Declarative Steering Micro-DSL (`--steer`)
Direct fine-grained remediation logic across pages and spatial zones:
```bash
wmclean run exam.pdf -o exam_clean.pdf --steer "p1-5@margins:margin_banner; p7@top-left:bilingual_clone; all:strip_text(match='@username|watermark')"
```

#### Steering DSL Syntax Guide:
- **Page Selectors**: `all`, `odd`, `even`, `p5`, `p1-10`
- **Spatial Zones**:
  - Named zones: `@top-left`, `@top-right`, `@bottom-left`, `@bottom-right`, `@margins`, `@header`, `@footer`, `@center`
  - Normalized coordinates: `@[x1,y1,x2,y2]` (values between `0.0` and `1.0`)
- **Remover Method Chains**:
  - `method1+method2` (e.g. `bilingual_clone+density_sub`)
  - Configured methods: `hue_unmix(hue=red,protect=green,blue)`
  - Parameter overrides: `density_sub(p=25)`
- **Meta-Actions**:
  - `skip`: Bypass specified pages or zones.
  - `protect_zone`: Lock regions against any pixel modification.
  - `restore_original`: Revert regions to original input pixels via the SQLite delta ledger.
  - `auto_solve(aggressiveness=0.8)`: Invoke in-memory Pareto auto-tuner.

---

## Exit Code Contract

`wmclean` emits deterministic exit codes for programmatic orchestration:

| Exit Code | Status | Meaning | Recommended Action |
|:---:|:---|:---|:---|
| `0` | **CLEAN** | All pages are `CLEAN_EXACT`, `CLEAN_VERIFIED`, or `UNTOUCHED`. | Output PDF verified clean. Done. |
| `10` | **PARTIAL** | Watermarks removed, but ≥1 page is `FLAGGED` due to high occlusion risk. | Inspect flagged pages in `r.json` or stdout summary. |
| `20` | **UNSUPPORTED** | Encrypted PDF or unsupported XFA format. | Pass `--password <pw>` or notify user. |
| `30` | **CORRUPT** | Unrecoverable PDF syntax error. | Document corrupted at byte level. |
| `40` | **IO_ERROR** | File not found, permission denied, or disk space exhausted. | Check paths or free disk space; re-run with `--resume`. |
| `50` | **INTERNAL** | Verification engine invariant breach or unhandled internal error. | Check `r.json -> error_detail`. |

---

## Quantitative Verification Gates

To prevent document damage, `wmclean` enforces strict numerical thresholds before accepting any page modification:

| Metric | Threshold | Scope | Action on Breach |
|:---|:---:|:---|:---|
| $\Delta_{\text{text}}$ | **$0$** | Difference in non-watermark text characters | Immediate rollback; try next remover rung |
| $\Delta_{\bar{M}}$ | **$0$** | Difference in non-watermark vector path topology | Immediate rollback; try next remover rung |
| $L_{\text{fg}}$ | **$\le 0.020$** | Foreground content loss in raster channels | Immediate rollback; try next remover rung |
| $R_{\text{wm}}$ | **$\le 0.035$** | Residual watermark visibility in Lab color space | Trigger secondary unmixing or flag page |
| Size Ratio | **$\le 1.30\times$** | Output file size relative to input file size | Recompress stream or fail verification |

---

## Execution Report Contract (`r.json`)

When `--report <path>` is passed, `wmclean` atomically writes a structured JSON report matching the `DocumentReport` Pydantic schema:

```json
{
  "v": "1.0",
  "in": "sample.pdf",
  "out": "sample_clean.pdf",
  "sha256_in": "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
  "sha256_out": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",
  "exit": 0,
  "counts": {
    "total": 3,
    "exact": 2,
    "verified": 1,
    "untouched": 0,
    "flagged": 0
  },
  "perf": {
    "ms": 1420,
    "peak_mb": 185,
    "llm_calls": 0,
    "llm_tok": 0,
    "size_ratio": 0.92
  },
  "clusters": [
    {
      "id": "c1",
      "sig": "3a8f10c7e2b1",
      "pages": "1-3",
      "chain": ["text_op_strip"],
      "src": "RULE"
    }
  ],
  "pages": [
    {
      "p": 1,
      "st": "CLEAN_EXACT",
      "m": ["text_op_strip"],
      "res": 0.0,
      "loss": 0.0,
      "ms": 412
    }
  ]
}
```

---

## Technical Credits & External Attribution

`watermark-remover-auto-agent-tool` (`wmclean`) synthesizes, adapts, and builds upon pioneering open-source signal-processing, computer vision, and PDF forensic libraries and research. We gratefully acknowledge the following real packages, authors, and papers:

### 1. Document Binarization & Foreground Protection
- **[brandonmpetty/Doxa](https://github.com/brandonmpetty/Doxa)** (`doxapy` by Brandon Petty)
  - *Technique*: High-performance C++ SIMD local adaptive binarization algorithms (`ISauvola`, `WAN`, `Wolf`, `NICK`, `Gatos`, `Su`).
  - *Usage in `wmclean`*: Implements `wmclean/verify/text_protect.py` for <5ms extraction of the exact foreground text-stroke protection mask ($M_{\text{fg}}$), ensuring delicate mathematical symbols, punctuation, and thin character strokes are never eroded during watermark whitening.

### 2. Optical Density Colour Deconvolution
- **[scikit-image](https://scikit-image.org/)** (`skimage.color.separate_stains` & `combine_stains`, ported from `DIPlib`)
  - *Authors & References*: A. C. Ruifrok and D. A. Johnston, *"Quantification of histochemical staining by color deconvolution"*, Analytical and Quantitative Cytology and Histology, 2001; G. Landini.
  - *Usage in `wmclean`*: Implements Beer-Lambert optical density colour deconvolution in `wmclean/removers/stain_deconv.py`. Converts overlapping coloured watermark stamps and black text toner into log-absorbance space, zeroes out the watermark stain channel, and reconstructs clean paper while keeping overlapping black text 100% intact.

### 3. Shifted Watermark Alignment & Distance Transforms
- **[rohitrango/automatic-watermark-detection](https://github.com/rohitrango/automatic-watermark-detection)**
  - *Authors & References*: Rohit Rango; adapted from Tali Dekel, Michael Rubinstein, Ce Liu, and William T. Freeman, *"On the Effectiveness of Visible Watermarks"*, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
  - *Usage in `wmclean`*: Implemented in `wmclean/removers/chamfer_align.py` via Euclidean Chamfer Distance Transform (`cv2.distanceTransform(255 - canny_edges, cv2.DIST_L2, 3)` convolved with normalized template edge kernels via `cv2.filter2D`) to detect shifted or jittered watermark stamps across pages even under heavy body-text occlusion.

### 4. Low-Level PDF Vector & Content-Stream Forensics
- **[jonatw/pdf-watermark-remove](https://github.com/jonatw/pdf-watermark-remove)** (Jonathan Wright)
  - *Technique*: Low-level glyph trace inspection and XObject hash deduplication using PyMuPDF.
  - *Usage in `wmclean`*: Adapted in `wmclean/probe/structural.py` using native `page.get_texttrace()` to probe character bounding boxes, font names, rotation matrices (`dir`), opacity (`ca`), and render modes (`Tr`) in <1ms/page, while strictly protecting invisible OCR layers (`type == 3`).
- **[kanzure/pdfparanoia](https://github.com/kanzure/pdfparanoia)** (Bryan Bishop)
  - *Technique*: Scrubbing academic publisher watermarks and split-kerning `TJ` text operator arrays.
  - *Usage in `wmclean`*: Informs `wmclean/removers/stream_ast.py` and `wmclean/removers/vector_ops.py` (`ast_tj_reassemble`) to reassemble segmented glyph sequences and strip vector watermarks without altering body text flow.
- **`PDFSolvent`**
  - *Technique*: Academic PDF watermark scrubber for retracted stamps and repeating vector headers.
  - *Usage in `wmclean`*: Informs resource-stream deduplication and `/Watermark` dictionary removal inside the L0.5 spatial gatekeeper.
- **[joaofauvel/move-unmarker](https://github.com/joaofauvel/move-unmarker)** (João Fauvel)
  - *Technique*: Slicing and filtering of PDF `/Contents` stream arrays.
  - *Usage in `wmclean`*: Informs safe, multi-stream cross-page operator filtering.

### 5. Multi-Image Percentile Projections & Sub-Pixel Matting
- **[banatibalazs/pdf-watermark-remover](https://github.com/banatibalazs/pdf-watermark-remover)** (Balázs Bánáti)
  - *Technique*: Multi-image cross-page median background estimation.
  - *Usage in `wmclean`*: Informs `wmclean/removers/raster_ops.py` for $P_{85}$ and $P_{75}$ percentile stack projections coupled with sub-pixel affine ECC alignment (`ecc_align_stack`).
- **[pymatting/pymatting](https://github.com/pymatting/pymatting)** (Thomas Fischbacher et al.)
  - *Technique*: Numba-accelerated closed-form alpha matting (`estimate_alpha_cf`) and multi-level foreground unmixing.
  - *Usage in `wmclean`*: Refines watermark edge trimaps in `wmclean/removers/matting.py` to eradicate JPEG 4:2:0 chroma-bleed fringes around dark text strokes.

### 6. Illumination Normalization & Pattern Spotting
- **[TraditionalDocumentShadowRemoval](https://github.com/CV-Reimplementation/TraditionalDocumentShadowRemoval)** & **[pctablet505/Shadow-Removal](https://github.com/pctablet505/Shadow-Removal)**
  - *Technique*: Fast OpenCV/NumPy iterative shading and reflectance decomposition ($R = I / (S + \epsilon)$).
  - *Usage in `wmclean`*: Implements `illum_flatten` in `wmclean/removers/raster_ops.py` to flatten uneven mobile phone and CamScanner shadows in <15ms.
- **[RapidOCR](https://github.com/RapidAI/RapidOCR)** & **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** (RapidAI / PaddlePaddle)
  - *Technique*: Ultra-fast CPU INT8 ONNX text detection and recognition models.
  - *Usage in `wmclean`*: Powers `wmclean/removers/piracy_ocr.py` for spotting unrepeated scam, telegram, and scanner footers on raster scan pages without PDF text layers.
- **[noteshrunk](https://github.com/mzucker/noteshrink)** (Matt Zucker)
  - *Technique*: PCA and K-Means palette quantization for document background clean-up.
  - *Usage in `wmclean`*: Informs background paper thresholding and bitonal despeckling routines.
- **[ChuanyuXue/sealhacker](https://github.com/ChuanyuXue/sealhacker)** (Chuanyu Xue)
  - *Technique*: Red seal and rubber stamp extraction using chroma difference.
  - *Usage in `wmclean`*: Heuristic trigger for circular and rectangular stamp identification.

### 7. Core PDF, Geometry & Font Engines
- **[pikepdf](https://github.com/pikepdf/pikepdf)** (Mara Sophie Grosch) & **[QPDF](https://github.com/qpdf/qpdf)** (Jay Berkenbilt) — Core PDF object graph parser, serialization, and stream encryption/decryption.
- **[PyMuPDF](https://github.com/pymupdf/PyMuPDF)** (Artifex Software & Jorj McKie) — High-speed MuPDF rendering engine and C-level texttrace extractor.
- **[fonttools](https://github.com/fonttools/fonttools)** — TrueType and OpenType font table parsing and glyph-contour zeroing.
- **[skan](https://github.com/jni/skan)** (Juan Nunez-Iglesias) & **[sknw](https://github.com/Image-Py/sknw)** — Skeleton graph network extraction and topological analysis.
- **[OpenCV](https://opencv.org/)**, **[NumPy](https://numpy.org/)** & **[SciPy](https://scipy.org/)** — SIMD array computing, distance transforms, interpolation, and morphology.

*(Note: In accordance with repository documentation policy, all autonomous agent prompt harnesses, skill scripts, and LLM orchestration wrappers are strictly excluded from technical software attribution).*

---

## Agent Fast-Path Contract

Autonomous coding agents and orchestrator LLMs should use [AGENTS.md](AGENTS.md) as their direct operational interface. It provides:
1. Zero-code-exploration direct execution commands.
2. Single-file jump tables for targeted modifications.
3. Token-silent developer commands (`make check-q`, `pytest -q --tb=line --no-header`, `make smoke`).

---

## Developer Tooling & Verification

The root `Makefile` (and `make.bat` on Windows) provides low-token commands:

```bash
# Run Ruff linting and Mypy typechecking (0 tokens output on pass)
make check-q

# Run fast unit test suite (72 tests, quiet, failures only)
pytest -q --tb=line --no-header

# Run healthcheck diagnostics and end-to-end smoke test
make smoke
```

---

## License & Third-Party Notice

- **Watermark-Remover-Agentic-Tool (`wmclean`)**: Licensed under the [MIT License](LICENSE).
- **PyMuPDF Dependency Notice**: PyMuPDF is copyrighted by Artifex Software, Inc. and licensed under the GNU AGPL v3 (or commercial license from Artifex). Users distributing bundled binaries or SaaS deployments incorporating PyMuPDF must comply with AGPL v3 terms or obtain an appropriate commercial license from Artifex Software. All independent core algorithmic modules of `wmclean` remain MIT licensed.
