# Contributing to watermark-remover-auto-agent-tool

Thank you for your interest in improving `watermark-remover-auto-agent-tool` (`wmclean`).

This project is built for high-throughput, deterministic watermark removal with **zero tolerance for false positives** on non-watermark content. It is designed to be operated both by human engineers and autonomous AI agents with near-zero token overhead.

---

## Architecture Overview

`wmclean` uses a decoupled, high-performance forensic architecture:

```
[Input PDF]
    │
    ▼
[L0 Probe]       ── Page geometry, annotations, OCGs, text candidates, raster stats
    │
    ▼
[L0.5 Gatekeeper]── Heuristic bypass (clean pages -> UNTOUCHED) & sub-pixel ROI-Lock
    │
    ▼
[L1 Cluster]     ── Canonical feature hashing & SQLite signature cache
    │
    ▼
[L2 Router]      ── Deterministic scoring matrix (or 1-call bounded LLM fallback)
    │
    ▼
[L3 Removers]    ── Vector AST strippers + Closed-form alpha matting + Stain deconvolution
    │
    ▼
[L4 Verify]      ── Exact vector gates (Δ_text=0, Δ_M̄=0) & glyph bounding-box shield
    │
    ▼
[L5 Assemble]    ── Stream re-encoding, Flate compression, atomic output swap
    │
    ▼
[Output PDF + r.json Report]
```

---

## Core Invariants

Any contribution must strictly maintain these invariants:

1. **Zero False Positives on Text**:
   - `clean_ref` text content must never be mutated or lost ($\Delta_{\text{text}} = 0$).
   - If a watermark cannot be cleanly removed without risking text or diagram integrity, the page **must** resolve to `FLAGGED` (exit code `10`) or remain `UNTOUCHED`. Never corrupt body text.
2. **CPU-Only & Memory Bounded**:
   - Hard RAM ceiling of 2.0 GB. Zero GPU dependencies.
   - All C-heap objects (`pikepdf.Pdf`, `fitz.Document`, `sqlite3.Connection`) must use explicit context managers or deterministic cleanup.
3. **Atomic Writes**:
   - Never write directly to target outputs. Always write to `<target>.tmp.<pid>` and swap using `os.replace()`.
4. **Token Economy**:
   - CLI stdout output must remain capped at ≤ 25 lines / 1,200 characters (~300 tokens).
   - Use `make check-q` and `make test-q` for zero-token-waste verification.

---

## Development Setup

Clone the repository and install dependencies:

### Using `uv` (Fast & Recommended)
```bash
git clone https://github.com/Dev-Solanki-09/watermark-remover-auto-agent-tool.git
cd watermark-remover-auto-agent-tool
uv sync --all-extras
```

### Using standard `pip`
```bash
git clone https://github.com/Dev-Solanki-09/watermark-remover-auto-agent-tool.git
cd watermark-remover-auto-agent-tool
pip install -e ".[dev]"
```

Verify your environment with `wmclean doctor`:

```bash
wmclean doctor
```

Expected output:
```
[PASS] Python 3.14.x (>= 3.10)
[PASS] Core dependencies (pikepdf, pymupdf, cv2, numpy, PIL, pydantic, sqlite3)
[PASS] Temp directory write permissions OK
[PASS] SQLite WAL journal mode supported
[PASS] System RAM check passed
```

---

## Developer Verification Commands

We provide quiet targets in the root `Makefile` (and `make.bat` for Windows) that emit **0 tokens on success**:

```bash
# Fast lint & typecheck (silent on pass):
make check-q

# Fast unit test suite (60 tests, quiet, failures only):
make test-q

# End-to-end smoke test (verifies doctor + synthetic run):
make smoke
```

---

## Adding a New Watermark Remover or Intelligence Heuristic

If you are adding a new remover rung to L3 or intelligence heuristic:

1. Add the enum member to `RemoverMethod` in `wmclean/schemas.py`.
2. Implement the remover function in `wmclean/removers/vector_ops.py` (vector), `wmclean/removers/stream_ast.py` (content stream AST), or `wmclean/removers/raster_ops.py` / `matting.py` (raster).
3. If adding a pre-filter heuristic, place it in `wmclean/intelligence/gatekeeper.py` or `roi_lock.py`.
4. Register the method in `wmclean/orchestrator.py` `method_map`.
5. Add routing rules in `wmclean/router/rules.py`.
6. Add unit tests in `tests/test_vector_ops.py`, `tests/test_raster_ops.py`, or `tests/test_sota_extensions.py`.
7. Run `make check-q` and `make test-q` to verify that all 60+ tests pass.

---

## Pull Request Guidelines

- Branch names should be descriptive (e.g. `feat/bitonal-morphology`, `fix/ocg-lookup`).
- Ensure all tests pass before opening a PR (`make check-q && make test-q`).
- Do not commit large binary test PDFs. Use programmatic generators in `tests/fixture_gen.py`.

---

## Acknowledgements & Prior Art

We gratefully acknowledge the developers of existing open-source PDF watermark removal projects whose research and tools paved the way:

- **[kanzure/pdfparanoia](https://github.com/kanzure/pdfparanoia)** for split-kerning array (`TJ`) scrubbing and structural publisher watermark stripping using `pikepdf`.
- **[chazeon/PDF-Watermark-Remover](https://github.com/chazeon/pdf-watermark-remover)** for PDF dictionary traversal concepts.
- **[jamespan/pdf-unstamper](https://github.com/jamespan/pdf-unstamper)** for content-stream text operator matching patterns.
- **[jonatw/pdf-watermark-remover](https://github.com/jonatw/pdf-watermark-remover)** for content-stream token filtering and structural PDF object unlinking.
- **[Goshin/Remove-PDF-Watermark](https://github.com/Goshin/Remove-PDF-Watermark)** for normalized template cross-correlation approaches.
- **[LJSthu/Python-Remove-Watermark](https://github.com/LJSthu/Python-Remove-Watermark)** for $\Delta E$ raster color distance masking.
- **[watermark-remover-cli](https://pypi.org/project/watermark-remover-cli)** for HSV saturation separation.
- The authors and maintainers of **[pikepdf](https://github.com/pikepdf/pikepdf)**, **[PyMuPDF](https://github.com/pymupdf/PyMuPDF)**, and **[qpdf](https://github.com/qpdf/qpdf)**.

