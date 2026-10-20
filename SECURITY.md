# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## Reporting a Vulnerability

We take the security of PDF parsing and processing very seriously. PDFs are complex, untrusted binary formats that can contain deeply nested objects, malicious stream compressions, or corrupt cross-reference tables.

If you discover a security vulnerability (such as a memory leak, parser crash, arbitrary file read/write, or denial-of-service vulnerability), please **do not** open a public issue. Instead, report it privately:

- **Email**: Send vulnerability reports with proof-of-concept files to `security@wmclean.org` (or contact maintainers via GitHub Private Vulnerability Reporting).
- **Details to Include**:
  - Sample PDF demonstrating the issue (minimized if possible).
  - Environment details: OS, Python version, `pikepdf` and `PyMuPDF` versions.
  - Expected vs. actual behavior (e.g. unexpected memory spike or process crash).

We will acknowledge receipt within 48 hours and coordinate a fix and patch release before public disclosure.

---

## Security & Isolation Invariants

`watermark-remover-auto-agent-tool` (`wmclean`) enforces defensive invariants across all execution paths:

1. **Deterministic Exception Firewall**:
   - The CLI and orchestrator catch all unexpected exceptions (`OSError`, `MemoryError`, parser faults).
   - No raw python tracebacks or system paths are dumped to unauthenticated logs.
   - Faults are safely mapped to deterministic exit codes (`30` for corrupt syntax, `40` for I/O bounds, `50` for internal errors).

2. **Atomic File Swapping**:
   - The tool never writes directly to the destination PDF or report JSON.
   - All writes occur on temporary files (`<target>.tmp.<pid>`) in the same directory and are committed via `os.replace()` only after successful completion and verification.
   - In-place operations (`wmclean run input.pdf -o input.pdf`) safely decouple reads from writes.

3. **Memory Safety & Resource Guardrails**:
   - **No GPU Requirement**: 100% CPU-only execution with native SIMD acceleration.
   - **RAM Ceiling**: Hard memory guard with peak RSS monitoring (< 2.0 GB).
   - **Resource Management**: All `pikepdf.Pdf`, `pymupdf.Document`, and `sqlite3.Connection` instances use deterministic context managers and explicit buffer releases to prevent C-heap exhaustion during large batch operations.

4. **Zero Code Execution**:
   - PDF JavaScript (`/JS`), Launch actions (`/Launch`), and executable attachments are neither executed nor rendered.
