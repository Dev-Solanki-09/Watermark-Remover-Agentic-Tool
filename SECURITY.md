# Security Policy

## Supported Versions

We provide active security updates, vulnerability patches, and bug fixes for the current release stream:

| Version | Supported |
| :--- | :---: |
| `0.1.6-beta` (`0.1.6b0`) | :white_check_mark: |
| `< 0.1.6-beta` | :x: |

---

## Reporting a Vulnerability

We take the security of PDF parsing, rendering, and content manipulation very seriously. PDFs are complex, untrusted binary formats that can contain deeply nested objects, malicious stream compressions, or corrupt cross-reference tables.

If you discover a security vulnerability (such as a memory leak, parser crash, arbitrary file read/write, or denial-of-service condition), please **do not** open a public issue. Instead, report it privately:

- **GitHub Advisory**: Open a confidential security advisory at [GitHub Security Advisories](https://github.com/Dev-Solanki-09/Watermark-Remover-Agentic-Tool/security/advisories/new).
- **Details to Include**:
  - Sample PDF demonstrating the issue (minimized if possible, redacting any sensitive or confidential data).
  - Environment details: OS, Python version, `pikepdf`, `PyMuPDF`, and `OpenCV` versions.
  - Expected vs. actual behavior (e.g., unexpected memory spike or process crash).

We will acknowledge receipt within 48 hours and coordinate a fix and patch release before public disclosure.

---

## Security & Isolation Invariants

`watermark-remover-auto-agent-tool` (`wmclean`) enforces defensive invariants across all execution paths:

### 1. Deterministic Exception Firewall
- The CLI and orchestrator catch all unexpected exceptions (`OSError`, `MemoryError`, parser faults).
- No raw python tracebacks or absolute system paths are dumped to unauthenticated logs.
- Faults are safely mapped to deterministic exit codes:
  - `20`: UNSUPPORTED (encrypted/XFA PDF)
  - `30`: CORRUPT (unrecoverable PDF syntax)
  - `40`: IO / RESOURCE ERROR (disk, memory, or permissions issue)
  - `50`: INTERNAL INVARIANT BREACH

### 2. Atomic File Swapping & In-Place Safety
- The tool never writes directly to the destination PDF or report JSON.
- All writes occur on temporary swap files (`<target>.tmp.<pid>`) in the same directory and are committed via `os.replace()` only after successful completion and verification.
- In-place operations (`wmclean run input.pdf -o input.pdf`) safely decouple reads from writes.

### 3. Fail-Closed Forensic Verification (L4 Verification Gates)
- **Zero Authentic Body Text Loss**: Strict PikePDF AST verification enforces $\Delta_{\text{text}} = 0$.
- **Foreground Stroke Loss Bound**: C++ SIMD Sauvola binarization enforces $L_{\text{fg}} \le 0.020$ (loss $< 2.0\%$).
- **Residual Watermark Bound**: $R_{\text{wm}} \le 0.035$ in Lab color space.
- **Outside-ROI Invariance**: Hard NumPy array equality assertion ($\Delta_{\text{outside}} = 0$).
- If any gate fails, the page transactionally rolls back to `FLAGGED` or `UNTOUCHED`. Authentic content is never corrupted.

### 4. Memory Safety & Resource Guardrails
- **No GPU Requirement**: 100% CPU-only execution with native SIMD acceleration.
- **RAM Ceiling**: Hard memory guard with peak RSS monitoring (< 2.0 GB).
- **Resource Management**: All `pikepdf.Pdf`, `fitz.Document`, and `sqlite3.Connection` instances use deterministic context managers and explicit buffer releases to prevent C-heap exhaustion during large batch operations.

### 5. Local-First & Zero Cloud Telemetry
- `wmclean` operates 100% locally on CPU.
- Zero outbound network requests, zero telemetry, and zero third-party cloud API dependencies.
- Completely safe for processing HIPAA-regulated healthcare documents, confidential legal contracts, and intellectual property.

### 6. Agent Token-Cap Firewall
- CLI stdout output is programmatically capped at $\le 300$ tokens ($\le 25$ lines / 1,200 characters) to prevent inadvertent dumping of document text into external LLM prompts.
- Zero-vision `peek` commands render high-density ASCII topologies (< 85 tokens) to protect agent context windows.

### 7. Zero Code Execution
- PDF JavaScript (`/JS`), Launch actions (`/Launch`), and executable attachments are neither executed nor rendered.

### 8. Document Sanitization & Secret Isolation
- Repository-level `.gitignore` rules prevent accidental commits of personal PDFs, client files, diagnostic artifacts, and API credentials.
