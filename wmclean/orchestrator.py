"""Two-pass streaming coordinator, checkpoint/resume manager, RAM watchdog.

Pass 1: L0 Probe all pages → L1 Cluster → L2 Route
Pass 2: Per-page L3 Remove → L4 Verify → L5 Assemble
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pikepdf

from wmclean.schemas import (
    CountsReport,
    DocumentReport,
    ExitCode,
    PageReport,
    PageState,
    PerfReport,
    RemoverMethod,
)


class Orchestrator:
    """Main orchestrator for single-file watermark removal."""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        report_path: Optional[str] = None,
        password: Optional[str] = None,
        resume: bool = False,
        enable_ml: bool = False,
        allow_size_growth: bool = False,
        allow_break_signatures: bool = True,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.report_path = Path(report_path) if report_path else None
        self.password = password
        self.resume = resume
        self.enable_ml = enable_ml
        self.allow_size_growth = allow_size_growth
        self.allow_break_signatures = allow_break_signatures

    def execute(self) -> int:
        """Run the full pipeline. Returns exit code."""
        start_time = time.monotonic()

        # --- Open input PDF ---
        try:
            pdf = self._open_pdf()
        except PasswordError:
            self._emit_stdout("UNSUPPORTED", ExitCode.UNSUPPORTED, "ERR_ENCRYPTED")
            return ExitCode.UNSUPPORTED.value
        except CorruptError:
            self._emit_stdout("CORRUPT", ExitCode.CORRUPT, "ERR_CORRUPT_PDF")
            return ExitCode.CORRUPT.value

        # --- Pass 1: Probe → Cluster → Route ---
        probe_records = self._pass1_probe(pdf)
        clusters, router_decision = self._pass1_cluster_and_route(probe_records)

        # --- Pass 2: Remove → Verify → Assemble ---
        page_reports = self._pass2_process(pdf, probe_records, router_decision)

        # --- Save output atomically ---
        all_untouched = all(pr.st == PageState.UNTOUCHED for pr in page_reports)
        tmp_out = self.output_path.with_name(f"{self.output_path.name}.tmp.{os.getpid()}")
        if all_untouched and self.input_path.resolve() != self.output_path.resolve():
            import shutil

            pdf.close()
            shutil.copy2(str(self.input_path), str(tmp_out))
        else:
            pdf.save(str(tmp_out))
            pdf.close()
        os.replace(tmp_out, self.output_path)

        # --- Compute results ---
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        exit_code = self._compute_exit_code(page_reports)

        # --- Write report ---
        if self.report_path:
            self._write_report(page_reports, elapsed_ms, exit_code)

        # --- Stdout summary ---
        counts = self._count_states(page_reports)
        self._emit_stdout_summary(counts, elapsed_ms, exit_code, page_reports)

        return exit_code.value

    def _open_pdf(self) -> pikepdf.Pdf:
        """Open input PDF, handling encryption and corruption."""
        try:
            return pikepdf.open(
                str(self.input_path),
                password=self.password or "",
                attempt_recovery=True,
            )
        except pikepdf.PasswordError:
            raise PasswordError(f"Encrypted PDF requires password: {self.input_path}")
        except pikepdf.PdfError as exc:
            raise CorruptError(f"Corrupt PDF: {self.input_path}: {exc}")

    def _pass1_probe(self, pdf: pikepdf.Pdf) -> list:
        """L0: Probe all pages. Returns list of ProbeRecords."""
        import pymupdf

        from wmclean.probe.raster_stats import probe_raster
        from wmclean.probe.structural import probe_structural

        records = []
        fitz_doc = None
        try:
            fitz_doc = pymupdf.open(str(self.input_path))
            if self.password:
                fitz_doc.authenticate(self.password)
        except Exception:
            fitz_doc = None

        try:
            for page_num in range(len(pdf.pages)):
                fitz_page = fitz_doc[page_num] if fitz_doc and page_num < len(fitz_doc) else None
                record = probe_structural(pdf, page_num, fitz_page=fitz_page)
                record = probe_raster(pdf, page_num, record, fitz_doc=fitz_doc)
                records.append(record)
        finally:
            if fitz_doc is not None:
                fitz_doc.close()

        return records

    def _pass1_cluster_and_route(self, probe_records: list) -> tuple:
        """L1 + L2: Cluster pages and route removal chains."""
        from wmclean.cluster.signature import compute_clusters
        from wmclean.router.rules import route_deterministic

        clusters = compute_clusters(probe_records)
        decision = route_deterministic(clusters, probe_records)
        return clusters, decision

    def _pass2_process(self, pdf: pikepdf.Pdf, probe_records: list, decision) -> list[PageReport]:
        """L3 → L4 → L5 for each page with checkpoint/resume and VerificationEngine."""
        from wmclean.cluster.signature import compute_signature
        from wmclean.removers.raster_ops import (
            raster_alpha_inv,
            raster_bitonal_despeckle,
            raster_cc_inpaint,
            raster_color_unmix,
            raster_fft_notch,
            raster_luma_sep,
            raster_template_whiten,
        )
        from wmclean.removers.vector_ops import (
            annot_del,
            artifact_strip,
            clip_scope_strip,
            ocg_strip,
            pattern_sh_strip,
            rev_rollback,
            smask_gs_strip,
            text_op_strip,
            vec_path_strip,
            xobj_do_strip,
        )
        from wmclean.verify.engine import VerificationEngine

        method_map = {
            RemoverMethod.ANNOT_DEL: annot_del,
            RemoverMethod.OCG_STRIP: ocg_strip,
            RemoverMethod.ARTIFACT_STRIP: artifact_strip,
            RemoverMethod.TEXT_OP_STRIP: text_op_strip,
            RemoverMethod.CLIP_SCOPE_STRIP: clip_scope_strip,
            RemoverMethod.XOBJ_DO_STRIP: xobj_do_strip,
            RemoverMethod.VEC_PATH_STRIP: vec_path_strip,
            RemoverMethod.PATTERN_SH_STRIP: pattern_sh_strip,
            RemoverMethod.SMASK_GS_STRIP: smask_gs_strip,
            RemoverMethod.REV_ROLLBACK: rev_rollback,
            RemoverMethod.RASTER_TEMPLATE_WHITEN: raster_template_whiten,
            RemoverMethod.RASTER_ALPHA_INV: raster_alpha_inv,
            RemoverMethod.RASTER_COLOR_UNMIX: raster_color_unmix,
            RemoverMethod.RASTER_LUMA_SEP: raster_luma_sep,
            RemoverMethod.RASTER_FFT_NOTCH: raster_fft_notch,
            RemoverMethod.RASTER_CC_INPAINT: raster_cc_inpaint,
            RemoverMethod.RASTER_BITONAL_DESPECKLE: raster_bitonal_despeckle,
        }

        action_map = {act.sig_hash: act for act in decision.clusters}
        engine = VerificationEngine()
        page_reports = []
        state_file = self.output_path.with_name(f"{self.output_path.name}.wmclean_state.json")
        saved_state = {}
        if self.resume and state_file.exists():
            try:
                saved_state = json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                saved_state = {}

        for i, record in enumerate(probe_records):
            p_num = i + 1
            if self.resume and str(p_num) in saved_state:
                pr = PageReport.model_validate(saved_state[str(p_num)])
                page_reports.append(pr)
                continue

            t0 = time.monotonic()
            sig = compute_signature(record)
            action = action_map.get(sig.sig_hash)

            if not action or action.action == "SKIP" or not action.chain:
                pr = PageReport(
                    p=p_num,
                    st=PageState.UNTOUCHED,
                    ms=int((time.monotonic() - t0) * 1000),
                )
                page_reports.append(pr)
                if self.resume:
                    saved_state[str(p_num)] = pr.model_dump()
                continue

            chain_funcs: list[tuple[RemoverMethod, Any, Any]] = []
            for m in action.chain:
                fn = method_map.get(m)
                if fn:
                    chain_funcs.append((m, fn, action.params))

            if not chain_funcs:
                pr = PageReport(
                    p=p_num,
                    st=PageState.UNTOUCHED,
                    ms=int((time.monotonic() - t0) * 1000),
                )
                page_reports.append(pr)
                continue

            state, applied, v_result = engine.apply_with_rollback(
                pdf=pdf,
                page_idx=i,
                page_kind=record.kind,
                remover_funcs=chain_funcs,
                known_wm_texts=action.params.target_texts,
            )

            pr = PageReport(
                p=p_num,
                st=state,
                m=applied,
                res=round(v_result.residual, 4) if v_result.residual else 0.0,
                loss=round(v_result.content_loss, 4) if v_result.content_loss else 0.0,
                ms=int((time.monotonic() - t0) * 1000),
                err=v_result.failure_code.value if v_result.failure_code else None,
            )
            page_reports.append(pr)
            if self.resume:
                saved_state[str(p_num)] = pr.model_dump()
                try:
                    state_file.write_text(json.dumps(saved_state), encoding="utf-8")
                except Exception:
                    pass

        if self.resume and state_file.exists():
            try:
                state_file.unlink()
            except Exception:
                pass

        return page_reports

    def _compute_exit_code(self, page_reports: list[PageReport]) -> ExitCode:
        """Determine process exit code from page states."""
        has_flagged = any(pr.st == PageState.FLAGGED for pr in page_reports)
        if has_flagged:
            return ExitCode.PARTIAL
        return ExitCode.SUCCESS

    def _count_states(self, page_reports: list[PageReport]) -> CountsReport:
        """Count page states."""
        counts = CountsReport(total=len(page_reports))
        for pr in page_reports:
            if pr.st == PageState.CLEAN_EXACT:
                counts.exact += 1
            elif pr.st == PageState.CLEAN_VERIFIED:
                counts.verified += 1
            elif pr.st == PageState.UNTOUCHED:
                counts.untouched += 1
            elif pr.st == PageState.FLAGGED:
                counts.flagged += 1
        return counts

    def _write_report(
        self, page_reports: list[PageReport], elapsed_ms: int, exit_code: ExitCode
    ) -> None:
        """Write r.json report atomically."""
        if not self.report_path:
            return
        counts = self._count_states(page_reports)
        in_size = self.input_path.stat().st_size
        out_size = self.output_path.stat().st_size if self.output_path.exists() else 0

        report = DocumentReport.model_validate(
            {
                "in": str(self.input_path),
                "out": str(self.output_path),
                "sha256_in": self._file_sha256(self.input_path),
                "sha256_out": self._file_sha256(self.output_path) if self.output_path.exists() else "",
                "exit": exit_code.value,
                "counts": counts,
                "perf": PerfReport(
                    ms=elapsed_ms,
                    peak_mb=self._get_peak_rss_mb(),
                    size_ratio=round(out_size / in_size, 4) if in_size > 0 else 1.0,
                ),
                "pages": page_reports,
            }
        )
        report_json = report.model_dump_json(indent=2, by_alias=True)
        tmp_report = self.report_path.with_name(f"{self.report_path.name}.tmp.{os.getpid()}")
        tmp_report.write_text(report_json, encoding="utf-8")
        os.replace(tmp_report, self.report_path)

    def _emit_stdout_summary(
        self,
        counts: CountsReport,
        elapsed_ms: int,
        exit_code: ExitCode,
        page_reports: list[PageReport],
    ) -> None:
        """Print ≤300-token stdout summary."""
        status_map = {
            ExitCode.SUCCESS: "OK",
            ExitCode.PARTIAL: "PARTIAL",
            ExitCode.UNSUPPORTED: "UNSUPPORTED",
            ExitCode.CORRUPT: "CORRUPT",
            ExitCode.IO_ERROR: "IO_ERROR",
            ExitCode.INTERNAL: "INTERNAL_ERROR",
        }
        status = status_map.get(exit_code, "UNKNOWN")

        in_size_mb = self.input_path.stat().st_size / (1024 * 1024)
        out_size_mb = (
            self.output_path.stat().st_size / (1024 * 1024)
            if self.output_path.exists()
            else 0.0
        )
        ratio = out_size_mb / in_size_mb if in_size_mb > 0 else 1.0

        lines = [
            f"WMCLEAN v{__import__('wmclean').__version__} | {self.input_path.name} -> {self.output_path.name} | status={status} (exit {exit_code.value})",
            f"pages={counts.total} | CLEAN_EXACT={counts.exact} | CLEAN_VERIFIED={counts.verified} | UNTOUCHED={counts.untouched} | FLAGGED={counts.flagged}",
            f"size_in={in_size_mb:.1f}MB | size_out={out_size_mb:.1f}MB ({ratio:.2f}x) | time={elapsed_ms / 1000:.1f}s | ram_peak={self._get_peak_rss_mb()}MB | llm_calls=0",
        ]

        flagged = [pr for pr in page_reports if pr.st == PageState.FLAGGED]
        if flagged:
            lines.append("flagged_pages:")
            for pr in flagged[:20]:
                lines.append(f"  p{pr.p}: {pr.err or 'UNKNOWN'}")
            if len(flagged) > 20:
                lines.append(f"  ... +{len(flagged) - 20} more in r.json")

        if self.report_path:
            lines.append(f"report={self.report_path}")

        print("\n".join(lines))

    def _emit_stdout(self, status: str, exit_code: ExitCode, reason: str) -> None:
        """Print minimal error stdout."""
        print(
            f"WMCLEAN v{__import__('wmclean').__version__} | {self.input_path.name} | "
            f"status={status} (exit {exit_code.value}) reason={reason}"
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """Compute SHA-256 of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _get_peak_rss_mb() -> int:
        """Get peak resident set size in MB."""
        if sys.platform != "win32":
            try:
                import resource

                usage = resource.getrusage(resource.RUSAGE_SELF)
                if sys.platform == "darwin":
                    return int(usage.ru_maxrss / (1024 * 1024))
                return int(usage.ru_maxrss / 1024)
            except Exception:
                pass
        try:
            import psutil

            return int(psutil.Process().memory_info().peak_wset / (1024 * 1024))
        except Exception:
            return 0


class PasswordError(Exception):
    """Raised when PDF requires a password not provided."""


class CorruptError(Exception):
    """Raised when PDF is unrecoverably corrupt."""
