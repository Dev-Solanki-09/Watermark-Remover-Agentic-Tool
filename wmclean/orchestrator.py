"""Two-pass streaming coordinator, checkpoint/resume manager, RAM watchdog.

Pass 1: L0 Probe all pages → L1 Cluster → L2 Route
Pass 2: Per-page L3 Remove → L4 Verify → L5 Assemble
"""

from __future__ import annotations

import hashlib
import json
import logging
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
    RemoverParams,
)

logger = logging.getLogger(__name__)


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
        steer_dsl: Optional[str] = None,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.report_path = Path(report_path) if report_path else None
        self.password = password
        self.resume = resume
        self.enable_ml = enable_ml
        self.allow_size_growth = allow_size_growth
        self.allow_break_signatures = allow_break_signatures
        self.steer_dsl = steer_dsl

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

        from wmclean.probe.raster_stats import RasterReservoir, probe_raster
        from wmclean.probe.structural import probe_structural

        records = []
        fitz_doc = None
        try:
            fitz_doc = pymupdf.open(str(self.input_path))
            if self.password:
                fitz_doc.authenticate(self.password)
        except Exception:
            fitz_doc = None

        reservoir = RasterReservoir()
        self.reservoir = reservoir
        try:
            for page_num in range(len(pdf.pages)):
                fitz_page = fitz_doc[page_num] if fitz_doc and page_num < len(fitz_doc) else None
                record = probe_structural(pdf, page_num, fitz_page=fitz_page)
                record = probe_raster(pdf, page_num, record, reservoir=reservoir, fitz_doc=fitz_doc)
                records.append(record)

            # Compute cross-page document frequency for vector path candidates
            total_doc_pages = len(records)
            if total_doc_pages > 0:
                path_page_counts: dict[str, set[int]] = {}
                for p_idx, rec in enumerate(records):
                    for pc in rec.path_cand:
                        path_page_counts.setdefault(pc.hash, set()).add(p_idx)
                for rec in records:
                    for pc in rec.path_cand:
                        pc.doc_freq = len(path_page_counts.get(pc.hash, set())) / float(
                            total_doc_pages
                        )

            # Subdocument-scoped template computation (Apex v6.0 Gate 1)
            from wmclean.intelligence.gatekeeper import Gatekeeper

            gk = Gatekeeper(target_dpi=150)
            subdoc_ids = gk.segment_subdocuments(records)

            sample_by_page = {}
            for bucket in reservoir.buckets.values():
                for p_idx, s_grey in zip(bucket.page_indices, bucket.samples):
                    sample_by_page[p_idx] = s_grey

            subdoc_groups: dict[int, list[int]] = {}
            for p_idx, s_id in enumerate(subdoc_ids):
                subdoc_groups.setdefault(s_id, []).append(p_idx)

            import numpy as np

            subdoc_templates = {}
            from wmclean.probe.raster_stats import _compute_template_ncc

            for s_id, p_indices in subdoc_groups.items():
                s_samples = [sample_by_page[idx] for idx in p_indices if idx in sample_by_page]
                if len(s_samples) >= 3:
                    target_shape = s_samples[0].shape
                    comp = [s for s in s_samples if s.shape == target_shape]
                    if len(comp) >= 3:
                        subdoc_templates[s_id] = np.median(np.stack(comp, axis=0), axis=0).astype(
                            np.uint8
                        )

            self.subdoc_templates = subdoc_templates

            # Update every raster page with its subdocument-scoped template NCC
            for p_idx, rec in enumerate(records):
                if rec.ras_stats and rec.ras_stats.has_raster:
                    s_id = subdoc_ids[p_idx] if p_idx < len(subdoc_ids) else 0
                    s_tmpl = subdoc_templates.get(s_id)
                    page_sample = sample_by_page.get(p_idx)
                    if (
                        s_tmpl is not None
                        and page_sample is not None
                        and page_sample.shape == s_tmpl.shape
                    ):
                        doc_shield = None
                        try:
                            from wmclean.intelligence.doc_shield import parse_doc_shield

                            doc_shield = parse_doc_shield(page_sample)
                        except Exception:
                            doc_shield = None
                        ncc, cov = _compute_template_ncc(page_sample, s_tmpl, doc_shield=doc_shield)
                        rec.ras_stats.tmpl_ncc = ncc
                        rec.ras_stats.tmpl_cov = cov
                        if ncc >= 0.5:
                            rec.ras_stats.pos_stab = ncc
                        if ncc >= 0.35:
                            from wmclean.probe.raster_stats import _estimate_watermark_alpha

                            rec.ras_stats.est_alpha, rec.ras_stats.est_wm_rgb = (
                                _estimate_watermark_alpha(page_sample, s_tmpl)
                            )
                        else:
                            rec.ras_stats.est_alpha = 0.0
                    else:
                        rec.ras_stats.tmpl_ncc = 0.0
                        rec.ras_stats.tmpl_cov = 0.0
                        rec.ras_stats.pos_stab = 0.0
                        rec.ras_stats.est_alpha = 0.0
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
        from wmclean.intelligence.gatekeeper import Gatekeeper
        from wmclean.removers.bilingual_clone import bilingual_twin_clone
        from wmclean.removers.density_sub import density_sub
        from wmclean.removers.font_zero import font_table_zero
        from wmclean.removers.maxtree_filter import maxtree_filter
        from wmclean.removers.mupdf_layer_redact import mupdf_layer_redact
        from wmclean.removers.piracy_ocr import piracy_ocr_spotter, swt_morph_reconstruct
        from wmclean.removers.pointwise_unmix import pointwise_hue_and_lut
        from wmclean.removers.raster_ops import (
            illum_flatten,
            inverted_banner_strip,
            raster_alpha_inv,
            raster_bitonal_despeckle,
            raster_cc_inpaint,
            raster_color_unmix,
            raster_dark_stamp_consensus,
            raster_fft_notch,
            raster_hue_selective_unmix,
            raster_luma_sep,
            raster_margin_banner,
            raster_registered_outline_clean,
            raster_template_whiten,
        )
        from wmclean.removers.skeleton_graph import skeleton_graph
        from wmclean.removers.stain_deconv import beer_lambert_stain_unmix
        from wmclean.removers.surgical_lama import surgical_lama
        from wmclean.removers.vector_ops import (
            annot_del,
            artifact_strip,
            ast_tj_reassemble,
            clip_scope_strip,
            contents_array_slice,
            ocg_strip,
            pattern_sh_strip,
            rev_rollback,
            smask_gs_strip,
            text_op_strip,
            vec_path_strip,
            vec_spatial_band_cluster,
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
            RemoverMethod.AST_TJ_REASSEMBLE: ast_tj_reassemble,
            RemoverMethod.VEC_SPATIAL_BAND_CLUSTER: vec_spatial_band_cluster,
            RemoverMethod.CONTENTS_ARRAY_SLICE: contents_array_slice,
            RemoverMethod.FONT_TABLE_ZERO: font_table_zero,
            RemoverMethod.MUPDF_LAYER_REDACT: mupdf_layer_redact,
            RemoverMethod.RASTER_TEMPLATE_WHITEN: raster_template_whiten,
            RemoverMethod.RASTER_ALPHA_INV: raster_alpha_inv,
            RemoverMethod.RASTER_COLOR_UNMIX: raster_color_unmix,
            RemoverMethod.RASTER_HUE_SELECTIVE_UNMIX: raster_hue_selective_unmix,
            RemoverMethod.RASTER_LUMA_SEP: raster_luma_sep,
            RemoverMethod.RASTER_FFT_NOTCH: raster_fft_notch,
            RemoverMethod.RASTER_CC_INPAINT: raster_cc_inpaint,
            RemoverMethod.RASTER_BITONAL_DESPECKLE: raster_bitonal_despeckle,
            RemoverMethod.RASTER_MARGIN_BANNER: raster_margin_banner,
            RemoverMethod.INVERTED_BANNER_STRIP: inverted_banner_strip,
            RemoverMethod.RASTER_DARK_STAMP_CONSENSUS: raster_dark_stamp_consensus,
            RemoverMethod.RASTER_REGISTERED_OUTLINE_CLEAN: raster_registered_outline_clean,
            RemoverMethod.PIRACY_OCR_SPOTTER: piracy_ocr_spotter,
            RemoverMethod.ILLUM_FLATTEN: illum_flatten,
            RemoverMethod.SWT_MORPH_RECONSTRUCT: swt_morph_reconstruct,
            RemoverMethod.BEER_LAMBERT_STAIN_UNMIX: beer_lambert_stain_unmix,
            RemoverMethod.BILINGUAL_TWIN_CLONE: bilingual_twin_clone,
            RemoverMethod.DENSITY_SUB: density_sub,
            RemoverMethod.POINTWISE_HUE_AND_LUT: pointwise_hue_and_lut,
            RemoverMethod.MAXTREE_FILTER: maxtree_filter,
            RemoverMethod.SKELETON_GRAPH: skeleton_graph,
            RemoverMethod.SURGICAL_LAMA: surgical_lama,
        }

        action_map = {act.sig_hash: act for act in decision.clusters}
        engine = VerificationEngine()
        gatekeeper = Gatekeeper(target_dpi=150)
        dispatch_plans = gatekeeper.plan_document(
            probe_records, reservoir=getattr(self, "reservoir", None)
        )
        plan_map = {p.page: p for p in dispatch_plans}

        parsed_steer_rules = []
        if getattr(self, "steer_dsl", None) and self.steer_dsl is not None:
            try:
                from wmclean.intelligence.steering_dsl import parse_steer_dsl

                parsed_steer_rules = parse_steer_dsl(self.steer_dsl)
            except Exception as exc:
                logger.warning("Failed to parse --steer DSL: %s", exc)

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
            page_plan = plan_map.get(p_num)

            # L0.5 Gatekeeper negative presence veto (unless overridden by --steer):
            matching_steer = [
                r for r in parsed_steer_rules if r.matches_page(p_num, len(probe_records))
            ]
            if not matching_steer and page_plan and page_plan.is_untouched:
                pr = PageReport(
                    p=p_num,
                    st=PageState.UNTOUCHED,
                    ms=int((time.monotonic() - t0) * 1000),
                )
                page_reports.append(pr)
                if self.resume:
                    saved_state[str(p_num)] = pr.model_dump()
                continue

            if not matching_steer and (not action or action.action == "SKIP" or not action.chain):
                if not (page_plan and page_plan.tasks):
                    pr = PageReport(
                        p=p_num,
                        st=PageState.UNTOUCHED,
                        ms=int((time.monotonic() - t0) * 1000),
                    )
                    page_reports.append(pr)
                    if self.resume:
                        saved_state[str(p_num)] = pr.model_dump()
                    continue

            chain_funcs: list[
                tuple[RemoverMethod, Any, RemoverParams, Optional[tuple[int, int, int, int]]]
            ] = []
            default_params = action.params if action else RemoverParams()

            if matching_steer:
                steer_chain: list[
                    tuple[RemoverMethod, Any, RemoverParams, Optional[tuple[int, int, int, int]]]
                ] = []
                is_skip = False
                for r in matching_steer:
                    for act in r.actions:
                        if act.name in ("skip", "restore_original"):
                            is_skip = True
                            break
                        elif act.name == "protect_zone":
                            pass
                        elif act.name == "auto_solve":
                            steer_chain.append(
                                (
                                    RemoverMethod.POINTWISE_HUE_AND_LUT,
                                    pointwise_hue_and_lut,
                                    RemoverParams(),
                                    None,
                                )
                            )
                        else:
                            try:
                                meth = RemoverMethod(act.name)
                                fn = method_map.get(meth)
                                if fn:
                                    z_bbox = None
                                    if r.zones and r.zones[0].name != "full":
                                        p_obj = pdf.pages[i]
                                        mb = [float(x) for x in p_obj.mediabox]
                                        w_px = int((mb[2] - mb[0]) * 150.0 / 72.0)
                                        h_px = int((mb[3] - mb[1]) * 150.0 / 72.0)
                                        z_bbox = r.zones[0].to_pixel_bbox(w_px, h_px)
                                    steer_chain.append(
                                        (meth, fn, RemoverParams(**act.params), z_bbox)
                                    )
                            except Exception:
                                pass
                if is_skip:
                    pr = PageReport(
                        p=p_num,
                        st=PageState.UNTOUCHED,
                        ms=int((time.monotonic() - t0) * 1000),
                    )
                    page_reports.append(pr)
                    if self.resume:
                        saved_state[str(p_num)] = pr.model_dump()
                    continue
                if steer_chain:
                    chain_funcs = steer_chain
            elif page_plan and page_plan.tasks:
                for t in page_plan.tasks:
                    fn = method_map.get(t.method)
                    if fn:
                        p = t.params or default_params
                        chain_funcs.append((t.method, fn, p, t.roi_bbox))
            elif action and action.chain:
                for m in action.chain:
                    fn = method_map.get(m)
                    if fn:
                        chain_funcs.append((m, fn, default_params, None))

            if not chain_funcs:
                pr = PageReport(
                    p=p_num,
                    st=PageState.UNTOUCHED,
                    ms=int((time.monotonic() - t0) * 1000),
                )
                page_reports.append(pr)
                continue

            tmpl = None
            if hasattr(self, "subdoc_templates") and self.subdoc_templates:
                p_plan = plan_map.get(p_num)
                s_id = p_plan.subdoc_id if p_plan else 0
                tmpl = self.subdoc_templates.get(s_id)
            if tmpl is None and hasattr(self, "reservoir") and self.reservoir:
                p_obj = pdf.pages[i]
                mb = [float(x) for x in p_obj.mediabox]
                w_pt = mb[2] - mb[0]
                h_pt = mb[3] - mb[1]
                tw = int(w_pt * 150.0 / 72.0)
                th = int(h_pt * 150.0 / 72.0)
                tmpl = self.reservoir.get_template(tw, th)

            state, applied, v_result = engine.apply_with_rollback(
                pdf=pdf,
                page_idx=i,
                page_kind=record.kind,
                remover_funcs=chain_funcs,
                known_wm_texts=action.params.target_texts if (action and action.params) else None,
                template=tmpl,
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
                "sha256_out": self._file_sha256(self.output_path)
                if self.output_path.exists()
                else "",
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
            self.output_path.stat().st_size / (1024 * 1024) if self.output_path.exists() else 0.0
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
