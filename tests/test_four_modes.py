"""Tests for the 4 execution modes (FAST, ACCURATE, ADAPTIVE, HYBRID) and ExecutionPolicy enforcement."""

import json
from pathlib import Path

from tests.fixture_gen import ALL_FIXTURE_GENERATORS
from wmclean.cli import main
from wmclean.orchestrator import Orchestrator
from wmclean.schemas import DocumentReport, ExecutionMode, ExecutionPolicy


def test_mode_policy_configuration():
    """Verify default policy matrices for FAST, ACCURATE, ADAPTIVE, HYBRID."""
    fast_p = ExecutionPolicy.for_mode(ExecutionMode.FAST)
    assert fast_p.mode == ExecutionMode.FAST
    assert fast_p.allow_llm is False
    assert fast_p.candidate_limit == 1
    assert fast_p.evaluate_multiple_candidates is False

    acc_p = ExecutionPolicy.for_mode(ExecutionMode.ACCURATE)
    assert acc_p.mode == ExecutionMode.ACCURATE
    assert acc_p.allow_llm is True
    assert acc_p.enforce_four_invariants is True

    adap_p = ExecutionPolicy.for_mode(ExecutionMode.ADAPTIVE)
    assert adap_p.mode == ExecutionMode.ADAPTIVE
    assert adap_p.candidate_limit == 1
    assert adap_p.enforce_four_invariants is True

    hyb_p = ExecutionPolicy.for_mode(ExecutionMode.HYBRID)
    assert hyb_p.mode == ExecutionMode.HYBRID
    assert hyb_p.candidate_limit >= 3
    assert hyb_p.evaluate_multiple_candidates is True



def test_orchestrator_fast_mode_disallows_llm(tmp_path: Path):
    """FAST mode must strictly disable LLM and ML regardless of caller flags."""
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 0>>endobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \ntrailer<</Size 3/Root 1 0 R>>\nstartxref\n99\n%%EOF\n")
    out_pdf = tmp_path / "out.pdf"

    orch = Orchestrator(
        input_path=str(dummy_pdf),
        output_path=str(out_pdf),
        mode="fast",
        enable_ml=True,
    )
    assert orch.mode == ExecutionMode.FAST
    assert orch.policy.allow_llm is False
    assert orch.enable_ml is False


def test_orchestrator_runs_in_fast_mode(fixtures_dir: Path, tmp_path: Path):
    """Verify FAST mode executes end-to-end and writes mode in report."""
    wm_path, _, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    out_pdf = tmp_path / "out_fast.pdf"
    rep_json = tmp_path / "rep_fast.json"

    orch = Orchestrator(
        input_path=str(wm_path),
        output_path=str(out_pdf),
        report_path=str(rep_json),
        mode=ExecutionMode.FAST,
    )
    code = orch.execute()
    assert code == 0
    assert out_pdf.exists()
    assert rep_json.exists()

    rep = DocumentReport.model_validate_json(rep_json.read_text(encoding="utf-8"))
    assert rep.mode == ExecutionMode.FAST
    assert rep.exit == 0


def test_orchestrator_runs_in_hybrid_mode(fixtures_dir: Path, tmp_path: Path):
    """Verify HYBRID mode generates multiple candidates and selects passing candidate."""
    wm_path, _, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    out_pdf = tmp_path / "out_hyb.pdf"
    rep_json = tmp_path / "rep_hyb.json"

    orch = Orchestrator(
        input_path=str(wm_path),
        output_path=str(out_pdf),
        report_path=str(rep_json),
        mode="hybrid",
    )
    code = orch.execute()
    assert code == 0
    assert out_pdf.exists()

    rep = DocumentReport.model_validate_json(rep_json.read_text(encoding="utf-8"))
    assert rep.mode == ExecutionMode.HYBRID
    assert rep.exit == 0


def test_cli_mode_flag_parsing(fixtures_dir: Path, tmp_path: Path):
    """CLI flag --mode must propagate to DocumentReport."""
    wm_path, _, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    out_pdf = tmp_path / "out_cli.pdf"
    rep_json = tmp_path / "rep_cli.json"

    exit_code = main(["run", str(wm_path), "-o", str(out_pdf), "--mode", "accurate", "--report", str(rep_json)])
    assert exit_code == 0
    assert rep_json.exists()

    rep_data = json.loads(rep_json.read_text(encoding="utf-8"))
    assert rep_data.get("mode") == "accurate"
