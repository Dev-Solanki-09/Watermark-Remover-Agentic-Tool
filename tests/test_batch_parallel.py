"""Tests for CLI batch processing, bounded parallel execution, and aggregate BatchReport."""

import shutil
from pathlib import Path

from tests.fixture_gen import ALL_FIXTURE_GENERATORS
from wmclean.cli import main
from wmclean.schemas import BatchReport, ExecutionMode


def test_batch_execution_with_aggregate_report(fixtures_dir: Path, tmp_path: Path):
    """Test batch processing across multiple PDFs produces an aggregate BatchReport."""
    in_dir = tmp_path / "in_batch"
    out_dir = tmp_path / "out_batch"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Place two synthetic fixtures in input directory
    wm1, _, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    wm2, _, _ = ALL_FIXTURE_GENERATORS["FX_G02"](fixtures_dir)

    shutil.copy(wm1, in_dir / "doc1.pdf")
    shutil.copy(wm2, in_dir / "doc2.PDF")  # Test case-insensitivity (.PDF)

    rep_json = tmp_path / "batch_report.json"

    # Run batch with workers=2 and mode=fast
    exit_code = main([
        "batch",
        str(in_dir),
        "-o",
        str(out_dir),
        "--mode",
        "fast",
        "--workers",
        "2",
        "--report",
        str(rep_json),
    ])

    assert exit_code == 0
    assert (out_dir / "doc1.pdf").exists()
    assert (out_dir / "doc2.PDF").exists()
    assert rep_json.exists()

    report = BatchReport.model_validate_json(rep_json.read_text(encoding="utf-8"))
    assert report.total_files == 2
    assert report.success_count == 2
    assert report.error_count == 0
    assert report.mode == ExecutionMode.FAST
    assert len(report.files) == 2



def test_batch_recursive_discovery(fixtures_dir: Path, tmp_path: Path):
    """Test batch --recursive discovers files in nested subdirectories."""
    in_dir = tmp_path / "in_nested"
    sub_dir = in_dir / "subfolder" / "level2"
    out_dir = tmp_path / "out_nested"
    sub_dir.mkdir(parents=True, exist_ok=True)

    wm1, _, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    shutil.copy(wm1, sub_dir / "nested_doc.pdf")

    rep_json = tmp_path / "nested_report.json"

    exit_code = main([
        "batch",
        str(in_dir),
        "-o",
        str(out_dir),
        "--recursive",
        "--mode",
        "fast",
        "--workers",
        "1",
        "--report",
        str(rep_json),
    ])

    assert exit_code == 0
    assert rep_json.exists()

    report = BatchReport.model_validate_json(rep_json.read_text(encoding="utf-8"))
    assert report.total_files == 1
    assert report.success_count == 1

