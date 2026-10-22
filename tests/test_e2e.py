"""End-to-End integration tests for wmclean pipeline and CLI across synthetic fixtures."""

from pathlib import Path

import pikepdf

from tests.fixture_gen import ALL_FIXTURE_GENERATORS
from wmclean.cli import main
from wmclean.schemas import DocumentReport


def test_e2e_vector_g01_annot(fixtures_dir: Path, tmp_path: Path):
    """E2E test: FX_G01 annotation watermark removed and verified with CLEAN_EXACT."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    out_pdf = tmp_path / "out_g01.pdf"
    report_json = tmp_path / "report_g01.json"

    exit_code = main(["run", str(wm_path), "-o", str(out_pdf), "--report", str(report_json)])
    assert exit_code == 0
    assert out_pdf.exists()
    assert report_json.exists()

    report = DocumentReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    assert report.exit == 0
    assert report.counts.exact >= 1
    assert report.counts.flagged == 0
    assert report.perf.size_ratio <= 1.30


def test_e2e_vector_g02_ocg(fixtures_dir: Path, tmp_path: Path):
    """E2E test: FX_G02 OCG layer watermark removed and verified."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G02"](fixtures_dir)
    out_pdf = tmp_path / "out_g02.pdf"
    report_json = tmp_path / "report_g02.json"

    exit_code = main(["run", str(wm_path), "-o", str(out_pdf), "--report", str(report_json)])
    assert exit_code == 0
    assert out_pdf.exists()

    report = DocumentReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    assert report.exit == 0
    assert report.counts.exact >= 1


def test_e2e_vector_g04_text(fixtures_dir: Path, tmp_path: Path):
    """E2E test: FX_G04 vector text watermark removed and verified."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G04"](fixtures_dir)
    out_pdf = tmp_path / "out_g04.pdf"
    report_json = tmp_path / "report_g04.json"

    exit_code = main(["run", str(wm_path), "-o", str(out_pdf), "--report", str(report_json)])
    assert exit_code == 0
    assert out_pdf.exists()

    report = DocumentReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    assert report.exit == 0
    assert report.counts.exact >= 1


def test_e2e_raster_g11_color(fixtures_dir: Path, tmp_path: Path):
    """E2E test: FX_G11 colour raster watermark unmixed and verified."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G11"](fixtures_dir)
    out_pdf = tmp_path / "out_g11.pdf"
    report_json = tmp_path / "report_g11.json"

    exit_code = main(["run", str(wm_path), "-o", str(out_pdf), "--report", str(report_json)])
    assert exit_code == 0
    assert out_pdf.exists()

    report = DocumentReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    assert report.exit == 0
    assert report.counts.verified >= 1


def test_e2e_raster_g15_bitonal(fixtures_dir: Path, tmp_path: Path):
    """E2E test: FX_G15 bitonal despeckled and verified."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G15"](fixtures_dir)
    out_pdf = tmp_path / "out_g15.pdf"
    report_json = tmp_path / "report_g15.json"

    exit_code = main(["run", str(wm_path), "-o", str(out_pdf), "--report", str(report_json)])
    assert exit_code == 0
    assert out_pdf.exists()

    report = DocumentReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    assert report.exit == 0
    assert report.counts.verified >= 1


def test_e2e_g19_opaque_flagged(fixtures_dir: Path, tmp_path: Path):
    """E2E test: FX_G19 opaque occlusion triggers FLAGGED with exit code 10."""
    wm_path, _, _ = ALL_FIXTURE_GENERATORS["FX_G19"](fixtures_dir)
    out_pdf = tmp_path / "out_g19.pdf"
    report_json = tmp_path / "report_g19.json"

    exit_code = main(["run", str(wm_path), "-o", str(out_pdf), "--report", str(report_json)])
    assert exit_code == 10  # PARTIAL completion
    assert out_pdf.exists()

    report = DocumentReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    assert report.exit == 10
    assert report.counts.flagged >= 1


def test_e2e_idempotency_clean_pdf(fixtures_dir: Path, tmp_path: Path):
    """E2E test: Clean PDF input produces 100% UNTOUCHED pages with zero mutation."""
    _, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G01"](fixtures_dir)
    out_pdf = tmp_path / "out_clean.pdf"
    report_json = tmp_path / "report_clean.json"

    exit_code = main(["run", str(clean_path), "-o", str(out_pdf), "--report", str(report_json)])
    assert exit_code == 0

    report = DocumentReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    assert report.exit == 0
    assert report.counts.untouched == report.counts.total
    assert report.counts.exact == 0
    assert report.counts.flagged == 0


def test_e2e_m04_encrypted(fixtures_dir: Path, tmp_path: Path):
    """E2E test: Encrypted PDF with password decrypts, cleans, and passes."""
    clean_path = tmp_path / "clean_locked.pdf"
    locked_pdf = tmp_path / "locked.pdf"

    # Create a locked PDF with non-empty user password
    from tests.fixture_gen import create_seed_vector

    create_seed_vector(clean_path, pages=1)

    pdf = pikepdf.open(str(clean_path))
    enc = pikepdf.Encryption(user="secret123", owner="owner123")
    pdf.save(str(locked_pdf), encryption=enc)
    pdf.close()

    out_pdf = tmp_path / "out_locked.pdf"
    report_json = tmp_path / "report_locked.json"

    # With correct password "secret123" -> exit code 0
    exit_code = main(
        [
            "run",
            str(locked_pdf),
            "-o",
            str(out_pdf),
            "--password",
            "secret123",
            "--report",
            str(report_json),
        ]
    )
    assert exit_code == 0
    assert out_pdf.exists()

    # Without password -> exit code 20 (UNSUPPORTED/ENCRYPTED)
    out_fail = tmp_path / "out_fail.pdf"
    exit_fail = main(["run", str(locked_pdf), "-o", str(out_fail)])
    assert exit_fail == 20
