"""Test Tier 1 Two-Regime Fill and Fast Patch Tournament."""

import time

from tests.benchmarks.fixture_generator import generate_benchmark_suite
from wmclean.removers.two_regime import two_regime_fill
from wmclean.router.tournament import run_remover_tournament
from wmclean.schemas import RemoverMethod
from wmclean.verify.raster import verify_raster


def test_two_regime_fill_bm01():
    """Two-regime fill must cleanly remove diagonal watermark on BM_01 and pass verification."""
    fixtures = generate_benchmark_suite(quick=True)
    f01 = fixtures[0]  # BM_01_diag_gray_150

    cleaned = two_regime_fill(f01.watermarked_bgr)
    v_res = verify_raster(f01.watermarked_bgr, cleaned)

    assert v_res.passed is True
    assert v_res.residual <= 0.035
    assert v_res.content_loss <= 0.02


def test_fast_patch_tournament_speed_and_selection():
    """Tournament must execute on a 256x256 crop in < 50ms and pick a valid remover."""
    fixtures = generate_benchmark_suite(quick=True)
    f01 = fixtures[0]

    t0 = time.perf_counter()
    best_method, best_func, best_score = run_remover_tournament(f01.watermarked_bgr)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    assert duration_ms < 50.0, f"Tournament took {duration_ms:.1f}ms, expected < 50ms"
    assert best_method in (RemoverMethod.TWO_REGIME_FILL, RemoverMethod.RASTER_TEMPLATE_WHITEN)
    assert best_score < 0.10
