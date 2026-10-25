"""L4 Verification & Escalation Engine package."""

from wmclean.verify.engine import VerificationEngine
from wmclean.verify.exact import verify_exact
from wmclean.verify.hardened_math_qa import (
    FourInvariantQAResult,
    evaluate_four_invariant_qa,
    remove_periodic_tiled_watermark_fft,
)
from wmclean.verify.raster import verify_raster
from wmclean.verify.text_protect import compute_fg_protection_mask, measure_stroke_recall_loss

__all__ = [
    "verify_exact",
    "verify_raster",
    "VerificationEngine",
    "compute_fg_protection_mask",
    "measure_stroke_recall_loss",
    "FourInvariantQAResult",
    "evaluate_four_invariant_qa",
    "remove_periodic_tiled_watermark_fft",
]

