"""L4 Verification & Escalation Engine package."""

from wmclean.verify.engine import VerificationEngine
from wmclean.verify.exact import verify_exact
from wmclean.verify.raster import verify_raster
from wmclean.verify.text_protect import compute_fg_protection_mask, measure_stroke_recall_loss

__all__ = [
    "verify_exact",
    "verify_raster",
    "VerificationEngine",
    "compute_fg_protection_mask",
    "measure_stroke_recall_loss",
]

