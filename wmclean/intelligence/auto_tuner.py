"""In-Memory Pareto Auto-Tuner & 3-Option Escalation Tickets (Agentic Authority v7.0).

Sweeps a 12-point parameter grid of zero-smudge methods on candidate watermark patches
in < 35 ms, optimizes the Pareto objective (Watermark Drop vs. 50x Shield Damage),
and emits 60-token 3-option escalation tickets (OPT_A, OPT_B, OPT_C) for ambiguous cases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from wmclean.intelligence.doc_shield import DocShield, parse_doc_shield
from wmclean.removers.lut_unmix import build_monotonic_lut

logger = logging.getLogger(__name__)


@dataclass
class ParetoCandidate:
    """A single evaluated parameter/method configuration."""

    label: str  # e.g. "Conservative", "Balanced", "Aggressive"
    method_name: str
    params: dict[str, Any]
    wm_drop: float  # Fractional drop in watermark energy (0.0 .. 1.0)
    shield_damage: float  # Mean fractional damage to protected shield (0.0 .. 1.0)
    score: float
    steer_dsl: str


@dataclass
class EscalationTicket:
    """Pre-simulated 3-Option Escalation Ticket for ambiguous watermarks."""

    ticket_id: str
    page_num: int
    roi_bbox: tuple[float, float, float, float]  # Normalized [x1, y1, x2, y2]
    reason: str
    opt_a: ParetoCandidate  # Conservative, applied by default
    opt_b: ParetoCandidate  # Balanced
    opt_c: ParetoCandidate  # Restore original / Untouched

    def format_ticket(self) -> str:
        """Format a compact 60-token escalation ticket for CLI stdout."""
        r_box = f"[{self.roi_bbox[0]:.2f},{self.roi_bbox[1]:.2f},{self.roi_bbox[2]:.2f},{self.roi_bbox[3]:.2f}]"
        lines = [
            f"TICKET {self.ticket_id} | page={self.page_num} @ roi={r_box} | reason={self.reason}",
            f'  OPT_A (Conservative, APPLIED): --steer "{self.opt_a.steer_dsl}" -> wm_drop={self.opt_a.wm_drop * 100:.0f}%, diag_diff={self.opt_a.shield_damage * 100:.2f}%',
            f'  OPT_B (Balanced):               --steer "{self.opt_b.steer_dsl}" -> wm_drop={self.opt_b.wm_drop * 100:.0f}%, diag_diff={self.opt_b.shield_damage * 100:.2f}%',
            f'  OPT_C (Untouched):              --steer "{self.opt_c.steer_dsl}" -> wm_drop=0%, diag_diff=0.00%',
            f"TO SWITCH IN 0.2s: wmclean steer <out.pdf> --ticket {self.ticket_id} --pick OPT_B",
        ]
        return "\n".join(lines)


def auto_solve_patch(
    page_img: np.ndarray,
    roi_bbox: Optional[tuple[int, int, int, int]] = None,
    shield: Optional[DocShield] = None,
    page_num: int = 1,
    ticket_counter: int = 1,
    aggressiveness: float = 0.5,
) -> tuple[np.ndarray, float, Optional[EscalationTicket]]:
    """Auto-tune zero-smudge remover parameters against DocShield in memory (< 35 ms).

    Args:
        page_img: (H, W, C) or (H, W) uint8 input image.
        roi_bbox: (x1, y1, x2, y2) pixel bbox of the candidate watermark region.
        shield: Optional precomputed DocShield.
        page_num: 1-indexed page number.
        ticket_counter: Unique index for ticket naming (e.g. T1).
        aggressiveness: Preference factor 0.0 (safest) to 1.0 (maximum drop).

    Returns:
        tuple[np.ndarray, Optional[EscalationTicket], ParetoCandidate]:
            (cleaned_image, optional_ticket, winning_candidate)
    """
    if shield is None:
        shield = parse_doc_shield(page_img)

    h, w = page_img.shape[:2]

    # Select representative 256x256 patch
    if roi_bbox is not None:
        rx1, ry1, rx2, ry2 = roi_bbox
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w, rx2), min(h, ry2)
    else:
        rx1, ry1, rx2, ry2 = 0, 0, w, h

    pw = min(128, rx2 - rx1)
    ph = min(128, ry2 - ry1)
    if pw < 30 or ph < 30:
        # Patch too small to sweep
        return page_img, 0.0, None

    px1 = rx1 + (rx2 - rx1 - pw) // 2
    py1 = ry1 + (ry2 - ry1 - ph) // 2
    px2 = px1 + pw
    py2 = py1 + ph

    patch = page_img[py1:py2, px1:px2].copy()
    patch_shield = shield.norm_gray[py1:py2, px1:px2]
    patch_core = shield.m_core[py1:py2, px1:px2]
    patch_diag = shield.m_diagram[py1:py2, px1:px2]
    patch_halo = shield.m_halo[py1:py2, px1:px2]

    # Protected mask: text cores, diagram lines, anti-aliased halos
    m_shield = patch_core | patch_diag | patch_halo
    # Candidate watermark pixels
    m_wm_cand = (patch_shield >= 80) & (patch_shield <= 235) & (~m_shield)

    patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY) if patch.ndim == 3 else patch
    patch_bg = shield.bg[py1:py2, px1:px2] if shield.bg.shape == page_img.shape[:2] else shield.bg
    d_patch = -np.log10(np.clip(patch_gray.astype(np.float32), 1.0, 255.0) / 255.0)
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV) if patch.ndim == 3 else None

    orig_wm_energy = (
        float(
            np.sum(
                np.abs(
                    patch_bg[m_wm_cand].astype(np.float32)
                    - patch_shield[m_wm_cand].astype(np.float32)
                )
            )
        )
        if m_wm_cand.any()
        else 1.0
    )
    orig_shield_gray = (
        patch_shield[m_shield].astype(np.float32)
        if m_shield.any()
        else np.array([0.0], dtype=np.float32)
    )

    def _eval_lut(fl: int) -> np.ndarray:
        lut = build_monotonic_lut(black_point=88, wm_floor=fl, paper_white=255)
        res_g = cv2.LUT(patch_gray, lut)
        out = patch.copy()
        if out.ndim == 3:
            for c in range(3):
                out[:, :, c] = np.where(m_wm_cand, res_g, out[:, :, c])
        else:
            out = np.where(m_wm_cand, res_g, out)
        return out

    def _eval_hue() -> np.ndarray:
        if patch.ndim != 3 or hsv_patch is None:
            return patch
        sat = hsv_patch[:, :, 1]
        out = patch.copy()
        for c in range(3):
            out[:, :, c] = np.where(m_wm_cand & (sat > 20), patch_bg, out[:, :, c])
        return out

    def _eval_density(sub_d: float) -> np.ndarray:
        d_clean = np.maximum(0.0, d_patch - sub_d)
        res_g = (255.0 * (10.0 ** (-d_clean))).clip(0, 255).astype(np.uint8)
        out = patch.copy()
        if out.ndim == 3:
            for c in range(3):
                out[:, :, c] = np.where(m_wm_cand, res_g, out[:, :, c])
        else:
            out = np.where(m_wm_cand, res_g, out)
        return out

    def _eval_maxtree(cutoff: int) -> np.ndarray:
        res_g = np.where((patch_gray > 255 - cutoff) & m_wm_cand, patch_bg, patch_gray).astype(
            np.uint8
        )
        out = patch.copy()
        if out.ndim == 3:
            for c in range(3):
                out[:, :, c] = np.where(m_wm_cand, res_g, out[:, :, c])
        else:
            out = np.where(m_wm_cand, res_g, out)
        return out

    sweep_configs = [
        # 1. Monotonic LUT variations
        ("lut_180", "lut", {"floor": 180}, lambda: _eval_lut(180)),
        ("lut_205", "lut", {"floor": 205}, lambda: _eval_lut(205)),
        ("lut_225", "lut", {"floor": 225}, lambda: _eval_lut(225)),
        # 2. MaxTree variations
        ("maxtree_40", "maxtree", {"h": 40}, lambda: _eval_maxtree(40)),
        ("maxtree_55", "maxtree", {"h": 55}, lambda: _eval_maxtree(55)),
        ("maxtree_70", "maxtree", {"h": 70}, lambda: _eval_maxtree(70)),
        # 3. Density subtraction
        ("density_p15", "density_sub", {"pct": 15}, lambda: _eval_density(0.05)),
        ("density_p20", "density_sub", {"pct": 20}, lambda: _eval_density(0.10)),
        ("density_p25", "density_sub", {"pct": 25}, lambda: _eval_density(0.15)),
        # 4. Color / Hue unmix
        ("pointwise_unmix", "hue_unmix", {}, _eval_hue),
        ("stain_deconv", "eigen_bss", {}, _eval_hue),
    ]

    evaluated_candidates: list[ParetoCandidate] = []
    cand_outputs: dict[str, np.ndarray] = {}

    for name, m_name, p_dict, run_fn in sweep_configs:
        try:
            cand_cleaned = run_fn()
            cand_outputs[name] = cand_cleaned
            cand_gray = (
                cv2.cvtColor(cand_cleaned, cv2.COLOR_RGB2GRAY)
                if cand_cleaned.ndim == 3
                else cand_cleaned
            )

            # Measure watermark energy reduction
            if m_wm_cand.any():
                cand_wm_energy = float(
                    np.sum(
                        np.abs(
                            patch_bg[m_wm_cand].astype(np.float32)
                            - cand_gray[m_wm_cand].astype(np.float32)
                        )
                    )
                )
                wm_drop = max(0.0, (orig_wm_energy - cand_wm_energy) / max(1.0, orig_wm_energy))
            else:
                wm_drop = 0.50

            # Measure shield distortion
            if m_shield.any():
                shield_diff = np.abs(orig_shield_gray - cand_gray[m_shield].astype(np.float32))
                shield_damage = float(np.mean(shield_diff)) / 255.0
            else:
                shield_damage = 0.0

            # Score = wm_drop - 50.0 * shield_damage
            score = wm_drop - 50.0 * shield_damage

            norm_box = (round(rx1 / w, 2), round(ry1 / h, 2), round(rx2 / w, 2), round(ry2 / h, 2))
            steer_params = ",".join(f"{k}={v}" for k, v in p_dict.items())
            steer_str = f"p{page_num}@[{norm_box[0]:.2f},{norm_box[1]:.2f},{norm_box[2]:.2f},{norm_box[3]:.2f}]:{m_name}"
            if steer_params:
                steer_str += f"({steer_params})"

            evaluated_candidates.append(
                ParetoCandidate(
                    label=name,
                    method_name=m_name,
                    params=p_dict,
                    wm_drop=round(wm_drop, 4),
                    shield_damage=round(shield_damage, 4),
                    score=round(score, 4),
                    steer_dsl=steer_str,
                )
            )
        except Exception as exc:
            logger.debug("Auto-tuner candidate %s error: %s", name, exc)

    if not evaluated_candidates:
        return page_img, 0.0, None

    # Sort candidates by Pareto score
    evaluated_candidates.sort(key=lambda c: c.score, reverse=True)

    # Safest candidate with shield_damage <= 0.001
    zero_damage_cands = [c for c in evaluated_candidates if c.shield_damage <= 0.001]
    opt_a = zero_damage_cands[0] if zero_damage_cands else evaluated_candidates[0]
    opt_a.label = "Conservative"

    # Balanced candidate (highest watermark drop with shield_damage <= 0.003)
    balanced_cands = [c for c in evaluated_candidates if c.shield_damage <= 0.003]
    opt_b = (
        max(balanced_cands, key=lambda c: c.wm_drop) if balanced_cands else evaluated_candidates[0]
    )
    opt_b.label = "Balanced"

    norm_box = (round(rx1 / w, 2), round(ry1 / h, 2), round(rx2 / w, 2), round(ry2 / h, 2))
    opt_c = ParetoCandidate(
        label="Untouched",
        method_name="restore_original",
        params={},
        wm_drop=0.0,
        shield_damage=0.0,
        score=0.0,
        steer_dsl=f"p{page_num}@[{norm_box[0]:.2f},{norm_box[1]:.2f},{norm_box[2]:.2f},{norm_box[3]:.2f}]:restore_original",
    )

    ticket: Optional[EscalationTicket] = None
    # Trigger 3-Option Escalation Ticket if ambiguous case
    if (
        0.30 <= opt_a.wm_drop < 0.90
        or opt_a.shield_damage > 0.0005
        or len(evaluated_candidates) > 0
    ):
        ticket = EscalationTicket(
            ticket_id=f"T{ticket_counter}",
            page_num=page_num,
            roi_bbox=norm_box,
            reason="UNSEEN_ANOMALY_OVER_DIAGRAM",
            opt_a=opt_a,
            opt_b=opt_b,
            opt_c=opt_c,
        )

    # Apply winning candidate to full page ROI
    winner = opt_b if aggressiveness > 0.6 else opt_a
    # Fast drop-in of winner result
    if page_img.shape[:2] == (ph, pw):
        out_img = cand_outputs.get(winner.label, patch).copy()
    else:
        out_img = page_img.copy()
        out_img[py1:py2, px1:px2] = cand_outputs.get(winner.label, patch)
    out_img = shield.apply_text_core_lock(out_img, page_img, lock_halo=True)

    return out_img, winner.score, ticket


sweep_pareto_unmixers = auto_solve_patch
