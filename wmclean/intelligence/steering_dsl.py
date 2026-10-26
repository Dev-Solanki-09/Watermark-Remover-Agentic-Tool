"""Declarative Steering Micro-DSL Parser (Agentic Authority Layer v7.0).

Allows calling AI/LLM orchestrators to steer watermark removal on any specific
page, named zone, layer, or parameter in < 20 tokens without writing Python code.

Grammar:
    <rule> ::= <page_sel> ["@" <zone_sel>] ":" <method_chain> ["(" <params> ")"]
    <rules> ::= <rule> [";" <rule>]*

Examples:
    "p1-10@top-left:bilingual_clone+density_sub"
    "p14-23@bottom-right:margin_banner"
    "p24-33:hue_unmix(hue=red,protect=green,blue)"
    "all:strip_text(match='@username|watermark|scanner')"
    "p41@ROI_1:maxtree(h=65); p10@top-left:restore_original"
    "p10@[0.05,0.12,0.45,0.18]:protect_zone"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from wmclean.schemas import RemoverMethod

logger = logging.getLogger(__name__)

# Standard named spatial zones mapped to normalized [x1, y1, x2, y2] (0.0 .. 1.0)
NAMED_ZONES: dict[str, tuple[float, float, float, float]] = {
    "top-left": (0.0, 0.0, 0.50, 0.25),
    "top-right": (0.50, 0.0, 1.0, 0.25),
    "bottom-left": (0.0, 0.88, 0.50, 1.0),
    "bottom-right": (0.50, 0.88, 1.0, 1.0),
    "header": (0.0, 0.0, 1.0, 0.10),
    "footer": (0.0, 0.90, 1.0, 1.0),
    "center": (0.12, 0.18, 0.88, 0.82),
    "col-left": (0.0, 0.08, 0.50, 0.92),
    "col-right": (0.50, 0.08, 1.0, 0.92),
    "margins": (0.0, 0.0, 1.0, 0.12),
    "margin-top": (0.0, 0.0, 1.0, 0.12),
    "margin-bottom": (0.0, 0.88, 1.0, 1.0),
    "margin-left": (0.0, 0.0, 0.12, 1.0),
    "margin-right": (0.88, 0.0, 1.0, 1.0),
}

# Alias mapping for friendly short names in DSL
METHOD_ALIASES: dict[str, str] = {
    "bilingual_clone": RemoverMethod.BILINGUAL_TWIN_CLONE.value,
    "twin_clone": RemoverMethod.BILINGUAL_TWIN_CLONE.value,
    "density_sub": RemoverMethod.DENSITY_SUB.value,
    "optical_density_sub": RemoverMethod.DENSITY_SUB.value,
    "hue_unmix": RemoverMethod.POINTWISE_HUE_AND_LUT.value,
    "pointwise_unmix": RemoverMethod.POINTWISE_HUE_AND_LUT.value,
    "lut": RemoverMethod.POINTWISE_HUE_AND_LUT.value,
    "maxtree": RemoverMethod.MAXTREE_FILTER.value,
    "maxtree_filter": RemoverMethod.MAXTREE_FILTER.value,
    "skan_graph": RemoverMethod.SKELETON_GRAPH.value,
    "skeleton_graph": RemoverMethod.SKELETON_GRAPH.value,
    "margin_banner": RemoverMethod.RASTER_MARGIN_BANNER.value,
    "banner": RemoverMethod.RASTER_MARGIN_BANNER.value,
    "piracy_ocr": RemoverMethod.PIRACY_OCR_SPOTTER.value,
    "ocr_spotter": RemoverMethod.PIRACY_OCR_SPOTTER.value,
    "eigen_bss": RemoverMethod.BEER_LAMBERT_STAIN_UNMIX.value,
    "stain_deconv": RemoverMethod.BEER_LAMBERT_STAIN_UNMIX.value,
    "mupdf_redact": RemoverMethod.MUPDF_LAYER_REDACT.value,
    "font_zero": RemoverMethod.FONT_TABLE_ZERO.value,
    "text_op_strip": RemoverMethod.TEXT_OP_STRIP.value,
    "strip_text": RemoverMethod.TEXT_OP_STRIP.value,
    "text_strip": RemoverMethod.TEXT_OP_STRIP.value,
    "ast_tj": RemoverMethod.AST_TJ_REASSEMBLE.value,
    "annot_del": RemoverMethod.ANNOT_DEL.value,
    "ocg_strip": RemoverMethod.OCG_STRIP.value,
    "artifact_strip": RemoverMethod.ARTIFACT_STRIP.value,
    "strip_artifact": RemoverMethod.ARTIFACT_STRIP.value,
    "artifact": RemoverMethod.ARTIFACT_STRIP.value,
    "lama": RemoverMethod.SURGICAL_LAMA.value,
}



@dataclass
class SteerZone:
    """Target spatial zone."""

    name: str
    norm_bbox: tuple[float, float, float, float]  # [x1, y1, x2, y2] normalized 0.0..1.0

    def to_pixel_bbox(self, width_px: int, height_px: int) -> tuple[int, int, int, int]:
        """Convert normalized bbox to integer pixel coordinates."""
        x1, y1, x2, y2 = self.norm_bbox
        return (
            max(0, round(x1 * width_px)),
            max(0, round(y1 * height_px)),
            min(width_px, round(x2 * width_px)),
            min(height_px, round(y2 * height_px)),
        )


@dataclass
class SteerAction:
    """Action or remover specification with parameters."""

    name: str  # remover name or meta-action ('skip', 'restore_original', 'protect_zone', 'shield', 'auto_solve')
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SteerRule:
    """A single parsed steering rule."""

    page_selector: str
    zones: list[SteerZone] = field(default_factory=list)
    actions: list[SteerAction] = field(default_factory=list)

    def matches_page(self, page_num: int, total_pages: int = 9999) -> bool:
        """Check if this rule applies to 1-indexed page_num."""
        sel = self.page_selector.strip().lower()
        if sel == "all":
            return True
        if sel == "odd":
            return (page_num % 2) != 0
        if sel == "even":
            return (page_num % 2) == 0

        tokens = sel.split(",")
        for tok in tokens:
            t = tok.strip()
            if t.startswith("p"):
                t = t[1:]
            if "-" in t:
                parts = t.split("-", 1)
                try:
                    p_start = int(parts[0])
                    p_end = int(parts[1])
                    if p_start <= page_num <= p_end:
                        return True
                except ValueError:
                    continue
            else:
                try:
                    if int(t) == page_num:
                        return True
                except ValueError:
                    continue
        return False


def _parse_params_str(param_str: str) -> dict[str, Any]:
    """Parse comma-separated key=val parameters."""
    params: dict[str, Any] = {}
    if not param_str:
        return params

    tokens = re.split(r",(?![^\[]*\])(?![^\'\"]*[\'\"])", param_str)
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            try:
                if "." in v:
                    params[k] = float(v)
                else:
                    params[k] = int(v)
            except ValueError:
                if "," in v:
                    params[k] = [x.strip() for x in v.split(",")]
                else:
                    params[k] = v
        else:
            params[tok] = True
    return params


def _parse_zone(
    zone_str: str, known_rois: Optional[dict[str, tuple[float, float, float, float]]] = None
) -> list[SteerZone]:
    """Parse zone token (named zone, ROI_x, or explicit [x1,y1,x2,y2])."""
    zone_str = zone_str.strip()
    if (zone_str.startswith("[") and zone_str.endswith("]")) or (
        zone_str.startswith("(") and zone_str.endswith(")")
    ):
        inner = zone_str[1:-1].strip()
        try:
            coords = [float(c.strip()) for c in inner.split(",")]
            if len(coords) == 4:
                return [
                    SteerZone(name="custom", norm_bbox=(coords[0], coords[1], coords[2], coords[3]))
                ]
        except ValueError:
            pass

    zones: list[SteerZone] = []
    sub_zones = [z.strip() for z in zone_str.split(",") if z.strip()]
    for z in sub_zones:
        z_lower = z.lower()
        if z_lower in NAMED_ZONES:
            zones.append(SteerZone(name=z_lower, norm_bbox=NAMED_ZONES[z_lower]))
        elif known_rois and z in known_rois:
            zones.append(SteerZone(name=z, norm_bbox=known_rois[z]))
        else:
            zones.append(SteerZone(name=z, norm_bbox=(0.0, 0.0, 1.0, 1.0)))

    return zones if zones else [SteerZone(name="page", norm_bbox=(0.0, 0.0, 1.0, 1.0))]


def parse_steer_dsl(
    dsl_text: str,
    known_rois: Optional[dict[str, tuple[float, float, float, float]]] = None,
) -> list[SteerRule]:
    """Parse a semicolon-separated steering DSL string into a list of SteerRules.

    Grammar:
        <page_sel> [@<zone_sel>] : <method_chain> [(<params>)]
    """
    rules: list[SteerRule] = []
    if not dsl_text or not dsl_text.strip():
        return rules

    raw_rules = [r.strip() for r in dsl_text.split(";") if r.strip()]

    valid_actions = {
        "skip",
        "restore_original",
        "protect_zone",
        "shield",
        "auto_solve",
        *(m.value for m in RemoverMethod),
        *METHOD_ALIASES.keys(),
    }

    for raw in raw_rules:
        if ":" not in raw:
            raise ValueError(f"Invalid steering DSL syntax (missing ':' in rule): '{raw}'")
        left, right = raw.split(":", 1)
        left = left.strip()
        right = right.strip()

        page_sel = left
        zones: list[SteerZone] = []
        if "@" in left:
            p_part, z_part = left.split("@", 1)
            page_sel = p_part.strip()
            zones = _parse_zone(z_part, known_rois=known_rois)
        else:
            zones = [SteerZone(name="full", norm_bbox=(0.0, 0.0, 1.0, 1.0))]

        actions: list[SteerAction] = []
        method_tokens = right.split("+")
        for m_tok in method_tokens:
            m_tok = m_tok.strip()
            if not m_tok:
                continue

            params: dict[str, Any] = {}
            action_name = m_tok
            if "(" in m_tok and m_tok.endswith(")"):
                idx_paren = m_tok.index("(")
                action_name = m_tok[:idx_paren].strip()
                p_content = m_tok[idx_paren + 1 : -1].strip()
                params = _parse_params_str(p_content)

            canonical_name = METHOD_ALIASES.get(action_name.lower(), action_name)
            if (
                action_name.lower() not in valid_actions
                and canonical_name not in valid_actions
            ):
                raise ValueError(
                    f"Unknown steering action '{action_name}'. Must be one of: {sorted(valid_actions)}"
                )
            actions.append(SteerAction(name=canonical_name, params=params))

        if actions:
            rules.append(SteerRule(page_selector=page_sel, zones=zones, actions=actions))

    return rules


parse_steering_dsl = parse_steer_dsl

