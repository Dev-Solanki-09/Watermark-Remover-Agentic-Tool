"""Content-stream tokenizer + PDF Graphics State machine.

This is the foundation for all vector removers. It wraps pikepdf's
`parse_content_stream` with a stateful graphics-state tracker that
maintains the CTM, text matrix, font, text rendering mode, clipping
state, and ExtGState stack as operators are processed.

Key invariant: state-mutating operators (cm, Tm, Td, gs, q, Q) are
NEVER deleted during watermark removal — only painting operators
(S, f, B, Tj, Do, sh, etc.) are replaced with no-ops or excised.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

import pikepdf


@dataclass
class GraphicsState:
    """Snapshot of the PDF Graphics State at a given point in the content stream."""

    # Current Transformation Matrix (6-element affine: a, b, c, d, e, f)
    ctm: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    # Text state
    tm: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    font: Optional[str] = None
    font_size: float = 0.0
    tr: int = 0  # text rendering mode
    tl: float = 0.0  # text leading
    tc: float = 0.0  # character spacing
    tw: float = 0.0  # word spacing

    # Colour state
    fill_color: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    stroke_color: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    fill_cs: str = "/DeviceGray"
    stroke_cs: str = "/DeviceGray"

    # Alpha / blend
    ca: float = 1.0  # non-stroking alpha
    CA: float = 1.0  # stroking alpha
    bm: str = "/Normal"  # blend mode
    smask: bool = False

    # Clipping
    clip_pending: bool = False  # W or W* was issued, awaiting paint op

    def copy(self) -> "GraphicsState":
        """Deep copy of this state."""
        return copy.deepcopy(self)


def _multiply_matrices(a: list[float], b: list[float]) -> list[float]:
    """Multiply two 3×3 affine matrices represented as [a,b,c,d,e,f]."""
    return [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]


def _rotation_degrees(ctm: list[float]) -> float:
    """Extract rotation angle from CTM (in degrees)."""
    return math.degrees(math.atan2(ctm[1], ctm[0]))


class StreamProcessor:
    """Processes a content stream with full graphics state tracking.

    Usage:
        processor = StreamProcessor(page, resources)
        for entry in processor.iterate():
            # entry.operator, entry.operands, entry.state
            # Decide whether to keep or remove this operator
        filtered = processor.rebuild(keep_indices)
    """

    @dataclass
    class Entry:
        """A single operator in the content stream with its tracked state."""

        index: int
        operator: str
        operands: list
        state: GraphicsState
        raw: tuple  # (operands, operator) from pikepdf

    def __init__(self, page: pikepdf.Page, resources: Optional[pikepdf.Dictionary] = None):
        self._page = page
        self._resources = resources or page.get("/Resources", pikepdf.Dictionary())
        self._entries: list[StreamProcessor.Entry] = []
        self._parsed = False

    def iterate(self) -> list["StreamProcessor.Entry"]:
        """Parse and iterate the content stream with state tracking.

        Returns list of Entry objects with fully tracked graphics state.
        """
        if self._parsed:
            return self._entries

        try:
            cs = pikepdf.parse_content_stream(self._page)
        except Exception:
            self._parsed = True
            return self._entries

        state = GraphicsState()
        state_stack: list[GraphicsState] = []
        ext_gstate_dict = (
            self._resources.get("/ExtGState", pikepdf.Dictionary())
            if self._resources is not None
            else pikepdf.Dictionary()
        )

        idx = 0
        for operands, operator in cs:
            op = str(operator)

            # --- State-mutating operators ---
            if op == "q":
                state_stack.append(state.copy())

            elif op == "Q":
                if state_stack:
                    state = state_stack.pop()

            elif op == "cm" and len(operands) == 6:
                m = [float(o) for o in operands]
                state.ctm = _multiply_matrices(m, state.ctm)

            elif op == "Tm" and len(operands) == 6:
                state.tm = [float(o) for o in operands]

            elif op == "Td" and len(operands) == 2:
                tx, ty = float(operands[0]), float(operands[1])
                state.tm[4] += tx
                state.tm[5] += ty

            elif op == "TD" and len(operands) == 2:
                tx, ty = float(operands[0]), float(operands[1])
                state.tl = -ty
                state.tm[4] += tx
                state.tm[5] += ty

            elif op == "T*":
                state.tm[5] -= state.tl

            elif op == "Tf" and len(operands) >= 2:
                state.font = str(operands[0])
                state.font_size = float(operands[1])

            elif op == "Tr" and len(operands) >= 1:
                state.tr = int(float(operands[0]))

            elif op == "TL" and len(operands) >= 1:
                state.tl = float(operands[0])

            elif op == "Tc" and len(operands) >= 1:
                state.tc = float(operands[0])

            elif op == "Tw" and len(operands) >= 1:
                state.tw = float(operands[0])

            elif op == "gs" and len(operands) >= 1:
                gs_name = str(operands[0])
                gs_dict = ext_gstate_dict.get(gs_name)
                if gs_dict is not None:
                    try:
                        gs = gs_dict.resolve() if hasattr(gs_dict, "resolve") else gs_dict
                        if "/ca" in gs:
                            state.ca = float(gs["/ca"])
                        if "/CA" in gs:
                            state.CA = float(gs["/CA"])
                        if "/BM" in gs:
                            state.bm = str(gs["/BM"])
                        state.smask = "/SMask" in gs and str(gs.get("/SMask", "")) != "/None"
                    except Exception:
                        pass

            elif op in ("g", "G") and len(operands) >= 1:
                val = [float(operands[0])]
                if op == "g":
                    state.fill_color = val
                    state.fill_cs = "/DeviceGray"
                else:
                    state.stroke_color = val
                    state.stroke_cs = "/DeviceGray"

            elif op in ("rg", "RG") and len(operands) >= 3:
                val = [float(o) for o in operands[:3]]
                if op == "rg":
                    state.fill_color = val
                    state.fill_cs = "/DeviceRGB"
                else:
                    state.stroke_color = val
                    state.stroke_cs = "/DeviceRGB"

            elif op in ("k", "K") and len(operands) >= 4:
                val = [float(o) for o in operands[:4]]
                if op == "k":
                    state.fill_color = val
                    state.fill_cs = "/DeviceCMYK"
                else:
                    state.stroke_color = val
                    state.stroke_cs = "/DeviceCMYK"

            elif op in ("W", "W*"):
                state.clip_pending = True

            # Record entry with current state snapshot
            entry = StreamProcessor.Entry(
                index=idx,
                operator=op,
                operands=list(operands),
                state=state.copy(),
                raw=(operands, operator),
            )
            self._entries.append(entry)

            # Clear clip pending after paint op
            if op in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n", "Do", "sh"):
                state.clip_pending = False

            idx += 1

        self._parsed = True
        return self._entries

    def rebuild(self, keep_indices: set[int]) -> list[tuple]:
        """Rebuild content stream keeping only operators at specified indices.

        Returns list of (operands, operator) tuples suitable for
        pikepdf.unparse_content_stream().
        """
        return [
            entry.raw for entry in self._entries if entry.index in keep_indices
        ]

    def rebuild_filtered(self, remove_indices: set[int]) -> list[tuple]:
        """Rebuild content stream removing operators at specified indices."""
        return [
            entry.raw for entry in self._entries if entry.index not in remove_indices
        ]

    def rebuild_with_replacements(
        self,
        remove_indices: set[int],
        replace_map: Optional[dict[int, tuple]] = None,
    ) -> list[tuple]:
        """Rebuild content stream applying removals and operator replacements.

        replace_map maps entry index to a new (operands, operator) tuple, e.g.
        to replace painting operators ('S') with no-ops ('n').
        """
        replace_map = replace_map or {}
        result = []
        for entry in self._entries:
            if entry.index in remove_indices:
                continue
            if entry.index in replace_map:
                result.append(replace_map[entry.index])
            else:
                result.append(entry.raw)
        return result


# Aliases for API compatibility
ContentStreamAST = StreamProcessor
OperationNode = StreamProcessor.Entry

