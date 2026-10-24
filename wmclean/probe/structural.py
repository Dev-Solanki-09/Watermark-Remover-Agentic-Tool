"""L0 Structural Probe — extract page geometry, annotations, OCGs, artifacts,
ExtGState, text spans, XObject candidates, vector path hashes, and revision history.

This module NEVER modifies the input PDF.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Optional

import pikepdf

from wmclean.schemas import (
    AnnotCandidate,
    ArtifactCandidate,
    ExtGStateInfo,
    OCGCandidate,
    PageGeometry,
    PageKind,
    PathCandidate,
    ProbeRecord,
    RasterStats,
    TextCandidate,
    XObjCandidate,
)

# ---------------------------------------------------------------------------
# Dynamic text normalization patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL_RE = re.compile(r"https?://[^\s)\"'>]+")
_DATE_RE = re.compile(
    r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"
    r"|\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_DIGIT_RUN_RE = re.compile(r"\b\d{4,}\b")  # long digit runs (IDs, serials, IPs)

# Watermark keyword patterns for text classification
_WM_KEYWORDS = re.compile(
    r"(?i)\b(?:CONFIDENTIAL|DRAFT|SAMPLE|DO\s+NOT\s+COPY|COPY|SPECIMEN|PREVIEW|"
    r"WATERMARK|DOWNLOADED\s+BY|PRIVILEGED|INTERNAL\s+USE|NOT\s+FOR\s+DISTRIBUTION)\b"
)


def normalize_text(raw: str) -> str:
    """Normalize dynamic per-page text to a canonical template.

    Replaces emails with <EMAIL>, URLs with <URL>, dates with <DATE>,
    and long digit runs with <D>.
    """
    s = _EMAIL_RE.sub("<EMAIL>", raw)
    s = _URL_RE.sub("<URL>", s)
    s = _DATE_RE.sub("<DATE>", s)
    s = _DIGIT_RUN_RE.sub("<D>", s)
    return s.strip()


def _text_sig(norm_str: str, font: str, size: float, rot_deg: float) -> str:
    """Compute a short hash signature for a text watermark candidate."""
    key = f"{norm_str}|{font}|{size:.1f}|{rot_deg:.0f}"
    return "t_" + hashlib.sha256(key.encode()).hexdigest()[:6]


# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------


def _get_page_geometry(page: pikepdf.Page) -> PageGeometry:
    """Extract MediaBox, CropBox, and /Rotate from a page."""
    mb = [float(x) for x in page.mediabox]
    cb = mb[:]
    if "/CropBox" in page:
        cb = [float(x) for x in page["/CropBox"]]
    rot = int(page.get("/Rotate", 0)) % 360
    return PageGeometry(mb=mb, cb=cb, rot=rot)


# ---------------------------------------------------------------------------
# Annotation extraction
# ---------------------------------------------------------------------------


_WM_ANNOT_SUBTYPES = {"/Watermark", "/Stamp", "/FreeText", "/Widget"}


def _extract_annots(page: pikepdf.Page) -> list[AnnotCandidate]:
    """Extract watermark-type annotation candidates."""
    results = []
    annots = page.get("/Annots")
    if annots is None:
        return results

    for annot_ref in annots:
        try:
            annot = annot_ref.resolve() if hasattr(annot_ref, "resolve") else annot_ref
        except Exception:
            continue

        sub = str(annot.get("/Subtype", ""))
        if sub not in _WM_ANNOT_SUBTYPES:
            continue

        rect = annot.get("/Rect")
        if rect is None:
            continue

        bb = [float(x) for x in rect]
        xref = annot_ref.objgen[0] if hasattr(annot_ref, "objgen") else 0
        results.append(AnnotCandidate(xref=xref, sub=sub, bb=bb))

    return results


# ---------------------------------------------------------------------------
# OCG extraction
# ---------------------------------------------------------------------------


def _extract_ocgs(pdf: pikepdf.Pdf) -> list[OCGCandidate]:
    """Extract Optional Content Groups from document root."""
    results = []
    root = pdf.Root
    oc_props = root.get("/OCProperties")
    if oc_props is None:
        return results

    ocgs = oc_props.get("/OCGs")
    if ocgs is None:
        return results

    for ocg_ref in ocgs:
        try:
            ocg = ocg_ref.resolve() if hasattr(ocg_ref, "resolve") else ocg_ref
        except Exception:
            continue

        name = str(ocg.get("/Name", ""))
        intent = str(ocg.get("/Intent", "/View"))
        xref = ocg_ref.objgen[0] if hasattr(ocg_ref, "objgen") else 0
        results.append(OCGCandidate(xref=xref, name=name, intent=intent))

    return results


# ---------------------------------------------------------------------------
# Content stream artifact extraction
# ---------------------------------------------------------------------------


def _extract_artifacts(page: pikepdf.Page) -> list[ArtifactCandidate]:
    """Parse content stream to find /Artifact (/Subtype /Watermark) BDC..EMC spans."""
    results = []
    try:
        cs = pikepdf.parse_content_stream(page)
    except Exception:
        return results

    op_idx = 0
    bdc_stack: list[dict] = []

    for operands, operator in cs:
        op_str = str(operator)

        if op_str == "BDC" and len(operands) >= 2:
            tag = str(operands[0])
            props = operands[1] if len(operands) > 1 else None
            bdc_stack.append({"tag": tag, "props": props, "start": op_idx})

        elif op_str == "EMC" and bdc_stack:
            frame = bdc_stack.pop()
            if frame["tag"] == "/Artifact":
                # Check for /Subtype /Watermark in the properties
                props = frame.get("props")
                sub = ""
                if props is not None and hasattr(props, "get"):
                    sub = str(props.get("/Subtype", ""))
                elif isinstance(props, pikepdf.Dictionary):
                    sub = str(props.get("/Subtype", ""))

                if sub == "/Watermark":
                    results.append(
                        ArtifactCandidate(
                            sub="/Watermark",
                            op_span=[frame["start"], op_idx],
                        )
                    )

        op_idx += 1

    return results


# ---------------------------------------------------------------------------
# ExtGState transparency extraction
# ---------------------------------------------------------------------------


def _extract_gs_alpha(page: pikepdf.Page) -> list[ExtGStateInfo]:
    """Extract ExtGState entries with transparency or blend mode information."""
    results = []
    resources = page.get("/Resources")
    if resources is None:
        return results

    ext_gstate = resources.get("/ExtGState")
    if ext_gstate is None:
        return results

    for name, gs_ref in ext_gstate.items():
        try:
            gs = gs_ref.resolve() if hasattr(gs_ref, "resolve") else gs_ref
        except Exception:
            continue

        ca = float(gs["/ca"]) if "/ca" in gs else None
        CA = float(gs["/CA"]) if "/CA" in gs else None
        bm = str(gs.get("/BM", "")) or None
        smask = "/SMask" in gs and str(gs["/SMask"]) != "/None"

        # Only include if there's transparency or non-normal blend mode info
        if ca is not None or CA is not None or (bm and bm != "/Normal") or smask:
            results.append(ExtGStateInfo(name=str(name), ca=ca, CA=CA, bm=bm, smask=smask))

    return results


# ---------------------------------------------------------------------------
# Text watermark candidate extraction
# ---------------------------------------------------------------------------


def _extract_text_candidates(
    page: pikepdf.Page, page_num: int, total_pages: int, fitz_page=None
) -> list[TextCandidate]:
    """Extract text spans that match watermark patterns.

    Uses PyMuPDF low-level character/stroke trace inspection (page.get_texttrace())
    for 1ms exact vector/text watermark probing (adapted from jonatw/pdf-watermark-remove).
    Falls back to text dictionary if get_texttrace is unavailable.
    """
    results: list[TextCandidate] = []
    if fitz_page is not None:
        # Fast-path: PyMuPDF >= 1.28.2 page.get_texttrace() (1ms/page)
        if hasattr(fitz_page, "get_texttrace"):
            try:
                traces = fitz_page.get_texttrace()
                for trace in traces:
                    t_type = trace.get("type", 0)
                    # Protect invisible OCR layer (render mode 3) - never candidate for removal
                    if t_type == 3:
                        continue

                    dx, dy = trace.get("dir", (1.0, 0.0))
                    rot_rad = math.atan2(dy, dx)
                    rot_deg = round(math.degrees(rot_rad), 1)
                    is_diagonal = abs(rot_deg) > 5.0

                    opacity = float(trace.get("opacity", 1.0))
                    is_translucent = opacity < 0.85
                    is_stroke_outline = t_type == 1

                    font_size = float(trace.get("size", 0.0))
                    font = str(trace.get("font", ""))

                    color_raw = trace.get("color", (0.0, 0.0, 0.0))
                    if isinstance(color_raw, (tuple, list)) and len(color_raw) >= 3:
                        r, g, b = float(color_raw[0]), float(color_raw[1]), float(color_raw[2])
                    elif isinstance(color_raw, int):
                        r = ((color_raw >> 16) & 0xFF) / 255.0
                        g = ((color_raw >> 8) & 0xFF) / 255.0
                        b = (color_raw & 0xFF) / 255.0
                    else:
                        r, g, b = 0.0, 0.0, 0.0

                    is_light = r > 0.4 or g > 0.4 or b > 0.4

                    chars = trace.get("chars", [])
                    raw_text = "".join(chr(c[0]) for c in chars if c[0] > 0).strip()
                    norm = normalize_text(raw_text) if raw_text else "<EMPTY>"
                    has_kw = bool(_WM_KEYWORDS.search(raw_text)) if raw_text else False

                    # Watermark candidate grouping:
                    # - Diagonal orientation
                    # - Translucent opacity (ca < 0.85)
                    # - Stroke-only outline (type == 1)
                    # - Watermark keyword match with large/light/rotated text
                    is_wm = (
                        (
                            is_diagonal
                            and (has_kw or font_size >= 14.0 or is_translucent or is_stroke_outline)
                        )
                        or (is_translucent and (has_kw or font_size >= 14.0 or is_stroke_outline))
                        or (is_stroke_outline and (has_kw or font_size >= 14.0 or is_diagonal))
                        or (has_kw and (font_size >= 20.0 or is_light or is_diagonal))
                    )

                    if is_wm:
                        bb = [float(x) for x in trace.get("bbox", [0, 0, 0, 0])]
                        sig = _text_sig(norm, font, font_size, rot_deg)
                        results.append(
                            TextCandidate(
                                sig=sig,
                                norm_str=norm,
                                rot_deg=rot_deg,
                                bb=bb,
                                font=font,
                                size=font_size,
                                color=[round(r, 2), round(g, 2), round(b, 2)],
                                tr=t_type,
                                ca=opacity,
                                doc_freq=1.0,
                            )
                        )
                return results
            except Exception:
                pass

        # Fallback: get_text("dict")
        try:
            text_dict = fitz_page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # text blocks only
                    continue
                for line in block.get("lines", []):
                    dx, dy = line.get("dir", (1.0, 0.0))
                    rot_rad = math.atan2(dy, dx)
                    rot_deg = round(math.degrees(rot_rad), 1)

                    for span in line.get("spans", []):
                        raw_text = span.get("text", "").strip()
                        if not raw_text:
                            continue

                        norm = normalize_text(raw_text)
                        has_kw = bool(_WM_KEYWORDS.search(raw_text))
                        is_rotated = abs(rot_deg) > 5.0
                        font_size = float(span.get("size", 0.0))
                        int_color = span.get("color", 0)
                        r = ((int_color >> 16) & 0xFF) / 255.0
                        g = ((int_color >> 8) & 0xFF) / 255.0
                        b = (int_color & 0xFF) / 255.0
                        is_light = r > 0.4 or g > 0.4 or b > 0.4

                        # Watermark candidate if: rotated text (watermark/title angle) OR keyword text that is large/light/rotated
                        if (is_rotated and (has_kw or font_size >= 14.0)) or (
                            has_kw and (font_size >= 20.0 or is_light or is_rotated)
                        ):
                            font = span.get("font", "")
                            bb = [float(x) for x in span.get("bbox", [0, 0, 0, 0])]
                            sig = _text_sig(norm, font, font_size, rot_deg)

                            results.append(
                                TextCandidate(
                                    sig=sig,
                                    norm_str=norm,
                                    rot_deg=rot_deg,
                                    bb=bb,
                                    font=font,
                                    size=font_size,
                                    color=[round(r, 2), round(g, 2), round(b, 2)],
                                    tr=0,
                                    ca=None,
                                    doc_freq=1.0,
                                )
                            )
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# XObject candidate extraction
# ---------------------------------------------------------------------------


def _extract_xobj_candidates(
    page: pikepdf.Page, kind: Optional[PageKind] = None, fitz_page=None
) -> list[XObjCandidate]:
    """Extract image and form XObject candidates from page resources.

    Container Form XObjects (which encapsulate fonts, text streams, or child objects)
    are traversed so that only actual leaf images/stamps are identified as candidates.
    """
    results = []
    resources = page.get("/Resources")
    if resources is None:
        return results

    def _collect_leaf_xobjects(res, depth=0):
        if depth > 4 or res is None:
            return []
        xo_dict = res.get("/XObject")
        if not xo_dict:
            return []
        leaves = []
        for name, xobj_ref in xo_dict.items():
            try:
                xobj = xobj_ref.resolve() if hasattr(xobj_ref, "resolve") else xobj_ref
            except Exception:
                continue
            subtype = str(xobj.get("/Subtype", ""))
            xref = xobj_ref.objgen[0] if hasattr(xobj_ref, "objgen") else 0

            if subtype == "/Image":
                leaves.append((str(name), xref, "IMAGE", xobj))
            elif subtype == "/Form":
                f_res = xobj.get("/Resources")
                is_container = False
                if f_res is not None and ("/Font" in f_res or "/XObject" in f_res):
                    is_container = True
                if is_container:
                    leaves.extend(_collect_leaf_xobjects(f_res, depth + 1))
                else:
                    leaves.append((str(name), xref, "FORM", xobj))
        return leaves

    leaf_xobjs = _collect_leaf_xobjects(resources)

    # Scanned pages have a single full-page background image which is the document itself
    if kind in (PageKind.SCAN, PageKind.SCAN_OCR) and len(leaf_xobjs) == 1:
        return results

    for name, xref, xobj_kind, xobj in leaf_xobjs:
        bb = [0.0, 0.0, 0.0, 0.0]
        if xobj_kind == "FORM" and "/BBox" in xobj:
            bb = [float(x) for x in xobj["/BBox"]]
        elif xobj_kind == "IMAGE":
            w = float(xobj.get("/Width", 0))
            h = float(xobj.get("/Height", 0))
            bb = [0.0, 0.0, w, h]

        results.append(
            XObjCandidate(
                name=name,
                xref=xref,
                kind=xobj_kind,
                bb=bb,
                doc_freq=0.0,  # computed after all pages probed
            )
        )

    return results


# ---------------------------------------------------------------------------
# Vector path candidate extraction
# ---------------------------------------------------------------------------


def _extract_path_candidates(page: pikepdf.Page) -> list[PathCandidate]:
    """Extract untagged vector path candidates from content stream.

    Computes topology hashes for path sequences and identifies those
    with watermark-like characteristics (alpha, repetition, diagonal).
    """
    results = []
    try:
        cs = pikepdf.parse_content_stream(page)
    except Exception:
        return results

    # Collect path sequences between save/restore blocks
    path_ops = []
    path_start = None
    op_count = 0
    current_alpha = None
    current_stroke_color = None
    current_fill_color = None

    for operands, operator in cs:
        op_str = str(operator)

        if op_str == "gs" and operands:
            try:
                gs_name = str(operands[0])
                res = page.get("/Resources")
                if res and "/ExtGState" in res:
                    egs = res["/ExtGState"].get(gs_name)
                    if egs:
                        resolved = egs.resolve() if hasattr(egs, "resolve") else egs
                        if "/ca" in resolved:
                            current_alpha = float(resolved["/ca"])
                        elif "/CA" in resolved:
                            current_alpha = float(resolved["/CA"])
            except Exception:
                pass

        elif op_str == "RG" and len(operands) >= 3:
            try:
                current_stroke_color = tuple(float(x) for x in operands[:3])
            except Exception:
                pass
        elif op_str == "rg" and len(operands) >= 3:
            try:
                current_fill_color = tuple(float(x) for x in operands[:3])
            except Exception:
                pass
        elif op_str == "G" and len(operands) >= 1:
            try:
                v = float(operands[0])
                current_stroke_color = (v, v, v)
            except Exception:
                pass
        elif op_str == "g" and len(operands) >= 1:
            try:
                v = float(operands[0])
                current_fill_color = (v, v, v)
            except Exception:
                pass
        elif op_str == "K" and len(operands) >= 4:
            try:
                k = float(operands[3])
                v = max(0.0, 1.0 - k)
                current_stroke_color = (v, v, v)
            except Exception:
                pass
        elif op_str == "k" and len(operands) >= 4:
            try:
                k = float(operands[3])
                v = max(0.0, 1.0 - k)
                current_fill_color = (v, v, v)
            except Exception:
                pass

        elif op_str in ("m", "l", "c", "v", "y", "re"):
            if path_start is None:
                path_start = op_count
            path_ops.append((op_str, [str(o) for o in operands]))

        elif op_str in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*"):
            if path_ops and len(path_ops) >= 2:
                # Compute topology hash
                topo_key = "|".join(f"{op}:{','.join(args)}" for op, args in path_ops)
                h = "p_" + hashlib.sha256(topo_key.encode()).hexdigest()[:6]

                is_stroke = op_str in ("S", "s", "B", "B*", "b", "b*")
                is_fill = op_str in ("f", "F", "f*", "B", "B*", "b", "b*")
                light_stroke = (
                    is_stroke
                    and current_stroke_color
                    and all(c >= 0.65 for c in current_stroke_color)
                )
                light_fill = (
                    is_fill and current_fill_color and all(c >= 0.65 for c in current_fill_color)
                )
                is_light = bool(light_stroke or light_fill)

                results.append(
                    PathCandidate(
                        hash=h,
                        ops=len(path_ops),
                        bb=[0.0, 0.0, 0.0, 0.0],  # refined later with CTM
                        ca=current_alpha,
                        is_clip=False,
                        doc_freq=0.0,  # computed after all pages probed
                        is_light=is_light,
                    )
                )
            path_ops = []
            path_start = None

        elif op_str == "n":
            # No-op paint — might be clipping setup
            if path_ops:
                path_ops = []
                path_start = None

        op_count += 1

    return results


# ---------------------------------------------------------------------------
# Page kind classification
# ---------------------------------------------------------------------------


def _classify_page_kind(page: pikepdf.Page, fitz_page=None) -> PageKind:
    """Classify page as VECTOR, SCAN, SCAN_OCR, HYBRID, or BITONAL."""
    if fitz_page is not None:
        try:
            text = fitz_page.get_text().strip()
            images = fitz_page.get_images()
            has_vector_text = len(text) > 0
            has_images = len(images) > 0
            if has_images:
                for img in images:
                    if len(img) > 4 and img[4] == 1:
                        return PageKind.BITONAL
            if has_images and has_vector_text:
                return PageKind.HYBRID
            elif has_images:
                return PageKind.SCAN
            elif has_vector_text:
                return PageKind.VECTOR
            else:
                return PageKind.VECTOR
        except Exception:
            pass

    resources = page.get("/Resources")
    has_images = False
    has_vector_text = False

    def _inspect_xobjects(res, depth=0):
        nonlocal has_images, has_vector_text
        if depth > 4 or res is None:
            return
        xo_dict = res.get("/XObject")
        if xo_dict:
            for _, xobj_ref in xo_dict.items():
                try:
                    xo = xobj_ref.resolve() if hasattr(xobj_ref, "resolve") else xobj_ref
                    st = str(xo.get("/Subtype", ""))
                    if st == "/Image":
                        has_images = True
                    elif st == "/Form":
                        f_res = xo.get("/Resources")
                        if f_res:
                            if "/Font" in f_res:
                                has_vector_text = True
                            _inspect_xobjects(f_res, depth + 1)
                except Exception:
                    continue
        if "/Font" in res:
            has_vector_text = True

    if resources:
        _inspect_xobjects(resources)

    # Check content stream for text operators
    try:
        cs = pikepdf.parse_content_stream(page)
        for operands, operator in cs:
            op_str = str(operator)
            if op_str in ("Tj", "TJ", "'", '"'):
                has_vector_text = True
                break
    except Exception:
        pass

    if has_images and has_vector_text:
        return PageKind.HYBRID
    elif has_images:
        return PageKind.SCAN
    elif has_vector_text:
        return PageKind.VECTOR
    else:
        return PageKind.VECTOR  # empty or minimal page


# ---------------------------------------------------------------------------
# Revision history inspection
# ---------------------------------------------------------------------------


def _check_revisions(pdf: pikepdf.Pdf) -> Optional[int]:
    """Check if a clean pre-watermark revision exists.

    Returns the byte offset of the clean prior revision if found, else None.
    """
    try:
        filename = getattr(pdf, "filename", None)
        if filename and os.path.exists(str(filename)):
            with open(str(filename), "rb") as f:
                data = f.read()
            matches = [m.start() for m in re.finditer(b"%%EOF", data)]
            if len(matches) > 1:
                # Prior revision ends at matches[0] + 5
                return matches[0] + 5
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main probe entry point
# ---------------------------------------------------------------------------


def probe_structural(pdf: pikepdf.Pdf, page_num: int, fitz_page=None) -> ProbeRecord:
    """Run L0 structural probe on a single page.

    Returns a ProbeRecord with all structural fields populated.
    Raster stats are left as defaults — filled in by probe_raster().

    This function NEVER modifies the input PDF.
    """
    page = pdf.pages[page_num]
    total_pages = len(pdf.pages)

    geom = _get_page_geometry(page)
    kind = _classify_page_kind(page, fitz_page=fitz_page)
    rev_clean = _check_revisions(pdf)
    annots = _extract_annots(page)
    ocgs = _extract_ocgs(pdf)  # document-level, same for all pages
    artifacts = _extract_artifacts(page)
    gs_alpha = _extract_gs_alpha(page)
    text_cand = _extract_text_candidates(page, page_num, total_pages, fitz_page=fitz_page)
    xobj_cand = _extract_xobj_candidates(page, kind=kind, fitz_page=fitz_page)
    path_cand = _extract_path_candidates(page)

    return ProbeRecord(
        p=page_num + 1,  # 1-indexed
        kind=kind,
        geom=geom,
        rev_clean=rev_clean,
        annots=annots,
        ocgs=ocgs,
        artifacts=artifacts,
        gs_alpha=gs_alpha,
        text_cand=text_cand,
        xobj_cand=xobj_cand,
        path_cand=path_cand,
        ras_stats=RasterStats(),  # populated by raster_stats.py
    )


# ---------------------------------------------------------------------------
# SQLite Spill for Large Documents (Task 1.2.2)
# ---------------------------------------------------------------------------


class ProbeSpillDB:
    """Temporary SQLite database to spill ProbeRecords for large documents.

    Keeps RAM usage strictly bounded (O(1)) regardless of whether the PDF
    has 10 pages or 50,000 pages.
    """

    def __init__(self, db_path: Optional[str] = None):
        import sqlite3
        import tempfile

        self.temp_file: Any = None
        if db_path is None:
            self.temp_file = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
            self.db_path = self.temp_file.name
            self.temp_file.close()
        else:
            self.db_path = db_path

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS probe_records (page_num INT PRIMARY KEY, json_data TEXT)"
        )
        self.conn.commit()

    def save(self, record: ProbeRecord) -> None:
        """Save a ProbeRecord to SQLite spill."""
        data = record.model_dump_json()
        self.conn.execute(
            "INSERT OR REPLACE INTO probe_records (page_num, json_data) VALUES (?, ?)",
            (record.p, data),
        )
        self.conn.commit()

    def get(self, page_num: int) -> Optional[ProbeRecord]:
        """Retrieve a ProbeRecord by 1-indexed page number."""
        cur = self.conn.cursor()
        cur.execute("SELECT json_data FROM probe_records WHERE page_num = ?", (page_num,))
        row = cur.fetchone()
        if row:
            return ProbeRecord.model_validate_json(row[0])
        return None

    def get_all(self) -> list[ProbeRecord]:
        """Retrieve all ProbeRecords ordered by page number."""
        cur = self.conn.cursor()
        cur.execute("SELECT json_data FROM probe_records ORDER BY page_num ASC")
        return [ProbeRecord.model_validate_json(row[0]) for row in cur.fetchall()]

    def close(self) -> None:
        """Close database and delete temporary file."""
        self.conn.close()
        if self.temp_file is not None and os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
