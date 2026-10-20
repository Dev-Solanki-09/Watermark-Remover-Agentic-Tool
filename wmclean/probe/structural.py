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
            results.append(
                ExtGStateInfo(name=str(name), ca=ca, CA=CA, bm=bm, smask=smask)
            )

    return results


# ---------------------------------------------------------------------------
# Text watermark candidate extraction
# ---------------------------------------------------------------------------


def _extract_text_candidates(
    page: pikepdf.Page, page_num: int, total_pages: int, fitz_page=None
) -> list[TextCandidate]:
    """Extract text spans that match watermark patterns.

    Uses PyMuPDF text dictionary inspection for fast, precise span
    extraction (bbox, font, size, rotation, color, normalized string).
    """
    results = []
    if fitz_page is not None:
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
                        is_light = (r > 0.4 or g > 0.4 or b > 0.4)

                        # Watermark candidate if: rotated text (watermark/title angle) OR keyword text that is large/light/rotated
                        if (is_rotated and (has_kw or font_size >= 14.0)) or (has_kw and (font_size >= 20.0 or is_light or is_rotated)):
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


def _extract_xobj_candidates(page: pikepdf.Page, kind: Optional[PageKind] = None) -> list[XObjCandidate]:
    """Extract image and form XObject candidates from page resources."""
    results = []
    resources = page.get("/Resources")
    if resources is None:
        return results

    xobjects = resources.get("/XObject")
    if xobjects is None:
        return results

    # Scanned pages have a single full-page background image which is the document itself
    if kind == PageKind.SCAN and len(xobjects) == 1:
        return results

    for name, xobj_ref in xobjects.items():
        try:
            xobj = xobj_ref.resolve() if hasattr(xobj_ref, "resolve") else xobj_ref
        except Exception:
            continue

        subtype = str(xobj.get("/Subtype", ""))
        if subtype == "/Image":
            xobj_kind = "IMAGE"
        elif subtype == "/Form":
            xobj_kind = "FORM"
        else:
            continue

        xref = xobj_ref.objgen[0] if hasattr(xobj_ref, "objgen") else 0

        # BBox for form XObjects; for images we need CTM from content stream
        bb = [0.0, 0.0, 0.0, 0.0]
        if xobj_kind == "FORM" and "/BBox" in xobj:
            bb = [float(x) for x in xobj["/BBox"]]

        results.append(
            XObjCandidate(
                name=str(name),
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

    for operands, operator in cs:
        op_str = str(operator)

        if op_str == "gs" and operands:
            # Track alpha from ExtGState — will be refined with actual gs lookup
            pass

        if op_str in ("m", "l", "c", "v", "y", "re"):
            if path_start is None:
                path_start = op_count
            path_ops.append((op_str, [str(o) for o in operands]))

        elif op_str in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*"):
            if path_ops and len(path_ops) >= 2:
                # Compute topology hash
                topo_key = "|".join(f"{op}:{','.join(args)}" for op, args in path_ops)
                h = "p_" + hashlib.sha256(topo_key.encode()).hexdigest()[:6]

                results.append(
                    PathCandidate(
                        hash=h,
                        ops=len(path_ops),
                        bb=[0.0, 0.0, 0.0, 0.0],  # refined later with CTM
                        ca=current_alpha,
                        is_clip=False,
                        doc_freq=0.0,  # computed after all pages probed
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


def _classify_page_kind(page: pikepdf.Page) -> PageKind:
    """Classify page as VECTOR, SCAN, SCAN_OCR, HYBRID, or BITONAL."""
    resources = page.get("/Resources")
    has_images = False
    has_vector_text = False

    if resources:
        xobjects = resources.get("/XObject")
        if xobjects:
            for name, xobj_ref in xobjects.items():
                try:
                    xobj = xobj_ref.resolve() if hasattr(xobj_ref, "resolve") else xobj_ref
                    if str(xobj.get("/Subtype", "")) == "/Image":
                        has_images = True
                        # Check for bitonal
                        bpc = int(xobj.get("/BitsPerComponent", 8))
                        if bpc == 1:
                            return PageKind.BITONAL
                except Exception:
                    continue

        fonts = resources.get("/Font")
        if fonts and len(fonts) > 0:
            has_vector_text = True

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


def probe_structural(
    pdf: pikepdf.Pdf, page_num: int, fitz_page=None
) -> ProbeRecord:
    """Run L0 structural probe on a single page.

    Returns a ProbeRecord with all structural fields populated.
    Raster stats are left as defaults — filled in by probe_raster().

    This function NEVER modifies the input PDF.
    """
    page = pdf.pages[page_num]
    total_pages = len(pdf.pages)

    geom = _get_page_geometry(page)
    kind = _classify_page_kind(page)
    rev_clean = _check_revisions(pdf)
    annots = _extract_annots(page)
    ocgs = _extract_ocgs(pdf)  # document-level, same for all pages
    artifacts = _extract_artifacts(page)
    gs_alpha = _extract_gs_alpha(page)
    text_cand = _extract_text_candidates(page, page_num, total_pages, fitz_page=fitz_page)
    xobj_cand = _extract_xobj_candidates(page, kind=kind)
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
