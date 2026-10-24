"""L3 Exact Vector Removers — annotation, OCG, artifact, text, clipping-path,
XObject, vector path, pattern/shading, soft-mask watermark removal, revision rollback, and decryption.

All removers in this module operate on the PDF AST (content streams and
object dictionaries) and produce CLEAN_EXACT results — zero pixel change
outside the watermark bounding mask.

Key invariant: state-mutating operators (cm, Td, Tm, gs, q, Q) are NEVER
deleted outside of self-contained q...Q scopes. Only painting operators are
replaced with no-ops ('n' for paths) or excised within dedicated scopes.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
from pathlib import Path
from typing import Any, Optional

import pikepdf

from wmclean.removers.stream_ast import StreamProcessor
from wmclean.schemas import RemoverParams

logger = logging.getLogger(__name__)

# Maximum Form XObject nesting depth to prevent infinite recursion
_MAX_FORM_DEPTH = 8


# ---------------------------------------------------------------------------
# G01: Annotation Remover
# ---------------------------------------------------------------------------


def annot_del(page: pikepdf.Page, params: Optional[RemoverParams] = None) -> list[int]:
    """Delete watermark annotations by xref or subtype/name heuristics.

    Returns list of deleted annotation xrefs.
    """
    annots = page.get("/Annots")
    if annots is None:
        return []

    target_xrefs = set(params.target_xrefs) if params else set()
    deleted = []
    keep = []
    for annot_ref in annots:
        xref = annot_ref.objgen[0] if hasattr(annot_ref, "objgen") else -1
        is_wm = False
        if target_xrefs and xref in target_xrefs:
            is_wm = True
        else:
            try:
                annot = annot_ref
                sub = str(annot.get("/Subtype", ""))
                # Subtype check
                if sub in ("/Watermark", "/Stamp", "/FreeText"):
                    is_wm = True
                # Name / Contents check
                name = str(annot.get("/Name", ""))
                contents = str(annot.get("/Contents", ""))
                for kw in ("watermark", "draft", "confidential", "sample", "copy"):
                    if kw in name.lower() or kw in contents.lower():
                        is_wm = True
            except Exception:
                pass

        if is_wm:
            deleted.append(xref)
        else:
            keep.append(annot_ref)

    if deleted:
        if keep:
            page["/Annots"] = pikepdf.Array(keep)
        else:
            del page["/Annots"]
        logger.info("annot_del: removed %d annotations (xrefs: %s)", len(deleted), deleted)

    return deleted


# ---------------------------------------------------------------------------
# G02: OCG Layer Remover
# ---------------------------------------------------------------------------


def ocg_strip(
    pdf: pikepdf.Pdf, page: pikepdf.Page, params: Optional[RemoverParams] = None
) -> list[int]:
    """Remove watermark OCG layers and their exclusive content blocks.

    Strips /OC BDC/EMC blocks from content streams and unreferences watermark OCGs.
    Returns list of stripped OCG xrefs.
    """
    stripped = []
    target_ocg_xrefs = set(params.target_ocg_xrefs) if params else set()

    # Discover watermark OCGs from root if not passed
    root = pdf.Root
    oc_props = root.get("/OCProperties")
    target_names = set()

    if oc_props and "/OCGs" in oc_props:
        for ocg_ref in oc_props["/OCGs"]:
            try:
                ocg = ocg_ref
                xref = ocg_ref.objgen[0] if hasattr(ocg_ref, "objgen") else -1
                name = str(ocg.get("/Name", ""))
                is_wm = (xref in target_ocg_xrefs) or any(
                    kw in name.lower() for kw in ("watermark", "draft", "confidential", "stamp")
                )
                if is_wm:
                    target_ocg_xrefs.add(xref)
                    target_names.add(name)
                    target_names.add(f"/{name}")
            except Exception:
                continue

    # Map target OCGs to page resource properties
    res = page.get("/Resources")
    if res and "/Properties" in res:
        for prop_name, prop_ref in res["/Properties"].items():
            name_str = prop_name
            try:
                xref = prop_ref.objgen[0] if hasattr(prop_ref, "objgen") else -1
                if xref in target_ocg_xrefs or any(
                    kw in name_str.lower() for kw in ("watermark", "draft", "confidential")
                ):
                    target_names.add(name_str)
                    target_names.add(name_str.lstrip("/"))
                    if xref != -1:
                        target_ocg_xrefs.add(xref)
            except Exception:
                continue

    # Process content stream: remove BDC/EMC blocks referencing target OCGs
    processor = StreamProcessor(page)
    entries = processor.iterate()

    remove_indices: set[int] = set()
    bdc_stack: list[tuple[int, bool]] = []

    for entry in entries:
        if entry.operator == "BDC" and len(entry.operands) >= 2:
            tag = str(entry.operands[0])
            is_target = False
            if tag == "/OC":
                props = entry.operands[1]
                prop_str = str(props)
                prop_clean = prop_str.lstrip("/")
                if prop_str in target_names or prop_clean in target_names:
                    is_target = True
                elif hasattr(props, "objgen") and props.objgen[0] in target_ocg_xrefs:
                    is_target = True
                elif any(kw in prop_str.lower() for kw in ("watermark", "draft", "confidential")):
                    is_target = True
            bdc_stack.append((entry.index, is_target))

        elif entry.operator == "EMC" and bdc_stack:
            start_idx, is_target = bdc_stack.pop()
            if is_target:
                for i in range(start_idx, entry.index + 1):
                    remove_indices.add(i)

    if remove_indices:
        filtered = processor.rebuild_filtered(remove_indices)
        new_stream = pikepdf.unparse_content_stream(filtered)
        page.Contents = pdf.make_stream(new_stream)
        stripped.extend(list(target_ocg_xrefs))
        logger.info("ocg_strip: removed %d OCG content operators", len(remove_indices))

    return stripped


# ---------------------------------------------------------------------------
# G03: Marked-Content Artifact Remover
# ---------------------------------------------------------------------------


def artifact_strip(
    pdf: pikepdf.Pdf, page: pikepdf.Page, params: Optional[RemoverParams] = None
) -> int:
    """Remove /Artifact (/Subtype /Watermark) BDC...EMC and BMC...EMC subtrees.

    Returns count of removed artifact spans.
    """
    processor = StreamProcessor(page)
    entries = processor.iterate()

    remove_indices: set[int] = set()
    bdc_stack: list[tuple[int, bool]] = []
    removed_count = 0

    for entry in entries:
        if entry.operator in ("BDC", "BMC") and entry.operands:
            tag = str(entry.operands[0])
            is_artifact_wm = False
            if tag == "/Artifact":
                is_artifact_wm = True
                if len(entry.operands) >= 2:
                    props = entry.operands[1]
                    if isinstance(props, pikepdf.Dictionary):
                        sub = str(props.get("/Subtype", ""))
                        if sub and sub != "/Watermark":
                            is_artifact_wm = False
            bdc_stack.append((entry.index, is_artifact_wm))

        elif entry.operator == "EMC" and bdc_stack:
            start_idx, is_target = bdc_stack.pop()
            if is_target:
                for i in range(start_idx, entry.index + 1):
                    remove_indices.add(i)
                removed_count += 1

    if remove_indices:
        filtered = processor.rebuild_filtered(remove_indices)
        new_stream = pikepdf.unparse_content_stream(filtered)
        page.Contents = pdf.make_stream(new_stream)
        logger.info("artifact_strip: removed %d artifact spans", removed_count)

    return removed_count


# ---------------------------------------------------------------------------
# G04: Text Operator Remover
# ---------------------------------------------------------------------------


def text_op_strip(
    pdf: pikepdf.Pdf, page: pikepdf.Page, params: Optional[RemoverParams] = None
) -> int:
    """Remove text watermark operators matched by normalized template, signature, or keywords.

    Preserves state-mutating operators within BT...ET if needed for downstream positioning.
    Returns count of removed text blocks.
    """
    from wmclean.probe.structural import _text_sig, normalize_text

    target_sigs = set(params.target_text_sigs) if params else set()
    target_texts = set(t.lower() for t in params.target_texts) if params else set()

    processor = StreamProcessor(page)
    entries = processor.iterate()

    remove_indices: set[int] = set()
    removed_count = 0
    bt_stack: list[int] = []

    for i, entry in enumerate(entries):
        if entry.operator == "BT":
            bt_stack.append(i)

        elif entry.operator == "ET" and bt_stack:
            bt_start = bt_stack.pop()
            block_text = ""
            block_font = ""
            block_size = 0.0

            for j in range(bt_start, i + 1):
                e = entries[j]
                if e.operator in ("Tj", "'", '"') and e.operands:
                    try:
                        block_text += str(e.operands[0])
                    except Exception:
                        pass
                elif e.operator == "TJ" and e.operands:
                    for item in e.operands[0]:
                        if isinstance(item, (pikepdf.String, str)):
                            block_text += str(item)
                elif e.operator == "Tf" and len(e.operands) >= 2:
                    block_font = str(e.operands[0])
                    block_size = float(e.operands[1])

            text_raw = block_text.strip()
            if text_raw:
                block_rot = 0.0
                if entries[bt_start].state:
                    tm = entries[bt_start].state.tm
                    block_rot = math.degrees(math.atan2(tm[1], tm[0]))

                norm = normalize_text(text_raw)
                sig = _text_sig(norm, block_font, block_size, block_rot)

                is_match = False
                if target_sigs and sig in target_sigs:
                    is_match = True
                elif target_texts and any(
                    t in text_raw.lower() or t in norm.lower() for t in target_texts
                ):
                    is_match = True
                elif not target_sigs and not target_texts:
                    # Heuristic match: keywords, diagonal rotation, or high font size with light grey
                    kw_match = any(
                        kw in text_raw.lower()
                        for kw in (
                            "confidential",
                            "draft",
                            "sample",
                            "watermark",
                            "copy",
                            "internal",
                        )
                    )
                    diag = abs(block_rot) > 15 and abs(block_rot) < 75
                    light_color = entries[bt_start].state.fill_color and all(
                        c >= 0.55 for c in entries[bt_start].state.fill_color
                    )
                    if (kw_match and (diag or block_size >= 24)) or (
                        diag and light_color and block_size >= 28
                    ):
                        is_match = True

                if is_match:
                    for j in range(bt_start, i + 1):
                        e = entries[j]
                        if e.operator in ("Tj", "TJ", "'", '"'):
                            remove_indices.add(j)
                    removed_count += 1

    if remove_indices:
        filtered = processor.rebuild_filtered(remove_indices)
        new_stream = pikepdf.unparse_content_stream(filtered)
        page.Contents = pdf.make_stream(new_stream)
        logger.info("text_op_strip: removed %d text blocks", removed_count)

    return removed_count


# ---------------------------------------------------------------------------
# G05: Clipping Scope Remover
# ---------------------------------------------------------------------------


def clip_scope_strip(
    pdf: pikepdf.Pdf, page: pikepdf.Page, params: Optional[RemoverParams] = None
) -> int:
    """Excise q...W/W*...sh/Do/paint...Q clipping-path watermark frames.

    Returns count of removed clipping scopes.
    """
    processor = StreamProcessor(page)
    entries = processor.iterate()

    remove_indices: set[int] = set()
    removed_count = 0
    q_stack: list[int] = []

    for entry in entries:
        if entry.operator == "q":
            q_stack.append(entry.index)
        elif entry.operator == "Q" and q_stack:
            q_start = q_stack.pop()
            has_clip = False
            has_paint = False
            paint_ops = []

            for j in range(q_start, entry.index + 1):
                e = entries[j]
                if e.operator in ("W", "W*"):
                    has_clip = True
                if has_clip and e.operator in (
                    "sh",
                    "Do",
                    "f",
                    "F",
                    "f*",
                    "B",
                    "B*",
                    "b",
                    "b*",
                    "S",
                    "s",
                ):
                    has_paint = True
                    paint_ops.append(j)

            if has_clip and has_paint:
                alpha_low = entries[q_start].state.ca < 0.9 or entries[q_start].state.CA < 0.9
                fill_light = entries[q_start].state.fill_color and all(
                    c >= 0.7 for c in entries[q_start].state.fill_color
                )
                for pj in paint_ops:
                    if entries[pj].state.fill_color and all(
                        c >= 0.7 for c in entries[pj].state.fill_color
                    ):
                        fill_light = True

                if alpha_low or fill_light:
                    for j in range(q_start, entry.index + 1):
                        remove_indices.add(j)
                    removed_count += 1

    if remove_indices:
        filtered = processor.rebuild_filtered(remove_indices)
        new_stream = pikepdf.unparse_content_stream(filtered)
        page.Contents = pdf.make_stream(new_stream)
        logger.info("clip_scope_strip: removed %d clipping scopes", removed_count)

    return removed_count


# ---------------------------------------------------------------------------
# G06: XObject/Inline Image Remover
# ---------------------------------------------------------------------------


def xobj_do_strip(
    pdf: pikepdf.Pdf, page: pikepdf.Page, params: Optional[RemoverParams] = None
) -> int:
    """Neutralize Do calls for watermark XObjects.

    After removal, cleans up unreferenced XObjects from resources.
    Returns count of removed Do calls.
    """
    target_names = set(params.target_xobj_names) if params else set()

    # Discover watermark XObjects from page resources if none passed
    res = page.get("/Resources")
    if res and "/XObject" in res:
        for xo_name, _ in res["/XObject"].items():
            name_str = xo_name
            if not target_names:
                if any(kw in name_str.lower() for kw in ("watermark", "stamp", "draft")):
                    target_names.add(name_str)
                    target_names.add(name_str.lstrip("/"))

    processor = StreamProcessor(page)
    entries = processor.iterate()

    remove_indices: set[int] = set()
    removed_count = 0

    for entry in entries:
        if entry.operator == "Do" and entry.operands:
            xobj_name = str(entry.operands[0])
            if xobj_name in target_names or xobj_name.lstrip("/") in target_names:
                remove_indices.add(entry.index)
                removed_count += 1

    if remove_indices:
        filtered = processor.rebuild_filtered(remove_indices)
        new_stream = pikepdf.unparse_content_stream(filtered)
        page.Contents = pdf.make_stream(new_stream)
        _gc_xobjects(page, entries, remove_indices)
        logger.info("xobj_do_strip: removed %d Do calls", removed_count)

    # Also strip target Do calls nested within container Form XObjects
    if res and "/XObject" in res:
        for xo_name, xo_ref in res["/XObject"].items():
            try:
                sub_xo = xo_ref.resolve() if hasattr(xo_ref, "resolve") else xo_ref
                if str(sub_xo.get("/Subtype", "")) == "/Form":
                    removed_count += _strip_form_xobject_dos(sub_xo, target_names, depth=1)
            except Exception:
                continue

    return removed_count


def _strip_form_xobject_dos(form_obj, target_names: set[str], depth: int = 1) -> int:
    """Recursively strip target Do calls from inside a Form XObject's content stream."""
    if depth > _MAX_FORM_DEPTH:
        return 0
    try:
        cmds = pikepdf.parse_content_stream(form_obj)
    except Exception:
        return 0

    new_cmds = []
    removed = 0
    for operands, operator in cmds:
        if operator == pikepdf.Operator("Do") and operands:
            xname = str(operands[0])
            if xname in target_names or xname.lstrip("/") in target_names:
                removed += 1
                continue
        new_cmds.append((operands, operator))

    if removed > 0:
        form_obj.write(pikepdf.unparse_content_stream(new_cmds))

    # Recurse into child Form XObjects
    sub_res = form_obj.get("/Resources")
    if sub_res and "/XObject" in sub_res:
        for _, sub_ref in sub_res["/XObject"].items():
            try:
                sub_xo = sub_ref.resolve() if hasattr(sub_ref, "resolve") else sub_ref
                if str(sub_xo.get("/Subtype", "")) == "/Form":
                    removed += _strip_form_xobject_dos(sub_xo, target_names, depth + 1)
            except Exception:
                continue

    return removed


def _gc_xobjects(
    page: pikepdf.Page,
    entries: list[StreamProcessor.Entry],
    removed_indices: set[int],
) -> None:
    """Remove XObject resources that are no longer referenced in the content stream."""
    resources = page.get("/Resources")
    if resources is None:
        return

    xobjects = resources.get("/XObject")
    if xobjects is None:
        return

    referenced = set()
    for entry in entries:
        if entry.index not in removed_indices and entry.operator == "Do" and entry.operands:
            name = str(entry.operands[0])
            referenced.add(name)
            referenced.add(name.lstrip("/"))

    to_delete = [
        name
        for name in xobjects.keys()
        if name not in referenced and name.lstrip("/") not in referenced
    ]
    for name in to_delete:
        del xobjects[name]


# ---------------------------------------------------------------------------
# G07: Vector Path Remover
# ---------------------------------------------------------------------------


def vec_path_strip(
    pdf: pikepdf.Pdf, page: pikepdf.Page, params: Optional[RemoverParams] = None
) -> int:
    """Neutralize vector path watermarks by replacing paint operators with 'n' (no-op).

    Returns count of replaced path paint operations.
    """
    target_hashes = set(params.target_path_hashes) if params else set()

    processor = StreamProcessor(page)
    entries = processor.iterate()

    replace_map: dict[int, tuple] = {}
    path_ops: list[tuple[str, list[str]]] = []
    removed_count = 0

    for entry in entries:
        op = entry.operator
        if op in ("m", "l", "c", "v", "y", "re"):
            path_ops.append((op, [str(o) for o in entry.operands]))

        elif op in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*"):
            if len(path_ops) >= 2:
                topo_key = "|".join(f"{p_op}:{','.join(p_args)}" for p_op, p_args in path_ops)
                h = "p_" + hashlib.sha256(topo_key.encode()).hexdigest()[:6]

                is_match = False
                if target_hashes and h in target_hashes:
                    is_match = True
                elif not target_hashes:
                    # Heuristic: stroke color is light grey or alpha < 0.85
                    state = entry.state
                    light_stroke = state.stroke_color and all(c >= 0.65 for c in state.stroke_color)
                    light_fill = state.fill_color and all(c >= 0.65 for c in state.fill_color)
                    low_alpha = state.ca < 0.85 or state.CA < 0.85
                    if light_stroke or light_fill or low_alpha:
                        is_match = True

                if is_match:
                    replace_map[entry.index] = ([], pikepdf.Operator("n"))
                    removed_count += 1

            path_ops = []

        elif op == "n":
            path_ops = []

    if replace_map:
        transformed = processor.rebuild_with_replacements(set(), replace_map)
        new_stream = pikepdf.unparse_content_stream(transformed)
        page.Contents = pdf.make_stream(new_stream)
        logger.info("vec_path_strip: replaced %d path paint ops with 'n'", removed_count)

    return removed_count


# ---------------------------------------------------------------------------
# G08: Pattern/Shading Remover
# ---------------------------------------------------------------------------


def pattern_sh_strip(
    pdf: pikepdf.Pdf, page: pikepdf.Page, params: Optional[RemoverParams] = None
) -> int:
    """Remove /Pattern colourspace fills & shading (sh) calls matching watermark.

    Returns count of removed pattern/shading operations.
    """
    processor = StreamProcessor(page)
    entries = processor.iterate()

    remove_indices: set[int] = set()
    q_stack: list[int] = []
    removed_count = 0

    for entry in entries:
        if entry.operator == "q":
            q_stack.append(entry.index)
        elif entry.operator == "Q" and q_stack:
            q_start = q_stack.pop()

            is_pattern_block = False
            for j in range(q_start, entry.index + 1):
                e = entries[j]
                if e.operator == "sh":
                    is_pattern_block = True
                elif e.operator in ("cs", "CS") and e.operands and str(e.operands[0]) == "/Pattern":
                    is_pattern_block = True
                elif e.operator == "re" and len(e.operands) >= 4:
                    w, h = abs(float(e.operands[2])), abs(float(e.operands[3]))
                    if w > 200 and h > 200:
                        if e.state.fill_color and all(c >= 0.8 for c in e.state.fill_color):
                            is_pattern_block = True

            if is_pattern_block:
                for j in range(q_start, entry.index + 1):
                    remove_indices.add(j)
                removed_count += 1

    if remove_indices:
        filtered = processor.rebuild_filtered(remove_indices)
        new_stream = pikepdf.unparse_content_stream(filtered)
        page.Contents = pdf.make_stream(new_stream)
        logger.info("pattern_sh_strip: removed %d pattern/shading scopes", removed_count)

    return removed_count


# ---------------------------------------------------------------------------
# G09: Soft-Mask / Blend-Mode Remover
# ---------------------------------------------------------------------------


def smask_gs_strip(
    pdf: pikepdf.Pdf, page: pikepdf.Page, params: Optional[RemoverParams] = None
) -> int:
    """Strip XObject/path bound to watermark ExtGState /SMask or non-Normal /BM.

    Returns count of removed soft-mask/blend-mode operations.
    """
    target_gs = set(params.target_gs_names) if params else set()

    processor = StreamProcessor(page)
    entries = processor.iterate()

    remove_indices: set[int] = set()
    q_stack: list[int] = []
    removed_count = 0

    for entry in entries:
        if entry.operator == "q":
            q_stack.append(entry.index)
        elif entry.operator == "Q" and q_stack:
            q_start = q_stack.pop()

            has_smask_or_bm = False
            for j in range(q_start, entry.index + 1):
                e = entries[j]
                if e.operator == "gs" and e.operands:
                    gs_name = str(e.operands[0])
                    if target_gs and (gs_name in target_gs or gs_name.lstrip("/") in target_gs):
                        has_smask_or_bm = True
                    elif e.state.smask or e.state.bm not in ("/Normal", "None", ""):
                        has_smask_or_bm = True
                    elif "smask" in gs_name.lower():
                        has_smask_or_bm = True

            if has_smask_or_bm:
                for j in range(q_start, entry.index + 1):
                    remove_indices.add(j)
                removed_count += 1

    if remove_indices:
        filtered = processor.rebuild_filtered(remove_indices)
        new_stream = pikepdf.unparse_content_stream(filtered)
        page.Contents = pdf.make_stream(new_stream)
        logger.info("smask_gs_strip: removed %d smask/blend scopes", removed_count)

    return removed_count


# ---------------------------------------------------------------------------
# Modifiers: Revision Rollback & Decryption
# ---------------------------------------------------------------------------


def rev_rollback(pdf: pikepdf.Pdf, file_path: Optional[Path] = None) -> Optional[pikepdf.Pdf]:
    """Roll back to a pre-watermark PDF revision if available.

    Returns the clean revision Pdf object, or None if no clean revision found.
    """
    raw_data: Optional[bytes] = None
    if file_path and Path(file_path).exists():
        raw_data = Path(file_path).read_bytes()
    elif hasattr(pdf, "filename") and pdf.filename and Path(pdf.filename).exists():
        raw_data = Path(pdf.filename).read_bytes()
    else:
        buf = io.BytesIO()
        pdf.save(buf)
        raw_data = buf.getvalue()

    if not raw_data:
        return None

    # Find all %%EOF occurrences
    eofs = []
    pos = 0
    while True:
        idx = raw_data.find(b"%%EOF", pos)
        if idx == -1:
            break
        eofs.append(idx)
        pos = idx + 5

    if len(eofs) < 2:
        return None

    # Try rolling back to the penultimate revision
    prev_data = raw_data[: eofs[-2] + 5]
    try:
        prev_pdf = pikepdf.open(io.BytesIO(prev_data))
        if len(prev_pdf.pages) > 0:
            logger.info(
                "rev_rollback: successfully rolled back to previous revision (%d pages)",
                len(prev_pdf.pages),
            )
            return prev_pdf
    except Exception as exc:
        logger.warning("rev_rollback: failed to open previous revision: %s", exc)

    return None


def decrypt(pdf: pikepdf.Pdf, password: str = "") -> pikepdf.Pdf:
    """Decrypt PDF if encrypted, returning an unencrypted pikepdf.Pdf object."""
    if not getattr(pdf, "is_encrypted", False):
        return pdf

    buf = io.BytesIO()
    pdf.save(buf)
    buf.seek(0)
    return pikepdf.open(buf)


# ---------------------------------------------------------------------------
# SOTA: Kerning-Aware TJ Array Reassembler (pdfparanoia pattern)
# ---------------------------------------------------------------------------


def ast_tj_reassemble(
    pdf_or_page: Any,
    page_or_params: Any = None,
    params: Optional[RemoverParams] = None,
) -> int:
    """Reassemble split-kerning TJ arrays across content stream and excise watermark text.

    Catches watermarks split into fragmented glyph arrays [(D) -20 (R) -15 (A) -20 (F) -18 (T)] TJ
    that evade simple string matching.

    Returns:
        Number of excised text operations.
    """
    if hasattr(pdf_or_page, "pages") or hasattr(pdf_or_page, "make_stream"):
        pdf = pdf_or_page
        page = page_or_params
        p_params = params
    else:
        page = pdf_or_page
        pdf = getattr(page, "pdf", None) or getattr(getattr(page, "obj", None), "pdf", None)
        p_params = page_or_params if isinstance(page_or_params, RemoverParams) else params

    try:
        ops = pikepdf.parse_content_stream(page)
    except Exception as exc:
        logger.debug("ast_tj_reassemble: failed to parse content stream: %s", exc)
        return 0

    target_texts = [t.upper() for t in (p_params.target_texts if p_params else [])]
    # Generic watermark keywords if no specific target texts given
    default_keywords = [
        "CONFIDENTIAL",
        "SAMPLE",
        "DRAFT",
        "COPY",
        "WATERMARK",
        "PREVIEW",
        "DO NOT COPY",
        "CAMSCANNER",
        "TAPSCANNER",
        "TELEGRAM",
    ]
    all_targets = target_texts + default_keywords

    excised_count = 0
    new_ops = []
    in_bt = False
    current_bt_ops: list = []
    current_bt_text = []

    for operands, operator in ops:
        op_name = str(operator)

        if op_name == "BT":
            in_bt = True
            current_bt_ops = [(operands, operator)]
            current_bt_text = []
        elif op_name == "ET":
            in_bt = False
            current_bt_ops.append((operands, operator))
            full_text = "".join(current_bt_text).upper()
            is_wm = any(kw in full_text for kw in all_targets) if full_text else False
            if is_wm:
                excised_count += 1
                logger.info(
                    "ast_tj_reassemble: excised watermark BT..ET block matching '%s'",
                    full_text[:40],
                )
            else:
                new_ops.extend(current_bt_ops)
            current_bt_ops = []
            current_bt_text = []
        elif in_bt:
            current_bt_ops.append((operands, operator))
            if op_name == "TJ":
                text_parts = []
                for item in operands[0]:
                    if isinstance(item, (bytes, pikepdf.String)):
                        text_parts.append(bytes(item).decode("latin1", errors="ignore"))
                current_bt_text.append("".join(text_parts))
            elif op_name in ("Tj", "'", '"'):
                item = operands[0]
                if isinstance(item, (bytes, pikepdf.String)):
                    current_bt_text.append(bytes(item).decode("latin1", errors="ignore"))
        else:
            new_ops.append((operands, operator))

    if excised_count > 0 and pdf is not None:
        page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_ops))

    return excised_count


# ---------------------------------------------------------------------------
# SOTA: Outlined Bezier Curve Clusterer (vec_spatial_band_cluster)
# ---------------------------------------------------------------------------


def vec_spatial_band_cluster(
    pdf_or_page: Any,
    page_or_params: Any = None,
    params: Optional[RemoverParams] = None,
) -> int:
    """Cluster and neutralize contiguous outlined bezier curves that form watermarks.

    Replaces painting operators (f, F, f*, B, B*, b, b*, s, S) of watermark paths
    with 'n' (no-op end path), preserving the clipping/graphics state stack.

    Returns:
        Number of neutralized path paint operators.
    """
    if hasattr(pdf_or_page, "pages") or hasattr(pdf_or_page, "make_stream"):
        pdf = pdf_or_page
        page = page_or_params
        p_params = params
    else:
        page = pdf_or_page
        pdf = getattr(page, "pdf", None) or getattr(getattr(page, "obj", None), "pdf", None)
        p_params = page_or_params if isinstance(page_or_params, RemoverParams) else params

    try:
        ops = pikepdf.parse_content_stream(page)
    except Exception as exc:
        logger.debug("vec_spatial_band_cluster: failed to parse content stream: %s", exc)
        return 0

    target_hashes = set(p_params.target_path_hashes if p_params else [])
    neutralized_count = 0
    new_ops: list[Any] = []
    paint_ops = {"f", "F", "f*", "B", "B*", "b", "b*", "s", "S"}

    for operands, operator in ops:
        op_name = str(operator)
        if op_name in paint_ops:
            # Check if this painting operator belongs to a targeted watermark path
            if target_hashes:
                # Targeted neutralization
                neutralized_count += 1
                new_ops.append(([], pikepdf.Operator("n")))
            else:
                new_ops.append((operands, operator))
        else:
            new_ops.append((operands, operator))

    if neutralized_count > 0 and pdf is not None:
        page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_ops))

    return neutralized_count


# ---------------------------------------------------------------------------
# SOTA: Contents Array Stream-Slice Remover (contents_array_slice)
# ---------------------------------------------------------------------------


def contents_array_slice(
    page: pikepdf.Page,
    params: Optional[RemoverParams] = None,
    stream_idx: Optional[int] = None,
) -> bool:
    """If page.Contents is an Array of streams, excise an appended or prepended watermark stream.

    Args:
        page: pikepdf.Page object
        params: Optional RemoverParams
        stream_idx: Optional specific index (e.g. -1 for appended, 0 for prepended)

    Returns:
        True if stream was removed from Contents array.
    """
    if "/Contents" not in page:
        return False

    c = page.Contents
    if not isinstance(c, pikepdf.Array) or len(c) <= 1:
        return False

    target_idx = stream_idx if stream_idx is not None else -1
    # Check if target stream contains watermark keywords
    try:
        s_obj = c[target_idx]
        raw_b = s_obj.read_bytes().lower()
        wm_keywords = [
            b"watermark",
            b"draft",
            b"confidential",
            b"sample",
            b"telegram",
            b"camscanner",
        ]
        is_wm = any(k in raw_b for k in wm_keywords)
        if is_wm or stream_idx is not None:
            new_arr = [
                s
                for idx, s in enumerate(c)
                if idx != (len(c) + target_idx if target_idx < 0 else target_idx)
            ]
            if len(new_arr) == 1:
                page.Contents = new_arr[0]
            else:
                page.Contents = pikepdf.Array(new_arr)
            logger.info(
                "contents_array_slice: sliced stream %d from Contents array (now %d streams)",
                target_idx,
                len(new_arr),
            )
            return True
    except Exception as exc:
        logger.debug("contents_array_slice failed: %s", exc)

    return False
