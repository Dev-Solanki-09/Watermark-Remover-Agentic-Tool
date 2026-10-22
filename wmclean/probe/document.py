"""L0 Document Probe — run full structural and raster probing across all pages of a document."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pikepdf
import pymupdf

from wmclean.probe.raster_stats import RasterReservoir, probe_raster
from wmclean.probe.structural import probe_structural
from wmclean.schemas import ProbeRecord


def probe_document(file_path: str, password: Optional[str] = None) -> List[ProbeRecord]:
    """Run full L0 structural and raster probe on all pages of a document.

    Args:
        file_path: Path to the PDF file.
        password: Optional PDF decryption password.

    Returns:
        List of ProbeRecord instances, one per page.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    pdf = pikepdf.open(str(path), password=password or "")
    fitz_doc = None
    try:
        fitz_doc = pymupdf.open(str(path))
        if password:
            fitz_doc.authenticate(password)
    except Exception:
        fitz_doc = None

    records: List[ProbeRecord] = []
    reservoir = RasterReservoir()

    try:
        total_pages = len(pdf.pages)
        for page_num in range(total_pages):
            fitz_page = fitz_doc[page_num] if fitz_doc and page_num < len(fitz_doc) else None
            record = probe_structural(pdf, page_num, fitz_page=fitz_page)
            record = probe_raster(pdf, page_num, record, reservoir=reservoir, fitz_doc=fitz_doc)
            records.append(record)

        # Cross-page doc_freq for paths
        if records:
            path_counts: dict[str, set[int]] = {}
            for p_idx, rec in enumerate(records):
                for pc in rec.path_cand:
                    path_counts.setdefault(pc.hash, set()).add(p_idx)
            for rec in records:
                for pc in rec.path_cand:
                    pc.doc_freq = len(path_counts.get(pc.hash, set())) / float(len(records))

        return records
    finally:
        if fitz_doc:
            fitz_doc.close()
        pdf.close()
