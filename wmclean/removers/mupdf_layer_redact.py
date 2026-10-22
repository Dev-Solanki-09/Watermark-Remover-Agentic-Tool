"""PyMuPDF C++ Layer-Selective Redaction Engine.

Uses PyMuPDF's (fitz) native C++ redaction engine to surgically strip
only one specific carrier layer (text, raster image, or line-art) inside
a bounding box while ensuring the other layers remain 100% byte-untouched.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import pymupdf as fitz

logger = logging.getLogger(__name__)

PDF_REDACT_TEXT_REMOVE: int = getattr(fitz, "PDF_REDACT_TEXT_REMOVE", 0)
PDF_REDACT_TEXT_NONE: int = getattr(fitz, "PDF_REDACT_TEXT_NONE", 2)
PDF_REDACT_IMAGE_NONE: int = getattr(fitz, "PDF_REDACT_IMAGE_NONE", 0)
PDF_REDACT_IMAGE_PIXELS: int = getattr(fitz, "PDF_REDACT_IMAGE_PIXELS", 1)
PDF_REDACT_LINE_ART_NONE: int = getattr(fitz, "PDF_REDACT_LINE_ART_NONE", 0)


def mupdf_layer_redact(
    page: fitz.Page,
    bbox: Union[tuple[float, float, float, float], list[float], fitz.Rect],
    strip_layer: str = "text",  # "text", "image", or "both"
    text_mode: int = PDF_REDACT_TEXT_REMOVE,
    image_mode: int = PDF_REDACT_IMAGE_NONE,
    graphics_mode: int = PDF_REDACT_LINE_ART_NONE,
    redact_layer: Optional[str] = None,
) -> bool:
    """Apply C++ layer-selective redaction to a page within a specified bounding box.

    Args:
        page: PyMuPDF fitz.Page object.
        bbox: Bounding box (x0, y0, x1, y1) in PDF point coordinates.
        strip_layer: Convenience selector:
                     - "text": remove text only, keep images and line art intact.
                     - "image": blank image pixels only, keep text and line art intact.
                     - "custom": use explicitly provided modes.
        text_mode: PyMuPDF PDF_REDACT_TEXT_* flag.
        image_mode: PyMuPDF PDF_REDACT_IMAGE_* flag.
        graphics_mode: PyMuPDF PDF_REDACT_LINE_ART_* flag.
        redact_layer: Optional alias for strip_layer.

    Returns:
        bool: True if redaction succeeded.
    """
    if redact_layer is not None:
        strip_layer = redact_layer
    rect = fitz.Rect(bbox)
    if rect.is_empty or rect.is_infinite:
        logger.warning("mupdf_layer_redact: invalid or empty bbox %s", bbox)
        return False

    if strip_layer == "text":
        t_mode = PDF_REDACT_TEXT_REMOVE
        i_mode = PDF_REDACT_IMAGE_NONE
        g_mode = PDF_REDACT_LINE_ART_NONE
    elif strip_layer == "image":
        t_mode = PDF_REDACT_TEXT_NONE
        i_mode = PDF_REDACT_IMAGE_PIXELS
        g_mode = PDF_REDACT_LINE_ART_NONE
    else:
        t_mode = text_mode
        i_mode = image_mode
        g_mode = graphics_mode

    try:
        # Add redaction annotation (invisible, no fill)
        annot = page.add_redact_annot(rect)
        if annot is None:
            return False

        # Apply C++ layer-selective redactions
        page.apply_redactions(
            images=i_mode,
            graphics=g_mode,
            text=t_mode,
        )
        logger.info(
            "mupdf_layer_redact: stripped %s layer in bbox [%.1f, %.1f, %.1f, %.1f]",
            strip_layer,
            rect.x0,
            rect.y0,
            rect.x1,
            rect.y1,
        )
        return True
    except Exception as exc:
        logger.warning("mupdf_layer_redact failed on page %d: %s", page.number, exc)
        return False
