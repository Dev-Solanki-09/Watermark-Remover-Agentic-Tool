"""Lazy-loading dependency wrappers for SOTA vision & OCR libraries.

Ensures zero startup overhead (<100ms import time) and graceful degradation
if optional packages are unavailable.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RAPIDOCR_INSTANCE: Optional[Any] = None
_RAPIDOCR_CHECKED: bool = False

_PYMATTING_MODULE: Optional[Any] = None
_PYMATTING_CHECKED: bool = False


def has_cv2_contrib() -> bool:
    """Check if opencv-contrib extra modules (e.g. ximgproc) are available."""
    try:
        import cv2

        return hasattr(cv2, "ximgproc")
    except Exception:
        return False


def has_scipy() -> bool:
    """Check if SciPy is available."""
    return importlib.util.find_spec("scipy") is not None


def has_skimage() -> bool:
    """Check if scikit-image is available."""
    return importlib.util.find_spec("skimage") is not None


def has_pymatting() -> bool:
    """Check if pymatting is available."""
    return importlib.util.find_spec("pymatting") is not None


def get_pymatting() -> Optional[Any]:
    """Lazy-load and return pymatting module if installed."""
    global _PYMATTING_MODULE, _PYMATTING_CHECKED
    if _PYMATTING_CHECKED:
        return _PYMATTING_MODULE

    _PYMATTING_CHECKED = True
    try:
        import pymatting

        _PYMATTING_MODULE = pymatting
        logger.debug("pymatting loaded successfully")
    except Exception as exc:
        logger.debug("pymatting not available: %s", exc)
        _PYMATTING_MODULE = None

    return _PYMATTING_MODULE


def has_rapidocr() -> bool:
    """Check if rapidocr-onnxruntime or rapidocr is available."""
    return (
        importlib.util.find_spec("rapidocr_onnxruntime") is not None
        or importlib.util.find_spec("rapidocr") is not None
    )


def get_rapidocr() -> Optional[Any]:
    """Lazy-load and return RapidOCR singleton engine if installed."""
    global _RAPIDOCR_INSTANCE, _RAPIDOCR_CHECKED
    if _RAPIDOCR_CHECKED:
        return _RAPIDOCR_INSTANCE

    _RAPIDOCR_CHECKED = True
    try:
        if importlib.util.find_spec("rapidocr_onnxruntime") is not None:
            from rapidocr_onnxruntime import RapidOCR

            _RAPIDOCR_INSTANCE = RapidOCR()
            logger.debug("RapidOCR (rapidocr_onnxruntime) initialized")
        elif importlib.util.find_spec("rapidocr") is not None:
            from rapidocr import RapidOCR

            _RAPIDOCR_INSTANCE = RapidOCR()
            logger.debug("RapidOCR initialized")
    except Exception as exc:
        logger.debug("RapidOCR initialization failed: %s", exc)
        _RAPIDOCR_INSTANCE = None

    return _RAPIDOCR_INSTANCE


_DOXAPY_MODULE: Optional[Any] = None
_DOXAPY_CHECKED: bool = False


def has_doxapy() -> bool:
    """Check if doxapy C++ SIMD binarization library is available."""
    return importlib.util.find_spec("doxapy") is not None


def get_doxapy() -> Optional[Any]:
    """Lazy-load and return doxapy module if installed."""
    global _DOXAPY_MODULE, _DOXAPY_CHECKED
    if _DOXAPY_CHECKED:
        return _DOXAPY_MODULE

    _DOXAPY_CHECKED = True
    try:
        import doxapy

        _DOXAPY_MODULE = doxapy
        logger.debug("doxapy loaded successfully")
    except Exception as exc:
        logger.debug("doxapy not available: %s", exc)
        _DOXAPY_MODULE = None

    return _DOXAPY_MODULE


_FONTTOOLS_MODULE: Optional[Any] = None
_FONTTOOLS_CHECKED: bool = False


def has_fonttools() -> bool:
    """Check if fonttools is available."""
    return importlib.util.find_spec("fontTools") is not None


def get_fonttools() -> Optional[Any]:
    """Lazy-load and return fontTools module if installed."""
    global _FONTTOOLS_MODULE, _FONTTOOLS_CHECKED
    if _FONTTOOLS_CHECKED:
        return _FONTTOOLS_MODULE

    _FONTTOOLS_CHECKED = True
    try:
        import fontTools

        _FONTTOOLS_MODULE = fontTools
        logger.debug("fontTools loaded successfully")
    except Exception as exc:
        logger.debug("fontTools not available: %s", exc)
        _FONTTOOLS_MODULE = None

    return _FONTTOOLS_MODULE


_SKAN_MODULE: Optional[Any] = None
_SKAN_CHECKED: bool = False


def has_skan() -> bool:
    """Check if skan skeleton analysis library is available."""
    return importlib.util.find_spec("skan") is not None


def get_skan() -> Optional[Any]:
    """Lazy-load and return skan module if installed."""
    global _SKAN_MODULE, _SKAN_CHECKED
    if _SKAN_CHECKED:
        return _SKAN_MODULE

    _SKAN_CHECKED = True
    try:
        import skan

        _SKAN_MODULE = skan
        logger.debug("skan loaded successfully")
    except Exception as exc:
        logger.debug("skan not available: %s", exc)
        _SKAN_MODULE = None

    return _SKAN_MODULE


_SKNW_MODULE: Optional[Any] = None
_SKNW_CHECKED: bool = False


def has_sknw() -> bool:
    """Check if sknw skeleton network library is available."""
    return importlib.util.find_spec("sknw") is not None


def get_sknw() -> Optional[Any]:
    """Lazy-load and return sknw module if installed."""
    global _SKNW_MODULE, _SKNW_CHECKED
    if _SKNW_CHECKED:
        return _SKNW_MODULE

    _SKNW_CHECKED = True
    try:
        import sknw

        _SKNW_MODULE = sknw
        logger.debug("sknw loaded successfully")
    except Exception as exc:
        logger.debug("sknw not available: %s", exc)
        _SKNW_MODULE = None

    return _SKNW_MODULE
