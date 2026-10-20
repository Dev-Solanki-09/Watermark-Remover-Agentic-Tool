"""Unit tests for Phase 4: L3 Raster Removers (FX_G10–FX_G16), Virtual Stitching, and L5 Encoder."""

from pathlib import Path

import cv2
import numpy as np
import pikepdf
import pymupdf

from tests.fixture_gen import ALL_FIXTURE_GENERATORS
from wmclean.assemble.encoder import encode_image
from wmclean.removers.raster_ops import (
    raster_alpha_inv,
    raster_bitonal_despeckle,
    raster_cc_inpaint,
    raster_color_unmix,
    raster_fft_notch,
    raster_luma_sep,
)
from wmclean.removers.stitch import slice_repack, virtual_stitch
from wmclean.schemas import RemoverParams
from wmclean.verify.raster import verify_raster


def _load_first_page_image(pdf_path: Path, dpi: int = 150) -> np.ndarray:
    """Helper to extract page 0 as RGB numpy array via PyMuPDF."""
    with pymupdf.open(str(pdf_path)) as doc:
        pix = doc[0].get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            return img[:, :, :3]
        if pix.n == 1:
            return np.stack([img, img, img], axis=2)
        return img


def test_raster_alpha_inv_g10(fixtures_dir: Path):
    """FX_G10: Light grey alpha-blended raster watermark removed via alpha inversion."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G10"](fixtures_dir)
    wm_img = _load_first_page_image(wm_path, dpi=150)

    # Watermark text in G10 has luminance ~235 (blended on white paper)
    grey = cv2.cvtColor(wm_img, cv2.COLOR_RGB2GRAY) if wm_img.ndim == 3 else wm_img
    h, w = grey.shape[:2]
    roi = np.zeros((h, w), dtype=bool)
    roi[int(h * 0.45):int(h * 0.65), int(w * 0.10):int(w * 0.90)] = True
    mask = (((grey > 200) & (grey < 248)) & roi).astype(np.uint8) * 255

    # Template has watermark value ~190
    template = np.full_like(grey, 190)

    params = RemoverParams(alpha_clamp=0.30)
    cleaned = raster_alpha_inv(wm_img, template, mask, params)

    assert cleaned.shape == wm_img.shape
    v_res = verify_raster(wm_img, cleaned, wm_mask=(mask > 0))
    assert v_res.passed is True
    assert v_res.residual <= 0.035
    assert v_res.content_loss <= 0.020


def test_raster_color_unmix_g11(fixtures_dir: Path):
    """FX_G11: Coloured saturated watermark removed via LAB chroma separation."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G11"](fixtures_dir)
    wm_img = _load_first_page_image(wm_path, dpi=150)

    # Detect high saturation pixels as watermark mask
    hsv = cv2.cvtColor(wm_img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    mask = (sat > 30).astype(np.uint8) * 255

    params = RemoverParams()
    cleaned = raster_color_unmix(wm_img, mask, params)

    assert cleaned.shape == wm_img.shape
    # Watermark region in cleaned image should have saturated color eliminated
    hsv_clean = cv2.cvtColor(cleaned, cv2.COLOR_RGB2HSV)
    assert np.mean(hsv_clean[:, :, 1][mask > 0]) < 10.0


def test_raster_luma_sep_g12(fixtures_dir: Path):
    """FX_G12: Varying-position grey watermark separated via luminance thresholding."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G12"](fixtures_dir)
    wm_img = _load_first_page_image(wm_path, dpi=150)
    clean_ref = _load_first_page_image(clean_path, dpi=150)

    # Use ground-truth watermark diff mask
    diff = cv2.absdiff(wm_img, clean_ref)
    diff_grey = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) if diff.ndim == 3 else diff
    mask = (diff_grey > 8).astype(np.uint8) * 255

    params = RemoverParams(luma_thresh=140)
    cleaned = raster_luma_sep(wm_img, mask, params)

    assert cleaned.shape == wm_img.shape
    v_res = verify_raster(wm_img, cleaned, wm_mask=(mask > 0))
    assert v_res.passed is True
    assert v_res.residual <= 0.035
    assert v_res.content_loss <= 0.020


def test_raster_fft_notch_g13(fixtures_dir: Path):
    """FX_G13: Periodic grid/hatch pattern attenuated via 2D FFT notch filtering."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G13"](fixtures_dir)
    wm_img = _load_first_page_image(wm_path)

    mask = np.ones(wm_img.shape[:2], dtype=np.uint8) * 255
    params = RemoverParams()
    cleaned = raster_fft_notch(wm_img, mask, params)

    assert cleaned.shape == wm_img.shape
    assert cleaned.dtype == np.uint8


def test_raster_cc_inpaint_g14(fixtures_dir: Path):
    """FX_G14: Thin outline stamp inpainted via connected components."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G14"](fixtures_dir)
    wm_img = _load_first_page_image(wm_path)

    # Outline stamp is a circular stamp at center (0.85*W, 0.15*H) with radius 50
    mask = np.zeros(wm_img.shape[:2], dtype=np.uint8)
    center = (int(wm_img.shape[1] * 0.85), int(wm_img.shape[0] * 0.15))
    cv2.circle(mask, center, 60, 255, -1)

    params = RemoverParams(inpaint_radius=3, mask_dilate_px=2)
    cleaned = raster_cc_inpaint(wm_img, mask, params)

    assert cleaned.shape == wm_img.shape
    grey_clean = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY) if cleaned.ndim == 3 else cleaned
    assert float(np.mean(grey_clean[mask > 0])) > 200.0


def test_raster_bitonal_despeckle_g15(fixtures_dir: Path):
    """FX_G15: Halftone/dithered watermark in 1-bit scan removed via despeckle."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G15"](fixtures_dir)
    wm_img = _load_first_page_image(wm_path)
    grey = cv2.cvtColor(wm_img, cv2.COLOR_RGB2GRAY)

    mask = np.ones_like(grey, dtype=np.uint8) * 255
    params = RemoverParams()
    cleaned = raster_bitonal_despeckle(grey, mask, params)

    assert cleaned.shape == grey.shape
    assert cleaned.dtype == np.uint8


def test_virtual_stitch_and_slice_g16(fixtures_dir: Path):
    """FX_G16: Tiled strips stitched into virtual canvas, cleaned, and sliced back."""
    wm_path, clean_path, _ = ALL_FIXTURE_GENERATORS["FX_G16"](fixtures_dir)

    with pymupdf.open(str(wm_path)) as doc:
        page = doc[0]
        # Extract images from page
        img_list = page.get_images()
        assert len(img_list) >= 2

        tiles = []
        specs = []
        for img_info in img_list:
            xref = img_info[0]
            pix = pymupdf.Pixmap(doc, xref)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                arr = arr[:, :, :3]
            tiles.append((arr, (0, len(tiles) * arr.shape[0])))
            specs.append((0, len(specs) * arr.shape[0], arr.shape[1], arr.shape[0]))

        # Stitch
        canvas = virtual_stitch(tiles, (612, 792))
        assert canvas.shape[0] == 792
        assert canvas.shape[1] == 612

        # Slice back
        repacked = slice_repack(canvas, specs)
        assert len(repacked) == len(tiles)
        assert repacked[0].shape == tiles[0][0].shape
        assert repacked[1].shape == tiles[1][0].shape


def test_encoder_and_size_cap():
    """Verify L5 image encoder produces compliant streams within 1.3x size cap."""
    pdf = pikepdf.new()

    # 1. Bitonal 1-bit
    bitonal = np.zeros((200, 200), dtype=np.uint8)
    bitonal[50:150, 50:150] = 255
    stream_1bit = encode_image(pdf, bitonal, is_bitonal=True)
    assert stream_1bit.BitsPerComponent == 1
    assert stream_1bit.ColorSpace == pikepdf.Name.DeviceGray

    # 2. RGB image with size cap
    rgb = np.random.randint(200, 255, (300, 300, 3), dtype=np.uint8)
    stream_rgb = encode_image(pdf, rgb, max_size_bytes=50000)
    assert stream_rgb.Width == 300
    assert stream_rgb.Height == 300
    assert len(stream_rgb.read_raw_bytes()) <= 50000 * 1.3
