"""L3 Removers — registry of deterministic vector & raster watermark removers."""

from wmclean.removers.chamfer_align import chamfer_align_template, extract_chamfer_edges
from wmclean.removers.matting import closed_form_matte_unmix, guided_filter_matte
from wmclean.removers.piracy_ocr import (
    find_anomaly_candidates,
    piracy_ocr_spotter,
    swt_morph_reconstruct,
)
from wmclean.removers.raster_ops import (
    ecc_align_stack,
    illum_flatten,
    inverted_banner_strip,
    raster_alpha_inv,
    raster_bitonal_despeckle,
    raster_cc_inpaint,
    raster_color_unmix,
    raster_dark_stamp_consensus,
    raster_fft_notch,
    raster_hue_selective_unmix,
    raster_luma_sep,
    raster_margin_banner,
    raster_registered_outline_clean,
    raster_template_whiten,
)
from wmclean.removers.stain_deconv import (
    beer_lambert_stain_unmix,
    estimate_stain_vectors,
)
from wmclean.removers.stitch import slice_repack, virtual_stitch
from wmclean.removers.stream_ast import GraphicsState, StreamProcessor
from wmclean.removers.vector_ops import (
    annot_del,
    artifact_strip,
    ast_tj_reassemble,
    clip_scope_strip,
    ocg_strip,
    pattern_sh_strip,
    rev_rollback,
    smask_gs_strip,
    text_op_strip,
    vec_path_strip,
    vec_spatial_band_cluster,
    xobj_do_strip,
)

__all__ = [
    "chamfer_align_template",
    "extract_chamfer_edges",
    "closed_form_matte_unmix",
    "guided_filter_matte",
    "find_anomaly_candidates",
    "piracy_ocr_spotter",
    "swt_morph_reconstruct",
    "beer_lambert_stain_unmix",
    "estimate_stain_vectors",
    "virtual_stitch",
    "slice_repack",
    "ContentStreamAST",
    "StreamProcessor",
    "GraphicsState",
    "annot_del",
    "artifact_strip",
    "ast_tj_reassemble",
    "clip_scope_strip",
    "ocg_strip",
    "pattern_sh_strip",
    "rev_rollback",
    "smask_gs_strip",
    "text_op_strip",
    "vec_path_strip",
    "vec_spatial_band_cluster",
    "xobj_do_strip",
    "raster_template_whiten",
    "raster_alpha_inv",
    "raster_color_unmix",
    "raster_hue_selective_unmix",
    "raster_luma_sep",
    "raster_fft_notch",
    "raster_cc_inpaint",
    "raster_bitonal_despeckle",
    "raster_margin_banner",
    "inverted_banner_strip",
    "raster_dark_stamp_consensus",
    "raster_registered_outline_clean",
    "ecc_align_stack",
    "illum_flatten",
]
