"""Compact Pydantic models for Probe, Signature, Router, Verify, and Report schemas.

Every layer communicates through these typed contracts. Enums are closed sets —
the router can only select from RemoverMethod, and failures map deterministically
to FailureCode.
"""

from __future__ import annotations

import enum
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PageKind(str, enum.Enum):
    """Axis-1 page base classification."""

    VECTOR = "VECTOR"
    SCAN = "SCAN"
    SCAN_OCR = "SCAN_OCR"
    HYBRID = "HYBRID"
    BITONAL = "BITONAL"


class PageState(str, enum.Enum):
    """Final per-page outcome after L4 verification."""

    CLEAN_EXACT = "CLEAN_EXACT"
    CLEAN_VERIFIED = "CLEAN_VERIFIED"
    UNTOUCHED = "UNTOUCHED"
    FLAGGED = "FLAGGED"


class CarrierFlag(str, enum.Enum):
    """Watermark carrier type detected during L0 probe."""

    ANNOT_WM = "ANNOT_WM"
    ANNOT_STAMP = "ANNOT_STAMP"
    ANNOT_FREETEXT = "ANNOT_FREETEXT"
    OCG = "OCG"
    ARTIFACT_WM = "ARTIFACT_WM"
    TEXT_VECTOR = "TEXT_VECTOR"
    CLIP_PATH = "CLIP_PATH"
    XOBJ_IMAGE = "XOBJ_IMAGE"
    XOBJ_FORM = "XOBJ_FORM"
    INLINE_IMAGE = "INLINE_IMAGE"
    VEC_PATH = "VEC_PATH"
    PATTERN = "PATTERN"
    SHADING = "SHADING"
    SMASK = "SMASK"
    BLEND_MODE = "BLEND_MODE"
    RASTER_BAKED = "RASTER_BAKED"
    RASTER_BITONAL = "RASTER_BITONAL"
    REVISION_APPENDABLE = "REVISION_APPENDABLE"


class RemoverMethod(str, enum.Enum):
    """Closed enum of L3 remover rungs. Router may only select from this set."""

    ANNOT_DEL = "annot_del"
    OCG_STRIP = "ocg_strip"
    ARTIFACT_STRIP = "artifact_strip"
    TEXT_OP_STRIP = "text_op_strip"
    CLIP_SCOPE_STRIP = "clip_scope_strip"
    XOBJ_DO_STRIP = "xobj_do_strip"
    VEC_PATH_STRIP = "vec_path_strip"
    PATTERN_SH_STRIP = "pattern_sh_strip"
    SMASK_GS_STRIP = "smask_gs_strip"
    REV_ROLLBACK = "rev_rollback"
    AST_TJ_REASSEMBLE = "ast_tj_reassemble"
    VEC_SPATIAL_BAND_CLUSTER = "vec_spatial_band_cluster"
    RASTER_TEMPLATE_WHITEN = "raster_template_whiten"
    RASTER_ALPHA_INV = "raster_alpha_inv"
    RASTER_COLOR_UNMIX = "raster_color_unmix"
    RASTER_HUE_SELECTIVE_UNMIX = "raster_hue_selective_unmix"
    RASTER_LUMA_SEP = "raster_luma_sep"
    RASTER_FFT_NOTCH = "raster_fft_notch"
    RASTER_CC_INPAINT = "raster_cc_inpaint"
    RASTER_BITONAL_DESPECKLE = "raster_bitonal_despeckle"
    RASTER_MARGIN_BANNER = "raster_margin_banner"
    RASTER_REGISTERED_OUTLINE_CLEAN = "raster_registered_outline_clean"
    RASTER_DARK_STAMP_CONSENSUS = "raster_dark_stamp_consensus"
    PIRACY_OCR_SPOTTER = "piracy_ocr_spotter"
    ILLUM_FLATTEN = "illum_flatten"
    INVERTED_BANNER_STRIP = "inverted_banner_strip"
    SWT_MORPH_RECONSTRUCT = "swt_morph_reconstruct"
    BEER_LAMBERT_STAIN_UNMIX = "beer_lambert_stain_unmix"
    RASTER_ML_INPAINT = "raster_ml_inpaint"


class FailureCode(str, enum.Enum):
    """Deterministic failure codes — every error maps to exactly one of these."""

    ERR_OPAQUE_OCCLUSION = "ERR_OPAQUE_OCCLUSION"
    ERR_HIGH_RESIDUAL = "ERR_HIGH_RESIDUAL"
    ERR_VEC_VERIFY_BREACH = "ERR_VEC_VERIFY_BREACH"
    ERR_DIAGRAM_INTEGRITY = "ERR_DIAGRAM_INTEGRITY"
    ERR_SIZE_CAP_EXCEEDED = "ERR_SIZE_CAP_EXCEEDED"
    ERR_ENCRYPTED = "ERR_ENCRYPTED"
    ERR_UNSUPPORTED_XFA = "ERR_UNSUPPORTED_XFA"
    ERR_CORRUPT_PDF = "ERR_CORRUPT_PDF"
    ERR_IO_RESOURCE = "ERR_IO_RESOURCE"


class ExitCode(int, enum.Enum):
    """Process exit codes."""

    SUCCESS = 0
    PARTIAL = 10  # ≥1 page FLAGGED
    UNSUPPORTED = 20  # encrypted / XFA
    CORRUPT = 30
    IO_ERROR = 40
    INTERNAL = 50


class RouterSource(str, enum.Enum):
    """How the router decision was made."""

    RULE = "RULE"
    LLM = "LLM"
    SIGNATURE_DB = "SIGNATURE_DB"
    LLM_FALLBACK_TO_RULE = "LLM_FALLBACK_TO_RULE"


# ---------------------------------------------------------------------------
# L0 Probe sub-models
# ---------------------------------------------------------------------------


class PageGeometry(BaseModel):
    """Page box coordinates and rotation."""

    mb: list[float] = Field(description="MediaBox [x0, y0, x1, y1]")
    cb: list[float] = Field(description="CropBox [x0, y0, x1, y1]")
    rot: int = Field(default=0, description="Page /Rotate value (0, 90, 180, 270)")


class AnnotCandidate(BaseModel):
    """Watermark annotation candidate detected in L0."""

    xref: int
    sub: str = Field(description="Annotation /Subtype string, e.g. '/Watermark'")
    bb: list[float] = Field(description="Bounding box [x0, y0, x1, y1]")


class OCGCandidate(BaseModel):
    """Optional Content Group candidate."""

    xref: int
    name: str
    intent: str = Field(default="/View")


class ArtifactCandidate(BaseModel):
    """Marked content /Artifact span candidate."""

    mcid: Optional[int] = None
    sub: str = Field(description="Artifact /Subtype, e.g. '/Watermark'")
    op_span: list[int] = Field(description="[start_op_index, end_op_index] in content stream")


class ExtGStateInfo(BaseModel):
    """ExtGState transparency/blend information."""

    name: str = Field(description="Resource name, e.g. '/GS1'")
    ca: Optional[float] = Field(default=None, description="Non-stroking alpha")
    CA: Optional[float] = Field(default=None, description="Stroking alpha")  # noqa: N815
    bm: Optional[str] = Field(default=None, description="Blend mode, e.g. '/Multiply'")
    smask: bool = Field(default=False, description="Whether /SMask is present")


class TextCandidate(BaseModel):
    """Text watermark candidate with normalized template."""

    sig: str = Field(description="Short hash signature, e.g. 't_8f3a1c'")
    norm_str: str = Field(description="Normalized text with <EMAIL>, <D>, <DATE> placeholders")
    rot_deg: float = Field(default=0.0, description="Rotation in degrees")
    bb: list[float] = Field(description="Bounding box [x0, y0, x1, y1]")
    font: str = Field(description="Font resource name, e.g. '/F2'")
    size: float = Field(description="Font size in points")
    color: list[float] = Field(description="Fill color components")
    tr: int = Field(default=0, description="Text rendering mode (0-7)")
    ca: Optional[float] = Field(default=None, description="Effective alpha")
    doc_freq: float = Field(
        default=0.0,
        description="Fraction of document pages containing this text pattern",
    )


class XObjCandidate(BaseModel):
    """Image/Form XObject watermark candidate."""

    name: str = Field(description="XObject resource name, e.g. '/Im3'")
    xref: int
    kind: str = Field(description="'IMAGE' or 'FORM'")
    bb: list[float] = Field(description="Bounding box [x0, y0, x1, y1]")
    doc_freq: float = Field(default=0.0)
    is_strip: bool = Field(default=False, description="Whether this is a tiled/strip image")


class PathCandidate(BaseModel):
    """Untagged vector path watermark candidate."""

    hash: str = Field(description="Topology hash, e.g. 'p_44d19e'")
    ops: int = Field(description="Number of path operators")
    bb: list[float] = Field(description="Bounding box [x0, y0, x1, y1]")
    ca: Optional[float] = Field(default=None, description="Effective alpha")
    is_clip: bool = Field(default=False)
    doc_freq: float = Field(default=0.0)


class RasterStats(BaseModel):
    """Raster-layer statistics computed at 100 DPI."""

    has_raster: bool = False
    is_bitonal: bool = False
    is_tiled_strips: bool = False
    strip_count: int = 1
    ocr_spans: int = 0
    tmpl_cov: float = Field(default=0.0, description="Template coverage fraction of page")
    tmpl_ncc: float = Field(default=0.0, description="NCC of template vs page region")
    pos_stab: float = Field(default=0.0, description="Position stability across pages (0-1)")
    sat_p95: float = Field(default=0.0, description="95th percentile saturation")
    est_alpha: float = Field(default=0.0, description="Estimated watermark alpha")
    est_wm_rgb: list[int] = Field(
        default_factory=lambda: [200, 200, 200],
        description="Estimated watermark RGB colour",
    )


class ProbeRecord(BaseModel):
    """L0 per-page probe output. Stored in SQLite spill table (~1.2 KB/page)."""

    p: int = Field(description="1-indexed page number")
    kind: PageKind
    geom: PageGeometry
    rev_clean: Optional[int] = Field(
        default=None,
        description="Revision offset of a clean pre-watermark revision, or None",
    )
    annots: list[AnnotCandidate] = Field(default_factory=list)
    ocgs: list[OCGCandidate] = Field(default_factory=list)
    artifacts: list[ArtifactCandidate] = Field(default_factory=list)
    gs_alpha: list[ExtGStateInfo] = Field(default_factory=list)
    text_cand: list[TextCandidate] = Field(default_factory=list)
    xobj_cand: list[XObjCandidate] = Field(default_factory=list)
    path_cand: list[PathCandidate] = Field(default_factory=list)
    ras_stats: RasterStats = Field(default_factory=RasterStats)


# ---------------------------------------------------------------------------
# L1 Cluster Signature
# ---------------------------------------------------------------------------


class RasterBucket(BaseModel):
    """Raster characteristic bucket for clustering."""

    bitonal: bool = False
    periodic_fft: bool = False
    chroma_sep: bool = False
    tmpl_stable: bool = False


class ClusterSignature(BaseModel):
    """L1 cluster signature — pages with identical sig_hash share a removal chain."""

    sig_hash: str = Field(description="SHA-256 hex digest of canonicalized features")
    page_kind: PageKind
    carrier_flags: list[CarrierFlag] = Field(default_factory=list)
    quant_geom_aspect: float = Field(description="Quantized CropBox aspect ratio")
    norm_text_pattern: Optional[str] = Field(
        default=None,
        description="Normalized watermark text pattern, e.g. 'DOWNLOADED BY <EMAIL> ON <DATE>'",
    )
    rot_bucket: int = Field(default=0, description="Quantized rotation (0, 45, 90, ...)")
    color_bucket: str = Field(default="UNKNOWN")
    raster_bucket: RasterBucket = Field(default_factory=RasterBucket)
    proven_chain: Optional[list[RemoverMethod]] = Field(default=None)
    proven_params: Optional[dict] = Field(default=None)
    historical_pass_rate: Optional[float] = Field(default=None)


# ---------------------------------------------------------------------------
# L2 Router Decision
# ---------------------------------------------------------------------------


class RemoverParams(BaseModel):
    """Bounded numeric parameters for L3 removers."""

    target_xrefs: list[int] = Field(default_factory=list)
    target_text_sigs: list[str] = Field(default_factory=list)
    target_texts: list[str] = Field(default_factory=list)
    target_path_hashes: list[str] = Field(default_factory=list)
    target_xobj_names: list[str] = Field(default_factory=list)
    target_ocg_xrefs: list[int] = Field(default_factory=list)
    target_gs_names: list[str] = Field(default_factory=list)
    alpha_clamp: float = Field(default=0.85, ge=0.1, le=0.92)
    luma_thresh: int = Field(default=165, ge=100, le=250)
    mask_dilate_px: int = Field(default=2, ge=0, le=5)
    inpaint_radius: int = Field(default=3, ge=1, le=7)


class ClusterAction(BaseModel):
    """Router decision for a single cluster."""

    sig_hash: str
    action: str = Field(description="'REMOVE' or 'SKIP'")
    chain: list[RemoverMethod] = Field(default_factory=list)
    params: RemoverParams = Field(default_factory=RemoverParams)


class RouterDecision(BaseModel):
    """L2 router output — one decision per cluster, validated by Pydantic."""

    clusters: list[ClusterAction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# L4 Verify Result
# ---------------------------------------------------------------------------


class VerifyResult(BaseModel):
    """L4 verification output for a single page."""

    passed: bool
    residual: float = Field(default=0.0, description="Watermark residual R_wm")
    content_loss: float = Field(default=0.0, description="Foreground content loss L_fg")
    outside_mask_diff: float = Field(default=0.0, description="Max pixel diff outside mask")
    text_diff: int = Field(default=0, description="Edit distance of non-watermark text tuples")
    reason: Optional[str] = Field(default=None, description="Failure reason if not passed")
    failure_code: Optional[FailureCode] = Field(default=None)


# ---------------------------------------------------------------------------
# Report (r.json) Schema
# ---------------------------------------------------------------------------


class PageReport(BaseModel):
    """Per-page entry in r.json."""

    p: int
    st: PageState
    cl: Optional[str] = Field(default=None, description="Cluster ID")
    m: list[str] = Field(default_factory=list, description="Methods applied")
    err: Optional[str] = Field(default=None, description="FailureCode if FLAGGED")
    res: float = Field(default=0.0, description="Watermark residual")
    loss: float = Field(default=0.0, description="Content loss")
    ms: int = Field(default=0, description="Processing time in milliseconds")


class ClusterReport(BaseModel):
    """Per-cluster summary in r.json."""

    id: str
    sig: str
    pages: str = Field(description="Page range string, e.g. '1-41,43-120'")
    chain: list[str]
    src: RouterSource


class PerfReport(BaseModel):
    """Performance metrics in r.json."""

    ms: int = Field(description="Total processing time in milliseconds")
    peak_mb: int = Field(description="Peak RSS in MB")
    llm_calls: int = Field(default=0)
    llm_tok: int = Field(default=0, description="Total LLM tokens consumed")
    size_ratio: float = Field(default=1.0, description="output_bytes / input_bytes")


class CountsReport(BaseModel):
    """Page state counts."""

    total: int = 0
    exact: int = 0
    verified: int = 0
    untouched: int = 0
    flagged: int = 0


class DocumentReport(BaseModel):
    """Top-level r.json schema (version 1.0)."""

    v: str = "1.0"
    input: str = Field(alias="in", serialization_alias="in")
    out: str
    sha256_in: str = ""
    sha256_out: str = ""
    exit: int = 0
    counts: CountsReport = Field(default_factory=CountsReport)
    perf: PerfReport = Field(default_factory=lambda: PerfReport(ms=0, peak_mb=0))
    clusters: list[ClusterReport] = Field(default_factory=list)
    pages: list[PageReport] = Field(default_factory=list)
    error_detail: Optional[str] = Field(default=None, description="Deterministic exception detail on failure")

    model_config = {"populate_by_name": True}
