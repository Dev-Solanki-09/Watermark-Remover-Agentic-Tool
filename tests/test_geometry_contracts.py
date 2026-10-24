from wmclean.deps import get_deskew, get_page_dewarp, has_deskew, has_page_dewarp
from wmclean.schemas import GeometricModality, GeometryProbeRecord


def test_geometric_modality_enum():
    assert GeometricModality.GEOM_FLAT.value == "GEOM_FLAT"
    assert GeometricModality.GEOM_SKEW_RIGID.value == "GEOM_SKEW_RIGID"
    assert GeometricModality.GEOM_GUTTER_CURVE.value == "GEOM_GUTTER_CURVE"
    assert GeometricModality.GEOM_WAVY_BUMP.value == "GEOM_WAVY_BUMP"


def test_geometry_probe_record_instantiation():
    rec = GeometryProbeRecord(
        modality=GeometricModality.GEOM_FLAT,
        skew_angle_deg=0.05,
        spine_dip_px=0.8,
        waviness_std_px=0.4,
    )
    assert rec.modality == GeometricModality.GEOM_FLAT
    assert rec.is_flat is True


def test_dewarp_deps():
    assert has_deskew() is True
    assert get_deskew() is not None
    assert has_page_dewarp() is True
    assert get_page_dewarp() is not None
