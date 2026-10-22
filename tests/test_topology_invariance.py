"""Test topological Betti-number invariance and skeleton stitching."""

import cv2
import numpy as np

from tests.benchmarks.fixture_generator import draw_benzene_ring
from wmclean.verify.topology import (
    compute_betti_numbers,
    stitch_broken_skeletons,
    verify_topological_invariance,
)


def test_isolated_ring_topology():
    """A clean closed loop must have exactly 1 component and 1 closed loop."""
    img = np.full((100, 100), 255, dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, 0, thickness=2)
    binary = (img < 150).astype(np.uint8) * 255

    b0, b1 = compute_betti_numbers(binary)
    assert b0 == 1, f"Expected 1 connected component, got {b0}"
    assert b1 == 1, f"Expected exactly 1 closed loop, got {b1}"


def test_broken_ring_detected_and_stitched():
    """Breaking a loop drops beta_1 to 0; skeleton stitching restores beta_1 to 1."""
    img = np.full((100, 100), 255, dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, 0, thickness=2)
    binary = (img < 150).astype(np.uint8) * 255

    # Break circle with a 4-pixel eraser gap on the left edge (x=20, y=50)
    broken = binary.copy()
    broken[48:52, 18:22] = 0

    b0_broken, b1_broken = compute_betti_numbers(broken)
    assert b1_broken == 0, f"Broken loop should have 0 closed loops, got {b1_broken}"

    # Stitch skeleton
    wm_mask = np.zeros_like(binary)
    wm_mask[40:60, 10:30] = 255  # simulated watermark zone covering the gap
    restored = stitch_broken_skeletons(broken, wm_mask, max_gap_px=10)

    b0_restored, b1_restored = compute_betti_numbers(restored)
    assert b1_restored == 1, f"Stitching should restore the closed loop (beta_1=1), got {b1_restored}"


def test_benzene_molecule_topology():
    """Benzene molecule (ring + resonance circle + OH) has exact known topological invariants."""
    img = np.full((120, 120), 255, dtype=np.uint8)
    draw_benzene_ring(img, (60, 60), radius=30, line_width=2)
    binary = img < 150
    b0, b1 = compute_betti_numbers(binary)
    # 4 components (hexagon, resonance circle, O, H) and 2 closed loops (annulus hole, letter O)
    assert b0 == 4
    assert b1 == 2


def test_topological_invariance_verifier():
    """Verify that verify_topological_invariance passes on clean and fails on broken diagrams."""
    clean = np.full((160, 160, 3), 255, dtype=np.uint8)
    draw_benzene_ring(clean, (80, 80), radius=35, line_width=2)

    diag_mask = (cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) < 150)

    # Identical clean passes
    passed, db0, db1, msg = verify_topological_invariance(clean, clean.copy(), diagram_mask=diag_mask)
    assert passed is True
    assert db1 == 0

    # Broken ring fails
    corrupted = clean.copy()
    corrupted[78:82, 42:48] = (255, 255, 255)
    passed_broken, db0, db1, msg = verify_topological_invariance(clean, corrupted, diagram_mask=diag_mask)
    assert passed_broken is False
    assert db1 == -1, f"Expected loop loss (db1=-1), got {db1}"
