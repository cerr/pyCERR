"""Tests for contour rasterization: mask -> contour -> mask round-trip.

Uses the bundled DICOM radiomics phantom (no network required). A synthetic
axis-aligned box mask is imported as a structure (mask -> polygon contours via
``importStructureMask``), then rasterized back to a mask (polygon -> mask via
``getStrMask`` / ``generateRastersegs``) and compared against the input. This
exercises the core, otherwise-untested segmentation geometry path in
``cerr.dataclasses.structure`` and ``cerr.contour.rasterseg``.
"""
import os
import numpy as np

from cerr import datasets
from cerr import plan_container as pc
from cerr.dataclasses import structure as structr
from cerr.contour.rasterseg import getStrMask

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def _load_phantom():
    return pc.loadDcmDir(phantom_dir)


def _dice(a, b):
    """Dice overlap coefficient between two boolean masks."""
    a = a.astype(bool)
    b = b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return 2.0 * int(np.logical_and(a, b).sum()) / denom


def _make_box_mask(planC, scanNum=0):
    """A solid, axis-aligned box occupying the middle of the scan grid."""
    nRows, nCols, nSlc = planC.scan[scanNum].getScanSize()
    mask = np.zeros((nRows, nCols, nSlc), dtype=bool)
    # Middle-half in-plane, a few central slices: robust to any phantom size
    # and kept off the image edges.
    r0, r1 = nRows // 4, nRows - nRows // 4
    c0, c1 = nCols // 4, nCols - nCols // 4
    s0 = max(1, nSlc // 2 - 1)
    s1 = min(nSlc - 1, nSlc // 2 + 2)
    mask[r0:r1, c0:c1, s0:s1] = True
    return mask


def test_box_mask_roundtrip():
    """An axis-aligned filled box should survive mask -> contour -> mask."""
    planC = _load_phantom()
    mask = _make_box_mask(planC)
    nStructBefore = len(planC.structure)

    planC = structr.importStructureMask(mask, 0, 'test_box', planC)
    assert len(planC.structure) == nStructBefore + 1
    structNum = len(planC.structure) - 1

    rtMask = getStrMask(structNum, planC)
    assert rtMask.shape == mask.shape

    # Round-tripped mask is non-empty and lives on exactly the same slices.
    assert rtMask.any()
    np.testing.assert_array_equal(np.where(mask.any(axis=(0, 1)))[0],
                                  np.where(rtMask.any(axis=(0, 1)))[0])

    # An axis-aligned filled box must round-trip exactly (Dice == 1).
    assert _dice(mask, rtMask) == 1.0


def test_roundtrip_voxel_count():
    """Rasterized voxel count must exactly match the input box."""
    planC = _load_phantom()
    mask = _make_box_mask(planC)
    planC = structr.importStructureMask(mask, 0, 'test_box', planC)
    structNum = len(planC.structure) - 1
    rtMask = getStrMask(structNum, planC)
    nIn = int(mask.sum())
    nOut = int(rtMask.sum())
    assert nIn > 0
    assert nOut == nIn


def test_real_structure_mask_nonempty():
    """The phantom's bundled RTSTRUCT rasterizes to a non-empty, in-grid mask."""
    planC = _load_phantom()
    assert len(planC.structure) > 0
    mask = getStrMask(0, planC)
    nRows, nCols, nSlc = planC.scan[0].getScanSize()
    assert mask.shape == (nRows, nCols, nSlc)
    assert mask.any()
