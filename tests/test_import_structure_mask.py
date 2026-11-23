"""Tests that importStructureMask contours the mask's bounding box exactly as it would the whole slice.

Uses the radiomics phantom shipped with pyCERR and its GTV-1 structure, so the masks are real
contours on a real scan grid rather than boxes.
"""
import os

import numpy as np
import pytest

from cerr import datasets
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


@pytest.fixture(scope='module')
def planC():
    """Phantom planC; structure 0 is GTV-1. Imported structures are appended to it."""
    return pc.loadDcmDir(phantom_dir)


@pytest.fixture(scope='module')
def gtvMask(planC):
    """GTV-1 as a binary mask on the scan grid: rows/columns well inside the slice."""
    mask3M = rs.getStrMask(planC.structure[0], planC)
    assert mask3M.any(), "GTV-1 is empty"
    return mask3M


def _polygons(mask3M, planC, minPoints=8):
    """Contours of mask3M in (row, col, slc) units, keyed by slice.

    Segments shorter than minPoints are dropped, so the marker voxels that force the
    uncropped path do not enter the comparison.
    """
    planC = pc.importStructureMask(mask3M, 0, 'test', planC)
    polyList, _, _ = planC.structure[-1].getContourPolygons(planC, rcsFlag=True)
    bySlc = {}
    for poly in polyList:
        if poly.shape[0] >= minPoints:
            bySlc.setdefault(int(round(poly[0, 2])), []).append(np.round(poly[:, :2], 6))
    return bySlc


def _assertSamePolygons(got, expected):
    assert sorted(got) == sorted(expected), "contours on different slices"
    for slc in expected:
        gotV = sorted(got[slc], key=lambda p: (p.shape[0], p[0, 0], p[0, 1]))
        expV = sorted(expected[slc], key=lambda p: (p.shape[0], p[0, 0], p[0, 1]))
        assert len(gotV) == len(expV), f"slice {slc}: segment count"
        for g, e in zip(gotV, expV):
            assert np.array_equal(g, e), f"slice {slc}: contour points differ"


def test_contours_match_uncropped(planC, gtvMask):
    """Cropping to the mask's bounding box gives the contours of the whole slice.

    The reference run adds one voxel at each in-plane corner of every occupied slice, so the
    bounding box is the full slice and no cropping happens; the GTV's own contours must match.
    """
    numRows, numCols, _ = planC.scan[0].getScanSize()
    full = gtvMask.copy()
    occupied = gtvMask.any(axis=(0, 1))
    for r in (0, numRows - 1):
        for c in (0, numCols - 1):
            full[r, c, occupied] = True

    cropped = _polygons(gtvMask, planC)
    assert cropped, "no contours imported"
    _assertSamePolygons(cropped, _polygons(full, planC))


def test_contours_match_uncropped_at_the_grid_edge(planC, gtvMask):
    """Same, for a mask shifted onto the first row and column (padding at the array edge)."""
    rowsV = np.flatnonzero(gtvMask.any(axis=(1, 2)))
    colsV = np.flatnonzero(gtvMask.any(axis=(0, 2)))
    edge = np.roll(np.roll(gtvMask, -int(rowsV[0]), axis=0), -int(colsV[0]), axis=1)
    assert edge[0].any() or edge[:, 0].any(), "mask does not reach the grid edge"

    numRows, numCols, _ = planC.scan[0].getScanSize()
    full = edge.copy()
    occupied = edge.any(axis=(0, 1))
    for r in (0, numRows - 1):
        for c in (0, numCols - 1):
            full[r, c, occupied] = True

    _assertSamePolygons(_polygons(edge, planC), _polygons(full, planC))


def test_contours_shift_with_the_mask(planC, gtvMask):
    """A translated mask gives translated contours: the cropped region's offset is restored."""
    dRow, dCol = 7, 11
    moved = np.roll(np.roll(gtvMask, dRow, axis=0), dCol, axis=1)
    expected = {slc: [p + np.array([dRow, dCol]) for p in polys]
                for slc, polys in _polygons(gtvMask, planC).items()}
    _assertSamePolygons(_polygons(moved, planC), expected)


def test_mask_round_trips_through_contours(planC, gtvMask):
    """The imported contours rasterize back to the mask they came from."""
    planC = pc.importStructureMask(gtvMask, 0, 'test', planC)
    assert np.array_equal(rs.getStrMask(planC.structure[-1], planC), gtvMask)
