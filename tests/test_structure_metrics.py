"""Tests for cerr.contour.structure_metrics on geometry with known answers."""
import os

import numpy as np
import pytest
from shapely.geometry import box

from cerr import datasets
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.contour import structure_metrics as sm

phantom_dir = os.path.join(os.path.dirname(datasets.__file__), 'radiomics_phantom_dicom', 'pat_1')
SPACING = (1.2, 0.8, 3.0)          # mm per array axis: rows, cols, slices (anisotropic)


def _box(shape=(30, 30, 12), rows=(10, 20), cols=(10, 20), slcs=(3, 8)):
    m = np.zeros(shape, dtype=bool)
    m[rows[0]:rows[1], cols[0]:cols[1], slcs[0]:slcs[1]] = True
    return m


def test_overlap_and_distance_for_shifted_box():
    ref = _box()
    test = np.roll(ref, 2, axis=1)                        # 2 columns = 1.6 mm
    m = sm.compareMasks(ref, test, SPACING)
    assert m['Dice'] == pytest.approx(0.8)
    assert m['Jaccard'] == pytest.approx(8 / 12)
    assert m['Sensitivity'] == pytest.approx(0.8) and m['PPV'] == pytest.approx(0.8)
    # Shift along columns is measured with the column spacing, not the row spacing
    assert m['HD_mm'] == pytest.approx(1.6)
    assert m['HD95_mm'] == pytest.approx(1.6)
    assert m['RefVolume_cc'] == pytest.approx(500 * np.prod(SPACING) / 1000)
    assert m['VolumeRatio'] == pytest.approx(1.0)
    assert m['SurfaceDice_2mm'] == pytest.approx(1.0)
    assert m['SurfaceDice_1mm'] < 1.0

    m = sm.compareMasks(ref, np.roll(ref, 1, axis=0), SPACING)
    assert m['HD_mm'] == pytest.approx(1.2)


def test_identical_and_empty_masks():
    ref = _box()
    m = sm.compareMasks(ref, ref, SPACING)
    assert m['Dice'] == 1 and m['HD_mm'] == 0 and m['MSD_mm'] == 0
    m = sm.compareMasks(ref, np.zeros_like(ref), SPACING)
    assert m['Dice'] == 0 and np.isnan(m['HD_mm']) and m['Sensitivity'] == 0
    m = sm.compareMasks(np.zeros_like(ref), np.zeros_like(ref), SPACING)
    assert np.isnan(m['Dice'])
    with pytest.raises(ValueError):
        sm.compareMasks(ref, ref[:-1], SPACING)


def test_added_path_length_on_polygons():
    ref = box(0, 0, 4, 4)                                 # cm
    # Identical contours: nothing to redraw
    assert sm.addedPathLengthSlice(ref, ref, 0.1) == pytest.approx(0)
    # Test missing on this slice: the whole reference perimeter
    assert sm.addedPathLengthSlice(ref, None, 0.1) == pytest.approx(16)
    assert sm.addedPathLengthSlice(None, ref, 0.1) == pytest.approx(16)
    assert sm.addedPathLengthSlice(None, None, 0.1) == 0
    # Test shifted 0.5 cm in x with a 0.1 cm band: the left edge (4) lies outside
    # the grown test, the right edge inside the shrunk test except within 0.1 of
    # the corners (3.8), and 0.4 of the top and bottom edges lies outside (0.8).
    test = box(0.5, 0, 4.5, 4)
    assert sm.addedPathLengthSlice(ref, test, 0.1) == pytest.approx(8.6, abs=0.01)
    # Within tolerance: a 0.05 cm shift costs nothing
    assert sm.addedPathLengthSlice(ref, box(0.05, 0, 4.05, 4), 0.1) == pytest.approx(0, abs=1e-6)


def test_slice_polygon_keeps_holes():
    class Seg:
        def __init__(self, xy):
            self.points = np.array([[x, y, 0.0] for x, y in xy])

    class Ctr:
        def __init__(self, segs):
            self.segments = segs

    outer = Seg([(0, 0), (4, 0), (4, 4), (0, 4)])
    inner = Seg([(1, 1), (3, 1), (3, 3), (1, 3)])
    ring = sm.slicePolygon(Ctr([outer, inner]))
    assert ring.area == pytest.approx(16 - 4)
    assert ring.length == pytest.approx(16 + 8)           # both boundaries count
    assert sm.slicePolygon([]) is None
    assert sm.slicePolygon(Ctr([Seg([(0, 0), (1, 1)])])) is None


@pytest.fixture(scope="module")
def phantom():
    planC = pc.loadDcmDir(phantom_dir)
    gtv3M = rs.getStrMask(0, planC)
    planC = pc.importStructureMask(gtv3M, 0, 'GTV copy', planC)          # 1
    planC = pc.importStructureMask(np.roll(gtv3M, 5, axis=1), 0, 'GTV shifted', planC)   # 2
    missing3M = gtv3M.copy()
    slices = np.unique(np.where(gtv3M)[2])
    missing3M[:, :, slices[len(slices) // 2]] = False
    planC = pc.importStructureMask(missing3M, 0, 'GTV one slice missing', planC)   # 3
    return planC, slices[len(slices) // 2]


def test_added_path_length_on_structures(phantom):
    planC, missingSlc = phantom
    total, perSlice = sm.addedPathLength(1, 1, planC)
    assert total == 0 and perSlice.shape == (planC.scan[0].getScanSize()[2],)

    total, perSlice = sm.addedPathLength(1, 3, planC)
    refPerimeterMm = sm.slicePolygon(planC.structure[1].contour[missingSlc]).length * 10
    assert perSlice[missingSlc] == pytest.approx(refPerimeterMm)
    assert total == pytest.approx(refPerimeterMm, rel=0.05)    # the other slices agree

    # Swapping roles: the missing slice is now test-only... and then reference-less
    totalSwap, _ = sm.addedPathLength(3, 1, planC)
    assert totalSwap == pytest.approx(total)
    totalNoTestOnly, perNoTestOnly = sm.addedPathLength(3, 1, planC, includeTestOnlySlices=False)
    assert perNoTestOnly[missingSlc] == 0

    shifted, _ = sm.addedPathLength(1, 2, planC)
    loose, _ = sm.addedPathLength(1, 2, planC, toleranceMm=10.0)
    assert shifted > 0 and loose < shifted


def test_compare_structure_pair(phantom):
    planC, _ = phantom
    m = sm.compareStructurePair('GTV copy', 'GTV shifted', planC)
    dx = planC.scan[0].getScanSpacing()[0] * 10
    assert 0.8 < m['Dice'] < 1
    assert m['CentroidDistance_mm'] == pytest.approx(5 * dx, rel=1e-6)
    assert m['HD_mm'] == pytest.approx(5 * dx, rel=0.01)
    assert m['APL_mm'] > 0
    assert sm.getSpacingMm(0, planC) == pytest.approx(
        tuple(planC.scan[0].getScanSpacing()[[1, 0, 2]] * 10))


def test_structures_on_different_scans_are_rejected(phantom):
    planC, _ = phantom
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    planC2 = pc.importScanArray(planC.scan[0].getScanArray(), xV, yV, zV, 'CT', 0, planC)
    planC2 = pc.importStructureMask(rs.getStrMask(0, planC2), 1, 'on scan 1', planC2)
    with pytest.raises(ValueError, match='different scans'):
        sm.compareStructurePair(0, len(planC2.structure) - 1, planC2)
    del planC2.structure[-1]
    del planC2.scan[-1]
