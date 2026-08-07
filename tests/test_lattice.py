"""Tests for cerr.imrtp.lattice (lattice / SFRT target creation + boost wiring).

The geometry tests build a synthetic water phantom with a spherical target
and exercise vertex placement, sphere painting and the peak/valley
invariants -- no optional dependencies required.

The optimization-wiring test drives the lattice boost end-to-end through the
pyRadPlan bridge and is skipped when pyRadPlan is unavailable.
"""

import numpy as np
import pytest

import SimpleITK as sitk

from cerr import plan_container as pc
from cerr.imrtp import lattice as lat


# --------------------------------------------------------------------------
# Phantom construction
# --------------------------------------------------------------------------

VOXEL_MM = 2.0
SHAPE_ZYX = (60, 120, 120)          # slices, rows, cols
GTV_RADIUS_MM = 40.0


def _makePhantomPlanC(tmp_path):
    """Water-box phantom with a central spherical GTV, loaded via NIfTI."""
    nz, ny, nx = SHAPE_ZYX
    hu = np.full(SHAPE_ZYX, -1000.0, dtype=np.float32)   # air
    hu[4:-4, 8:-8, 8:-8] = 0.0                           # water box

    img = sitk.GetImageFromArray(hu)
    img.SetSpacing([VOXEL_MM] * 3)
    img.SetOrigin([0.0, 0.0, 0.0])
    niiFile = str(tmp_path / "phantom.nii.gz")
    sitk.WriteImage(img, niiFile)

    planC = pc.loadNiiScan(niiFile, imageType="CT SCAN")

    ctrZ, ctrY, ctrX = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    distMm = VOXEL_MM * np.sqrt((zz - ctrZ) ** 2 + (yy - ctrY) ** 2
                                + (xx - ctrX) ** 2)
    maskZYX = distMm <= GTV_RADIUS_MM

    scanImg = planC.scan[0].getSitkImage()
    maskImg = sitk.GetImageFromArray(maskZYX.astype(np.uint8))
    maskImg.CopyInformation(scanImg)
    # (z,y,x) DICOM slice order -> pyCERR (row, col, slice)
    mask3M = np.transpose(sitk.GetArrayFromImage(maskImg), (1, 2, 0)).astype(bool)
    planC = pc.importStructureMask(mask3M, 0, "GTV", planC)
    return planC


def _structIndex(planC, name):
    return [i for i, s in enumerate(planC.structure)
            if s.structureName == name][0]


def _mask(planC, name):
    import cerr.contour.rasterseg as rs
    return rs.getStrMask(_structIndex(planC, name), planC)


# --------------------------------------------------------------------------
# Geometry tests
# --------------------------------------------------------------------------

@pytest.fixture
def phantomPlanC(tmp_path):
    return _makePhantomPlanC(tmp_path)


def test_createLattice_addsPeakAndValley(phantomPlanC):
    planC = phantomPlanC
    planC, verts = lat.createLattice(
        planC, 0, sphereDiameter=12.0, latticeSpacing=20.0,
        latticeType='bcc', innerMargin=8.0, units='mm',
        addValley=True, addIndividualSpheres=False)

    assert len(verts) > 0
    names = [s.structureName for s in planC.structure]
    assert "GTV_lattice_peaks" in names
    assert "GTV_lattice_valley" in names


def test_createLattice_peakValleyInvariants(phantomPlanC):
    planC = phantomPlanC
    planC, verts = lat.createLattice(
        planC, 0, sphereDiameter=12.0, latticeSpacing=20.0,
        latticeType='bcc', innerMargin=8.0, units='mm', addValley=True)

    gtv = _mask(planC, "GTV")
    peaks = _mask(planC, "GTV_lattice_peaks")
    valley = _mask(planC, "GTV_lattice_valley")

    assert peaks.sum() > 0
    # peaks lie strictly inside the GTV
    assert np.all(peaks <= gtv)
    # peaks and valley partition the GTV exactly
    assert np.array_equal(peaks | valley, gtv)
    assert not np.any(peaks & valley)


def test_createLattice_individualSpheres(phantomPlanC):
    planC = phantomPlanC
    planC, verts = lat.createLattice(
        planC, 0, sphereDiameter=12.0, latticeSpacing=20.0,
        latticeType='sc', innerMargin=8.0, units='mm',
        addValley=False, addIndividualSpheres=True)

    vertexStructs = [s for s in planC.structure
                     if s.structureName.startswith("GTV_vertex_")]
    assert len(vertexStructs) == len(verts)


def test_createLattice_bccDenserThanSc(phantomPlanC):
    """BCC adds body-centred points, so it yields more vertices than SC."""
    import copy
    scPlanC = copy.deepcopy(phantomPlanC)
    _, scVerts = lat.createLattice(
        scPlanC, 0, sphereDiameter=12.0, latticeSpacing=20.0,
        latticeType='sc', innerMargin=8.0, units='mm', addValley=False)
    _, bccVerts = lat.createLattice(
        phantomPlanC, 0, sphereDiameter=12.0, latticeSpacing=20.0,
        latticeType='bcc', innerMargin=8.0, units='mm', addValley=False)
    assert len(bccVerts) > len(scVerts)


def test_createLattice_noVerticesFitRaises(phantomPlanC):
    # innerMargin larger than the GTV inradius leaves no room for any vertex.
    with pytest.raises(ValueError, match="No lattice vertices fit"):
        lat.createLattice(phantomPlanC, 0, sphereDiameter=15.0,
                          latticeSpacing=20.0, innerMargin=100.0, units='mm')


def test_createLattice_avoidanceRing(phantomPlanC):
    """The avoidance ring is a shell around the peaks: inside the GTV, disjoint
    from the peaks, and part of the valley."""
    planC = phantomPlanC
    planC, verts = lat.createLattice(
        planC, 0, sphereDiameter=12.0, latticeSpacing=20.0,
        latticeType='bcc', innerMargin=8.0, units='mm',
        addValley=True, addAvoidanceRing=True, ringThickness=3.0)

    ringNum = _structIndex(planC, "GTV_lattice_ring")
    gtv    = _mask(planC, "GTV")
    peaks  = _mask(planC, "GTV_lattice_peaks")
    valley = _mask(planC, "GTV_lattice_valley")
    ring   = _mask(planC, "GTV_lattice_ring")

    assert ring.sum() > 0
    assert np.all(ring <= gtv)            # ring stays inside the target
    assert not np.any(ring & peaks)       # ring excludes the peaks
    assert np.all(ring <= valley)         # ring is part of the valley
    assert ringNum >= 0


def test_createLatticeVertices_unitsAgnostic(phantomPlanC):
    """The low-level vertex generator works directly in grid (cm) units."""
    import cerr.contour.rasterseg as rs
    xV, yV, zV = phantomPlanC.scan[0].getScanXYZVals()
    gtv = rs.getStrMask(0, phantomPlanC)
    verts = lat.createLatticeVertices(
        gtv, xV, yV, zV, latticeSpacing=2.0, latticeType='bcc',
        innerMargin=0.8, requireSphereInside=True, sphereRadius=0.6)
    assert verts.shape[1] == 3
    assert len(verts) > 0


def test_invalidLatticeType(phantomPlanC):
    with pytest.raises(ValueError, match="latticeType"):
        lat.createLattice(phantomPlanC, 0, latticeType='fcc')


def test_invalidUnits(phantomPlanC):
    with pytest.raises(ValueError, match="units"):
        lat.createLattice(phantomPlanC, 0, units='inches')


# --------------------------------------------------------------------------
# Optimization-wiring tests
# --------------------------------------------------------------------------

def test_latticeBoostObjectives_structure():
    """Objective builder wires peaks as target, valley as overdosing cap."""
    prp = pytest.importorskip("cerr.imrtp.pyradplan_bridge")
    if not getattr(prp, "_PYRADPLAN_AVAILABLE", False):
        pytest.skip("pyRadPlan not installed")
    from pyRadPlan.optimization.objectives import (SquaredDeviation,
                                                   SquaredOverdosing,
                                                   SquaredUnderdosing)

    objectives, targets = lat.latticeBoostObjectives(
        peakStructNum=1, valleyStructNum=2, gtvStructNum=0,
        peakDose=15.0)

    assert targets == [1, 0]
    assert isinstance(objectives[1][0], SquaredDeviation)
    assert isinstance(objectives[2][0], SquaredOverdosing)
    assert isinstance(objectives[0][0], SquaredUnderdosing)
    # peak target driven to the prescription dose
    assert objectives[1][0].d_ref == pytest.approx(15.0)
    # default valley cap is 30% of the peak dose (pyRadPlan param: d_max)
    assert objectives[2][0].d_max == pytest.approx(0.3 * 15.0)
    # GTV minimum-dose coverage defaults to the valley cap (param: d_min)
    assert objectives[0][0].d_min == pytest.approx(0.3 * 15.0)


def test_latticeBoostObjectives_numericOarCap():
    """A plain number in oarObjectives becomes an overdose cap (no pyRadPlan
    Objective needed from the caller)."""
    prp = pytest.importorskip("cerr.imrtp.pyradplan_bridge")
    if not getattr(prp, "_PYRADPLAN_AVAILABLE", False):
        pytest.skip("pyRadPlan not installed")
    from pyRadPlan.optimization.objectives import SquaredOverdosing

    objectives, _ = lat.latticeBoostObjectives(
        peakStructNum=1, peakDose=15.0, oarObjectives={5: 2.0})
    assert isinstance(objectives[5][0], SquaredOverdosing)
    assert objectives[5][0].d_max == pytest.approx(2.0)


def test_optimizeLatticeBoost_endToEnd(tmp_path):
    """Full lattice boost through the pyRadPlan bridge; peaks outdose valley."""
    pytest.importorskip("pyRadPlan")
    from cerr.dataclasses import beams as bms

    planC = _makePhantomPlanC(tmp_path)
    planC, verts = lat.createLattice(
        planC, 0, sphereDiameter=12.0, latticeSpacing=20.0,
        latticeType='bcc', innerMargin=8.0, units='mm', addValley=True)
    peakNum = _structIndex(planC, "GTV_lattice_peaks")
    valleyNum = _structIndex(planC, "GTV_lattice_valley")

    # isocenter at the GTV centroid (DICOM mm)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(planC.structure[0].getSitkImage(planC),
                            sitk.sitkUInt8))
    isoMm = np.array(stats.GetCentroid(1))

    w, doseNum, planC = lat.optimizeLatticeBoost(
        planC, peakNum, valleyStructNum=valleyNum, gtvStructNum=0,
        scanNum=0, gantryAngles=(0.0, 90.0), isoCenter=isoMm,
        peakDose=15.0, bixelWidth=8.0,
        doseGridResolution={"x": 4.0, "y": 4.0, "z": 4.0})

    assert doseNum == len(planC.dose) - 1
    assert np.all(np.asarray(w) >= 0)

    # Mean dose in the peaks should exceed the valley (peak-to-valley effect).
    import cerr.contour.rasterseg as rs
    dose3M = planC.dose[doseNum].doseArray
    peaks = rs.getStrMask(peakNum, planC)
    valley = rs.getStrMask(valleyNum, planC)
    assert dose3M[peaks].mean() > dose3M[valley].mean()


def test_optimizeLatticeBoost_invalidEngine(phantomPlanC):
    with pytest.raises(ValueError, match="engine"):
        lat.optimizeLatticeBoost(phantomPlanC, 0, engine='montecarlo')


def _addBodyStruct(planC):
    """Add a BODY structure = the full water box, for the QIB/PortPy dataset."""
    import cerr.contour.rasterseg as rs
    gtv = rs.getStrMask(0, planC)
    body = np.zeros_like(gtv)
    body[8:-8, 8:-8, 4:-4] = True     # matches the water box in _makePhantomPlanC
    body = body | gtv
    pc.importStructureMask(body, 0, "BODY", planC)
    return _structIndex(planC, "BODY")


def test_optimizeLatticeBoost_qibEngine(tmp_path):
    """QIB engine (native pyCERR PB + PortPy); peaks outdose valley."""
    pytest.importorskip("portpy")

    planC = _makePhantomPlanC(tmp_path)
    bodyNum = _addBodyStruct(planC)
    planC, verts = lat.createLattice(
        planC, 0, sphereDiameter=12.0, latticeSpacing=20.0,
        latticeType='bcc', innerMargin=8.0, units='mm', addValley=True,
        addAvoidanceRing=True, ringThickness=3.0)
    peakNum = _structIndex(planC, "GTV_lattice_peaks")
    valleyNum = _structIndex(planC, "GTV_lattice_valley")
    ringNum = _structIndex(planC, "GTV_lattice_ring")

    # ring cap given as a plain number -> no pyRadPlan Objective needed
    w, doseNum, planC = lat.optimizeLatticeBoost(
        planC, peakNum, valleyStructNum=valleyNum, bodyStructNum=bodyNum,
        scanNum=0, gantryAngles=(0.0, 90.0, 180.0, 270.0),
        peakDose=15.0, valleyMaxDose=4.5, oarObjectives={ringNum: 2.0},
        engine='qib', solver='SCS')

    assert doseNum == len(planC.dose) - 1
    import cerr.contour.rasterseg as rs
    dose3M = planC.dose[doseNum].doseArray
    peaks = rs.getStrMask(peakNum, planC)
    valley = rs.getStrMask(valleyNum, planC)
    assert dose3M[peaks].mean() > dose3M[valley].mean()
