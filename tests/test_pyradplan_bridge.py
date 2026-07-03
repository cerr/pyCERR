"""Phantom-based test for cerr.imrtp.pyradplan_bridge.

Builds a synthetic water phantom (HU=0 box in air) with a spherical
target, fabricates a two-beam RTPLAN-like Beams object on planC, runs
pyRadPlan beamlet dose calculation + fluence optimization through the
bridge, and checks that the resulting dose lands on the target.

Requires the optional dependency pyRadPlan; skipped when unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("pyRadPlan")

import SimpleITK as sitk  # noqa: E402

from cerr import plan_container as pc  # noqa: E402
from cerr.dataclasses import beams as bms  # noqa: E402
from cerr.imrtp import pyradplan_bridge as prp  # noqa: E402


# --------------------------------------------------------------------------
# Phantom construction
# --------------------------------------------------------------------------

VOXEL_MM = 4.0
SHAPE_ZYX = (40, 64, 64)          # slices, rows, cols
SPHERE_RADIUS_MM = 15.0


def _makePhantomPlanC(tmp_path, flipOrientation=False):
    """Water-box phantom with a central spherical target, loaded via NIfTI.

    When ``flipOrientation`` is True the CT is written with a flipped
    (``imageOrientationPatient = [-1,0,0,0,-1,0]``) axial geometry -- the
    non-identity direction that broke pyRadPlan's ray tracer on clinical
    data and that :func:`ctFromScan` now reorients away.
    """
    nz, ny, nx = SHAPE_ZYX
    hu = np.full(SHAPE_ZYX, -1000.0, dtype=np.float32)   # air
    hu[4:-4, 8:-8, 8:-8] = 0.0                           # water box

    img = sitk.GetImageFromArray(hu)
    img.SetSpacing([VOXEL_MM] * 3)
    if flipOrientation:
        img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
        img.SetOrigin([(nx - 1) * VOXEL_MM, (ny - 1) * VOXEL_MM, 0.0])
    else:
        img.SetOrigin([0.0, 0.0, 0.0])
    niiFile = str(tmp_path / "phantom.nii.gz")
    sitk.WriteImage(img, niiFile)

    planC = pc.loadNiiScan(niiFile, imageType="CT SCAN")

    # spherical target mask at the phantom center, in pyCERR (r, c, s) order
    ctrZ, ctrY, ctrX = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    distMm = VOXEL_MM * np.sqrt((zz - ctrZ) ** 2 + (yy - ctrY) ** 2
                                + (xx - ctrX) ** 2)
    maskZYX = distMm <= SPHERE_RADIUS_MM

    scanImg = planC.scan[0].getSitkImage()
    maskImg = sitk.GetImageFromArray(maskZYX.astype(np.uint8))
    maskImg.CopyInformation(scanImg)
    mask3M = prp.doseArrayFromSitk(maskImg, planC, 0).astype(bool)
    planC = pc.importStructureMask(mask3M, 0, "PTV", planC)
    return planC


def _targetCentroidMm(planC):
    """Physical (DICOM mm) centroid of structure 0."""
    maskImg = planC.structure[0].getSitkImage(planC)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(maskImg, sitk.sitkUInt8))
    return np.array(stats.GetCentroid(1))


def _addBeams(planC, isoMm, gantryAngles=(0.0, 90.0)):
    """Fabricate a minimal RTPLAN-like Beams entry on planC."""
    beamSeqs = []
    for i, g in enumerate(gantryAngles):
        cp = bms.ControlPointSequence()
        cp.ControlPointIndex = 0
        cp.GantryAngle = float(g)
        cp.PatientSupportAngle = 0.0
        cp.NominalBeamEnergy = 6.0
        cp.IsocenterPosition = np.asarray(isoMm, dtype=np.float64)

        bs = bms.BeamSeq()
        bs.BeamNumber = i + 1
        bs.BeamName = "B%d" % (i + 1)
        bs.BeamType = "STATIC"
        bs.RadiationType = "PHOTON"
        bs.TreatmentDeliveryType = "TREATMENT"
        bs.SourceAxisDistance = 1000.0
        bs.NumberOfControlPoints = 1
        bs.ControlPointSequence = np.array([cp])
        beamSeqs.append(bs)

    beams = bms.Beams()
    beams.RTPlanLabel = "phantomPlan"
    beams.BeamSequence = np.array(beamSeqs)
    beams.BeamUID = "RP.test.pyradplan"
    planC.beams.append(beams)
    return planC


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def phantomPipeline(tmp_path_factory):
    """Run the full bridge pipeline once and share results across tests."""
    tmp_path = tmp_path_factory.mktemp("prp_phantom")
    planC = _makePhantomPlanC(tmp_path)
    isoMm = _targetCentroidMm(planC)
    planC = _addBeams(planC, isoMm)

    ct, cst, pln = prp.planFromPlanC(
        planC, scanNum=0, beamsNum=0,
        objectives={0: [prp.squaredDeviation(2.0)]},
        prescribedDose=2.0,
        bixelWidth=8.0,
        doseGridResolution={"x": 4.0, "y": 4.0, "z": 4.0})
    stf, dij = prp.calcDoseInfluence(ct, cst, pln)
    return {"planC": planC, "isoMm": isoMm,
            "ct": ct, "cst": cst, "pln": pln, "stf": stf, "dij": dij}


def test_beamGeometryFromBeams(phantomPipeline):
    geom = prp.beamGeometryFromBeams(phantomPipeline["planC"], 0)
    assert geom["gantry_angles"] == [0.0, 90.0]
    assert geom["couch_angles"] == [0.0, 0.0]
    assert geom["radiation_mode"] == "photons"
    assert geom["iso_degenerate"] is False
    np.testing.assert_allclose(geom["iso_center"][0],
                               phantomPipeline["isoMm"], atol=1e-6)


def test_degenerateIsocenterFallback(tmp_path):
    """A [0,0,0] RTPLAN isocenter should warn and fall back to the target."""
    planC = _makePhantomPlanC(tmp_path)
    ptvCtr = _targetCentroidMm(planC)
    # beams with the degenerate placeholder isocenter
    planC = _addBeams(planC, np.zeros(3), gantryAngles=(0.0,))

    geom = prp.beamGeometryFromBeams(planC, 0)
    assert geom["iso_degenerate"] is True

    with pytest.warns(UserWarning, match="placeholder"):
        ct, cst, pln = prp.planFromPlanC(
            planC, scanNum=0, beamsNum=0,
            objectives={0: [prp.squaredDeviation(2.0)]},
            structNums=[0], targetStructNums=[0],
            bixelWidth=8.0, prescribedDose=2.0,
            doseGridResolution={"x": 4.0, "y": 4.0, "z": 4.0})

    iso = np.asarray(pln.prop_stf["iso_center"])
    assert not np.all(iso == 0.0)
    np.testing.assert_allclose(iso[0], ptvCtr, atol=1e-6)


def test_explicitIsoCenterOverridesFallback(tmp_path):
    """An explicit isoCenter takes precedence over the degenerate RTPLAN."""
    planC = _makePhantomPlanC(tmp_path)
    planC = _addBeams(planC, np.zeros(3), gantryAngles=(0.0,))
    explicit = np.array([10.0, 20.0, 30.0])
    ct, cst, pln = prp.planFromPlanC(
        planC, scanNum=0, beamsNum=0,
        objectives={0: [prp.squaredDeviation(2.0)]},
        structNums=[0], targetStructNums=[0], isoCenter=explicit,
        bixelWidth=8.0, doseGridResolution={"x": 4.0, "y": 4.0, "z": 4.0})
    np.testing.assert_allclose(np.asarray(pln.prop_stf["iso_center"])[0],
                               explicit, atol=1e-6)


def test_dijBeamletMatrix(phantomPipeline):
    dij = phantomPipeline["dij"]
    nBixels = int(np.sum([b.num_of_bixels_per_ray.sum()
                          for b in phantomPipeline["stf"].beams])) \
        if hasattr(phantomPipeline["stf"], "beams") else None
    doseMat = dij.physical_dose.flat[0] \
        if isinstance(dij.physical_dose, np.ndarray) else dij.physical_dose
    assert doseMat.shape[1] > 1            # multiple beamlets
    assert doseMat.nnz > 0                 # non-empty influence
    if nBixels:
        assert doseMat.shape[1] == nBixels


def test_unitWeightDoseLandsOnTarget(phantomPipeline):
    planC = phantomPipeline["planC"]
    dij = phantomPipeline["dij"]
    doseMat = dij.physical_dose.flat[0] \
        if isinstance(dij.physical_dose, np.ndarray) else dij.physical_dose
    w = np.ones(doseMat.shape[1])
    result = dij.compute_result_ct_grid(w)
    doseNum = prp.importDoseToPlanC(planC, result["physical_dose"],
                                    scanNum=0, fractionGroupID="unitWt")

    dose3M = planC.dose[doseNum].doseArray
    assert dose3M.shape == tuple(planC.scan[0].getScanSize())
    assert dose3M.max() > 0

    # dose inside the target must be much higher than the phantom periphery
    from cerr.contour import rasterseg as rs
    mask3M = rs.getStrMask(planC.structure[0], planC)
    targetMean = dose3M[mask3M].mean()
    assert targetMean > 0.5 * dose3M.max()

    # high-dose centroid should sit near the isocenter (both beams cross there)
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    hot = dose3M >= 0.7 * dose3M.max()
    rr, cc, ss = np.nonzero(hot)
    hotCtrCerr = np.array([xV[cc].mean(), yV[rr].mean(), zV[ss].mean(), 1.0])
    hotCtrMm = (planC.scan[0].cerrToDcmTransM @ hotCtrCerr)[:3]
    assert np.linalg.norm(hotCtrMm - phantomPipeline["isoMm"]) < 15.0


def test_optimizedDoseMatchesPrescription(phantomPipeline):
    planC = phantomPipeline["planC"]
    w, doseNum, planC = prp.optimizeAndImportDose(
        planC, phantomPipeline["ct"], phantomPipeline["cst"],
        phantomPipeline["stf"], phantomPipeline["dij"],
        phantomPipeline["pln"], scanNum=0, fractionGroupID="optimized")

    assert np.all(w >= 0) and w.max() > 0
    dose3M = planC.dose[doseNum].doseArray

    from cerr.contour import rasterseg as rs
    from scipy import ndimage
    mask3M = rs.getStrMask(planC.structure[0], planC)
    # evaluate the target core (eroded mask) to avoid partial-volume
    # boundary voxels of the coarser dose grid
    coreMask = ndimage.binary_erosion(mask3M, iterations=2)
    coreMean = dose3M[coreMask].mean()
    # optimizer should deliver the 2 Gy prescription to the target (+/- 20%)
    assert abs(coreMean - 2.0) / 2.0 < 0.2
    # and spare the periphery: mean outside target well below prescription
    outside = dose3M[~mask3M & (dose3M > 0)]
    assert outside.mean() < coreMean


def test_flippedOrientationDoseLandsOnTarget(tmp_path):
    """A CT with flipped imageOrientationPatient must still work.

    Regression for clinical scans with ``imageOrientationPatient =
    [-1,0,0,0,-1,0]`` (non-identity SimpleITK direction), which crashed
    pyRadPlan's ray tracer ("attempt to get argmax of an empty sequence")
    before ctFromScan reoriented the CT and doseArrayFromSitk resampled the
    dose back onto the original scan grid.
    """
    planC = _makePhantomPlanC(tmp_path, flipOrientation=True)
    # confirm the phantom really has the flipped (non-identity) orientation
    imgOri = np.asarray(planC.scan[0].scanInfo[0].imageOrientationPatient)
    assert not np.allclose(imgOri, [1, 0, 0, 0, 1, 0])

    isoMm = _targetCentroidMm(planC)
    planC = _addBeams(planC, isoMm, gantryAngles=(0.0, 90.0))

    dij, stf, doseNum, planC = prp.calcBeamletDoseAndImport(
        planC, scanNum=0, beamsNum=0, bixelWidth=8.0,
        structNums=[0], targetStructNums=[0])

    dose3M = planC.dose[doseNum].doseArray
    assert dose3M.shape == tuple(planC.scan[0].getScanSize())
    assert dose3M.max() > 0

    from cerr.contour import rasterseg as rs
    mask3M = rs.getStrMask(planC.structure[0], planC)
    assert dose3M[mask3M].mean() > 0.5 * dose3M.max()

    # high-dose region must coincide with the target, not a mirrored location
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    hot = dose3M >= 0.7 * dose3M.max()
    rr, cc, ss = np.nonzero(hot)
    hotCtrCerr = np.array([xV[cc].mean(), yV[rr].mean(), zV[ss].mean(), 1.0])
    hotCtrMm = (planC.scan[0].cerrToDcmTransM @ hotCtrCerr)[:3]
    assert np.linalg.norm(hotCtrMm - isoMm) < 15.0
