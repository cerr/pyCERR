"""Tests for cerr.gamma (3-D gamma-index dose comparison).

All tests are self-contained: synthetic doses with known shifts / dose
differences whose gamma values are predictable in closed form. No optional
dependencies are used.
"""

import os
import tempfile

import numpy as np
import pytest
import SimpleITK as sitk

from cerr import gamma
from cerr import plan_container as pc


def _gaussianDose(shape=(40, 40, 24), spacing_mm=2.0, amp=100.0):
    """A smooth 3-D Gaussian dose bump centred in the volume (mm axes)."""
    n0, n1, n2 = shape
    a0 = (np.arange(n0) - (n0 - 1) / 2) * spacing_mm
    a1 = (np.arange(n1) - (n1 - 1) / 2) * spacing_mm
    a2 = (np.arange(n2) - (n2 - 1) / 2) * spacing_mm
    Z0, Z1, Z2 = np.meshgrid(a0, a1, a2, indexing='ij')
    sig = 20.0
    dose = amp * np.exp(-(Z0**2 + Z1**2 + Z2**2) / (2 * sig**2))
    return dose, (a0, a1, a2), spacing_mm


# --------------------------------------------------------------------------
# Self-consistency (no optional deps)
# --------------------------------------------------------------------------

def test_identicalDoseAllPass():
    dose, _, sp = _gaussianDose()
    g = gamma.gammaDose3d(dose, dose, (sp, sp, sp),
                          distAgreement=3.0, doseAgreement=3.0,
                          thresholdDose=0.2 * dose.max())
    # identical distributions -> gamma == 0 everywhere it is evaluated
    valid = ~np.isnan(g)
    assert np.all(g[valid] < 1e-9)
    assert gamma.gammaPassRate(g) == pytest.approx(1.0)


def test_largeUniformOffsetFails():
    dose, _, sp = _gaussianDose()
    # +30% of max added everywhere: dose diff (30 Gy) >> 3% DD, and the bump is
    # broad so no nearby voxel is within DTA either -> most voxels fail.
    evalDose = dose + 0.30 * dose.max()
    g = gamma.gammaDose3d(dose, evalDose, (sp, sp, sp),
                          distAgreement=3.0, doseAgreement=3.0,
                          thresholdDose=0.2 * dose.max())
    assert gamma.gammaPassRate(g) < 0.5


def test_smallScalingMostlyPasses():
    dose, _, sp = _gaussianDose()
    evalDose = dose * 1.02          # 2% scaling, under a 3% global DD
    g = gamma.gammaDose3d(dose, evalDose, (sp, sp, sp),
                          distAgreement=3.0, doseAgreement=3.0,
                          thresholdDose=0.2 * dose.max())
    assert gamma.gammaPassRate(g) > 0.99


def test_localVsGlobal():
    """A fixed absolute offset fails more of the low-dose region under a local
    criterion than under a global one."""
    dose, _, sp = _gaussianDose()
    evalDose = dose + 2.0           # +2 Gy everywhere
    gGlobal = gamma.gammaDose3d(dose, evalDose, (sp, sp, sp),
                                doseAgreementType='global',
                                thresholdDose=0.2 * dose.max())
    gLocal = gamma.gammaDose3d(dose, evalDose, (sp, sp, sp),
                               doseAgreementType='local',
                               thresholdDose=0.2 * dose.max())
    assert gamma.gammaPassRate(gLocal) <= gamma.gammaPassRate(gGlobal)


def test_shapeMismatchRaises():
    a = np.zeros((10, 10, 10))
    b = np.zeros((10, 10, 9))
    with pytest.raises(ValueError, match="same shape"):
        gamma.gammaDose3d(a, b, (2, 2, 2))


# --------------------------------------------------------------------------
# Known-answer tests (closed-form gamma for simple perturbations)
# --------------------------------------------------------------------------

def test_flatDoseKnownDifference():
    """On a spatially flat dose, DTA cannot help, so gamma == doseDiff / DD.

    ref = 50 everywhere; eval = 50 + delta. With a global DD criterion of
    ``doseAgreement`` percent of ``normalizationDose``, gamma is exactly
    delta / (doseAgreement/100 * normalizationDose) at every voxel.
    """
    shape = (20, 20, 12)
    ref = np.full(shape, 50.0)
    normDose = 100.0
    dd_pct = 3.0                       # DD_abs = 3 Gy
    for delta, expected in [(1.5, 0.5), (3.0, 1.0), (0.75, 0.25)]:
        evalDose = ref + delta
        g = gamma.gammaDose3d(ref, evalDose, (2.0, 2.0, 2.0),
                              distAgreement=3.0, doseAgreement=dd_pct,
                              thresholdDose=0.0, doseAgreementType='global',
                              normalizationDose=normDose)
        valid = ~np.isnan(g)
        assert np.allclose(g[valid], expected, atol=1e-6)


def test_linearGradientKnownShift():
    """For a pure spatial shift ``s`` of a linear gradient ``g`` (dose/mm),
    the minimum gamma has the closed form

        gamma = s*g / sqrt(DD^2 + (g*DTA)^2)

    (Low 1998). Here g=1 Gy/mm, s=3 mm, DD=3 Gy, DTA=3 mm -> gamma = 3/sqrt(18).
    """
    n0, n1, n2 = 60, 8, 8
    sp = 1.0                                   # 1 mm voxels along the gradient
    g = 1.0                                    # 1 Gy/mm
    idx0 = np.arange(n0)[:, None, None]
    ref = np.broadcast_to(g * idx0 * sp, (n0, n1, n2)).astype(float)
    shift_mm = 3.0
    # eval(x) = ref(x - shift): a pure spatial shift of the gradient
    evalDose = ref - g * shift_mm

    dd, dta = 3.0, 3.0
    expected = shift_mm * g / np.sqrt(dd ** 2 + (g * dta) ** 2)   # ~0.7071

    gm = gamma.gammaDose3d(
        ref, evalDose, (sp, sp, sp),
        distAgreement=dta, doseAgreement=dd, thresholdDose=1e-6,
        doseAgreementType='global', normalizationDose=100.0,   # DD_abs = 3 Gy
        maxSearchDistance=2 * dta, distSampleRate=10)

    # interior slab, away from the boundaries the search runs out of samples
    interior = gm[15:45, :, :]
    interior = interior[~np.isnan(interior)]
    assert np.nanmedian(interior) == pytest.approx(expected, abs=0.02)


# --------------------------------------------------------------------------
# planC-level entry points (scan-grid gamma + per-structure pass rates)
# --------------------------------------------------------------------------

def _phantomWithTwoDoses(tmp_path, offsetGy=0.0):
    """Water-box planC with a spherical structure and two doses.

    The evaluated dose is the reference plus a constant ``offsetGy`` so the
    per-structure gamma pass rate is analytically predictable.
    """
    nz, ny, nx, vox = 24, 48, 48, 3.0
    hu = np.full((nz, ny, nx), -1000.0, dtype=np.float32)
    hu[3:-3, 6:-6, 6:-6] = 0.0
    img = sitk.GetImageFromArray(hu)
    img.SetSpacing([vox] * 3)
    img.SetOrigin([0.0, 0.0, 0.0])
    nii = os.path.join(tmp_path, "ph.nii.gz")
    sitk.WriteImage(img, nii)
    planC = pc.loadNiiScan(nii, imageType="CT SCAN")

    xV, yV, zV = planC.scan[0].getScanXYZVals()
    cx, cy, cz = xV.mean(), yV.mean(), zV.mean()
    Xc = xV[None, :, None]; Yc = yV[:, None, None]; Zc = zV[None, None, :]
    r2 = (Xc - cx) ** 2 + (Yc - cy) ** 2 + (Zc - cz) ** 2
    sphere = r2 <= 3.0 ** 2                    # 3 cm sphere
    pc.importStructureMask(sphere, 0, "SPHERE", planC)

    # a smooth reference dose peaked at the centre, and eval = ref + offset
    ref3M = 60.0 * np.exp(-r2 / (2 * 2.0 ** 2))
    pc.importDoseArray(ref3M, xV, yV, zV, planC, 0,
                       {"fractionGroupID": "ref", "units": "GRAYS"})
    pc.importDoseArray(ref3M + offsetGy, xV, yV, zV, planC, 0,
                       {"fractionGroupID": "eval", "units": "GRAYS"})
    return planC


def test_gammaForScanIdenticalDoses(tmp_path):
    planC = _phantomWithTwoDoses(str(tmp_path), offsetGy=0.0)
    gamma3M, passRate = gamma.gammaDose3dForScan(
        0, 1, 0, planC, distAgreement=3.0, doseAgreement=3.0,
        thresholdFraction=0.2)
    assert passRate == pytest.approx(1.0)
    assert gamma3M.shape == tuple(int(v) for v in planC.scan[0].getScanSize())


def test_gammaByStructure(tmp_path):
    planC = _phantomWithTwoDoses(str(tmp_path), offsetGy=0.0)
    gamma3M, _ = gamma.gammaDose3dForScan(0, 1, 0, planC)
    sphereNum = [i for i, s in enumerate(planC.structure)
                 if s.structureName == "SPHERE"][0]
    rows = gamma.gammaByStructure(gamma3M, [sphereNum], planC)
    assert len(rows) == 1
    assert rows[0]["structureName"] == "SPHERE"
    assert rows[0]["numEvaluated"] > 0
    assert rows[0]["passRate"] == pytest.approx(1.0)     # identical doses
