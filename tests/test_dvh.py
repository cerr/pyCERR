"""Tests for DVH computation (``cerr.dvh``) against an analytic uniform-dose case.

Builds a synthetic box structure and a spatially-uniform dose on the bundled
DICOM phantom grid (no network). With a constant dose ``D`` over the whole grid
the DVH is exactly predictable: every structure voxel receives ``D``, the
integrated histogram volume equals the structure volume, and the
Dx/Vx/MOHx/meanDose summary metrics all collapse to ``D``. This pins down
``getDVH`` (mask integration + dose interpolation) and the histogram metrics,
which are otherwise only exercised indirectly.
"""
import os
import numpy as np

from cerr import datasets
from cerr import plan_container as pc
from cerr.dataclasses import structure as structr
from cerr.contour.rasterseg import getStrMask
from cerr import dvh

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')

UNIFORM_DOSE = 40.0  # Gy


def _setup():
    """Phantom planC + a box structure + a spatially-uniform dose grid."""
    planC = pc.loadDcmDir(phantom_dir)
    nRows, nCols, nSlc = planC.scan[0].getScanSize()

    mask = np.zeros((nRows, nCols, nSlc), dtype=bool)
    r0, r1 = nRows // 4, nRows - nRows // 4
    c0, c1 = nCols // 4, nCols - nCols // 4
    s0 = max(1, nSlc // 2 - 1)
    s1 = min(nSlc - 1, nSlc // 2 + 2)
    mask[r0:r1, c0:c1, s0:s1] = True
    planC = structr.importStructureMask(mask, 0, 'dvh_box', planC)
    structNum = len(planC.structure) - 1

    # Spatially-uniform dose on the scan grid.
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    dose3M = np.full((nRows, nCols, nSlc), UNIFORM_DOSE, dtype=float)
    planC = pc.importDoseArray(dose3M, xV, yV, zV, planC, 0)
    doseNum = len(planC.dose) - 1
    return planC, structNum, doseNum


def test_getDVH_uniform_dose():
    """Every structure voxel sees the uniform dose."""
    planC, structNum, doseNum = _setup()
    dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)
    assert isErr == 0
    assert len(dosesV) > 0
    np.testing.assert_allclose(dosesV, UNIFORM_DOSE, rtol=1e-3, atol=1e-3)


def test_dvh_volume_matches_mask():
    """Integrated DVH volume equals (voxel count) x (voxel volume)."""
    planC, structNum, doseNum = _setup()
    dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)

    xV, yV, zV = planC.scan[0].getScanXYZVals()
    dx = abs(xV[1] - xV[0])
    dy = abs(yV[1] - yV[0])
    dz = abs(zV[1] - zV[0])
    voxVol = dx * dy * dz  # cc

    nVox = int(getStrMask(structNum, planC).sum())
    assert nVox > 0
    np.testing.assert_allclose(volsV.sum(), nVox * voxVol, rtol=0.02)


def test_dvh_metrics_uniform_dose():
    """Dx / Vx / MOHx / meanDose collapse to the uniform dose."""
    planC, structNum, doseNum = _setup()
    dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)
    binWidth = 0.2
    doseBinsV, volHistV = dvh.doseHist(dosesV, volsV, binWidth)

    # Fraction of volume below D is the whole structure; above D is none.
    np.testing.assert_allclose(
        dvh.Vx(doseBinsV, volHistV, UNIFORM_DOSE - 1.0, volumeType=1),
        1.0, atol=1e-6)
    assert dvh.Vx(doseBinsV, volHistV, UNIFORM_DOSE + 1.0, volumeType=1) == 0

    # Dose to the hottest 50% of volume, mean of hottest 90%, and mean dose
    # all equal D to within one bin.
    assert abs(dvh.Dx(doseBinsV, volHistV, 50.0, volumeType=1) - UNIFORM_DOSE) <= binWidth
    assert abs(dvh.MOHx(doseBinsV, volHistV, 90) - UNIFORM_DOSE) <= binWidth
    assert abs(dvh.meanDose(doseBinsV, volHistV) - UNIFORM_DOSE) <= binWidth
