"""Functional tests for ANTs (antspyx) registration (``cerr.registration.ants_reg``).

Registers a synthetically shifted copy of the bundled phantom back onto the
original and checks that the recovered scan matches the fixed scan and that a
moving structure warps into fixed space. Also exercises the mask and
landmark-based initial-alignment code paths. Skipped when antspyx is absent.
Fully offline using the bundled phantom.
"""
import os
import numpy as np
import pytest

from cerr import datasets
from cerr import plan_container as pc
from cerr.utils import mask as maskUtils

ants = pytest.importorskip("ants")

from cerr.registration import ants_reg  # noqa: E402
from cerr.contour.rasterseg import getStrMask  # noqa: E402

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def _shiftedMovingPlanC(shiftVoxels=(6, 4, 2)):
    """Fixed planC plus a moving planC that is the fixed scan shifted by voxels."""
    basePlanC = pc.loadDcmDir(phantom_dir)
    scan3M = basePlanC.scan[0].getScanArray()
    xV, yV, zV = basePlanC.scan[0].getScanXYZVals()
    moved3M = np.roll(scan3M, shift=shiftVoxels, axis=(0, 1, 2))
    movPlanC = pc.importScanArray(moved3M, xV, yV, zV, 'CT', 0, basePlanC)
    return basePlanC, movPlanC, xV, yV, zV


def test_ants_rigid_register_and_warp(tmp_path):
    basePlanC, planC, xV, yV, zV = _shiftedMovingPlanC()
    # scan 0 = fixed, scan 1 = shifted moving (same planC)
    numScans0 = len(planC.scan)

    planC = ants_reg.registerScansAnts(
        planC, 0, planC, 1, transformSaveDir=str(tmp_path),
        typeOfTransform='Rigid')

    # A warped scan and a deform object were added.
    assert len(planC.scan) == numScans0 + 1
    assert len(planC.deform) == 1
    deformS = planC.deform[-1]
    assert deformS.registrationTool == 'ants'
    assert deformS.deformOutFileType == 'ants'
    assert os.path.exists(deformS.deformOutFilePath)

    fixed3M = planC.scan[0].getScanArray().astype(float)
    warped3M = planC.scan[-1].getScanArray().astype(float)
    moving3M = planC.scan[1].getScanArray().astype(float)
    assert warped3M.shape == fixed3M.shape

    def corr(a, b):
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # Warped-vs-fixed should be much better aligned than moving-vs-fixed.
    assert corr(warped3M, fixed3M) > corr(moving3M, fixed3M)
    assert corr(warped3M, fixed3M) > 0.95


def test_ants_warp_structure(tmp_path):
    basePlanC, planC, xV, yV, zV = _shiftedMovingPlanC()

    # Put a structure on the moving scan (scan 1): a centered box.
    movSize = planC.scan[1].getScanSize()
    movMask3M = np.zeros(movSize, dtype=bool)
    r0, r1 = movSize[0] // 3, 2 * movSize[0] // 3
    c0, c1 = movSize[1] // 3, 2 * movSize[1] // 3
    movMask3M[r0:r1, c0:c1, :] = True
    planC = pc.importStructureMask(movMask3M, 1, 'movBox', planC)
    movStrNum = len(planC.structure) - 1

    planC = ants_reg.registerScansAnts(
        planC, 0, planC, 1, transformSaveDir=str(tmp_path),
        typeOfTransform='Rigid')
    deformS = planC.deform[-1]

    numStr0 = len(planC.structure)
    planC = ants_reg.warpStructuresAnts(planC, 0, planC, [movStrNum], deformS)
    assert len(planC.structure) == numStr0 + 1
    warpedMask = getStrMask(len(planC.structure) - 1, planC)
    assert warpedMask.any()
    # Warped structure is associated with the fixed scan.
    assert planC.structure[-1].assocScanUID == planC.scan[0].scanUID


def test_ants_masks_and_landmarks(tmp_path):
    """Mask and landmark code paths run and improve alignment."""
    basePlanC, planC, xV, yV, zV = _shiftedMovingPlanC(shiftVoxels=(5, 5, 0))

    baseMask3M = maskUtils.getPatientOutline(planC.scan[0].getScanArray())
    movMask3M = maskUtils.getPatientOutline(planC.scan[1].getScanArray())

    # Landmarks in pyCERR virtual coords (cm): a few voxel centers on each scan.
    # Moving is the fixed scan rolled by (5 rows, 5 cols); build matching pairs.
    rows = [20, 40, 30, 50]
    cols = [20, 30, 40, 25]
    slcs = [2, 4, 6, 8]
    baseLm = np.array([[xV[c], yV[r], zV[s]] for r, c, s in zip(rows, cols, slcs)])
    movLm = np.array([[xV[c + 5], yV[r + 5], zV[s]]
                      for r, c, s in zip(rows, cols, slcs)])

    planC = ants_reg.registerScansAnts(
        planC, 0, planC, 1, transformSaveDir=str(tmp_path),
        typeOfTransform='Rigid',
        baseMask3M=baseMask3M, movMask3M=movMask3M,
        baseLandmarksM=baseLm, movLandmarksM=movLm,
        landmarkCoordSys='cerr', landmarkTransformType='rigid')

    deformS = planC.deform[-1]
    assert deformS.deformParams['usedLandmarks'] is True

    fixed3M = planC.scan[0].getScanArray().astype(float)
    warped3M = planC.scan[-1].getScanArray().astype(float)

    def corr(a, b):
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert corr(warped3M, fixed3M) > 0.95
