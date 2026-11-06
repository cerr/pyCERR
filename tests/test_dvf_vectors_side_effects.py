"""``register.getDvfVectors`` must not modify the deformation it samples.

It nudges the z end points of the DVF grid outward before interpolating, and
``Deform.getDVFXYZVals`` returns ``deform.zValuesV`` itself, so editing that
array in place moved the stored first/last slice by 1e-3 cm on every call.
Fully offline using the bundled phantom; no Qt needed.
"""
import os

import numpy as np
import pytest

from cerr import datasets
from cerr import plan_container as pc
from cerr.registration import register

sitk = pytest.importorskip("SimpleITK")

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')
SHIFT_MM = (5.0, -3.0, 2.0)


def _constantDvfPlan(tmp_path):
    """Phantom planC with a DVF that shifts every voxel by SHIFT_MM (LPS mm)."""
    planC = pc.loadDcmDir(phantom_dir)
    scanObj = planC.scan[0]
    numRows, numCols, numSlcs = scanObj.getScanSize()
    xV, yV, zV = scanObj.getScanXYZVals()
    dvfArr = np.zeros((numSlcs, numRows, numCols, 3), dtype=np.float32)
    for comp, val in enumerate(SHIFT_MM):
        dvfArr[..., comp] = val
    img = sitk.GetImageFromArray(dvfArr, isVector=True)
    img.SetSpacing([10 * abs(xV[1] - xV[0]), 10 * abs(yV[0] - yV[1]), 10 * abs(zV[1] - zV[0])])
    img.SetOrigin([float(v) for v in scanObj.scanInfo[0].imagePositionPatient])
    img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    dvfFile = str(tmp_path / "dvf.nii.gz")
    sitk.WriteImage(img, dvfFile)
    return pc.loadNiiVf(dvfFile, 0, planC)


def test_getdvfvectors_does_not_modify_the_deform(tmp_path):
    planC = _constantDvfPlan(tmp_path)
    deformS = planC.deform[-1]
    zBefore = np.array(deformS.zValuesV, dtype=float, copy=True)
    first = register.getDvfVectors(deformS, planC, 0, outputResV=[0.4] * 3)
    second = register.getDvfVectors(deformS, planC, 0, outputResV=[0.4] * 3)
    np.testing.assert_array_equal(np.asarray(deformS.zValuesV, dtype=float), zBefore)
    # and repeated calls give the same vectors
    np.testing.assert_array_equal(first, second)
