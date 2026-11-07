"""``saveToH5`` writes planC.deform, and ``loadFromH5`` must read it back.

Before: loadFromH5 never called loadH5Deform, so every deformation vector field
was silently lost on a save/load round trip; loadH5Deform also iterated the H5
items unsorted (Item_10 before Item_2). Fully offline using the bundled phantom.
"""
import os

import numpy as np
import pytest

from cerr import datasets
from cerr import plan_container as pc

sitk = pytest.importorskip("SimpleITK")

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def _addConstantDvf(planC, tmp_path, shiftMm, name):
    scanObj = planC.scan[0]
    numRows, numCols, numSlcs = scanObj.getScanSize()
    xV, yV, zV = scanObj.getScanXYZVals()
    dvfArr = np.zeros((numSlcs, numRows, numCols, 3), dtype=np.float32)
    for comp, val in enumerate(shiftMm):
        dvfArr[..., comp] = val
    img = sitk.GetImageFromArray(dvfArr, isVector=True)
    img.SetSpacing([10 * abs(xV[1] - xV[0]), 10 * abs(yV[0] - yV[1]), 10 * abs(zV[1] - zV[0])])
    img.SetOrigin([float(v) for v in scanObj.scanInfo[0].imagePositionPatient])
    img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    f = str(tmp_path / f"{name}.nii.gz")
    sitk.WriteImage(img, f)
    return pc.loadNiiVf(f, 0, planC)


def test_deform_survives_h5_roundtrip(tmp_path):
    planC = pc.loadDcmDir(phantom_dir)
    shifts = [(float(i), -1.0, 0.5) for i in range(11)]          # > 10 items: checks ordering
    for i, sh in enumerate(shifts):
        planC = _addConstantDvf(planC, tmp_path, sh, f"dvf{i}")
    h5File = str(tmp_path / "plan.h5")
    pc.saveToH5(planC, h5File)

    planH5 = pc.loadFromH5(h5File)
    assert len(planH5.deform) == len(planC.deform) == 11
    for orig, back in zip(planC.deform, planH5.deform):
        assert back.deformUID == orig.deformUID
        np.testing.assert_array_equal(np.asarray(back.dvfMatrix), np.asarray(orig.dvfMatrix))
        np.testing.assert_allclose(np.asarray(back.zValuesV, float), np.asarray(orig.zValuesV, float))


def test_subset_of_deforms(tmp_path):
    planC = pc.loadDcmDir(phantom_dir)
    planC = _addConstantDvf(planC, tmp_path, (1.0, 0, 0), "a")
    planC = _addConstantDvf(planC, tmp_path, (2.0, 0, 0), "b")
    h5File = str(tmp_path / "plan.h5")
    pc.saveToH5(planC, h5File, deformNumV=[1])
    planH5 = pc.loadFromH5(h5File)
    assert [d.deformUID for d in planH5.deform] == [planC.deform[1].deformUID]
