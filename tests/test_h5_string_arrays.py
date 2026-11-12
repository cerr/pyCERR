"""saveToH5/loadFromH5 with metadata that holds several strings (a numpy unicode array or a list).

loadDose builds dose_meta.deidentificationMethodDescription with np.append, so a DICOM export
with a de-identification sequence gives a '<U..' array, which h5py cannot store:
TypeError: No conversion path for dtype: dtype('<U70').
"""
import os

import numpy as np

from cerr import plan_container as pc
from cerr import datasets


def _planC_with_string_arrays():
    phantom = os.path.join(os.path.dirname(datasets.__file__), "radiomics_phantom_dicom", "pat_1")
    planC = pc.loadDcmDir(phantom)
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    planC = pc.importDoseArray(np.ones(planC.scan[0].getScanSize(), dtype=np.float32), xV, yV, zV, planC, 0)
    d = planC.dose[0]
    d.deidentificationMethodDescription = np.array(
        ["113100: Basic Application Confidentiality Profile", "113107: Retain Longitudinal Temporal "
         "Information Modified Dates Option"])          # '<U84', as loadDose builds it
    d.refBeamNumber = ["1", "2", "3"]                    # a list of str
    d.doseType = "PHYSICAL"
    return planC


def test_string_arrays_round_trip(tmp_path):
    planC = _planC_with_string_arrays()
    f = os.path.join(tmp_path, "p.h5")
    pc.saveToH5(planC, f)                                # raised TypeError before the fix
    back = pc.loadFromH5(f)
    d = back.dose[0]
    assert list(d.deidentificationMethodDescription) == list(planC.dose[0].deidentificationMethodDescription)
    assert all(isinstance(x, str) for x in d.deidentificationMethodDescription)
    assert list(d.refBeamNumber) == ["1", "2", "3"]
    assert d.doseType == "PHYSICAL"
    np.testing.assert_array_equal(d.doseArray, planC.dose[0].doseArray)


def test_numeric_arrays_unchanged(tmp_path):
    planC = _planC_with_string_arrays()
    planC.dose[0].deidentificationMethodDescription = np.array([], dtype=float)
    f = os.path.join(tmp_path, "p.h5")
    pc.saveToH5(planC, f)
    back = pc.loadFromH5(f)
    assert back.dose[0].doseArray.dtype == np.float32
