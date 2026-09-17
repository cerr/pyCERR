"""Tests for structure JSON serialization (``getJsonList`` / ``importJson``).

Structures reach the JSON encoders from several loaders, and not all of them
produce plain Python values. A structure imported from NIfTI got its default
color from ``getColorForStructNum`` as a list of ``np.int64``, and
``getJsonList`` failed on it with ``TypeError: Unexpected type int64`` while the
same structure loaded from RTSTRUCT serialized fine.
"""
import json
import os

import numpy as np
import pytest

from cerr import datasets
from cerr import plan_container as pc
from cerr.contour.rasterseg import getStrMask
from cerr.dataclasses import structure as cerrStr

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


@pytest.fixture(scope='module')
def planCWithNiiStruct(tmp_path_factory):
    """Phantom planC plus a copy of its GTV re-imported through NIfTI."""
    planC = pc.loadDcmDir(phantom_dir)
    niiFile = os.path.join(str(tmp_path_factory.mktemp('seg')), 'seg.nii.gz')
    planC.structure[0].saveNii(niiFile, planC)
    planC = pc.loadNiiStructure(niiFile, 0, planC, labels_dict={'GTV_nii': 1})
    return planC, len(planC.structure) - 1


def test_default_color_is_native_ints():
    for structNum in (0, 5, 27, 28, np.int64(3)):
        color = cerrStr.getColorForStructNum(structNum)
        assert len(color) == 3
        assert all(type(c) is int for c in color), color


def test_nifti_imported_structure_serializes(planCWithNiiStruct):
    planC, structNum = planCWithNiiStruct
    strList = cerrStr.getJsonList(structNum, planC)
    assert len(strList) == 1
    strDict = json.loads(strList[0])
    assert strDict['structureName'] == 'GTV_nii'
    assert all(type(c) is int for c in strDict['structureColor'])


def test_json_roundtrip_preserves_mask(planCWithNiiStruct):
    planC, structNum = planCWithNiiStruct
    origMask3M = getStrMask(structNum, planC)
    strList = cerrStr.getJsonList(structNum, planC)

    # importJson skips structures whose strUID is already present, so remove
    # the original before importing it back.
    removed = planC.structure.pop(structNum)
    try:
        planC = cerrStr.importJson(planC, strList=strList)
        newNum = len(planC.structure) - 1
        assert planC.structure[newNum].strUID == removed.strUID
        np.testing.assert_array_equal(getStrMask(newNum, planC), origMask3M)
    finally:
        # leave the module fixture as it was for any later test
        planC.structure[-1] = removed


def test_encoder_converts_numpy_values_on_any_field(planCWithNiiStruct):
    """Fields restored from HDF5 or pickles may be arrays or numpy scalars."""
    planC, structNum = planCWithNiiStruct
    strObj = planC.structure[structNum]
    saved = (strObj.structureColor, strObj.roiGenerationDescription)
    try:
        strObj.structureColor = np.array([10, 20, 30], dtype=np.uint8)
        strObj.roiGenerationDescription = np.float32(1.5)
        strDict = json.loads(cerrStr.getJsonList(structNum, planC)[0])
        assert strDict['structureColor'] == [10, 20, 30]
        assert strDict['roiGenerationDescription'] == pytest.approx(1.5)
    finally:
        strObj.structureColor, strObj.roiGenerationDescription = saved


def test_encoder_still_rejects_unsupported_types(planCWithNiiStruct):
    planC, structNum = planCWithNiiStruct
    strObj = planC.structure[structNum]
    saved = strObj.roiGenerationAlgorithm
    try:
        strObj.roiGenerationAlgorithm = object()
        with pytest.raises(TypeError, match='Unexpected type object'):
            cerrStr.getJsonList(structNum, planC)
    finally:
        strObj.roiGenerationAlgorithm = saved
