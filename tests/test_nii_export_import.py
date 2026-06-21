from cerr import datasets
import os
from cerr import plan_container as pc
import numpy as np

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),'radiomics_phantom_dicom')
pat_names = ['pat_1', 'pat_2', 'pat_3', 'pat_4']
all_pat_dirs = [os.path.join(phantom_dir, pat) for pat in pat_names]
dcm_dir = all_pat_dirs[0]

def test_scan_export_import():
    planC = pc.loadDcmDir(dcm_dir)
    scanNiiFile = 'scan_from_cerr.nii.gz'
    scanNum = 0
    planC.scan[scanNum].saveNii(scanNiiFile)
    imageType = 'CT SCAN'
    scanOrientation = ''
    planC = pc.loadNiiScan(scanNiiFile, imageType, scanOrientation, planC)
    scanArrayDcm = planC.scan[scanNum].getScanArray()
    scanArrayNii = planC.scan[scanNum+1].getScanArray()
    np.testing.assert_almost_equal(scanArrayDcm, scanArrayNii)


def test_dose_export_import(tmp_path):
    # pyCERR has no DICOM RTDOSE export; NIfTI is the supported dose export.
    planC = pc.loadDcmDir(dcm_dir)
    nRows, nCols, nSlc = planC.scan[0].getScanSize()
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    # Non-uniform dose (ramp across columns) to exercise array fidelity.
    ramp = np.linspace(0.0, 60.0, nCols, dtype=float)
    dose3M = np.broadcast_to(ramp[None, :, None], (nRows, nCols, nSlc)).copy()
    planC = pc.importDoseArray(dose3M, xV, yV, zV, planC, 0)

    doseNiiFile = str(tmp_path / 'dose_from_cerr.nii.gz')
    planC.dose[0].saveNii(doseNiiFile)
    planC = pc.loadNiiDose(doseNiiFile, 0, planC)

    doseDcm = planC.dose[0].doseArray
    doseNii = planC.dose[1].doseArray
    np.testing.assert_allclose(doseNii, doseDcm, rtol=1e-3, atol=1e-3)


def test_structure_export_import(tmp_path):
    # NIfTI structure (label-map) export/import - the File > Export structure
    # path (saveNiiStructure / loadNiiStructure).
    from cerr.contour import rasterseg as rs
    planC = pc.loadDcmDir(dcm_dir)
    assert len(planC.structure) >= 1
    strNum = 0
    name = planC.structure[strNum].structureName
    origMask = rs.getStrMask(strNum, planC).astype(bool)

    strNiiFile = str(tmp_path / 'struct_from_cerr.nii.gz')
    pc.saveNiiStructure(strNiiFile, {name: 1}, planC, [strNum])

    nStruct0 = len(planC.structure)
    planC = pc.loadNiiStructure(strNiiFile, 0, planC, {name: 1})
    assert len(planC.structure) > nStruct0

    newMask = rs.getStrMask(nStruct0, planC).astype(bool)
    inter = int(np.logical_and(origMask, newMask).sum())
    dice = 2.0 * inter / (int(origMask.sum()) + int(newMask.sum()))
    assert dice > 0.98
