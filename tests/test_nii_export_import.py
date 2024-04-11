from cerr import datasets
import os
from cerr import plan_container as pc
import numpy as np

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),'radiomics_phantom_dicom')
pat_names = ['PAT1', 'PAT2', 'PAT3', 'PAT4']
all_pat_dirs = [os.path.join(phantom_dir, pat) for pat in pat_names]
dcm_dir = all_pat_dirs[0]

def test_scan_export_import():
    planC = pc.load_dcm_dir(dcm_dir)
    scanNiiFile = 'scan_from_cerr.nii.gz'
    scanNum = 0
    planC.scan[scanNum].save_nii(scanNiiFile)
    imageType = 'CT SCAN'
    scanOrientation = ''
    planC = pc.load_nii_scan(scanNiiFile, imageType, scanOrientation, planC)
    scanArrayDcm = planC.scan[scanNum].getScanArray()
    scanArrayNii = planC.scan[scanNum+1].getScanArray()
    np.testing.assert_almost_equal(scanArrayDcm, scanArrayNii)
