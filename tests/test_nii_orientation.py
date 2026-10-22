"""
 Orientation of NIfTI imports: loadNiiScan reorients to LPS by default, and
 doses / structures from NIfTI stay aligned with the reoriented scan.
"""
import os

import numpy as np
import SimpleITK as sitk

from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.dataclasses import scan as scn
from cerr.dvh import getDVH

currPath = os.path.abspath(__file__)
cerrPath = os.path.join(os.path.dirname(os.path.dirname(currPath)), 'cerr')
dataDir = os.path.join(cerrPath, 'datasets', 'sample_ct', 'dosimetric_model_test_data')
scanFile = os.path.join(dataDir, 'scan.nii')
doseFile = os.path.join(dataDir, 'dose.nii')
maskFile = os.path.join(dataDir, 'mask_(testStr1 U testStr3).nii')

LPS_ORI = [1, 0, 0, 0, 1, 0]


def load_sample(direction=None):
    if direction is None:
        planC = pc.loadNiiScan(scanFile, 'CT SCAN')
    else:
        planC = pc.loadNiiScan(scanFile, 'CT SCAN', direction)
    planC = pc.loadNiiDose(doseFile, 0, planC)
    planC = pc.loadNiiStructure(maskFile, 0, planC)
    return planC


def test_sample_file_is_not_lps():
    # Guards the premise of the tests below: the sample is stored RAS
    img = sitk.ReadImage(scanFile)
    assert sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection()) == 'RAS'


def test_default_orientation_is_lps():
    planC = load_sample()
    for sInfo in planC.scan[0].scanInfo:
        np.testing.assert_array_equal(sInfo.imageOrientationPatient, LPS_ORI)
    np.testing.assert_array_equal(planC.dose[0].imageOrientationPatient.ravel(), LPS_ORI)


def test_native_orientation_opt_out():
    for optOut in ['', None]:
        planC = pc.loadNiiScan(scanFile, 'CT SCAN', optOut)
        np.testing.assert_array_equal(planC.scan[0].scanInfo[0].imageOrientationPatient,
                                      [-1, 0, 0, 0, -1, 0])


def test_lps_is_native_flipped_in_plane():
    nativeC = load_sample('')
    lpsC = load_sample()
    nativeScan3M = nativeC.scan[0].getScanArray()
    lpsScan3M = lpsC.scan[0].getScanArray()
    np.testing.assert_array_equal(lpsScan3M, nativeScan3M[::-1, ::-1, :])
    np.testing.assert_allclose(lpsC.dose[0].doseArray, nativeC.dose[0].doseArray[::-1, ::-1, :])
    nativeMask3M = rs.getStrMask(0, nativeC)
    lpsMask3M = rs.getStrMask(0, lpsC)
    assert lpsMask3M.sum() > 0
    np.testing.assert_array_equal(lpsMask3M, nativeMask3M[::-1, ::-1, :])


def test_dose_and_structure_aligned_with_lps_scan():
    nativeC = load_sample('')
    lpsC = load_sample()

    # Same voxels: identical HU under the mask, identical DVH
    nativeMask3M = rs.getStrMask(0, nativeC)
    lpsMask3M = rs.getStrMask(0, lpsC)
    np.testing.assert_array_equal(np.sort(lpsC.scan[0].getScanArray()[lpsMask3M]),
                                  np.sort(nativeC.scan[0].getScanArray()[nativeMask3M]))
    nativeDoseV, nativeVolV = getDVH(0, 0, nativeC)[:2]
    lpsDoseV, lpsVolV = getDVH(0, 0, lpsC)[:2]
    np.testing.assert_allclose(np.sort(lpsDoseV), np.sort(nativeDoseV))
    np.testing.assert_allclose(np.sum(lpsVolV), np.sum(nativeVolV))

    # Dose grid coincides with the scan grid in virtual coordinates
    sx, sy, sz = lpsC.scan[0].getScanXYZVals()
    dx, dy, dz = lpsC.dose[0].getDoseXYZVals()
    np.testing.assert_allclose(dx, sx, atol=1e-6)
    np.testing.assert_allclose(dy, sy, atol=1e-6)
    np.testing.assert_allclose(dz, sz, atol=1e-6)


def test_dose_uses_associated_scan(tmp_path):
    # A single-slice scan 0 has no slice order (flipSliceOrderFlag is False), unlike
    # the multi-slice scan the dose belongs to. The dose must follow its own scan.
    img = sitk.ReadImage(scanFile)
    oneSliceFile = str(tmp_path / 'one_slice.nii.gz')
    sitk.WriteImage(img[:, :, 0:1], oneSliceFile)
    planC = pc.loadNiiScan(oneSliceFile, 'CT SCAN')
    planC = pc.loadNiiScan(scanFile, 'CT SCAN', initplanC=planC)
    assert scn.flipSliceOrderFlag(planC.scan[0]) != scn.flipSliceOrderFlag(planC.scan[1])
    planC = pc.loadNiiDose(doseFile, 1, planC)

    refC = load_sample()
    np.testing.assert_allclose(planC.dose[0].doseArray, refC.dose[0].doseArray)
    np.testing.assert_allclose(planC.dose[0].getDoseXYZVals()[2], refC.dose[0].getDoseXYZVals()[2])


def test_4d_nifti_reoriented_per_volume(tmp_path):
    img = sitk.ReadImage(scanFile, sitk.sitkFloat32)
    vol4 = sitk.JoinSeries([img, img * 2])
    vol4File = str(tmp_path / 'scan_4d.nii.gz')
    sitk.WriteImage(vol4, vol4File)

    planC = pc.loadNiiScan(vol4File, 'CT SCAN')
    refC = pc.loadNiiScan(scanFile, 'CT SCAN')
    assert len(planC.scan) == 2
    for s in planC.scan:
        np.testing.assert_array_equal(s.scanInfo[0].imageOrientationPatient, LPS_ORI)
    ref3M = refC.scan[0].getScanArray().astype(float)
    np.testing.assert_allclose(planC.scan[0].getScanArray(), ref3M)
    np.testing.assert_allclose(planC.scan[1].getScanArray(), 2 * ref3M)


def test_4d_label_map_aligned_with_lps_scan(tmp_path):
    # 4-D label maps are resampled onto the scan like 3-D ones
    mask = sitk.ReadImage(maskFile)
    mask4 = sitk.JoinSeries([mask, mask * 2])
    mask4File = str(tmp_path / 'mask_4d.nii.gz')
    sitk.WriteImage(mask4, mask4File)

    planC = pc.loadNiiScan(scanFile, 'CT SCAN')
    planC = pc.loadNiiStructure(maskFile, 0, planC)
    planC = pc.loadNiiStructure(mask4File, 0, planC, {'A': 1, 'B': 2})
    ref3M = rs.getStrMask(0, planC)
    assert ref3M.sum() > 0
    np.testing.assert_array_equal(rs.getStrMask(1, planC), ref3M)
    np.testing.assert_array_equal(rs.getStrMask(2, planC), ref3M)
