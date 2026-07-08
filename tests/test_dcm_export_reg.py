"""Round-trip test for REG DICOM export (``cerr.dcm_export.reg_iod``).

Builds a synthetic deformation vector field on the bundled phantom's grid,
imports it as a deform object, exports it to a Deformable Spatial
Registration (REG) DICOM file, and verifies the grid geometry and vector
data are preserved exactly. Fully offline using the bundled phantom.
"""
import os
import numpy as np
import SimpleITK as sitk
from pydicom import dcmread

from cerr import datasets
from cerr import plan_container as pc
from cerr.dcm_export import reg_iod

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def test_reg_export_roundtrip(tmp_path):
    planC = pc.loadDcmDir(phantom_dir)
    scanObj = planC.scan[0]
    numRows, numCols, numSlcs = scanObj.getScanSize()

    # Grid geometry from the scan (mm, DICOM LPS)
    xV, yV, zV = scanObj.getScanXYZVals()
    spacing = [10 * abs(xV[1] - xV[0]), 10 * abs(yV[0] - yV[1]),
               10 * abs(zV[1] - zV[0])]
    origin = [float(v) for v in scanObj.scanInfo[0].imagePositionPatient]

    # Synthetic smooth DVF, distinct x/y/z components (mm)
    dvfArr = np.zeros((numSlcs, numRows, numCols, 3), dtype=np.float32)
    zz, yy, xx = np.meshgrid(np.arange(numSlcs), np.arange(numRows),
                             np.arange(numCols), indexing='ij')
    dvfArr[..., 0] = 0.5 * xx
    dvfArr[..., 1] = -0.25 * yy
    dvfArr[..., 2] = 0.1 * zz

    dvfImg = sitk.GetImageFromArray(dvfArr, isVector=True)
    dvfImg.SetSpacing(spacing)
    dvfImg.SetOrigin(origin)
    dvfImg.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])  # LPS-aligned
    dvfFile = os.path.join(str(tmp_path), 'dvf.nii.gz')
    sitk.WriteImage(dvfImg, dvfFile)

    # Import as a deform object and export to REG DICOM
    planC = pc.loadNiiVf(dvfFile, 0, planC)
    assert len(planC.deform) == 1

    regFile = os.path.join(str(tmp_path), 'reg_export.dcm')
    reg_iod.create(0, regFile, planC, movScanNum=0,
                   seriesOpts={'seriesDescription': 'pyCERR REG export test'})
    assert os.path.exists(regFile)
    assert os.path.getsize(regFile) > 0

    ds = dcmread(regFile)
    assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.3"
    assert ds.Modality == "REG"
    assert ds.PatientID == scanObj.scanInfo[0].patientID
    assert ds.StudyInstanceUID == scanObj.scanInfo[0].studyInstanceUID
    assert ds.FrameOfReferenceUID == scanObj.scanInfo[0].frameOfReferenceUID

    # Referenced series covers every slice of the scan
    refSeries = ds.ReferencedSeriesSequence[0]
    assert refSeries.SeriesInstanceUID == scanObj.scanInfo[0].seriesInstanceUID
    refSops = {item.ReferencedSOPInstanceUID
               for item in refSeries.ReferencedInstanceSequence}
    assert refSops == {sInfo.sopInstanceUID for sInfo in scanObj.scanInfo}

    # Two items: fixed (identity) and moving (with DVF grid)
    defSeq = ds.DeformableRegistrationSequence
    assert len(defSeq) == 2
    assert defSeq[0].RegistrationTypeCodeSequence[0].CodeValue == "125021"
    assert defSeq[1].RegistrationTypeCodeSequence[0].CodeValue == "125024"

    grid = defSeq[1].DeformableRegistrationGridSequence[0]
    assert list(grid.GridDimensions) == [numCols, numRows, numSlcs]
    np.testing.assert_allclose([float(v) for v in grid.GridResolution],
                               spacing, atol=1e-4)
    np.testing.assert_allclose([float(v) for v in grid.ImagePositionPatient],
                               origin, atol=1e-4)

    # Vector data round-trips bit-for-bit (x fastest, then rows, then slices)
    vecData = np.frombuffer(grid.VectorGridData, dtype=np.float32)
    expected = dvfArr.reshape(-1, 3).ravel()
    np.testing.assert_array_equal(vecData, expected)

    # Pre/post rigid matrices default to identity
    preMat = defSeq[1].PreDeformationMatrixRegistrationSequence[0]
    np.testing.assert_array_equal(
        np.array(preMat.FrameOfReferenceTransformationMatrix).reshape(4, 4),
        np.eye(4))
