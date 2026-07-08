"""Round-trip test for DICOM REG import (``cerr.plan_container.loadDcmReg``).

Builds a synthetic deformation vector field on the bundled phantom's grid,
exports it to a Deformable Spatial Registration (REG) DICOM file via
``cerr.dcm_export.reg_iod``, re-imports it with ``loadDcmReg``, and checks that
the recovered ``Deform`` object matches the one produced by importing the same
DVF directly from NIfTI with ``loadNiiVf``. Fully offline using the bundled
phantom.
"""
import os
import numpy as np
import SimpleITK as sitk

from cerr import datasets
from cerr import plan_container as pc
from cerr.dcm_export import reg_iod

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def _writeSyntheticDvf(scanObj, dvfFile):
    """Write a smooth synthetic LPS-mm DVF on the scan grid; return the array."""
    numRows, numCols, numSlcs = scanObj.getScanSize()
    xV, yV, zV = scanObj.getScanXYZVals()
    spacing = [10 * abs(xV[1] - xV[0]), 10 * abs(yV[0] - yV[1]),
               10 * abs(zV[1] - zV[0])]
    origin = [float(v) for v in scanObj.scanInfo[0].imagePositionPatient]

    dvfArr = np.zeros((numSlcs, numRows, numCols, 3), dtype=np.float32)
    zz, yy, xx = np.meshgrid(np.arange(numSlcs), np.arange(numRows),
                             np.arange(numCols), indexing='ij')
    dvfArr[..., 0] = 0.5 * xx
    dvfArr[..., 1] = -0.25 * yy
    dvfArr[..., 2] = 0.1 * zz

    img = sitk.GetImageFromArray(dvfArr, isVector=True)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])  # LPS-aligned
    sitk.WriteImage(img, dvfFile)
    return dvfArr


def test_reg_import_matches_nii_import(tmp_path):
    planC = pc.loadDcmDir(phantom_dir)
    scanObj = planC.scan[0]

    dvfFile = os.path.join(str(tmp_path), 'dvf.nii.gz')
    _writeSyntheticDvf(scanObj, dvfFile)

    # Reference: import the DVF straight from NIfTI.
    planC = pc.loadNiiVf(dvfFile, 0, planC)
    refDeform = planC.deform[-1]

    # Export to REG DICOM, then re-import.
    regFile = os.path.join(str(tmp_path), 'reg.dcm')
    reg_iod.create(0, regFile, planC, movScanNum=0)

    numDeformBefore = len(planC.deform)
    planC = pc.loadDcmReg(regFile, planC)
    assert len(planC.deform) == numDeformBefore + 1
    regDeform = planC.deform[-1]

    assert regDeform.registrationTool == 'dicom'
    assert regDeform.baseScanUID == scanObj.scanUID

    # Geometry and vector field match the NIfTI-imported reference.
    np.testing.assert_allclose(regDeform.dvfMatrix, refDeform.dvfMatrix,
                               rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(regDeform.zValuesV, refDeform.zValuesV, atol=1e-4)
    np.testing.assert_allclose(regDeform.imagePositionPatientV,
                               refDeform.imagePositionPatientV, atol=1e-4)
    np.testing.assert_allclose(regDeform.imageOrientationPatient,
                               refDeform.imageOrientationPatient, atol=1e-6)
    np.testing.assert_allclose(regDeform.dx, refDeform.dx, atol=1e-6)
    np.testing.assert_allclose(regDeform.dy, refDeform.dy, atol=1e-6)
    np.testing.assert_allclose(regDeform.cerrToDcmTransM,
                               refDeform.cerrToDcmTransM, rtol=1e-4, atol=1e-4)

    # Pre/post rigid matrices default to identity on export.
    np.testing.assert_array_equal(
        np.array(regDeform.deformParams['preDeformationMatrix']), np.eye(4))


def test_reg_import_via_loaddcmdir(tmp_path):
    """loadDcmDir picks up a REG file dropped alongside the CT slices."""
    import glob
    import shutil

    planC = pc.loadDcmDir(phantom_dir)
    dvfFile = os.path.join(str(tmp_path), 'dvf.nii.gz')
    _writeSyntheticDvf(planC.scan[0], dvfFile)
    planC = pc.loadNiiVf(dvfFile, 0, planC)

    exportDir = tmp_path / 'reg_dir'
    exportDir.mkdir()
    for f in glob.glob(os.path.join(phantom_dir, 'DCM_IMG_*.dcm')):
        shutil.copy(f, str(exportDir))
    regFile = os.path.join(str(exportDir), 'reg.dcm')
    reg_iod.create(0, regFile, planC, movScanNum=0)

    planC2 = pc.loadDcmDir(str(exportDir))
    assert len(planC2.scan) == 1
    assert len(planC2.deform) == 1
    assert planC2.deform[0].registrationTool == 'dicom'
    # REG references the CT's frame of reference, so the base scan resolves.
    assert planC2.deform[0].baseScanUID == planC2.scan[0].scanUID


def _writeRigidRegFile(regFile, baseFOR, movFOR, matrix4x4):
    """Write a minimal DICOM Spatial Registration (rigid REG, 66.1) file."""
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ImplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.66.1'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(regFile, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.Modality = 'REG'
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesDescription = 'rigid test REG'
    ds.FrameOfReferenceUID = baseFOR

    def _regItem(forUID, m):
        mtx = Dataset()
        mtx.FrameOfReferenceTransformationMatrixType = 'RIGID'
        mtx.FrameOfReferenceTransformationMatrix = \
            [float(v) for v in np.asarray(m).flatten()]
        mtxReg = Dataset()
        mtxReg.MatrixSequence = Sequence([mtx])
        item = Dataset()
        item.FrameOfReferenceUID = forUID
        item.MatrixRegistrationSequence = Sequence([mtxReg])
        return item

    ds.RegistrationSequence = Sequence([
        _regItem(baseFOR, np.eye(4)),
        _regItem(movFOR, matrix4x4),
    ])
    ds.save_as(regFile)


def test_rigid_reg_import_and_warp(tmp_path):
    """Rigid (66.1) REG imports as a matrix and warpScanRigid aligns the scan."""
    from cerr.registration import register

    planC = pc.loadDcmDir(phantom_dir)
    scan3M = planC.scan[0].getScanArray()
    xV, yV, zV = planC.scan[0].getScanXYZVals()

    # Moving scan: fixed scan rolled by (+6 rows, +4 cols) on the same grid.
    rowShift, colShift = 6, 4
    moved3M = np.roll(scan3M, shift=(rowShift, colShift), axis=(0, 1))
    planC = pc.importScanArray(moved3M, xV, yV, zV, 'CT', 0, planC)
    movScanNum = len(planC.scan) - 1

    # True rigid transform (moving -> fixed, LPS mm) for this roll: content at
    # fixed voxel (r, c) sits at moving voxel (r+6, c+4), so the moving->fixed
    # mapping subtracts the shift along the row/col direction cosines.
    colSpacing = 10 * abs(xV[1] - xV[0])   # mm along IOP row cosines (+x)
    rowSpacing = 10 * abs(yV[0] - yV[1])   # mm along IOP col cosines (+y)
    matrix = np.eye(4)
    matrix[0, 3] = -colShift * colSpacing
    matrix[1, 3] = -rowShift * rowSpacing

    baseFOR = planC.scan[0].scanInfo[0].frameOfReferenceUID
    movFOR = '1.2.826.0.1.3680043.8.498.1'  # distinct dummy FOR
    regFile = os.path.join(str(tmp_path), 'rigid_reg.dcm')
    _writeRigidRegFile(regFile, baseFOR, movFOR, matrix)

    numDeform0 = len(planC.deform)
    planC = pc.loadDcmReg(regFile, planC)
    assert len(planC.deform) == numDeform0 + 1
    deformS = planC.deform[-1]
    assert deformS.deformOutFileType == 'rigid'
    assert deformS.registrationTool == 'dicom'
    assert deformS.baseScanUID == planC.scan[0].scanUID
    np.testing.assert_allclose(
        np.array(deformS.deformParams['rigidMatrix']), matrix)
    assert deformS.deformParams['rigidMatrixType'] == 'RIGID'

    # Warp the moving scan back onto the fixed grid.
    planC = register.warpScanRigid(planC, 0, planC, movScanNum, deformS)
    warped3M = planC.scan[-1].getScanArray().astype(float)
    fixed3M = planC.scan[0].getScanArray().astype(float)
    assert warped3M.shape == fixed3M.shape

    # Grid-aligned translation: interior should match essentially exactly.
    m = 12
    np.testing.assert_allclose(warped3M[m:-m, m:-m, :],
                               fixed3M[m:-m, m:-m, :], atol=1e-3)

    # matrixDirection='fixedToMoving' uses the matrix as-is (not inverted), so
    # the warp goes the opposite way and does NOT recover the fixed scan.
    planC = register.warpScanRigid(planC, 0, planC, movScanNum, deformS,
                                   matrixDirection='fixedToMoving')
    reversed3M = planC.scan[-1].getScanArray().astype(float)
    assert not np.allclose(reversed3M[m:-m, m:-m, :],
                           fixed3M[m:-m, m:-m, :], atol=1e-3)
    # It matches the moving scan shifted twice (2x the shift).
    moving3M = planC.scan[movScanNum].getScanArray().astype(float)
    doubled3M = np.roll(moving3M, shift=(rowShift, colShift), axis=(0, 1))
    np.testing.assert_allclose(reversed3M[2 * m:-2 * m, 2 * m:-2 * m, :],
                               doubled3M[2 * m:-2 * m, 2 * m:-2 * m, :],
                               atol=1e-3)


def test_deformable_reg_matrices_only(tmp_path):
    """A 66.3 REG with pre/post matrices but no grid imports as rigid."""
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ImplicitVRLittleEndian, generate_uid

    planC = pc.loadDcmDir(phantom_dir)
    baseFOR = planC.scan[0].scanInfo[0].frameOfReferenceUID
    movFOR = '1.2.826.0.1.3680043.8.498.2'

    preM = np.eye(4)
    preM[:3, 3] = [5.0, -3.0, 2.0]
    postM = np.eye(4)
    postM[:3, 3] = [1.0, 1.0, 0.0]

    regFile = os.path.join(str(tmp_path), 'matrices_only_reg.dcm')
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.66.3'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(regFile, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.Modality = 'REG'
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    baseItem = Dataset()
    baseItem.SourceFrameOfReferenceUID = baseFOR
    movItem = Dataset()
    movItem.SourceFrameOfReferenceUID = movFOR
    for seqName, m in [('PreDeformationMatrixRegistrationSequence', preM),
                       ('PostDeformationMatrixRegistrationSequence', postM)]:
        mtx = Dataset()
        mtx.FrameOfReferenceTransformationMatrixType = 'RIGID'
        mtx.FrameOfReferenceTransformationMatrix = \
            [float(v) for v in m.flatten()]
        setattr(movItem, seqName, Sequence([mtx]))
    ds.DeformableRegistrationSequence = Sequence([baseItem, movItem])
    ds.save_as(regFile)

    planC = pc.loadDcmReg(regFile, planC)
    deformS = planC.deform[-1]
    assert deformS.deformOutFileType == 'rigid'
    assert deformS.baseScanUID == planC.scan[0].scanUID
    np.testing.assert_allclose(
        np.array(deformS.deformParams['rigidMatrix']),
        np.matmul(postM, preM))
