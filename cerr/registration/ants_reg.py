"""ANTs (antspyx) based image registration for pyCERR.

This module registers pyCERR scans using the `antspyx` package. It mirrors the
plastimatch-based workflow in :mod:`cerr.registration.register` (register, then
warp scans / structures / dose), and adds two features that ANTs supports well:

* Fixed and/or moving **masks** to restrict the similarity metric to a region
  of interest (e.g. patient outline, lungs).
* **Landmark**-based initial alignment: a rigid / similarity / affine transform
  is fit to paired points and used as the ``initial_transform`` for the
  subsequent ANTs stages.

The resulting transformation is stored as a composite ITK transform (``.h5``)
in a :class:`cerr.dataclasses.deform.Deform` object appended to ``planC.deform``
(``registrationTool == 'ants'``, ``deformOutFileType == 'ants'``).

antspyx is an optional dependency. Install it with ``pip install antspyx`` or
``pip install "pyCERR[ants]"``.
"""

import os
import shutil
import tempfile

import numpy as np

import cerr.plan_container as pc
from cerr.dataclasses import deform as cerrDeform
from cerr.utils import uid


def _importAnts():
    """Import antspyx lazily with an informative error if it is missing."""
    try:
        import ants
    except ImportError as e:
        raise ImportError(
            "antspyx is required for ANTs registration. Install it with "
            "`pip install antspyx` or `pip install \"pyCERR[ants]\"`."
        ) from e
    return ants


def _asTransformList(transforms):
    """Normalize an ANTs transform result to a list of path strings.

    ``ants.registration`` returns a single string when
    ``write_composite_transform=True`` and a list otherwise.
    """
    if transforms is None:
        return []
    if isinstance(transforms, (list, tuple)):
        return list(transforms)
    return [transforms]


def landmarksToDicomPhysical(landmarksM, scanObj, coordSys='dicom'):
    """Convert landmark points to DICOM physical (LPS, mm) coordinates.

    ANTs operates in the physical (world) space of the images it reads. A nii
    written by pyCERR (:meth:`cerr.dataclasses.scan.Scan.getSitkImage`) encodes
    DICOM physical coordinates, so landmark points must be supplied in - or
    converted to - that same LPS/mm space before being handed to ANTs.

    Args:
        landmarksM (numpy.ndarray): N x 3 array of (x, y, z) landmark points.
        scanObj (cerr.dataclasses.scan.Scan): scan the landmarks belong to;
            used for the pyCERR -> DICOM transform when ``coordSys == 'cerr'``.
        coordSys (str): coordinate system of the input points. ``'dicom'``
            (default) if they are already DICOM physical LPS/mm, or ``'cerr'``
            if they are pyCERR virtual coordinates (x, y, z in cm).

    Returns:
        numpy.ndarray: N x 3 array of points in DICOM physical LPS/mm.
    """
    landmarksM = np.asarray(landmarksM, dtype=float)
    if landmarksM.ndim != 2 or landmarksM.shape[1] != 3:
        raise ValueError("Landmarks must be an N x 3 array of (x, y, z) points.")
    if coordSys.lower() == 'dicom':
        return landmarksM
    if coordSys.lower() == 'cerr':
        numPts = landmarksM.shape[0]
        homogM = np.hstack((landmarksM, np.ones((numPts, 1))))
        dcmM = np.matmul(scanObj.cerrToDcmTransM, homogM.T).T[:, :3]
        return dcmM
    raise ValueError("landmarkCoordSys must be 'dicom' or 'cerr'.")


def _fitLandmarkTransform(baseLandmarksM, movLandmarksM, baseScanObj,
                          movScanObj, coordSys, transformType, dirpath):
    """Fit an initial transform from paired landmarks and write it to disk.

    The transform maps moving-image physical points onto fixed-image physical
    points, matching the direction ANTs expects for ``initial_transform``.

    Returns:
        str: path to the written ITK transform (.mat) file.
    """
    ants = _importAnts()

    baseDcmM = landmarksToDicomPhysical(baseLandmarksM, baseScanObj, coordSys)
    movDcmM = landmarksToDicomPhysical(movLandmarksM, movScanObj, coordSys)
    if baseDcmM.shape[0] != movDcmM.shape[0]:
        raise ValueError("Fixed and moving landmark arrays must have the same "
                         "number of points.")

    # fit_transform_to_paired_points(moving, fixed) returns an ITK transform
    # mapping fixed -> moving physical points, i.e. the convention ANTs expects
    # for an initial_transform that resamples the moving image into fixed space.
    # Pass numpy arrays (not DataFrames) - the DataFrame path mishandles the
    # transform center in antspyx.
    initTx = ants.fit_transform_to_paired_points(movDcmM, baseDcmM,
                                                 transform_type=transformType)
    initTxFile = os.path.join(dirpath, 'landmark_init.mat')
    ants.write_transform(initTx, initTxFile)
    return initTxFile


def registerScansAnts(basePlanC, baseScanIndex, movPlanC, movScanIndex,
                      transformSaveDir=None,
                      typeOfTransform='antsRegistrationSyNQuick[s]',
                      baseMask3M=None, movMask3M=None, maskAllStages=True,
                      baseLandmarksM=None, movLandmarksM=None,
                      landmarkCoordSys='dicom', landmarkTransformType='rigid',
                      initialTransform=None,
                      outputFilePrefix='ants_reg',
                      antsRegistrationKwargs=None):
    """Register a moving scan to a fixed scan using ANTs (antspyx).

    Args:
        basePlanC (cerr.plan_container.PlanC): plan container with the fixed
            (target) scan.
        baseScanIndex (int): index of the fixed scan in ``basePlanC.scan``.
        movPlanC (cerr.plan_container.PlanC): plan container with the moving
            scan. May be the same object as ``basePlanC``.
        movScanIndex (int): index of the moving scan in ``movPlanC.scan``.
        transformSaveDir (str): directory in which to write the composite
            transform files. If ``None``, a persistent temporary directory is
            created and its path stored on the deform object.
        typeOfTransform (str): ANTs transform preset passed to
            ``ants.registration`` (e.g. ``'Rigid'``, ``'Affine'``, ``'SyN'``,
            ``'antsRegistrationSyNQuick[s]'``, ``'antsRegistrationSyN[bo]'``).
        baseMask3M (numpy.ndarray): optional 3D boolean mask on the fixed scan
            grid restricting the similarity metric.
        movMask3M (numpy.ndarray): optional 3D boolean mask on the moving scan
            grid.
        maskAllStages (bool): if True, apply the masks at every registration
            stage (affine and deformable), not just the final stage.
        baseLandmarksM (numpy.ndarray): optional N x 3 fixed-scan landmark
            points for initial alignment.
        movLandmarksM (numpy.ndarray): optional N x 3 moving-scan landmark
            points (paired with ``baseLandmarksM``).
        landmarkCoordSys (str): coordinate system of the landmark points,
            ``'dicom'`` (LPS/mm) or ``'cerr'`` (virtual x, y, z in cm).
        landmarkTransformType (str): transform to fit to the landmarks,
            one of ``'rigid'``, ``'similarity'``, ``'affine'``.
        initialTransform: optional initial transform passed to
            ``ants.registration`` (path to an ITK transform, a list of paths,
            or ``'identity'``). Ignored when landmarks are supplied.
        outputFilePrefix (str): filename prefix for the written transforms.
        antsRegistrationKwargs (dict): optional extra keyword arguments
            forwarded to ``ants.registration``.

    Returns:
        cerr.plan_container.PlanC: ``basePlanC`` with the warped moving scan
        appended to ``basePlanC.scan`` and a new ``Deform`` object appended to
        ``basePlanC.deform``.
    """
    ants = _importAnts()

    if antsRegistrationKwargs is None:
        antsRegistrationKwargs = {}

    if transformSaveDir is None:
        transformSaveDir = tempfile.mkdtemp()
    else:
        os.makedirs(transformSaveDir, exist_ok=True)

    # Write fixed/moving scans (and masks) to nii in a temporary work dir.
    dirpath = tempfile.mkdtemp()
    fixedImgNii = os.path.join(dirpath, 'fixed.nii.gz')
    movingImgNii = os.path.join(dirpath, 'moving.nii.gz')
    basePlanC.scan[baseScanIndex].saveNii(fixedImgNii)
    movPlanC.scan[movScanIndex].saveNii(movingImgNii)

    imgAntsBase = ants.image_read(fixedImgNii)
    imgAntsMov = ants.image_read(movingImgNii)

    maskAntsBase = None
    maskAntsMov = None
    if baseMask3M is not None:
        fixedMaskNii = os.path.join(dirpath, 'fixed_mask.nii.gz')
        basePlanC = pc.importStructureMask(baseMask3M, baseScanIndex, 'mask', basePlanC)
        basePlanC.structure[-1].saveNii(fixedMaskNii, basePlanC)
        del basePlanC.structure[-1]
        maskAntsBase = ants.image_read(fixedMaskNii)
    if movMask3M is not None:
        movingMaskNii = os.path.join(dirpath, 'moving_mask.nii.gz')
        movPlanC = pc.importStructureMask(movMask3M, movScanIndex, 'mask', movPlanC)
        movPlanC.structure[-1].saveNii(movingMaskNii, movPlanC)
        del movPlanC.structure[-1]
        maskAntsMov = ants.image_read(movingMaskNii)

    # Build the initial transform: landmarks take precedence over an explicitly
    # supplied initial_transform.
    hasLandmarks = baseLandmarksM is not None and movLandmarksM is not None
    if hasLandmarks:
        initTxFile = _fitLandmarkTransform(
            baseLandmarksM, movLandmarksM,
            basePlanC.scan[baseScanIndex], movPlanC.scan[movScanIndex],
            landmarkCoordSys, landmarkTransformType, transformSaveDir)
        initialTransform = [initTxFile]

    txPath = os.path.join(transformSaveDir, outputFilePrefix)
    regResult = ants.registration(
        fixed=imgAntsBase, moving=imgAntsMov, type_of_transform=typeOfTransform,
        initial_transform=initialTransform,
        mask=maskAntsBase, moving_mask=maskAntsMov,
        mask_all_stages=maskAllStages,
        write_composite_transform=True, outprefix=txPath,
        **antsRegistrationKwargs)

    # Warp the moving scan and add it to the fixed plan container.
    warpedScanImage = ants.apply_transforms(
        fixed=imgAntsBase, moving=imgAntsMov,
        transformlist=regResult['fwdtransforms'], interpolator='linear')
    warpedScanNii = os.path.join(dirpath, 'warped_scan.nii.gz')
    ants.image_write(warpedScanImage, warpedScanNii)
    imageType = movPlanC.scan[movScanIndex].scanInfo[0].imageType
    basePlanC = pc.loadNiiScan(warpedScanNii, imageType, '', basePlanC)

    # The forward composite maps points from fixed -> moving; it is the
    # transform used to resample moving images into fixed space.
    # With write_composite_transform=True, ANTs returns each transform as a
    # single path string (not a list), so normalize carefully - list(str)
    # would split the path into characters.
    fwdTransforms = _asTransformList(regResult['fwdtransforms'])
    invTransforms = _asTransformList(regResult['invtransforms'])

    deform = cerrDeform.Deform()
    deform.deformUID = uid.createUID("deform")
    deform.baseScanUID = basePlanC.scan[baseScanIndex].scanUID
    deform.movScanUID = movPlanC.scan[movScanIndex].scanUID
    deform.deformOutFileType = 'ants'
    deform.deformOutFilePath = fwdTransforms[0]
    deform.registrationTool = 'ants'
    deform.algorithm = typeOfTransform
    deform.algorithmParams = dict(antsRegistrationKwargs)
    deform.deformParams = {
        'fwdtransforms': fwdTransforms,
        'invtransforms': invTransforms,
        'inverseTransform': invTransforms[0] if invTransforms else '',
        'usedLandmarks': bool(hasLandmarks),
        'landmarkTransformType': landmarkTransformType if hasLandmarks else '',
    }
    basePlanC.deform.append(deform)

    shutil.rmtree(dirpath, ignore_errors=True)

    return basePlanC


def _applyAntsTransform(basePlanC, baseScanIndex, movingNii, deformS,
                        interpolator, inverse=False):
    """Resample a moving nii into fixed space using a stored ANTs transform.

    Returns:
        str: path to the warped nii file (inside a temp dir the caller owns).
    """
    ants = _importAnts()
    dirpath = os.path.dirname(movingNii)
    fixedImgNii = os.path.join(dirpath, 'ref.nii.gz')
    basePlanC.scan[baseScanIndex].saveNii(fixedImgNii)

    if inverse:
        transformList = deformS.deformParams.get('invtransforms',
                                                 [deformS.deformParams.get('inverseTransform', '')])
    else:
        transformList = deformS.deformParams.get('fwdtransforms',
                                                 [deformS.deformOutFilePath])

    fixedImg = ants.image_read(fixedImgNii)
    movingImg = ants.image_read(movingNii)
    warpedImg = ants.apply_transforms(fixed=fixedImg, moving=movingImg,
                                      transformlist=transformList,
                                      interpolator=interpolator)
    warpedNii = os.path.join(dirpath, 'warped.nii.gz')
    ants.image_write(warpedImg, warpedNii)
    return warpedNii


def warpScanAnts(basePlanC, baseScanIndex, movPlanC, movScanIndex, deformS,
                 interpolator='linear'):
    """Warp a moving scan into fixed space using a stored ANTs transform.

    Args:
        basePlanC (cerr.plan_container.PlanC): plan container with the fixed scan.
        baseScanIndex (int): index of the fixed scan in ``basePlanC.scan``.
        movPlanC (cerr.plan_container.PlanC): plan container with the moving scan.
        movScanIndex (int): index of the moving scan in ``movPlanC.scan``.
        deformS (cerr.dataclasses.deform.Deform): ANTs deform object
            (``registrationTool == 'ants'``).
        interpolator (str): ANTs interpolator (default ``'linear'``).

    Returns:
        cerr.plan_container.PlanC: ``basePlanC`` with the warped scan appended.
    """
    dirpath = tempfile.mkdtemp()
    movingNii = os.path.join(dirpath, 'moving.nii.gz')
    movPlanC.scan[movScanIndex].saveNii(movingNii)
    warpedNii = _applyAntsTransform(basePlanC, baseScanIndex, movingNii,
                                    deformS, interpolator)
    imageType = movPlanC.scan[movScanIndex].scanInfo[0].imageType
    basePlanC = pc.loadNiiScan(warpedNii, imageType, '', basePlanC)
    shutil.rmtree(dirpath, ignore_errors=True)
    return basePlanC


def warpStructuresAnts(basePlanC, baseScanIndex, movPlanC, movStrNumV, deformS,
                       interpolator='genericLabel'):
    """Warp moving structures into fixed space using a stored ANTs transform.

    Args:
        basePlanC (cerr.plan_container.PlanC): plan container with the fixed scan.
        baseScanIndex (int): index of the fixed scan in ``basePlanC.scan``.
        movPlanC (cerr.plan_container.PlanC): plan container with the moving
            structures.
        movStrNumV (list): indices of moving structures in ``movPlanC.structure``.
        deformS (cerr.dataclasses.deform.Deform): ANTs deform object.
        interpolator (str): ANTs interpolator for labels (default
            ``'genericLabel'``; ``'nearestNeighbor'`` is also common).

    Returns:
        cerr.plan_container.PlanC: ``basePlanC`` with the warped structures appended.
    """
    if not isinstance(movStrNumV, (list, np.ndarray)):
        movStrNumV = [movStrNumV]
    dirpath = tempfile.mkdtemp()
    movingNii = os.path.join(dirpath, 'structure.nii.gz')
    for strNum in movStrNumV:
        structName = movPlanC.structure[strNum].structureName
        movPlanC.structure[strNum].saveNii(movingNii, movPlanC)
        warpedNii = _applyAntsTransform(basePlanC, baseScanIndex, movingNii,
                                        deformS, interpolator)
        # loadNiiStructure expects labels_dict as {structureName: label}; the
        # saved structure nii is a binary mask with label value 1.
        basePlanC = pc.loadNiiStructure(warpedNii, baseScanIndex, basePlanC,
                                        {structName: 1})
    shutil.rmtree(dirpath, ignore_errors=True)
    return basePlanC


def warpDoseAnts(basePlanC, baseScanIndex, movPlanC, movDoseIndexV, deformS,
                 interpolator='linear'):
    """Warp moving dose distributions into fixed space using an ANTs transform.

    Args:
        basePlanC (cerr.plan_container.PlanC): plan container with the fixed scan.
        baseScanIndex (int): index of the fixed scan in ``basePlanC.scan``.
        movPlanC (cerr.plan_container.PlanC): plan container with the moving dose.
        movDoseIndexV (int or list): indices of moving dose in ``movPlanC.dose``.
        deformS (cerr.dataclasses.deform.Deform): ANTs deform object.
        interpolator (str): ANTs interpolator (default ``'linear'``).

    Returns:
        cerr.plan_container.PlanC: ``basePlanC`` with the warped dose appended.
    """
    if not isinstance(movDoseIndexV, (list, np.ndarray)):
        movDoseIndexV = [movDoseIndexV]
    dirpath = tempfile.mkdtemp()
    movingNii = os.path.join(dirpath, 'dose.nii.gz')
    for movDoseIndex in movDoseIndexV:
        doseName = movPlanC.dose[movDoseIndex].fractionGroupID
        doseUnits = movPlanC.dose[movDoseIndex].doseUnits
        movPlanC.dose[movDoseIndex].saveNii(movingNii)
        warpedNii = _applyAntsTransform(basePlanC, baseScanIndex, movingNii,
                                        deformS, interpolator)
        basePlanC = pc.loadNiiDose(warpedNii, baseScanIndex, basePlanC, doseName)
        basePlanC.dose[-1].imageOrientationPatient = \
            basePlanC.scan[baseScanIndex].scanInfo[0].imageOrientationPatient
        basePlanC.dose[-1].doseUnits = doseUnits
    shutil.rmtree(dirpath, ignore_errors=True)
    return basePlanC
