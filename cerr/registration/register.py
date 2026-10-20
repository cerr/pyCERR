"""The register module defines routines for registering and deforming scans, structures and dose distributions.
It also provides routines for calculating the vector field and Jacobian of a deformation.

Registration uses ANTs (:mod:`cerr.registration.ants_reg`; ``pip install "pycerr[ants]"``).
plastimatch is still supported through ``registrationTool='plastimatch'`` but is deprecated.
"""

import os
import tempfile
import shutil
from cerr.utils import uid
from cerr.dataclasses import deform as cerrDeform
from cerr.dataclasses import scan as scn
from cerr.dataclasses import structure as cerrStr
import cerr.plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.utils.mask import getSurfacePoints, computeBoundingBox
from cerr.utils.interp import finterp3
from cerr.radiomics import preprocess
import numpy as np
import subprocess
import warnings

# ANTs (antspyx) based registration lives in a companion module; re-export the
# public entry points so they are reachable as cerr.registration.register.*.
# The module does not import antspyx at import time, so this stays lightweight.
from cerr.registration.ants_reg import (
    registerScansAnts,
    warpScanAnts,
    warpStructuresAnts,
    warpDoseAnts,
    landmarksToDicomPhysical,
    calcVectorFieldAnts,
)


#: ANTs transform presets used for the ``deforAlgorithm`` names of :func:`registerScans`.
#: Any other value is passed to ``ants.registration`` as the ``type_of_transform``.
ANTS_TRANSFORMS = {
    'rigid': 'Rigid',
    'affine': 'Affine',
    'bsplines': 'antsRegistrationSyNQuick[s]',
    'deformable': 'antsRegistrationSyNQuick[s]',
}

_PLASTIMATCH_DEPRECATION = ("plastimatch support in cerr.registration is deprecated and will be "
                            "removed in a future release; ANTs is the default "
                            "(pip install \"pycerr[ants]\").")


def _isAntsDeform(deformS):
    return (deformS.registrationTool or '').lower() == 'ants' \
        or deformS.deformOutFileType == 'ants'


_ITK_FIELD_EXTENSIONS = ('.nii', '.nii.gz', '.mha', '.mhd', '.nrrd')


def _antsCanWarp(deformS):
    """True for an ANTs transform, or a vector field stored in a file ITK reads."""
    if _isAntsDeform(deformS):
        return True
    path = deformS.deformOutFilePath or ''
    return deformS.deformOutFileType == 'vf' and path.lower().endswith(_ITK_FIELD_EXTENSIONS) \
        and os.path.exists(path)


def _isRigidDeform(deformS):
    return deformS.deformOutFileType == 'rigid'


def _warnPlastimatch():
    warnings.warn(_PLASTIMATCH_DEPRECATION, FutureWarning, stacklevel=3)


def registerScans(basePlanC, baseScanIndex, movPlanC, movScanIndex, transformSaveDir,
                  deforAlgorithm='bsplines', registrationTool='ants',
                  baseMask3M=None, movMask3M=None, inputCmdFile=None, outputFilePrefix=None):
    """Routine to register a moving scan to a fixed scan. The warped moving scan
    and a deformation object are added to basePlanC.

    Registration uses ANTs (:func:`cerr.registration.ants_reg.registerScansAnts`),
    which needs the ``ants`` extra (``pip install "pycerr[ants]"``).
    ``registrationTool='plastimatch'`` is deprecated.

    Args:
        basePlanC (cerr.plan_container.PlanC): pyCERR plan container containing fixed target scan
        baseScanIndex (int): integer, identifies target scan in basePlanC
        movPlanC (cerr.plan_container.PlanC): pyCERR plan container containing moving scan
        movScanIndex (int): integer, identifies moving scan in movPlanC
        transformSaveDir (str): Directory to save transformation file
        deforAlgorithm (str): 'bsplines' (default; deformable), 'deformable', 'affine' or 'rigid'.
            For ANTs, any other value is used as the ANTs transform preset, e.g. 'SyN' or
            'antsRegistrationSyN[s]' (see ``ANTS_TRANSFORMS``). For plastimatch it selects the
            command file from cerr/registration/settings when inputCmdFile is not given.
        registrationTool (str): 'ants' (default) or 'plastimatch' (deprecated).
        baseMask3M (numpy.ndarray): optional, 3D binary mask in target space
        movMask3M (numpy.ndarray): optional, 3D binary mask in moving space
        inputCmdFile (str): optional, path to a plastimatch command file. Not used with ANTs.
        outputFilePrefix (str): optional, prefix for the transform file names

    Returns:
        cerr.plan_container.PlanC: plan container object basePlanC with an element added to planC.deform attribute

    Example:
        Deformable registration of scan 1 to scan 0, then propagate a structure::

            from cerr.registration import register

            planC = register.registerScans(planC, 0, planC, 1, '/path/to/transforms')
            deformS = planC.deform[-1]
            planC = register.warpStructures(planC, 0, planC, [movStructNum], deformS)

    """
    if registrationTool.lower() == 'ants':
        if inputCmdFile is not None:
            warnings.warn("inputCmdFile is a plastimatch option and is ignored with ANTs; "
                          "pass an ANTs preset as deforAlgorithm instead.", stacklevel=2)
        typeOfTransform = ANTS_TRANSFORMS.get(str(deforAlgorithm).lower(), deforAlgorithm)
        return registerScansAnts(basePlanC, baseScanIndex, movPlanC, movScanIndex,
                                 transformSaveDir=transformSaveDir,
                                 typeOfTransform=typeOfTransform,
                                 baseMask3M=baseMask3M, movMask3M=movMask3M,
                                 outputFilePrefix=outputFilePrefix or 'ants_reg')
    if registrationTool.lower() != 'plastimatch':
        raise ValueError("registrationTool must be 'ants' or 'plastimatch', not %r"
                         % registrationTool)
    _warnPlastimatch()

    # create temporary directory to hold registration files
    dirpath = tempfile.mkdtemp()

    # Write nii files for base and moving scans in dirpath
    moving_img_nii = os.path.join(dirpath, 'moving.nii.gz')
    fixed_img_nii = os.path.join(dirpath, 'fixed.nii.gz')
    moving_mask_nii = os.path.join(dirpath, 'moving_mask.nii.gz')
    fixed_mask_nii = os.path.join(dirpath, 'fixed_mask.nii.gz')
    warped_img_nii = os.path.join(dirpath, 'warped_moving.nii.gz')
    basePlanC.scan[baseScanIndex].saveNii(fixed_img_nii)
    movPlanC.scan[movScanIndex].saveNii(moving_img_nii)
    if baseMask3M is not None:
        basePlanC = pc.importStructureMask(baseMask3M, baseScanIndex, 'mask', basePlanC)
        maskStrNum = len(basePlanC.structure) - 1
        #pc.saveNiiStructure(fixed_mask_nii, maskStrNum, basePlanC)
        basePlanC.structure[maskStrNum].saveNii(fixed_mask_nii, basePlanC)
        del basePlanC.structure[-1]
    if movMask3M is not None:
        movPlanC = pc.importStructureMask(movMask3M, movScanIndex, 'mask', movPlanC)
        maskStrNum = len(movPlanC.structure) - 1
        #pc.saveNiiStructure(moving_mask_nii, maskStrNum, movPlanC)
        movPlanC.structure[maskStrNum].saveNii(moving_mask_nii, movPlanC)
        del movPlanC.structure[-1]

    if inputCmdFile is None or not os.path.exists(inputCmdFile):
        if baseMask3M is not None and movMask3M is not None:
            if deforAlgorithm == 'affine':
                plmCmdFile = 'plastimatch_ct_ct_intra_pt_w_masks_affine.txt'
            elif deforAlgorithm == 'bsplines':
                plmCmdFile = 'plastimatch_ct_ct_intra_pt_w_masks_bsplines.txt'
        else:
            if deforAlgorithm == 'affine':
                plmCmdFile = 'plastimatch_ct_ct_intra_pt_affine.txt'
            elif deforAlgorithm == 'bsplines':
                plmCmdFile = 'plastimatch_ct_ct_intra_pt_bsplines.txt'

        regDir = os.path.dirname(os.path.abspath(__file__))
        inputCmdFile = os.path.join(regDir,'settings',plmCmdFile)
        #cmdFilePathDest = os.path.join(dirpath, plmCmdFile)
        #shutil.copyfile(cmdFilePathSrc, cmdFilePathDest)

    # Read xform_out name
    xform_out = ''
    cmdFileObj = open(inputCmdFile, 'r')
    for line in cmdFileObj:
        if 'xform_out' in line:
            xform_out = line.rsplit('=')
            if len(xform_out) == 2:
                xform_out = xform_out[1].strip()
            break

    deformOutFileType = '' # plm_bspline_coeffs

    # Filename to save bsplines coeffficients
    bspSourcePath = os.path.join(dirpath, xform_out)
    if outputFilePrefix is not None:
        bspDestPath = os.path.join(transformSaveDir, outputFilePrefix + '_' + xform_out)
    else:
        bspDestPath = os.path.join(transformSaveDir, xform_out)

    # Command string to call registration tool
    if registrationTool.lower() == 'plastimatch':
        plm_reg_cmd = "plastimatch register " + inputCmdFile

    currDir = os.getcwd()
    os.chdir(dirpath)
    #os.system(plm_reg_cmd)
    sts = subprocess.Popen(plm_reg_cmd, shell=True).wait()
    os.chdir(currDir)

    # Add warped scan to planC
    imageType = movPlanC.scan[movScanIndex].scanInfo[0].imageType
    direction = ''
    basePlanC = pc.loadNiiScan(warped_img_nii, imageType, direction, basePlanC)

    # Copy output to the user-specified directory
    shutil.copyfile(bspSourcePath, bspDestPath)

    # Create a deform object and add to planC
    deform = cerrDeform.Deform()
    deform.deformUID = uid.createUID("deform")
    deform.baseScanUID = basePlanC.scan[baseScanIndex].scanUID
    deform.movScanUID = movPlanC.scan[movScanIndex].scanUID
    deform.deformOutFileType = deformOutFileType
    deform.deformOutFilePath = bspDestPath
    deform.registrationTool = registrationTool
    deform.algorithm = deforAlgorithm

    # Append to base planc
    basePlanC.deform.append(deform)

    # Remove temporary directory
    shutil.rmtree(dirpath)

    return basePlanC


def warpScan(basePlanC, baseScanIndex, movPlanC, movScanIndex, deformS):
    """Routine to deform moving scan to fixed scan based on input transformation

    Args:
        basePlanC (cerr.plan_container.PlanC): pyCERR plan container containing fixed target scan
        baseScanIndex (int): index of fixed scan in planC.scan
        movPlanC (cerr.plan_container.PlanC): pyCERR plan container containing moving scan
        movScanIndex (int): index of moving scan in planC.scan
        deformS (cerr.dataclasses.deform.Deform): pyCERR's deformation transformation object

    Returns:
        cerr.plan_container.PlanC: plan container object basePlanC with the deformed scan added to planC.scan attribute

    ANTs transforms and vector fields stored as NIfTI/MHA files are applied with ANTs,
    rigid matrices directly, and plastimatch B-spline transforms with plastimatch
    (deprecated).
    """
    if _antsCanWarp(deformS):
        return warpScanAnts(basePlanC, baseScanIndex, movPlanC, movScanIndex, deformS)
    if _isRigidDeform(deformS):
        return warpScanRigid(basePlanC, baseScanIndex, movPlanC, movScanIndex, deformS)
    _warnPlastimatch()

    dirpath = tempfile.mkdtemp()
    fixed_img_nii = os.path.join(dirpath, 'ref.nii.gz')
    moving_img_nii = os.path.join(dirpath, 'ctmoving.nii.gz')
    warped_img_nii = os.path.join(dirpath, 'warped.nii.gz')
    bsplines_coeff_file = deformS.deformOutFilePath
    basePlanC.scan[baseScanIndex].saveNii(fixed_img_nii)
    movPlanC.scan[movScanIndex].saveNii(moving_img_nii)


    plm_warp_str_cmd = "plastimatch warp --input " + moving_img_nii + \
                  " --output-img " + warped_img_nii + \
                  " --xf " + bsplines_coeff_file + \
                  " --fixed " + fixed_img_nii

    currDir = os.getcwd()
    os.chdir(dirpath)
    os.system(plm_warp_str_cmd)
    os.chdir(currDir)

    imageType = movPlanC.scan[movScanIndex].scanInfo[0].imageType
    direction = ''
    basePlanC = pc.loadNiiScan(warped_img_nii, imageType, direction, basePlanC)

    # Remove temporary directory
    shutil.rmtree(dirpath)

    return basePlanC


def _applyRigidToNii(movingNii, warpedNii, fixedNii, deformS, interpolator,
                     matrixDirection='movingToFixed'):
    """Resample a moving nii onto the fixed grid using a stored rigid matrix.

    SimpleITK's resampling transform maps output (fixed) points to input
    (moving) points, i.e. it needs the fixed -> moving matrix.

    Args:
        matrixDirection (str): direction of the stored matrix.
            ``'movingToFixed'`` (default) follows the DICOM Spatial Registration
            convention (moving frame -> base/fixed frame), so its inverse is
            used as the sitk transform. ``'fixedToMoving'`` is for vendor files
            written in the opposite direction; the matrix is used as-is.
    """
    import SimpleITK as sitk

    if deformS.deformOutFileType != 'rigid':
        raise ValueError("deformS does not hold a rigid registration "
                         "(deformOutFileType != 'rigid').")
    if matrixDirection not in ('movingToFixed', 'fixedToMoving'):
        raise ValueError("matrixDirection must be 'movingToFixed' or "
                         "'fixedToMoving', got " + str(matrixDirection))
    matrix = np.array(deformS.deformParams['rigidMatrix'], dtype=float)

    fixedImg = sitk.ReadImage(fixedNii)
    movingImg = sitk.ReadImage(movingNii)

    # sitk needs the fixed -> moving matrix.
    fixedToMoving = matrix if matrixDirection == 'fixedToMoving' \
        else np.linalg.inv(matrix)
    tfm = sitk.AffineTransform(3)
    tfm.SetMatrix(fixedToMoving[:3, :3].ravel())
    tfm.SetTranslation(fixedToMoving[:3, 3])

    interpMap = {'linear': sitk.sitkLinear,
                 'nearestNeighbor': sitk.sitkNearestNeighbor,
                 'bspline': sitk.sitkBSpline}
    if interpolator not in interpMap:
        raise ValueError("Unsupported interpolator: " + str(interpolator))
    bgVal = float(sitk.GetArrayViewFromImage(movingImg).min())
    warpedImg = sitk.Resample(movingImg, fixedImg, tfm,
                              interpMap[interpolator], bgVal,
                              movingImg.GetPixelID())
    sitk.WriteImage(warpedImg, warpedNii)


def warpScanRigid(basePlanC, baseScanIndex, movPlanC, movScanIndex, deformS,
                  interpolator='linear', matrixDirection='movingToFixed'):
    """Warp a moving scan onto a fixed scan grid using a rigid registration.

    Applies the 4x4 matrix stored on a rigid deform object (e.g. imported from
    a DICOM Spatial Registration REG file via
    :func:`cerr.plan_container.loadDcmReg`).

    Args:
        basePlanC (cerr.plan_container.PlanC): plan container with the fixed scan.
        baseScanIndex (int): index of the fixed scan in ``basePlanC.scan``.
        movPlanC (cerr.plan_container.PlanC): plan container with the moving scan.
        movScanIndex (int): index of the moving scan in ``movPlanC.scan``.
        deformS (cerr.dataclasses.deform.Deform): rigid deform object
            (``deformOutFileType == 'rigid'``).
        interpolator (str): 'linear' (default), 'nearestNeighbor' or 'bspline'.
        matrixDirection (str): direction of the stored matrix,
            ``'movingToFixed'`` (default, DICOM convention) or
            ``'fixedToMoving'`` (for vendor files written the other way, or if
            the warp comes out reversed).

    Returns:
        cerr.plan_container.PlanC: ``basePlanC`` with the warped scan appended.
    """
    dirpath = tempfile.mkdtemp()
    fixedNii = os.path.join(dirpath, 'ref.nii.gz')
    movingNii = os.path.join(dirpath, 'moving.nii.gz')
    warpedNii = os.path.join(dirpath, 'warped.nii.gz')
    basePlanC.scan[baseScanIndex].saveNii(fixedNii)
    movPlanC.scan[movScanIndex].saveNii(movingNii)

    _applyRigidToNii(movingNii, warpedNii, fixedNii, deformS, interpolator,
                     matrixDirection)

    imageType = movPlanC.scan[movScanIndex].scanInfo[0].imageType
    basePlanC = pc.loadNiiScan(warpedNii, imageType, '', basePlanC)

    shutil.rmtree(dirpath)
    return basePlanC


def warpStructuresRigid(basePlanC, baseScanIndex, movPlanC, movStrNumV,
                        deformS, matrixDirection='movingToFixed'):
    """Warp moving structures onto a fixed scan grid using a rigid registration.

    Args:
        basePlanC (cerr.plan_container.PlanC): plan container with the fixed scan.
        baseScanIndex (int): index of the fixed scan in ``basePlanC.scan``.
        movPlanC (cerr.plan_container.PlanC): plan container with the moving
            structures.
        movStrNumV (list): indices of moving structures in ``movPlanC.structure``.
        deformS (cerr.dataclasses.deform.Deform): rigid deform object
            (``deformOutFileType == 'rigid'``).
        matrixDirection (str): direction of the stored matrix,
            ``'movingToFixed'`` (default, DICOM convention) or
            ``'fixedToMoving'``.

    Returns:
        cerr.plan_container.PlanC: ``basePlanC`` with the warped structures appended.
    """
    if not isinstance(movStrNumV, (list, np.ndarray)):
        movStrNumV = [movStrNumV]
    dirpath = tempfile.mkdtemp()
    fixedNii = os.path.join(dirpath, 'ref.nii.gz')
    movingNii = os.path.join(dirpath, 'structure.nii.gz')
    warpedNii = os.path.join(dirpath, 'warped.nii.gz')
    basePlanC.scan[baseScanIndex].saveNii(fixedNii)
    for strNum in movStrNumV:
        structName = movPlanC.structure[strNum].structureName
        movPlanC.structure[strNum].saveNii(movingNii, movPlanC)
        _applyRigidToNii(movingNii, warpedNii, fixedNii, deformS,
                         'nearestNeighbor', matrixDirection)
        basePlanC = pc.loadNiiStructure(warpedNii, baseScanIndex, basePlanC,
                                        {structName: 1})
    shutil.rmtree(dirpath)
    return basePlanC


def warpDose(basePlanC, baseScanIndex, movPlanC, movDoseIndexV, deformS):
    """Routine to deform moving dose to fixed scan based on input transformation

    Args:
        basePlanC (cerr.plan_container.PlanC): pyCERR plan container containing fixed target scan
        baseScanIndex (int): index of fixed scan in planC.scan
        movPlanC (cerr.plan_container.PlanC): pyCERR plan container containing moving scan
        movDoseIndexV (int): index of moving dose in planC.dose
        deformS (cerr.dataclasses.deform.Deform): pyCERR's deformation transformation object

    Returns:
        cerr.plan_container.PlanC: plan container object basePlanC with the deformed dose added to planC.dose attribute

    ANTs transforms and vector fields stored as NIfTI/MHA files are applied with ANTs;
    plastimatch B-spline transforms with plastimatch (deprecated).
    """
    if _antsCanWarp(deformS):
        return warpDoseAnts(basePlanC, baseScanIndex, movPlanC, movDoseIndexV, deformS)
    _warnPlastimatch()

    if not isinstance(movDoseIndexV, (list, np.ndarray)):
        movDoseIndexV = [movDoseIndexV]

    dirpath = tempfile.mkdtemp()
    fixed_img_nii = os.path.join(dirpath, 'ref.nii.gz')
    moving_dose_nii = os.path.join(dirpath, 'dose.nii.gz')
    warped_dose_nii = os.path.join(dirpath, 'warped.nii.gz')
    bsplines_coeff_file = deformS.deformOutFilePath
    basePlanC.scan[baseScanIndex].saveNii(fixed_img_nii)
    currDir = os.getcwd()
    os.chdir(dirpath)
    for movDoseIndex in movDoseIndexV:
        doseName = movPlanC.dose[movDoseIndex].fractionGroupID
        doseUnits = movPlanC.dose[movDoseIndex].doseUnits
        movPlanC.dose[movDoseIndex].saveNii(moving_dose_nii)
        plm_warp_str_cmd = "plastimatch warp --input " + moving_dose_nii + \
                      " --output-img " + warped_dose_nii + \
                      " --xf " + bsplines_coeff_file + \
                      " --fixed " + fixed_img_nii + \
                      " --interpolation linear"
        os.system(plm_warp_str_cmd)
        basePlanC = pc.loadNiiDose(warped_dose_nii, baseScanIndex, basePlanC, doseName)
        basePlanC.dose[-1].imageOrientationPatient = basePlanC.scan[baseScanIndex].scanInfo[0].imageOrientationPatient
        basePlanC.dose[-1].doseUnits = doseUnits

    os.chdir(currDir)

    return basePlanC


def warpStructures(basePlanC, baseScanIndex, movPlanC, movStrNumV, deformS):
    """Routine to deform moving structures to fixed scan based on input transformation

    Args:
        basePlanC (cerr.plan_container.PlanC): pyCERR plan container containing fixed target scan
        baseScanIndex (int): index of fixed scan in planC.scan
        movPlanC (cerr.plan_container.PlanC): pyCERR plan container containing moving scan
        movStrNumV (int): list of indices of moving structures in planC.structure
        deformS (cerr.dataclasses.deform.Deform): pyCERR's deformation transformation object

    Returns:
        cerr.plan_container.PlanC: plan container object basePlanC with the deformed structures added to planC.structure attribute

    ANTs transforms and vector fields stored as NIfTI/MHA files are applied with ANTs,
    rigid matrices directly, and plastimatch B-spline transforms with plastimatch
    (deprecated).
    """
    if _antsCanWarp(deformS):
        return warpStructuresAnts(basePlanC, baseScanIndex, movPlanC, movStrNumV, deformS)
    if _isRigidDeform(deformS):
        return warpStructuresRigid(basePlanC, baseScanIndex, movPlanC, movStrNumV, deformS)
    _warnPlastimatch()

    # dirpath = tempfile.mkdtemp()
    # rtst_warped_path = os.path.join(dirpath, 'struct.nii.gz')
    dirpath = tempfile.mkdtemp()
    fixed_img_nii = os.path.join(dirpath, 'ref.nii.gz')
    moving_str_nii = os.path.join(dirpath, 'structure.nii.gz')
    warped_str_nii = os.path.join(dirpath, 'warped.nii.gz')
    bsplines_coeff_file = deformS.deformOutFilePath
    basePlanC.scan[baseScanIndex].saveNii(fixed_img_nii)
    currDir = os.getcwd()
    os.chdir(dirpath)
    for strNum in movStrNumV:
        #movScanNum = scn.getScanNumFromUID(movPlanC.structure[strNum].assocScanUID, movPlanC)
        structName = movPlanC.structure[strNum].structureName
        movPlanC.structure[strNum].saveNii(moving_str_nii, movPlanC)
        plm_warp_str_cmd = "plastimatch warp --input " + moving_str_nii + \
                      " --output-img " + warped_str_nii + \
                      " --xf " + bsplines_coeff_file + \
                      " --fixed " + fixed_img_nii + \
                      " --interpolation nn"
        os.system(plm_warp_str_cmd)
        # loadNiiStructure expects labels_dict as {structureName: label}; the
        # saved/warped structure nii is a binary mask with label value 1.
        basePlanC = pc.loadNiiStructure(warped_str_nii, baseScanIndex, basePlanC, {structName: 1})

    os.chdir(currDir)

    # Remove temporary directory
    print(dirpath)
    #shutil.rmtree(dirpath)

    return basePlanC


def calcVectorField(deformS, planC, baseScanNum, transformSaveDir):
    """Routine to generate a vector field file from input pyCERR transformation object

    Args:
        deformS (cerr.dataclasses.deform.Deform): pyCERR's deformation transformation object
        planC (cerr.plan_container.PlanC): pyCERR's plan container
        baseScanNum (int): index of fixed scan from planC.scan
        transformSaveDir (str): path to directory for storing the vector field file

    Returns:
        cerr.plan_container.PlanC: plan container object with a 'vf' deform object, holding the
        sampled vector field, appended to planC.deform

    ANTs transforms are sampled with
    :func:`cerr.registration.ants_reg.calcVectorFieldAnts`; plastimatch B-spline
    transforms with plastimatch (deprecated).
    """
    if _isAntsDeform(deformS):
        return calcVectorFieldAnts(deformS, planC, baseScanNum, transformSaveDir)
    _warnPlastimatch()

    # create temporary directory to hold registration files
    dirpath = tempfile.mkdtemp()

    # Get base scan index
    #fixed_img_nii = os.path.join(dirpath, 'ref.nii.gz')
    #planC.scan[baseScanNum].save_nii(fixed_img_nii)
    numRows, numCols, numSlcs = planC.scan[baseScanNum].getScanSize()

    # Write nii files for base and moving scans in dirpath
    vf_nii_src = os.path.join(dirpath, 'vf.nii.gz')
    vf_nii_dest = os.path.join(transformSaveDir, 'vf.nii.gz')

    # Get x,y,z coordinate of the 1st voxel
    xV, yV, zV = planC.scan[baseScanNum].getScanXYZVals()
    spacing = [10*(xV[1]-xV[0]), 10*(yV[0]-yV[1]), 10*(zV[1]-zV[0])]
    if scn.flipSliceOrderFlag(planC.scan[baseScanNum]):
        cerrImgPatPos = [xV[0], yV[0], zV[-1], 1]
    else:
        cerrImgPatPos = [xV[0], yV[0], zV[0], 1]
    dcmImgPos = np.matmul(planC.scan[baseScanNum].cerrToDcmTransM, cerrImgPatPos)[:3]

    bsplines_coeff_file = deformS.deformOutFilePath

    plm_warp_str_cmd = "plastimatch xf-convert --input " + bsplines_coeff_file + \
                  " --output " + vf_nii_src + \
                  " --output-type vf" + \
                  " --dim \"" + str(numCols) + " " + str(numRows) + " " + str(numSlcs) + "\""\
                  " --spacing \"" + str(spacing[0]) + ' ' + str(spacing[1]) + ' ' + str(spacing[2]) + "\""\
                  " -- origin \"" + str(dcmImgPos[0]) + ' ' + str(dcmImgPos[1]) + ' ' + str(dcmImgPos[2]) + "\""

    print("======== Plastimatch command =======")
    print(plm_warp_str_cmd)

    currDir = os.getcwd()
    os.chdir(dirpath)
    os.system(plm_warp_str_cmd)
    os.chdir(currDir)

    # Copy output to the user-specified directory
    shutil.copyfile(vf_nii_src, vf_nii_dest)

    # Create a deform object and add to planC
    deform = cerrDeform.Deform()
    deform.deformUID = uid.createUID("deform")
    deform.baseScanUID = deformS.baseScanUID
    deform.movScanUID = deformS.baseScanUID
    deform.deformOutFileType = "vf"
    deform.deformOutFilePath = vf_nii_dest
    deform.registrationTool = deformS.registrationTool
    deform.algorithm = deformS.algorithm

    # Append to base planc
    planC.deform.append(deform)

    # Remove temporary directory
    shutil.rmtree(dirpath)

    return planC


def calcJacobian(deformS, planC, tool='ants'):
    """Routine to compute the Jacobian determinant of a deformation and add it to planC as a scan

    The determinant is the local volume change of the mapping from the fixed scan to
    the moving scan: 1 means no change, below 1 contraction, above 1 expansion, and
    values at or below 0 mean the mapping folds.

    Args:
        deformS (cerr.dataclasses.deform.Deform): pyCERR's deformation transformation object:
            an ANTs registration, or a 'vf' deform object holding a vector field file
            (e.g. from :func:`calcVectorField`).
        planC (cerr.plan_container.PlanC): pyCERR's plan container
        tool (str): optional. 'ants' (default) or 'plastimatch' (deprecated).

    Returns:
        cerr.plan_container.PlanC: plan container with the Jacobian appended to planC.scan
        (image type 'Jacobian'), on the grid of the deformation's fixed scan.

    Example:
        ::

            from cerr.registration import register

            planC = register.registerScans(planC, 0, planC, 1, '/path/to/transforms')
            planC = register.calcJacobian(planC.deform[-1], planC)
            jac3M = planC.scan[-1].getScanArray()
    """
    if tool == 'plastimatch':
        _warnPlastimatch()
        if deformS.deformOutFileType != 'vf':
            raise ValueError("plastimatch Jacobian needs a 'vf' deform object; "
                             "use calcVectorField first.")
        dirpath = tempfile.mkdtemp()
        jacobian_nii = os.path.join(dirpath, 'jacobian.nii.gz')
        plm_jacobian_str_cmd = "plastimatch jacobian --input " + deformS.deformOutFilePath + \
                      " --output-img " + jacobian_nii
        os.system(plm_jacobian_str_cmd)
        planC = pc.loadNiiScan(jacobian_nii, "Jacobian", '', planC)
        shutil.rmtree(dirpath)
        return planC
    if tool != 'ants':
        raise ValueError("tool must be 'ants' or 'plastimatch', not %r" % tool)

    from cerr.registration.ants_reg import _composeAntsField, _importAnts
    ants = _importAnts()
    baseScanNum = scn.getScanNumFromUID(deformS.baseScanUID, planC)
    dirpath = tempfile.mkdtemp()
    if _isAntsDeform(deformS):
        dvfFile = _composeAntsField(deformS, planC, baseScanNum, dirpath)
    elif deformS.deformOutFileType == 'vf' and deformS.deformOutFilePath \
            and os.path.exists(deformS.deformOutFilePath):
        dvfFile = deformS.deformOutFilePath
    else:
        shutil.rmtree(dirpath, ignore_errors=True)
        raise ValueError("calcJacobian needs an ANTs deform object or a 'vf' deform object "
                         "with a vector field file (deformOutFileType=%r)."
                         % deformS.deformOutFileType)
    fixedNii = os.path.join(dirpath, 'fixed.nii.gz')
    planC.scan[baseScanNum].saveNii(fixedNii)
    jacImg = ants.create_jacobian_determinant_image(ants.image_read(fixedNii), dvfFile,
                                                    do_log=False)
    jacNii = os.path.join(dirpath, 'jacobian.nii.gz')
    ants.image_write(jacImg, jacNii)
    planC = pc.loadNiiScan(jacNii, "Jacobian", '', planC)
    shutil.rmtree(dirpath, ignore_errors=True)
    return planC


def _requireDvf(deformS):
    """Raise a clear error when a deform object has no sampled vector field."""
    if cerrDeform.hasDvfMatrix(deformS):
        return
    tool = (deformS.registrationTool or '').lower()
    if tool == 'ants' or deformS.deformOutFileType == 'ants':
        hint = ("sample it first with cerr.registration.ants_reg.calcVectorFieldAnts"
                "(deformS, planC, baseScanNum) and use the deform object it appends")
    elif deformS.deformOutFileType == 'rigid':
        hint = "a rigid registration has no per-voxel vector field"
    else:
        hint = ("sample it first with cerr.registration.register.calcVectorField "
                "(plastimatch) and use the deform object it appends")
    raise ValueError("This deform object (registrationTool=%r, deformOutFileType=%r) "
                     "carries no deformation vector field: %s."
                     % (deformS.registrationTool, deformS.deformOutFileType, hint))


def getDvfVectors(deformS, planC, scanNum, outputResV=[0, 0, 0], structNum=None, surfFlag=False):
    """Routine to obtain dvf vectors

    Args:
        deformS (cerr.dataclasses.deform.Deform): pyCERR's deformation transformation object
        planC (cerr.plan_container.PlanC): pyCERR's plan container
        scanNum (int): index for scan in planC.scan
        outputResV (ist): optional, grid resolution for output vectors. Used only when surfFlag is False.
        structNum (int): optional, index of planC.structure to restrict the vector calculation
        surfFlag (bool): optional, flag to calculate vectors on the surface of structNum

    Returns:
        (numpy.ndarray): (n x 2 x 3) vectors where 1st element in the 2nd dimension represents the start coordinates in
            pyCERR r,c,s units and 2nd element in the 2nd dimension represents deformation y,x,z in cm in pyCERR virtual
            coordinates.

    """

    #assocScanNum = scn.getScanNumFromUID(planC.structure[structNum].assocScanUID, planC)
    xValsV, yValsV, zValsV = planC.scan[scanNum].getScanXYZVals()
    zStart = zValsV[0]

    # structNum is optional: without one the vectors cover the whole scan, and
    # there is no mask to build.
    mask3M = None if structNum is None else rs.getStrMask(structNum, planC)

    origRes = planC.scan[scanNum].getScanSpacing()
    pixelSpacingZ = origRes[2]

    if structNum is not None and surfFlag:
        # # Get surface points from structure contours
        # rcsSurfPolygons =  cerrStr.getContourPolygons(structNum, planC, True)
        # rcsSurfPoints = np.array(rcsSurfPolygons[0])
        # for poly in rcsSurfPolygons[1:]:
        #     rcsSurfPoints = np.append(rcsSurfPoints, poly, axis=0)
        # rSurfV = rcsSurfPoints[:, 0]
        # cSurfV = rcsSurfPoints[:, 1]
        # sSurfV = rcsSurfPoints[:, 2]
        #
        # xSurfV = xValsV[cSurfV.astype(int)]
        # ySurfV = yValsV[rSurfV.astype(int)]
        # zSurfV = zValsV[sSurfV.astype(int)]

        rowV, colV, slcV = getSurfacePoints(mask3M)
        xSurfV = xValsV[colV]
        ySurfV = yValsV[rowV]
        zSurfV = zValsV[slcV]

    elif not surfFlag:
        if structNum is not None and isinstance(structNum,(int,float)):
            # mask3M = rs.getStrMask(structNum, planC)
            rmin,rmax,cmin,cmax,smin,smax,_ = computeBoundingBox(mask3M)
            xValsV = xValsV[cmin:cmax+1]
            yValsV = yValsV[rmin:rmax+1]
            zValsV = zValsV[smin:smax+1]

        # = np.absolute(np.median(np.diff(xValsV)))
        #pixelSpacingY = np.absolute(np.median(np.diff(yValsV)))
        #pixelSpacingZ = np.absolute(np.median(np.diff(zValsV)))
        #origRes = [pixelSpacingX, pixelSpacingY, pixelSpacingZ]

        # origRes = planC.scan[scanNum].getScanSpacing()
        # pixelSpacingZ = origRes[2]

        outputResV = [outputResV[i] if outputResV[i] > 0 else origRes[i] for i,_ in enumerate(outputResV)]

        grid_resample_method = 'center'
        [xResampleV,yResampleV,zResampleV] = preprocess.getResampledGrid(outputResV,\
                                                xValsV, yValsV, zValsV, grid_resample_method)
        xM, yM, zM = np.meshgrid(xResampleV,yResampleV,zResampleV)
        xSurfV = xM.flatten()
        ySurfV = yM.flatten()
        zSurfV = zM.flatten()

    scaleX = planC.scan[scanNum].scanInfo[0].grid2Units
    scaleY = planC.scan[scanNum].scanInfo[0].grid1Units
    imageSizeV = [planC.scan[scanNum].scanInfo[0].sizeOfDimension1,
                  planC.scan[scanNum].scanInfo[0].sizeOfDimension2]

    # Get any offset of CT scans to apply (neg) to structures
    xCTOffset = planC.scan[scanNum].scanInfo[0].xOffset \
        if planC.scan[scanNum].scanInfo[0].xOffset else 0
    yCTOffset = planC.scan[scanNum].scanInfo[0].yOffset \
        if planC.scan[scanNum].scanInfo[0].yOffset else 0

    rSurfV, cSurfV = rs.aapmtom(xSurfV/scaleX, ySurfV/scaleY, xCTOffset / scaleX,
                         yCTOffset / scaleY, imageSizeV)

    sSurfV = (zSurfV - zStart) / pixelSpacingZ


    # Get x,y,z deformations at selected points
    _requireDvf(deformS)
    xV, yV, zV = deformS.getDVFXYZVals()
    delta = 1e-8
    zV[0] = zV[0] - 1e-3
    zV[-1] = zV[-1] + 1e-3
    xFieldV = np.asarray([xV[0] - delta, xV[1] - xV[0], xV[-1] + delta])
    yFieldV = np.asarray([yV[0] + delta, yV[1] - yV[0], yV[-1] - delta])
    zFieldV = np.asarray(zV)
    xDeformM = deformS.dvfMatrix[:,:,:,0]
    yDeformM = deformS.dvfMatrix[:,:,:,1]
    zDeformM = deformS.dvfMatrix[:,:,:,2]
    xDeformV = finterp3(xSurfV,ySurfV,zSurfV,xDeformM,xFieldV,yFieldV,zFieldV)
    yDeformV = finterp3(xSurfV,ySurfV,zSurfV,yDeformM,xFieldV,yFieldV,zFieldV)
    zDeformV = finterp3(xSurfV,ySurfV,zSurfV,zDeformM,xFieldV,yFieldV,zFieldV)

    # Convert xDeformV,yDeformV,zDeformV to CERR virtual coordinates (cm)
    xDeformV, yDeformV, zDeformV = dvfToCerrVirtual(xDeformV, yDeformV,
                                                    zDeformV, planC, scanNum)
    numPts = len(yDeformV)
    vectors = np.empty((numPts,2,3), dtype=np.float32)
    rcsFlag = True # an input argument?
    if rcsFlag: # (r,c,s) image coordinates
        #dx = np.abs(np.median(np.diff(xValsV)))
        #dy = np.abs(np.median(np.diff(yValsV)))
        #dz = np.abs(np.median(np.diff(zValsV)))
        # Convert CERR virtual coords to DICOM Image coords
        for i in range(numPts):
            vectors[i,0,:] = [rSurfV[i], cSurfV[i], sSurfV[i]]
            #vectors[i,1,:] = [yDeformV[i]/dy, xDeformV[i]/dx, zDeformV[i]/dz]
            vectors[i,1,:] = [yDeformV[i], xDeformV[i], zDeformV[i]]
        # deformMedian = np.median(vectors, axis = 0)[1,:]
        # vectors[:,1,:] -= deformMedian
    else: # (x,y,z) physical coordinates
        if cerrDeform.flipSliceOrderFlag(deformS):
            zDeformV = - zDeformV
        for i in range(numPts):
            vectors[i,0,:] = [ySurfV[i], xSurfV[i], zSurfV[i]]
            vectors[i,1,:] = [yDeformV[i], xDeformV[i], zDeformV[i]]
    return vectors


def dvfToCerrVirtual(xDeformV, yDeformV, zDeformV, planC, scanNum):
    """Convert DVF components from DICOM patient (LPS, mm) to pyCERR virtual
    coordinates (cm).

    ``Deform.dvfMatrix`` stores displacements as DICOM LPS millimetres whatever
    the source (RTREG, plastimatch ``.vf``, NIfTI), while pyCERR's virtual
    coordinate system - the one the viewers draw in - is right-handed and in
    centimetres. The scan's ``cerrToDcmTransM`` carries both the rotation and
    the cm->mm scaling, so inverting it and subtracting the image of the origin
    (a displacement is a difference of positions, so the translation must drop
    out) gives the displacement in virtual cm.

    Args:
        xDeformV, yDeformV, zDeformV (np.ndarray): 1-D arrays of DVF components
            in DICOM patient coordinates (mm).
        planC (cerr.plan_container.PlanC): pyCERR's plan container.
        scanNum (int): index of the scan defining the virtual coordinate system.

    Returns:
        tuple: ``(xV, yV, zV)`` displacement components in pyCERR virtual
        coordinates, in cm.
    """
    onesV = np.ones_like(xDeformV)
    zeroV = np.zeros_like(xDeformV)
    dcmXyzM = np.vstack((xDeformV, yDeformV, zDeformV, onesV))
    dcmZeroM = np.vstack((zeroV, zeroV, zeroV, onesV))
    invM = np.linalg.inv(planC.scan[scanNum].cerrToDcmTransM)
    cerrXYZM = np.matmul(invM, dcmXyzM)[:3] - np.matmul(invM, dcmZeroM)[:3]
    return cerrXYZM[0, :], cerrXYZM[1, :], cerrXYZM[2, :]


#: Default ceiling on the number of sample points when ``getDvfGrid`` picks the
#: output resolution itself. Three float32 components at this size are ~24 MB,
#: and one arrow per sample is already denser than any slice view can show.
DVF_GRID_MAX_POINTS = 2_000_000


def getDvfGrid(deformS, planC, scanNum, outputResV=None, structNum=None,
               surfFlag=False, subtractMedian=False, dtype=np.float32,
               maxPoints=DVF_GRID_MAX_POINTS):
    """Sample a deformation vector field on a regular grid, for slice display.

    The grid counterpart of :func:`getDvfVectors`, which returns the same
    vectors scattered (an ``(n, 2, 3)`` array for Napari's ``Vectors`` layer).
    Sampling, restriction and median-subtraction follow that routine exactly -
    the same ``outputResV`` / ``structNum`` / ``surfFlag`` controls, the same
    coordinate transform - but the result comes back as three dense component
    arrays plus the grid's own axes, so a viewer can quiver any plane of it
    without searching a point cloud for the vectors near that plane.

    Everything is in pyCERR's own convention: displacements are in **virtual
    x, y, z centimetres** (x along the columns, increasing left to right; y
    along the rows, DECREASING down the image; z along the slices), not
    Napari's row/col/slice frame with its flipped y.

    Args:
        deformS (cerr.dataclasses.deform.Deform): pyCERR deformation object.
        planC (cerr.plan_container.PlanC): pyCERR's plan container.
        scanNum (int): index of the scan whose grid frames the sampling.
        outputResV (list): output grid resolution ``[dx, dy, dz]`` in cm; a 0
            entry means that axis's native scan spacing. ``None`` (the default)
            starts from the native spacing and coarsens all three axes by a
            common factor until the grid fits ``maxPoints``.
        structNum (int): optional index into ``planC.structure``. The sampling
            is framed on that structure's bounding box and vectors outside the
            structure itself are dropped.
        surfFlag (bool): with ``structNum``, keep only the vectors on the
            structure's SURFACE - the deformation of a boundary is usually what
            a registration is judged on, and interior arrows hide it.
        subtractMedian (bool): subtract the median displacement of the sampled
            vectors, leaving the LOCAL deformation. A registration that is
            mostly a bulk shift otherwise draws one uniform arrow field in which
            no local detail is visible.
        dtype: floating dtype of the returned components (default float32 -
            display data, where float64 doubles the footprint for no gain).
        maxPoints (int): sample-count ceiling used when ``outputResV`` is
            ``None``.

    Returns:
        tuple: ``(comps, axes, info)`` where

        - **comps** (*list*): ``[dRow, dCol, dSlc]``, each of shape
          ``(len(yV), len(xV), len(zV))``, holding the displacement component
          along the grid's row (virtual y), column (virtual x) and slice
          (virtual z) directions in cm. Zero wherever no vector is drawn -
          outside the DVF's field of view, or outside the selected structure.
        - **axes** (*tuple*): ``(xV, yV, zV)``, the grid's physical coordinates
          in cm (x ascending, y descending, z ascending, as elsewhere in
          pyCERR).
        - **info** (*dict*): ``'valid'`` (bool array, True where a vector was
          sampled), ``'resolution'`` (the grid spacing actually used, cm),
          ``'median'`` (the subtracted median displacement in virtual cm, zeros
          when ``subtractMedian`` is False) and ``'nVectors'``.
    """
    xValsV, yValsV, zValsV = planC.scan[scanNum].getScanXYZVals()
    origRes = planC.scan[scanNum].getScanSpacing()

    mask3M = None
    if structNum is not None:
        mask3M = rs.getStrMask(structNum, planC)
        if surfFlag:
            # getSurfacePoints returns the mask's boundary voxels; a surface
            # mask keeps the same nearest-voxel lookup as the solid one.
            rowV, colV, slcV = getSurfacePoints(mask3M)
            surf3M = np.zeros_like(mask3M, dtype=bool)
            surf3M[rowV, colV, slcV] = True
            mask3M = surf3M
        rmin, rmax, cmin, cmax, smin, smax, _ = computeBoundingBox(mask3M)
        xValsV = xValsV[cmin:cmax + 1]
        yValsV = yValsV[rmin:rmax + 1]
        zValsV = zValsV[smin:smax + 1]

    def _grid(resV):
        return preprocess.getResampledGrid(np.asarray(resV, dtype=float),
                                           xValsV, yValsV, zValsV, 'center')

    if outputResV is None:
        # Auto: the native spacing, coarsened isotropically until the sample
        # count fits. Coarsening every axis by the same factor keeps the arrow
        # spacing even, which a per-axis fit would not. The first factor is an
        # estimate - getResampledGrid rounds each axis up, so keep growing it
        # until the grid it actually returns fits.
        nPts = len(xValsV) * len(yValsV) * len(zValsV)
        factor = max(1.0, np.ceil((nPts / float(maxPoints)) ** (1 / 3.0)))
        while True:
            resV = [r * factor for r in origRes]
            xR, yR, zR = _grid(resV)
            if len(xR) * len(yR) * len(zR) <= maxPoints:
                break
            factor += 1.0
    else:
        resV = [outputResV[i] if outputResV[i] > 0 else origRes[i]
                for i in range(3)]
        xR, yR, zR = _grid(resV)
    xM, yM, zM = np.meshgrid(xR, yR, zR)      # 'xy' -> (rows, cols, slices)
    shape = xM.shape
    xQ, yQ, zQ = xM.flatten(), yM.flatten(), zM.flatten()

    # DVF grid. finterp3 takes the x/y axes as (start, delta, end) and needs the
    # end points nudged outward so boundary samples are not rejected as
    # out-of-bounds; z is passed in full (slice spacing may be non-uniform).
    _requireDvf(deformS)
    xV, yV, zV = deformS.getDVFXYZVals()
    zV = np.asarray(zV, dtype=float).copy()          # copy: do not edit deformS
    delta = 1e-8
    zV[0] = zV[0] - 1e-3
    zV[-1] = zV[-1] + 1e-3
    xFieldV = np.asarray([xV[0] - delta, xV[1] - xV[0], xV[-1] + delta])
    yFieldV = np.asarray([yV[0] + delta, yV[1] - yV[0], yV[-1] - delta])
    defV = [finterp3(xQ, yQ, zQ, deformS.dvfMatrix[:, :, :, comp],
                     xFieldV, yFieldV, zV) for comp in range(3)]
    # NaN marks a sample outside the DVF's field of view, where there is no
    # deformation to show. Tracked as a mask rather than folded into the values,
    # so a genuinely zero displacement stays distinguishable from no data.
    valid = np.isfinite(defV[0]) & np.isfinite(defV[1]) & np.isfinite(defV[2])
    defV = [np.nan_to_num(d) for d in defV]
    xDeformV, yDeformV, zDeformV = dvfToCerrVirtual(defV[0], defV[1], defV[2],
                                                    planC, scanNum)

    if mask3M is not None:
        # Nearest scan voxel of each sample point: the structure mask is on the
        # scan grid, and interpolating it onto a coarser grid would erode a
        # one-voxel-thick surface away entirely.
        valid &= _nearestMaskValues(mask3M, xQ, yQ, zQ, planC, scanNum)

    medianV = np.zeros(3)
    if subtractMedian and valid.any():
        medianV = np.array([np.median(yDeformV[valid]),
                            np.median(xDeformV[valid]),
                            np.median(zDeformV[valid])])
        yDeformV = yDeformV - medianV[0]
        xDeformV = xDeformV - medianV[1]
        zDeformV = zDeformV - medianV[2]

    comps = []
    for d in (yDeformV, xDeformV, zDeformV):          # [row(y), col(x), slc(z)]
        d = np.where(valid, d, 0.0)
        comps.append(d.reshape(shape).astype(dtype))
    info = {"valid": valid.reshape(shape), "resolution": list(resV),
            "median": medianV, "nVectors": int(valid.sum())}
    return comps, (xR, yR, zR), info


def _nearestMaskValues(mask3M, xQ, yQ, zQ, planC, scanNum):
    """Value of a scan-grid mask at each query point, by nearest voxel."""
    xValsV, yValsV, zValsV = planC.scan[scanNum].getScanXYZVals()
    cIdx = np.clip(np.rint((xQ - xValsV[0]) / (xValsV[1] - xValsV[0])),
                   0, len(xValsV) - 1).astype(int)
    rIdx = np.clip(np.rint((yQ - yValsV[0]) / (yValsV[1] - yValsV[0])),
                   0, len(yValsV) - 1).astype(int)
    sIdx = np.clip(np.searchsorted(zValsV, zQ), 0, len(zValsV) - 1)
    # searchsorted gives the upper neighbour; step back where the lower one is
    # nearer (slice spacing may be non-uniform, so this cannot be a division).
    lower = np.clip(sIdx - 1, 0, len(zValsV) - 1)
    takeLower = np.abs(zQ - zValsV[lower]) < np.abs(zQ - zValsV[sIdx])
    sIdx = np.where(takeLower, lower, sIdx)
    return mask3M[rIdx, cIdx, sIdx].astype(bool)
