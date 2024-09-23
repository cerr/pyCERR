"""The register module defines routines for registering and deforming scans, structures and does distributions.
It also provies routines for calculating vector field and Jacobian from deformation transformation.

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


def registerScans(basePlanC, baseScanIndex, movPlanC, movScanIndex, transformSaveDir,
                  deforAlgorithm='bsplines', registrationTool='plastimatch',
                  baseMask3M=None, movMask3M=None, inputCmdFile=None):
    """

    Args:
        basePlanC (cerr.plan_container.PlanC): pyCERR plan container containing fixed target scan
        baseScanIndex (int): integer, identifies target scan in basePlanC
        movPlanC (cerr.plan_container.PlanC): pyCERR plan container containing moving scan
        movScanIndex (int): integer, identifies moving scan in movPlanC
        transformSaveDir (str): Directory to save transformation file
        registration_tool (str): registration software to use ('PLASTIMATCH','ELASTIX','ANTS')
        baseMask3M (numpy.ndarray): optional, 3D or 4D binary mask(s) in target space
        movMask3M (numpy.ndarray): optional, 3D or 4D binary mask(s) in moving space
        inputCmdFile (str): optional, path to registration command file

    Returns:
        cerr.plan_container.PlanC: plan container object basePlanC with an element added to planC.deform attribute

    """

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
        pc.saveNiiStructure(fixed_mask_nii, maskStrNum, basePlanC)
        del basePlanC.structure[-1]
    if movMask3M is not None:
        movPlanC = pc.importStructureMask(movMask3M, movScanIndex, 'mask', movPlanC)
        maskStrNum = len(movPlanC.structure) - 1
        pc.saveNiiStructure(moving_mask_nii, maskStrNum, movPlanC)
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
    """

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
    """

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
    """

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
        basePlanC = pc.loadNiiStructure(warped_str_nii, baseScanIndex, basePlanC, {1: structName})

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
        cerr.plan_container.PlanC: plan container object basePlanC with the deformed structures added to planC.structure attribute

    """

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


def calcJacobian(deformS, planC, tool='plastimatch'):
    """Routine to compute Jacobian of vector field

    Args:
        deformS (cerr.dataclasses.deform.Deform): pyCERR's deformation transformation object
        planC (cerr.plan_container.PlanC): pyCERR's plan container
        tool (str): optional. tool to use for Jacobian calculation, defaults to 'plastimatch'

    Returns:

    """

    if deformS.deformOutFileType != 'vf':
        return

    if tool == 'plastimatch':

        # create temporary directory to hold registration files
        dirpath = tempfile.mkdtemp()

        # Command for Jacobian calculation
        jacobian_nii = os.path.join(dirpath, 'jacobian.nii.gz')
        plm_jacobian_str_cmd = "plastimatch jacobian --input " + deformS.deformOutFilePath + \
                      " --output-img " + jacobian_nii

        os.system(plm_jacobian_str_cmd)

        # Add Jacobian to planC scan
        direction = ''
        planC = pc.loadNiiScan(jacobian_nii, "Jacobian", direction, planC)

        # Remove temporary directory
        shutil.rmtree(dirpath)

    return planC


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

    mask3M = rs.getStrMask(structNum, planC)

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

    # Convert xDeformV,yDeformV,zDeformV to CERR virtual coordinates
    onesV = np.ones_like(xDeformV)
    zeroV = np.zeros_like(xDeformV)
    dcmXyzM = np.vstack((xDeformV,yDeformV,zDeformV, onesV))
    dcmZeroM = np.vstack((zeroV,zeroV,zeroV, onesV))
    deformPos = np.matmul(np.linalg.inv(planC.scan[scanNum].cerrToDcmTransM), dcmXyzM)[:3]
    zeroPos = np.matmul(np.linalg.inv(planC.scan[scanNum].cerrToDcmTransM), dcmZeroM)[:3]
    cerrXYZM = deformPos - zeroPos
    # DVF in mm (Note that CERR coordinate system is in cm.)
    xDeformV = cerrXYZM[0,:]
    yDeformV = cerrXYZM[1,:]
    zDeformV = cerrXYZM[2,:]
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
