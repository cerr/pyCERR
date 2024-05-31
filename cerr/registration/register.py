"""register module.

Ths register module defines modules for.

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
from cerr.utils.mask import getSurfacePoints, compute_boundingbox
from cerr.utils.interp import finterp3
from cerr.radiomics import preprocess
import numpy as np

def register_scans(basePlanC, baseScanIndex, movPlanC, movScanIndex, transformSaveDir):

    # create temporary directory to hold registration files
    dirpath = tempfile.mkdtemp()

    # Write nii files for base and moving scans in dirpath
    moving_img_nii = os.path.join(dirpath, 'ctmoving.nii.gz')
    fixed_img_nii = os.path.join(dirpath, 'ctfixed.nii.gz')
    basePlanC.scan[baseScanIndex].save_nii(fixed_img_nii)
    movPlanC.scan[movScanIndex].save_nii(moving_img_nii)

    plmCmdFile = 'plastimatch_ct_ct_intra_pt.txt'
    regDir = os.path.dirname(os.path.abspath(__file__))
    cmdFilePathSrc = os.path.join(regDir,'settings',plmCmdFile)
    #cmdFilePathDest = os.path.join(dirpath, plmCmdFile)
    #shutil.copyfile(cmdFilePathSrc, cmdFilePathDest)

    # Filename to save bsplines coeffficients
    bspSourcePath = os.path.join(dirpath, 'bspline_coefficients.txt')
    bspDestPath = os.path.join(transformSaveDir, 'bspline_coefficients.txt')

    plm_reg_cmd = "plastimatch register " + cmdFilePathSrc

    currDir = os.getcwd()
    os.chdir(dirpath)
    os.system(plm_reg_cmd)
    os.chdir(currDir)

    # Copy output to the user-specified directory
    shutil.copyfile(bspSourcePath, bspDestPath)

    # Create a deform object and add to planC
    deform = cerrDeform.Deform()
    deform.deformUID = uid.createUID("deform")
    deform.baseScanUID = basePlanC.scan[baseScanIndex].scanUID
    deform.movScanUID = movPlanC.scan[movScanIndex].scanUID
    deform.deformOutFileType = "plm_bspline_coeffs"
    deform.deformOutFilePath = bspDestPath
    deform.registrationTool = 'plastimatch'
    deform.algorithm = 'bsplines'

    # Append to base planc
    basePlanC.deform.append(deform)

    # Remove temporary directory
    shutil.rmtree(dirpath)

    return basePlanC


def warp_scan(basePlanC, baseScanIndex, movPlanC, movScanIndex, deformS):
    dirpath = tempfile.mkdtemp()
    fixed_img_nii = os.path.join(dirpath, 'ref.nii.gz')
    moving_img_nii = os.path.join(dirpath, 'ctmoving.nii.gz')
    warped_img_nii = os.path.join(dirpath, 'warped.nii.gz')
    bsplines_coeff_file = deformS.deformOutFilePath
    basePlanC.scan[baseScanIndex].save_nii(fixed_img_nii)
    movPlanC.scan[movScanIndex].save_nii(moving_img_nii)


    plm_warp_str_cmd = "plastimatch warp --input " + moving_img_nii + \
                  " --output-img " + warped_img_nii + \
                  " --xf " + bsplines_coeff_file + \
                  " --referenced-ct " + fixed_img_nii

    currDir = os.getcwd()
    os.chdir(dirpath)
    os.system(plm_warp_str_cmd)
    os.chdir(currDir)

    imageType = movPlanC.scan[movScanIndex].scanInfo[0].imageType
    direction = ''
    basePlanC = pc.load_nii_scan(warped_img_nii, imageType, direction, basePlanC)

    # Remove temporary directory
    shutil.rmtree(dirpath)

    return basePlanC


def warp_dose():
    pass

def warp_structures(basePlanC, baseScanIndex, movPlanC, movStrNumV, deformS):
    # dirpath = tempfile.mkdtemp()
    # rtst_warped_path = os.path.join(dirpath, 'struct.nii.gz')
    dirpath = tempfile.mkdtemp()
    fixed_img_nii = os.path.join(dirpath, 'ref.nii.gz')
    moving_str_nii = os.path.join(dirpath, 'structure.nii.gz')
    warped_str_nii = os.path.join(dirpath, 'warped.nii.gz')
    bsplines_coeff_file = deformS.deformOutFilePath
    basePlanC.scan[baseScanIndex].save_nii(fixed_img_nii)
    currDir = os.getcwd()
    os.chdir(dirpath)
    for strNum in movStrNumV:
        #movScanNum = scn.getScanNumFromUID(movPlanC.structure[strNum].assocScanUID, movPlanC)
        structName = movPlanC.structure[strNum].structureName
        movPlanC.structure[strNum].save_nii(moving_str_nii, movPlanC)
        plm_warp_str_cmd = "plastimatch warp --input " + moving_str_nii + \
                      " --output-img " + warped_str_nii + \
                      " --xf " + bsplines_coeff_file + \
                      " --fixed " + fixed_img_nii + \
                      " --interpolation nn"
        os.system(plm_warp_str_cmd)
        basePlanC = pc.load_nii_structure(warped_str_nii, baseScanIndex, basePlanC, {1: structName})

    os.chdir(currDir)

    # Remove temporary directory
    print(dirpath)
    #shutil.rmtree(dirpath)

    return basePlanC

def calc_vector_field(deformS, planC, baseScanNum, transformSaveDir):

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

def calc_jacobian(deformS, planC, tool='plastimatch'):

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
        planC = pc.load_nii_scan(jacobian_nii, "Jacobian", direction, planC)

        # Remove temporary directory
        shutil.rmtree(dirpath)

    return planC


def get_dvf_vectors(deformS, planC, structNum, surfFlag=False, outputResV=[0,0,0]):

    assocScanNum = scn.getScanNumFromUID(planC.structure[structNum].assocScanUID, planC)
    xValsV, yValsV, zValsV = planC.scan[assocScanNum].getScanXYZVals()
    zStart = zValsV[0]

    if structNum is not None and surfFlag:
        # Get surface points from structure contours
        rcsSurfPolygons =  cerrStr.getContourPolygons(structNum, planC, True)
        rcsSurfPoints = np.array(rcsSurfPolygons[0])
        for poly in rcsSurfPolygons[1:]:
            rcsSurfPoints = np.append(rcsSurfPoints, poly, axis=0)
        rSurfV = rcsSurfPoints[:, 0]
        cSurfV = rcsSurfPoints[:, 1]
        sSurfV = rcsSurfPoints[:, 2]

        xSurfV = xValsV[cSurfV.astype(int)]
        ySurfV = yValsV[rSurfV.astype(int)]
        zSurfV = zValsV[sSurfV.astype(int)]

    elif not surfFlag:
        if structNum is not None:
            mask3M = rs.getStrMask(structNum, planC)
            rmin,rmax,cmin,cmax,smin,smax,_ = compute_boundingbox(mask3M)
            xValsV = xValsV[cmin:cmax+1]
            yValsV = yValsV[rmin:rmax+1]
            zValsV = zValsV[smin:smax+1]

        pixelSpacingX = np.absolute(np.median(np.diff(xValsV)))
        pixelSpacingY = np.absolute(np.median(np.diff(yValsV)))
        pixelSpacingZ = np.absolute(np.median(np.diff(zValsV)))
        origRes = [pixelSpacingX, pixelSpacingY, pixelSpacingZ]

        outputResV = [outputResV[i] if outputResV[i] > 0 else origRes[i] for i,_ in enumerate(outputResV)]

        grid_resample_method = 'center'
        [xResampleV,yResampleV,zResampleV] = preprocess.getResampledGrid(outputResV,\
                                                xValsV, yValsV, zValsV, grid_resample_method)
        xM, yM, zM = np.meshgrid(xResampleV,yResampleV,zResampleV)
        xSurfV = xM.flatten()
        ySurfV = yM.flatten()
        zSurfV = zM.flatten()

        scaleX = planC.scan[assocScanNum].scanInfo[0].grid2Units
        scaleY = planC.scan[assocScanNum].scanInfo[0].grid1Units
        imageSizeV = [planC.scan[assocScanNum].scanInfo[0].sizeOfDimension1,
                      planC.scan[assocScanNum].scanInfo[0].sizeOfDimension2]

        # Get any offset of CT scans to apply (neg) to structures
        xCTOffset = planC.scan[assocScanNum].scanInfo[0].xOffset \
            if planC.scan[assocScanNum].scanInfo[0].xOffset else 0
        yCTOffset = planC.scan[assocScanNum].scanInfo[0].yOffset \
            if planC.scan[assocScanNum].scanInfo[0].yOffset else 0

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
    # Convert x,y,zDeformV to CERR virtual coordinates
    onesV = np.ones_like(xDeformV)
    zeroV = np.zeros_like(xDeformV)
    dcmXyzM = np.vstack((xDeformV,yDeformV,zDeformV, onesV))
    dcmZeroM = np.vstack((zeroV,zeroV,zeroV, onesV))
    deformPos = np.matmul(np.linalg.inv(planC.scan[assocScanNum].cerrToDcmTransM), dcmXyzM)[:3]
    zeroPos = np.matmul(np.linalg.inv(planC.scan[assocScanNum].cerrToDcmTransM), dcmZeroM)[:3]
    cerrXYZM = deformPos - zeroPos
    # DVF in mm (Note that CERR coordinate system is in cm.)
    xDeformV = cerrXYZM[0,:] * 10
    yDeformV = cerrXYZM[1,:] * 10
    zDeformV = cerrXYZM[2,:] * 10
    numPts = len(yDeformV)
    vectors = np.empty((numPts,2,3), dtype=np.float32)
    rcsFlag = True # an input argument?
    if rcsFlag: # (r,c,s) image coordinates
        dx = np.abs(np.median(np.diff(xValsV)))
        dy = np.abs(np.median(np.diff(yValsV)))
        dz = np.abs(np.median(np.diff(zValsV)))
        # Convert CERR virtual coords to DICOM Image coords
        for i in range(numPts):
            vectors[i,0,:] = [rSurfV[i], cSurfV[i], sSurfV[i]]
            #vectors[i,1,:] = [yDeformV[i]/dy, xDeformV[i]/dx, zDeformV[i]/dz]
            vectors[i,1,:] = [-yDeformV[i], xDeformV[i], zDeformV[i]]
        deformMedian = np.median(vectors, axis = 0)[1,:]
        vectors[:,1,:] -= deformMedian
    else: # (x,y,z) physical coordinates
        if cerrDeform.flipSliceOrderFlag(deformS):
            zDeformV = - zDeformV
        for i in range(numPts):
            vectors[i,0,:] = [-ySurfV[i], xSurfV[i], zSurfV[i]]
            vectors[i,1,:] = [-yDeformV[i], xDeformV[i], zDeformV[i]]
    return vectors
