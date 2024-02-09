import os
import tempfile
import shutil
from cerr.utils import uid
from cerr.dataclasses import deform as cerrDeform
import cerr.plan_container as pc

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
    deform = cerrDeform.Deform
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
    basePlanC = pc.load_nii_scan(warped_img_nii, imageType, basePlanC)

    # Remove temporary directory
    shutil.rmtree(dirpath)

    return basePlanC


def warp_dose():
    pass

def warp_structures():
    # dirpath = tempfile.mkdtemp()
    # rtst_warped_path = os.path.join(dirpath, 'struct.nii.gz')
    pass
