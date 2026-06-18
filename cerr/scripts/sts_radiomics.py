from cerr.dataclasses import structure as cerrStruct
import os
from cerr import plan_container as pc
import SimpleITK as sitk
import numpy as np
import cerr.dataclasses.scan as cerrScn
import cerr.viewer as vwr
from cerr.contour import rasterseg as rs
import pandas as pd
from cerr.dataclasses import scan as cerrScn
import SimpleITK as sitk

if __name__ == "__main__":


    ptDir = r'M:\Data\soft_tissue_sarcoma_DrBozzo\expanded_images\T1\RIA_16-1123_000_000009'
    ptDir = r'M:\Data\soft_tissue_sarcoma_DrBozzo\expanded_images\T1\RIA_16-1123_000_000003'
    dcm_scan_dir = r'M:\Data\soft_tissue_sarcoma_DrBozzo\expanded_images\T1\RIA_16-1123_000_000003'
    imageDir = os.path.join(ptDir,'image')
    segDir = os.path.join(ptDir, 'mask')

    planC = pc.loadDcmDir(ptDir)

    viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
            vwr.showNapari([0], [0], [], {}, planC, '2d')

    # Apply N4 bias field correction filter
    scanNum = 0
    scanITKObj = planC.scan[scanNum].getSitkImage()
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    outImg = corrector.Execute(scanITKObj)
    logBiasFieldImg = corrector.GetLogBiasFieldAsImage(scanITKObj)
    logBiasField3M = cerrScn.getCERRScanArrayFromITK(logBiasFieldImg, scanNum, planC)
    correctedImageFullResolution = planC.scan[scanNum].getScanArray() / np.exp(logBiasField3M)


    # Add bias field corrected array to planC
    x, y, z = planC.scan[scanNum].getScanXYZVals()
    planC = pc.importScanArray(correctedImageFullResolution, x, y, z, 'N4 Corrected', scanNum, planC)

    viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
            vwr.showNapari([0,1], [0,1], [], {}, planC, '2d')

    niiScanFile = r'M:\Data\soft_tissue_sarcoma_DrBozzo\niiPyCERR\T1\scan_n4.nii.gz'
    niiStructFile = r'M:\Data\soft_tissue_sarcoma_DrBozzo\niiPyCERR\T1\struct.nii.gz'

    planC.scan[0].saveNii(niiScanFile)
    planC.structure[0].saveNii(niiStructFile, planC)

    planC=  pc.loadNiiStructure(niiStructFile, 1, planC, {1: 'tumor'})

    from cerr.contour import rasterseg as rs
    mask1 = rs.getStrMask(0, planC)
    mask2 = rs.getStrMask(1, planC)
