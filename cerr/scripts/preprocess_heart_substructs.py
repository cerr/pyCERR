from cerr import plan_container as pc
import os
from cerr import viewer as vwr
from skimage.io import imsave, imshow
import cerr.contour.rasterseg as rs
import numpy as np
from cerr.dataclasses import structure as cerrStr
from cerr.utils import bbox
import os

if __name__ == "__main__":

    # RTOG0617 sample cases
    # dataUrl="https://mskcc.box.com/shared/static/psde8yjjtz55mlos8gbdi0gnr3gcr5ao.gz"

    atriaLabelDict = {1: 'DL_Atria'}
    heartSubSegDict = {2: 'AORTA', 3: 'DL_LA',
                       4: 'DL_LV', 5: 'DL_RA',
                       6: 'DL_RV', 7: 'DL_IVC',
                       8: 'DL_SVC', 9: 'DL_PA'}
    heartSegDict = {1: 'DL_heart'}
    periLabelDict = {1: 'DL_Pericardium'}
    ventriLabelDict = {1: 'DL_Ventricles'}

    dcm_dir = r"L:\Data\RTOG0617\DICOM\NSCLC-Cetuximab_flat_dirs\0617-381906_09-09-2000-87022"
    niiDir = r'C:\software\DL_seg_models\CT_Heart_DeepLab'

    # Patient ID
    _, patID = os.path.split(dcm_dir)

    # Load DICOM to planC
    planC = pc.loadDcmDir(dcm_dir)

    # Scan number to segment
    origScanNum = 0

    # List of Structure names
    strNames = [s.structureName for s in planC.structure]
    numOrigStructs = len(strNames)

    # Find total lung contour
    lungInd = cerrStr.getMatchingIndex('Lung_Total', strNames, 'exact')

    # Get lung extents
    mask3M = rs.getStrMask(lungInd[0], planC)
    rmin,rmax,cmin,cmax,smin,smax,_ = bbox.computeBoundingBox(mask3M)

    # Create cropped scan
    x,y,z = planC.scan[0].getScanXYZVals()
    xCropV = x[cmin:cmax]
    yCropV = y[rmin:rmax]
    zCropV = z[smin:smax]
    scan3M = planC.scan[0].getScanArray()
    scanCrop3M = scan3M[rmin:rmax,cmin:cmax,smin:smax]
    planC = pc.importScanArray(scanCrop3M, xCropV, yCropV, zCropV, 'CT SCAN', 0, planC)

    croppedScanNum = origScanNum + 1
    ptNiiDir = os.path.join(niiDir, patID)

    # Save cropped scan to nii
    niiInDir = os.path.join(ptNiiDir, 'inNii')
    if not os.path.exists(niiInDir):
        os.mkdir(niiInDir)
    niiScanFile = os.path.join(niiInDir, patID + '_ct.nii.gz') #r'C:\software\DL_seg_models\CT_Heart_DeepLab\test_data\inNii\scan_pt1_ct.nii.gz'
    planC.scan[croppedScanNum].saveNii(niiScanFile)

    # Seg output dir
    niiOutDir = os.path.join(ptNiiDir, 'outNii')

    # Segmentation output file names and labels dictionary
    atriaSegFile = os.path.join(niiInDir, patID + '_ct_atria.nii.gz')
    heartSubSegFile = os.path.join(niiInDir, patID + '_ct_heart.nii.gz')
    heartSegFile = os.path.join(niiInDir, patID + '_ct_heartStructure.nii.gz')
    periSegFile = os.path.join(niiInDir, patID + '_ct_pericardium.nii.gz')
    ventriSegFile = os.path.join(niiInDir, patID + '_ct_ventricles.nii.gz')
    segFileAndDict = [(atriaSegFile, atriaLabelDict),(heartSubSegFile, heartSubSegDict),
                      (heartSegFile, heartSegDict), (periLabelDict, periLabelDict),
                      (ventriSegFile, ventriLabelDict)]

    # Import segmentation to cropped scan in planC
    for segFile, segDict in segFileAndDict:
        planC = pc.loadNiiStructure(segFile, croppedScanNum, planC, segDict)

    # Copy Segmentatrion to the original scan in planC
    numStructs = len(planC.structure)
    for strNum in range(numOrigStructs, numStructs):
        planC = cerrStr.copyToScan(strNum, origScanNum, planC)


    # Visualize segmentation
    scanNum = [0]
    doseNum = []
    strNum = [0,1,2]
    displayMode = '2d'
    vectDict = {}
    viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
            vwr.showNapari(scanNum, strNum, doseNum, vectDict, planC, displayMode)
