import glob
import os

import SimpleITK as sitk
import numpy as np
import subprocess
import sys

import cerr.plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.dataclasses import scan as cerrScn
from cerr.dataclasses import structure as cerrStr
from cerr.dcm_export import rtstruct_iod
from cerr.utils import mask
from cerr.utils.ai_pipeline import createSessionDir, getScanNumFromIdentifier
from cerr.utils.image_proc import resizeScanAndMask


# Map output labels to structure names
structToLabelMap = {1: "Liver",
                    2: "Bowel_Lg",
                    3: "Bowel_Sm",
                    4: "Duostomach"}

segStructList = list(structToLabelMap.values())
labels = list(structToLabelMap.keys())
numAIStrs = len(labels)


def writeFile(data, dirName, fileName, inputImg):
    """ Write mask to NIfTI file """
    outFile = os.path.join(dirName, fileName + '.nii.gz')
    data = np.flip(data, 2)
    dataImg = sitk.GetImageFromArray(data)
    dataImg.CopyInformation(inputImg)
    sitk.WriteImage(dataImg, outFile)


def processInputData(scanIdx, outlineStructName, structToLabelMap, planC):
    """Pre-process scan and mask for input to model"""

    # --------------------------------------------------
    #          Identify input structure indices
    # ---------------------------------------------------
    structList = [str.structureName for str in planC.structure]
    if structToLabelMap is not None:

        segStructList = list(structToLabelMap.values())
        labels = list(structToLabelMap.keys())
        structNumV = np.zeros(len(labels), dtype=int)

        for numLabel in range(len(labels)):
            structName = structToLabelMap[labels[numLabel]]
            matchIdxV = cerrStr.getMatchingIndex(structName,
                                                 structList,
                                                 matchCriteria='exact')
            structNumV[numLabel] = matchIdxV[0]

        # Extract label map
        print(structNumV)
        maskList = cerrStr.getMaskList(structNumV,
                                       planC,
                                       labelDict=structToLabelMap)

        mask4M = np.array(maskList)
        mask4M = np.moveaxis(mask4M, 0, -1)  # nRows x nCols x nSlc x nlabels
    else:
        mask4M = None

    # --------------------------------------------------
    #          Process input scan
    # ---------------------------------------------------
    modality = 'MR'
    scan3M = planC.scan[scanIdx].getScanArray()
    scanSizeV = np.shape(scan3M)

    # 1. Crop to  patient outline

    ## Extract outline
    outlineIdx = structList.index(outlineStructName) \
        if outlineStructName in structList else None

    if outlineIdx is None:
        # Generate outline mask
        threshold = 0.03
        outline3M = mask.getPatientOutline(scan3M,
                                           threshold,
                                           normFlag=True)

        planC = pc.importStructureMask(outline3M,
                                       scanIdx,
                                       outlineStructName,
                                       planC,
                                       None)
    else:
        # Load outline mask
        outline3M = rs.getStrMask(outlineIdx, planC)

    ## Crop scan
    cropMask4M = None
    minr, maxr, minc, maxc, mins, maxs, _ = mask.computeBoundingBox(outline3M)
    cropScan3M = scan3M[minr:maxr, minc:maxc, mins:maxs]
    cropScanSizeV = np.shape(cropScan3M)
    if mask4M is not None:
        cropMask4M = mask4M[minr:maxr, minc:maxc, mins:maxs, :]

    ##    Crop grid
    gridS = planC.scan[scanIdx].getScanXYZVals()
    cropGridS = (gridS[0][minc:maxc],
                 gridS[1][minr:maxr],
                 gridS[2][mins:maxs])

    # 2. Resize scan
    ## Crop scan in-plane
    outputImgSizeV = [128, 192, cropScanSizeV[2]]
    method = 'bicubic'
    procScan3M, procMask4M, resizeGridS = resizeScanAndMask(cropScan3M,
                                                            cropMask4M,
                                                            cropGridS,
                                                            outputImgSizeV,
                                                            method)
    ## Pad scan along slices
    resizeScanShape = procScan3M.shape
    outputImgSizeV = [resizeScanShape[0], resizeScanShape[1], 128]
    method = 'padslices'
    procPadScan3M, procPadMask4M, padGridS = resizeScanAndMask(procScan3M, \
                                                               procMask4M, \
                                                               resizeGridS, \
                                                               outputImgSizeV, \
                                                               method)

    # --------------------------------------------------
    #    Import processed scan & mask to planC
    # ---------------------------------------------------
    ## Import scan
    planC = pc.importScanArray(procPadScan3M,
                               padGridS[0],
                               padGridS[1],
                               padGridS[2],
                               modality,
                               scanIdx,
                               planC)
    processedScanIdx = len(planC.scan) - 1

    ## Import mask
    processedStrIdxV = []
    if procPadMask4M is not None:
        for structIndex in range(len(segStructList)):
            structName = segStructList[structIndex]
            procPadMask3M = procPadMask4M[:, :, :, structIndex]
            planC = pc.importStructureMask(procPadMask3M,
                                           processedScanIdx,
                                           structName,
                                           planC,
                                           None)
            processedStrIdxV.append(len(planC.structure) - 1)
    else:
        processedStrIdxV = None

    return processedScanIdx, processedStrIdxV, padGridS


def reverseTransformations(data3M, dataType, numDims, scanNum, outSizeDict, inputGridS, planC):
    ## Undo padding
    outputScan3M = None
    resizedDimsV = outSizeDict['pad']
    method = 'unpadslices'
    _, unPadData3M, unPadGridS = resizeScanAndMask(outputScan3M,
                                                   data3M,
                                                   inputGridS,
                                                   resizedDimsV,
                                                   method)
    ## Undo resizing
    outputImgSizeV = outSizeDict['resize']
    if dataType == 'mask':
        method = 'nearest'
    else:
        method = 'bicubic'
    _, resizeData3M, resizeGridS = resizeScanAndMask(outputScan3M,
                                                     unPadData3M,
                                                     unPadGridS,
                                                     outputImgSizeV,
                                                     method)

    ## Undo cropping
    [minr, maxr, minc, maxc, mins, maxs] = outSizeDict['bbox']
    baseImgSizeV = list(planC.scan[scanNum].getScanSize())
    fullData4M = np.zeros(baseImgSizeV + [numAIStrs])
    fullData4M[minr:maxr, minc:maxc, mins:maxs, :] = resizeData3M

    return fullData4M


def postProcAndImportResults(modOutputPath, dataType, baseScanIdx, inputGridS,
                             outlineStructName, planC, structToLabelMap=None):
    """ Reverse pre-processing transformations and import auto-segmentations to planC"""

    # --------------------------------------------------
    #      Undo pre-processing transformations
    # ---------------------------------------------------
    ## Extract extents of patient outline
    baseImgSizeV = list(planC.scan[baseScanIdx].getScanSize())
    structList = [struct.structureName for struct in planC.structure]
    outlineIdx = structList.index(outlineStructName)
    outline3M = rs.getStrMask(outlineIdx, planC)
    minr, maxr, minc, maxc, mins, maxs, _ = mask.computeBoundingBox(outline3M)
    [xValsV, yValsV, zValsV] = planC.scan[baseScanIdx].getScanXYZVals()
    physExtentsV = [yValsV[minr], yValsV[maxr],
                    xValsV[minc], xValsV[maxc],
                    zValsV[mins], zValsV[maxs]]

    nSlc = maxs - mins
    padDimsV = [128, 192, nSlc]
    outputImgSizeV = [maxr - minr, maxc - minc, nSlc]
    cropDimsV = [minr, maxr, minc, maxc, mins, maxs]
    outSizeDict = {'pad': padDimsV, 'resize': outputImgSizeV, 'bbox': cropDimsV}

    if dataType == 'mask':
        # Read AI-generated mask
        niiGlob = glob.glob(os.path.join(modOutputPath, '*.nii.gz'))
        print('Importing ' + niiGlob[0] + '...')
        outputMask = sitk.ReadImage(niiGlob[0])
        outputMask3M = sitk.GetArrayFromImage(outputMask)
        numStrOrig = len(planC.structure)
        numAIStrs = len(structToLabelMap)
        # Reverse transformations
        fullData4M = reverseTransformations(outputMask3M, dataType, numAIStrs,
                                            baseScanIdx, outSizeDict, inputGridS, planC)
        # Import to planC
        numComponents = 1
        for numLabel in range(numAIStrs):
            binMask = fullData4M[:, :, :, numLabel]

        structName = 'AI_' + structToLabelMap[numLabel + 1]
        planC = cerrStr.importStructureMask(binMask,
                                            baseScanIdx,
                                            structName,
                                            planC,
                                            None)

        # Post-process and replace input structure in planC
        structNumV = len(planC.structure) - 1
        importMask3M, planC = cerrStr.getLargestConnComps(structNumV,
                                                          numComponents,
                                                          planC,
                                                          saveFlag=True,
                                                          replaceFlag=True,
                                                          procSructName=structName)
    elif dataType == 'dvf':
        niiGlob = glob.glob(os.path.join(modOutputPath, '*.nii.gz'))
        print('Importing ' + niiGlob[0] + '...')
        outputDVF = sitk.ReadImage(niiGlob[0])
        outputDVF4M = sitk.GetArrayFromImage(outputDVF)
        outputSizeV = outputDVF4M.shape
        fullData4M = np.zeros([baseImgSizeV + [3]])  # DVF on orig scan

        for nDim in range(outputDVF4M.shape[3]):

            DVF3M = np.squeeze(outputDVF4M[:, :, :, nDim])
            procDVF3M = reverseTransformations(DVF3M, dataType, 3,
                                               baseScanIdx, outSizeDict, inputGridS, planC)
            if nDim == 1:
                scaleFactor = abs(physExtentsV[1] - physExtentsV[0]) / (outputSizeV[1] - 1)
            elif nDim == 0:
                scaleFactor = abs(physExtentsV[3] - physExtentsV[2]) / (outputSizeV[2] - 1)
            else:
                DVF3M = -DVF3M  # Reverse Z axis
                scaleFactor = abs(physExtentsV[5] - physExtentsV[4]) / (outputSizeV[3] - 1)

            procDVF3M = procDVF3M * scaleFactor * 10  # Convert to mm
            fullData4M[:, :, :, nDim] = procDVF3M

    return fullData4M, planC

def main(inputDicomPath, outputDicomPath, sessionPath, condaEnvPath):
    """
    Function to apply pretrained model
    Args:
        inputDicomPath (string): Path to input DICOM data structured as:
            inputDicomPath
                            | ------Pat1
                                        | ---Week1
                                                 | ------REG_img1.dcm
                                                         REG_img2.dcm
                                                         .
                                                         .
                                                         .
                                                         REG_RTSRTUCT
                                        | ---Week2
                                                | ------img1.dcm
                                                        img2.dcm
                                                        .
                                                        .
                                                        .

        outputDicomPath (string):  Path to write DICOM outputs
        sessionPath (string)    :  Path to directory for writing temporary files
        condaEnvPath (string)   :  Path to conda env containing python libraries, pre-trained model,
                                           and inference script

    """

    # Create output and session dirs
    if not os.path.exists(outputDicomPath):
        os.mkdir(outputDicomPath)

    if not os.path.exists(sessionPath):
        os.mkdir(sessionPath)

    ptList = os.listdir(inputDicomPath)
    inputSubDirs = ['Masks']

    # Paths to activation and inference scripts
    activateScript = os.path.join(condaEnvPath, 'bin', 'activate')
    scriptPath = os.path.join(condaEnvPath, 'MRI_Pancreas_Fullshot_AnatomicCtxShape',
                              'model_wrapper', 'run_inference_first_to_last_nii.py')

    # Create session dir to store temporary data
    modInputPath, modOutputPath = createSessionDir(sessionPath,
                                                   inputDicomPath,
                                                   inputSubDirs)

    # Import DICOM scan to planC
    planC = pc.loadDcmDir(inputDicomPath)
    numExistingStructs = len(planC.structure)

    # ------------------------------
    #  Pre-processing transformations
    # ------------------------------

    scanNum = 0
    modality = 'MR'
    outlineStructName = 'patient_outline'

    # 1. Base scan
    identifier = {"seriesDate": "last"}
    baseScanIdx = getScanNumFromIdentifier(identifier, planC)[0]
    exportLabelMap = None
    procBaseScanIdx, __, procBaseGridS = processInputData(baseScanIdx,
                                                          outlineStructName,
                                                          exportLabelMap,
                                                          planC)

    # 2. Moving scan
    identifier = {"seriesDate": "first"}
    movScanIdx = getScanNumFromIdentifier(identifier, planC)[0]
    exportLabelMap = structToLabelMap
    procMovScanIdx, procMovStrIdxV, procMovGridS = \
        processInputData(movScanIdx,
                         outlineStructName,
                         exportLabelMap,
                         planC)

    # Export processed inputs to NIfTI
    inputFile = ptList[0]
    baseScanFile = os.path.join(modInputPath,
                                f"{inputFile}_MR SCAN_last_scan_3D.nii.gz")
    movScanFile = os.path.join(modInputPath,
                               f"{inputFile}_MR SCAN_first_scan_3D.nii.gz")
    movMaskFile = os.path.join(modInputPath, inputSubDirs[0], \
                               f"{inputFile}_MR SCAN_first_4D.nii.gz")
    planC.scan[procBaseScanIdx].saveNii(baseScanFile)
    planC.scan[procMovScanIdx].saveNii(movScanFile)
    pc.saveNiiStructure(movMaskFile,
                        procMovStrIdxV,
                        planC,
                        labelDict=structToLabelMap,
                        dim=4)

    # Apply pretrained AI model
    subprocess.run(f"source {activateScript} && python {scriptPath} \
                        {modInputPath} {modOutputPath}",
                   capture_output=False, shell=True, executable="/bin/bash")

    # Import segmentations to planC
    __, planC = postProcAndImportResults(modOutputPath, 'mask',
                                         baseScanIdx, procMovGridS,
                                         outlineStructName, planC, structToLabelMap)

    newNumStructs = len(planC.structure)

    # Locate DVF file
    modDVFpath = os.path.join(modOutputPath, 'DVF')
    modDVFfiles = [f for f in os.listdir(modDVFpath) if f.endswith('.nii.gz')]
    if not modDVFfiles:
        raise FileNotFoundError("No NIfTI DVF file found in output directory.")
    DVFfile = os.path.join(modDVFpath, modDVFfiles[0])
    procDVF4M, planC = postProcAndImportResults(modDVFpath, 'DVF',
                                                baseScanIdx, procMovGridS,
                                                outlineStructName, planC)
    writeFile(procDVF4M, modDVFpath, modDVFfiles[0], movScanFile)

    # Export segmentations to DICOM
    structFileName = inputFile + '_AI_seg.dcm'
    structFilePath = os.path.join(outputDicomPath, structFileName)
    structNumV = np.arange(numExistingStructs + 1, newNumStructs)
    indOrigV = np.array([cerrScn.getScanNumFromUID( \
        planC.structure[structNum].assocScanUID, planC) \
        for structNum in structNumV], dtype=int)
    structsToExportV = structNumV[indOrigV == baseScanIdx]
    seriesDescription = "AI Generated"
    exportOpts = {'seriesDescription': seriesDescription}
    rtstruct_iod.create(structsToExportV,
                        structFilePath,
                        planC,
                        exportOpts)

    # Export DVF to DICOM
    DVFFileName = ptList[0] + '_AI_DVF.dcm'
    DFVDCMfile = os.path.join(outputDicomPath, DVFFileName)
    scriptPath = os.path.join(condaEnvPath, 'reg2dcm/reg2dcm.py')
    reg2dcm_cmd = ['python ', scriptPath, ' -b ', movScanFile, ' -m ', baseScanFile, ' -d ',
                   DVFfile , ' -o ', DFVDCMfile]
    subprocess.run(['source activate ', condaEnvPath, ' && ', reg2dcm_cmd])


    return 0

if __name__ == '__main__':
    inputDicomPath = sys.argv[1]
    outputDicomPath = sys.argv[2]
    sessionPath = sys.argv[3]
    condaEnvPath = sys.argv[4]
    main(inputDicomPath, outputDicomPath, sessionPath, condaEnvPath)