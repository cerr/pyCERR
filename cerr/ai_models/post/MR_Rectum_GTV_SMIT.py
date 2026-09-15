import os
import glob
import numpy as np
from cerr import plan_container as pc
from cerr.dataclasses import structure as cerrStr
from cerr.utils import mask as maskUtils
from cerr.dcm_export import rtstruct_iod


STR_TO_LABEL_MAP = {1: "GTV"}
LABEL_TO_STR_MAP = {value: key for key, value in STR_TO_LABEL_MAP.items()}


def postProcAndImportSeg(planC, procScanNum, scanNum, userInputs, outDir):
    """Import GTV segmentation, retain its largest connected component, and fill holes.

    Args:
        planC: pyCERR plan container from processInputData.
        procScanNum (int): Index of the (processed) scan used for inference in planC.
        scanNum (int): Index of the original scan in planC.
        userInputs (dict): Must contain
            'input_path'  - original input path (DICOM dir or NIfTI file)
            'output_path' - directory for final output
        outDir (str): Directory containing the model output (NIfTI) (outDir: session_path/output/).

    Returns:
        planC: Updated plan container with the post-processed GTV structure.
    """
    inputPath = userInputs['input_path']
    outputPath = userInputs['output_path']
    numComponents = 1

    # Find NIfTI output
    niiGlob = glob.glob(os.path.join(outDir, '*.nii.gz'))
    if not niiGlob:
        raise FileNotFoundError(
            f"Error. No segmentation output files found in {outDir}."
        )

    # Import mask to planC
    gtvStructNum = len(planC.structure)
    planC = pc.loadNiiStructure(niiGlob[0], procScanNum, planC, labels_dict=LABEL_TO_STR_MAP)

    # Retain largest connected component
    strName = STR_TO_LABEL_MAP[1]
    largestCompMask3M, planC = cerrStr.getLargestConnComps(
        gtvStructNum, numComponents, planC=planC,
        saveFlag=True, replaceFlag=True, procSructName=strName)

    # Fill holes and replace the structure with the processed mask
    filledMask3M = maskUtils.fillHoles(largestCompMask3M)
    planC = cerrStr.importStructureMask(filledMask3M, scanNum, strName, planC,
                                         structNum=gtvStructNum)

    structsToExportV = np.array([gtvStructNum])

    # Export
    ptID = os.path.basename(inputPath.rstrip('/\\'))

    if os.path.isdir(inputPath):
        # to DICOM RTSTRUCT
        os.makedirs(outputPath, exist_ok=True)
        structFileName = f"{ptID}_MR_Rectum_GTV_SMIT_AI_seg.dcm"
        structFilePath = os.path.join(outputPath, structFileName)
        exportOpts = {'seriesDescription': 'AI Generated'}
        rtstruct_iod.create(structsToExportV, structFilePath, planC, exportOpts)
    else:
        # to NIfTI
        structNiiFile = os.path.join(outputPath, f"{ptID}_MR_Rectum_GTV_SMIT_AI_seg.nii.gz")
        pc.saveNiiStructure(structNiiFile, LABEL_TO_STR_MAP, planC, strNumV=structsToExportV)

    return planC
