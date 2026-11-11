"""Postprocessing for MR_Rectum_GTV_SMIT:
import the GTV segmentation, keep its largest connected component, and fill holes.
"""
import os
import glob
import numpy as np
from cerr import plan_container as pc
from cerr.dataclasses import structure as cerrStr
from cerr.utils import mask as maskUtils


STR_TO_LABEL_MAP = {1: "GTV"}
LABEL_TO_STR_MAP = {value: key for key, value in STR_TO_LABEL_MAP.items()}


def postProcAndImportSeg(planC, procScanNum, scanNum, userInputs, outDir):
    """Import GTV segmentation, retain its largest connected component, and fill holes.

    Args:
        planC: pyCERR plan container from processInputData.
        procScanNum (int): Index of the (processed) scan used for inference in planC.
        scanNum (int): Index of the original scan in planC.
        userInputs (dict): userInputs of the run (not used).
        outDir (str): Directory containing the model output (NIfTI) (outDir: session_path/output/).

    Returns:
        tuple: (planC, structNumV)
            planC      - updated plan container;
            structNumV - list of the indices of the output structures in planC; run_utils.main
                         writes them to output_path in place of the model's output.
    """
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

    return planC, [int(s) for s in structsToExportV]
