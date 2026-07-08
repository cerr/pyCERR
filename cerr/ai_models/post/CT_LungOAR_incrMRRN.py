import os
import glob
import numpy as np
from cerr import plan_container as pc
from cerr.dataclasses import scan as cerrScn
from cerr.dataclasses import structure as cerrStr
from cerr.utils import mask
from cerr.dcm_export import rtstruct_iod


STR_TO_LABEL_MAP = {
    1: "Lung_Left",
    2: "Lung_Right",
    3: "Heart",
    4: "Esophagus",
    5: "Cord",
    6: "PBT"
}
LABEL_TO_STR_MAP = {value: key for key, value in STR_TO_LABEL_MAP.items()}


def postProcAndImportSeg(planC, procScanNum, scanNum, userInputs, outDir):
    """Largest connected component is retained per structure.

    Args:
        planC: pyCERR plan container from processInputData.
        procScanNum (int): Scan index of resized scan in planC.
        scanNum (int): Scan index of original scan in planC.
        userInputs (dict): Must contain
            'input_path'  - original input path (DICOM dir or NIfTI file)
            'output_path' - directory for final output
        outDir (str): Directory containing model NIfTI output (session_path/output/).

    Returns:
        planC: Updated plan container with imported and post-processed structures.
    """
    inputPath = userInputs['input_path']
    outputPath = userInputs['output_path']
    numLabel = len(STR_TO_LABEL_MAP)
    numComponents = 1

    # Find NIfTI output
    niiGlob = glob.glob(os.path.join(outDir, '*.nii.gz'))
    if not niiGlob:
        raise FileNotFoundError(
            f"No .nii.gz output files found in {outDir}."
        )

    # Import label map to planC on resized scan
    numStrOrig = len(planC.structure)
    planC = pc.loadNiiStructure(niiGlob[0], procScanNum, planC,labels_dict=LABEL_TO_STR_MAP)
    cpyStrNumV = np.arange(numStrOrig, len(planC.structure))
    numExistingStructs = numStrOrig

    # Copy structures to original scan and retain largest connected component
    for label in range(numLabel):
        planC = cerrStr.copyToScan(cpyStrNumV[label], scanNum, planC)
        origStr = len(planC.structure) - 1
        strName = STR_TO_LABEL_MAP[label + 1]
        __, planC = cerrStr.getLargestConnComps(origStr, numComponents, planC=planC,
                           saveFlag=True, replaceFlag=True, procSructName=strName)

    newNumStructs = len(planC.structure)
    structNumV = np.arange(numExistingStructs, newNumStructs)

    # Determine which structures are on the original scan
    indOrigV = np.array([cerrScn.getScanNumFromUID(planC.structure[s].assocScanUID, planC) for s in structNumV],
                        dtype=int)
    structsToExportV = structNumV[indOrigV == scanNum]

    for s in structsToExportV:
        name = planC.structure[s].structureName
        print(f"struct {s}: name={name!r}  type={type(name)}")

    # Export
    ptID = os.path.basename(inputPath.rstrip('/'))
    if os.path.isdir(inputPath):
        # to DICOM RTSTRUCT
        os.makedirs(outputPath, exist_ok=True)
        structFileName = f"{ptID}_CT_LungOAR_incrMRRN_AI_seg.dcm"
        structFilePath = os.path.join(outputPath, structFileName)
        exportOpts = {'seriesDescription': 'AI Generated'}
        rtstruct_iod.create(structsToExportV, structFilePath, planC, exportOpts)
    else:
        # to NIfTI
        labelDict = {v: k for k, v in STR_TO_LABEL_MAP.items()}
        structNiiFile = os.path.join(outputPath, f"{ptID}_CT_LungOAR_incrMRRN_AI_seg.nii.gz")
        pc.saveNiiStructure(structNiiFile, labelDict, planC, strNumV=structsToExportV)

    return planC