import os
import numpy as np
from cerr import plan_container as pc
from cerr.utils.ai_pipeline import getScanNumFromIdentifier
from cerr.utils import image_proc


REQUIRED_INPUTS = {
    'input_path': {'required': True},
    'output_path': {'required': True},
    'session_path': {'required': True}
}

def processInputData(userInputs):
    """Load input scan, resize to 512x512 and return processed planC.

    Args:
        userInputs (dict): Must contain 'input_path' (DICOM dir or NIfTI file)
                           and 'session_path' (directory for temporary files)

    Returns:
        tuple: (planC, procScanNum, scanNum)
            planC       - plan container with original and resized scans
            procScanNum - scan index of resized scan in planC
            scanNum     - scan index of original scan in planC
    """
    inputPath = userInputs['input_path']
    sessionPath = userInputs['session_path']
    modality = 'CT'

    # Create session input dir
    modInputPath = os.path.join(sessionPath, 'input')
    os.makedirs(modInputPath, exist_ok=True)

    # Load input into planC
    if os.path.isdir(inputPath):
        planC = pc.loadDcmDir(inputPath)
    elif inputPath.endswith('.nii') or inputPath.endswith('.nii.gz'):
        planC = pc.loadNiiScan(inputPath, imageType='CT SCAN')
    else:
        raise ValueError(f"Unsupported input path: {inputPath}. "
                         f"Must be a DICOM directory or NIfTI file.")

    # Identify CT scan
    scanIdS = {'imageType': 'CT SCAN'}
    matchScanV = getScanNumFromIdentifier(scanIdS, planC, False)
    scanNum = matchScanV[0]

    # Extract scan
    scan3M = planC.scan[scanNum].getScanArray()
    mask3M = None
    gridS = planC.scan[scanNum].getScanXYZVals()

    # Resize to 512x512
    inputImgSizeV = np.shape(scan3M)
    outputImgSizeV = [512, 512, inputImgSizeV[2]]
    method = 'padorcrop3d'
    procScan3M, __, resizeGridS = image_proc.resizeScanAndMask(scan3M, mask3M, gridS, outputImgSizeV, method)

    # Import resized scan into planC
    planC = pc.importScanArray(procScan3M, resizeGridS[0], resizeGridS[1], resizeGridS[2], modality, scanNum, planC)
    procScanNum = len(planC.scan) - 1

    # Export resized scan to session dir input
    ptID = os.path.basename(inputPath.rstrip('/\\'))
    scanNiiFile = os.path.join(modInputPath, f"{ptID}_scan_3D.nii.gz")
    planC.scan[procScanNum].saveNii(scanNiiFile)

    return planC, procScanNum, scanNum


