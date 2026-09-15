import os
from cerr import plan_container as pc
from cerr.utils.ai_pipeline import getScanNumFromIdentifier

REQUIRED_INPUTS = {
    'input_path': {'required': True},
    'output_path': {'required': True},
    'session_path': {'required': True}
}


def processInputData(userInputs):
    """Load input MR scan and export it to NIfTI for model inference.

    Args:
        userInputs (dict): Must contain 'input_path' (DICOM dir or NIfTI file)
                           and 'session_path' (directory for temporary files)

    Returns:
        tuple: (planC, procScanNum, scanNum, sessionUserInputs)
            planC       - plan container with the loaded scan
            procScanNum - scan index used for inference (same as scanNum)
            scanNum     - scan index of original scan in planC
            sessionUserInputs - userInputs updated with session input/output paths
    """
    inputPath = userInputs['input_path']
    sessionPath = userInputs['session_path']
    modality = 'MR'

    # Create session input/output dirs
    modInputPath = os.path.join(sessionPath, 'input')
    modOutputPath = os.path.join(sessionPath, 'output')
    os.makedirs(modInputPath, exist_ok=True)
    os.makedirs(modOutputPath, exist_ok=True)

    # Load input into planC
    if os.path.isdir(inputPath):
        planC = pc.loadDcmDir(inputPath)
    elif inputPath.endswith('.nii') or inputPath.endswith('.nii.gz'):
        planC = pc.loadNiiScan(inputPath, imageType='MR SCAN')
    else:
        raise ValueError(f"Unsupported input path: {inputPath}. "
                         f"Must be a DICOM directory or NIfTI file.")

    # Identify MR scan
    scanIdS = {'imageType': 'MR SCAN'}
    matchScanV = getScanNumFromIdentifier(scanIdS, planC, False)
    scanNum = matchScanV[0]
    procScanNum = scanNum

    # Export scan to session dir input
    ptID = os.path.basename(inputPath.rstrip('/\\'))
    scanNiiFile = os.path.join(modInputPath, f"{ptID}_scan_3D.nii.gz")
    planC.scan[scanNum].saveNii(scanNiiFile)

    sessionUserInputs = userInputs.copy()
    sessionUserInputs['input_path'] = modInputPath
    sessionUserInputs['output_path'] = modOutputPath
    return planC, procScanNum, scanNum, sessionUserInputs
