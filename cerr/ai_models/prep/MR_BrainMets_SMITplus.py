"""Preprocessing for MR_BrainMets_SMITplus:
N4 bias correction,  resampling to 0.5 x 0.5 x 1.0 mm, skull-stripping, cropping to the largest-CC brain bounding box.

Relies on pre-processing scripts installed in model directory (run_utils passes this in as userInputs['model_dir']).
"""
import os
import sys
import math
import importlib.util
import SimpleITK as sitk

from cerr import plan_container as pc
from cerr.utils.ai_pipeline import getScanNumFromIdentifier

VOXEL_FOR_N4 = 2.0                 # mm - downsampling for fast bias-field fit
OUTPUT_SPACING = [0.5, 0.5, 1.0]   # mm - final (stripped) image spacing

REQUIRED_INPUTS = {
    'input_path': {'required': True},
    'output_path': {'required': True},
    'session_path': {'required': True},
    'model_dir': {'required': True},
    'gpu': {'required': False}
}


def cropToStructure(images, structure, margin=0):
    """Crop images to the bounding box of the largest connected component in `structure`."""
    cc = sitk.ConnectedComponentImageFilter().Execute(structure)
    lssif = sitk.LabelShapeStatisticsImageFilter(); lssif.Execute(cc)
    if lssif.GetNumberOfLabels() == 0:
        return images
    biggestVol = 0; biggestIdx = 0
    for i in range(1, lssif.GetNumberOfLabels() + 1):
        if lssif.GetNumberOfPixels(i) > biggestVol:
            biggestVol = lssif.GetNumberOfPixels(i); biggestIdx = i
    if biggestIdx == 0:
        return images
    bb = list(lssif.GetBoundingBox(biggestIdx))
    out = []
    for img in images:
        b = list(bb)
        b[0] = max(bb[0] - margin, 0); b[1] = max(bb[1] - margin, 0); b[2] = max(bb[2] - margin, 0)
        b[3] = min(bb[3] + margin + bb[0] - b[0], img.GetSize()[0] - b[0])
        b[4] = min(bb[4] + margin + bb[1] - b[1], img.GetSize()[1] - b[1])
        b[5] = min(bb[5] + margin + bb[2] - b[2], img.GetSize()[2] - b[2])
        out.append(sitk.RegionOfInterest(img, b[3:], b[:3]))
    return out


def loadSynthStripper(modelDir, gpu):
    """Apply mri_synthstrip.py."""
    synthstripPath = os.path.join(modelDir, 'mri_synthstrip.py')
    weightsPath = os.path.join(modelDir, 'synthstrip.1.pt')
    if not os.path.isfile(synthstripPath):
        raise FileNotFoundError(f"mri_synthstrip.py not found at: {synthstripPath}")
    if not os.path.isfile(weightsPath):
        raise FileNotFoundError(f"SynthStrip weights not found at: {weightsPath}")

    spec = importlib.util.spec_from_file_location("mri_synthstrip", synthstripPath)
    mri_synthstrip = importlib.util.module_from_spec(spec)
    sys.modules["mri_synthstrip"] = mri_synthstrip
    spec.loader.exec_module(mri_synthstrip)

    return mri_synthstrip.SynthStrip(weightsPath, gpu=gpu)


def skullStrip(rawFile, strippedFile, maskFile, stripper):
    """Run N4 correction + resample + SynthStrip + crop on a raw T1c NIfTI file."""
    rawMr = sitk.Cast(sitk.ReadImage(rawFile), sitk.sitkFloat64)

    # --- N4 bias correction: fit on 2mm downsample, apply log-bias to full res ---
    oldSpacing, oldSize = rawMr.GetSpacing(), rawMr.GetSize()
    n4Size = tuple(int(math.ceil(oldSpacing[i] * oldSize[i] / VOXEL_FOR_N4)) for i in range(3))
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([VOXEL_FOR_N4] * 3); rs.SetSize(n4Size)
    rs.SetOutputOrigin(rawMr.GetOrigin()); rs.SetOutputDirection(rawMr.GetDirection())
    rs.SetInterpolator(sitk.sitkLinear)
    down = rs.Execute(rawMr)
    otsu = sitk.OtsuThreshold(down, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter(); corrector.Execute(down, otsu)
    logBias = sitk.Cast(corrector.GetLogBiasFieldAsImage(rawMr), sitk.sitkFloat64)
    corrected = rawMr / sitk.Exp(logBias)

    # --- resample to 0.5 x 0.5 x 1.0 mm ---
    cs, csz = corrected.GetSpacing(), corrected.GetSize()
    outSize = tuple(int(math.ceil(cs[i] * csz[i] / OUTPUT_SPACING[i])) for i in range(3))
    rs.SetOutputSpacing(OUTPUT_SPACING); rs.SetSize(outSize)
    rs.SetOutputOrigin(corrected.GetOrigin()); rs.SetOutputDirection(corrected.GetDirection())
    rs.SetInterpolator(sitk.sitkLinear); rs.SetTransform(sitk.Transform())
    resampled = rs.Execute(corrected)

    # --- SynthStrip (needs a file on disk) ---
    t1cTmpPath = strippedFile + '_tmp.nii.gz'
    sitk.WriteImage(resampled, t1cTmpPath)
    maskTmpPath = maskFile + '.tmp.nii.gz'
    stripper.eval(t1cTmpPath, maskFile=maskTmpPath, border=1)
    brainMask = sitk.Cast(sitk.ReadImage(maskTmpPath), sitk.sitkUInt8)

    # --- apply mask + crop to brain bbox ---
    stripped = sitk.MaskImageFilter().Execute(resampled, brainMask)
    cropStripped, cropMask = cropToStructure([stripped, brainMask], brainMask, margin=0)

    sitk.WriteImage(cropStripped, strippedFile)
    sitk.WriteImage(cropMask, maskFile)
    for f in (t1cTmpPath, maskTmpPath):
        try: os.remove(f)
        except Exception: pass


def processInputData(userInputs):
    """Load input MR scan and apply skull-stripping.

    Args:
        userInputs (dict): Must contain 'input_path' (DICOM dir or NIfTI file),
                            'session_path' (directory for temporary files) and
                            'model_dir' (installed model directory containing
                            mri_synthstrip.py and synthstrip.1.pt). Optional 'gpu'
                            (int, default 0; use -1 for CPU).

    Returns:
        tuple: (planC, procScanNum, scanNum, sessionUserInputs)
            planC       - plan container with original and skull-stripped scans
            procScanNum - scan index of skull-stripped scan in planC
            scanNum     - scan index of original scan in planC
            sessionUserInputs - userInputs updated with session input/output paths
    """
    inputPath = userInputs['input_path']
    sessionPath = userInputs['session_path']
    modelDir = userInputs['model_dir']
    gpu = userInputs.get('gpu', 0)
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

    # Export raw scan to session dir for SimpleITK processing
    ptID = os.path.basename(inputPath.rstrip('/\\'))
    rawNiiFile = os.path.join(modInputPath, f"{ptID}_raw_t1c.nii.gz")
    planC.scan[scanNum].saveNii(rawNiiFile)

    # Skull-strip
    stripper = loadSynthStripper(modelDir, gpu)
    strippedFile = os.path.join(modInputPath, f"{ptID}_scan_3D.nii.gz")
    maskFile = os.path.join(modInputPath, f"{ptID}_brainmask.nii.gz")
    skullStrip(rawNiiFile, strippedFile, maskFile, stripper)

    # Import skull-stripped scan into planC
    planC = pc.loadNiiScan(strippedFile, imageType=modality + ' SCAN', initplanC=planC)
    procScanNum = len(planC.scan) - 1

    sessionUserInputs = userInputs.copy()
    sessionUserInputs['input_path'] = modInputPath
    sessionUserInputs['output_path'] = modOutputPath
    return planC, procScanNum, scanNum, sessionUserInputs
