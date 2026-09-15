import glob
import os
import json
import yaml
import numpy as np
from cerr import plan_container as pc
from cerr.dataclasses import structure as cerrStr
from cerr.dcm_export import rtstruct_iod


def postProcAndImportSeg(planC, procScanNum, scanNum, userInputs, outDir):
    """Import each GTV as a uniquely-named structure through reading JSON label-to-structure map.

    Args:
        planC: pyCERR plan container from the pre-processing step.
        procScanNum (int): Scan index of the skull-stripped scan in planC (the
                            grid seg.nii.gz is on).
        scanNum (int): Scan index of the original scan in planC.
        userInputs (dict): Must contain 'input_path' (original input), 'model_dir'
                            (installed model directory, for run_spec.yaml) and
                            'output_path' or 'session_output' (final delivery dir).
        outDir (str): Directory containing model output (session_output/),
                       including seg.nii.gz / seg_in_raw.nii.gz and the
                       structure-to-label-map JSON.

    Returns:
        planC: Updated plan container with AI GTVs.
    """
    inputPath = userInputs['input_path']
    outputPath = userInputs.get('output_path') or userInputs.get('session_output')
    modelDir = userInputs['model_dir']

    runSpecFile = os.path.join(modelDir, 'run_spec.yaml')
    with open(runSpecFile, 'r') as f:
        runSpec = yaml.safe_load(f)
    mapFileName = runSpec.get('outputs', {}).get('structureToLabelMap', 'structureToLabelMap.json')

    niiSegDir = os.path.join(outDir, 'nii_seg')
    mapFiles = glob.glob(os.path.join(niiSegDir, '*_labels.json'))
    if not mapFiles:
        raise FileNotFoundError(f"No *_labels.json found in {niiSegDir}")
    with open(mapFiles[0], 'r') as f:
        labelToName = json.load(f)
    labelsDict = {name: int(label) for label, name in labelToName.items()}

    if not labelsDict:
        # No lesions detected
        return planC

    # Copy to original MR if available, otherwise use the stripped/cropped-space.
    rawSegFiles = glob.glob(os.path.join(niiSegDir, '*_img_in_raw.nii.gz'))
    strippedSegFiles = glob.glob(os.path.join(niiSegDir, '*_img.nii.gz'))
    if rawSegFiles:
        segFile, assocScanNum = rawSegFiles[0], scanNum
    elif strippedSegFiles:
        segFile, assocScanNum = strippedSegFiles[0], procScanNum
    else:
        raise FileNotFoundError(f"No segmentation NIfTI found in {niiSegDir}")

    numOrigStructs = len(planC.structure)
    planC = pc.loadNiiStructure(segFile, assocScanNum, planC, labels_dict=labelsDict)
    structNumV = np.arange(numOrigStructs, len(planC.structure))

    # Map structures to original scan
    if assocScanNum != scanNum:
        mappedStructNumV = []
        for s in structNumV:
            planC = cerrStr.copyToScan(s, scanNum, planC)
            mappedStructNumV.append(len(planC.structure) - 1)
        structsToExportV = np.array(mappedStructNumV)
    else:
        structsToExportV = structNumV

    # Export
    ptID = os.path.basename(inputPath.rstrip('/\\'))
    if os.path.isdir(inputPath):
        os.makedirs(outputPath, exist_ok=True)
        structFileName = f"{ptID}_MR_BrainMets_SMITplus_AI_seg.dcm"
        structFilePath = os.path.join(outputPath, structFileName)
        exportOpts = {'seriesDescription': 'AI Generated'}
        rtstruct_iod.create(structsToExportV, structFilePath, planC, exportOpts)
    else:
        structNiiFile = os.path.join(outputPath, f"{ptID}_MR_BrainMets_SMITplus_AI_seg.nii.gz")
        pc.saveNiiStructure(structNiiFile, labelsDict, planC, strNumV=structsToExportV)

    return planC
