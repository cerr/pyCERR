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

    mapFile = os.path.join(outDir, mapFileName)
    if not os.path.isfile(mapFile):
        raise FileNotFoundError(f"{mapFileName} not found in {outDir}")
    with open(mapFile, 'r') as f:
        mapJson = json.load(f)
    labelsDict = {entry['structureName']: entry['value'] for entry in mapJson['strNameToLabelMap']}

    if not labelsDict:
        # No lesions detected
        return planC

    # Copy to original MR if availalbe, otherwise use the stripped/cropped-space.
    rawSegFile = os.path.join(outDir, 'seg_in_raw.nii.gz')
    strippedSegFile = os.path.join(outDir, 'seg.nii.gz')
    if os.path.isfile(rawSegFile):
        segFile, assocScanNum = rawSegFile, scanNum
    elif os.path.isfile(strippedSegFile):
        segFile, assocScanNum = strippedSegFile, procScanNum
    else:
        raise FileNotFoundError(f"No segmentation NIfTI found in {outDir}")

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
