"""Postprocessing for MR_BrainMets_SMITplus:
import each metastasis as a separately named structure, using the model's label-to-structure map.
"""
import glob
import os
import json
import yaml
import numpy as np
from cerr import plan_container as pc
from cerr.dataclasses import structure as cerrStr


def postProcAndImportSeg(planC, procScanNum, scanNum, userInputs, outDir):
    """Import each GTV as a uniquely-named structure through reading JSON label-to-structure map.

    Args:
        planC: pyCERR plan container from the pre-processing step.
        procScanNum (int): Scan index of the skull-stripped scan in planC (the
                            grid seg.nii.gz is on).
        scanNum (int): Scan index of the original scan in planC.
        userInputs (dict): Must contain 'model_dir' (installed model directory, for run_spec.yaml).
        outDir (str): Directory containing model output (session_output/),
                       including seg.nii.gz / seg_in_raw.nii.gz and the
                       structure-to-label-map JSON.

    Returns:
        tuple: (planC, structNumV)
            planC      - updated plan container;
            structNumV - list of the indices of the output structures in planC; run_utils.main
                         writes them to output_path in place of the model's output.
    """
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
        return planC, []

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

    return planC, [int(s) for s in structsToExportV]
