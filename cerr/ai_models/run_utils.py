"""Module for running pretrained AI models from CERR model repositories (e.g. https://github.com/cerr/smit).

The model runs in its repository's uv environment as

    <entrypoint> --model <MODEL> --data_dir <input dir> --results_dir <output dir> [--<key> <value> ...]

and writes <case>_seg.nii.gz label maps or labels.json (label value -> structure name) to the output dir.
"""
import os
import re
import sys
import json
import shutil
import importlib.util
import subprocess

from pathlib import Path
from cerr import plan_container as pc
from cerr.dcm_export import rtstruct_iod
from cerr.ai_models.install_utils import (getModel, getRepoPath, getEnvPython,
                                          getWeightsPath, validateModelName)

# userInputs keys consumed by pyCERR; all other keys are passed to the model as --<key> <value>
RESERVED_INPUTS = ('input_path', 'output_path', 'session_path', 'model_dir', 'gpu')
SEG_PATTERN = '*_seg.*'
OUTPUT_EXTS_REGEX = r'\.(?:nii\.gz|dcm)'  # what a trailing '.*' in an output file pattern matches


def main(model, installDir, userInputs, verbose=False):
    """
        Run pretrained AI model.

    Args:
        model (int or str): Model number or name (see install_utils.listModels).
        installDir (str): Path to model install dir.
        userInputs: Dictionary of arguments provided by the user:
            'input_path'   - Path to DICOM or NIfTI inputs;
            'output_path'  - Results directory;
            'session_path' - [optional] Scratch directory
            'gpu'          - [optional] GPU index, sets CUDA_VISIBLE_DEVICES for the model;
            any other key is passed to the model as --<key> <value> (True: --<key>; False/None: omitted).
        verbose (bool): [optional, default:False] Print stdout if True.

    Returns:
        subprocess.CompletedProcess: the model run.
    """
    modelName, repo, entrypoint = getModel(model)
    repoPath = getRepoPath(repo, installDir)
    validateModelName(modelName, repoPath)
    if not getWeightsPath(repoPath, modelName) and 'pretrained_model_path' not in userInputs:
        raise FileNotFoundError(
            f"Weights for {modelName} are not installed. Install them with "
            f"ai_models.install({modelName!r}, installDir, weightsUrl='<link you received>') "
            f"(or weightsFile='<downloaded file>').")
    modelPath = repoPath.as_posix()
    planC = None
    procScanNum = None
    scanNum = None
    sessionUserInputs = None

    # Apply pre-processing
    prepPath = getProcScript('prep', modelName)
    if prepPath:
        preproc = loadModule("preproc", prepPath)
        prepUserInputs = {**userInputs, 'model_dir': modelPath}
        planC, procScanNum, scanNum, sessionUserInputs = preproc.processInputData(prepUserInputs)

    # Build model run command
    runInputs = sessionUserInputs if sessionUserInputs else userInputs
    cmd, env = buildCommand(repoPath, modelName, runInputs, entrypoint)
    sessionOutDir = runInputs['output_path']
    os.makedirs(sessionOutDir, exist_ok=True)

    # Apply the model
    print(f"Running {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=modelPath, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Model inference failed.\nSTDERR:\n{result.stderr}")
    if verbose:
        print(result.stdout)

    # Apply post-processing
    outputPath = str(userInputs['output_path'])
    postPath = getProcScript('post', modelName)
    if postPath:
        postproc = loadModule("postproc", postPath)
        postUserInputs = {**userInputs, 'model_dir': modelPath}
        planC, structNumV = postproc.postProcAndImportSeg(planC, procScanNum, scanNum,
                                                          postUserInputs, sessionOutDir)
        # Re-export to output_path
        inputPath = str(userInputs['input_path'])
        labelDict = readLabels(sessionOutDir, modelName, installDir)
        fileName = getCaseName(inputPath)
        exportSeg(planC, structNumV, fileName, os.path.isdir(inputPath), modelName,
                  labelDict, outputPath)

    elif os.path.abspath(sessionOutDir) != os.path.abspath(outputPath):
        # copy the model's output from the session directory
        shutil.copytree(sessionOutDir, outputPath, dirs_exist_ok=True)

    return result


def getCaseName(path):
    """Case name of an input: the DICOM directory or NIfTI file name, without .nii/.nii.gz.

    Args:
        path (str): DICOM directory or NIfTI file.

    Returns:
        str: case name.
    """
    return re.sub(r'\.nii(\.gz)?$', '', os.path.basename(path.rstrip('/\\')))


def exportSeg(planC, structNumV, caseName, isDicom, modelName, labels, outputPath):
    """Write structures to outputPath or a label map with labels.json.

    Args:
        planC (plan_container.PlanC): pyCERR plan container.
        structNumV (list): indices of the structures to write.
        caseName (str): case name, for the output file names.
        isDicom (bool): write RTSTRUCTs (DICOM input) instead of NIfTI label maps.
        modelName (str): model name, for <case>_<MODEL>_RTSTRUCT.dcm.
        labels (dict): output file pattern -> {label value: structure name} (see normalizeLabels).
        outputPath (str): output directory.
    """
    os.makedirs(outputPath, exist_ok=True)
    names = [planC.structure[s].structureName for s in structNumV]
    exported = set()
    for pattern, fileLabels in labels.items():
        nameToLabel = {name: label for label, name in fileLabels.items()}
        fileStructV = [s for s, name in zip(structNumV, names) if name in nameToLabel]
        if not fileStructV:
            continue
        if isDicom:
            fileName = (f'{caseName}_{modelName}_RTSTRUCT.dcm' if pattern == SEG_PATTERN
                        else pattern[:-2].replace('*', caseName) + '.dcm')
            rtstruct_iod.create(fileStructV, os.path.join(outputPath, fileName), planC,
                                {'seriesDescription': 'AI Generated'})
        else:
            fileName = pattern[:-2].replace('*', caseName) + '.nii.gz'
            pc.saveNiiStructure(os.path.join(outputPath, fileName), nameToLabel, planC, strNumV=fileStructV)
        exported.update(fileStructV)
    if not isDicom:
        rawLabels = {str(k): v for k, v in labels[SEG_PATTERN].items()} if list(labels) == [SEG_PATTERN] else \
            {pattern: {str(k): v for k, v in fileLabels.items()} for pattern, fileLabels in labels.items()}
        with open(os.path.join(outputPath, 'labels.json'), 'w') as f:
            json.dump({'model': modelName, 'labels': rawLabels}, f, indent=2)
    skipped = [names[i] for i, s in enumerate(structNumV) if s not in exported]
    if skipped:
        print(f"Warning: structures not in the {modelName} labels were not written: {', '.join(skipped)}")


def getProcScript(kind, modelName):
    """pyCERR pre- or post-processing script for a model, or None.

    Args:
        kind (str): 'prep' or 'post'.
        modelName (str): model name.

    Returns:
        pathlib.Path or None: cerr/ai_models/<kind>/<modelName>.py, if it exists.
    """
    path = Path(__file__).parent / kind / f'{modelName}.py'
    return path if path.is_file() else None


def loadModule(name, path):
    """
    Load model-specific pre/post-processing module(s).

    Args:
        name (str): module name.
        path (str or pathlib.Path): module file.

    Returns:
        module: the loaded module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stageInput(inputPath, userInputs):
    """Directory to pass to the model as --data_dir.

    A single NIfTI file is copied to <session_path>/input (or <output_path>/input), since the
    model segments every scan in a folder.

    Args:
        inputPath (str): DICOM directory, folder of inputs, or NIfTI file.
        userInputs (dict): userInputs of the run ('session_path' or 'output_path' for the copy).

    Returns:
        str: directory to pass as --data_dir.
    """
    if os.path.isdir(inputPath):
        return inputPath
    if not (inputPath.endswith('.nii') or inputPath.endswith('.nii.gz')):
        raise ValueError(f"Unsupported input path: {inputPath}. "
                         f"Must be a DICOM directory or a NIfTI file.")
    if not os.path.isfile(inputPath):
        raise FileNotFoundError(f"Input file not found: {inputPath}")
    stageDir = os.path.join(userInputs.get('session_path') or userInputs['output_path'], 'input')
    os.makedirs(stageDir, exist_ok=True)
    shutil.copy(inputPath, os.path.join(stageDir, os.path.basename(inputPath)))
    return stageDir


def additionalInputs(entrypoint, option):
    """True if the entrypoint script defines a command-line option (e.g. '--session_dir').

    Args:
        entrypoint (str or pathlib.Path): model entrypoint script.
        option (str): command-line option, e.g. '--session_dir'.

    Returns:
        bool: True if the option appears quoted in the script source.
    """
    try:
        source = Path(entrypoint).read_text(errors='ignore')
    except OSError:
        return False
    return f"'{option}'" in source or f'"{option}"' in source


def buildCommand(repoPath, modelName, userInputs, entrypoint):
    """
        Constructs the subprocess command to run the model, based on user inputs.

    Args:
        repoPath (pathlib.Path): Path to the installed model repository.
        modelName (str): Model name from models.yaml.
        userInputs: Dictionary of arguments provided by the user (see main).
        entrypoint (str): Model's inference script, relative to repoPath.

    Returns:
        tuple: (command list, environment dict) for subprocess.run
    """
    repoPath = Path(repoPath).resolve()  # commands run with cwd=repoPath
    pythonExe = getEnvPython(repoPath)
    if not pythonExe.exists():
        raise FileNotFoundError(f"Python binary not found in the uv env at: {pythonExe}")
    entrypoint = repoPath / entrypoint
    if not entrypoint.exists():
        raise FileNotFoundError(f"Model entrypoint not found at: {entrypoint}")

    for argName in ('input_path', 'output_path'):
        if argName not in userInputs:
            raise ValueError(f"Missing required argument: {argName}")
    dataDir = stageInput(str(userInputs['input_path']), userInputs)

    cmd = [pythonExe.as_posix(), entrypoint.as_posix(), '--model', modelName,
           '--data_dir', dataDir, '--results_dir', str(userInputs['output_path'])]
    if userInputs.get('session_path') and additionalInputs(entrypoint, '--session_dir'):
        cmd.extend(['--session_dir', str(userInputs['session_path'])])

    # Pass the remaining inputs as model options
    for argName, val in userInputs.items():
        if argName in RESERVED_INPUTS or val is None or val is False:
            continue
        flag = f"--{argName}"
        if val is True:
            cmd.append(flag)
        elif isinstance(val, (list, tuple)):
            cmd.extend([flag, *[str(v) for v in val]])
        else:
            cmd.extend([flag, str(val)])

    # Locate the model env
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
    if userInputs.get('gpu') is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(userInputs['gpu'])
    if sys.platform == 'darwin':
        # Apple-silicon GPU: run operations MPS lacks on the CPU
        env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

    return cmd, env


def normalizeLabels(labels):
    """Output file pattern -> {label value: structure name} from either form of a model's labels:

        {label1: 'struct_name1', label2: 'struct_name2'}              one label map per case, <case>_seg.nii.gz (OR)
        {'*_file_pattern1.*': {label1: 'struct_name1'},               several outputs per case, '*' standing for
         '*_file_pattern2.*': {label2: 'struct_name2'}, ...}          the case name and '.*' for .nii.gz or .dcm

    Args:
        labels (dict): labels from models.yaml or labels.json, in either form.

    Returns:
        dict: output file pattern -> {label value (int): structure name}.
    """
    if not labels:
        raise ValueError("No labels found for the model outputs.")
    isFileMap = [isinstance(val, dict) for val in labels.values()]
    if not any(isFileMap):
        return {SEG_PATTERN: {int(k): v for k, v in labels.items()}}
    if not all(isFileMap):
        raise ValueError("labels must map either label values to names or output files to label maps, not both.")
    for pattern in labels:
        if not (pattern.endswith('.*') and pattern[:-2].count('*') == 1):
            raise ValueError(f"Output file pattern {pattern!r} must contain one '*' (the case name) "
                             f"and end in .* (.nii.gz or .dcm).")
    return {pattern: {int(k): v for k, v in fileLabels.items()} for pattern, fileLabels in labels.items()}


def readLabels(outputPath, model=None, installDir=None):
    """Output file pattern -> {label value: structure name} for a model's outputs (see normalizeLabels).
    Read from <outputPath>/labels.json, written by the model; else from the model's models.yaml entry.

    Args:
        outputPath (str): model output directory.
        model (int or str, optional): model number or name, for the models.yaml fallback.
        installDir (str, optional): model install directory, for the models.yaml fallback.

    Returns:
        dict: output file pattern -> {label value (int): structure name}.
    """
    labelsFile = os.path.join(outputPath, 'labels.json')
    if os.path.isfile(labelsFile):
        with open(labelsFile, 'r') as f:
            labels = json.load(f)['labels']
    elif model is not None and installDir:
        modelName, repo, _ = getModel(model)
        labels = validateModelName(modelName, getRepoPath(repo, installDir)).get('labels') or {}
    else:
        raise FileNotFoundError(f"No labels.json in {outputPath}; pass model and installDir.")
    return normalizeLabels(labels)


def findOutputs(outputPath, patterns, caseName=None):
    """Case name -> {pattern: file} for the model outputs in outputPath matching the file patterns.

    '*' in a pattern is the case name; a trailing '.*' matches .nii.gz or .dcm only.

    Args:
        outputPath (str): model output directory.
        patterns (iterable of str): output file patterns.
        caseName (str, optional): only return this case.

    Returns:
        dict: case name -> {pattern: file path}.
    """
    fileNames = sorted(os.listdir(outputPath)) if os.path.isdir(outputPath) else []
    cases = {}
    for pattern in patterns:
        stem, ext = (pattern[:-2], OUTPUT_EXTS_REGEX) if pattern.endswith('.*') else (pattern, '')
        prefix, suffix = stem.split('*')
        regex = re.compile(f'{re.escape(prefix)}(?P<case>.+){re.escape(suffix)}{ext}')
        for fileName in fileNames:
            match = regex.fullmatch(fileName)
            if match and (caseName is None or match['case'] == caseName):
                cases.setdefault(match['case'], {})[pattern] = os.path.join(outputPath, fileName)
    return cases


def importSeg(model, installDir, outputPath, scanNum, planC, caseName=None):
    """
    Import segmentations from the model's output directory into planC.
    Note: Loads <case>_<MODEL>_RTSTRUCT.dcm for DICOM inputs and
          <case>_seg.nii.gz (with label names from JSON) for NIfTI inputs.
          A model with several outputs per case lists them in labels by file pattern
          (see normalizeLabels); all are loaded, .dcm files as RTSTRUCTs and .nii.gz files with
          the pattern's labels.
    Args:
        model (int or str): Model number or name (see install_utils.listModels).
        installDir (str): Path to model install directory.
        outputPath (str): Path to directory containing model outputs.
        scanNum (int): Scan number in planC to associate structures with.
        planC (plan_container.planC): pyCERR plan container.
        caseName (str, optional): Case to import when outputPath holds several.

    Returns:
        planC: Updated plan container with auto-segmented structures.
    """
    modelName, _, _ = getModel(model)
    labels = readLabels(outputPath, modelName, installDir)
    # A model with one label map per case writes <case>_<MODEL>_RTSTRUCT.dcm for DICOM inputs
    rtstructPattern = f'*_{modelName}_RTSTRUCT.dcm'
    cases = findOutputs(outputPath, [rtstructPattern], caseName) if list(labels) == [SEG_PATTERN] else {}
    patterns = [rtstructPattern] if cases else list(labels)
    cases = cases or findOutputs(outputPath, patterns, caseName)
    if not cases:
        raise FileNotFoundError(f"No {modelName} output ({', '.join(labels)}) found in {outputPath}.")
    if len(cases) > 1:
        raise RuntimeError(f"Outputs of several cases found in {outputPath}: {', '.join(sorted(cases))}. ")
    case, files = next(iter(cases.items()))

    missing = [pattern for pattern in patterns if pattern not in files]
    if missing:
        raise FileNotFoundError(f"{modelName} outputs missing for case {case}: {', '.join(missing)}.")
    for pattern in patterns:
        if files[pattern].endswith('.dcm'):
            planC = pc.loadDcmDir(files[pattern], initplanC=planC)
        else:
            loadDict = {name: label for label, name in labels[pattern].items()}
            planC = pc.loadNiiStructure(files[pattern], scanNum, planC, labels_dict=loadDict)

    return planC


def listInputs(model, installDir):
    """Print the required and optional userInputs for a given model.

    Includes REQUIRED_INPUTS defined in the pre-processing script, if available, and the model's
    own options.

    Args:
        model (int or str): Model number or name (see install_utils.listModels).
        installDir (str): Path to model install directory.
    """
    modelName, repo, entrypoint = getModel(model)
    repoPath = getRepoPath(repo, installDir)
    entry = validateModelName(modelName, repoPath)

    print(f"\nModel: {modelName}")
    if entry.get('description'):
        print(entry['description'])
    print("=" * 50)
    print(f"{'Input':<25} {'Required':<10}")
    print("-" * 50)

    prepPath = getProcScript('prep', modelName)
    if prepPath:
        preproc = loadModule("preproc", prepPath)
        if not hasattr(preproc, 'REQUIRED_INPUTS'):
            raise AttributeError(f"Pre-processing script {prepPath.name} does not define REQUIRED_INPUTS.")
        inputs = {name: meta.get('required', True) for name, meta in preproc.REQUIRED_INPUTS.items()
                  if name != 'model_dir'}  # supplied by run_utils
    else:
        inputs = {'input_path': True, 'output_path': True, 'session_path': False}
    inputs.setdefault('gpu', False)
    for name, required in inputs.items():
        print(f"  {name:<23} {'Yes' if required else 'No':<10}")
    print("=" * 50)

    # Model options
    pythonExe = getEnvPython(repoPath)
    entrypoint = repoPath / entrypoint
    if pythonExe.exists() and entrypoint.exists():
        try:
            result = subprocess.run([pythonExe.as_posix(), entrypoint.as_posix(), '--help'],
                                    capture_output=True, text=True, cwd=repoPath.as_posix())
        except OSError as e:
            print(f"Error listing model options: {e}")
            return
        if result.returncode == 0:
            print("Other keys are passed to the model as --<key> <value>. Model options:")
            print(result.stdout)
