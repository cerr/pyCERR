"""Module for running pretrained AI models available through the model_installer."""
import os
import sys
import glob
import yaml
import importlib
import importlib.util
import subprocess
import shutil

from pathlib import Path
from cerr import plan_container as pc
from cerr.ai_models.install_utils import validateModelNum


def main(modelNum, installDir, userInputs, verbose=False):
    """
        Run pretrained AI model.

    Args:
        modelNum (int): Model number to install (see model_installer for available models)
        installDir (str): Path to model install dir.
        userInputs: Dictionary of arguments provided by the user.
        verbose (bool): [optional, default:False] Print stdout if True.
    """
    installPath = Path(installDir)
    modelName = validateModelNum(modelNum)
    modelBase = installPath / modelName
    modelPath = modelBase.as_posix()

    # Check for pre/post processing scripts
    runSpecFile = (modelBase / 'run_spec.yaml').as_posix()
    try:
        with open(runSpecFile, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Run spec file not found at: {runSpecFile}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing run spec YAML: {e}")
    prepConfig = config.get('preprocessing', {})
    prepScript = prepConfig.get('script')
    prepMode = prepConfig.get('mode', 'in_process')
    postScript = config.get('postprocessing', {}).get('script')
    planC = None
    procScanNum = None
    scanNum = None
    sessionUserInputs = None

    # Apply pre-processing
    if prepScript:
        if prepMode == 'in_process':
            prepPath = Path(__file__).parent / 'prep' / prepScript
            if not prepPath.exists():
                raise FileNotFoundError(f"Pre-processing script not found at: {prepPath}")
            spec = importlib.util.spec_from_file_location("preproc", prepPath)
            preproc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(preproc)
            prepUserInputs = {**userInputs, 'model_dir': modelPath}
            planC, procScanNum, scanNum, sessionUserInputs = preproc.processInputData(prepUserInputs)
        elif prepMode == 'subprocess':
            planC, procScanNum, scanNum, sessionUserInputs = runSubprocessPrep(
                modelBase, prepScript, userInputs, verbose=verbose)
        else:
            raise ValueError(
                f"Unknown preprocessing mode: {prepMode!r}. Expected 'in_process' or 'subprocess'.")

    # Build model run command
    if sessionUserInputs:
        cmd, bashExe = buildCommand(modelBase, sessionUserInputs)
        sessionOutDir = sessionUserInputs.get('session_output') or sessionUserInputs.get('output_path')
    else:
        cmd, bashExe = buildCommand(modelBase, userInputs)
        sessionOutDir = userInputs.get('session_output') or userInputs.get('output_path')

    # Apply the model
    print(f"Running {cmd}")
    result = subprocess.run(cmd, shell=True, executable=bashExe,
                            capture_output=True, text=True, cwd=modelPath)
    if result.returncode != 0:
        raise RuntimeError(f"Model inference failed.\n"f"STDERR:\n{result.stderr}")
    if verbose:
        print(result.stdout)

    # Apply post-processing
    if postScript:
        postPath = Path(__file__).parent / 'post' / postScript
        if not postPath.exists():
            raise FileNotFoundError(f"Post-processing script not found at: {postPath}")
        spec = importlib.util.spec_from_file_location("postproc", postPath)
        postproc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(postproc)
        postUserInputs = {**userInputs, 'model_dir': modelPath}
        __ = postproc.postProcAndImportSeg(planC, procScanNum, scanNum,
                                           postUserInputs, sessionOutDir)

    return result


def runSubprocessPrep(modelBase, prepScript, userInputs, verbose=False):
    """Run pre-processing as a subprocess inside the model's own env where additional libraries are required.

    Syntax:
        <script> --image <raw_nifti> --out_dir <session_input_dir> --gpu <gpu>

    Args:
        modelBase (pathlib.Path): Path to the installed model.
        prepScript (str): Filename of the pre-processing entrypoint, relative to modelBase.
        userInputs (dict): Must contain 'input_path' and 'session_path'. Optional 'gpu'.
        verbose (bool): Print subprocess stdout if True.

    Returns:
        tuple: (planC, procScanNum, scanNum, sessionUserInputs)
    """
    inputPath = userInputs['input_path']
    sessionPath = userInputs['session_path']
    modelPath = modelBase.as_posix()
    gpu = userInputs.get('gpu', 0)

    sessionInputDir = os.path.join(sessionPath, 'input')
    sessionOutputDir = os.path.join(sessionPath, 'output')
    os.makedirs(sessionInputDir, exist_ok=True)
    os.makedirs(sessionOutputDir, exist_ok=True)

    # Load original scan into planC
    if os.path.isdir(inputPath):
        planC = pc.loadDcmDir(inputPath)
    elif inputPath.endswith('.nii') or inputPath.endswith('.nii.gz'):
        planC = pc.loadNiiScan(inputPath)
    else:
        raise ValueError(f"Unsupported input path: {inputPath}. "
                         f"Must be a DICOM directory or NIfTI file.")
    scanNum = len(planC.scan) - 1

    # Export to NIfTI for pre-processing
    ptID = os.path.basename(inputPath.rstrip('/\\'))
    rawNiiFile = os.path.join(sessionInputDir, f"{ptID}_raw.nii.gz")
    planC.scan[scanNum].saveNii(rawNiiFile)

    # Locate the pre-processing entrypoint
    prepPath = modelBase / prepScript
    if not prepPath.exists():
        raise FileNotFoundError(f"Pre-processing script not found at: {prepPath}")
    envPath = modelBase / '.venv'
    if sys.platform == "win32":
        pythonExe = envPath / "Scripts" / "python.exe"
        activateScript = envPath / "Scripts" / "activate"
        bashExe = shutil.which("bash")
    else:
        pythonExe = envPath / "bin" / "python"
        activateScript = envPath / "bin" / "activate"
        bashExe = '/bin/bash'
    if not pythonExe.exists():
        raise FileNotFoundError(f"Python binary not found in the uv env at: {pythonExe}")
    if not activateScript.exists():
        raise FileNotFoundError(f"Activate script not found in the uv env at: {activateScript}")

    cmdStr = (f"{pythonExe.as_posix()} {prepPath.as_posix()} "
              f"--image {rawNiiFile} --out_dir {sessionInputDir} --gpu {gpu}")
    fullCmd = f"source {activateScript.as_posix()} && {cmdStr}"

    print(f"Running {fullCmd}")
    result = subprocess.run(fullCmd, shell=True, executable=bashExe,
                            capture_output=True, text=True, cwd=modelPath)
    if result.returncode != 0:
        raise RuntimeError(f"Pre-processing failed.\nSTDERR:\n{result.stderr}")
    if verbose:
        print(result.stdout)

    # Import processed NIfTI
    candidates = [f for f in glob.glob(os.path.join(sessionInputDir, '*.nii.gz'))
                  if f != rawNiiFile and 'mask' not in os.path.basename(f).lower()]
    if not candidates:
        raise FileNotFoundError(f"No pre-processed NIfTI output found in {sessionInputDir}")
    planC = pc.loadNiiScan(sorted(candidates)[0], initplanC=planC)
    procScanNum = len(planC.scan) - 1

    sessionUserInputs = userInputs.copy()
    sessionUserInputs['session_input'] = sessionInputDir
    sessionUserInputs['session_output'] = sessionOutputDir
    return planC, procScanNum, scanNum, sessionUserInputs


def buildCommand(modelPath, userInputs):
    """
        Reads the model's run specification YAML file and constructs the
        subprocess command to run the model, based on user inputs.

    Args:
        modelPath (pathlib.Path): Path to the installed model.
        userInputs: Dictionary of arguments provided by the user.
    """

    runSpec = (modelPath / 'run_spec.yaml')
    envPath = (modelPath / '.venv')

    if sys.platform == "win32":
        pythonExe = envPath / "Scripts" / "python.exe"
        activateScript = envPath / "Scripts" / "activate"  # bash-compatible
        bashExe = shutil.which("bash")
    else:
        pythonExe = envPath / "bin" / "python"
        activateScript = modelPath / ".venv" / "bin" / "activate"
        bashExe = '/bin/bash'
    if not pythonExe.exists():
        raise FileNotFoundError(f"Python binary not found in the uv env at: {pythonExe}")
    if not activateScript.exists():
        raise FileNotFoundError(f"Activate script not found in the uv env at: {activateScript}")

    # Read the run specs
    try:
        runSpecFile = runSpec.as_posix()
        with open(runSpecFile, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Manifest file not found at: {runSpecFile}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

    # Set path to inference wrapper
    execConfig = config["execution"]
    inferenceWrapper = (modelPath / execConfig["entrypoint"]).as_posix()

    cmd = [pythonExe.as_posix().replace("\\", "/"), inferenceWrapper]

    # Set args
    for arg in execConfig.get("arguments", []):
        argName = arg["name"]

        # Handle Positional Arguments
        if arg["type"] == "positional":
            if arg["required"] and argName not in userInputs:
                raise ValueError(f"Missing required argument: {argName}")
            cmd.append(str(userInputs[argName]))

        # Handle optional flags (e.g., --gpu 1)
        elif arg["type"] == "flag":
            val = userInputs.get(argName, arg.get("default"))
            if val is not None:
                cmd.extend([arg["flag_string"], str(val)])

        # Handle boolean switches (e.g., --verbose)
        elif arg["type"] == "boolean_flag":
            isTrue = userInputs.get(argName, arg.get("default", False))
            if isTrue:
                cmd.append(arg["flag_string"])

    cmdStr = " ".join(cmd)
    fullCmd = (
        f"source {activateScript.as_posix()} && "
        f"PYTHONPATH={modelPath.as_posix()}:$PYTHONPATH "
        f"{cmdStr}"
    )

    return fullCmd, bashExe


def importSeg(modelNum, installDir, outputPath, scanNum, planC):
    """
    Import segmentations from the model's output directory into planC.
    Note: Expected output filename patterns and label-maps are read from
          the model's run_spec.yaml.
    Args:
        modelNum (int): Model number (see model_installer for available models).
        installDir (str): Path to model install directory.
        outputPath (str): Path to directory containing model outputs.
        scanNum (int): Scan number in planC to associate structures with.
        planC (plan_container.planC): pyCERR plan container.

    Returns:
        planC: Updated plan container with auto-segmented structures.
    """

    installPath = Path(installDir)
    modelName = validateModelNum(modelNum)
    modelBase = installPath / modelName

    # Read run_spec.yaml
    runSpecFile = (modelBase / 'run_spec.yaml').as_posix()
    try:
        with open(runSpecFile, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing run spec file: {runSpecFile}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing run spec YAML: {e}")

    # Read expected file names and label mapping
    if 'outputs' not in config:
        raise ValueError(f"No 'outputs' section found in run_spec.yaml for model {modelName}.")
    outputsConfig = config['outputs']

    for pattern, labelDict in outputsConfig.items():
        niiMatches = glob.glob(os.path.join(outputPath, pattern))
        dcmPattern = pattern.replace('.nii.gz', '.dcm')
        dcmMatches = glob.glob(os.path.join(outputPath, dcmPattern))

        if not niiMatches and not dcmMatches:
            raise FileNotFoundError(
                f"No output file found matching pattern '{pattern}' "
                f"(or '{dcmPattern}') in {outputPath}."
            )

        if niiMatches and dcmMatches:
            raise RuntimeError(
                f"Both NIfTI and DICOM matches found for pattern '{pattern}' "
                f"in {outputPath}. Expected only one output type."
            )
        else:
            if niiMatches:
                maskFilePath = niiMatches[0]
                planC = pc.loadNiiStructure(maskFilePath, scanNum, planC,
                                            labels_dict=labelDict)
            else:
                dcmFilePath = dcmMatches[0]
                planC = pc.loadDcmDir(dcmFilePath, initplanC=planC)

    return planC


def listInputs(modelNum, installDir):
    """Print the required and optional userInputs for a given model.

    Includes REQUIRED_INPUTS defined in the processing script, if available along with model inputs
    from run_spec.yaml arguments.

    Args:
        modelNum (int): Model number (see model_installer for available models)
        installDir (str): Path to model install directory.
    """

    installPath = Path(installDir)
    modelName = validateModelNum(modelNum)
    modelBase = installPath / modelName

    # Load run_spec.yaml
    runSpecFile = (modelBase / 'run_spec.yaml').as_posix()
    try:
        with open(runSpecFile, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Run spec file not found at: {runSpecFile}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing run spec YAML: {e}")

    prepConfig = config.get('preprocessing', {})
    prepScript = prepConfig.get('script')
    prepMode = prepConfig.get('mode', 'in_process')

    print(f"\nModel: {modelName}")
    print("=" * 50)
    print(f"{'Input':<25} {'Required':<10}")
    print("-" * 50)

    if prepScript and prepMode == 'in_process':
        # Load REQUIRED_INPUTS
        prepPath = Path(__file__).parent / 'prep' / prepScript
        if not prepPath.exists():
            raise FileNotFoundError(
                f"Pre-processing script not found at: {prepPath}"
            )
        spec = importlib.util.spec_from_file_location("preproc", prepPath)
        preproc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preproc)

        if not hasattr(preproc, 'REQUIRED_INPUTS'):
            raise AttributeError(
                f"Pre-processing script {prepScript} does not define "
                f"REQUIRED_INPUTS."
            )

        for name, meta in preproc.REQUIRED_INPUTS.items():
            required = 'Yes' if meta.get('required', True) else 'No'
            print(f"  {name:<23} {required:<10}")

    else:
        # Read from run_spec.yaml
        args = config.get('execution', {}).get('single', {}).get('arguments', [])
        for arg in args:
            name = arg.get('name', '')
            required = 'Yes' if arg.get('required', False) else 'No'
            print(f"  {name:<23} {required:<10}")

    print("=" * 50)