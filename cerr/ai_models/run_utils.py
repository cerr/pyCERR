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

def main(modelNum, installDir, mode, userInputs, verbose=False):
    """
        Run pretrained AI model.

    Args:
        modelNum (int): Model number to install (see model_installer for available models)
        installDir (str): Path to model install dir.
        mode (str) : The execution mode to use ('single' or 'batch').
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
    prepScript = config.get('preprocessing', {}).get('script')
    postScript = config.get('postprocessing', {}).get('script')
    planC = None
    procScanNum = None
    scanNum = None

    # Apply pre-processing
    if prepScript:
        prepPath = Path(__file__).parent / 'prep' / prepScript
        if not prepPath.exists():
            raise FileNotFoundError(f"Pre-processing script not found at: {prepPath}")
        spec = importlib.util.spec_from_file_location("preproc", prepPath)
        preproc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preproc)
        planC, procScanNum, scanNum = preproc.processInputData(userInputs)

    # Build model run command
    sessionPath = userInputs.get('session_path')
    if prepScript and sessionPath:
        sessionUserInputs = dict(userInputs)
        sessionUserInputs['session_input'] = os.path.join(sessionPath, 'input')
        sessionUserInputs['session_output'] = os.path.join(sessionPath, 'output')
        os.makedirs(sessionUserInputs['session_output'], exist_ok=True)
        cmd, bashExe = buildCommand(modelBase, mode, sessionUserInputs)
    else:
        cmd, bashExe = buildCommand(modelBase, mode, userInputs)

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
        outDir = os.path.join(sessionPath, 'output')
        status = postproc.postProcAndImportSeg(planC, procScanNum, scanNum, userInputs, outDir)

    return result


def buildCommand(modelPath, mode, userInputs):
    """
        Reads the model's run specification YAML file and constructs the
        subprocess command based on user inputs.

    Args:
        modelPath (pathlib.Path): Path to the installed model.
        mode (str) : The execution mode to use ('single' or 'batch').
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
    validModes = ['single','batch']
    if mode not in validModes:
        raise ValueError(
        f"Invalid execution mode '{mode}'."
        f"Available modes: {validModes}"
        )
    execConfig = config["execution"][mode]
    inferenceWrapper = (modelPath /execConfig["entrypoint"]).as_posix()

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

    prepScript = config.get('preprocessing', {}).get('script')

    print(f"\nModel: {modelName}")
    print("=" * 50)
    print(f"{'Input':<25} {'Required':<10}")
    print("-" * 50)

    if prepScript:
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