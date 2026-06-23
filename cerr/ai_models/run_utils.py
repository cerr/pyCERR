"""Module for running pretrained AI models available through the model_installer."""
import sys
import yaml
import subprocess

from pathlib import Path
from cerr.ai_models.install_utils import validateModelNum

def main(modelNum, installDir, mode, userInputs):
    """
        Run pretrained AI model.

    Args:
        modelNum (int): Model number to install (see model_installer for available models)
        installDir (str): Path to model install dir.
        mode (str) : The execution mode to use ('single' or 'batch').
        userInputs: Dictionary of arguments provided by the user.
    """
    installPath = Path(installDir)
    modelName = validateModelNum(modelNum)
    runSpec = (installPath / modelName / 'run_spec.yaml')
    envPath = (installPath / modelName / '.venv')
    cmd = buildCommand(envPath, runSpec, mode, userInputs)

    print(f"Running {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result


def buildCommand(envPath, runSpec, mode, userInputs):
    """
        Reads the model's run specification YAML file and constructs the
        subprocess command based on user inputs.

    Args:
        envPath (pathlib.Path): Path to the uv environment for model execution.
        runSpec (pathlib.Path): Path to the run_spec.yaml file.
        mode (str) : The execution mode to use ('single' or 'batch').
        userInputs: Dictionary of arguments provided by the user.
    """

    if sys.platform == "win32":
        pythonExe = envPath / "Scripts" / "python.exe"
    else:
        pythonExe = envPath / "bin" / "python"
    if not pythonExe.exists():
        raise FileNotFoundError(f"Python binary not found in the uv env at: {pythonExe}")

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
    inferenceWrapper = execConfig["entrypoint"]
    cmd = [str(pythonExe), inferenceWrapper]

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

    return cmd