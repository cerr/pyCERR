"""Module for installing AI models via the model_installer (https://github.com/cerr/model_installer.git)."""

import os
import subprocess
import shutil
import platform
import importlib.resources


def main(modelNum, installDir, credentials=None):
    """Install from a library of pretrained AI models (weights and uv environment).

    Args:
        modelNum (int): model number to install (see model_installer for available models)
        installDir (str): path to directory where model weights and uv environment will be installed
        credentials (str, optional): GitHub credentials in format "user:token" for private repos.
                                     Required for model 12 (CT_Lung_OAR_SMITplus).

    """
    checkSystemDependencies()
    modelName = validateModelNum(modelNum)
    validateCredentials(modelNum, modelName, credentials)

    scriptPath = getScriptPath()
    cmd = buildCommand(scriptPath, modelNum, installDir, credentials)
    runInstaller(cmd)


def checkSystemDependencies():
    """Checks for required system dependencies.
    On Linux, macOS requires git and curl.
    On Windows, requires Git Bash with git and curl available.
    """
    osName = platform.system()

    if osName == "Windows" and shutil.which("bash") is None:
        raise EnvironmentError(
            "Git Bash is required but not found. "
            "Please install Git for Windows before proceeding: https://gitforwindows.org"
        )

    missingDeps = [dep for dep in ["git", "curl"] if shutil.which(dep) is None]
    if missingDeps:
        raise EnvironmentError(
            f"The following required dependencies are missing: {', '.join(missingDeps)}. "
            f"Please install them before proceeding."
        )


def listModels():
    """List all available models from the model_installer.
    """
    scriptPath = getScriptPath()

    models = {}
    modelNum = 1
    while True:
        cmd = buildBashCommand(scriptPath, ["-n", str(modelNum)])
        try:
            result = subprocess.run(cmd,capture_output=True,text=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Timed out while listing models at model number {modelNum}.")

        # Extract model name
        lines = [line.strip() for line in result.stdout.strip().split("\n")
                 if line.strip() and line.strip() != "Error"]
        modelName = lines[-1] if lines else ""
        if not modelName or modelName == "Error" or result.returncode != 0:
            break

        models[modelNum] = modelName
        modelNum += 1

    if not models:
        raise RuntimeError("No models found. Check that model_installer is correctly installed.")

    # Display
    for num, name in models.items():
        print(f"  {num}. {name}")

    return models


def validateModelNum(modelNum):
    """Check that modelNum input to installer.sh is valid.

    Args:
        modelNum (int): model number to validate
    """
    scriptPath = getScriptPath()
    cmd = buildBashCommand(scriptPath, ["-n", str(modelNum)])


    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timed out while validating model number {modelNum}.")
    except FileNotFoundError:
        raise RuntimeError(f"Could not execute installer script at {scriptPath}.")

    errMsg = f"Invalid model number: {modelNum}. Run installer.sh -h to see available models."
    if result.returncode != 0:
        raise ValueError(errMsg)

    lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    modelName = lines[-1] if lines else ""

    if modelName in ("Error",  ""):
        raise ValueError(errMsg)

    return modelName


def validateCredentials(modelNum, modelName, credentials):
    """Check that valid credentials are provided for private repos.

    Args:
        modelNum (int): model number to install
        modelName (str): model name returned by validateModelNum
        credentials (str or None): GitHub credentials in format "user:token"
    """

    repoUrl = f"https://github.com/cerr/{modelName}.git"

    env = os.environ.copy()
    if credentials is not None:
        env["GIT_TERMINAL_PROMPT"] = "0"
        repoUrl = f"https://{credentials}@github.com/cerr/{modelName}.git"

    result = subprocess.run(
        ["git", "ls-remote", repoUrl],
        capture_output=True,
        text=True,
        timeout=30,
        env=env
    )

    if result.returncode == 0:
        return

    stderr = result.stderr.lower()
    if credentials:
        stderr = stderr.replace(credentials, "********")
        cleanStderr = result.stderr.replace(credentials, "********")
    else:
        cleanStderr = result.stderr

    isAuthError = "authentication" in stderr or "could not read" in stderr or "403" in stderr

    if isAuthError:
        if credentials is None:
            raise ValueError(
                f"Model {modelNum} ({modelName}) is a private repository. "
                f"GitHub credentials are required to install this model. "
                f"Please provide credentials in the format 'user:token'."
            )
        else:
            raise ValueError(
                f"Model {modelNum} ({modelName}) is a private repository. "
                f"The provided credentials were rejected. "
                f"Please verify your credentials are in the format 'user:token' "
                f"and that the token has repository access."
            )
    else:
        raise RuntimeError(
            f"Failed to access repository for model {modelNum} ({modelName}).\n"
            f"STDERR:\n{cleanStderr}"
        )


def getScriptPath():
    """Locate the installer.sh script from the model_installer package.

    Returns:
        pathlib.Path: path to installer.sh
    """
    try:
        scriptPath = importlib.resources.files("model_installer") / "installer.sh"
    except (ModuleNotFoundError, TypeError):
        raise RuntimeError(
            "The model_installer package is not installed. "
            "Install pyCERR with the model_installer extra: "
            "pip install pyCERR[model_installer]"
        )

    if not scriptPath.is_file():
        raise RuntimeError(
            f"installer.sh not found at expected location: {scriptPath}. "
            "The model_installer package may be improperly installed."
        )

    return scriptPath


def buildBashCommand(scriptPath, args):
    """Builds a command list to invoke installer.sh, accounting for OS differences.

    Args:
        scriptPath (pathlib.Path): path to installer.sh
        args (list): list of argument strings to pass to the script

    Returns:
        list: command arguments for subprocess.run
    """
    bash_exec = shutil.which("bash") or "bash"
    cmd = [bash_exec, str(scriptPath)] + args
    
    return cmd


def buildCommand(scriptPath, modelNum, installDir, credentials):
    """Build the subprocess command list for invoking installer.sh.

    Args:
        scriptPath (pathlib.Path): path to installer.sh
        modelNum (int): model number to install
        installDir (str): path to installation directory
        credentials (str or None): GitHub credentials in format "user:token"

    Returns:
        list: command arguments for subprocess.run
    """
    args = [
        "-m", str(modelNum),
        "-d", str(installDir),
        "-p", "U",
    ]

    if credentials is not None:
        args.extend(["-u", credentials])

    return buildBashCommand(scriptPath, args)


def runInstaller(cmd):
    """Execute model_installer as a subprocess.

    Args:
        cmd (list): command arguments for subprocess.run
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Could not execute bash. Ensure bash is available on your system."
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Model installation failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )