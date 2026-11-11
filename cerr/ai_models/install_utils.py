"""Module for installing AI models from CERR model repositories (e.g. https://github.com/cerr/<repo_name>).

Supported models are logged in cerr/ai_models/model_list.py.

A model repository serves one or more models. It is cloned to ``<installDir>/<repo>`` and holds:

    models.yaml          model registry (one entry per model)
    get_weights.py       installs a model's weights into the repository's weights/ folder
    requirements/*.txt   pip requirements, installed in name order into a uv environment (.venv)
    uv_config.txt        optional: python_version=<ver> and uv_flags=<extra uv pip flags>
"""

import os
import sys
import shlex
import shutil
import platform
import subprocess
import tempfile
from pathlib import Path

import yaml

from cerr.ai_models.model_list import MODELS, resolveModel

REPO_URL = 'https://github.com/cerr/{repo}.git'
WEIGHT_EXTS = ('.safetensors', '.pt')  # preferred first, as in get_weights.py
ENV_BUILD_MARKER = '.cerr_build_incomplete'  # in .venv while buildEnv is installing requirements


def main(model, installDir, weightsUrl=None, weightsFile=None, credentials=None,
         forceWeights=False, rebuildEnv=False, update=True):
    """Install a pretrained AI model: clone its repository, build the uv environment and install the weights.

    Models in the same repository share inference scripts and an environment; each model only adds its weights.

    Args:
        model (int or str): model number or name (see listModels).
        installDir (str): directory to clone model repositories into.
        weightsUrl (str, optional): download link for the weights, as received from the model maintainers.
        weightsFile (str, optional): already-downloaded weights file or archive (instead of weightsUrl).
            With neither, the weights are downloaded from the source listed in models.yaml, if any.
        credentials (str, optional): GitHub credentials "user:token", for private repositories.
        forceWeights (bool, optional): replace weights that are already installed.
        rebuildEnv (bool, optional): rebuild the uv environment even if it exists.
        update (bool, optional): pull the latest code if the repository is already cloned.

    Returns:
        pathlib.Path: path to the installed weights file.
    """
    if weightsUrl and weightsFile:
        raise ValueError("Pass weightsUrl or weightsFile, not both.")
    modelName, repo, _ = getModel(model)
    checkSystemDependencies()
    repoPath = cloneRepo(repo, installDir, credentials=credentials, update=update)
    validateModelName(modelName, repoPath)
    buildEnv(repoPath, rebuild=rebuildEnv)
    return getWeights(repoPath, modelName, weightsUrl=weightsUrl, weightsFile=weightsFile,
                      force=forceWeights)


def checkSystemDependencies():
    """Check that git and uv (https://docs.astral.sh/uv/) are on the PATH."""
    missingDeps = [dep for dep in ["git", "uv"] if shutil.which(dep) is None]
    if missingDeps:
        raise EnvironmentError(
            f"The following required dependencies are missing: {', '.join(missingDeps)}. "
            f"Please install them before proceeding (uv: https://docs.astral.sh/uv/getting-started/installation/)."
        )

#=======================================
# Clone model repository
#=======================================

def getRepoUrl(repo):
    """GitHub URL of a model repository.

    Args:
        repo (str): repository name (https://github.com/cerr/<repo>).

    Returns:
        str: clone URL of the repository.
    """
    return REPO_URL.format(repo=repo)


def getRepoPath(repo, installDir):
    """Path to the local install of a model repository.

    Args:
        repo (str): repository name.
        installDir (str or pathlib.Path): directory model repositories are cloned into.

    Returns:
        pathlib.Path: absolute path to <installDir>/<repo>.
    """
    return (Path(installDir) / repo).resolve()


def cloneRepo(repo, installDir, credentials=None, update=True):
    """Clone a model repository to <installDir>/<repo>, or pull the latest code if it is already there.

    Args:
        repo (str): repository name (https://github.com/cerr/<repo>).
        installDir (str): directory to clone into.
        credentials (str, optional): GitHub credentials "user:token", for private repositories.
            They are used for the clone only and are not stored in the clone's git config.
        update (bool, optional): run 'git pull --ff-only' in an existing clone.

    Returns:
        pathlib.Path: path to the clone.
    """
    url = getRepoUrl(repo)
    repoPath = getRepoPath(repo, installDir)
    authUrl = url.replace("https://", f"https://{credentials}@", 1) if credentials else url
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"

    if (repoPath / '.git').is_dir():
        if not update:
            return repoPath
        print(f"Updating {repoPath}")
        cmd = ["git", "-C", str(repoPath), "pull", "--ff-only", authUrl]
    elif repoPath.exists() and any(repoPath.iterdir()):
        raise RuntimeError(f"{repoPath} exists and is not a git clone of {url}. Remove it or choose another installDir.")
    else:
        print(f"Cloning {url} to {repoPath}")
        repoPath.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", authUrl, str(repoPath)]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        stderr = result.stderr.replace(credentials, "********") if credentials else result.stderr
        raise RuntimeError(f"git failed for {url}.\nSTDERR:\n{stderr}")
    return repoPath

#=======================================
# Get model info
#=======================================

def loadModelRegistry(repoPath):
    """Read the models section of <repoPath>/models.yaml.

    Args:
        repoPath (str or pathlib.Path): path to the model repository clone.

    Returns:
        dict: model name -> registry entry.
    """
    registryFile = Path(repoPath) / 'models.yaml'
    try:
        with open(registryFile, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)['models']
    except FileNotFoundError:
        raise FileNotFoundError(f"Model registry not found at: {registryFile}.")
    except (yaml.YAMLError, KeyError, TypeError) as e:
        raise ValueError(f"Error reading model registry {registryFile}: {e}")


def validateModelName(modelName, repoPath):
    """Check that modelName is listed in the repository's models.yaml; return its registry entry.

    Args:
        modelName (str): model name.
        repoPath (str or pathlib.Path): path to the model repository clone.

    Returns:
        dict: the model's entry in models.yaml.
    """
    registry = loadModelRegistry(repoPath)
    if modelName not in registry:
        raise ValueError(f"Unknown model {modelName!r} in {repoPath}. Available: {', '.join(registry)}")
    return registry[modelName]


def listModels(installDir=None):
    """List all models indicating where weights have been installed.

    Args:
        installDir (str, optional): model install directory; with it, the installed status is shown.

    Returns:
        dict: {model number: model name}
    """
    registries = {}
    installedWeights = {}
    models = {}
    for modelNum, modelEntry in MODELS.items():
        modelName, repo = modelEntry['name'], modelEntry.get('repo')
        models[modelNum] = modelName
        if not repo or not modelEntry.get('entrypoint'):
            status = 'not supported via pyCERR'
        else:
            status = repo
            repoPath = getRepoPath(repo, installDir) if installDir else None
            if repoPath is not None and (repoPath / 'models.yaml').is_file():
                if repo not in registries:
                    registries[repo] = loadModelRegistry(repoPath)
                entry = registries[repo].get(modelName)
                if entry is None:
                    status += ', not in models.yaml'
                else:
                    if repo not in installedWeights:
                        installedWeights[repo] = listInstalledWeights(repoPath)
                    if installedWeights[repo].get(modelName):
                        status += ', weights installed'
                    else:
                        status += ', weights not installed'
        print(f"  {modelNum:>3}. {modelName:<40} [{status}]")
    return models


def getModel(model):
    """Model name, repository and entrypoint for a supported model.

    Args:
        model (int or str): model number or name (see listModels).

    Returns:
        tuple: (modelName, repo, entrypoint)
    """
    modelNum, entry = resolveModel(model)
    modelName, repo, entrypoint = entry['name'], entry.get('repo'), entry.get('entrypoint')
    if not repo or not entrypoint:
        raise ValueError(f"Model {modelNum} ({modelName}) is not yet supported via pyCERR.")
    return modelName, repo, entrypoint

#=======================================
# Install model env
#=======================================

def getEnvPython(repoPath):
    """Path to the Python executable of the repository's uv environment (.venv).

    Args:
        repoPath (str or pathlib.Path): path to the model repository clone.

    Returns:
        pathlib.Path: path to the environment's Python executable (which may not exist yet).
    """
    envPath = Path(repoPath) / '.venv'
    if sys.platform == "win32":
        return envPath / "Scripts" / "python.exe"
    return envPath / "bin" / "python"


def readUvConfig(repoPath):
    """Read python_version and uv_flags from <repoPath>/uv_config.txt.

    Args:
        repoPath (str or pathlib.Path): path to the model repository clone.

    Returns:
        tuple: (pythonVersion or None, list of extra uv pip flags)
    """
    configFile = Path(repoPath) / 'uv_config.txt'
    pythonVersion, uvFlags = None, []
    if not configFile.is_file():
        return pythonVersion, uvFlags
    for line in configFile.read_text().splitlines():
        key, sep, val = line.strip().partition('=')
        if not sep:
            continue
        if key.strip() == 'python_version' and val.strip():
            pythonVersion = val.strip()
        elif key.strip() == 'uv_flags':
            uvFlags = shlex.split(val)
    return pythonVersion, uvFlags


def _stripPytorchIndex(reqFile, outDir):
    """Copy of a requirements file without PyTorch's own package index lines.

    PyTorch publishes no CUDA builds for macOS, so torch is installed from the default index there.

    Args:
        reqFile (str or pathlib.Path): requirements file.
        outDir (str or pathlib.Path): directory to write the copy to.

    Returns:
        pathlib.Path: the copy, or the original path if the file has no such lines.
    """
    lines = Path(reqFile).read_text().splitlines()
    kept = [ln for ln in lines if not (ln.strip().startswith(('--extra-index-url', '--index-url', '-i ',
                                                              '--find-links', '-f '))
                                       and 'download.pytorch.org' in ln)]
    if len(kept) == len(lines):
        return Path(reqFile)
    outFile = Path(outDir) / Path(reqFile).name
    outFile.write_text("\n".join(kept) + "\n")
    return outFile


def buildEnv(repoPath, rebuild=False):
    """Build the repository's uv environment in <repoPath>/.venv.

    Creates the environment with the Python version from uv_config.txt, then installs every
    requirements/*.txt in name order with the uv_flags from uv_config.txt.

    Args:
        repoPath (str or pathlib.Path): path to the model repository clone.
        rebuild (bool, optional): rebuild even if the environment exists.

    Returns:
        pathlib.Path: path to the environment's Python executable.
    """
    repoPath = Path(repoPath).resolve()  # commands run with cwd=repoPath
    pythonExe = getEnvPython(repoPath)
    buildMarker = repoPath / '.venv' / ENV_BUILD_MARKER
    if pythonExe.exists() and not rebuild:
        if not buildMarker.exists():
            print(f"Using existing environment {repoPath / '.venv'}")
            return pythonExe
        print(f"Rebuilding incomplete environment {repoPath / '.venv'} (a previous build did not finish)")

    reqFiles = sorted((repoPath / 'requirements').glob('*.txt'))
    if not reqFiles:
        raise FileNotFoundError(f"No requirements/*.txt found in {repoPath}")
    pythonVersion, uvFlags = readUvConfig(repoPath)

    venvCmd = ["uv", "venv", str(repoPath / '.venv'), "--clear"]
    if pythonVersion:
        venvCmd += ["--python", pythonVersion]
    _runUv(venvCmd, repoPath)

    # Removed only once every requirements file is installed, so a failed build is not reused
    buildMarker.touch()
    with tempfile.TemporaryDirectory() as tmpDir:
        for reqFile in reqFiles:
            if platform.system() == "Darwin":
                reqFile = _stripPytorchIndex(reqFile, tmpDir)
            _runUv(["uv", "pip", "install", "--python", str(pythonExe), *uvFlags, "-r", str(reqFile)], repoPath)
    buildMarker.unlink()
    return pythonExe


def _runUv(cmd, cwd):
    """Run a uv command, raising if it fails.

    Args:
        cmd (list): uv command and its arguments.
        cwd (str or pathlib.Path): directory to run the command in.

    Raises:
        RuntimeError: if uv exits with a non-zero status; the message carries its output.
    """
    print(f"Running {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Building the model environment failed.\nSTDOUT:\n{result.stdout}\n"
                           f"STDERR:\n{result.stderr}")

#=======================================
# Download model weights
#=======================================

def listInstalledWeights(repoPath):
    """Installed weights of each model in a repository.

    Read from the repository's models.yaml and weights/ folder, without get_weights.py or the model's
    environment. Follows the layout get_weights.py installs: weights/<MODEL>.safetensors (or .pt), or a
    non-empty folder weights/<MODEL>/ for weights made of several files.

    Args:
        repoPath (str or pathlib.Path): path to the model repository clone.

    Returns:
        dict: {model name: path to the installed weights, or None if not installed}
    """
    weightsDir = Path(repoPath) / 'weights'
    installed = {}
    for modelName in loadModelRegistry(repoPath):
        installed[modelName] = None
        for path in [weightsDir / f'{modelName}{ext}' for ext in WEIGHT_EXTS] + [weightsDir / modelName]:
            if path.is_file() or (path.is_dir() and any(path.iterdir())):
                installed[modelName] = path
                break
    return installed


def getWeightsPath(repoPath, modelName):
    """Installed weights of a model (see listInstalledWeights), or None if not installed.

    Args:
        repoPath (str or pathlib.Path): path to the model repository clone.
        modelName (str): model name.

    Returns:
        pathlib.Path or None: weights file or folder.
    """
    return listInstalledWeights(repoPath).get(modelName)


def getWeights(repoPath, modelName, weightsUrl=None, weightsFile=None, force=False):
    """Install a model's weights with the repository's get_weights.py, run in its uv environment.

    Args:
        repoPath (str or pathlib.Path): path to the model repository clone.
        modelName (str): model name from models.yaml.
        weightsUrl (str, optional): download link for the weights.
        weightsFile (str, optional): already-downloaded weights file or archive.
        force (bool, optional): replace weights that are already installed.

    Returns:
        pathlib.Path: path to the installed weights file.
    """
    repoPath = Path(repoPath).resolve()  # commands run with cwd=repoPath
    installed = getWeightsPath(repoPath, modelName)
    if installed and not force:
        print(f"Weights for {modelName} already installed: {installed}")
        return installed

    pythonExe = getEnvPython(repoPath)
    if not pythonExe.exists():
        raise FileNotFoundError(f"Python binary not found in the uv env at: {pythonExe}. Run buildEnv first.")
    cmd = [str(pythonExe), "get_weights.py", "--model", modelName]
    if weightsUrl:
        cmd += ["--url", weightsUrl]
    elif weightsFile:
        cmd += ["--file", str(Path(weightsFile).resolve())]
    if force:
        cmd.append("--force")

    print(f"Installing weights for {modelName}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repoPath)
    if result.returncode != 0:
        raise RuntimeError(f"Installing weights for {modelName} failed.\n"
                           f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    print(result.stdout)
    return getWeightsPath(repoPath, modelName)
