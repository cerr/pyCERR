"""pyCERR sample datasets, with on-demand download for large data.

Most sample datasets are shipped inside the installed package. Large sample
data (e.g. the head & neck CT) is excluded from the PyPI distribution to keep
it small, and downloaded on first use from the pyCERR GitHub repository, then
cached locally in a per-user data directory.

Example::

    from cerr import datasets
    dcmDir = datasets.fetch_sample_data('head_and_neck')   # downloads if needed

The download cache location is, in order of precedence:

* ``$CERR_DATA_DIR``                              (if set)
* ``%LOCALAPPDATA%\\cerr\\datasets``               (Windows)
* ``$XDG_CACHE_HOME/cerr/datasets`` or
  ``~/.cache/cerr/datasets``                       (Linux / macOS)
"""

import json
import os
import urllib.request

_DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))

# Source repository for on-demand datasets (the data lives in the repo; it is
# only excluded from the built wheel/sdist - see MANIFEST.in / pyproject.toml).
_REPO = "cerr/pyCERR"
_REF = "main"
_API = "https://api.github.com/repos/{repo}/contents/{path}?ref={ref}"

# Public dataset name -> path relative to the cerr/datasets directory.
_REMOTE_DATASETS = {
    "head_and_neck": os.path.join("sample_ct", "head_and_neck"),
}


def list_remote_datasets():
    """Return the names of datasets available via :func:`fetch_sample_data`."""
    return sorted(_REMOTE_DATASETS)


def get_cache_dir():
    """Return the per-user directory where downloaded datasets are cached."""
    base = os.environ.get("CERR_DATA_DIR")
    if not base:
        if os.name == "nt":
            base = os.path.join(
                os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "cerr")
        else:
            base = os.path.join(
                os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                "cerr")
    return os.path.join(base, "datasets")


def _has_data(path):
    return os.path.isdir(path) and bool(os.listdir(path))


def _download_dir(repoPath, localPath):
    """Recursively download a repo directory's files into ``localPath``."""
    apiUrl = _API.format(repo=_REPO, path=repoPath.replace(os.sep, "/"), ref=_REF)
    req = urllib.request.Request(
        apiUrl, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:
        entries = json.load(resp)
    os.makedirs(localPath, exist_ok=True)
    for ent in entries:
        target = os.path.join(localPath, ent["name"])
        if ent["type"] == "dir":
            _download_dir(ent["path"], target)
        elif ent["type"] == "file":
            urllib.request.urlretrieve(ent["download_url"], target)


def fetch_sample_data(name, force=False):
    """Return the local path to a sample dataset, downloading it if needed.

    The data is returned from the installed package if it shipped there (e.g. a
    source checkout). Otherwise it is downloaded from the pyCERR GitHub
    repository and cached under the per-user cache directory (see
    :func:`get_cache_dir`). Requires network access on the first call only.

    Args:
        name (str): dataset name, e.g. ``'head_and_neck'`` (see
            :func:`list_remote_datasets`).
        force (bool): re-download even if a local copy already exists.

    Returns:
        str: absolute path to the dataset directory.
    """
    if name not in _REMOTE_DATASETS:
        raise KeyError(
            "Unknown sample dataset %r. Available: %s"
            % (name, ", ".join(list_remote_datasets())))
    relPath = _REMOTE_DATASETS[name]

    # 1) Already shipped inside the package (source checkout / older wheel).
    bundledPath = os.path.join(_DATASETS_DIR, relPath)
    if not force and _has_data(bundledPath):
        return bundledPath

    # 2) Per-user cache; download there on first use.
    cachePath = os.path.join(get_cache_dir(), relPath)
    if force or not _has_data(cachePath):
        repoPath = "cerr/datasets/" + relPath.replace(os.sep, "/")
        print("Downloading sample dataset %r from %s (%s) to %s..."
              % (name, _REPO, _REF, cachePath))
        _download_dir(repoPath, cachePath)
    return cachePath
