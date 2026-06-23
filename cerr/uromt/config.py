"""urOMT configuration.

The MATLAB urOMT stores everything under a single ``cfg`` struct, split by
``getParams.m`` into (a) data/path settings and (b) model/algorithm settings.

In pyCERR the data settings come straight from ``planC`` (the longitudinal
scans and the ROI structure), so only the **model/algorithm** settings live in
a JSON file (``settings/uromt_model_settings.json``). :class:`UROMTConfig`
merges the two into one object that the pipeline (Part 1 / Part 2) consumes -
the analog of ``cfg``.
"""

import json
import os

import numpy as np

_DEFAULT_SETTINGS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "settings", "uromt_model_settings.json")


def loadModelSettings(settingsFile=None):
    """Load the urOMT model/algorithm settings from a JSON file.

    Args:
        settingsFile (str): path to a JSON settings file. If ``None`` the
            bundled ``settings/uromt_model_settings.json`` is used.

    Returns:
        dict: model settings (sigma, dt, nt, alpha, beta, ...).
    """
    if settingsFile is None:
        settingsFile = _DEFAULT_SETTINGS
    with open(settingsFile) as f:
        s = json.load(f)
    return {k: v for k, v in s.items() if not k.startswith("_")}


class UROMTConfig:
    """Resolved urOMT configuration (model settings + planC-derived data).

    Attributes (model, from JSON): ``sigma, dt, nt, alpha, beta, niter_pcg,
    maxUiter, dTri, reinitR, smooth, do_resize, size_factor``.

    Attributes (data, from planC, filled by :func:`cerr.uromt.data.prepareData`):
    ``scanNumV`` (selected time-point scan indices), ``structNum``,
    ``spacing`` ([dx,dy,dz] cm), ``trueSize`` (ROI dims), ``mask`` (3-D ROI),
    ``vol`` (list of preprocessed frame arrays).
    """

    def __init__(self, settings, scanNumV, structNum):
        self.settings = settings
        for k, v in settings.items():
            setattr(self, k, v)
        self.bc = "open" if int(self.dTri) == 3 else "closed"

        # data fields (populated by prepareData)
        self.scanNumV = list(scanNumV)
        self.structNum = structNum
        self.spacing = settings.get("spacing", None)
        self.trueSize = None
        self.mask = None
        self.vol = []                 # list of 3-D np.ndarray, one per frame
        self.bbox = None              # (minr,maxr,minc,maxc,mins,maxs)

    def selectedTimeIndices(self, nScans):
        """1-based first_time:time_jump:last_time -> 0-based positions into the
        supplied scan list (mirrors getData's frame selection)."""
        t = self.settings.get("time", {}) or {}
        first = int(t.get("first_time", 1))
        jump = int(t.get("time_jump", 1))
        last = t.get("last_time", None)
        last = nScans if last is None else int(last)
        first = max(1, first)
        last = min(nScans, last)
        return [i - 1 for i in range(first, last + 1, jump)]

    def __repr__(self):
        return ("UROMTConfig(sigma=%g, dt=%g, nt=%d, alpha=%g, beta=%g, "
                "dTri=%s, frames=%d)"
                % (self.sigma, self.dt, self.nt, self.alpha, self.beta,
                   self.bc, len(self.vol)))


def buildConfig(scanNumV, structNum, settingsFile=None):
    """Create a :class:`UROMTConfig` from a scan list, an ROI structure index,
    and a model-settings JSON file."""
    settings = loadModelSettings(settingsFile)
    return UROMTConfig(settings, scanNumV, structNum)
