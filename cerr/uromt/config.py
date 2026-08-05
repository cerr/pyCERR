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

# Former setting names, still accepted in settings files and in buildConfig
# overrides so older JSONs and scripts keep working.
_ALIASES = {"gpu": "useGPU", "threads": "numThreads"}

_YES = {"yes", "y", "true", "t", "on", "1"}
_NO = {"no", "n", "false", "f", "off", "0", "none", ""}


def parseYesNo(value, default=False):
    """Interpret a yes/no-style setting as a bool.

    Accepts ``'yes'/'no'`` (and ``y/n``, ``true/false``, ``on/off``), the
    numbers 0/1, booleans, and ``None`` (-> ``default``). Anything else raises,
    so a typo cannot silently read as "off".
    """
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in _YES:
        return True
    if s in _NO:
        return False
    raise ValueError("expected a yes/no value, got %r" % (value,))


def getUseGPU(cfg):
    """Resolve the ``useGPU`` setting off a config object (legacy: ``gpu``)."""
    v = getattr(cfg, "useGPU", None)
    if v is None:
        v = getattr(cfg, "gpu", None)
    return parseYesNo(v, default=False)


def getNumThreads(cfg):
    """Resolve the ``numThreads`` setting off a config object (legacy:
    ``threads``). ``0`` means auto; see :func:`cerr.uromt.kernels.setNumThreads`.
    """
    v = getattr(cfg, "numThreads", None)
    if v is None:
        v = getattr(cfg, "threads", None)
    return 0 if v is None else int(v)


def _applyAliases(settings):
    """Rewrite legacy keys onto their current names (in place)."""
    for old, new in _ALIASES.items():
        if old in settings:
            settings[new] = settings.pop(old)
    return settings


def loadModelSettings(settingsFile=None):
    """Load the urOMT model/algorithm settings from a JSON file.

    Args:
        settingsFile (str): path to a JSON settings file. If ``None`` the
            bundled ``settings/uromt_model_settings.json`` is used.

    Returns:
        dict: model settings (sigma, dt, nt, alpha, beta, ...). Legacy keys
        (``gpu``, ``threads``) are renamed to their current equivalents
        (``useGPU``, ``numThreads``).
    """
    if settingsFile is None:
        settingsFile = _DEFAULT_SETTINGS
    with open(settingsFile) as f:
        s = json.load(f)
    return _applyAliases({k: v for k, v in s.items() if not k.startswith("_")})


class UROMTConfig:
    """Resolved urOMT configuration (model settings + planC-derived data).

    Attributes (model, from JSON): ``sigma, dt, nt, alpha, beta, eta, niter_pcg,
    maxUiter, solver, dTri, reinitR, smooth, smooth_method, smooth_dt,
    do_resize, size_factor`` (``eta`` weights the velocity H1-smoothness penalty
    Gamma4; 0 = off). ROI / post-processing attributes: ``mask_dilate`` (grow
    the ROI mask before it is used to report maps - the reference ``cfg.dilate``)
    and ``peclet_floor`` (Peclet denominator floor fraction).
    Performance attributes: ``fft_pad`` (grow the ROI box to an
    FFT-friendly size - the diffusion DCT is far cheaper on 2/3/5-smooth grid
    dims), ``numThreads`` (CPU threads for the kernels and the diffusion solve;
    default 0 = auto, 1 = single-threaded) and ``useGPU`` (``'yes'``/``'no'``,
    default ``'no'``; run the solver on cupy - see :mod:`cerr.uromt.gpu`). Read
    the last two through :func:`getNumThreads` / :func:`getUseGPU`, which also
    honour the legacy ``threads``/``gpu`` names.
    Concentration-conversion (DCE) attributes:
    ``convertToConc, T10, r1, basePts, TR, FA, conc_clip``.

    Attributes (data, from planC, filled by :func:`cerr.uromt.data.prepareData`):
    ``scanNumV`` (time-point scan indices; pass ``None`` to have
    :func:`cerr.uromt.data.prepareData` infer the acquisition order), ``structNum``,
    ``spacing`` ([row,col,slice] mm, always read from planC), ``trueSize``
    (ROI dims), ``mask`` (3-D ROI),
    ``vol`` (list of preprocessed frame arrays).
    """

    def __init__(self, settings, scanNumV=None, structNum=None):
        self.settings = settings
        for k, v in settings.items():
            setattr(self, k, v)
        self.bc = "open" if int(self.dTri) == 3 else "closed"

        # data fields (populated by prepareData). scanNumV may be None/empty -
        # prepareData then infers the time-point order from the scan metadata.
        self.scanNumV = [] if scanNumV is None else list(scanNumV)
        self.structNum = structNum
        self.spacing = None        # always set from planC (mm) by prepareData
        self.trueSize = None
        self.mask = None
        self.vol = []                 # list of 3-D np.ndarray, one per frame
        self.bbox = None              # (minr,maxr,minc,maxc,mins,maxs)

        # optional source-indicator chi (MATLAB K): a structure index whose
        # mask marks where the relative source r may act. None -> K = 1
        # everywhere. prepareData crops/resizes it like the data and stores the
        # flattened (N,) indicator in self.chi for the numerics.
        self.chiStructNum = settings.get("chiStructNum", None)
        self.chi = None

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


def buildConfig(scanNumV=None, structNum=None, settingsFile=None, **overrides):
    """Create a :class:`UROMTConfig` from a scan list, an ROI structure index,
    and a model-settings JSON file.

    ``scanNumV=None`` leaves the time-point list unset;
    :func:`cerr.uromt.data.prepareData` then orders every scan in the planC by
    acquisition time. ``structNum=None`` means the whole scan (no ROI).

    Keyword ``overrides`` replace individual settings without editing the JSON,
    e.g. ``buildConfig(scans, roi, useGPU='yes', maxUiter=3, fft_pad=0)``.
    Unknown keys raise, so a typo cannot silently do nothing; the legacy names
    ``gpu``/``threads`` are accepted as aliases for ``useGPU``/``numThreads``.
    """
    settings = loadModelSettings(settingsFile)
    overrides = _applyAliases(dict(overrides))
    unknown = set(overrides) - set(settings)
    if unknown:
        raise ValueError("unknown urOMT setting(s): %s"
                         % ", ".join(sorted(unknown)))
    settings.update(overrides)
    return UROMTConfig(settings, scanNumV, structNum)
