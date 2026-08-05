"""urOMT (unbalanced regularized Optimal Mass Transport) for pyCERR.

A modular pipeline ported from the MATLAB urOMT ``driver_RatBrain.m``
(https://github.com/xinan-nancy-chen/urOMT). The longitudinal DCE-MRI time
points live in ``planC.scan`` and the ROI in ``planC.structure``; urOMT
model/algorithm settings come from a JSON file.

Pipeline stages:

* **Part 1** - :func:`cerr.uromt.data.prepareData` (concentration conversion,
  load & preprocess frames + mask).
* **Part 2** - :func:`cerr.uromt.solver.solveUROMT` (run the urOMT optimization).
* **Part 3** - :func:`cerr.uromt.analyze.runEULA` (Eulerian speed/rate/Peclet/flux maps).
* **Part 4** - :func:`cerr.uromt.analyze.runGLAD` (Lagrangian transport pathlines).
* **Part 5** - :mod:`cerr.uromt.viz` (napari overlays of the fields & pathlines).

Top-level convenience::

    from cerr.uromt import runUROMT
    result = runUROMT(planC, structNum=0, settingsFile=None)

Part 2 on its own is :func:`cerr.uromt.solver.solveUROMT`, which takes a
prepared ``cfg`` rather than a ``planC``.
"""

import warnings

from cerr.uromt.config import buildConfig, loadModelSettings, UROMTConfig
from cerr.uromt.data import prepareData
from cerr.uromt.analyze import runEULA, runGLAD


def runUROMT(planC, structNum=None, scanNumV=None, settingsFile=None,
             analyze=True, saveToPlanC=True, **settingsOverrides):
    """Run urOMT Parts 1-4 on a planC and store the run in ``planC.urOMT``.

    Part 1 (concentration + preprocessing), Part 2 (optimization) and, when
    ``analyze`` is set, Parts 3-4 (Eulerian + Lagrangian post-processing). The
    inputs and outputs are bundled into a :class:`cerr.dataclasses.uromt.UROMT`
    object and appended to ``planC.urOMT`` (created on demand, like
    ``planC.im``), so the results travel with the plan container.

    Args:
        planC: pyCERR plan container holding the longitudinal scans.
        structNum (int): ROI structure index (``None`` -> whole scan).
        scanNumV (list[int]): scan indices, ordered by time point. ``None``
            (the default) orders *all* scans in the planC by acquisition time
            via :func:`cerr.mri_metrics.dce_mri.getScanOrder`, which raises if
            the timing metadata is inconsistent - pass the list explicitly to
            use a subset or to override the inferred order.
        settingsFile (str): urOMT model-settings JSON (``None`` -> bundled).
        analyze (bool): also run :func:`runEULA`/:func:`runGLAD`.
        saveToPlanC (bool): store the run on ``planC.urOMT``.
        **settingsOverrides: individual model settings to override without
            editing the JSON, e.g. ``useGPU='yes'``, ``numThreads=4``,
            ``maxUiter=3``, ``fft_pad=0``.

    Returns:
        int: the index into ``planC.urOMT`` when ``saveToPlanC`` (access the run
        as ``planC.urOMT[idx]``); otherwise the raw result dict.
    """
    from cerr.uromt.solver import solveUROMT    # lazy (heavy numerics)
    from cerr.dataclasses.uromt import buildFromConfig, saveUROMTToPlan
    cfg = buildConfig(scanNumV, structNum, settingsFile, **settingsOverrides)
    cfg = prepareData(cfg, planC)      # infers scanNumV when it is None
    result = solveUROMT(cfg)
    Eul = runEULA(result) if analyze else None
    Lag = runGLAD(result) if analyze else None
    if analyze:
        result["Eul"], result["Lag"] = Eul, Lag
    if saveToPlanC:
        obj = buildFromConfig(cfg, result, Eul, Lag)
        return saveUROMTToPlan(planC, obj)
    return result


def runUROMTPipeline(*args, **kwargs):
    """Deprecated alias for :func:`runUROMT`."""
    warnings.warn("runUROMTPipeline is deprecated; use cerr.uromt.runUROMT.",
                  DeprecationWarning, stacklevel=2)
    return runUROMT(*args, **kwargs)
