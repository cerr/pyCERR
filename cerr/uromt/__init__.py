"""urOMT (unbalanced regularized Optimal Mass Transport) for pyCERR.

A modular pipeline ported from the MATLAB urOMT ``driver_RatBrain.m``
(https://github.com/xinan-nancy-chen/urOMT). The longitudinal DCE-MRI time
points live in ``planC.scan`` and the ROI in ``planC.structure``; urOMT
model/algorithm settings come from a JSON file.

Pipeline stages:

* **Part 1** - :func:`cerr.uromt.data.prepareData` (concentration conversion,
  load & preprocess frames + mask).
* **Part 2** - :func:`cerr.uromt.solver.runUROMT` (run the urOMT optimization).
* **Part 3** - :func:`cerr.uromt.analyze.runEULA` (Eulerian speed/rate/Peclet/flux maps).
* **Part 4** - :func:`cerr.uromt.analyze.runGLAD` (Lagrangian transport pathlines).
* **Part 5** - :mod:`cerr.uromt.viz` (napari overlays of the fields & pathlines).

Top-level convenience::

    from cerr.uromt import runUROMTPipeline
    result = runUROMTPipeline(planC, scanNumV=[0, 1, 2, 3], structNum=0,
                              settingsFile=None)
"""

from cerr.uromt.config import buildConfig, loadModelSettings, UROMTConfig
from cerr.uromt.data import prepareData
from cerr.uromt.analyze import runEULA, runGLAD


def runUROMTPipeline(planC, scanNumV, structNum=None, settingsFile=None,
                     analyze=True, saveToPlanC=True):
    """Run urOMT Parts 1-4 on a planC and store the run in ``planC.urOMT``.

    Part 1 (concentration + preprocessing), Part 2 (optimization) and, when
    ``analyze`` is set, Parts 3-4 (Eulerian + Lagrangian post-processing). The
    inputs and outputs are bundled into a :class:`cerr.dataclasses.uromt.UROMT`
    object and appended to ``planC.urOMT`` (created on demand, like
    ``planC.im``), so the results travel with the plan container.

    Args:
        planC: pyCERR plan container holding the longitudinal scans.
        scanNumV (list[int]): scan indices, ordered by time point.
        structNum (int): ROI structure index (``None`` -> whole scan).
        settingsFile (str): urOMT model-settings JSON (``None`` -> bundled).
        analyze (bool): also run :func:`runEULA`/:func:`runGLAD`.
        saveToPlanC (bool): store the run on ``planC.urOMT``.

    Returns:
        int: the index into ``planC.urOMT`` when ``saveToPlanC`` (access the run
        as ``planC.urOMT[idx]``); otherwise the raw result dict.
    """
    from cerr.uromt.solver import runUROMT          # lazy (heavy numerics)
    from cerr.dataclasses.uromt import buildFromConfig, saveUROMTToPlan
    cfg = buildConfig(scanNumV, structNum, settingsFile)
    cfg = prepareData(cfg, planC)
    result = runUROMT(cfg)
    Eul = runEULA(result) if analyze else None
    Lag = runGLAD(result) if analyze else None
    if analyze:
        result["Eul"], result["Lag"] = Eul, Lag
    if saveToPlanC:
        obj = buildFromConfig(cfg, result, Eul, Lag)
        return saveUROMTToPlan(planC, obj)
    return result
