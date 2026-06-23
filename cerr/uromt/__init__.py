"""urOMT (unbalanced regularized Optimal Mass Transport) for pyCERR.

A modular pipeline ported from the MATLAB urOMT ``driver_RatBrain.m``
(https://github.com/xinan-nancy-chen/urOMT). The longitudinal DCE-MRI time
points live in ``planC.scan`` and the ROI in ``planC.structure``; urOMT
model/algorithm settings come from a JSON file.

Pipeline stages:

* **Part 1** - :func:`cerr.uromt.data.prepareData` (load & preprocess frames + mask).
* **Part 2** - :func:`cerr.uromt.solver.runUROMT` (run the urOMT optimization).

Top-level convenience::

    from cerr.uromt import runUROMTPipeline
    result = runUROMTPipeline(planC, scanNumV=[0, 1, 2, 3], structNum=0,
                              settingsFile=None)
"""

from cerr.uromt.config import buildConfig, loadModelSettings, UROMTConfig
from cerr.uromt.data import prepareData


def runUROMTPipeline(planC, scanNumV, structNum=None, settingsFile=None):
    """Run urOMT Part 1 (data prep) then Part 2 (optimization).

    Args:
        planC: pyCERR plan container holding the longitudinal scans.
        scanNumV (list[int]): scan indices, ordered by time point.
        structNum (int): ROI structure index (``None`` -> whole scan).
        settingsFile (str): urOMT model-settings JSON (``None`` -> bundled).

    Returns:
        dict: urOMT results (see :func:`cerr.uromt.solver.runUROMT`).
    """
    from cerr.uromt.solver import runUROMT          # lazy (heavy numerics)
    cfg = buildConfig(scanNumV, structNum, settingsFile)
    cfg = prepareData(cfg, planC)
    return runUROMT(cfg)
