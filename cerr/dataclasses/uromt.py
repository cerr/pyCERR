"""urOMT result container stored on the plan container.

Mirrors :class:`cerr.dataclasses.imrt.IM` (stored on ``planC.im``): a urOMT run
- both its inputs and its outputs - is kept as a :class:`UROMT` object in the
dynamically-created ``planC.urOMT`` list, so results travel with the planC (and
persist through pickle) instead of living on the viewer.

Fields:

* ``UROMTSetup``    - inputs: model ``settings``, ``scanNumV`` (time-point scans),
  ``structNum`` (ROI), the resolved ``frameScanNums``, the preprocessed
  concentration frames ``vol``, the ROI ``mask``/``bbox``/``spacing``/``trueSize``.
* ``UROMTResult``   - solver output of :func:`cerr.uromt.solver.runUROMT`
  (per-interval ``u``, ``r``, ``rho``, ``gamma`` plus grid metadata).
* ``UROMTEulerian`` - :func:`cerr.uromt.analyze.runEULA` output (speed/rate/Peclet/flux maps).
* ``UROMTLagrangian`` - :func:`cerr.uromt.analyze.runGLAD` output (pathlines, speed/Peclet lines).
* ``UROMTUID``      - unique id.
"""

from dataclasses import dataclass, field

from cerr.utils import uid


def get_empty_dict():
    return {}


@dataclass
class UROMT:
    UROMTSetup: dict = field(default_factory=get_empty_dict)
    UROMTResult: dict = field(default_factory=get_empty_dict)
    UROMTEulerian: dict = field(default_factory=get_empty_dict)
    UROMTLagrangian: dict = field(default_factory=get_empty_dict)
    UROMTUID: str = ""


def getUROMTList(planC):
    """Return ``planC.urOMT`` (the list of :class:`UROMT` runs), creating it if
    the plan container does not have one yet (PlanC has no native ``urOMT``
    attribute, exactly like ``planC.im``)."""
    if not hasattr(planC, "urOMT") or planC.urOMT is None:
        planC.urOMT = []
    return planC.urOMT


def buildFromConfig(cfg, result, Eul=None, Lag=None):
    """Assemble a :class:`UROMT` object from a solved config + result dicts.

    Stores the inputs to the solver (settings + the preprocessed concentration
    frames and ROI geometry from ``cfg``) alongside the outputs.
    """
    setup = dict(
        settings=getattr(cfg, "settings", {}),
        scanNumV=list(getattr(cfg, "scanNumV", []) or []),
        structNum=getattr(cfg, "structNum", None),
        frameScanNums=list(getattr(cfg, "frameScanNums", []) or []),
        vol=getattr(cfg, "vol", []),
        mask=getattr(cfg, "mask", None),
        bbox=getattr(cfg, "bbox", None),
        spacing=getattr(cfg, "spacing", None),
        trueSize=getattr(cfg, "trueSize", None),
        chi=getattr(cfg, "chi", None),
    )
    return UROMT(UROMTSetup=setup, UROMTResult=result,
                 UROMTEulerian=Eul or {}, UROMTLagrangian=Lag or {},
                 UROMTUID=uid.createUID("UROMT"))


def saveUROMTToPlan(planC, uromtObj, index=None):
    """Store ``uromtObj`` on ``planC.urOMT``; append when ``index`` is None,
    else overwrite. Returns the (0-based) index where it was stored."""
    lst = getUROMTList(planC)
    if index is None or index >= len(lst):
        lst.append(uromtObj)
        return len(lst) - 1
    lst[index] = uromtObj
    return index
