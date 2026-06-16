"""cerr.imrtp
~~~~~~~~~~~~~

IMRT planning (IMRTP) module for pyCERR - Python port of Matlab CERR's
``IMRTP`` directory (``IMRTPGui.m``, ``initIMRTProblem.m`` et al.).

Quick start::

    from cerr import plan_container as pc
    from cerr.imrtp import IMRTPGui

    planC = pc.loadDcmDir(r'C:/data/myPlan')
    IMRTPGui(planC)

After pressing *Go* with a *Recompute* action selected, the resulting dose
is appended to ``planC.dose`` and the IM problem is stored on ``planC.im``.

Dose engines: the QIB photon pencil-beam algorithm is ported in
:mod:`cerr.imrtp.dosecalc` (with DPM / VMC++ Monte-Carlo placeholders) and
registered automatically.  Register additional engines with
:func:`cerr.imrtp.imrtp.registerEngine`; a self-contained ``'PB-Demo'``
engine is included for end-to-end testing.
"""

from .imrtp_problem import (IMRTProblem, IMBeam, IMGoal, IMParams, VMCParams,
                            Isocenter, Beamlet, initIMRTProblem,
                            createDefaultBeam, conditionBeam,
                            addEquispacedBeams, addGoal, getIMList,
                            saveIMToPlan)
from .imrtp import runIMRTP, doseToPlanC, registerEngine, availableEngines

# Register the bundled dose engines (QIB + Monte-Carlo placeholders).
try:
    from . import dosecalc
    from .dosecalc import recalcDose
except Exception as _e:  # pragma: no cover - e.g. scipy missing
    import warnings as _w
    _w.warn('cerr.imrtp.dosecalc could not be loaded (%s); only '
            'user-registered engines will be available.' % _e)


def IMRTPGui(*args, **kwargs):
    """Lazy wrapper so that headless installs can import cerr.imrtp without
    Qt being present."""
    from .imrtp_gui import IMRTPGui as _gui
    return _gui(*args, **kwargs)
