"""cerr.imrtp.dosecalc
~~~~~~~~~~~~~~~~~~~

Dose-calculation engines for pyCERR's IMRTP module.

Implemented:
    QIB     Quadrant Infinite Beam photon pencil-beam algorithm (Ahnesjo
            analytical kernel with quadrant-integral lookup tables),
            ported from Matlab CERR ``IMRTP/getQIBDose.m`` and friends.

Placeholders (interfaces defined, ports pending -- see
https://github.com/cerr/CERR/tree/master/IMRTP/recompDose):
    DPM     Dose Planning Method Monte Carlo
    VMC++   Voxel Monte Carlo

Importing this package registers 'QIB' (and the 'DPM' / 'VMC++'
placeholders) with the :mod:`cerr.imrtp` engine registry, so they appear
in the IMRTP GUI's *algorithm* drop-down and work with
``cerr.imrtp.runIMRTP``.

High-level entry point (port of ``recompDose/call_doseRecal.m``)::

    from cerr.imrtp.dosecalc import recalcDose
    doseNum, dose3D, im = recalcDose(planC, im, structNumsV=[1, 2],
                                     sampleRateV=[2, 2])

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

from .qib_data import QIBDataS, loadPBData
from .qib import (getQIBDose, getPBConsts, applyIMRTCompression,
                  rtogVectors2Gantry, gantry2RTOGVectors)
from .raytrace import (ScanGrid, getTargetSurfacePoints, getPBRays,
                       getPBRayData, setBeamRayData, CTTrace)
from .influence import generateQIBInfluence, getIMDose
from .montecarlo import (MCRecompParams, beam2MCdose,
                         calcDoseByBeamMeterset, dpmEngine, vmcEngine,
                         MC_SOLVER_DPM, MC_SOLVER_VMC, MC_SOLVER_QIB)
from .recalc import recalcDose, structDoseMetric
from .rtplan import imFromRTPlan, beamsFromRTPlan, dcmIsocenterToCerr

__all__ = [
    'QIBDataS', 'loadPBData',
    'getQIBDose', 'getPBConsts', 'applyIMRTCompression',
    'rtogVectors2Gantry', 'gantry2RTOGVectors',
    'ScanGrid', 'getTargetSurfacePoints', 'getPBRays', 'getPBRayData',
    'setBeamRayData', 'CTTrace',
    'generateQIBInfluence', 'getIMDose',
    'MCRecompParams', 'beam2MCdose', 'calcDoseByBeamMeterset',
    'recalcDose', 'structDoseMetric', 'qibEngine',
    'imFromRTPlan', 'beamsFromRTPlan', 'dcmIsocenterToCerr',
]


# --------------------------------------------------------------------------
# Engine registration with cerr.imrtp
# --------------------------------------------------------------------------

def qibEngine(im, planC, statusCallback=None):
    """QIB dose engine for the cerr.imrtp registry.

    Computes the beamlet influence over all goal structures and returns
    the unit-weight (open-field) dose on the scan grid.
    """
    from cerr.dataclasses import structure as structr
    generateQIBInfluence(im, planC, statusCallback)
    structNumsV = sorted({structr.getStructNumFromUID(g.strUID, planC)
                          for g in im.goals})
    return getIMDose(im, None, structNumsV, planC)


def _register():
    from .. import imrtp as _imrun
    _imrun.registerEngine('QIB', qibEngine)
    _imrun.registerEngine('DPM', dpmEngine)
    _imrun.registerEngine('VMC++', vmcEngine)


_register()
