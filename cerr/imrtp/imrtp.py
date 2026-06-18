"""imrtp
~~~~~~~~

Driver for IMRT beamlet/dose computation in pyCERR; the role played by
``IMRTP.m`` + ``dose2CERR.m`` in Matlab CERR.

Matlab CERR shipped three dose engines (QIB, VMC++, DPM) that rely on large
external data files / executables and have not been ported to Python.  This
module therefore exposes a small *engine registry*: any callable with the
signature

    doseArray = engine(im, planC, statusCallback)

can be registered under a name and selected through the GUI's
``IM Parameters > algorithm`` field.  A simple, self-contained
``'PB-Demo'`` divergent pencil-beam engine is provided so that the GUI is
testable end-to-end; it is **for geometry/QA demonstration only and is not
dosimetrically validated**.

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np

from cerr.contour import rasterseg as rs
from cerr.dataclasses import structure as structr
from . import imrtp_problem as imp


# --------------------------------------------------------------------------
# Engine registry
# --------------------------------------------------------------------------

_ENGINES: Dict[str, Callable] = {}


def registerEngine(name: str, fn: Callable):
    """Register a dose engine: ``fn(im, planC, statusCallback) -> dose3M``.

    ``dose3M`` must be a numpy array on the associated scan grid
    (rows x cols x slices), in Gy or arbitrary units."""
    _ENGINES[name] = fn


def availableEngines():
    """Names of registered dose engines (analog of VMCPresent/QIBPresent)."""
    return list(_ENGINES.keys())


def runIMRTP(im: imp.IMRTProblem, planC,
             statusCallback: Optional[Callable[[str, float], None]] = None):
    """Compute dose for all beams of ``im`` using the selected engine
    (port of ``IMRTP.m``'s top-level flow).

    Returns:
        np.ndarray: total dose on the scan grid (rows x cols x slices).
    """
    def status(msg, frac=None):
        if statusCallback is not None:
            statusCallback(msg, frac)

    if not im.beams:
        raise ValueError('No beams defined. Add at least one beam.')
    if not im.goals:
        raise ValueError('No structures selected. Add at least one structure.')

    alg = im.params.algorithm
    if alg not in _ENGINES:
        raise NotImplementedError(
            "Dose engine '%s' is not available in this pyCERR installation. "
            "The Matlab CERR engines (QIB, VMC++, DPM) have not been ported; "
            "register a Python engine with cerr.imrtp.registerEngine() or "
            "use the built-in 'PB-Demo' engine." % alg)

    # Resolve auto fields on every beam before calculation.
    for b in im.beams:
        imp.conditionBeam(b, im, planC,
                          getattr(b, 'autoFieldsOverride', None))

    status('Running %s dose calculation...' % alg, 0.0)
    dose3M = _ENGINES[alg](im, planC, status)
    status('Dose calculation complete.', 1.0)
    im.isFresh = True
    return dose3M


def doseToPlanC(dose3M: np.ndarray, im: imp.IMRTProblem, planC,
                fractionGroupID: Optional[str] = None) -> int:
    """Insert a computed dose into planC (port of ``dose2CERR.m``).

    Returns the new dose index."""
    from cerr import plan_container as pc
    scanNum = im.assocScanNum(planC)
    xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
    if fractionGroupID is None:
        fractionGroupID = im.name
    planC = pc.importDoseArray(dose3M, xV, yV, zV, planC, scanNum)
    planC.dose[-1].fractionGroupID = fractionGroupID
    return len(planC.dose) - 1


# --------------------------------------------------------------------------
# Built-in demonstration engine ('PB-Demo')
# --------------------------------------------------------------------------

def _pbDemoEngine(im: imp.IMRTProblem, planC, status):
    """Simple divergent-beam exponential-attenuation engine.

    For each beam, dose at a voxel is
        w(angle off-axis) * exp(-mu * depth) / r^2-like falloff,
    restricted to a cone covering the (PB-margin expanded) targets.
    Intended only to exercise the GUI / data flow.
    """
    MU = 0.045          # effective attenuation per cm (very rough, 18 MV-ish)
    scanNum = im.assocScanNum(planC)
    scanObj = planC.scan[scanNum]
    xV, yV, zV = scanObj.getScanXYZVals()
    nRows, nCols, nSlcs = len(yV), len(xV), len(zV)

    # Target mask defines the field aperture.
    targetMask = np.zeros((nRows, nCols, nSlcs), dtype=bool)
    for g in im.targetGoals():
        strNum = structr.getStructNumFromUID(g.strUID, planC)
        targetMask |= rs.getStrMask(strNum, planC).astype(bool)
    if not targetMask.any():
        raise ValueError('Target structure mask is empty.')

    rT, cT, sT = np.where(targetMask)
    tgtPts = np.stack([xV[cT], yV[rT], zV[sT]], axis=1)   # N x 3, cm
    margin = max([g.PBMargin for g in im.targetGoals()] + [0.0])

    X, Y, Z = np.meshgrid(xV, yV, zV, indexing='xy')      # rows x cols x slcs
    doseTotal = np.zeros((nRows, nCols, nSlcs), dtype=np.float32)

    nB = len(im.beams)
    for k, beam in enumerate(im.beams):
        status('PB-Demo: beam %d/%d (gantry %g deg)'
               % (k + 1, nB, beam.gantryAngle), k / max(nB, 1))
        src = np.asarray(beam.sourcePos(), dtype=float)
        iso = np.asarray([beam.isocenter.x, beam.isocenter.y,
                          beam.isocenter.z], dtype=float)
        axis = iso - src
        sad = np.linalg.norm(axis)
        axis = axis / sad

        # Aperture: max angular radius of target points (plus margin) about axis.
        v = tgtPts - src
        d = v @ axis
        perp = np.linalg.norm(v - np.outer(d, axis), axis=1)
        apertureTan = np.max((perp + margin) / np.maximum(d, 1e-6))

        # Voxel geometry relative to the source.
        Vx, Vy, Vz = X - src[0], Y - src[1], Z - src[2]
        depth = Vx * axis[0] + Vy * axis[1] + Vz * axis[2]   # along-axis dist
        r2 = Vx ** 2 + Vy ** 2 + Vz ** 2
        perpD = np.sqrt(np.maximum(r2 - depth ** 2, 0.0))
        inField = (depth > 0) & (perpD <= apertureTan * depth)

        # Radiological depth approximated by geometric depth past the patient
        # surface along the axis (entrance at the first in-field voxel depth).
        if inField.any():
            dEnt = depth[inField].min()
            beamDose = np.zeros_like(doseTotal)
            beamDose[inField] = (np.exp(-MU * (depth[inField] - dEnt))
                                 * (sad ** 2) / r2[inField]).astype(np.float32)
            doseTotal += beamDose

    if doseTotal.max() > 0:
        doseTotal *= 100.0 / doseTotal.max()   # normalize to 100 at max
    return doseTotal


registerEngine('PB-Demo', _pbDemoEngine)
