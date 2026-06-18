"""rtplan
~~~~~~

Build an IMRT problem from the beams of a DICOM RTPLAN stored on
``planC.beams`` -- the role of the beam-geometry section of CERR's
``recompDose/beam2MCdose.m`` (gantry / couch / collimator angles,
isocenter, SAD and nominal energy per beam).

Typical use (the ``call_doseRecal.m`` workflow on a DICOM dataset)::

    from cerr import plan_container as pc
    from cerr.imrtp.dosecalc import imFromRTPlan, recalcDose

    planC = pc.loadDcmDir(r'C:/data/myPlan')      # CT + RTSTRUCT + RTPLAN
    im = imFromRTPlan(planC, targetStructNum=ptvNum)
    doseNum, dose3D, im = recalcDose(planC, im, structNumsV=[...],
                                     sampleRateV=[...])

What is taken from the RTPLAN (as in beam2MCdose.m):
    * gantry angle, couch (PatientSupportAngle) and collimator angles of
      the first control point of each TREATMENT beam,
    * isocenter position (converted from DICOM mm to pyCERR virtual cm
      via ``scan.cerrToDcmTransM``),
    * source-axis distance (mm -> cm),
    * nominal beam energy (mapped to the nearest QIB kernel energy,
      6 or 18 MV, with a warning when not exact).

What is NOT taken: MLC leaf sequences / fluence maps, wedges and per-beam
metersets.  The QIB recomputation covers the target (+ ``PBMargin``) with
open beamlets; use ``beamletWeightsV`` for IMRT weights and the
clinical-dose scaling of :func:`~cerr.imrtp.dosecalc.recalcDose` to set
absolute dose.

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence

import numpy as np

from .. import imrtp_problem as imp

#: energies for which QIB kernel tables exist
QIB_ENERGIES = (6.0, 18.0)


# --------------------------------------------------------------------------
# Coordinate helpers
# --------------------------------------------------------------------------

def dcmIsocenterToCerr(isoDcm_mm, planC, scanNum: int):
    """DICOM patient coordinates (mm) -> pyCERR virtual coordinates (cm).

    Uses the scan's ``cerrToDcmTransM`` (pyCERR xyz cm -> DICOM mm); falls
    back to the simple HFS relation used by ``beam2MCdose.m``
    (x/10, -y/10, -z/10) if the matrix is unavailable.
    """
    iso = np.asarray(isoDcm_mm, dtype=np.float64).ravel()
    transM = getattr(planC.scan[scanNum], 'cerrToDcmTransM', None)
    if transM is not None and np.size(transM) == 16:
        inv = np.linalg.inv(np.asarray(transM, dtype=np.float64))
        out = inv @ np.append(iso, 1.0)
        return float(out[0]), float(out[1]), float(out[2])
    warnings.warn('scan.cerrToDcmTransM unavailable; assuming HFS '
                  '(x/10, -y/10, -z/10) for the isocenter conversion.')
    return iso[0] / 10.0, -iso[1] / 10.0, -iso[2] / 10.0


def _mapEnergy(nominalMV: float) -> float:
    """Map an RTPLAN nominal energy onto an available QIB kernel energy."""
    e = float(nominalMV or 0.0)
    if e in QIB_ENERGIES:
        return e
    mapped = min(QIB_ENERGIES, key=lambda q: abs(q - e))
    warnings.warn('QIB kernel data exists for 6 and 18 MV only; mapping '
                  'the RTPLAN nominal energy %g MV to %g MV.' % (e, mapped))
    return mapped


# --------------------------------------------------------------------------
# RTPLAN -> IMBeam list
# --------------------------------------------------------------------------

def beamsFromRTPlan(planC, beamsNum: int = 0, scanNum: Optional[int] = None,
                    energyMap: bool = True) -> List[imp.IMBeam]:
    """Convert the TREATMENT beams of ``planC.beams[beamsNum]`` to IMBeams.

    Mirrors the geometry import of ``beam2MCdose.m``: per beam the first
    control point supplies gantry / couch / collimator angles, nominal
    energy and isocenter; SAD comes from the beam sequence.  Source
    positions are derived as ``xRel = SAD sin(g)``, ``yRel = SAD cos(g)``,
    ``zRel = 0``, rotated by the couch angle.

    Args:
        planC:     plan container with an RTPLAN loaded.
        beamsNum:  index into ``planC.beams``.
        scanNum:   scan used for the DICOM->CERR isocenter transform
                   (default 0).
        energyMap: map nominal energies to the nearest QIB energy
                   (6 / 18 MV).  Set False to keep the RTPLAN value.

    Returns:
        list of :class:`~cerr.imrtp.imrtp_problem.IMBeam`.
    """
    if not len(planC.beams):
        raise ValueError('planC.beams is empty - no RTPLAN was loaded.')
    plan = planC.beams[beamsNum]
    scanNum = 0 if scanNum is None else scanNum

    beamSeq = np.atleast_1d(plan.BeamSequence)
    if beamSeq.size == 0:
        raise ValueError('RTPLAN %s has no BeamSequence.'
                         % getattr(plan, 'RTPlanLabel', beamsNum))

    imBeams: List[imp.IMBeam] = []
    for bs in beamSeq:
        if str(getattr(bs, 'TreatmentDeliveryType', '') or
               'TREATMENT').upper() not in ('', 'TREATMENT'):
            continue  # skip SETUP / OPEN_PORTFILM etc.
        cps = np.atleast_1d(bs.ControlPointSequence)
        if cps.size == 0:
            continue
        cp0 = cps[0]

        beam = imp.createDefaultBeam(len(imBeams) + 1)
        beam.beamDescription = ('%s' % (getattr(bs, 'BeamName', '') or
                                        getattr(bs, 'BeamDescription', '')))
        beam.beamNum = int(getattr(bs, 'BeamNumber', len(imBeams) + 1))
        beam.gantryAngle = float(getattr(cp0, 'GantryAngle', 0.0))
        beam.couchAngle = float(getattr(cp0, 'PatientSupportAngle', 0.0)
                                or 0.0)
        beam.collimatorAngle = float(getattr(cp0, 'BeamLimitingDeviceAngle',
                                             0.0) or 0.0)
        sad_mm = float(getattr(bs, 'SourceAxisDistance', 0.0) or 1000.0)
        beam.isodistance = sad_mm / 10.0

        nrgy = float(getattr(cp0, 'NominalBeamEnergy', 0.0) or 6.0)
        beam.beamEnergy = _mapEnergy(nrgy) if energyMap else nrgy
        beam.beamModality = getattr(bs, 'RadiationType', 'photons') \
            or 'photons'

        isoDcm = np.atleast_1d(getattr(cp0, 'IsocenterPosition',
                                       np.array([])))
        if isoDcm.size != 3:
            raise ValueError('Beam %s has no IsocenterPosition in its '
                             'first control point.' % beam.beamNum)
        x, y, z = dcmIsocenterToCerr(isoDcm, planC, scanNum)
        beam.isocenter = imp.Isocenter(x=x, y=y, z=z)

        # Source position relative to the isocenter (beam2MCdose.m):
        g = math.radians(beam.gantryAngle)
        xRel = beam.isodistance * math.sin(g)
        yRel = beam.isodistance * math.cos(g)
        zRel = 0.0
        c = math.radians(beam.couchAngle)
        beam.xRel = math.cos(c) * xRel - math.sin(c) * zRel
        beam.yRel = yRel
        beam.zRel = math.sin(c) * xRel + math.cos(c) * zRel
        # The isocenter and source position come from the RTPLAN; stop
        # conditionBeam from re-deriving them (relevant for couch != 0):
        beam.autoFieldsOverride = {'isocenter': False, 'sourceRel': False}
        imBeams.append(beam)

    if not imBeams:
        raise ValueError('No TREATMENT beams found in the RTPLAN.')
    return imBeams


# --------------------------------------------------------------------------
# RTPLAN -> IMRTProblem
# --------------------------------------------------------------------------

def imFromRTPlan(planC, beamsNum: int = 0,
                 targetStructNum: Optional[int] = None,
                 structNumsV: Optional[Sequence[int]] = None,
                 sampleRateV: Optional[Sequence[int]] = None,
                 scanNum: int = 0,
                 PBMargin: float = 0.5,
                 beamletDelta: float = 1.0,
                 name: Optional[str] = None) -> imp.IMRTProblem:
    """Build an :class:`IMRTProblem` from the RTPLAN of ``planC``.

    Args:
        planC:            plan container (CT + RTSTRUCT + RTPLAN loaded).
        beamsNum:         index into ``planC.beams``.
        targetStructNum:  ``planC.structure`` index of the target (PTV)
                          that the beamlets must cover.
        structNumsV:      additional structures where dose is computed
                          (default: target only); ``planC.structure``
                          indices.
        sampleRateV:      per-structure sample rates for ``structNumsV``
                          (default 2 each).
        scanNum:          associated scan.
        PBMargin:         beamlet margin around the target, cm
                          (beam2MCdose.m uses 0.5).
        beamletDelta:     beamlet width at the isocenter, cm.
        name:             IM name (default: the RTPLAN label).

    Returns:
        IMRTProblem ready for
        :func:`~cerr.imrtp.dosecalc.recalcDose` /
        :func:`~cerr.imrtp.runIMRTP`.
    """
    from cerr.dataclasses import scan as scn

    if targetStructNum is None:
        raise ValueError(
            'targetStructNum is required: pass the planC.structure index '
            'of the PTV/target the beams should cover. Available: %s'
            % [(i, s.structureName) for i, s in enumerate(planC.structure)])

    im = imp.initIMRTProblem(planC)
    plan = planC.beams[beamsNum]
    im.name = name or (getattr(plan, 'RTPlanLabel', '') or 'RTPLAN reCalc')

    im.beams = beamsFromRTPlan(planC, beamsNum, scanNum)
    for b in im.beams:
        b.beamletDelta_x = b.beamletDelta_y = float(beamletDelta)

    # Goals: the target first, then any extra dose structures.
    strNumsInScan = [i for i, s in enumerate(planC.structure)
                     if scn.getScanNumFromUID(s.assocScanUID, planC)
                     == scanNum]
    if structNumsV is None:
        structNumsV = [targetStructNum]
    if targetStructNum not in structNumsV:
        structNumsV = [targetStructNum] + list(structNumsV)
    if sampleRateV is None:
        sampleRateV = [2] * len(structNumsV)

    for n, r in zip(structNumsV, sampleRateV):
        g = imp.addGoal(im, strNumsInScan.index(n), planC)
        g.xySampleRate = int(r)
        if n == targetStructNum:
            g.isTarget = 'yes'
            g.PBMargin = float(PBMargin)
    return im
