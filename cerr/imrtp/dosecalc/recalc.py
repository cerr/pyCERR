"""recalc
~~~~~~

Dose recomputation driver -- Python port of CERR's
``IMRTP/recompDose/call_doseRecal.m``.

Workflow (as in the Matlab script):

    1. (optional CT downsample -- not yet ported, see note below)
    2. choose the solver: QIB (implemented) or DPM / VMC++ (placeholders,
       see :mod:`cerr.imrtp.dosecalc.montecarlo`)
    3. compute the beamlet influence for every beam over the requested
       structures / sample rates
    4. store the IM dosimetry on ``planC.im`` and assemble the 3-D dose
    5. optionally scale the result to a clinical dose using a DVH metric
       of a target structure (mean dose or D98, as in call_doseRecal.m)
    6. import the dose into ``planC.dose``

Example::

    from cerr.imrtp.dosecalc import recalcDose
    doseNum, dose3D, im = recalcDose(
        planC, im,
        structNumsV=[5, 6, 7], sampleRateV=[4, 2, 2],
        imrtpName='QIB reCalc',
        ptvStructNum=13, clinicDoseNum=0, scaleMetric='meanDose')

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from cerr import dvh as dvhMod

from .. import imrtp_problem as imp
from .influence import generateQIBInfluence, getIMDose
from .montecarlo import (MC_SOLVER_DPM, MC_SOLVER_QIB, MC_SOLVER_VMC,
                         calcDoseByBeamMeterset)

_SOLVER_NAMES = {MC_SOLVER_DPM: 'DPM', MC_SOLVER_VMC: 'VMC++',
                 MC_SOLVER_QIB: 'QIB'}


def _strNumFromUID(uid: str, planC) -> int:
    from cerr.dataclasses import structure as structr
    return structr.getStructNumFromUID(uid, planC)


def structDoseMetric(planC, structNum: int, doseNum: int,
                     metric: str = 'meanDose', x: float = 98.0,
                     binWidth: float = 0.01) -> float:
    """DVH metric of a structure for a given dose (Dx / meanDose helpers
    used by ``call_doseRecal.m`` for clinical-dose scaling)."""
    dosesV, volsV, isErr = dvhMod.getDVH(structNum, doseNum, planC)
    doseBinsV, volsHistV = dvhMod.doseHist(dosesV, volsV, binWidth)
    m = metric.lower()
    if m == 'meandose':
        return float(dvhMod.meanDose(doseBinsV, volsHistV))
    if m in ('dx', 'd98', 'd%d' % int(x)):
        return float(dvhMod.Dx(doseBinsV, volsHistV, x, 1))  # 1: percent
    raise ValueError("Unknown metric '%s' (use 'meanDose' or 'Dx')."
                     % metric)


def recalcDose(planC,
               im: imp.IMRTProblem,
               structNumsV: Optional[Sequence[int]] = None,
               sampleRateV: Optional[Sequence[int]] = None,
               algorithm: str = 'QIB',
               imrtpName: str = 'QIB reCalc',
               beamletWeightsV: Optional[Sequence[float]] = None,
               scatterThreshold: Optional[float] = None,
               ptvStructNum: Optional[int] = None,
               clinicDoseNum: Optional[int] = None,
               scaleMetric: str = 'meanDose',
               dxPercent: float = 98.0,
               saveIM: bool = True,
               statusCallback=None):
    """Recompute dose for an IMRT problem (port of ``call_doseRecal.m``).

    Args:
        planC:            pyCERR plan container.
        im:               IMRTProblem describing beams, goals and params
                          (build one with the IMRTP GUI or
                          ``cerr.imrtp.initIMRTProblem``).
        structNumsV:      structure indices where dose is computed.  If
                          None, the structures of ``im.goals`` are used.
        sampleRateV:      per-structure in-plane sample rates (powers of
                          2; e.g. skin 8, targets/critical structures 2).
                          Same length as ``structNumsV``.
        algorithm:        'QIB' (implemented), 'DPM' or 'VMC++'
                          (placeholders; see dosecalc.montecarlo).
        imrtpName:        name of the IM dosimetry stored on ``planC.im``.
        beamletWeightsV:  one weight per pencil beam (flat, beam-by-beam)
                          or None for open (unit-weight) fields.
        scatterThreshold: optional override of
                          ``im.params.Scatter.Threshold`` (QIB default in
                          call_doseRecal.m: 0.1).
        ptvStructNum:     target structure used to scale to the clinical
                          dose (skipped when ``clinicDoseNum`` is None).
        clinicDoseNum:    index of the clinical dose in ``planC.dose`` to
                          scale against.
        scaleMetric:      'meanDose' (call_doseRecal.m default) or 'Dx'.
        dxPercent:        x for the Dx metric (default 98).
        saveIM:           store the IM dosimetry on ``planC.im``.
        statusCallback:   optional ``fn(msg, frac)``.

    Returns:
        (doseNum, dose3D, im): index of the new dose in ``planC.dose``,
        the dose array, and the updated IMRTProblem.

    Note:
        The Matlab script optionally downsamples the CT by a factor of 2
        before computing (``getplanCDownSample``); CT downsampling is not
        yet available in pyCERR, so compute time is controlled through
        ``sampleRateV`` and the beamlet size instead.
    """
    from .. import imrtp as imrun   # runtime import to avoid cycles

    def status(msg, frac=None):
        if statusCallback:
            statusCallback(msg, frac)

    alg = {1: 'DPM', 2: 'VMC++', 3: 'QIB'}.get(algorithm, algorithm) \
        if isinstance(algorithm, int) else algorithm

    # ----- per-structure sample rates -----------------------------------
    if structNumsV is not None:
        if sampleRateV is None:
            sampleRateV = [2] * len(structNumsV)
        if len(sampleRateV) != len(structNumsV):
            raise ValueError('sampleRateV must match structNumsV in '
                             'length.')
        byUID = {g.strUID: g for g in im.goals}
        for n, r in zip(structNumsV, sampleRateV):
            uid = planC.structure[n].strUID
            if uid in byUID:
                byUID[uid].xySampleRate = int(r)
            else:
                # addGoal expects the index relative to the structures of
                # the associated scan:
                from cerr.dataclasses import scan as scn
                scanNum = im.assocScanNum(planC)
                inScan = [i for i, s in enumerate(planC.structure)
                          if scn.getScanNumFromUID(s.assocScanUID, planC)
                          == scanNum]
                g = imp.addGoal(im, inScan.index(n), planC)
                g.xySampleRate = int(r)
    else:
        structNumsV = sorted({_strNumFromUID(g.strUID, planC)
                              for g in im.goals})

    if scatterThreshold is not None:
        im.params.Scatter.Threshold = float(scatterThreshold)
    im.params.algorithm = alg

    # ----- resolve auto isocenters etc. ---------------------------------
    for b in im.beams:
        imp.conditionBeam(b, im, planC,
                          getattr(b, 'autoFieldsOverride', None))

    # ----- run the solver ------------------------------------------------
    if alg.upper() == 'QIB':
        status('Running QIB dose recomputation...', 0.0)
        generateQIBInfluence(im, planC, statusCallback)
        dose3D = getIMDose(im, beamletWeightsV, structNumsV, planC)
        doseName = imrtpName
    elif alg.upper() == 'DPM':
        # Matlab: beam2MCdose per beam, then calcDoseByBeamMeterset.
        calcDoseByBeamMeterset(planC, nhist=1e5, batch=101)  # placeholder
        return  # pragma: no cover
    elif alg.upper() in ('VMC++', 'VMC'):
        from .montecarlo import vmcEngine
        vmcEngine(im, planC, statusCallback)  # placeholder
        return  # pragma: no cover
    else:
        raise ValueError("Unknown algorithm '%s' (QIB, DPM or VMC++)."
                         % algorithm)

    # ----- store IM dosimetry on planC (as in call_doseRecal.m) ---------
    im.name = imrtpName
    im.isFresh = True
    im.solutions = (np.asarray(beamletWeightsV).tolist()
                    if beamletWeightsV is not None else [])
    if saveIM:
        imp.saveIMToPlan(im, planC)

    # ----- import dose ----------------------------------------------------
    status('Importing recomputed dose...', 0.95)
    doseNum = imrun.doseToPlanC(dose3D, im, planC,
                                fractionGroupID=doseName)

    # ----- scale to the clinical dose -------------------------------------
    if clinicDoseNum is not None and ptvStructNum is not None:
        status('Scaling to clinical dose...', 0.98)
        reCalcMetric = structDoseMetric(planC, ptvStructNum, doseNum,
                                        scaleMetric, dxPercent)
        clinicMetric = structDoseMetric(planC, ptvStructNum, clinicDoseNum,
                                        scaleMetric, dxPercent)
        if reCalcMetric > 0:
            factor = clinicMetric / reCalcMetric
            planC.dose[doseNum].doseArray = \
                planC.dose[doseNum].doseArray * factor
            dose3D = dose3D * factor
            status('Scaled recomputed dose by %.4f (%s of structure %d).'
                   % (factor, scaleMetric, ptvStructNum), 1.0)

    status('Dose recomputation complete.', 1.0)
    return doseNum, dose3D, im
