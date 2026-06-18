"""montecarlo
~~~~~~~~~~

Placeholders for the Monte-Carlo dose engines of CERR's
``IMRTP/recompDose`` toolbox (DPM and VMC++).

The Matlab originals
(https://github.com/cerr/CERR/tree/master/IMRTP/recompDose) drive compiled
MC executables and a measurement-based photon source model:

    beam2MCdose.m / beam2MCdose_with_QIB.m
        Build the fluence (leaf sequences, MLC transmission `leak`, wedges,
        tongue-and-groove, horn/softening corrections, source model) for
        one beam of an RTPLAN and run the selected solver.
    calcDoseByBeamMeterset.m
        Read per-beam MC dose from disk and combine using the plan
        metersets (DPM path of ``call_doseRecal.m``).

Porting these requires the DPM / VMC++ binaries, their phase-space /
spectrum inputs (e.g. ``6MV10x10MDA.spectrum``) and the accelerator-head
source model, none of which ship with pyCERR.  The functions below define
the intended Python interfaces and raise ``NotImplementedError`` with
pointers to the Matlab references; the parameter set mirrors
``call_doseRecal.m`` so existing workflows translate directly.

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

from dataclasses import dataclass

_RECOMP_URL = 'https://github.com/cerr/CERR/tree/master/IMRTP/recompDose'

#: solver codes used by call_doseRecal.m
MC_SOLVER_DPM = 1
MC_SOLVER_VMC = 2
MC_SOLVER_QIB = 3


@dataclass
class MCRecompParams:
    """Inputs of ``beam2MCdose.m`` (defaults from ``call_doseRecal.m``)."""
    leak: float = 0.032              # MLC leakage/transmission
    spectrum_File: str = '6MV10x10MDA.spectrum'
    nhist: float = 1e5               # histories per cm^2 (1M ~ 2% sigma)
    batch: int = 101                 # unique id per calculation
    OutputError: int = 0
    PBMaxWidth: float = 10.0
    gradsense: float = 25.0
    MCsolver: int = MC_SOLVER_DPM    # 1: DPM, 2: VMC++
    saveIM: int = 0
    sourceModel: int = 0             # 1 to use the source model
    doseToWater: int = 0
    fillWater: int = 0
    useWedge: int = 0
    inputPB: int = 0
    inputIM: int = 1
    Softening: int = 1
    UseFlatFilter: int = 1
    MLC: int = 0
    TongueGroove: int = 0
    interactiveMode: int = 0
    LS_flag: int = 0
    K: int = 1
    scatterThreshold: float = 11.2   # DPM default (QIB uses 0.1)


def beam2MCdose(im, planC, whichBeam: int,
                params: MCRecompParams = None, statusCallback=None):
    """PLACEHOLDER -- Monte-Carlo dose for one beam (DPM or VMC++).

    Intended port of ``beam2MCdose.m`` / ``beam2MCdose_with_QIB.m`` from
    the CERR ``recompDose`` toolbox: build the beam fluence map (including
    MLC leaf sequences, transmission, wedge and source model) and run the
    selected Monte-Carlo solver, returning the per-beam IM dosimetry and
    beamlet weights.
    """
    params = params or MCRecompParams()
    solver = 'DPM' if params.MCsolver == MC_SOLVER_DPM else 'VMC++'
    raise NotImplementedError(
        'The %s Monte-Carlo engine has not been ported to pyCERR. See the '
        'Matlab reference implementation in %s (beam2MCdose.m). Implement '
        'this function (or register a custom engine with '
        'cerr.imrtp.registerEngine) to enable Monte-Carlo recomputation.'
        % (solver, _RECOMP_URL))


def calcDoseByBeamMeterset(planC, nhist: float, batch: int):
    """PLACEHOLDER -- combine per-beam DPM doses using plan metersets.

    Intended port of ``calcDoseByBeamMeterset.m`` (DPM path of
    ``call_doseRecal.m``): reads the per-beam MC dose files written by the
    DPM solver and sums them weighted by the beam metersets.
    """
    raise NotImplementedError(
        'DPM dose summation has not been ported to pyCERR. See '
        'calcDoseByBeamMeterset.m in %s.' % _RECOMP_URL)


def dpmEngine(im, planC, statusCallback=None):
    """PLACEHOLDER engine entry ('DPM') for the cerr.imrtp registry."""
    raise NotImplementedError(
        "The 'DPM' Monte-Carlo engine is a placeholder. Port beam2MCdose.m "
        'from %s or register your own engine via '
        "cerr.imrtp.registerEngine('DPM', fn). The 'QIB' engine is "
        'available now.' % _RECOMP_URL)


def vmcEngine(im, planC, statusCallback=None):
    """PLACEHOLDER engine entry ('VMC++') for the cerr.imrtp registry."""
    raise NotImplementedError(
        "The 'VMC++' Monte-Carlo engine is a placeholder. Port "
        'generateVMCInfluence.m / beam2MCdose.m from %s or register your '
        "own engine via cerr.imrtp.registerEngine('VMC++', fn). The 'QIB' "
        'engine is available now.' % _RECOMP_URL)
