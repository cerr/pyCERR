"""imrtp_problem
~~~~~~~~~~~~~~~~

Python port of CERR's ``IMRTP/initIMRTProblem.m`` data structures for pyCERR.

Defines the IMRT problem container (:class:`IMRTProblem`) together with its
beam (:class:`IMBeam`), goal (:class:`IMGoal`), solution and parameter
sub-structures, mirroring the Matlab ``IM`` struct used by ``IMRTPGui.m``.

Coordinates are in pyCERR's virtual coordinate system (cm), consistent with
``planC.scan[scanNum].getScanXYZVals()``.

Based on:
    https://github.com/cerr/CERR/blob/master/IMRTP/initIMRTProblem.m
    https://github.com/cerr/CERR/blob/master/IMRTP/IMRTPGui.m

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import List, Optional, Union

import numpy as np

from cerr.utils import uid as uid_utils
from cerr.dataclasses import scan as scn
from cerr.dataclasses import structure as structr


# --------------------------------------------------------------------------
# Field metadata used by the GUI (port of the tables at the top of IMRTPGui.m)
# --------------------------------------------------------------------------

#: Beam-parameter fields shown in the "Beam Parameters" panel.  Nested fields
#: (isocenter.x etc.) are expressed as tuples.
BEAM_FIELD_NAMES = [
    ('beamNum',), ('beamModality',), ('beamEnergy',),
    ('isocenter', 'x'), ('isocenter', 'y'), ('isocenter', 'z'),
    ('isodistance',), ('arcAngle',), ('couchAngle',), ('collimatorAngle',),
    ('gantryAngle',), ('beamDescription',),
    ('beamletDelta_x',), ('beamletDelta_y',), ('dateOfCreation',), ('beamType',),
    ('zRel',), ('xRel',), ('yRel',), ('sigma_100',),
]

#: 1 -> user editable, 0 -> auto-computed (checkbox toggles auto calculation)
BEAM_FIELD_EDITABLE = [0, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                       1, 1, 1, 1, 0, 1, 0, 0, 0, 1]

BEAM_FIELD_IS_NUM = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 0, 1, 1, 0, 0, 1, 1, 1, 1]

BEAM_FIELD_CHOICES = {
    ('beamModality',): ['photons'],
    ('beamEnergy',): [6, 15, 18],
    ('beamType',): ['IM'],
}

#: QIB / DPM pencil-beam dose-calculation parameters
PARAM_NAMES = [
    ('algorithm',), ('DoseTerm',), ('ScatterMethod',),
    ('Scatter', 'Threshold'), ('Scatter', 'RandomStep'),
    ('xyDownsampleIndex',), ('numCTSamplePts',), ('cutoffDistance',),
]
PARAM_IS_NUM = [0, 0, 0, 1, 1, 1, 1, 1]
PARAM_CHOICES = {
    ('DoseTerm',): ['primary', 'nogauss+scatter', 'scatter',
                    'GaussPrimary', 'GaussPrimary+scatter'],
    ('ScatterMethod',): ['random', 'threshold', 'exponential'],
}
PARAM_DEFAULTS = [None, 'GaussPrimary+scatter', 'exponential', 0.01, 30, 1, 300, 4]

#: VMC++ Monte-Carlo parameters
MC_PARAM_NAMES = [
    ('NumParticles',), ('NumBatches',), ('scoreDoseToWater',), ('includeError',),
    ('monoEnergy',), ('spectrum',), ('repeatHistory',), ('splitPhotons',),
    ('photonSplitFactor',), ('base',), ('dimension',), ('skip',),
]
MC_PARAM_IS_NUM = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
MC_PARAM_CHOICES = {
    ('scoreDoseToWater',): ['Yes', 'No'],
    ('includeError',): ['Yes', 'No'],
    ('splitPhotons',): ['Yes', 'No'],
}
MC_PARAM_DEFAULTS = [50000, 10, 'Yes', 'No', None, '', 0.251, 'Yes', -40, 2, 60, 1]

#: Goal (structure) fields
GOAL_FIELDS = ['structNum', 'strUID', 'structName', 'isTarget',
               'PBMargin', 'xySampleRate']
GOAL_DEFAULTS = [1, '', '', 'no', 0.0, 2]


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class Isocenter:
    """Beam isocenter.  Any coordinate may be the string ``'COM'`` meaning
    "auto-compute from the center-of-mass of the target structures"."""
    x: Union[float, str] = 'COM'
    y: Union[float, str] = 'COM'
    z: Union[float, str] = 'COM'


@dataclass
class Beamlet:
    """Sparse per-structure beamlet influence record.

    One Beamlet instance corresponds to (structure i, pencil-beam j) and
    holds the non-zero dose contributions of that pencil beam to that
    structure, in the compressed format used by Matlab CERR."""
    structureName: str = ''
    format: str = 'uint8'          # storage format of influence values
    influence: Optional[np.ndarray] = None   # compressed influence values
    beamNum: int = 0
    fullLength: int = 0            # number of voxels in structure mask
    indexV: Optional[np.ndarray] = None      # voxel indices of non-zeros
    maxInfluenceVal: float = 0.0   # scale factor for compressed values
    lowDosePoints: Optional[np.ndarray] = None
    sampleRate: int = 1
    strUID: str = ''


@dataclass
class IMBeam:
    """One IMRT beam; port of ``IM.beams`` in initIMRTProblem.m."""
    beamNum: int = 1
    beamModality: str = 'photons'
    beamEnergy: float = 18
    isocenter: Isocenter = field(default_factory=Isocenter)
    isodistance: float = 100.0          # SAD, cm
    arcAngle: float = 0.0
    couchAngle: float = 0.0
    collimatorAngle: float = 0.0
    gantryAngle: float = 0.0            # degrees, IEC: 0 = beam from anterior
    beamDescription: str = 'IMRT beam'
    beamletDelta_x: float = 1.0         # beamlet width at iso, cm
    beamletDelta_y: float = 1.0
    dateOfCreation: str = field(default_factory=lambda: date.today().isoformat())
    beamType: str = 'IM'
    # Source position relative to isocenter; auto-computed from gantry angle.
    zRel: float = 0.0
    xRel: float = 0.0
    yRel: float = 0.0
    sigma_100: float = 0.4              # Gaussian source sigma at 100 cm (QIB)
    beamUID: str = field(default_factory=lambda: uid_utils.createUID('BEAM'))
    # beamlets[g][p]: list (over goals) of lists (over pencil beams) of Beamlet
    beamlets: list = field(default_factory=list)
    # Geometry of the beamlet grid on the isocenter plane (filled at run time)
    RTOGPBVectorsM: Optional[np.ndarray] = None
    xPBPosV: Optional[np.ndarray] = None
    yPBPosV: Optional[np.ndarray] = None

    def sourcePos(self):
        """Return absolute source position [x, y, z] (cm)."""
        iso = self.isocenter
        if any(isinstance(v, str) for v in (iso.x, iso.y, iso.z)):
            raise ValueError("Isocenter must be resolved (numeric) before "
                             "computing the source position. Call "
                             "conditionBeam() first.")
        return [iso.x + self.xRel, iso.y + self.yRel, iso.z + self.zRel]


@dataclass
class IMGoal:
    """One structure entry ("goal"); port of ``IM.goals``."""
    structNum: int = 1                  # relative structure index (1-based)
    strUID: str = ''
    structName: str = ''
    isTarget: str = 'no'                # 'yes' / 'no'
    PBMargin: float = 0.0               # pencil-beam margin around target, cm
    xySampleRate: int = 2               # in-plane voxel sampling rate

    def is_target(self) -> bool:
        return str(self.isTarget).lower() == 'yes'


@dataclass
class ScatterParams:
    Threshold: float = 0.01
    RandomStep: float = 30


@dataclass
class VMCParams:
    NumParticles: int = 50000
    NumBatches: int = 10
    scoreDoseToWater: str = 'Yes'
    includeError: str = 'No'
    monoEnergy: Optional[float] = None
    spectrum: str = ''
    repeatHistory: float = 0.251
    splitPhotons: str = 'Yes'
    photonSplitFactor: int = -40
    base: int = 2
    dimension: int = 60
    skip: int = 1


@dataclass
class IMParams:
    """Dose-calculation parameters; port of ``IM.params``."""
    algorithm: str = 'QIB'
    DoseTerm: str = 'GaussPrimary+scatter'
    ScatterMethod: str = 'exponential'
    Scatter: ScatterParams = field(default_factory=ScatterParams)
    xyDownsampleIndex: int = 1
    numCTSamplePts: int = 300
    cutoffDistance: float = 4
    VMC: VMCParams = field(default_factory=VMCParams)


@dataclass
class IMSolution:
    """Optimization output placeholder; port of ``IM.solutions``."""
    beamletWeights: Optional[np.ndarray] = None
    doseScale: float = 1.0
    doseArray: Optional[np.ndarray] = None
    solutionDescription: str = ''


@dataclass
class IMRTProblem:
    """The IM problem container; port of the Matlab ``IM`` struct."""
    name: str = 'IM doseSet 1'
    beams: List[IMBeam] = field(default_factory=list)
    goals: List[IMGoal] = field(default_factory=list)
    params: IMParams = field(default_factory=IMParams)
    solutions: List[IMSolution] = field(default_factory=list)
    assocScanUID: str = ''
    isFresh: bool = True                # False once geometry edits stale the beamlets
    IMUID: str = field(default_factory=lambda: uid_utils.createUID('IM'))

    # -- convenience -------------------------------------------------------
    def assocScanNum(self, planC) -> Optional[int]:
        try:
            return scn.getScanNumFromUID(self.assocScanUID, planC)
        except Exception:
            return None

    def targetGoals(self) -> List[IMGoal]:
        return [g for g in self.goals if g.is_target()]

    def clearBeamlets(self):
        """Invalidate all computed beamlets (geometry/goals changed)."""
        for b in self.beams:
            b.beamlets = []
        self.isFresh = False


# --------------------------------------------------------------------------
# Construction / conditioning helpers (ports of initIMRTProblem.m,
# createDefaultBeam and conditionBeam used by IMRTPGui.m)
# --------------------------------------------------------------------------

def initIMRTProblem(planC=None) -> IMRTProblem:
    """Create a fresh, empty IMRT problem (port of ``initIMRTProblem.m``).

    Args:
        planC: optional pyCERR plan container.  When given, the problem is
            associated with the first scan and named ``IM doseSet n+1``.
    """
    im = IMRTProblem()
    if planC is not None and len(planC.scan) > 0:
        im.assocScanUID = planC.scan[0].scanUID
        imList = getIMList(planC)
        im.name = 'IM doseSet ' + str(len(imList) + 1)
    return im


def createDefaultBeam(beamNum: int, planC=None, scanNum: int = 0) -> IMBeam:
    """Create a beam populated with GUI defaults (port of createDefaultBeam)."""
    beam = IMBeam(beamNum=beamNum,
                  beamDescription='Beam ' + str(beamNum))
    if planC is not None and len(planC.scan) > scanNum:
        si = planC.scan[scanNum].scanInfo[0]
        # Default isocenter at scan offset, like the commented fieldDefaults
        # in IMRTPGui.m; superseded by 'COM' when targets exist.
        beam.isocenter = Isocenter('COM', 'COM', 'COM')
        _ = si  # offsets kept for reference; COM auto-calc is the default
    return beam


def resolveIsocenter(beam: IMBeam, im: IMRTProblem, planC) -> Isocenter:
    """Resolve 'COM' isocenter coordinates from the targets' center of mass."""
    needsCOM = any(isinstance(v, str) and v.upper() == 'COM'
                   for v in (beam.isocenter.x, beam.isocenter.y, beam.isocenter.z))
    if not needsCOM:
        return beam.isocenter

    targets = im.targetGoals()
    if not targets:
        raise ValueError("Isocenter is set to 'COM' but no target structures "
                         "are selected. Mark at least one structure as a "
                         "target (isTarg) or enter a numeric isocenter.")
    comList = []
    for g in targets:
        strNum = structr.getStructNumFromUID(g.strUID, planC)
        comList.append(structr.calcIsocenter(strNum, planC))
    com = np.mean(np.asarray(comList, dtype=float), axis=0)

    iso = Isocenter(beam.isocenter.x, beam.isocenter.y, beam.isocenter.z)
    if isinstance(iso.x, str):
        iso.x = float(com[0])
    if isinstance(iso.y, str):
        iso.y = float(com[1])
    if isinstance(iso.z, str):
        iso.z = float(com[2])
    return iso


def conditionBeam(beam: IMBeam, im: IMRTProblem, planC,
                  autoFields=None) -> IMBeam:
    """Fill in auto-computed beam fields (port of ``conditionBeam``).

    Auto fields (when enabled): beamNum is left as set by the caller,
    isocenter is resolved from target COM, dateOfCreation refreshed, and the
    relative source position is derived from the gantry angle:

        xRel = isodistance * sin(gantry)
        yRel = isodistance * cos(gantry)
        zRel = 0
    """
    autoFields = autoFields if autoFields is not None else {}

    if autoFields.get('isocenter', True):
        beam.isocenter = resolveIsocenter(beam, im, planC)

    if autoFields.get('dateOfCreation', True):
        beam.dateOfCreation = date.today().isoformat()

    if autoFields.get('sourceRel', True):
        ga = math.radians(beam.gantryAngle)
        beam.xRel = beam.isodistance * math.sin(ga)
        beam.yRel = beam.isodistance * math.cos(ga)
        beam.zRel = 0.0
    return beam


def addEquispacedBeams(im: IMRTProblem, numBeams: int, startAngle: float,
                       planC=None, templateBeam: Optional[IMBeam] = None):
    """Add ``numBeams`` equispaced beams starting at ``startAngle`` degrees
    (port of the 'NEWEQUISPACED' GUI command)."""
    import copy
    angles = (startAngle + np.arange(numBeams) * 360.0 / numBeams) % 360.0
    for ang in angles:
        if templateBeam is not None:
            beam = copy.deepcopy(templateBeam)
        else:
            beam = createDefaultBeam(len(im.beams) + 1, planC)
        beam.beamNum = len(im.beams) + 1
        beam.gantryAngle = float(ang)
        beam.beamDescription = 'Beam %d (%g deg)' % (beam.beamNum, ang)
        beam.beamUID = uid_utils.createUID('BEAM')
        beam.beamlets = []
        im.beams.append(beam)
    im.isFresh = False
    return im


def addGoal(im: IMRTProblem, relStrNum: int, planC) -> IMGoal:
    """Append a goal for the relStrNum-th structure (0-based, relative to the
    structures associated with ``im.assocScanUID``); port of 'ADDGOAL'."""
    scanNum = im.assocScanNum(planC)
    strNumsInScan = [i for i, s in enumerate(planC.structure)
                     if scn.getScanNumFromUID(s.assocScanUID, planC) == scanNum]
    absStrNum = strNumsInScan[relStrNum]
    strObj = planC.structure[absStrNum]
    goal = IMGoal(structNum=relStrNum + 1,
                  strUID=strObj.strUID,
                  structName=strObj.structureName)
    im.goals.append(goal)
    return goal


# --------------------------------------------------------------------------
# planC interop
# --------------------------------------------------------------------------

def getIMList(planC) -> list:
    """Return the list of IM problems stored on the plan container, creating
    it if necessary (pyCERR's PlanC has no native ``im`` attribute yet)."""
    if not hasattr(planC, 'im') or planC.im is None:
        planC.im = []
    return planC.im


def saveIMToPlan(im: IMRTProblem, planC, index: Optional[int] = None) -> int:
    """Store ``im`` on planC; append when index is None, else overwrite.

    Returns the (0-based) index at which the problem was stored."""
    imList = getIMList(planC)
    if index is None or index >= len(imList):
        imList.append(im)
        return len(imList) - 1
    imList[index] = im
    return index


def asDict(im: IMRTProblem) -> dict:
    """JSON/pickle-friendly representation of the problem."""
    return asdict(im)
