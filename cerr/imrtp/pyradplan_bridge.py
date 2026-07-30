"""pyradplan_bridge
~~~~~~~~~~~~~~~~~

Bridge between pyCERR's ``planC`` container and pyRadPlan
(https://github.com/e0404/pyRadPlan) for beamlet dose-influence
calculation and fluence optimization.

The bridge converts:
    * ``planC.scan[scanNum]``       -> pyRadPlan ``CT`` (SimpleITK, DICOM mm),
    * ``planC.structure[...]``      -> pyRadPlan ``StructureSet`` (VOIs + objectives),
    * ``planC.beams[beamsNum]``     -> pyRadPlan ``Plan`` (``prop_stf`` from the
      first control point of each treatment beam: gantry / couch angles,
      isocenter and nominal energy, as in ``cerr.imrtp.dosecalc.rtplan``),
and imports the resulting dose cube back as a new element of ``planC.dose``.
The sparse beamlet dose-influence matrix (``dij``) is returned to the caller.

Typical use::

    from cerr import plan_container as pc
    from cerr.imrtp import pyradplan_bridge as prp

    planC = pc.loadDcmDir(r'C:/data/myPlan')   # CT + RTSTRUCT (+ RTPLAN)
    ct, cst, pln = prp.planFromPlanC(planC, scanNum=0, beamsNum=0,
                                     objectives={ptvNum: [prp.squaredDeviation(60.0)],
                                                 oarNum: [prp.squaredOverdosing(20.0)]})
    stf, dij = prp.calcDoseInfluence(ct, cst, pln)
    w, doseNum, planC = prp.optimizeAndImportDose(planC, ct, cst, stf, dij, pln,
                                                  scanNum=0)

Limitations (same as the QIB recompute path): MLC leaf sequences, wedges
and per-beam metersets of the RTPLAN are *not* reproduced -- the plan
geometry is re-planned with pyRadPlan beamlets.

Requires the optional dependency ``pyRadPlan`` (``pip install pyRadPlan``).

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence

import numpy as np

try:
    import SimpleITK as sitk
    from pyRadPlan import (calc_dose_influence, fluence_optimization,
                           generate_stf)
    from pyRadPlan.ct import create_ct
    from pyRadPlan.cst import StructureSet, create_voi
    from pyRadPlan.plan import IonPlan, PhotonPlan
    from pyRadPlan.optimization.objectives import (SquaredDeviation,
                                                   SquaredOverdosing,
                                                   SquaredUnderdosing,
                                                   MeanDose)
    _PYRADPLAN_AVAILABLE = True
    _PYRADPLAN_IMPORT_ERROR = None
except ImportError as _e:  # pragma: no cover - exercised only without pyRadPlan
    _PYRADPLAN_AVAILABLE = False
    _PYRADPLAN_IMPORT_ERROR = _e


def _requirePyRadPlan():
    if not _PYRADPLAN_AVAILABLE:
        raise ImportError(
            'pyRadPlan is required for cerr.imrtp.pyradplan_bridge. '
            'Install it with "pip install pyRadPlan". Original error: %s'
            % _PYRADPLAN_IMPORT_ERROR)


# --------------------------------------------------------------------------
# planC -> pyRadPlan converters
# --------------------------------------------------------------------------

#: DICOM orientation label used for pyRadPlan (identity direction cosines).
#: pyRadPlan's Siddon ray tracer assumes an axis-aligned CT; a scan whose
#: ``imageOrientationPatient`` is flipped (e.g. ``[-1,0,0,0,-1,0]``) yields a
#: non-identity SimpleITK direction and every ray misses the patient. We
#: reorient the CT/masks to this standard orientation and resample the dose
#: back onto the original scan grid on import (see :func:`doseArrayFromSitk`).
_PYRADPLAN_ORIENT = 'LPS'


def _reorientForPyRadPlan(img):
    """Reorient a SimpleITK image to :data:`_PYRADPLAN_ORIENT`.

    A no-op (up to a copy) when the image is already axis-aligned to that
    orientation, so scans with standard geometry are unaffected.
    """
    return sitk.DICOMOrient(img, _PYRADPLAN_ORIENT)


def ctFromScan(planC, scanNum: int = 0):
    """Convert ``planC.scan[scanNum]`` to a pyRadPlan ``CT``.

    Uses ``Scan.getSitkImage`` (DICOM patient space, mm, true HU with the
    CT offset applied) and reorients to axis-aligned LPS so pyRadPlan's ray
    tracer behaves for scans with a flipped ``imageOrientationPatient``.
    """
    _requirePyRadPlan()
    img = planC.scan[scanNum].getSitkImage()
    img = sitk.Cast(img, sitk.sitkFloat64)
    img = _reorientForPyRadPlan(img)
    return create_ct(cube_hu=img)


def cstFromStructs(planC, ct, structNums: Optional[Sequence[int]] = None,
                   objectives: Optional[Dict[int, list]] = None,
                   targetStructNums: Optional[Sequence[int]] = None):
    """Convert pyCERR structures to a pyRadPlan ``StructureSet``.

    Args:
        planC: plan container.
        ct: pyRadPlan CT created with :func:`ctFromScan` (same scan!).
        structNums: indices into ``planC.structure`` to convert. Defaults to
            all structures associated with the CT's scan.
        objectives: optional dict ``{structNum: [Objective, ...]}`` used for
            optimization (see :func:`squaredDeviation` etc.).
        targetStructNums: structures to mark as ``TARGET``; all others are
            ``OAR``. Structures with a SquaredDeviation/SquaredUnderdosing
            objective are auto-promoted to targets when this is None.
    """
    _requirePyRadPlan()
    if structNums is None:
        structNums = list(range(len(planC.structure)))
    objectives = objectives or {}

    if targetStructNums is None:
        targetStructNums = [s for s, objs in objectives.items()
                            if any(isinstance(o, (SquaredDeviation,
                                                  SquaredUnderdosing))
                                   for o in objs)]

    vois = []
    for s in structNums:
        struct = planC.structure[s]
        maskImg = struct.getSitkImage(planC)
        maskImg = _reorientForPyRadPlan(sitk.Cast(maskImg, sitk.sitkUInt8))
        voiType = 'TARGET' if s in targetStructNums else 'OAR'
        vois.append(create_voi(name=struct.structureName,
                               voi_type=voiType,
                               ct_image=ct,
                               mask=maskImg,
                               objectives=list(objectives.get(s, []))))
    return StructureSet(vois=vois, ct_image=ct)


def targetCentroidMm(planC, structNums: Sequence[int]) -> np.ndarray:
    """Center-of-mass of the union of ``structNums`` in DICOM LPS mm.

    Used as an isocenter fallback when the RTPLAN stores a degenerate
    ([0, 0, 0]) isocenter.
    """
    _requirePyRadPlan()
    union = None
    for s in structNums:
        m = sitk.Cast(planC.structure[s].getSitkImage(planC), sitk.sitkUInt8)
        union = m if union is None else (union | m)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(union)
    return np.asarray(stats.GetCentroid(1), dtype=np.float64)


def beamGeometryFromBeams(planC, beamsNum: int = 0) -> dict:
    """Extract per-beam geometry from ``planC.beams[beamsNum]``.

    Mirrors ``cerr.imrtp.dosecalc.rtplan.beamsFromRTPlan``: the first
    control point of each TREATMENT beam supplies gantry angle, couch
    (PatientSupportAngle) angle, isocenter (DICOM LPS mm) and nominal
    energy; radiation type comes from the beam sequence.

    Returns:
        dict with keys ``gantry_angles``, ``couch_angles``, ``iso_center``
        (n x 3, mm), ``energies``, ``radiation_mode`` ('photons'/'protons')
        and ``iso_degenerate`` (True when every beam's isocenter is the
        DICOM origin, which some verification/setup plans store as a
        placeholder -- see :func:`planFromPlanC` for the target-centroid
        fallback).
    """
    beams = planC.beams[beamsNum]
    gantry, couch, isos, energies, radTypes = [], [], [], [], []
    for bs in np.atleast_1d(beams.BeamSequence):
        if str(getattr(bs, 'TreatmentDeliveryType', '') or '').upper() \
                not in ('', 'TREATMENT'):
            continue
        cps = np.atleast_1d(bs.ControlPointSequence)
        if cps.size == 0:
            continue
        cp0 = cps[0]
        g = float(getattr(cp0, 'GantryAngle', 0.0))
        c = float(getattr(cp0, 'PatientSupportAngle', 0.0))
        gantry.append(0.0 if np.isnan(g) else g)
        couch.append(0.0 if np.isnan(c) else c)
        iso = np.asarray(getattr(cp0, 'IsocenterPosition', np.empty(0)),
                         dtype=np.float64).ravel()
        if iso.size != 3:
            raise ValueError('Beam %s of planC.beams[%d] has no '
                             'IsocenterPosition in its first control point.'
                             % (getattr(bs, 'BeamName', '?'), beamsNum))
        isos.append(iso)
        e = float(getattr(cp0, 'NominalBeamEnergy', np.nan))
        energies.append(e)
        radTypes.append(str(getattr(bs, 'RadiationType', 'PHOTON')).upper())

    if not gantry:
        raise ValueError('No treatment beams with control points found in '
                         'planC.beams[%d].' % beamsNum)

    radMode = 'protons' if all(r == 'PROTON' for r in radTypes) else 'photons'
    if radMode == 'photons' and any(r not in ('', 'PHOTON') for r in radTypes):
        warnings.warn('Mixed/unsupported radiation types %s; treating the '
                      'plan as photons.' % sorted(set(radTypes)))

    isoM = np.vstack(isos)
    isoDegenerate = bool(np.all(isoM == 0.0))
    if isoDegenerate:
        warnings.warn('All beams in planC.beams[%d] have IsocenterPosition '
                      '[0, 0, 0]; this is often a placeholder in '
                      'verification/setup RTPLANs rather than a real '
                      'treatment isocenter. Pass isoCenter=... (or a target '
                      'structure) to override.' % beamsNum)
    return {'gantry_angles': gantry,
            'couch_angles': couch,
            'iso_center': isoM,
            'energies': energies,
            'radiation_mode': radMode,
            'iso_degenerate': isoDegenerate}


def planFromPlanC(planC, scanNum: int = 0, beamsNum: Optional[int] = 0,
                  objectives: Optional[Dict[int, list]] = None,
                  structNums: Optional[Sequence[int]] = None,
                  targetStructNums: Optional[Sequence[int]] = None,
                  gantryAngles: Optional[Sequence[float]] = None,
                  couchAngles: Optional[Sequence[float]] = None,
                  isoCenter: Optional[np.ndarray] = None,
                  bixelWidth: float = 5.0,
                  machine: str = 'Generic',
                  prescribedDose: Optional[float] = None,
                  numOfFractions: int = 1,
                  doseGridResolution: Optional[Dict[str, float]] = None):
    """Build pyRadPlan ``(ct, cst, pln)`` from planC.

    Beam geometry is read from ``planC.beams[beamsNum]`` unless
    ``gantryAngles`` (and optionally ``couchAngles`` / ``isoCenter``) are
    given explicitly, in which case ``beamsNum`` may be None.

    Args:
        planC: plan container.
        scanNum: index of the CT scan in ``planC.scan``.
        beamsNum: index into ``planC.beams`` (None to use explicit angles).
        objectives: ``{structNum: [Objective, ...]}`` optimization objectives.
        structNums / targetStructNums: see :func:`cstFromStructs`.
        gantryAngles / couchAngles: explicit beam angles in degrees
            (override RTPLAN).
        isoCenter: explicit isocenter(s), DICOM LPS mm; a single 3-vector or
            an (nBeams x 3) array. Default: RTPLAN isocenter, or the target
            center-of-mass when no RTPLAN is used.
        bixelWidth: beamlet size at isocenter, mm.
        machine: pyRadPlan machine name ('Generic' ships with pyRadPlan).
        prescribedDose: prescription in Gy (used by optimization scaling).
        numOfFractions: number of fractions.
        doseGridResolution: e.g. ``{'x': 3.0, 'y': 3.0, 'z': 3.0}`` (mm);
            defaults to pyRadPlan's default dose grid.

    Returns:
        tuple: ``(ct, cst, pln)`` ready for :func:`calcDoseInfluence`.
    """
    _requirePyRadPlan()
    ct = ctFromScan(planC, scanNum)
    cst = cstFromStructs(planC, ct, structNums=structNums,
                         objectives=objectives,
                         targetStructNums=targetStructNums)

    radMode = 'photons'
    isoM = None
    isoDegenerate = False
    if gantryAngles is None:
        geom = beamGeometryFromBeams(planC, beamsNum if beamsNum else 0)
        gantryAngles = geom['gantry_angles']
        couchAngles = geom['couch_angles']
        isoM = geom['iso_center']
        radMode = geom['radiation_mode']
        isoDegenerate = geom['iso_degenerate']
    if couchAngles is None:
        couchAngles = [0.0] * len(gantryAngles)
    if isoCenter is not None:
        isoM = np.atleast_2d(np.asarray(isoCenter, dtype=np.float64))
        if isoM.shape[0] == 1:
            isoM = np.repeat(isoM, len(gantryAngles), axis=0)
    elif isoDegenerate:
        # RTPLAN isocenter is the [0,0,0] placeholder; fall back to the
        # target center-of-mass if a target structure is known.
        tgtNums = list(targetStructNums) if targetStructNums else \
            [s for s, objs in (objectives or {}).items()
             if any(isinstance(o, (SquaredDeviation, SquaredUnderdosing))
                    for o in objs)]
        if tgtNums:
            ctr = targetCentroidMm(planC, tgtNums)
            warnings.warn('Using target-structure centroid %s (mm) as the '
                          'isocenter in place of the [0,0,0] RTPLAN '
                          'placeholder.' % np.round(ctr, 1).tolist())
            isoM = np.repeat(ctr[None, :], len(gantryAngles), axis=0)

    planCls = IonPlan if radMode == 'protons' else PhotonPlan
    pln = planCls(radiation_mode=radMode, machine=machine)
    pln.num_of_fractions = numOfFractions
    if prescribedDose is not None:
        pln.prescribed_dose = prescribedDose
    pln.prop_stf = {'gantry_angles': [float(g) for g in gantryAngles],
                    'couch_angles': [float(c) for c in couchAngles],
                    'bixel_width': float(bixelWidth)}
    if isoM is not None:
        pln.prop_stf['iso_center'] = isoM
    if doseGridResolution is not None:
        # the dose engine expects a Grid; resample the CT grid to the
        # requested resolution
        pln.prop_dose_calc = {'dose_grid':
                              ct.grid.resample(doseGridResolution)}
    return ct, cst, pln


# --------------------------------------------------------------------------
# Dose calculation / optimization
# --------------------------------------------------------------------------

def calcDoseInfluence(ct, cst, pln):
    """Generate steering geometry and the beamlet dose-influence matrix.

    Returns:
        tuple: ``(stf, dij)`` -- pyRadPlan SteeringInformation and Dij.
        ``dij.physical_dose`` holds the sparse (voxels x beamlets)
        dose-influence matrix on the dose grid.
    """
    _requirePyRadPlan()
    stf = generate_stf(ct, cst, pln)
    dij = calc_dose_influence(ct, cst, stf, pln)
    return stf, dij


def doseArrayFromSitk(doseImg, planC, scanNum: int):
    """Convert a dose SimpleITK image back to pyCERR ``(row, col, slice)``.

    The pyRadPlan dose lives on the (reoriented) CT grid; it is first
    resampled onto ``Scan.getSitkImage``'s grid by world coordinates -- a
    no-op when they already coincide, and the step that undoes the LPS
    reorientation applied in :func:`ctFromScan`. It is then permuted to
    pyCERR order, the inverse of ``Scan.getSitkImage``: (z, y, x) DICOM
    slice order -> (row, col, slice) in pyCERR slice order.
    """
    from cerr.dataclasses.scan import flipSliceOrderFlag
    refImg = planC.scan[scanNum].getSitkImage()
    doseImg = sitk.Resample(sitk.Cast(doseImg, sitk.sitkFloat64), refImg,
                            sitk.Transform(), sitk.sitkLinear, 0.0,
                            sitk.sitkFloat64)
    dose3M = sitk.GetArrayFromImage(doseImg)
    if flipSliceOrderFlag(planC.scan[scanNum]):
        dose3M = np.flip(dose3M, axis=0)
    return np.transpose(dose3M, (1, 2, 0))


def importDoseToPlanC(planC, doseImg, scanNum: int = 0,
                      fractionGroupID: str = 'pyRadPlan',
                      units: str = 'GRAYS'):
    """Import a dose image (on the CT grid) into ``planC.dose``.

    Args:
        planC: plan container.
        doseImg: SimpleITK image or numpy array in pyCERR (r, c, s) order.
        scanNum: associated scan index.
        fractionGroupID: label stored on the dose object.
        units: dose units label.

    Returns:
        int: index of the new dose in ``planC.dose``.
    """
    from cerr import plan_container as pc
    if isinstance(doseImg, np.ndarray):
        dose3M = doseImg
    else:
        dose3M = doseArrayFromSitk(doseImg, planC, scanNum)
    scanSiz = planC.scan[scanNum].getScanSize()
    if tuple(dose3M.shape) != tuple(scanSiz):
        raise ValueError('Dose shape %s does not match scan grid %s; '
                         'resample the dose to the CT grid first.'
                         % (dose3M.shape, tuple(scanSiz)))
    xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
    doseInfo = {'fractionGroupID': fractionGroupID, 'units': units}
    planC = pc.importDoseArray(dose3M.astype(np.float64), xV, yV, zV, planC,
                               scanNum, doseInfo)
    return len(planC.dose) - 1


def optimizeAndImportDose(planC, ct, cst, stf, dij, pln, scanNum: int = 0,
                          fractionGroupID: str = 'pyRadPlan'):
    """Run fluence optimization and store the resulting dose in planC.

    Returns:
        tuple: ``(w, doseNum, planC)`` -- optimized beamlet weight vector,
        index of the imported dose in ``planC.dose`` and the container.
    """
    _requirePyRadPlan()
    w = fluence_optimization(ct, cst, stf, dij, pln)
    result = dij.compute_result_ct_grid(np.asarray(w, dtype=np.float64))
    doseImg = result['physical_dose']
    doseNum = importDoseToPlanC(planC, doseImg, scanNum=scanNum,
                                fractionGroupID=fractionGroupID)
    return np.asarray(w), doseNum, planC


def calcBeamletDoseAndImport(planC, w=None, scanNum: int = 0,
                             beamsNum: int = 0, bixelWidth: float = 5.0,
                             machine: str = 'Generic',
                             structNums: Optional[Sequence[int]] = None,
                             targetStructNums: Optional[Sequence[int]] = None,
                             fractionGroupID: str = 'pyRadPlan'):
    """One-call convenience: RTPLAN beams -> dij -> dose in planC.

    Computes the beamlet dose-influence matrix for the geometry of
    ``planC.beams[beamsNum]`` and imports the dose for beamlet weights
    ``w`` (unit weights when None) into ``planC.dose``.

    Returns:
        tuple: ``(dij, stf, doseNum, planC)``.
    """
    _requirePyRadPlan()
    ct, cst, pln = planFromPlanC(planC, scanNum=scanNum, beamsNum=beamsNum,
                                 structNums=structNums,
                                 targetStructNums=targetStructNums,
                                 bixelWidth=bixelWidth, machine=machine)
    stf, dij = calcDoseInfluence(ct, cst, pln)
    if w is None:
        w = np.ones(dij.total_num_of_bixels, dtype=np.float64)
    result = dij.compute_result_ct_grid(np.asarray(w, dtype=np.float64))
    doseNum = importDoseToPlanC(planC, result['physical_dose'],
                                scanNum=scanNum,
                                fractionGroupID=fractionGroupID)
    return dij, stf, doseNum, planC


# --------------------------------------------------------------------------
# Objective helpers (thin wrappers so callers need not import pyRadPlan)
# --------------------------------------------------------------------------

def squaredDeviation(dRef: float, priority: float = 100.0):
    """Target objective: penalize deviation from ``dRef`` Gy."""
    _requirePyRadPlan()
    return SquaredDeviation(d_ref=dRef, priority=priority)


def squaredOverdosing(dMax: float, priority: float = 100.0):
    """OAR objective: penalize dose above ``dMax`` Gy."""
    _requirePyRadPlan()
    # pyRadPlan names this parameter ``d_max`` (not ``d_ref``); passing the
    # wrong keyword is silently ignored and leaves the default 30 Gy in place.
    return SquaredOverdosing(d_max=dMax, priority=priority)


def squaredUnderdosing(dMin: float, priority: float = 100.0):
    """Target objective: penalize dose below ``dMin`` Gy."""
    _requirePyRadPlan()
    # pyRadPlan names this parameter ``d_min`` (not ``d_ref``).
    return SquaredUnderdosing(d_min=dMin, priority=priority)


def meanDose(dRef: float = 0.0, priority: float = 1.0):
    """OAR objective: penalize mean dose."""
    _requirePyRadPlan()
    return MeanDose(d_ref=dRef, priority=priority)
