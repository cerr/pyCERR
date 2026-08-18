"""gamma module.

3-D gamma-index dose comparison for pyCERR (Low et al., Med. Phys. 25(5),
1998). This is a Python port of the core routine behind MATLAB CERR's
``Dose -> Gamma 3D`` menu (``gammaDose3d.m``).

The gamma index combines a dose-difference (DD) criterion and a
distance-to-agreement (DTA) criterion into a single pass/fail map: for every
reference voxel, gamma is the minimum over the evaluated distribution of

    gamma = sqrt( (dist / DTA)^2 + (doseDiff / DD)^2 ),

so gamma <= 1 means the evaluated dose agrees with the reference within either
the dose or the distance tolerance. DD may be evaluated *globally* (a fixed
percentage of a normalization dose) or *locally* (a percentage of the local
reference dose).

Two entry points:

- :func:`gammaDose3d` -- core routine on two arrays already on the same grid.
- :func:`gammaDose3dForDoses` -- convenience wrapper that resamples one
  ``planC.dose`` onto another's grid (e.g. a clinical TPS dose vs a pyCERR
  dose) and returns the gamma map plus the pass rate.

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

import numpy as np
from scipy.ndimage import shift as _ndshift

import cerr.contour.rasterseg as rs


def gammaDose3d(doseRef3M, doseEval3M, spacingV,
                distAgreement=3.0, doseAgreement=3.0,
                thresholdDose=0.0, doseAgreementType='global',
                normalizationDose=None, maxSearchDistance=None,
                distSampleRate=3):
    """Compute the 3-D gamma index between two dose arrays on a common grid.

    Args:
        doseRef3M (np.ndarray): Reference dose, shape (nRows, nCols, nSlices).
            Reference voxels define the points at which gamma is evaluated.
        doseEval3M (np.ndarray): Evaluated dose, same shape and grid as the
            reference.
        spacingV (sequence of 3 floats): Voxel spacing in mm along the three
            array axes ``(axis0, axis1, axis2)`` (i.e. rows, cols, slices).
        distAgreement (float): Distance-to-agreement (DTA) criterion in mm.
        doseAgreement (float): Dose-difference (DD) criterion in percent.
        thresholdDose (float): Reference-dose cutoff (same dose units as the
            arrays). Voxels with ``doseRef3M < thresholdDose`` are excluded
            (set to NaN in the output).
        doseAgreementType (str): 'global' -- DD is ``doseAgreement`` percent of
            ``normalizationDose``; 'local' -- DD is ``doseAgreement`` percent of
            each voxel's reference dose.
        normalizationDose (float or None): Dose used for the global DD
            criterion. Defaults to the reference dose maximum.
        maxSearchDistance (float or None): Radius (mm) of the neighborhood
            searched around each reference voxel. Defaults to ``2 * DTA``.
        distSampleRate (int): Number of search samples per DTA along each axis
            (higher = finer distance resolution, slower). Default 3.

    Returns:
        np.ndarray: Gamma index per voxel (same shape), NaN where the reference
        dose is below ``thresholdDose``.
    """
    doseRef3M = np.asarray(doseRef3M, dtype=float)
    doseEval3M = np.asarray(doseEval3M, dtype=float)
    if doseRef3M.shape != doseEval3M.shape:
        raise ValueError("doseRef3M and doseEval3M must have the same shape; "
                         "got %s and %s" % (doseRef3M.shape, doseEval3M.shape))
    spacingV = np.asarray(spacingV, dtype=float)
    if spacingV.size != 3:
        raise ValueError("spacingV must have 3 elements (mm per array axis)")

    dta = float(distAgreement)
    if dta <= 0:
        raise ValueError("distAgreement (DTA) must be positive")
    if maxSearchDistance is None:
        maxSearchDistance = 2.0 * dta

    agreeType = doseAgreementType.lower()
    if agreeType == 'global':
        normDose = normalizationDose if normalizationDose is not None \
            else np.nanmax(doseRef3M)
        ddAbs = doseAgreement / 100.0 * float(normDose)     # scalar Gy
        if ddAbs <= 0:
            raise ValueError("global dose-difference criterion is zero; check "
                             "doseAgreement/normalizationDose")
    elif agreeType == 'local':
        with np.errstate(divide='ignore', invalid='ignore'):
            ddAbs = doseAgreement / 100.0 * doseRef3M       # per-voxel Gy
    else:
        raise ValueError("doseAgreementType must be 'global' or 'local'")

    # Search offsets on an isotropic lattice (mm) within the search sphere.
    step = dta / float(distSampleRate)
    nSteps = int(np.ceil(maxSearchDistance / step))
    offV = np.arange(-nSteps, nSteps + 1) * step

    gammaSq = np.full(doseRef3M.shape, np.inf)
    for o0 in offV:
        for o1 in offV:
            for o2 in offV:
                distMm = np.sqrt(o0 * o0 + o1 * o1 + o2 * o2)
                if distMm > maxSearchDistance:
                    continue
                # Sample eval at ref-index + offset: shift eval by -offset.
                voxShift = (-o0 / spacingV[0], -o1 / spacingV[1],
                            -o2 / spacingV[2])
                evalShift = _ndshift(doseEval3M, shift=voxShift, order=1,
                                     mode='constant', cval=np.nan)
                dDose = evalShift - doseRef3M
                distTerm = (distMm / dta) ** 2
                with np.errstate(divide='ignore', invalid='ignore'):
                    ddTerm = (dDose / ddAbs) ** 2
                g2 = distTerm + ddTerm
                # fmin ignores NaN (out-of-bounds / zero-local-dose voxels).
                np.fmin(gammaSq, g2, out=gammaSq)

    gamma3M = np.sqrt(gammaSq)
    # Exclude voxels with no valid reference dose (NaN, e.g. resampled outside
    # the dose field) or below the threshold.
    with np.errstate(invalid='ignore'):
        excluded = np.isnan(doseRef3M) | (doseRef3M < thresholdDose)
    gamma3M[excluded] = np.nan
    return gamma3M


def gammaPassRate(gamma3M):
    """Fraction of evaluated voxels (non-NaN) with gamma <= 1.

    Args:
        gamma3M (np.ndarray): Gamma map from :func:`gammaDose3d`.

    Returns:
        float: Pass rate in [0, 1], or NaN if no voxels were evaluated.
    """
    valid = ~np.isnan(gamma3M)
    n = int(np.count_nonzero(valid))
    if n == 0:
        return float('nan')
    return float(np.count_nonzero(gamma3M[valid] <= 1.0) / n)


def _doseOnGrid(doseObj, xV, yV, zV, shape):
    """Resample a pyCERR dose object onto the grid defined by (xV, yV, zV).

    Args:
        doseObj (cerr.dataclasses.dose.Dose): dose to resample.
        xV, yV, zV (np.ndarray): target grid coordinate vectors (cols, rows,
            slices) in pyCERR virtual cm.
        shape (tuple): target array shape (nRows, nCols, nSlices).

    Returns:
        np.ndarray: dose sampled on the target grid, shape ``shape``.
    """
    Yg, Xg, Zg = np.meshgrid(yV, xV, zV, indexing='ij')
    dose = np.asarray(doseObj.getDoseAt(Xg.ravel(), Yg.ravel(), Zg.ravel()),
                      dtype=float)
    return dose.reshape(shape)


def _spacingMmFromGrid(xV, yV, zV):
    """(row, col, slice) voxel spacing in mm from cm grid vectors."""
    return (abs(yV[1] - yV[0]) * 10.0,
            abs(xV[1] - xV[0]) * 10.0,
            abs(zV[1] - zV[0]) * 10.0)


def gammaDose3dForScan(refDoseNum, evalDoseNum, scanNum, planC,
                       distAgreement=3.0, doseAgreement=3.0,
                       thresholdFraction=0.2, doseAgreementType='global',
                       normalizationDose=None, maxSearchDistance=None,
                       distSampleRate=3):
    """Gamma between two doses, evaluated on a scan grid.

    Both doses are resampled onto ``planC.scan[scanNum]``'s grid so the gamma
    map aligns with that scan and its structures (see
    :func:`gammaByStructure`). This is the entry point used by the viewer's
    Gamma 3D tool.

    Args:
        refDoseNum (int): reference dose index in ``planC.dose``.
        evalDoseNum (int): evaluated dose index in ``planC.dose``.
        scanNum (int): scan index whose grid the gamma is computed on.
        planC (cerr.plan_container.PlanC): plan container.
        distAgreement, doseAgreement, thresholdFraction, doseAgreementType,
        normalizationDose, maxSearchDistance, distSampleRate: see
            :func:`gammaDose3d` / :func:`gammaDose3dForDoses`.

    Returns:
        tuple: ``(gamma3M, passRate)`` on the scan grid.
    """
    scan = planC.scan[scanNum]
    xV, yV, zV = scan.getScanXYZVals()
    shape = tuple(int(v) for v in scan.getScanSize())
    doseRef3M = _doseOnGrid(planC.dose[refDoseNum], xV, yV, zV, shape)
    doseEval3M = _doseOnGrid(planC.dose[evalDoseNum], xV, yV, zV, shape)
    spacingV = _spacingMmFromGrid(xV, yV, zV)
    thresholdDose = float(thresholdFraction) * float(np.nanmax(doseRef3M))
    gamma3M = gammaDose3d(
        doseRef3M, doseEval3M, spacingV,
        distAgreement=distAgreement, doseAgreement=doseAgreement,
        thresholdDose=thresholdDose, doseAgreementType=doseAgreementType,
        normalizationDose=normalizationDose,
        maxSearchDistance=maxSearchDistance, distSampleRate=distSampleRate)
    return gamma3M, gammaPassRate(gamma3M)


def gammaByStructure(gamma3M, structNumList, planC):
    """Per-structure gamma pass rates.

    Args:
        gamma3M (np.ndarray): gamma map on a scan grid (from
            :func:`gammaDose3dForScan`), NaN where not evaluated.
        structNumList (sequence of int): structures to tabulate. Each must be
            on the same scan grid as ``gamma3M``.
        planC (cerr.plan_container.PlanC): plan container.

    Returns:
        list of dict: one entry per structure with keys ``structNum``,
        ``structureName``, ``numEvaluated``, ``numPass`` and ``passRate``
        (NaN when no voxels of the structure were evaluated or the grids
        mismatch).
    """
    results = []
    for structNum in structNumList:
        entry = {'structNum': int(structNum),
                 'structureName': planC.structure[structNum].structureName,
                 'numEvaluated': 0, 'numPass': 0, 'passRate': float('nan')}
        mask3M = rs.getStrMask(structNum, planC)
        if mask3M.shape != gamma3M.shape:
            entry['note'] = 'structure/gamma grid mismatch'
            results.append(entry)
            continue
        g = gamma3M[mask3M.astype(bool)]
        valid = ~np.isnan(g)
        nEval = int(np.count_nonzero(valid))
        nPass = int(np.count_nonzero(g[valid] <= 1.0))
        entry['numEvaluated'] = nEval
        entry['numPass'] = nPass
        entry['passRate'] = (nPass / nEval) if nEval > 0 else float('nan')
        results.append(entry)
    return results


def gammaDose3dForDoses(refDoseNum, evalDoseNum, planC,
                        distAgreement=3.0, doseAgreement=3.0,
                        thresholdFraction=0.2, doseAgreementType='global',
                        normalizationDose=None, maxSearchDistance=None,
                        distSampleRate=3):
    """Gamma between two ``planC.dose`` objects, resampled to a common grid.

    The evaluated dose is interpolated onto the reference dose grid, so the two
    may originate from different systems/resolutions (e.g. a clinical TPS
    RTDOSE vs a pyCERR-computed dose). The dose-difference cutoff is given as a
    fraction of the reference dose maximum, mirroring the usual
    ``lower_percent_dose_cutoff`` convention.

    Args:
        refDoseNum (int): Index of the reference dose in ``planC.dose``.
        evalDoseNum (int): Index of the evaluated dose in ``planC.dose``.
        planC (cerr.plan_container.PlanC): pyCERR plan container.
        distAgreement (float): DTA in mm.
        doseAgreement (float): DD in percent.
        thresholdFraction (float): Reference-dose cutoff as a fraction of the
            reference maximum (e.g. 0.2 = exclude voxels below 20% of max).
        doseAgreementType (str): 'global' or 'local'.
        normalizationDose, maxSearchDistance, distSampleRate: see
            :func:`gammaDose3d`.

    Returns:
        tuple:
            - np.ndarray: gamma map on the reference grid.
            - float: gamma pass rate (fraction of evaluated voxels <= 1).
    """
    refDose = planC.dose[refDoseNum]
    evalDose = planC.dose[evalDoseNum]

    doseRef3M = np.asarray(refDose.doseArray, dtype=float)
    xR, yR, zR = refDose.getDoseXYZVals()
    nRows, nCols, nSlices = doseRef3M.shape
    if not (len(yR) == nRows and len(xR) == nCols and len(zR) == nSlices):
        raise ValueError("reference dose grid vectors do not match doseArray "
                         "shape %s (rows=%d, cols=%d, slices=%d)"
                         % (doseRef3M.shape, len(yR), len(xR), len(zR)))

    # Resample the evaluated dose onto the reference grid (pyCERR cm coords).
    doseEval3M = _doseOnGrid(evalDose, xR, yR, zR, doseRef3M.shape)

    # Voxel spacing in mm along (rows=y, cols=x, slices=z); grid is in cm.
    spacingV = _spacingMmFromGrid(xR, yR, zR)

    thresholdDose = float(thresholdFraction) * float(np.nanmax(doseRef3M))
    gamma3M = gammaDose3d(
        doseRef3M, doseEval3M, spacingV,
        distAgreement=distAgreement, doseAgreement=doseAgreement,
        thresholdDose=thresholdDose, doseAgreementType=doseAgreementType,
        normalizationDose=normalizationDose,
        maxSearchDistance=maxSearchDistance, distSampleRate=distSampleRate)
    return gamma3M, gammaPassRate(gamma3M)
