"""qib
~~~~~

QIB (Quadrant Infinite Beam) photon pencil-beam dose calculation.

Python port of Matlab CERR:
    IMRTP/getQIBDose.m            (the kernel evaluation, all DoseTerm modes)
    IMRTP/GetPBConsts.m           ('nearest' branch)
    IMRTP/applyIMRTCompression.m  (scatter compression of the influence)
    IMRTP/RTOGVectors2Gantry.m
    IMRTP/gantry2RTOGVectors.m

Algorithm (Ahnesjo et al., Med. Phys. 19, 263-273 (1992)): dose around a
pencil beam is A(z)exp(-a(z)r)/r + B(z)exp(-b(z)r)/r; the dose from a
finite-width beamlet is obtained by differencing precomputed quadrant
infinite integrals of the kernel.  An optional Gaussian smear of the
incident fluence models the finite source size (``sigma_100`` = projected
source sigma at 100 cm).

Outputs are in relative energy deposited per (g/cm^3) from a
point-monodirectional beam (same units as Matlab CERR).

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

import math

import numpy as np

from .qib_data import QIBDataS, loadPBData

LOW_DOSE_CUTOFF = 1e-4   # getQIBDose.m: cut all doses below this value

#: short cutoff (cm) around the beamlet for the fast primary component
_PRIMARY_CUTOFF = {6: 1.0, 8: 2.0, 18: 3.0}
_SCATTER_CUTOFF = 8.0


# --------------------------------------------------------------------------
# Depth-dependent kernel constants (GetPBConsts.m, 'nearest' branch)
# --------------------------------------------------------------------------

def getPBConsts(depthsV: np.ndarray, energy: float, qib: QIBDataS):
    """Nearest-neighbour lookup of the Ahnesjo A, a, B, b constants vs depth.

    Port of ``GetPBConsts.m`` with flag ``'nearest'`` (the branch used by
    ``generateQIBInfluence.m``).  Depths in g/cm^2 (radiological), clipped
    at the 50 cm dosimetry limit.
    """
    energy = int(round(float(energy)))
    if energy == 6:
        paramMat = qib.aahn6b
    elif energy == 18:
        paramMat = np.flipud(qib.aahn18b)
    else:
        raise ValueError(
            'QIB kernel data is only available for 6 and 18 MV beams '
            '(got beamEnergy=%s). Set beam.beamEnergy to 6 or 18.' % energy)

    maxDepth = 50.0
    depthsV = np.asarray(depthsV, dtype=np.float64)
    if np.any(depthsV > maxDepth):
        print('Warning!  Some depths exceed 50 cm, which is dosimetry limit!')
    depthsV = np.minimum(depthsV, maxDepth)

    # Matlab (1-based): index = round((d + 0.075)/0.15) + 1
    indexV = np.rint((depthsV + 0.075) / 0.15).astype(np.intp)
    indexV = np.clip(indexV, 0, paramMat.shape[0] - 1)

    scale = 100.0
    A_zV = paramMat[indexV, 1] * scale
    a_zV = paramMat[indexV, 2]
    B_zV = paramMat[indexV, 3] * scale
    b_zV = paramMat[indexV, 4]
    return A_zV, a_zV, B_zV, b_zV


# --------------------------------------------------------------------------
# Quadrant-integral lookup helper
# --------------------------------------------------------------------------

def _quadrantDiff(table: np.ndarray, qib: QIBDataS,
                  x1V, x2V, y1V, y2V):
    """Difference of quadrant infinite integrals over a rectangle.

    ``x1V..y2V`` are already converted to the table's *unscaled* coordinate
    system.  Mirrors the index arithmetic of ``getQIBDose.m`` (nearest-
    neighbour interpolation into the lookup table).
    """
    lim = table.shape[0]

    def toIdx(u):
        i = np.rint(u / qib.deltaQBM).astype(np.intp) + qib.QBMidIndexX
        return np.clip(i, 0, lim - 1)

    u1x, u2x = toIdx(x1V), toIdx(x2V)
    u1y, u2y = toIdx(y1V), toIdx(y2V)
    # Matlab linear index ind = UX + (UY-1)*S(1)  ->  table[UX, UY]
    return (table[u1x, u1y] - table[u2x, u1y]
            - table[u1x, u2y] + table[u2x, u2y])


# --------------------------------------------------------------------------
# getQIBDose
# --------------------------------------------------------------------------

def getQIBDose(posU: np.ndarray, radDepthV: np.ndarray,
               widthUX: np.ndarray, widthUY: np.ndarray,
               qib: QIBDataS,
               A_zV, a_zV, B_zV, b_zV,
               energy: float, how2Compute: str,
               gaussSigma_100: float, distV: np.ndarray) -> np.ndarray:
    """Fast pencil-beam dose calculation for photon beams.

    Port of ``getQIBDose.m``.

    Args:
        posU:        (N, 2) beamlet-frame coordinates (Xb, Yb) of the dose
                     points, cm.
        radDepthV:   radiological depths (g/cm^2) -- unused by the lookup
                     itself but kept for signature parity.
        widthUX/Y:   beamlet widths at each point's source distance, cm.
        qib:         kernel data from :func:`loadPBData`.
        A_zV..b_zV:  depth constants from :func:`getPBConsts`.
        energy:      nominal beam energy (6 / 18 MV).
        how2Compute: one of ``'primary'``, ``'scatter'``,
                     ``'nogauss+scatter'`` (a.k.a. ``'primary+scatter'``),
                     ``'GaussPrimary'``, ``'GaussPrimary+scatter'``.
        gaussSigma_100: projected source sigma at 100 cm (cm).
        distV:       distance from source to each dose point, cm (needed
                     for the Gaussian smear).

    Returns:
        Dose vector (relative energy deposited per (g/cm^3)).
    """
    how = how2Compute.lower()
    energy = int(round(float(energy)))

    xVa = np.asarray(posU[:, 0], dtype=np.float64)
    yVa = np.asarray(posU[:, 1], dtype=np.float64)
    widthUX = np.asarray(widthUX, dtype=np.float64)
    widthUY = np.asarray(widthUY, dtype=np.float64)
    dosesV = np.zeros_like(xVa)

    # ---- composite modes recurse, as in the Matlab original -------------
    if how in ('nogauss+scatter', 'primary+scatter'):
        return (getQIBDose(posU, radDepthV, widthUX, widthUY, qib,
                           A_zV, a_zV, B_zV, b_zV, energy, 'primary',
                           gaussSigma_100, distV)
                + getQIBDose(posU, radDepthV, widthUX, widthUY, qib,
                             A_zV, a_zV, B_zV, b_zV, energy, 'scatter',
                             gaussSigma_100, distV))
    if how == 'gaussprimary+scatter':
        return (getQIBDose(posU, radDepthV, widthUX, widthUY, qib,
                           A_zV, a_zV, B_zV, b_zV, energy, 'gaussprimary',
                           gaussSigma_100, distV)
                + getQIBDose(posU, radDepthV, widthUX, widthUY, qib,
                             A_zV, a_zV, B_zV, b_zV, energy, 'scatter',
                             gaussSigma_100, distV))

    if how == 'primary':
        cutoff = _PRIMARY_CUTOFF[energy]
        xCatch = widthUX / 2.0 + cutoff
        yCatch = widthUY / 2.0 + cutoff
        go = (np.abs(xVa) < xCatch) & (np.abs(yVa) < yCatch)
        if np.any(go):
            x1 = (xVa[go] - widthUX[go] / 2.0) * a_zV[go]
            x2 = (xVa[go] + widthUX[go] / 2.0) * a_zV[go]
            y1 = (yVa[go] - widthUY[go] / 2.0) * a_zV[go]
            y2 = (yVa[go] + widthUY[go] / 2.0) * a_zV[go]
            dA = _quadrantDiff(qib.QIB, qib, x1, x2, y1, y2)
            dosesV[go] += 2.0 * math.pi * dA * A_zV[go] / a_zV[go]

    elif how == 'scatter':
        cutoff = _SCATTER_CUTOFF
        xCatch = widthUX / 2.0 + cutoff
        yCatch = widthUY / 2.0 + cutoff
        go = (np.abs(xVa) < xCatch) & (np.abs(yVa) < yCatch)
        if np.any(go):
            x1 = (xVa[go] - widthUX[go] / 2.0) * b_zV[go]
            x2 = (xVa[go] + widthUX[go] / 2.0) * b_zV[go]
            y1 = (yVa[go] - widthUY[go] / 2.0) * b_zV[go]
            y2 = (yVa[go] + widthUY[go] / 2.0) * b_zV[go]
            dB = _quadrantDiff(qib.QIB, qib, x1, x2, y1, y2)
            dosesV[go] += 2.0 * math.pi * dB * B_zV[go] / b_zV[go]

    elif how == 'gaussprimary':
        # --- Gaussian smear of the primary component (Ahnesjo) -----------
        distV = np.asarray(distV, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma_zV = distV * (gaussSigma_100 / 100.0)
            s_zV = np.sqrt(qib.k1 / a_zV ** 2 + sigma_zV ** 2)
            tzV = a_zV / (1.0 + qib.k2 * a_zV ** 2 * sigma_zV ** 2)
            wzV = 1.0 / (1.0 + (sigma_zV ** 2 + a_zV ** 2)
                         / (sigma_zV ** 2
                            + (qib.k3 + qib.k4 / sigma_zV) ** 2))
        wzV = np.nan_to_num(wzV, nan=1.0)
        kGauss = (1.0 - wzV) * A_zV / a_zV   # Gaussian term coefficient
        kAV = wzV * A_zV / a_zV              # modified A-term coefficient

        cutoff = _PRIMARY_CUTOFF[energy]
        xCatch = widthUX / 2.0 + cutoff
        yCatch = widthUY / 2.0 + cutoff
        go = (np.abs(xVa) < xCatch) & (np.abs(yVa) < yCatch)
        if np.any(go):
            xR1 = xVa[go] - widthUX[go] / 2.0
            xR2 = xVa[go] + widthUX[go] / 2.0
            yR1 = yVa[go] - widthUY[go] / 2.0
            yR2 = yVa[go] + widthUY[go] / 2.0

            # Non-Gauss (sharp) primary part, scaled by t_z:
            dA = _quadrantDiff(qib.QIB, qib,
                               xR1 * tzV[go], xR2 * tzV[go],
                               yR1 * tzV[go], yR2 * tzV[go])
            dosesV[go] += 2.0 * math.pi * dA * kAV[go]

            # Gaussian primary part, scaled by 1/s_z:
            dG = _quadrantDiff(qib.QIBGauss, qib,
                               xR1 / s_zV[go], xR2 / s_zV[go],
                               yR1 / s_zV[go], yR2 / s_zV[go])
            dosesV[go] += 2.0 * math.pi * dG * kGauss[go]
    else:
        raise ValueError("Unknown DoseTerm '%s'" % how2Compute)

    dosesV[dosesV < LOW_DOSE_CUTOFF] = 0.0
    return dosesV


# --------------------------------------------------------------------------
# Scatter (influence) compression  --  applyIMRTCompression.m
# --------------------------------------------------------------------------

def applyIMRTCompression(params, doseV: np.ndarray) -> np.ndarray:
    """Selectively eliminate low-dose scatter components.

    Port of ``applyIMRTCompression.m``; ``params`` is an
    :class:`~cerr.imrtp.imrtp_problem.IMParams` (uses ``ScatterMethod``,
    ``Scatter.Threshold`` and ``Scatter.RandomStep``).
    """
    doseV = np.asarray(doseV, dtype=np.float64).copy()
    if doseV.size == 0:
        return doseV
    method = params.ScatterMethod.lower()
    thresh = params.Scatter.Threshold
    maxD = doseV.max()

    if method == 'exponential':
        # Keep low-dose points with probability proportional to their dose.
        doseVLow = doseV.copy()
        doseVLow[doseVLow > thresh * maxD] = 0.0
        lowInd = np.flatnonzero(doseVLow)
        if lowInd.size:
            order = np.argsort(doseVLow[lowInd], kind='stable')
            sortDose = doseVLow[lowInd][order]
            normSortDose = sortDose / sortDose.max()
            coins = np.random.rand(normSortDose.size)
            keepers = np.flatnonzero(normSortDose > coins)
            doseV[doseV < thresh * maxD] = 0.0
            keepIdx = lowInd[order[keepers]]
            doseV[keepIdx] = doseVLow[keepIdx]
    elif method in ('random', 'probabilistic'):
        step = int(params.Scatter.RandomStep)
        doseVLow = doseV.copy()
        doseVLow[doseVLow > thresh * maxD] = 0.0
        lowInd = np.flatnonzero(doseVLow)
        n = lowInd.size // step
        doseVLowDown = np.zeros_like(doseV)
        if n > 1:
            downInd = (np.round(np.random.rand(n) * (step - 1)).astype(int)
                       + np.arange(n) * step)
            downInd = downInd[:n - 1]
            downInd = downInd[downInd < doseV.size]
            doseVLowDown[downInd] = doseVLow[downInd]
        doseV[doseV < thresh * maxD] = 0.0
        doseV = doseV + doseVLowDown
    elif method == 'threshold':
        doseV[doseV < thresh * maxD] = 0.0
    else:
        raise ValueError('Invalid compression method in IM.params.'
                         'ScatterMethod: %s' % params.ScatterMethod)
    return doseV


# --------------------------------------------------------------------------
# Gantry <-> RTOG coordinate transforms
# --------------------------------------------------------------------------

def rtogVectors2Gantry(rtogVectorsM: np.ndarray,
                       gantryAngle: float) -> np.ndarray:
    """Rotate RTOG-frame vectors into the gantry frame.

    Port of ``RTOGVectors2Gantry.m``.  Gantry frame: Xb in-plane,
    Yb out-of-plane (along -z), Zb along the beam axis.
    """
    c = math.cos(math.radians(gantryAngle))
    s = math.sin(math.radians(gantryAngle))
    out = np.empty_like(np.asarray(rtogVectorsM, dtype=np.float64))
    r = np.asarray(rtogVectorsM, dtype=np.float64)
    out[:, 0] = c * r[:, 0] - s * r[:, 1]
    out[:, 1] = -r[:, 2]
    out[:, 2] = s * r[:, 0] + c * r[:, 1]
    return out


def gantry2RTOGVectors(gantryVectorsM: np.ndarray, gantryAngle: float,
                       couchAngle: float = 0.0) -> np.ndarray:
    """Rotate gantry-frame vectors back into the RTOG frame.

    Port of ``gantry2RTOGVectors.m`` (including couch rotation).
    """
    c = math.cos(math.radians(gantryAngle))
    s = math.sin(math.radians(gantryAngle))
    g = np.asarray(gantryVectorsM, dtype=np.float64)
    out = np.empty_like(g)
    out[:, 0] = s * g[:, 2] + c * g[:, 0]
    out[:, 1] = c * g[:, 2] - s * g[:, 0]
    out[:, 2] = -g[:, 1]
    if couchAngle not in (0.0, 360.0):
        cc = math.cos(math.radians(couchAngle))
        sc = math.sin(math.radians(couchAngle))
        pat = np.empty_like(out)
        pat[:, 0] = cc * out[:, 0] - sc * out[:, 2]
        pat[:, 1] = out[:, 1]
        pat[:, 2] = sc * out[:, 0] + cc * out[:, 2]
        out = pat
    return out
