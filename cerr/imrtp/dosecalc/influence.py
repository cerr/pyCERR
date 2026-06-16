"""influence
~~~~~~~~~

QIB influence (beamlet) generation and dose assembly.

Python port of Matlab CERR:
    IMRTP/generateQIBInfluence.m   (per-structure, per-beam, per-PB dose)
    IMRTP/getIMDose.m              (influence x beamlet weights -> dose3D)
    IMRTP/createIMBeamlet.m        (sparse beamlet records)

Beamlets are stored on each beam as ``beam.beamlets`` -- a flat list of
:class:`~cerr.imrtp.imrtp_problem.Beamlet`, one per (structure, pencil
beam), holding the non-zero dose contributions (float32, uncompressed) and
the voxel indices into the scan grid.

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy import ndimage

from cerr.contour import rasterseg as rs
from cerr.dataclasses import structure as structr

from .. import imrtp_problem as imp
from .qib import (applyIMRTCompression, getPBConsts, getQIBDose,
                  rtogVectors2Gantry)
from .qib_data import loadPBData
from .raytrace import ScanGrid, setBeamRayData


# --------------------------------------------------------------------------
# Structure sampling (getDown3Mask / getSurfaceExpand equivalents)
# --------------------------------------------------------------------------

def _samplePointsMask(mask: np.ndarray, sampleRate: int) -> np.ndarray:
    """Dose-calculation point mask for a structure at a given sample rate.

    For ``sampleRate == 1`` every structure voxel is used.  Otherwise a
    regular in-plane grid (every ``sampleRate``-th row/column) is used,
    augmented with the structure surface so that edges are never missed
    (mirrors ``getSurfaceExpand`` + ``getDown3Mask`` in
    ``generateQIBInfluence.m``).
    """
    if sampleRate == 1:
        return mask
    if 2 ** int(round(np.log2(sampleRate))) != sampleRate:
        raise ValueError('Sample factor must (currently) be a power of 2.')
    grid = np.zeros_like(mask)
    grid[::sampleRate, ::sampleRate, :] = True
    surface = mask & ~ndimage.binary_erosion(mask)
    return (mask & grid) | surface


def _structSamples(strNum: int, sampleRate: int, planC, grid: ScanGrid):
    """(scanIndV, x, y, z, mask, sampled) for a structure's calc points."""
    mask = rs.getStrMask(strNum, planC).astype(bool)
    sampled = _samplePointsMask(mask, int(sampleRate))
    r, c, s = np.nonzero(sampled)
    scanIndV = np.ravel_multi_index((r, c, s), grid.shape)
    return scanIndV, grid.xV[c], grid.yV[r], grid.zV[s], mask, sampled


# --------------------------------------------------------------------------
# generateQIBInfluence
# --------------------------------------------------------------------------

def generateQIBInfluence(im, planC, statusCallback=None):
    """Generate QIB beamlet influence for every goal structure and beam.

    Port of ``generateQIBInfluence.m``.  Fills ``beam.beamlets`` on every
    beam of ``im`` and returns ``im``.
    """
    def status(msg, frac=None):
        if statusCallback:
            statusCallback(msg, frac)

    qibData = loadPBData()

    # Ray-trace geometry for every beam (getPBList in IMRTP.m):
    grid = setBeamRayData(im, planC, statusCallback)

    # Unique ROI structure list (getROIStructureList in IMRTP.m):
    structRoiV, sampleRateV = [], []
    for g in im.goals:
        strNum = structr.getStructNumFromUID(g.strUID, planC)
        if strNum not in structRoiV:
            structRoiV.append(strNum)
            sampleRateV.append(int(g.xySampleRate))

    cutoff2 = float(im.params.cutoffDistance) ** 2
    doseFlag = im.params.DoseTerm

    for beam in im.beams:
        beam.beamlets = []

    for iStr, (strNum, rate) in enumerate(zip(structRoiV, sampleRateV)):
        scanIndV, xV, yV, zV, _, _ = _structSamples(strNum, rate, planC,
                                                    grid)
        pM = np.stack([xV, yV, zV], axis=1)
        numPts = pM.shape[0]
        strName = planC.structure[strNum].structureName
        strUID = planC.structure[strNum].strUID

        for j, beam in enumerate(im.beams):
            status('Computing dose to structure %s for beam %d...'
                   % (strName, j + 1), None)
            src = np.array([beam.x, beam.y, beam.z])
            pRelM = pM - src[None, :]
            numPBs = beam.RTOGPBVectorsM.shape[0]

            for pbNum in range(numPBs):
                if statusCallback and (pbNum % 25 == 0
                                       or pbNum == numPBs - 1):
                    status('Beam %d/%d, structure %s: PB %d of %d'
                           % (j + 1, len(im.beams), strName,
                              pbNum + 1, numPBs),
                           ((iStr + (j + (pbNum + 1) / numPBs)
                             / len(im.beams)) / len(structRoiV)))
                pbV = beam.RTOGPBVectorsM[pbNum]
                trace = beam.CTTraceS[pbNum]

                # Distance along the PB ray of each point's closest approach:
                distV = pRelM @ pbV
                qM = src[None, :] + distV[:, None] * pbV[None, :]
                rM = pM - qM
                rDist2 = np.einsum('ij,ij->i', rM, rM)
                goV = rDist2 < cutoff2
                if not np.any(goV):
                    beam.beamlets.append(_makeBeamlet(
                        np.empty(0), np.empty(0, dtype=np.intp), j,
                        goV.size, strName, strUID, rate))
                    continue

                rGoM = rM[goV]
                distGoV = distV[goV]

                gantryM = rtogVectors2Gantry(rGoM, beam.gantryAngle)
                xb, yb = gantryM[:, 0], gantryM[:, 1]

                tmpV = np.clip(distGoV, 0.001,
                               trace.distSamplePts.max() - 0.001)
                radDepthV = np.interp(
                    tmpV,
                    np.concatenate(([0.0], trace.distSamplePts)),
                    np.concatenate(([0.0], trace.cumDensityRay)))

                pbWidthY = (beam.beamletDelta_y * distGoV
                            / beam.isodistance)
                pbWidthX = (beam.beamletDelta_x * distGoV
                            / beam.isodistance)

                A_zV, a_zV, B_zV, b_zV = getPBConsts(
                    radDepthV, beam.beamEnergy, qibData)

                doseV = getQIBDose(np.stack([xb, yb], axis=1), radDepthV,
                                   pbWidthX, pbWidthY, qibData,
                                   A_zV, a_zV, B_zV, b_zV,
                                   beam.beamEnergy, doseFlag,
                                   beam.sigma_100, distGoV)

                # Inverse-square law (divide by depth-dependent beamlet
                # area scaled to the isocenter distance):
                doseV = (beam.beamletDelta_x * beam.beamletDelta_y
                         * doseV / (pbWidthX * pbWidthY))

                # Scatter compression:
                doseV = applyIMRTCompression(im.params, doseV)

                nz = doseV != 0
                beamlet = _makeBeamlet(doseV[nz],
                                       scanIndV[goV][nz], j,
                                       int(goV.size), strName, strUID,
                                       rate)
                beam.beamlets.append(beamlet)

    im.isFresh = True
    return im


def _makeBeamlet(doseV, indV, beamNum, fullLength, strName, strUID, rate):
    """Port of ``createIMBeamlet.m`` (uncompressed float32 storage)."""
    b = imp.Beamlet()
    b.format = 'float32'
    b.beamNum = beamNum
    b.fullLength = int(fullLength)
    b.structureName = strName
    b.strUID = strUID
    b.sampleRate = int(rate)
    b.influence = np.asarray(doseV, dtype=np.float32)
    b.indexV = np.asarray(indV, dtype=np.intp)
    b.maxInfluenceVal = float(b.influence.max()) if b.influence.size else 0.0
    return b


# --------------------------------------------------------------------------
# getIMDose: influence x weights -> 3-D dose
# --------------------------------------------------------------------------

def getIMDose(im, weightsV: Optional[Sequence[float]], structNumsV,
              planC) -> np.ndarray:
    """Assemble a 3-D dose distribution from beamlets and PB weights.

    Port of ``getIMDose.m``.

    Args:
        im:          IMRTProblem with computed beamlets.
        weightsV:    one weight per pencil beam (flat, ordered beam-by-beam
                     in PB order), or None for unit weights (open fields).
        structNumsV: structure indices over which dose is assembled.
        planC:       plan container.

    Returns:
        np.ndarray of dose on the scan grid (rows x cols x slices).

    Note:
        Where Matlab CERR fills sub-sampled structures with 3-D linear
        interpolation, this port uses nearest-sampled-point filling inside
        the structure mask.
    """
    scanNum = im.assocScanNum(planC)
    grid = ScanGrid(planC, scanNum)
    dose3D = np.zeros(grid.shape, dtype=np.float64)

    if np.isscalar(structNumsV):
        structNumsV = [int(structNumsV)]

    # Group each beam's beamlets by structure UID, preserving PB order:
    uidOf = {n: planC.structure[n].strUID for n in structNumsV}

    # Per-structure sample rates (from the stored beamlets):
    rateOf = {}
    for beam in im.beams:
        for b in beam.beamlets:
            rateOf.setdefault(b.strUID, b.sampleRate)

    # Coarsest first so that finer-sampled structures win in overlaps:
    order = sorted(structNumsV,
                   key=lambda n: -rateOf.get(uidOf[n], 1))

    # Flat PB weight vector -> per (beam, pb) lookup:
    pbCounts = [beam.RTOGPBVectorsM.shape[0] if beam.beamlets else 0
                for beam in im.beams]
    if weightsV is None:
        weightsV = np.ones(int(np.sum(pbCounts)))
    weightsV = np.asarray(weightsV, dtype=np.float64).ravel()
    if weightsV.size != int(np.sum(pbCounts)):
        raise ValueError('weightsV must have one entry per pencil beam '
                         '(%d expected, got %d).'
                         % (int(np.sum(pbCounts)), weightsV.size))
    wStart = np.concatenate(([0], np.cumsum(pbCounts)))

    for strNum in order:
        uid = uidOf[strNum]
        rate = rateOf.get(uid, 1)
        flat = np.zeros(int(np.prod(grid.shape)), dtype=np.float64)
        touched = np.zeros(flat.size, dtype=bool)

        for j, beam in enumerate(im.beams):
            pbIdx = 0
            for b in beam.beamlets:
                if b.strUID != uid:
                    continue
                if b.indexV is not None and b.indexV.size:
                    w = weightsV[wStart[j] + pbIdx]
                    flat[b.indexV] += w * b.influence
                    touched[b.indexV] = True
                pbIdx += 1

        doseStr = flat.reshape(grid.shape)

        mask = rs.getStrMask(strNum, planC).astype(bool)
        if rate != 1:
            # Fill non-sampled structure voxels from the nearest sampled
            # point (Matlab uses 3-D linear interpolation here):
            sampled = _samplePointsMask(mask, rate)
            if sampled.any():
                _, idx = ndimage.distance_transform_edt(
                    ~sampled, sampling=(grid.dy, grid.dx, grid.dz),
                    return_indices=True)
                filled = doseStr[idx[0], idx[1], idx[2]]
                dose3D[mask] = filled[mask]
        else:
            dose3D[mask] = doseStr[mask]

    return dose3D
