"""raytrace
~~~~~~~~

Pencil-beam geometry and CT ray tracing for IMRTP dose calculation.

Python port of Matlab CERR:
    IMRTP/IMRTP.m (getTargetSurfacePoints / getPBList helpers)
    IMRTP/getPBRays.m       (which PBs are needed to cover the target)
    IMRTP/getPBRayData.m    (radiological-depth trace through the CT)

Coordinates are pyCERR/CERR "virtual" coordinates as returned by
``planC.scan[n].getScanXYZVals()`` (x ascending, y descending, z ascending),
all in cm.  The scan array is indexed ``[row, col, slice]`` with row <-> y
and col <-> x.

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import ndimage

from cerr.contour import rasterseg as rs
from cerr.dataclasses import structure as structr

from .qib import gantry2RTOGVectors, rtogVectors2Gantry

WATER_CT_NUM = 1000.0   # getPBRayData.m: assumes water equals 1000
RAY_LENGTH = 500.0      # cm; length of rays passed through the CT


# --------------------------------------------------------------------------
# Scan-grid helpers
# --------------------------------------------------------------------------

class ScanGrid:
    """Cached geometry of a pyCERR scan, plus trilinear density sampling."""

    def __init__(self, planC, scanNum: int):
        scanObj = planC.scan[scanNum]
        xV, yV, zV = scanObj.getScanXYZVals()
        self.xV = np.asarray(xV, dtype=np.float64)        # ascending
        self.yV = np.asarray(yV, dtype=np.float64)        # descending
        self.zV = np.asarray(zV, dtype=np.float64)        # ascending
        self.dx = float(abs(self.xV[1] - self.xV[0]))
        self.dy = float(abs(self.yV[0] - self.yV[1]))
        self.dz = float(abs(self.zV[1] - self.zV[0])) if len(self.zV) > 1 \
            else 1.0
        self.shape = (len(self.yV), len(self.xV), len(self.zV))
        # CT numbers with water = 1000 (HU + 1000):
        hu = scanObj.getScanArray().astype(np.float64)
        self.ctNums = np.maximum(hu + WATER_CT_NUM, 0.0)
        # Bounding box of the scan (voxel centers +- half a voxel):
        self.boxMin = np.array([self.xV.min() - self.dx / 2,
                                self.yV.min() - self.dy / 2,
                                self.zV.min() - self.dz / 2])
        self.boxMax = np.array([self.xV.max() + self.dx / 2,
                                self.yV.max() + self.dy / 2,
                                self.zV.max() + self.dz / 2])

    def fracIndex(self, x, y, z):
        """Fractional [row, col, slice] indices of (x, y, z) points."""
        col = (np.asarray(x) - self.xV[0]) / self.dx
        row = (self.yV[0] - np.asarray(y)) / self.dy
        slc = np.interp(np.asarray(z), self.zV, np.arange(len(self.zV)))
        return row, col, slc

    def sampleCTNums(self, x, y, z):
        """Trilinear interpolation of CT numbers at (x, y, z), 0 outside."""
        row, col, slc = self.fracIndex(x, y, z)
        return ndimage.map_coordinates(self.ctNums,
                                       np.vstack([row, col, slc]),
                                       order=1, mode='constant', cval=0.0)


# --------------------------------------------------------------------------
# Target surface points (IMRTP.m: getTargetSurfacePoints / getSurface)
# --------------------------------------------------------------------------

def getTargetSurfacePoints(im, planC):
    """Surface points of all target structures, expanded by PBMargin + 0.5 cm.

    Returns (xS, yS, zS) coordinate vectors (cm) of the surface voxels of
    the (margin-expanded) union of the target structures, mirroring
    ``getSurface(structTargetV, PBMarginV + 0.5, ...)`` in ``IMRTP.m``.
    """
    scanNum = im.assocScanNum(planC)
    grid = ScanGrid(planC, scanNum)
    surf = np.zeros(grid.shape, dtype=bool)

    targets = [g for g in im.goals if str(g.isTarget).lower().startswith('y')]
    if not targets:
        raise ValueError('No target structures (isTarget = yes) defined.')

    seen = set()
    for g in targets:
        strNum = structr.getStructNumFromUID(g.strUID, planC)
        if strNum in seen:
            continue
        seen.add(strNum)
        mask = rs.getStrMask(strNum, planC).astype(bool)
        if not mask.any():
            continue
        margin = float(g.PBMargin) + 0.5
        # Expand by `margin` cm using an anisotropic distance transform.
        dist = ndimage.distance_transform_edt(
            ~mask, sampling=(grid.dy, grid.dx, grid.dz))
        expanded = dist <= margin
        surf |= expanded & ~ndimage.binary_erosion(expanded)

    r, c, s = np.nonzero(surf)
    if r.size == 0:
        raise ValueError('Target structure mask is empty.')
    return grid.xV[c], grid.yV[r], grid.zV[s], grid


# --------------------------------------------------------------------------
# Pencil-beam grid (getPBRays.m)
# --------------------------------------------------------------------------

def getPBRays(xS, yS, zS, beam):
    """Determine the pencil beams required to cover the target.

    Port of ``getPBRays.m``: target surface points are projected (through
    the source) onto the gantry frame at the isocenter distance, binned on
    the beamlet grid, row gaps are filled, and unit RTOG-frame direction
    vectors are generated for each pencil beam.

    Returns:
        (RTOGPBVectorsM, RTOGPBVectorsM_MC, PBMaskM, rowPBV, colPBV,
         xPBPosV, yPBPosV)
    """
    bdx = float(beam.beamletDelta_x)
    bdy = float(beam.beamletDelta_y)
    src = np.array(beam.sourcePos(), dtype=np.float64)

    dxyz = np.stack([np.asarray(xS) - src[0],
                     np.asarray(yS) - src[1],
                     np.asarray(zS) - src[2]], axis=1)
    dxyz /= np.linalg.norm(dxyz, axis=1, keepdims=True)

    gant = rtogVectors2Gantry(dxyz, beam.gantryAngle)
    normG = np.linalg.norm(gant, axis=1)
    xProj = gant[:, 0] / normG * beam.isodistance
    yProj = gant[:, 1] / normG * beam.isodistance

    minCol = int(np.floor(xProj.min() / bdx))
    maxCol = int(np.ceil(xProj.max() / bdx))
    minRow = int(np.floor(yProj.min() / bdy))
    maxRow = int(np.ceil(yProj.max() / bdy))
    edgesX = np.arange(minCol, maxCol + 1) * bdx
    edgesY = np.arange(minRow, maxRow + 1) * bdy

    xBin = np.clip(np.digitize(xProj, edgesX) - 1, 0, len(edgesX) - 1)
    yBin = np.clip(np.digitize(yProj, edgesY) - 1, 0, len(edgesY) - 1)

    pbMask = np.zeros((len(edgesY), len(edgesX)), dtype=np.uint8)
    pbMask[yBin, xBin] = 1
    # Fill gaps between the first and last marked column of each row:
    for rowNum in range(pbMask.shape[0]):
        cols = np.flatnonzero(pbMask[rowNum])
        if cols.size:
            pbMask[rowNum, cols.min():cols.max() + 1] = 1

    # Column-major enumeration order, as Matlab's find():
    colPBV, rowPBV = np.nonzero(pbMask.T)
    rowPBV = rowPBV.astype(np.intp)
    colPBV = colPBV.astype(np.intp)

    xPBPosV = edgesX[colPBV] + 0.5 * bdx
    yPBPosV = edgesY[rowPBV] + 0.5 * bdy

    gantry2 = np.stack([xPBPosV, yPBPosV,
                        -beam.isodistance * np.ones_like(xPBPosV)], axis=1)
    normG2 = np.linalg.norm(gantry2, axis=1, keepdims=True)
    rtogMC = gantry2RTOGVectors(gantry2, beam.gantryAngle, beam.couchAngle)
    rtog = gantry2RTOGVectors(gantry2 / normG2, beam.gantryAngle,
                              beam.couchAngle)
    return rtog, rtogMC, pbMask, rowPBV, colPBV, xPBPosV, yPBPosV


# --------------------------------------------------------------------------
# CT ray trace (getPBRayData.m)
# --------------------------------------------------------------------------

@dataclass
class CTTrace:
    """Cumulative radiological depth along one pencil-beam ray."""
    distSamplePts: np.ndarray = None   # cm from the source
    densityRay: np.ndarray = None      # g/cm^2 per sample interval
    cumDensityRay: np.ndarray = None   # cumulative g/cm^2


def _rayBoxT(src, dirV, boxMin, boxMax):
    """Slab-method ray/box intersection; returns (t0, t1) along src + t*dir,
    or None if the ray misses the box."""
    t0, t1 = -np.inf, np.inf
    for k in range(3):
        if abs(dirV[k]) < 1e-12:
            if src[k] < boxMin[k] or src[k] > boxMax[k]:
                return None
            continue
        ta = (boxMin[k] - src[k]) / dirV[k]
        tb = (boxMax[k] - src[k]) / dirV[k]
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
    if t1 <= t0:
        return None
    return t0, t1


def getPBRayData(xS, yS, zS, beam, numSamplePts: int, grid: ScanGrid):
    """Pencil-beam geometry plus cumulative-density CT traces for one beam.

    Port of ``getPBRayData.m``.  Returns ``(ctTraceList, RTOGPBVectorsM,
    RTOGPBVectorsM_MC, pbMask, rowPBV, colPBV, xPBPosV, yPBPosV)``.
    """
    (rtog, rtogMC, pbMask, rowPBV, colPBV,
     xPBPosV, yPBPosV) = getPBRays(xS, yS, zS, beam)

    src = np.array(beam.sourcePos(), dtype=np.float64)
    traces: List[CTTrace] = []
    nV = np.arange(numSamplePts)

    for i in range(rtog.shape[0]):
        delta = rtog[i] * RAY_LENGTH
        hit = _rayBoxT(src, delta, grid.boxMin, grid.boxMax)
        if hit is None:
            raise ValueError('PB Ray does not intersect CT scan.')
        tEnt, tExit = hit
        deltaT = (tExit - tEnt) / (numSamplePts - 1)
        tV = tEnt + nV * deltaT
        normD = float(np.linalg.norm(delta))

        sx = src[0] + tV * delta[0]
        sy = src[1] + tV * delta[1]
        sz = src[2] + tV * delta[2]
        ctV = grid.sampleCTNums(sx, sy, sz)

        tr = CTTrace()
        tr.distSamplePts = tV * normD
        tr.densityRay = deltaT * normD * ctV / WATER_CT_NUM
        tr.cumDensityRay = np.cumsum(tr.densityRay)
        traces.append(tr)

    return traces, rtog, rtogMC, pbMask, rowPBV, colPBV, xPBPosV, yPBPosV


def setBeamRayData(im, planC, statusCallback=None):
    """Populate ray-trace fields on every beam of ``im``.

    Port of the ``getPBList`` helper in ``IMRTP.m``: computes the PB
    direction matrix and CT traces, and sets the absolute source position
    (xRel/yRel/zRel + isocenter) on each beam.
    """
    xS, yS, zS, grid = getTargetSurfacePoints(im, planC)
    for i, beam in enumerate(im.beams):
        if statusCallback:
            statusCallback('Getting ray trace for beam %d...' % (i + 1),
                           i / max(len(im.beams), 1))
        (traces, rtog, rtogMC, pbMask, rowPBV, colPBV,
         xPos, yPos) = getPBRayData(xS, yS, zS, beam,
                                    int(im.params.numCTSamplePts), grid)
        beam.CTTraceS = traces
        beam.RTOGPBVectorsM = rtog
        beam.RTOGPBVectorsM_MC = rtogMC
        beam.PBMaskM = pbMask
        beam.rowPBV, beam.colPBV = rowPBV, colPBV
        beam.xPBPosV, beam.yPBPosV = xPos, yPos
        beam.x, beam.y, beam.z = beam.sourcePos()
    return grid
