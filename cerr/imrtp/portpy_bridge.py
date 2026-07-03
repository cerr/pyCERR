"""portpy_bridge
~~~~~~~~~~~~~

Bridge between pyCERR's ``planC`` container and PortPy
(https://github.com/PortPy-Project/PortPy).

PortPy is an *optimization* library: unlike a dose engine, it does not
compute dose.  It consumes a precomputed beamlet dose-influence matrix
``A`` (``d = A x``) plus CT / structure / voxel geometry, and solves a
CVXPy problem driven by clinical criteria.  Its own data come from a
curated Eclipse-extracted dataset ("cannot use your own dataset for
now"); this bridge fills that gap by emitting a PortPy-format patient
folder from any DICOM case pyCERR can load.

Phase 1 (this file, initial version) writes the geometry-only dataset --
CT, structures, optimization voxels and beam geometry -- so that PortPy's
``DataExplorer`` / ``CT`` / ``Structures`` / ``Beams`` load it back
unchanged.  The influence matrix ``A`` is attached in later phases
(from the pyRadPlan bridge or pyCERR's native QIB engine).

On-disk format produced (matching ``portpy.photon.data_explorer``)::

    <outDir>/<patientId>/
        CT_MetaData.json                 resolution/origin + CT_Data.h5 ref
        CT_Data.h5                       ct_hu_3d            (z, y, x)
        StructureSet_MetaData.json       per-structure name/volume + refs
        StructureSet_Data.h5             <name>/structure_mask_3d (z, y, x)
        OptimizationVoxels_MetaData.json ct_voxel_resolution/origin + ref
        OptimizationVoxels_Data.h5       ct_to_dose_voxel_map (z, y, x)
        Beams/Beam_<id>_MetaData.json    angles/energy/iso/beamlets refs
        Beams/Beam_<id>_Data.h5          beamlets/{position_x_mm,...}

Coordinate handling mirrors the pyRadPlan bridge: PortPy indexes voxels
as ``origin + index * resolution`` on an axis-aligned grid, so the CT and
masks are reoriented to LPS (identity direction) before writing.

Requires the optional dependency ``portpy`` (``pip install portpy``).

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence

import numpy as np

try:
    import SimpleITK as sitk
    import h5py
    _PORTPY_IO_AVAILABLE = True
    _PORTPY_IO_IMPORT_ERROR = None
except ImportError as _e:  # pragma: no cover
    _PORTPY_IO_AVAILABLE = False
    _PORTPY_IO_IMPORT_ERROR = _e


def _requireIO():
    if not _PORTPY_IO_AVAILABLE:
        raise ImportError('cerr.imrtp.portpy_bridge requires SimpleITK and '
                          'h5py (pip install portpy). Original error: %s'
                          % _PORTPY_IO_IMPORT_ERROR)


# --------------------------------------------------------------------------
# Geometry helpers (LPS, axis-aligned -- see module docstring)
# --------------------------------------------------------------------------

def _toLPS(img):
    """Reorient a SimpleITK image to axis-aligned LPS (identity direction)."""
    return sitk.DICOMOrient(img, 'LPS')


def _scanArraysLPS(planC, scanNum, gridRefImg=None):
    """Return (hu_zyx, origin_xyz_mm, res_xyz_mm, refImg) for a scan on LPS.

    When ``gridRefImg`` is given, the CT is resampled onto that grid (used
    when the PortPy voxel grid is pyRadPlan's coarser dose grid rather than
    the native CT grid).
    """
    img = _toLPS(sitk.Cast(planC.scan[scanNum].getSitkImage(), sitk.sitkFloat32))
    if gridRefImg is not None:
        img = sitk.Resample(img, gridRefImg, sitk.Transform(), sitk.sitkLinear,
                            -1000.0, sitk.sitkFloat32)
    hu = sitk.GetArrayFromImage(img)                       # (z, y, x)
    origin = list(img.GetOrigin())                         # (x, y, z) mm
    res = list(img.GetSpacing())                           # (x, y, z) mm
    return hu, origin, res, img


def doseGridRefImage(dij):
    """Build a SimpleITK reference image for pyRadPlan's dose grid.

    The dose grid is axis-aligned (identity direction); its voxel order in
    ``(z, y, x)`` (C-order over ``dimensions`` reversed) matches the row
    order of ``dij.physical_dose``.
    """
    _requireIO()
    dg = dij.dose_grid
    nx, ny, nz = (int(d) for d in dg.dimensions)
    ref = sitk.Image(int(nx), int(ny), int(nz), sitk.sitkFloat32)
    ref.SetOrigin([float(o) for o in dg.origin])
    ref.SetSpacing([float(r) for r in dg.resolution_vector])
    ref.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    return ref


def _maskArrayLPS(planC, structNum, refImg):
    """Return a structure's binary mask as (z, y, x) uint8 aligned to refImg."""
    m = sitk.Cast(planC.structure[structNum].getSitkImage(planC), sitk.sitkUInt8)
    m = _toLPS(m)
    # guard against any residual grid mismatch (e.g. differing extents)
    if (m.GetSize() != refImg.GetSize()
            or not np.allclose(m.GetOrigin(), refImg.GetOrigin(), atol=1e-3)):
        m = sitk.Resample(m, refImg, sitk.Transform(), sitk.sitkNearestNeighbor,
                          0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(m).astype(np.uint8)


# --------------------------------------------------------------------------
# Beamlet geometry
# --------------------------------------------------------------------------

def _nominalBeamletGrid(halfWidthMm, widthMm, heightMm):
    """A square open field of beamlets centered on the isocenter axis.

    Returns center positions (x, y) and per-beamlet width/height, all in mm
    in the beam's-eye-view / isocenter plane, matching PortPy's convention
    (``position_x_mm``/``position_y_mm`` are beamlet centers).

    Phase 1 uses a nominal field; later phases replace this with the ray
    geometry that the influence matrix ``A`` was actually computed on.
    """
    xs = np.arange(-halfWidthMm + widthMm / 2.0, halfWidthMm, widthMm)
    ys = np.arange(halfWidthMm - heightMm / 2.0, -halfWidthMm, -heightMm)
    xx, yy = np.meshgrid(xs, ys)
    px = xx.ravel().astype(np.float64)
    py = yy.ravel().astype(np.float64)
    w = np.full(px.shape, float(widthMm))
    h = np.full(px.shape, float(heightMm))
    return px, py, w, h


# --------------------------------------------------------------------------
# Calc box / optimization voxel map
# --------------------------------------------------------------------------

def _ctToDoseVoxelMap(calcMask):
    """Build PortPy's ct_to_dose_voxel_map: running index inside the calc box.

    Voxels inside ``calcMask`` get indices 0..N-1 (row order of ``A``);
    voxels outside are -1.
    """
    idxMap = np.full(calcMask.shape, -1, dtype=np.int64)
    inside = np.flatnonzero(calcMask.ravel())
    idxMap.ravel()[inside] = np.arange(inside.size, dtype=np.int64)
    return idxMap


# --------------------------------------------------------------------------
# Dataset writer
# --------------------------------------------------------------------------

def writePortpyDataset(planC, outDir: str, patientId: str,
                       scanNum: int = 0,
                       structNums: Optional[Sequence[int]] = None,
                       bodyStructNum: Optional[int] = None,
                       gantryAngles: Optional[Sequence[float]] = None,
                       couchAngles: Optional[Sequence[float]] = None,
                       collimatorAngles: Optional[Sequence[float]] = None,
                       isoCenter: Optional[Sequence[float]] = None,
                       energyMV: str = '6X',
                       machineName: str = 'pyCERR',
                       beamletWidthMm: float = 2.5,
                       beamletHeightMm: float = 2.5,
                       fieldHalfWidthMm: float = 60.0,
                       influence: Optional[Dict] = None,
                       gridRefImg=None,
                       rowIndexMap: Optional[np.ndarray] = None,
                       numRows: Optional[int] = None) -> str:
    """Write a PortPy-format patient dataset from ``planC``.

    Args:
        planC: plan container.
        outDir: parent directory; the patient folder is ``outDir/patientId``.
        patientId: PortPy patient id (folder name / DataExplorer patient_id).
        scanNum: CT scan index in ``planC.scan``.
        structNums: structures to export (default: all).  Include a target
            named 'PTV' for PortPy's default workflows.
        bodyStructNum: structure to use as the calc box (optimization
            voxels).  Default: a structure named BODY/External/SKIN if
            present, else the union of ``structNums``.
        gantryAngles / couchAngles / collimatorAngles / isoCenter: beam
            geometry.  Default: read from ``planC.beams[0]`` treatment beams;
            isoCenter falls back to the PTV/target centroid.
        energyMV / machineName: beam labels stored in the metadata.
        beamletWidthMm / beamletHeightMm: beamlet size (multiples of 2.5).
        fieldHalfWidthMm: half-size of the nominal square beamlet field.
        influence: optional ``{beamId: {'sparse': Nx3 array, 'tol': float,
            'beamlet_id': array, 'position_x_mm': .., 'position_y_mm': ..,
            'width_mm': .., 'height_mm': ..}}`` attaching the dose-influence
            matrix (later phases).  When None, a geometry-only dataset with
            nominal beamlets and no ``A`` is written.

    Returns:
        str: path to the patient folder (``outDir/patientId``).
    """
    _requireIO()
    if structNums is None:
        structNums = list(range(len(planC.structure)))
    structNums = list(structNums)

    patDir = os.path.join(outDir, patientId)
    beamsDir = os.path.join(patDir, 'Beams')
    os.makedirs(beamsDir, exist_ok=True)

    # --- CT ---
    hu, origin, res, refImg = _scanArraysLPS(planC, scanNum, gridRefImg=gridRefImg)
    with h5py.File(os.path.join(patDir, 'CT_Data.h5'), 'w') as f:
        f.create_dataset('ct_hu_3d', data=hu.astype(np.float32))
    _writeJson(os.path.join(patDir, 'CT_MetaData.json'), {
        'resolution_xyz_mm': [float(r) for r in res],
        'origin_xyz_mm': [float(o) for o in origin],
        'size_xyz': [int(s) for s in refImg.GetSize()],
        'ct_hu_3d_File': 'CT_Data.h5/ct_hu_3d',
    })

    # --- Structures ---
    masks = {s: _maskArrayLPS(planC, s, refImg) for s in structNums}
    names = {s: str(planC.structure[s].structureName) for s in structNums}
    voxVolCc = float(np.prod(res) / 1000.0)

    # calc box (optimization voxels)
    if bodyStructNum is None:
        bodyStructNum = _guessBodyStruct(planC, structNums)
    if bodyStructNum is not None:
        calcMask = masks[bodyStructNum].astype(bool)
    else:
        calcMask = np.zeros(hu.shape, dtype=bool)
        for s in structNums:
            calcMask |= masks[s].astype(bool)
    if rowIndexMap is not None:
        # rows are defined by an external influence matrix (e.g. pyRadPlan):
        # keep those row indices, mask out voxels outside the calc box
        idxMap = np.where(calcMask, rowIndexMap.astype(np.int64), -1)
    else:
        idxMap = _ctToDoseVoxelMap(calcMask)

    with h5py.File(os.path.join(patDir, 'StructureSet_Data.h5'), 'w') as f:
        for s in structNums:
            f.create_dataset('%s/structure_mask_3d' % names[s],
                             data=masks[s].astype(np.uint8))
    structMeta = []
    doseBox = idxMap >= 0
    for s in structNums:
        m = masks[s].astype(bool)
        volCc = float(m.sum()) * voxVolCc
        inBox = float((m & doseBox).sum()) * voxVolCc
        structMeta.append({
            'name': names[s],
            'volume_cc': volCc,
            'fraction_of_vol_in_calc_box': (inBox / volCc) if volCc > 0 else 0.0,
            'structure_mask_3d_File':
                'StructureSet_Data.h5/%s/structure_mask_3d' % names[s],
        })
    _writeJson(os.path.join(patDir, 'StructureSet_MetaData.json'), structMeta)

    # --- Optimization voxels ---
    with h5py.File(os.path.join(patDir, 'OptimizationVoxels_Data.h5'), 'w') as f:
        f.create_dataset('ct_to_dose_voxel_map', data=idxMap)
    _writeJson(os.path.join(patDir, 'OptimizationVoxels_MetaData.json'), {
        'ct_voxel_resolution_xyz_mm': [float(r) for r in res],
        'dose_voxel_resolution_xyz_mm': [float(r) for r in res],
        'ct_origin_xyz_mm': [float(o) for o in origin],
        'ct_to_dose_voxel_map_File':
            'OptimizationVoxels_Data.h5/ct_to_dose_voxel_map',
    })

    # --- Beams ---
    geom = _beamGeometry(planC, structNums, masks, refImg,
                         gantryAngles, couchAngles, collimatorAngles, isoCenter)
    ids = []
    for i, (g, c, coll) in enumerate(zip(geom['gantry'], geom['couch'],
                                         geom['collimator'])):
        beamId = i
        ids.append(beamId)
        inf = (influence or {}).get(beamId)
        if inf is not None and 'position_x_mm' in inf:
            px, py = np.asarray(inf['position_x_mm']), np.asarray(inf['position_y_mm'])
            w, h = np.asarray(inf['width_mm']), np.asarray(inf['height_mm'])
        else:
            px, py, w, h = _nominalBeamletGrid(fieldHalfWidthMm,
                                               beamletWidthMm, beamletHeightMm)
        _writeBeam(beamsDir, beamId, g, c, coll, geom['iso'], energyMV,
                   machineName, px, py, w, h, inf)

    _writeJson(os.path.join(patDir, 'PlannerBeams.json'), {'IDs': ids})
    return patDir


def influenceFromPyRadPlan(dij, stf):
    """Split a pyRadPlan ``dij`` into PortPy per-beam influence + geometry.

    Returns:
        tuple ``(influence, rowIndexMap, numRows)`` where ``influence`` is
        the ``{beamId: {...}}`` dict consumed by :func:`writePortpyDataset`
        (per-beam sparse ``Nx3`` (row, local-beamlet, dose), beamlet BEV
        positions and widths), ``rowIndexMap`` is the ``(z, y, x)`` array of
        A-row indices for the dose grid, and ``numRows = A.shape[0]``.
    """
    _requireIO()
    from scipy import sparse as _sp
    A = dij.physical_dose
    A = A.flat[0] if isinstance(A, np.ndarray) else A
    A = _sp.csc_matrix(A)
    numRows = int(A.shape[0])

    nx, ny, nz = (int(d) for d in dij.dose_grid.dimensions)
    rowIndexMap = np.arange(numRows, dtype=np.int64).reshape((nz, ny, nx))

    beamNum = np.asarray(dij.beam_num).ravel().astype(int)
    rayNum = np.asarray(dij.ray_num).ravel().astype(int)

    influence = {}
    for b, beam in enumerate(stf.beams):
        cols = np.flatnonzero(beamNum == b)
        rays = np.atleast_1d(beam.rays)
        # beamlet BEV in-plane position (x, z) of ray_pos_bev, one per column
        px, py, w, h = [], [], [], []
        sub = A[:, cols].tocoo()
        for k, col in enumerate(cols):
            rp = np.asarray(rays[rayNum[col]].ray_pos_bev, dtype=np.float64).ravel()
            px.append(rp[0])
            py.append(rp[2])
            w.append(float(beam.bixel_width))
            h.append(float(beam.bixel_width))
        # sub.col is already local 0..len(cols)-1 because A[:, cols] reindexes
        triplet = np.column_stack([sub.row.astype(np.float64),
                                   sub.col.astype(np.float64),
                                   sub.data.astype(np.float64)])
        # PortPy rebuilds the matrix via csr_matrix((val,(row,col))) and infers
        # the shape from the max indices; pad an explicit zero at the last
        # (voxel, beamlet) so every beam keeps full (numRows x nCols) shape.
        nCols = len(cols)
        sentinel = np.array([[numRows - 1, max(nCols - 1, 0), 0.0]])
        triplet = np.vstack([triplet, sentinel]) if triplet.size else sentinel
        influence[b] = {
            'sparse': triplet,
            'tol': 0.0,
            'position_x_mm': np.asarray(px),
            'position_y_mm': np.asarray(py),
            'width_mm': np.asarray(w),
            'height_mm': np.asarray(h),
        }
    return influence, rowIndexMap, numRows


def writePortpyFromPyRadPlan(planC, dij, stf, outDir: str, patientId: str,
                             scanNum: int = 0,
                             structNums: Optional[Sequence[int]] = None,
                             bodyStructNum: Optional[int] = None,
                             gantryAngles: Optional[Sequence[float]] = None,
                             couchAngles: Optional[Sequence[float]] = None,
                             isoCenter: Optional[Sequence[float]] = None,
                             energyMV: str = '6X',
                             machineName: str = 'pyCERR') -> str:
    """Write a PortPy dataset with the influence matrix from a pyRadPlan run.

    ``dij``/``stf`` come from ``cerr.imrtp.pyradplan_bridge.calcDoseInfluence``.
    The PortPy voxel grid is pyRadPlan's dose grid so the influence-matrix
    rows line up with the optimization voxels.
    """
    _requireIO()
    influence, rowIndexMap, numRows = influenceFromPyRadPlan(dij, stf)
    refImg = doseGridRefImage(dij)
    gantry = list(gantryAngles) if gantryAngles is not None \
        else [float(b.gantry_angle) for b in stf.beams]
    couch = list(couchAngles) if couchAngles is not None \
        else [float(b.couch_angle) for b in stf.beams]
    if isoCenter is None:
        isoCenter = np.asarray(stf.beams[0].iso_center, dtype=np.float64).ravel()
    return writePortpyDataset(
        planC, outDir=outDir, patientId=patientId, scanNum=scanNum,
        structNums=structNums, bodyStructNum=bodyStructNum,
        gantryAngles=gantry, couchAngles=couch, isoCenter=isoCenter,
        energyMV=energyMV, machineName=machineName,
        influence=influence, gridRefImg=refImg, rowIndexMap=rowIndexMap,
        numRows=numRows)


def influenceFromQIB(im, planC, scanNum: int = 0):
    """Split pyCERR's native QIB beamlets (``im``) into PortPy influence.

    ``im`` is an :class:`cerr.imrtp.imrtp_problem.IMRTProblem` whose beamlets
    have been filled by ``generateQIBInfluence``.  QIB stores, per beam, a
    structure-major list of pencil-beam beamlets; each carries scan-grid
    ``(row, col, slice)`` C-order voxel indices and float dose values.

    Returns:
        tuple ``(influence, rowIndexMap, numRows)`` -- see
        :func:`influenceFromPyRadPlan`.  ``rowIndexMap`` is on the scan (CT)
        grid in PortPy ``(z, y, x)`` order (reoriented to LPS, matching the
        structure masks written by :func:`writePortpyDataset`).
    """
    _requireIO()
    from cerr.dataclasses.scan import flipSliceOrderFlag
    scanObj = planC.scan[scanNum]
    nR, nC, nS = (int(v) for v in scanObj.getScanSize())

    allInds = [np.asarray(bl.indexV, dtype=np.int64)
               for beam in im.beams for bl in beam.beamlets
               if bl.indexV is not None and np.size(bl.indexV)]
    if not allInds:
        raise ValueError('No QIB beamlets found; run generateQIBInfluence '
                         'on the IM problem first.')
    optScan = np.unique(np.concatenate(allInds))
    numRows = int(optScan.size)
    rowOf = np.full(nR * nC * nS, -1, dtype=np.int64)
    rowOf[optScan] = np.arange(numRows, dtype=np.int64)

    influence = {}
    for b, beam in enumerate(im.beams):
        numPBs = int(beam.RTOGPBVectorsM.shape[0]) if beam.beamlets else 0
        if numPBs == 0:
            continue
        numStructs = len(beam.beamlets) // numPBs
        rows, cols, vals = [], [], []
        for c in range(numPBs):
            sis, dvs = [], []
            for iStr in range(numStructs):
                bl = beam.beamlets[iStr * numPBs + c]
                if bl.indexV is not None and np.size(bl.indexV):
                    sis.append(np.asarray(bl.indexV, dtype=np.int64))
                    dvs.append(np.asarray(bl.influence, dtype=np.float64))
            if not sis:
                continue
            # a voxel shared by overlapping structures gets the same pencil-beam
            # dose in each; keep it once (avoid double counting on csr assembly)
            usi, idx = np.unique(np.concatenate(sis), return_index=True)
            rows.append(rowOf[usi])
            cols.append(np.full(usi.size, c))
            vals.append(np.concatenate(dvs)[idx])
        if rows:
            triplet = np.column_stack([np.concatenate(rows).astype(np.float64),
                                       np.concatenate(cols).astype(np.float64),
                                       np.concatenate(vals)])
        else:
            triplet = np.zeros((0, 3))
        sentinel = np.array([[numRows - 1, max(numPBs - 1, 0), 0.0]])
        triplet = np.vstack([triplet, sentinel]) if triplet.size else sentinel
        influence[b] = {
            'sparse': triplet,
            'tol': 0.0,
            'position_x_mm': np.asarray(beam.xPBPosV, dtype=np.float64) * 10.0,
            'position_y_mm': np.asarray(beam.yPBPosV, dtype=np.float64) * 10.0,
            'width_mm': np.full(numPBs, float(beam.beamletDelta_x) * 10.0),
            'height_mm': np.full(numPBs, float(beam.beamletDelta_y) * 10.0),
        }

    # scan-grid (row, col, slice) row-index map -> PortPy (z, y, x) via the
    # same getSitkImage + LPS pipeline used for the structure masks
    rowMapRCS = np.full((nR, nC, nS), -1, dtype=np.int32)
    rowMapRCS.reshape(-1)[optScan] = np.arange(numRows, dtype=np.int32)
    sitkArr = np.transpose(rowMapRCS, (2, 0, 1))          # (slice, row, col)
    if flipSliceOrderFlag(scanObj):
        sitkArr = np.flip(sitkArr, axis=0)
    labelImg = sitk.GetImageFromArray(np.ascontiguousarray(sitkArr))
    labelImg.CopyInformation(scanObj.getSitkImage())
    rowIndexMap = sitk.GetArrayFromImage(_toLPS(labelImg)).astype(np.int64)
    return influence, rowIndexMap, numRows


def writePortpyFromQIB(planC, im, outDir: str, patientId: str,
                       scanNum: int = 0,
                       structNums: Optional[Sequence[int]] = None,
                       bodyStructNum: Optional[int] = None,
                       energyMV: str = '6X',
                       machineName: str = 'pyCERR') -> str:
    """Write a PortPy dataset using pyCERR's native QIB influence matrix.

    ``im`` must have QIB beamlets computed (``generateQIBInfluence``).  This
    path needs no pyRadPlan.
    """
    _requireIO()
    influence, rowIndexMap, numRows = influenceFromQIB(im, planC, scanNum)
    gantry = [float(b.gantryAngle) for b in im.beams if b.beamlets]
    couch = [float(b.couchAngle) for b in im.beams if b.beamlets]
    return writePortpyDataset(
        planC, outDir=outDir, patientId=patientId, scanNum=scanNum,
        structNums=structNums, bodyStructNum=bodyStructNum,
        gantryAngles=gantry, couchAngles=couch,
        energyMV=energyMV, machineName=machineName,
        influence=influence, rowIndexMap=rowIndexMap, numRows=numRows)


def _guessBodyStruct(planC, structNums):
    for s in structNums:
        nm = str(planC.structure[s].structureName).upper().replace('_', '').replace(' ', '')
        if nm in ('BODY', 'EXTERNAL', 'SKIN', 'PATIENT', 'OUTERCONTOUR'):
            return s
    return None


def _beamGeometry(planC, structNums, masks, refImg, gantryAngles, couchAngles,
                  collimatorAngles, isoCenter):
    """Resolve beam angles + isocenter (DICOM/LPS mm) for the dataset."""
    if gantryAngles is None:
        gantry, couch, coll = [], [], []
        for bs in np.atleast_1d(planC.beams[0].BeamSequence):
            if str(getattr(bs, 'TreatmentDeliveryType', '') or '').upper() \
                    not in ('', 'TREATMENT'):
                continue
            cps = np.atleast_1d(bs.ControlPointSequence)
            if cps.size == 0:
                continue
            cp0 = cps[0]
            gantry.append(float(getattr(cp0, 'GantryAngle', 0.0) or 0.0))
            couch.append(float(getattr(cp0, 'PatientSupportAngle', 0.0) or 0.0))
            coll.append(float(getattr(cp0, 'BeamLimitingDeviceAngle', 0.0) or 0.0))
        if not gantry:
            raise ValueError('No treatment beams found in planC.beams[0].')
    else:
        gantry = [float(g) for g in gantryAngles]
        couch = [float(c) for c in (couchAngles or [0.0] * len(gantry))]
        coll = [float(c) for c in (collimatorAngles or [0.0] * len(gantry))]

    if isoCenter is None:
        iso = _targetCentroidLPS(planC, structNums, masks, refImg)
    else:
        iso = np.asarray(isoCenter, dtype=np.float64).ravel()
    return {'gantry': gantry, 'couch': couch, 'collimator': coll,
            'iso': [float(iso[0]), float(iso[1]), float(iso[2])]}


def _targetCentroidLPS(planC, structNums, masks, refImg):
    """Centroid (LPS mm) of a target structure, for a fallback isocenter."""
    target = None
    for s in structNums:
        if 'PTV' in str(planC.structure[s].structureName).upper():
            target = s
            break
    if target is None:
        target = structNums[0]
    m = masks[target].astype(bool)
    zyx = np.array(np.nonzero(m)).mean(axis=1)          # (z, y, x) index
    # index -> physical LPS mm via the reference image
    return np.asarray(refImg.TransformContinuousIndexToPhysicalPoint(
        [float(zyx[2]), float(zyx[1]), float(zyx[0])]), dtype=np.float64)


def _writeBeam(beamsDir, beamId, gantry, couch, coll, iso, energyMV,
               machineName, px, py, w, h, inf):
    dataName = 'Beam_%s_Data.h5' % beamId
    beamletId = np.arange(px.size, dtype=np.int64)
    with h5py.File(os.path.join(beamsDir, dataName), 'w') as f:
        f.create_dataset('beamlets/id', data=beamletId)
        f.create_dataset('beamlets/position_x_mm', data=px.astype(np.float64))
        f.create_dataset('beamlets/position_y_mm', data=py.astype(np.float64))
        f.create_dataset('beamlets/width_mm', data=w.astype(np.float64))
        f.create_dataset('beamlets/height_mm', data=h.astype(np.float64))
        if inf is not None and 'sparse' in inf:
            f.create_dataset('inf_matrix_sparse',
                             data=np.asarray(inf['sparse'], dtype=np.float64))
    meta = {
        'ID': beamId,
        'gantry_angle': float(gantry),
        'collimator_angle': float(coll),
        'couch_angle': float(couch),
        'beam_modality': 'photon',
        'energy_MV': energyMV,
        'iso_center': {'x_mm': iso[0], 'y_mm': iso[1], 'z_mm': iso[2]},
        'MLC_name': machineName,
        'machine_name': machineName,
        'beamlets': {
            'id_File': '%s/beamlets/id' % dataName,
            'position_x_mm_File': '%s/beamlets/position_x_mm' % dataName,
            'position_y_mm_File': '%s/beamlets/position_y_mm' % dataName,
            'width_mm_File': '%s/beamlets/width_mm' % dataName,
            'height_mm_File': '%s/beamlets/height_mm' % dataName,
        },
    }
    if inf is not None and 'sparse' in inf:
        meta['influenceMatrixSparse_File'] = '%s/inf_matrix_sparse' % dataName
        meta['influenceMatrixSparse_tol'] = float(inf.get('tol', 0.0))
    _writeJson(os.path.join(beamsDir, 'Beam_%s_MetaData.json' % beamId), meta)


def _writeJson(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


# --------------------------------------------------------------------------
# Phase 4: clinical-criteria mapping + PortPy optimization
# --------------------------------------------------------------------------

def buildOptParams(targetName: str, prescriptionGy: float,
                   oarObjectives: Optional[Dict[str, float]] = None,
                   targetWeight: float = 100000.0,
                   oarWeight: float = 10000.0,
                   smoothnessWeight: float = 10.0) -> dict:
    """Build a PortPy ``opt_params`` dict from a simple prescription.

    The target gets quadratic under- and over-dose objectives at the
    prescription; each OAR in ``oarObjectives`` (``{name: dose_gy}``) gets a
    quadratic-overdose objective at its limit (``dose_gy=0`` for pure mean
    sparing).  A smoothness term regularizes the fluence.

    Args:
        targetName: target structure name (e.g. 'PTV').
        prescriptionGy: total prescription dose (Gy).
        oarObjectives: ``{oar_name: overdose_limit_gy}``.
        targetWeight / oarWeight / smoothnessWeight: objective weights.

    Returns:
        dict with an ``objective_functions`` list (and empty ``constraints``).
    """
    objs = [
        {'type': 'quadratic-underdose', 'structure_name': targetName,
         'dose_gy': float(prescriptionGy), 'weight': float(targetWeight)},
        {'type': 'quadratic-overdose', 'structure_name': targetName,
         'dose_gy': float(prescriptionGy), 'weight': float(targetWeight)},
    ]
    for name, limitGy in (oarObjectives or {}).items():
        objs.append({'type': 'quadratic-overdose', 'structure_name': name,
                     'dose_gy': float(limitGy), 'weight': float(oarWeight)})
    if smoothnessWeight:
        objs.append({'type': 'smoothness-quadratic',
                     'weight': float(smoothnessWeight)})
    return {'objective_functions': objs, 'constraints': []}


def buildClinicalCriteria(dataDir: str, patientId: str, prescriptionGy: float,
                          numFractions: int, diseaseSite: str = 'pyCERR'):
    """Construct a PortPy ``ClinicalCriteria`` for a prescription.

    Writes a minimal criteria JSON into the patient folder and loads it, so
    no bundled protocol is required.
    """
    import portpy.photon as pp
    cc = {
        'disease_site': diseaseSite,
        'protocol_name': 'pyCERR_%gGy_%dFx' % (prescriptionGy, numFractions),
        'pres_per_fraction_gy': float(prescriptionGy) / int(numFractions),
        'num_of_fractions': int(numFractions),
        'criteria': [],
    }
    path = os.path.join(dataDir, patientId, 'ClinicalCriteria_MetaData.json')
    _writeJson(path, cc)
    return pp.ClinicalCriteria(file_name=path)


def _importPortpyDose(planC, dose3dZYX, structs, scanNum, fractionGroupID):
    """Import a PortPy-grid dose (z, y, x) back into ``planC.dose``.

    Builds a SimpleITK image from the optimization-voxel grid geometry,
    resamples it onto the scan grid, and stores it via ``importDoseArray``.
    """
    from cerr import plan_container as pc
    from cerr.dataclasses.scan import flipSliceOrderFlag
    res = [float(v) for v in structs.opt_voxels_dict['ct_voxel_resolution_xyz_mm']]
    origin = [float(v) for v in structs.opt_voxels_dict['ct_origin_xyz_mm']]
    img = sitk.GetImageFromArray(np.ascontiguousarray(dose3dZYX.astype(np.float64)))
    img.SetSpacing(res)
    img.SetOrigin(origin)
    img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])

    ref = planC.scan[scanNum].getSitkImage()
    doseImg = sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, 0.0,
                            sitk.sitkFloat64)
    dose3M = sitk.GetArrayFromImage(doseImg)               # (z, y, x)
    if flipSliceOrderFlag(planC.scan[scanNum]):
        dose3M = np.flip(dose3M, axis=0)
    dose3M = np.transpose(dose3M, (1, 2, 0))               # (row, col, slice)
    xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
    pc.importDoseArray(dose3M, xV, yV, zV, planC, scanNum,
                       {'fractionGroupID': fractionGroupID, 'units': 'GRAYS'})
    return len(planC.dose) - 1


def optimizeAndImport(planC, dataDir: str, patientId: str,
                      prescriptionGy: float, numFractions: int = 1,
                      scanNum: int = 0, targetName: str = 'PTV',
                      oarObjectives: Optional[Dict[str, float]] = None,
                      solver: str = 'SCS', isBev: bool = True,
                      beamletWidthMm: Optional[float] = None,
                      beamletHeightMm: Optional[float] = None,
                      fractionGroupID: str = 'portpy'):
    """Load a PortPy dataset, optimize with PortPy, import dose into planC.

    The dataset must already be written (e.g. by
    :func:`writePortpyFromPyRadPlan` or :func:`writePortpyFromQIB`).

    Returns:
        tuple ``(sol, doseNum, planC)`` -- PortPy solution dict, index of the
        imported dose in ``planC.dose``, and the container.
    """
    _requireIO()
    import portpy.photon as pp

    data = pp.DataExplorer(data_dir=dataDir, patient_id=patientId)
    ct = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)
    infKw = {}
    if beamletWidthMm is not None:
        infKw = {'beamlet_width_mm': beamletWidthMm,
                 'beamlet_height_mm': beamletHeightMm}
    inf = pp.InfluenceMatrix(structs, beams, ct, target_structure=targetName,
                             is_bev=isBev, **infKw)

    cc = buildClinicalCriteria(dataDir, patientId, prescriptionGy, numFractions)
    plan = pp.Plan(structs, beams, inf, ct, cc)
    opt_params = buildOptParams(targetName, prescriptionGy, oarObjectives)
    opt = pp.Optimization(plan, opt_params=opt_params, clinical_criteria=cc)
    opt.create_cvxpy_problem()
    sol = opt.solve(solver=solver, verbose=False)

    # A @ x is per-fraction dose; scale to total prescription dose
    dose_1d = np.asarray(inf.A @ sol['optimal_intensity']) * numFractions
    dose_3d = inf.dose_1d_to_3d(dose_1d=dose_1d)
    doseNum = _importPortpyDose(planC, dose_3d, structs, scanNum,
                                fractionGroupID)
    return sol, doseNum, planC
