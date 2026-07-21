"""lattice module.

Tools for creating lattice (spatially fractionated) radiotherapy targets.

Lattice radiotherapy (LRT) and 3-D GRID therapy deliver a spatially
fractionated dose by boosting a regular arrangement of small, high-dose
"vertices" (spheres) embedded inside a gross tumor volume (GTV), while the
tissue between the vertices ("valleys") receives a lower dose. This module
generates the vertex lattice from an existing target structure and adds the
resulting peak/valley segmentations to a :class:`cerr.plan_container.PlanC`.

Typical use::

    import cerr.imrtp.lattice as lat
    planC, verticesXYZ = lat.createLattice(
        planC, structNum,               # index of the GTV structure
        sphereDiameter=15.0,            # mm
        latticeSpacing=30.0,            # mm, vertex centre-to-centre
        latticeType='bcc',             # 'sc' (simple cubic) or 'bcc'
        innerMargin=8.0,               # mm inset of vertices from GTV surface
        addValley=True,
    )

The geometry follows common LRT planning conventions (see e.g. Wu et al.,
"The technical and clinical implementation of LATTICE radiation therapy",
Radiat. Res. 2020): 1-1.5 cm diameter spheres on a 2-5 cm lattice, inset from
the target surface so that no sphere protrudes past the GTV.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

import cerr.plan_container as pc
import cerr.dataclasses.scan as scn
import cerr.contour.rasterseg as rs


def _getVoxelSpacing(xV, yV, zV):
    """Return the (dx, dy, dz) voxel spacing in mm from grid coordinate vectors."""
    dx = float(np.abs(np.median(np.diff(xV)))) if len(xV) > 1 else 1.0
    dy = float(np.abs(np.median(np.diff(yV)))) if len(yV) > 1 else 1.0
    dz = float(np.abs(np.median(np.diff(zV)))) if len(zV) > 1 else 1.0
    return dx, dy, dz


def createLatticeVertices(targetMask3M, xV, yV, zV,
                          latticeSpacing=30.0,
                          latticeType='bcc',
                          innerMargin=8.0,
                          offset=None,
                          requireSphereInside=False,
                          sphereRadius=0.0):
    """Generate lattice vertex centres inside a target mask.

    Candidate vertices are placed on an axis-aligned lattice spanning the
    bounding box of ``targetMask3M`` and retained only where the vertex centre
    lies at least ``innerMargin`` inside the target surface. All distance
    arguments are in the same units as ``xV``/``yV``/``zV`` (pyCERR cm coords).

    Args:
        targetMask3M (np.ndarray): Boolean mask (nRows, nCols, nSlices) of the target.
        xV (np.ndarray): x (column) grid coordinates (ascending).
        yV (np.ndarray): y (row) grid coordinates (descending, pyCERR convention).
        zV (np.ndarray): z (slice) grid coordinates.
        latticeSpacing (float): Vertex centre-to-centre spacing along each axis.
        latticeType (str): 'sc' for simple cubic or 'bcc' for body-centred cubic.
        innerMargin (float): Minimum distance of a vertex centre from the
            target surface. Prevents vertices being placed on the periphery.
        offset (np.ndarray or None): Optional length-3 (x, y, z) shift applied
            to the lattice. When None the lattice is centred on the target centroid.
        requireSphereInside (bool): When True, only keep vertices whose full
            sphere (radius ``sphereRadius``) fits inside the eroded region, i.e.
            the vertex centre is at least ``sphereRadius`` from the eroded
            boundary as well. Uses max(innerMargin, sphereRadius) effectively.
        sphereRadius (float): Sphere radius, used only when
            ``requireSphereInside`` is True.

    Returns:
        np.ndarray: Array of shape (N, 3) of retained vertex centres in (x, y, z).
    """
    latticeType = latticeType.lower()
    if latticeType not in ('sc', 'bcc'):
        raise ValueError("latticeType must be 'sc' or 'bcc', got %r" % latticeType)
    if latticeSpacing <= 0:
        raise ValueError("latticeSpacing must be positive")

    if not np.any(targetMask3M):
        return np.zeros((0, 3))

    dx, dy, dz = _getVoxelSpacing(xV, yV, zV)

    # Distance (mm) from every target voxel to the nearest background voxel.
    # sampling is ordered to match array axes: (rows->dy, cols->dx, slices->dz).
    distMm = distance_transform_edt(targetMask3M, sampling=(dy, dx, dz))

    effMargin = innerMargin
    if requireSphereInside:
        effMargin = max(innerMargin, sphereRadius)

    # Physical coordinate of every voxel centre (broadcast grids are cheap in 1-D).
    # We test candidate lattice points against the allowed region by nearest voxel.
    allowed3M = distMm >= effMargin
    if not np.any(allowed3M):
        return np.zeros((0, 3))

    # Target centroid in physical coordinates (for default centring).
    rows, cols, slcs = np.where(targetMask3M)
    cx = float(np.mean(xV[cols]))
    cy = float(np.mean(yV[rows]))
    cz = float(np.mean(zV[slcs]))
    if offset is None:
        centre = np.array([cx, cy, cz])
    else:
        centre = np.array([cx, cy, cz]) + np.asarray(offset, dtype=float)

    # Build lattice coordinate grid spanning the target bounding box, centred
    # on `centre` so that a lattice node coincides with the centroid.
    xLo, xHi = xV[cols].min(), xV[cols].max()
    yLo, yHi = yV[rows].min(), yV[rows].max()
    zLo, zHi = zV[slcs].min(), zV[slcs].max()

    def _axisNodes(cen, lo, hi):
        nLo = int(np.ceil((lo - cen) / latticeSpacing))
        nHi = int(np.floor((hi - cen) / latticeSpacing))
        return cen + np.arange(nLo, nHi + 1) * latticeSpacing

    xNodes = _axisNodes(centre[0], xLo, xHi)
    yNodes = _axisNodes(centre[1], yLo, yHi)
    zNodes = _axisNodes(centre[2], zLo, zHi)

    # Corner (simple-cubic) lattice points.
    candidates = []
    XX, YY, ZZ = np.meshgrid(xNodes, yNodes, zNodes, indexing='ij')
    candidates.append(np.column_stack((XX.ravel(), YY.ravel(), ZZ.ravel())))

    # Body-centred points: shift by half a cell in every axis.
    if latticeType == 'bcc':
        h = latticeSpacing / 2.0
        XXb, YYb, ZZb = np.meshgrid(xNodes + h, yNodes + h, zNodes + h, indexing='ij')
        candidates.append(np.column_stack((XXb.ravel(), YYb.ravel(), ZZb.ravel())))

    cand = np.vstack(candidates)

    # Keep candidates whose nearest voxel is in the allowed (eroded) region.
    kept = []
    for (px, py, pz) in cand:
        ci = int(np.argmin(np.abs(xV - px)))
        ri = int(np.argmin(np.abs(yV - py)))
        si = int(np.argmin(np.abs(zV - pz)))
        if allowed3M[ri, ci, si]:
            kept.append((px, py, pz))

    return np.asarray(kept) if kept else np.zeros((0, 3))


def paintSpheres(verticesXYZ, sphereRadius, xV, yV, zV, maskShape):
    """Rasterize spheres of a given radius at each vertex into a binary mask.

    Args:
        verticesXYZ (np.ndarray): (N, 3) vertex centres in (x, y, z) mm.
        sphereRadius (float): Sphere radius in the same units as xV/yV/zV.
        xV, yV, zV (np.ndarray): Grid coordinate vectors (pyCERR cm coords).
        maskShape (tuple): (nRows, nCols, nSlices) of the output mask.

    Returns:
        np.ndarray: Boolean mask with the union of all spheres set to True.
    """
    mask3M = np.zeros(maskShape, dtype=bool)
    r = float(sphereRadius)
    if r <= 0 or len(verticesXYZ) == 0:
        return mask3M

    for (xc, yc, zc) in verticesXYZ:
        colIdx = np.where(np.abs(xV - xc) <= r)[0]
        rowIdx = np.where(np.abs(yV - yc) <= r)[0]
        slcIdx = np.where(np.abs(zV - zc) <= r)[0]
        if len(colIdx) == 0 or len(rowIdx) == 0 or len(slcIdx) == 0:
            continue
        rr, cc, ss = np.meshgrid(rowIdx, colIdx, slcIdx, indexing='ij')
        dist2 = (xV[cc] - xc) ** 2 + (yV[rr] - yc) ** 2 + (zV[ss] - zc) ** 2
        inside = dist2 <= r ** 2
        mask3M[rr[inside], cc[inside], ss[inside]] = True

    return mask3M


def createLattice(planC, structNum,
                  sphereDiameter=15.0,
                  latticeSpacing=30.0,
                  latticeType='bcc',
                  innerMargin=8.0,
                  offset=None,
                  requireSphereInside=True,
                  addValley=True,
                  addIndividualSpheres=False,
                  vertexStructName=None,
                  valleyStructName=None,
                  units='mm'):
    """Create a lattice-radiotherapy vertex target from an existing structure.

    Generates a regular arrangement of high-dose spheres ("vertices"/"peaks")
    inside the target structure ``structNum`` and appends the result to
    ``planC.structure``. Optionally also creates the complementary "valley"
    structure (target minus vertices) and one structure per sphere.

    Args:
        planC (cerr.plan_container.PlanC): pyCERR plan container.
        structNum (int): Index of the target (GTV) structure in ``planC.structure``.
        sphereDiameter (float): Vertex sphere diameter (typically 10-15 mm).
        latticeSpacing (float): Vertex centre-to-centre spacing (typically 20-50 mm).
        latticeType (str): 'sc' (simple cubic) or 'bcc' (body-centred cubic).
        innerMargin (float): Minimum distance of a vertex centre from the target surface.
        offset (np.ndarray or None): Optional (x, y, z) shift of the lattice;
            None centres it on the target centroid.
        requireSphereInside (bool): When True (default) only keep vertices whose
            full sphere fits within the target inset by ``innerMargin``.
        addValley (bool): When True add a "valley" structure = target minus vertices.
        addIndividualSpheres (bool): When True add each sphere as its own structure.
        vertexStructName (str or None): Name for the combined vertex structure.
            Defaults to ``"<target>_lattice_peaks"``.
        valleyStructName (str or None): Name for the valley structure.
            Defaults to ``"<target>_lattice_valley"``.
        units (str): Units for ``sphereDiameter``, ``latticeSpacing``,
            ``innerMargin`` and ``offset``. 'mm' (default, clinical convention)
            or 'cm' (pyCERR's native grid units). Values are converted to the
            grid's centimetre coordinate system internally.

    Returns:
        tuple:
            - cerr.plan_container.PlanC: updated plan container.
            - np.ndarray: (N, 3) retained vertex centres in (x, y, z), pyCERR cm coords.
    """
    if structNum < 0 or structNum >= len(planC.structure):
        raise IndexError("structNum %d out of range" % structNum)

    units = units.lower()
    if units not in ('mm', 'cm'):
        raise ValueError("units must be 'mm' or 'cm', got %r" % units)
    # pyCERR's virtual coordinate grid is in centimetres; convert user inputs.
    scale = 0.1 if units == 'mm' else 1.0
    latticeSpacing_cm = latticeSpacing * scale
    innerMargin_cm = innerMargin * scale
    sphereRadius_cm = (sphereDiameter / 2.0) * scale
    offset_cm = None if offset is None else np.asarray(offset, dtype=float) * scale

    assocScanNum = scn.getScanNumFromUID(planC.structure[structNum].assocScanUID, planC)
    targetName = planC.structure[structNum].structureName
    xV, yV, zV = planC.scan[assocScanNum].getScanXYZVals()

    targetMask3M = rs.getStrMask(structNum, planC)
    if not np.any(targetMask3M):
        raise ValueError("Target structure '%s' has an empty mask" % targetName)

    verticesXYZ = createLatticeVertices(
        targetMask3M, xV, yV, zV,
        latticeSpacing=latticeSpacing_cm,
        latticeType=latticeType,
        innerMargin=innerMargin_cm,
        offset=offset_cm,
        requireSphereInside=requireSphereInside,
        sphereRadius=sphereRadius_cm,
    )

    if len(verticesXYZ) == 0:
        raise ValueError(
            "No lattice vertices fit inside '%s' with spacing=%g %s, "
            "innerMargin=%g %s, sphereDiameter=%g %s. Try smaller values."
            % (targetName, latticeSpacing, units, innerMargin, units,
               sphereDiameter, units))

    maskShape = targetMask3M.shape
    peaks3M = paintSpheres(verticesXYZ, sphereRadius_cm, xV, yV, zV, maskShape)

    # Constrain peaks to lie within the target (safety against grid rounding).
    peaks3M = np.logical_and(peaks3M, targetMask3M)

    if vertexStructName is None:
        vertexStructName = "%s_lattice_peaks" % targetName
    planC = pc.importStructureMask(peaks3M, assocScanNum, vertexStructName, planC)

    if addValley:
        valley3M = np.logical_and(targetMask3M, np.logical_not(peaks3M))
        if valleyStructName is None:
            valleyStructName = "%s_lattice_valley" % targetName
        planC = pc.importStructureMask(valley3M, assocScanNum, valleyStructName, planC)

    if addIndividualSpheres:
        for i, vtx in enumerate(verticesXYZ):
            sph3M = paintSpheres(vtx[np.newaxis, :], sphereRadius_cm, xV, yV, zV, maskShape)
            sph3M = np.logical_and(sph3M, targetMask3M)
            planC = pc.importStructureMask(
                sph3M, assocScanNum, "%s_vertex_%02d" % (targetName, i + 1), planC)

    return planC, verticesXYZ


# --------------------------------------------------------------------------
# Optimization wiring (pyRadPlan bridge)
# --------------------------------------------------------------------------

def latticeBoostObjectives(peakStructNum,
                           valleyStructNum=None,
                           gtvStructNum=None,
                           peakDose=15.0,
                           valleyMaxDose=None,
                           gtvMinDose=None,
                           peakPriority=100.0,
                           valleyPriority=60.0,
                           gtvPriority=80.0,
                           oarObjectives=None):
    """Build a pyRadPlan objectives dict for a lattice (SFRT) boost plan.

    Produces the ``{structNum: [Objective, ...]}`` mapping consumed by
    :func:`cerr.imrtp.pyradplan_bridge.planFromPlanC`, encoding the defining
    feature of lattice radiotherapy: drive the vertices ("peaks") to a high
    prescription dose while capping the surrounding tissue ("valley") at a
    low dose, producing the characteristic peak-to-valley dose ratio.

    Args:
        peakStructNum (int): Index of the lattice peaks structure
            (from :func:`createLattice`, ``<target>_lattice_peaks``).
        valleyStructNum (int or None): Index of the valley structure. When
            given, an overdosing objective caps its dose at ``valleyMaxDose``.
        gtvStructNum (int or None): Index of the whole target. When given, an
            underdosing objective enforces a minimum coverage ``gtvMinDose``.
        peakDose (float): Peak prescription dose in Gy at the vertices.
        valleyMaxDose (float or None): Max valley dose in Gy. Defaults to
            ``0.3 * peakDose`` (a peak-to-valley ratio of ~3), a common LRT choice.
        gtvMinDose (float or None): Minimum GTV dose in Gy. Defaults to
            ``valleyMaxDose`` so the whole target receives at least the valley dose.
        peakPriority, valleyPriority, gtvPriority (float): Objective weights.
        oarObjectives (dict or None): Optional extra ``{structNum: [Objective, ...]}``
            for organs at risk, merged into the returned dict.

    Returns:
        tuple:
            - dict: ``{structNum: [Objective, ...]}`` optimization objectives.
            - list: structure indices to mark as optimization targets
              (the peaks, and the GTV when ``gtvStructNum`` is given).
    """
    from cerr.imrtp import pyradplan_bridge as prp

    if valleyMaxDose is None:
        valleyMaxDose = 0.3 * peakDose
    if gtvMinDose is None:
        gtvMinDose = valleyMaxDose

    objectives = {peakStructNum: [prp.squaredDeviation(peakDose, priority=peakPriority)]}
    targetStructNums = [peakStructNum]

    if valleyStructNum is not None:
        objectives.setdefault(valleyStructNum, []).append(
            prp.squaredOverdosing(valleyMaxDose, priority=valleyPriority))

    if gtvStructNum is not None:
        objectives.setdefault(gtvStructNum, []).append(
            prp.squaredUnderdosing(gtvMinDose, priority=gtvPriority))
        targetStructNums.append(gtvStructNum)

    if oarObjectives:
        for s, objs in oarObjectives.items():
            objectives.setdefault(s, []).extend(objs)

    return objectives, targetStructNums


def optimizeLatticeBoost(planC, peakStructNum,
                         valleyStructNum=None,
                         gtvStructNum=None,
                         scanNum=0,
                         beamsNum=0,
                         gantryAngles=None,
                         couchAngles=None,
                         isoCenter=None,
                         peakDose=15.0,
                         valleyMaxDose=None,
                         gtvMinDose=None,
                         oarObjectives=None,
                         structNums=None,
                         bixelWidth=5.0,
                         machine='Generic',
                         numOfFractions=1,
                         doseGridResolution=None,
                         fractionGroupID='lattice'):
    """End-to-end lattice boost optimization through the pyRadPlan bridge.

    Wires the peak/valley structures created by :func:`createLattice` into a
    fluence-optimized plan: builds the pyRadPlan ``(ct, cst, pln)``, computes
    the beamlet dose-influence matrix, optimizes the fluence for the
    lattice-boost objectives, and imports the resulting dose into ``planC.dose``.

    Beam geometry is read from ``planC.beams[beamsNum]`` unless ``gantryAngles``
    is given explicitly (in which case ``beamsNum`` is ignored).

    Args:
        planC (cerr.plan_container.PlanC): plan container.
        peakStructNum, valleyStructNum, gtvStructNum: structure indices; see
            :func:`latticeBoostObjectives`.
        scanNum (int): CT scan index.
        beamsNum (int): index into ``planC.beams`` (used when ``gantryAngles`` is None).
        gantryAngles, couchAngles, isoCenter: explicit beam geometry
            (see :func:`cerr.imrtp.pyradplan_bridge.planFromPlanC`).
        peakDose, valleyMaxDose, gtvMinDose, oarObjectives: dose objectives
            (see :func:`latticeBoostObjectives`).
        structNums (list or None): structures to include in the cst; defaults
            to the peak/valley/GTV (and any OAR) structures referenced by the
            objectives.
        bixelWidth, machine, numOfFractions, doseGridResolution: passed to
            :func:`cerr.imrtp.pyradplan_bridge.planFromPlanC`.
        fractionGroupID (str): label stored on the imported dose.

    Returns:
        tuple: ``(w, doseNum, planC)`` -- optimized beamlet weights, index of
        the imported dose in ``planC.dose`` and the container.
    """
    from cerr.imrtp import pyradplan_bridge as prp

    objectives, targetStructNums = latticeBoostObjectives(
        peakStructNum,
        valleyStructNum=valleyStructNum,
        gtvStructNum=gtvStructNum,
        peakDose=peakDose,
        valleyMaxDose=valleyMaxDose,
        gtvMinDose=gtvMinDose,
        oarObjectives=oarObjectives,
    )

    if structNums is None:
        structNums = sorted(objectives.keys())

    ct, cst, pln = prp.planFromPlanC(
        planC, scanNum=scanNum,
        beamsNum=None if gantryAngles is not None else beamsNum,
        objectives=objectives,
        structNums=structNums,
        targetStructNums=targetStructNums,
        gantryAngles=gantryAngles,
        couchAngles=couchAngles,
        isoCenter=isoCenter,
        bixelWidth=bixelWidth,
        machine=machine,
        prescribedDose=peakDose,
        numOfFractions=numOfFractions,
        doseGridResolution=doseGridResolution,
    )
    stf, dij = prp.calcDoseInfluence(ct, cst, pln)
    w, doseNum, planC = prp.optimizeAndImportDose(
        planC, ct, cst, stf, dij, pln, scanNum=scanNum,
        fractionGroupID=fractionGroupID)
    return w, doseNum, planC
