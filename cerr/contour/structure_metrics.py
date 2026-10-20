"""structure_metrics module.

Metrics for comparing two segmentations of the same structure on the same scan,
for example an auto-segmentation against a manual reference, a propagated
contour against one drawn on the target scan, or two observers.

* Overlap: Dice, Jaccard, sensitivity, positive predictive value (PPV),
  volumes and volume ratio.
* Surface distance: Hausdorff distance (HD), its 95th percentile (HD95), mean
  surface distance (MSD) and surface Dice at given tolerances, computed with the
  ``surface_distance`` package installed with pyCERR.
* Centre-of-mass distance.
* Added path length (APL): the length of reference contour that lies outside a
  tolerance band around the test contour, an estimate of the contour a person
  would have to redraw to correct the test contour (Vaassen et al., Phys Imaging
  Radiat Oncol 2020). Computed from the contour polygons, slice by slice.

Typical use::

    from cerr.contour import structure_metrics as sm

    metrics = sm.compareStructurePair(refStructNum, testStructNum, planC)
    print(metrics['Dice'], metrics['HD95_mm'], metrics['APL_mm'])

To compare several observers of the same target (STAPLE, kappa), see
:mod:`cerr.contour.structure_consensus`.
"""

import numpy as np
import shapely
import surface_distance as sd
from shapely.geometry import Polygon

import cerr.contour.rasterseg as rs
import cerr.dataclasses.scan as scn


# ---------------------------------------------------------------------------#
#  Helpers
# ---------------------------------------------------------------------------#
def getSpacingMm(scanNum, planC):
    """Voxel spacing of a scan in mm, in the order of the mask array axes.

    pyCERR masks are indexed ``[row, col, slice]`` while
    :meth:`cerr.dataclasses.scan.Scan.getScanSpacing` returns ``(dx, dy, dz)``
    in cm. ``surface_distance`` and similar packages need the spacing per array
    axis, i.e. ``(dy, dx, dz)``.

    Args:
        scanNum (int): index of the scan in ``planC.scan``.
        planC (cerr.plan_container.PlanC): pyCERR plan container.

    Returns:
        tuple: ``(dy, dx, dz)`` in mm.
    """
    dx, dy, dz = planC.scan[scanNum].getScanSpacing() * 10
    return float(dy), float(dx), float(dz)


def _commonScan(refStructNum, testStructNum, planC):
    """Index of the scan both structures are associated with."""
    refScanNum = scn.getScanNumFromUID(planC.structure[refStructNum].assocScanUID, planC)
    testScanNum = scn.getScanNumFromUID(planC.structure[testStructNum].assocScanUID, planC)
    if refScanNum != testScanNum:
        raise ValueError("Structures %d and %d are on different scans (%s and %s). Copy one to the "
                         "other's scan first with cerr.dataclasses.structure.copyToScan."
                         % (refStructNum, testStructNum, refScanNum, testScanNum))
    return refScanNum


# ---------------------------------------------------------------------------#
#  Mask-based metrics
# ---------------------------------------------------------------------------#
def compareMasks(ref3M, test3M, spacingMm, surfaceTolerancesMm=(1.0, 2.0, 3.0),
                 hdPercentile=95):
    """Overlap and surface-distance metrics between two binary masks on one grid.

    Args:
        ref3M (np.ndarray): reference mask (bool), shape ``(rows, cols, slices)``.
        test3M (np.ndarray): test mask on the same grid.
        spacingMm (sequence of 3 floats): voxel spacing in mm per array axis,
            ``(dy, dx, dz)`` for pyCERR masks; see :func:`getSpacingMm`.
        surfaceTolerancesMm (sequence of float): tolerances for surface Dice, mm.
        hdPercentile (float): percentile for the robust Hausdorff distance.

    Returns:
        dict: with keys

            * ``Dice``, ``Jaccard``: overlap, 0 to 1.
            * ``Sensitivity``: fraction of the reference covered by the test.
            * ``PPV``: fraction of the test inside the reference.
            * ``HD_mm``: Hausdorff distance (largest surface distance).
            * ``HD<p>_mm``: ``hdPercentile`` percentile Hausdorff distance, e.g. ``HD95_mm``.
            * ``MSD_mm``: mean surface distance, averaged over both directions.
            * ``SurfaceDice_<t>mm``: fraction of both surfaces within ``t`` mm of the other.
            * ``RefVolume_cc``, ``TestVolume_cc``, ``VolumeRatio`` (test / reference).

        When either mask is empty, overlap metrics are 0 (both empty: NaN) and
        distance metrics are NaN.
    """
    ref3M = np.asarray(ref3M, dtype=bool)
    test3M = np.asarray(test3M, dtype=bool)
    if ref3M.shape != test3M.shape:
        raise ValueError("Masks must have the same shape, got %s and %s"
                         % (ref3M.shape, test3M.shape))
    spacingMm = tuple(float(s) for s in spacingMm)
    voxelCc = np.prod(spacingMm) / 1000.0

    nRef, nTest = int(ref3M.sum()), int(test3M.sum())
    nBoth = int(np.sum(ref3M & test3M))
    nEither = int(np.sum(ref3M | test3M))
    hdKey = "HD%g_mm" % hdPercentile

    metrics = {
        "Dice": 2.0 * nBoth / (nRef + nTest) if nRef + nTest else np.nan,
        "Jaccard": nBoth / nEither if nEither else np.nan,
        "Sensitivity": nBoth / nRef if nRef else np.nan,
        "PPV": nBoth / nTest if nTest else np.nan,
        "HD_mm": np.nan,
        hdKey: np.nan,
        "MSD_mm": np.nan,
        "RefVolume_cc": nRef * voxelCc,
        "TestVolume_cc": nTest * voxelCc,
        "VolumeRatio": nTest / nRef if nRef else np.nan,
    }
    for tol in surfaceTolerancesMm:
        metrics["SurfaceDice_%gmm" % tol] = np.nan

    if nRef and nTest:
        surf = sd.compute_surface_distances(ref3M, test3M, spacingMm)
        refToTest, testToRef = sd.compute_average_surface_distance(surf)
        metrics["HD_mm"] = float(sd.compute_robust_hausdorff(surf, 100))
        metrics[hdKey] = float(sd.compute_robust_hausdorff(surf, hdPercentile))
        metrics["MSD_mm"] = float((refToTest + testToRef) / 2)
        for tol in surfaceTolerancesMm:
            metrics["SurfaceDice_%gmm" % tol] = float(
                sd.compute_surface_dice_at_tolerance(surf, tol))
    elif not (nRef or nTest):
        metrics["Dice"] = np.nan
    return metrics


def centroidDistanceMm(refStructNum, testStructNum, planC):
    """Distance between the centres of mass of two structures, in mm.

    Args:
        refStructNum (int): reference structure index in ``planC.structure``.
        testStructNum (int): test structure index (same scan).
        planC (cerr.plan_container.PlanC): pyCERR plan container.

    Returns:
        float: distance in mm, NaN if either structure is empty.
    """
    scanNum = _commonScan(refStructNum, testStructNum, planC)
    xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
    centres = []
    for structNum in (refStructNum, testStructNum):
        rowV, colV, slcV = np.where(rs.getStrMask(structNum, planC))
        if rowV.size == 0:
            return np.nan
        centres.append(np.array([xV[colV].mean(), yV[rowV].mean(), zV[slcV].mean()]))
    return float(np.linalg.norm(centres[0] - centres[1]) * 10)


# ---------------------------------------------------------------------------#
#  Added path length
# ---------------------------------------------------------------------------#
def slicePolygon(contour):
    """Shapely geometry of one slice of a pyCERR structure.

    Segments are combined by symmetric difference (even-odd rule), so an inner
    segment cuts a hole, consistent with how pyCERR rasterizes contours.

    Args:
        contour (cerr.dataclasses.structure.Contour or list): one element of
            ``Structure.contour``; empty for a slice without contours.

    Returns:
        shapely geometry in pyCERR virtual x, y (cm), or None when the slice
        has no valid polygon.
    """
    if not contour:
        return None
    geom = None
    for seg in contour.segments:
        points = np.asarray(seg.points)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        xy = [(float(p[0]), float(p[1])) for p in points]
        if len(set(xy)) < 3:
            continue
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        geom = poly if geom is None else geom.symmetric_difference(poly)
    if geom is None or geom.is_empty or geom.area <= 0:
        return None
    return geom


def addedPathLengthSlice(refGeom, testGeom, toleranceCm=0.1):
    """Added path length on one slice.

    The length of the reference boundary that lies outside a band of
    ``toleranceCm`` on either side of the test boundary: inside the test
    contour shrunk by the tolerance, or outside it grown by the tolerance.

    Args:
        refGeom: shapely (Multi)Polygon of the reference, or None.
        testGeom: shapely (Multi)Polygon of the test, or None.
        toleranceCm (float): half-width of the tolerance band, in the units of
            the geometries (pyCERR uses cm).

    Returns:
        float: added path length in the units of the geometries. When only one
        of the two exists, its whole perimeter.
    """
    if refGeom is None and testGeom is None:
        return 0.0
    if testGeom is None:
        return float(refGeom.length)
    if refGeom is None:
        return float(testGeom.length)

    expanded = testGeom.buffer(toleranceCm, quad_segs=32)
    contracted = testGeom.buffer(-toleranceCm, quad_segs=32)
    band = expanded.difference(contracted) if not contracted.is_empty else expanded
    # Reference boundary outside the band: the part a person would redraw
    outside = refGeom.boundary.difference(band)
    return float(outside.length)


def addedPathLength(refStructNum, testStructNum, planC, toleranceMm=1.0,
                    includeTestOnlySlices=True):
    """Added path length (APL) between two structures on the same scan.

    APL estimates how much contour has to be redrawn to turn the test
    segmentation into the reference. On each slice it is the length of the
    reference contour farther than ``toleranceMm`` from the test contour. On
    slices where only the reference has a contour, the whole reference contour
    counts; where only the test has one, its whole contour counts too, unless
    ``includeTestOnlySlices`` is False (removing a contour is quick, so some
    studies count only added reference contour).

    Args:
        refStructNum (int): reference structure index in ``planC.structure``.
        testStructNum (int): test structure index (same scan).
        planC (cerr.plan_container.PlanC): pyCERR plan container.
        toleranceMm (float): half-width of the tolerance band in mm. Default 1 mm.
        includeTestOnlySlices (bool): count the test contour on slices where the
            reference has none. Default True.

    Returns:
        tuple: ``(totalMm, perSliceMm)`` with ``perSliceMm`` an array with one
        value per scan slice (0 where neither structure has a contour).
    """
    scanNum = _commonScan(refStructNum, testStructNum, planC)
    numSlices = int(planC.scan[scanNum].getScanSize()[2])
    refContours = planC.structure[refStructNum].contour
    testContours = planC.structure[testStructNum].contour
    toleranceCm = toleranceMm / 10.0

    perSliceCm = np.zeros(numSlices)
    for slc in range(numSlices):
        refGeom = slicePolygon(refContours[slc]) if slc < len(refContours) else None
        testGeom = slicePolygon(testContours[slc]) if slc < len(testContours) else None
        if refGeom is None and not includeTestOnlySlices:
            continue
        try:
            perSliceCm[slc] = addedPathLengthSlice(refGeom, testGeom, toleranceCm)
        except shapely.errors.GEOSException:
            perSliceCm[slc] = np.nan
    perSliceMm = perSliceCm * 10
    return float(np.nansum(perSliceMm)), perSliceMm


# ---------------------------------------------------------------------------#
#  Everything at once
# ---------------------------------------------------------------------------#
def compareStructurePair(refStructNum, testStructNum, planC,
                         surfaceTolerancesMm=(1.0, 2.0, 3.0), hdPercentile=95,
                         aplToleranceMm=1.0):
    """All metrics between a reference and a test structure on the same scan.

    Args:
        refStructNum (int or str): reference structure index or name.
        testStructNum (int or str): test structure index or name.
        planC (cerr.plan_container.PlanC): pyCERR plan container.
        surfaceTolerancesMm (sequence of float): tolerances for surface Dice, mm.
        hdPercentile (float): percentile for the robust Hausdorff distance.
        aplToleranceMm (float): tolerance for the added path length, mm.

    Returns:
        dict: the keys of :func:`compareMasks`, plus ``CentroidDistance_mm`` and
        ``APL_mm`` (total added path length).

    Example:
        Score an auto-segmentation against a manual contour::

            from cerr.contour import structure_metrics as sm

            m = sm.compareStructurePair('Heart', 'AI_Heart', planC)
            print('Dice %.3f  HD95 %.1f mm  APL %.0f mm'
                  % (m['Dice'], m['HD95_mm'], m['APL_mm']))
    """
    from cerr.dataclasses.structure import resolveStructRef
    refStructNum = resolveStructRef(refStructNum, planC)
    testStructNum = resolveStructRef(testStructNum, planC)
    scanNum = _commonScan(refStructNum, testStructNum, planC)
    metrics = compareMasks(rs.getStrMask(refStructNum, planC),
                           rs.getStrMask(testStructNum, planC),
                           getSpacingMm(scanNum, planC),
                           surfaceTolerancesMm=surfaceTolerancesMm,
                           hdPercentile=hdPercentile)
    metrics["CentroidDistance_mm"] = centroidDistanceMm(refStructNum, testStructNum, planC)
    metrics["APL_mm"], _ = addedPathLength(refStructNum, testStructNum, planC,
                                           toleranceMm=aplToleranceMm)
    return metrics
