"""rasterseg module.

Ths rasterseg module defines methods for creation of raster segments
from polygonal coordinates and to assemble binary mask from rastersegments.

"""

import numpy as np
from cerr.dataclasses import scan as scn


def polyFill(rowV, colV, xSize, ySize):
    """Rasterize a closed polygon into a binary 2-D mask using scanline filling.

    Converts a polygon defined by row/column vertex coordinates into a filled
    binary mask of the requested dimensions.  Flat (horizontal) edges and
    vertex intersections are handled explicitly so that adjacent polygons
    produce a consistent, non-overlapping fill.

    Args:
        rowV (array-like): Row (x) coordinates of the polygon vertices, in
            order.  The polygon is automatically closed (last point connects
            back to the first).
        colV (array-like): Column (y) coordinates matching ``rowV``.
        xSize (int): Number of rows in the output mask.
        ySize (int): Number of columns in the output mask.

    Returns:
        np.ndarray: Boolean mask of shape ``(xSize, ySize)`` where filled
        pixels are 1 and background pixels are 0.
    """
    # Vectorized scanline fill: O(edge crossings) rather than a Python loop over
    # scanlines x edges. Scanlines run along the column (y) axis. Semantics:
    # an edge covers scanlines ceil(top)..bottom but toggles only where y != bottom;
    # a flat edge on an integer scanline draws ceil(min row)..floor(max row); an
    # integer bottom vertex draws its own pixel; toggles within 1e-6 of an integer
    # snap to it; sorted toggles pair up as ceil(t0)..floor(t1). An odd crossing
    # count on a scanline raises IndexError. Pixels outside the image are clipped.
    result = np.zeros((xSize, ySize))
    rowV = np.asarray(rowV, dtype=float)
    colV = np.asarray(colV, dtype=float)
    if len(rowV) == 0:
        return result

    # Edges ordered top to bottom (y-wise); the polygon is closed
    nxt = np.roll(np.arange(len(rowV)), -1)
    p1x, p1y, p2x, p2y = rowV, colV, rowV[nxt], colV[nxt]
    swap = ~(p1y <= p2y)
    ax = np.where(swap, p2x, p1x)
    ay = np.where(swap, p2y, p1y)
    bx = np.where(swap, p1x, p2x)
    by = np.where(swap, p1y, p2y)
    if int(np.floor(np.max(by))) < int(np.ceil(np.min(ay))):
        return result

    # Flat edges lying on an integer scanline
    flat = ay == by
    for e in np.nonzero(flat & (ay == np.round(ay)) & (ay >= 0) & (ay < ySize))[0]:
        lo, hi = (bx[e], ax[e]) if ax[e] > bx[e] else (ax[e], bx[e])
        result[max(int(np.ceil(lo)), 0):int(np.floor(hi)) + 1, int(ay[e])] = 1

    # Bottom vertices on integer grid points
    ax, ay, bx, by = ax[~flat], ay[~flat], bx[~flat], by[~flat]
    vertex = (by == np.round(by)) & (bx == np.round(bx))         & (bx >= 0) & (bx < xSize) & (by >= 0) & (by < ySize)
    result[bx[vertex].astype(int), by[vertex].astype(int)] = 1

    # Toggle points of each edge on every scanline it crosses
    y0 = np.ceil(ay).astype(np.int64)
    y1 = np.floor(by).astype(np.int64)
    y1 = np.where(by == np.round(by), y1 - 1, y1)  # no toggle on the bottom scanline
    k = np.maximum(y1 - y0 + 1, 0)
    total = int(k.sum())
    if total == 0:
        return result
    e = np.repeat(np.arange(len(k)), k)
    y = y0[e] + (np.arange(total) - np.repeat(np.cumsum(k) - k, k))
    invslope = (bx[e] - ax[e]) / (by[e] - ay[e])
    t = ax[e] + invslope * (y.astype(float) - ay[e])

    # Snap to center if less than tolerance
    t_round = np.round(t)
    snap = np.abs(t_round - t) < 1e-6
    t[snap] = t_round[snap]

    # Sort toggles within each scanline and pair them up
    order = np.lexsort((t, y))
    t, y = t[order], y[order]
    if np.any(np.unique(y, return_counts=True)[1] % 2):
        raise IndexError("odd number of polygon crossings on a scanline")
    x1 = np.ceil(t[0::2]).astype(np.int64)
    x2 = np.floor(t[1::2]).astype(np.int64)
    cols = y[0::2]
    # Clip runs to the image
    x1 = np.maximum(x1, 0)
    x2 = np.minimum(x2, xSize - 1)
    keep = (x1 <= x2) & (cols >= 0) & (cols < ySize)
    x1, x2, cols = x1[keep], x2[keep], cols[keep]
    if len(x1) == 0:
        return result

    # Draw all runs at once with a difference array along rows
    r0, r1, c0, c1 = int(x1.min()), int(x2.max()), int(cols.min()), int(cols.max())
    runs = np.zeros((r1 - r0 + 2, c1 - c0 + 1), dtype=np.int32)
    np.add.at(runs, (x1 - r0, cols - c0), 1)
    np.add.at(runs, (x2 + 1 - r0, cols - c0), -1)
    result[r0:r1 + 1, c0:c1 + 1][np.cumsum(runs, axis=0)[:-1] > 0] = 1
    return result

# Example usage:
# rowV = [10, 20, 30, 20]
# colV = [5, 10, 5, 1]
# xSize = 50
# ySize = 50
# result = poly_fill(rowV, colV, xSize, ySize)


def xytom(xV, yV, sliceNum, planC, scanNum):
    """Convert physical (AAPM/DICOM) x/y coordinates to image row/column indices.

    Scales the supplied physical coordinates by the per-slice grid spacing and
    then delegates to :func:`aapmtom` to apply the CT offset and image-centre
    shift, returning zero-based row and column indices suitable for indexing
    into the scan array.

    Args:
        xV (np.ndarray): Physical x-coordinates (cm) of the contour points.
        yV (np.ndarray): Physical y-coordinates (cm) of the contour points.
        sliceNum (int): Zero-based slice index used to look up slice-specific
            grid spacing from ``planC``.
        planC (cerr.plan_container.PlanC): pyCERR plan container object.
        scanNum (int): Index of the scan set in ``planC.scan``.

    Returns:
        tuple:
            - **rowV** (np.ndarray): Zero-based row indices corresponding to
              the input coordinates.
            - **colV** (np.ndarray): Zero-based column indices corresponding
              to the input coordinates.
    """
    scaleX = planC.scan[scanNum].scanInfo[sliceNum].grid2Units
    scaleY = planC.scan[scanNum].scanInfo[sliceNum].grid1Units
    imageSizeV = [planC.scan[scanNum].scanInfo[sliceNum].sizeOfDimension1,
                  planC.scan[scanNum].scanInfo[sliceNum].sizeOfDimension2]

    # Get any offset of CT scans to apply (neg) to structures
    xCTOffset = planC.scan[scanNum].scanInfo[0].xOffset \
        if planC.scan[scanNum].scanInfo[0].xOffset else 0
    yCTOffset = planC.scan[scanNum].scanInfo[0].yOffset \
        if planC.scan[scanNum].scanInfo[0].yOffset else 0

    x2V = xV / scaleX
    y2V = yV / scaleY
    rowV, colV = aapmtom(x2V, y2V, xCTOffset / scaleX,
                         yCTOffset / scaleY, imageSizeV)

    return rowV, colV

# Example usage:
# xV = [10, 20, 30]
# yV = [5, 10, 15]
# sliceNum = 1
# planC = {...}  # PlanC data structure containing necessary information
# scanNum = 0
# rowV, colV = xytom(xV, yV, sliceNum, planC, scanNum)

def mask2scan(maskM, optS, sliceNum):
    """Convert a 2-D binary mask into raster-segment run-length rows.

    Detects the start and stop column positions of every contiguous run of
    ``True`` pixels in each row of ``maskM`` and encodes each run as a
    segment row containing the physical y/x start/stop coordinates, voxel
    width, slice number, and pixel indices.

    Args:
        maskM (np.ndarray): 2-D boolean mask of shape ``(rows, cols)`` for a
            single slice, where ``True`` indicates structure voxels.
        optS (dict): Option dictionary with the following required keys:

            - ``"ROIxVoxelWidth"`` (float): Voxel width in the x-direction (cm).
            - ``"ROIyVoxelWidth"`` (float): Voxel width in the y-direction (cm).
            - ``"xCTOffset"`` (float): CT x-offset (cm).
            - ``"yCTOffset"`` (float): CT y-offset (cm).
        sliceNum (int or float): Slice index to record in the segment array.

    Returns:
        np.ndarray: Array of shape ``(N, 8)`` where each row encodes one
        run-length segment:
        ``[y_start, x_start, x_stop, delta_x, slice_num, row_idx,
        col_start_idx, col_stop_idx]``.

    Raises:
        ValueError: If the derived y-start and y-stop values for any segment
            are inconsistent, indicating a conversion error.
    """
    delta_x = optS["ROIxVoxelWidth"]
    delta_y = optS["ROIyVoxelWidth"]

    sizV = maskM.shape
    mask2M = np.zeros((sizV[0], sizV[1] + 2), dtype=bool)
    mask2M[:, 1:-1] = maskM

    rotMask2M = mask2M.T.astype(np.int8)
    diffM = rotMask2M[1:, :] - rotMask2M[:-1, :]
    startM = diffM == 1
    stopM = diffM == -1

    j1V, i1V = np.where(startM.T)
    xStartV, yStartV = mtoaapm(j1V, i1V, sizV)
    xStartV = xStartV * delta_x + optS["xCTOffset"]
    yStartV = yStartV * delta_y + optS["yCTOffset"]

    j2V, i2V = np.where(stopM.T)
    xStopV, yStopV = mtoaapm(j2V, i2V - 1, sizV)
    xStopV = xStopV * delta_x + optS["xCTOffset"]
    yStopV = yStopV * delta_y + optS["yCTOffset"]

    if np.any(yStartV != yStopV):
        raise ValueError("Oops! Problem in converting a mask to scan segments. Check options.")

    z1V = np.ones(len(j1V)) * sliceNum
    segmentsM = np.column_stack((yStartV, xStartV, xStopV, np.ones(len(xStopV)) * delta_x, z1V, j1V, i1V, i2V - 1))

    return segmentsM

# Example usage:
# maskM = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0]])
# optS = {"ROIxVoxelWidth": 0.5, "ROIyVoxelWidth": 0.5, "xCTOffset": 0, "yCTOffset": 0}
# sliceNum = 1
# segmentsM = mask2scan(maskM, optS, sliceNum)

def aapmtom(xAAPMShifted, yAAPMShifted, xOffset, yOffset, ImageWidth, voxelSizeV=[1, 1]):
    """Convert AAPM/DICOM-shifted physical coordinates to zero-based image row/column indices.

    Applies CT-offset correction, optional voxel-size scaling, and the
    image-centre shift to map physical (AAPM convention) x/y coordinates into
    zero-based row and column indices.  Values are snapped to the nearest
    integer when they fall within a tight tolerance, avoiding floating-point
    artefacts at pixel boundaries.

    Args:
        xAAPMShifted (np.ndarray): Physical x-coordinates already shifted by
            the CT x-offset (cm).
        yAAPMShifted (np.ndarray): Physical y-coordinates already shifted by
            the CT y-offset (cm).
        xOffset (float): CT x-offset (cm) to subtract from the shifted
            coordinates.
        yOffset (float): CT y-offset (cm) to subtract from the shifted
            coordinates.
        ImageWidth (list[int]): Two-element list ``[nRows, nCols]`` giving the
            image dimensions in pixels.
        voxelSizeV (list[float], optional): Two-element list
            ``[yVoxelSize, xVoxelSize]`` in cm.  Defaults to ``[1, 1]``
            (i.e. coordinates are already in pixel units).

    Returns:
        tuple:
            - **Row** (np.ndarray): Zero-based row indices.
            - **Col** (np.ndarray): Zero-based column indices.
    """
    xOffset /= voxelSizeV[1]
    yOffset /= voxelSizeV[0]

    xAAPMShifted /= voxelSizeV[1]
    yAAPMShifted /= voxelSizeV[0]

    xAAPM = xAAPMShifted - xOffset
    yAAPM = yAAPMShifted - yOffset

    yAAPMReshifted = yAAPM - ImageWidth[0] / 2 - 0.5
    xAAPMReshifted = xAAPM + ImageWidth[1] / 2 + 0.5

    Row = -yAAPMReshifted - 1 # -1 to account for 0 based indexing in Python
    Col = xAAPMReshifted - 1 # -1 to account for 0 based indexing in Python

    snap_tol = 1e-5
    roundRow = np.round(Row)
    roundCol = np.round(Col)
    row_snap_inds = np.abs(Row-roundRow) < snap_tol
    col_snap_inds = np.abs(Col-roundCol) < snap_tol
    Row[row_snap_inds] =  roundRow[row_snap_inds]
    Col[col_snap_inds] = roundCol[col_snap_inds]

    return Row, Col

# Example usage:
# xAAPMShifted = [10, 20, 30]
# yAAPMShifted = [5, 10, 15]
# xOffset = 2
# yOffset = 3
# ImageWidth = [100, 150]
# voxelSizeV = [0.5, 0.5]
# Row, Col = aapmtom(xAAPMShifted, yAAPMShifted, xOffset, yOffset, ImageWidth, voxelSizeV)


def mtoaapm(Row, Col, Dims, gridUnits=[1, 1], offset=[0, 0]):
    """Convert zero-based image row/column indices to physical AAPM x/y coordinates.

    Inverts the AAPM-to-matrix mapping: given pixel row and column indices,
    computes the physical (AAPM convention) x and y coordinates by applying
    the image-centre shift, optional voxel-size scaling, and an additional
    coordinate offset.

    Args:
        Row (np.ndarray): Zero-based row indices into the image array.
        Col (np.ndarray): Zero-based column indices into the image array.
        Dims (list[int]): Two-element list ``[nRows, nCols]`` giving the image
            dimensions in pixels.
        gridUnits (list[float], optional): Two-element list
            ``[yGridUnit, xGridUnit]`` specifying voxel size in cm.  When both
            elements are 1 (default) coordinates are returned in pixel units.
        offset (list[float], optional): Two-element list ``[yOffset, xOffset]``
            added to the scaled coordinates (cm).  Defaults to ``[0, 0]``.

    Returns:
        tuple:
            - **xAAPM** (np.ndarray): Physical x-coordinates (cm) in the AAPM
              frame.
            - **yAAPM** (np.ndarray): Physical y-coordinates (cm) in the AAPM
              frame.
    """
    #yAAPMShifted = np.double(-np.double(Row) + Dims[0])
    #xAAPMShifted = np.double(Col)
    yAAPMShifted = -Row.astype(float) + Dims[0]
    xAAPMShifted = Col.astype(float)

    yOffset = Dims[0] / 2 + 1 - 0.5 # +1 to account for 0-based indexing in Python
    xOffset = Dims[1] / 2 - 1 + 0.5 # -1 to account for 0-based indexing in Python

    xAAPM = xAAPMShifted - xOffset
    yAAPM = yAAPMShifted - yOffset

    if len(gridUnits) > 1:
        xAAPM = xAAPM * gridUnits[1] + offset[1]
        yAAPM = yAAPM * gridUnits[0] + offset[0]

    return xAAPM, yAAPM

# Example usage:
# Row = [10, 20, 30]
# Col = [5, 10, 15]
# Dims = [100, 150]
# gridUnits = [0.5, 0.5]
# offset = [2, 3]
# xAAPM, yAAPM = mtoaapm(Row, Col, Dims, gridUnits, offset)


def getStrMask(str_num, planC):
    """Return a full 3-D binary mask for a structure across all scan slices.

    Looks up the raster segments for the specified structure, determines the
    associated scan dimensions, and assembles a boolean volumetric mask by
    calling :func:`raster_to_mask` and placing the per-slice masks into the
    correct slice positions.

    Args:
        str_num (int | float | np.integer | cerr.dataclasses.structure.Structure):
            Either the integer index of the structure in ``planC.structure``,
            or a structure object itself.
        planC (cerr.plan_container.PlanC): pyCERR plan container object.

    Returns:
        np.ndarray: Boolean array of shape ``(nRows, nCols, nSlices)`` where
        ``True`` indicates voxels belonging to the structure.

    Example:
        Volume of a structure and mean scan intensity inside it::

            import numpy as np
            from cerr.contour import rasterseg as rs

            mask3M = rs.getStrMask(0, planC)
            scanNum = planC.structure[0].getStructureAssociatedScan(planC)
            volumeCc = mask3M.sum() * np.prod(planC.scan[scanNum].getScanSpacing())
            meanVal = planC.scan[scanNum].getScanArray()[mask3M].mean()

    """
    if isinstance(str_num, (int, float, np.integer)):
        rasterSegments = planC.structure[str_num].rasterSegments
        assocScanUID = planC.structure[str_num].assocScanUID
    else:
        rasterSegments = str_num.rasterSegments
        assocScanUID = str_num.assocScanUID
    scan_num = scn.getScanNumFromUID(assocScanUID,planC)
    num_rows, num_cols, num_slcs = planC.scan[scan_num].getScanSize()
    mask3M = np.zeros((num_rows,num_cols,num_slcs),dtype = bool)
    slcMask3M,slicesV = raster_to_mask(rasterSegments, scan_num, planC)
    slicesV = np.asarray(slicesV,int)
    if len(slicesV) > 0:
        mask3M[:,:,slicesV] = slcMask3M
    return mask3M

def raster_to_mask(rasterSegments, scanNum, planC):
    """Convert raster segments to a stack of per-slice 2-D binary masks.

    Iterates over the run-length encoded raster segment rows and fills the
    corresponding pixel spans in a 3-D boolean array indexed by unique slice
    number.  Only the slices actually referenced in ``rasterSegments`` are
    included in the output stack.

    Args:
        rasterSegments (np.ndarray): Array of shape ``(N, >=9)`` where each
            row is a raster segment produced by :func:`generateRastersegs`.
            Column indices used:

            - ``[:, 5]`` – CT slice number (zero-based).
            - ``[:, 6]`` – Row index within the slice.
            - ``[:, 7]`` – Start column index of the filled run.
            - ``[:, 8]`` – Stop column index of the filled run (inclusive).
        scanNum (int): Index of the scan set in ``planC.scan``, used to
            determine the slice dimensions.
        planC (cerr.plan_container.PlanC): pyCERR plan container object.

    Returns:
        tuple:
            - **dataSet** (np.ndarray): Boolean array of shape
              ``(nRows, nCols, nUniqueSlices)`` containing the filled masks.
              Returns a single empty ``(nRows, nCols)`` slice when
              ``rasterSegments`` is empty.
            - **uniqueSlices** (np.ndarray | list): Sorted array of the unique
              CT slice indices represented in ``rasterSegments``, or an empty
              list when there are no segments.
    """
    # Get x, y size of each slice in this scanset
    siz = planC.scan[scanNum].getScanSize()
    x, y = siz[1], siz[0]

    # If no raster segments, return an empty mask
    if not np.any(rasterSegments):
        dataSet = np.zeros((y, x), dtype=bool)
        uniqueSlices = []
        return dataSet, uniqueSlices

    # Figure out how many unique slices we need
    uniqueSlices = np.unique(rasterSegments[:, 5]).astype(int)
    nUniqueSlices = len(uniqueSlices)
    dataSet = np.zeros((y, x, nUniqueSlices), dtype=bool)

    # Loop over raster segments and fill in the proper slice
    for i in range(rasterSegments.shape[0]):
        CTSliceNum = rasterSegments[i, 5]
        index = np.where(uniqueSlices == CTSliceNum)[0][0]
        dataSet[int(rasterSegments[i, 6]), int(rasterSegments[i, 7]):int(rasterSegments[i, 8]+1), index] = 1

    return dataSet, uniqueSlices

# Usage:
# dataSet, uniqueSlices = rasterToMask(rasterSegments, scanNum, planC)

def generateRastersegs(strObj, planC):
    """Generate raster-segment run-length encoding from a structure's contour polygons.

    For each slice that contains contour data, the polygon vertices are
    converted from physical coordinates to image row/column indices via
    :func:`xytom`, rasterized into a binary mask with :func:`polyFill`, and
    then encoded as run-length segments by :func:`mask2scan`.  The z-coordinate
    and voxel thickness are appended to each segment row, and all slices are
    stacked into a single array.

    Special-case handling ensures that single-point or duplicate-point contours
    produce at least one filled pixel.

    Args:
        strObj (cerr.dataclasses.structure.Structure): Structure object whose
            ``contour`` list provides the per-slice polygon data, and whose
            ``assocScanUID`` attribute identifies the parent scan.
        planC (cerr.plan_container.PlanC): pyCERR plan container object used
            to retrieve scan geometry.

    Returns:
        np.ndarray: Array of shape ``(N, 10)`` where each row encodes one
        raster segment:
        ``[z_value, y_start, x_start, x_stop, delta_x, slice_num, row_idx,
        col_start_idx, col_stop_idx, voxel_thickness]``.
        Returns an empty ``np.ndarray`` when the structure has no contour data.
    """
    scan_num = scn.getScanNumFromUID(strObj.assocScanUID,planC)
    num_rows, num_cols, num_slcs = planC.scan[scan_num].getScanSize()
    seg_opts = {"ROIxVoxelWidth": planC.scan[scan_num].scanInfo[0].grid2Units,
                   "ROIyVoxelWidth": planC.scan[scan_num].scanInfo[0].grid1Units,
                   "ROIImageSize": [num_rows,num_cols],
                   "xCTOffset": planC.scan[scan_num].scanInfo[0].xOffset,
                   "yCTOffset": planC.scan[scan_num].scanInfo[0].yOffset}
    segM = np.empty((0,0))
    for slc_num in range(len(strObj.contour)):
        if not strObj.contour[slc_num]:
            continue
        num_segs = len(strObj.contour[slc_num].segments)
        mask3M = np.zeros((num_rows,num_cols,num_segs), dtype=bool)
        for seg_num in range(num_segs):
            ptsM = strObj.contour[slc_num].segments[seg_num].points
            rowV,colV = xytom(ptsM[:, 0],
                              ptsM[:, 1],
                              slc_num, planC, scan_num)
            # Shift row, col indices off the edge of the image mask to the edge.
            rowV[rowV >= num_rows] = num_rows - 1
            rowV[rowV < 0] = 0
            colV[colV >= num_cols] = num_cols - 1
            colV[colV < 0] = 0
            zValue = ptsM[0, 2]
            maskM = polyFill(rowV, colV, num_rows, num_cols)
            if (ptsM.shape[0] == 1) or \
                (ptsM.shape[0] == 2 and np.all(np.equal(ptsM[0,:], ptsM[1,:]))):
                maskM[int(np.round(rowV[0])), int(np.round(colV[0]))] = 1
            mask3M[:,:,seg_num] = maskM
        maskM = np.sum(mask3M,2) == 1
        tempSegM = mask2scan(maskM,seg_opts,slc_num)
        zValuesV = np.ones((tempSegM.shape[0],1)) * zValue
        voxel_thickness = planC.scan[scan_num].scanInfo[slc_num].voxelThickness
        thicknessV = np.ones((tempSegM.shape[0],1))  * voxel_thickness
        if np.any(segM):
            segM = np.vstack((segM,np.hstack((zValuesV, tempSegM, thicknessV))))
        else:
            segM = np.hstack((zValuesV, tempSegM, thicknessV))
    return segM
