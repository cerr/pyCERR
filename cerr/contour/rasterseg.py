"""rasterseg module.

Ths rasterseg module defines methods for creation of raster segments
from polygonal coordinates and to assemble binary mask from rastersegments.

"""

import numpy as np
from cerr.dataclasses import scan as scn


def polyFill(rowV, colV, xSize, ySize):
    # Initialize the result matrix with zeros
    result = np.zeros((xSize, ySize))

    # Some contours have more than one segment. Treat them as separate polygons on the same image.
    # They shouldn't overlap, but if they do, the value of the resultant mask will be 1.0.
    pointCount = len(rowV)
    edgeList = np.zeros((pointCount, 4))

    # Iterate over each point in the contour
    for point in range(pointCount):
        if point == pointCount - 1:
            p1x, p1y = rowV[point], colV[point]
            p2x, p2y = rowV[0], colV[0]
        else:
            p1x, p1y = rowV[point], colV[point]
            p2x, p2y = rowV[point + 1], colV[point + 1]

        # Ensure that the edge is ordered from top to bottom (y-wise)
        if p1y <= p2y:
            edgeList[point, :] = [p1x, p1y, p2x, p2y]
        else:
            edgeList[point, :] = [p2x, p2y, p1x, p1y]

    # Get the relevant y-range to loop over
    minY = int(np.ceil(np.min(edgeList[:, 1])))
    maxY = int(np.floor(np.max(edgeList[:, 3])))

    # Loop over the relevant lines in the image.
    for y in range(minY, maxY + 1):
        # Get the active edges that cover the current y value
        indV = (edgeList[:, 1] <= y) & (edgeList[:, 3] >= y)
        activeEdges = edgeList[indV, :]

        # Create lists to store pixels to draw and intersections
        drawlist = np.empty(0,dtype=int)
        togglelist = np.empty(0,dtype=float)

        # Iterate over the active edges
        edgeCount = activeEdges.shape[0]
        for edge in range(edgeCount):
            p1x, p1y, p2x, p2y = activeEdges[edge, :]

            # Check if it's a flat line
            if p1y == p2y:
                if p1x > p2x:
                    drawlist = np.append(drawlist, np.arange(int(np.ceil(p2x)), int(np.floor(p1x)) + 1))
                else:
                    drawlist = np.append(drawlist, np.arange(int(np.ceil(p1x)), int(np.floor(p2x)) + 1))
            elif p2y == y and p2x == round(p2x):
                drawlist = np.append(drawlist,int(p2x))
            elif p2y != y:
                invslope = float(p2x - p1x) / float(p2y - p1y)
                togglelist = np.append(togglelist, p1x + (invslope * (y - p1y)))

        # Snap to center if less than tolerance
        snap_tol = 1e-6
        togglelistRound = np.round(togglelist)
        ind_snap = np.abs(togglelistRound - togglelist) < snap_tol
        togglelist[ind_snap] = togglelistRound[ind_snap]

        # Sort the toggle points in order
        togglelist.sort()

        # Iterate over pairs in the toggle list and draw the pixels accordingly
        for i in range(0, len(togglelist), 2):
            x1, x2 = int(np.ceil(togglelist[i])), int(np.floor(togglelist[i + 1]))
            result[x1:x2 + 1, y] = 1

        # Turn on the pixels in the drawlist
        result[drawlist, y] = 1

    return result

# Example usage:
# rowV = [10, 20, 30, 20]
# colV = [5, 10, 5, 1]
# xSize = 50
# ySize = 50
# result = poly_fill(rowV, colV, xSize, ySize)


def xytom(xV, yV, sliceNum, planC, scanNum):
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


def getStrMask(str_num,planC):
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
