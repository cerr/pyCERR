import numpy as np
from scipy.interpolate import interp2d

def finterp3(xInterpV, yInterpV, zInterpV, field3M, xFieldV, yFieldV, zFieldV, OOBV=None):
    siz = field3M.shape

    if OOBV is None:
        OOBV = np.nan

    xDelta = xFieldV[1]
    yDelta = yFieldV[1]

    # Check for row/column vector
    xInterpV = xInterpV.flatten()
    yInterpV = yInterpV.flatten()
    zInterpV = zInterpV.flatten()
    xFieldV = xFieldV.flatten()
    yFieldV = yFieldV.flatten()
    zFieldV = zFieldV.flatten()

    # Get r,c,s indices.
    cols = (xInterpV - (xFieldV[0] - xDelta)) / xDelta - 1
    rows = (yInterpV - (yFieldV[0] - yDelta)) / yDelta - 1
    if len(zFieldV) > 1:
        zInterpV[zInterpV < min(zFieldV)] = min(zFieldV)  # Assign DUMMY value
        zInterpV[zInterpV > max(zFieldV)] = max(zFieldV)  # Assign DUMMY value
        #binIndex = np.searchsorted(zFieldV, zInterpV)
        binIndex = np.digitize(zInterpV,zFieldV)
        yV = np.arange(0, len(zFieldV))
        dxV = np.diff(zFieldV)
        dxV = np.append(dxV, 1)  # DUMMY 1
        slopeV = 1.0 / dxV
        #slopeV[binIndex == len(zFieldV)-1] = 0
        slcs = yV[binIndex-1] + slopeV[binIndex-1] * (zInterpV - zFieldV[binIndex - 2])
        slcs[np.isnan(slcs)] = np.nan
    else:
        slcs = np.ones_like(cols)  # This effectively negates Z.  All values are in plane.  Bad idea?

    slcs = slcs - 1

    # Find indices out of bounds.
    colNaN = np.logical_or(cols >= siz[1], cols < 1)
    colLast = (cols - siz[1]) ** 2 < 1e-3
    yInterpColLastV = yInterpV[colLast]
    zInterpColLastV = zInterpV[colLast]

    rowNaN = np.logical_or(rows >= siz[0], rows < 1)
    rowLast = (rows - siz[0]) ** 2 < 1e-3
    xInterpRowLastV = xInterpV[rowLast]
    zInterpRowLastV = zInterpV[rowLast]

    slcNaN = np.logical_or(np.isnan(slcs), np.logical_or(slcs < 1, slcs >= siz[2]))
    slcLast = (slcs - siz[2]) ** 2 < 1e-3
    xInterpLastV = xInterpV[slcLast]
    yInterpLastV = yInterpV[slcLast]

    # Set those to a proxy 1.
    rows[rowNaN] = 0
    cols[colNaN] = 0
    slcs[slcNaN] = 0

    colFloor = np.floor(cols)
    colMod = cols - colFloor
    oneMinusColMod = (1 - colMod)

    rowFloor = np.floor(rows)
    rowMod = rows - rowFloor
    oneMinusRowMod = (1 - rowMod)

    slcFloor = np.floor(slcs)
    slcMod = slcs - slcFloor
    oneMinusSlcMod = (1 - slcMod)

    rowFloor = np.asarray(rowFloor,dtype=int)
    colFloor = np.asarray(colFloor,dtype=int)
    slcFloor = np.asarray(slcFloor,dtype=int)

    # Accumulate contribution from each voxel surrounding x,y,z point.
    interpV = field3M[rowFloor,colFloor,slcFloor] * oneMinusRowMod * oneMinusColMod * oneMinusSlcMod
    interpV += field3M[rowFloor+1,colFloor,slcFloor] * rowMod * oneMinusColMod * oneMinusSlcMod
    interpV += field3M[rowFloor,colFloor+1,slcFloor] * oneMinusRowMod * colMod * oneMinusSlcMod
    interpV += field3M[rowFloor+1,colFloor+1,slcFloor] * rowMod * colMod * oneMinusSlcMod
    interpV += field3M[rowFloor,colFloor,slcFloor+1] * oneMinusRowMod * oneMinusColMod * slcMod
    interpV += field3M[rowFloor+1,colFloor,slcFloor+1] * rowMod * oneMinusColMod * slcMod
    interpV += field3M[rowFloor,colFloor+1,slcFloor+1] * oneMinusRowMod * colMod * slcMod
    interpV += field3M[rowFloor+1,colFloor+1,slcFloor+1] * rowMod * colMod * slcMod


    # # Linear indices of lower bound contributing points.
    # INDEXLIST = (rowFloor + (colFloor - 1) * siz[0] + (slcFloor - 1) * siz[0] * siz[1]).astype(int)
    #
    # # Linear offsets when moving in 3D matrix.
    # oneRow = 1
    # oneCol = siz[0]
    # oneSlc = siz[0] * siz[1]
    #
    # # Accumulate contribution from each voxel surrounding x,y,z point.
    # interpV = field3M.flat[INDEXLIST] * oneMinusRowMod * oneMinusColMod * oneMinusSlcMod
    # interpV += field3M.flat[INDEXLIST + oneRow] * rowMod * oneMinusColMod * oneMinusSlcMod
    # interpV += field3M.flat[INDEXLIST + oneCol] * oneMinusRowMod * colMod * oneMinusSlcMod
    # interpV += field3M.flat[INDEXLIST + oneCol + oneRow] * rowMod * colMod * oneMinusSlcMod
    # interpV += field3M.flat[INDEXLIST + oneSlc] * oneMinusRowMod * oneMinusColMod * slcMod
    # interpV += field3M.flat[INDEXLIST + oneSlc + oneRow] * rowMod * oneMinusColMod * slcMod
    # interpV += field3M.flat[INDEXLIST + oneSlc + oneCol] * oneMinusRowMod * colMod * slcMod
    # interpV += field3M.flat[INDEXLIST + oneSlc + oneCol + oneRow] * rowMod * colMod * slcMod

    # Replace proxy 1s with out of bounds vals.
    interpV[rowNaN | colNaN | slcNaN] = OOBV

    # 2D interpolate last slice
    if any(slcLast):
        interpV[slcLast] = interp2d(xFieldV, yFieldV, field3M[:, :, -1])(xInterpLastV, yInterpLastV)

    if any(colLast):
        if len(zFieldV) > 1:
            interpV[colLast] = interp2d(yFieldV, zFieldV, np.squeeze(field3M[:, -1, :].T))(yInterpColLastV, zInterpColLastV)

    if any(rowLast):
        if len(zFieldV) > 1:
            interpV[rowLast] = interp2d(xFieldV, zFieldV, np.squeeze(field3M[-1, :, :].T))(xInterpRowLastV, zInterpRowLastV)

    return interpV



def finterp2(x, y, z, xi, yi, uniformFlag=0, outOfRangeVal=np.nan):
    """
    This Python version of the finterp2 function should now work
    similarly to the MATLAB version for regularly spaced
    matrices and uniform grids. The output zi will be a 2D array
    interpolated from the input z based on the provided xi and yi
    vectors. The uniformFlag parameter is optional and defaults to 0.
    The outOfRangeVal parameter is also optional and defaults to np.nan.
    """

    if uniformFlag == 1:
        cols = np.interp(xi, x, np.arange(1, len(x) + 1))
        rows = np.interp(yi, y, np.arange(1, len(y) + 1))

        colFloor = np.floor(cols)
        colMod = cols - colFloor
        colValues = z[:, colFloor.astype(int) - 1] * (1 - colMod) + z[:, colFloor.astype(int)] * colMod

        rowFloor = np.floor(rows)
        rowMod = rows - rowFloor
        rowModMinusOne = 1 - rowMod

        part1 = colValues[rowFloor.astype(int) - 1, :] * rowModMinusOne
        part2 = colValues[rowFloor.astype(int), :] * rowMod

        zi = np.full((len(yi), len(xi)), outOfRangeVal)
        zi[~np.isnan(rowFloor) & ~np.isnan(colFloor)] = part1 + part2
    else:
        siz = z.shape

        xDelta = x[1] - x[0]
        yDelta = y[1] - y[0]

        cols = (xi - (x[0] - xDelta)) / xDelta
        rows = (yi - (y[0] - yDelta)) / yDelta
        colNaN = np.logical_or(cols >= siz[1], cols < 0)
        rowNaN = np.logical_or(rows >= siz[0], rows < 0)

        rows[rowNaN] = 0
        cols[colNaN] = 0

        colFloor = np.floor(cols).astype(int)
        colMod = cols - colFloor
        oneMinusColMod = 1 - colMod

        rowFloor = np.floor(rows).astype(int)
        rowMod = rows - rowFloor
        oneMinusRowMod = 1 - rowMod

        INDEXLIST = rowFloor + colFloor * siz[0]
        v1 = z.flat[INDEXLIST] * oneMinusRowMod * oneMinusColMod
        v2 = z.flat[INDEXLIST + 1] * rowMod * oneMinusColMod
        v3 = z.flat[INDEXLIST + siz[0]] * oneMinusRowMod * colMod
        v4 = z.flat[INDEXLIST + siz[0] + 1] * rowMod * colMod

        zi = v1 + v2 + v3 + v4
        zi[rowNaN | colNaN] = np.nan

    return zi
