"""
Image pre- and post-processing transformations.
"""
import math
import numpy as np
from skimage.transform import resize
from cerr.utils.statistics import round

EPS = np.finfo(np.float32).eps


def resizeScanAndMask(scan3M, mask4M, gridS, outputImgSizeV, method, \
                      limitsM=None, preserveAspectFlag=False):
    """
    Function to resize input scan and mask using specified coordinates, method, and output dimensions.
    Supports preserving aspect ratio and slice-wise resizing within bounding box limits.

    Args:
        scan3M (np.ndarray): 3D input scan.
        mask4M (np.ndarray): 4D input mask of dimension [nRows x nCols x nSlices x nStructures]
                            (stack of binary masks representing various structures).
        gridS (tuple): Vectors (xV, yV, zV) for coordinates of input scan/mask (3D methods)
                       OR  matrices of slices-wise coordinates (xM, yM, zM) of input scan/mask (2D methods).
        outputImgSizeV (np.array): Output image dimensions [nRows, nCols, nSlices].
        method (string): Resizing method for input scan. Supported options:

                * 'padorcrop3d': pad or crop rows and cols symmetrically (each axis
                  independently) to the output size. The scan is padded with its minimum.
                * 'pad3d' / 'unpad3d': as 'padorcrop3d', but only pad / only crop.
                * 'bilinear', 'bicubic', 'nearest': interpolate each slice to the
                  output rows and cols. Masks are always resized with 'nearest'.
                * 'pad2d', 'unpad2d': crop a fixed-size window centred on the
                  per-slice bounding box in ``limitsM``, and its inverse.
                * 'padslices', 'unpadslices': add or remove slices at the end.

                For the 3D methods the returned grid gives the physical position
                of every output row, col and slice, so the result can be imported
                with :func:`cerr.plan_container.importScanArray` and lines up with
                the input scan.
        limitsM (np.ndarray): [optional, default=None] Extents of bounding box on each slice.
                              minr = limitsM[slcNum, 0], maxr = limitsM[slcNum, 1],
                              minc = limitsM[slcNum, 2], maxc = limitsM[slcNum, 3]
        preserveAspectFlag (bool): Flag to preserve input aspect ratio by first padding rows or
                              cols to a square with the mean background intensity. Used by
                              'padorcrop3d', 'bilinear', 'bicubic' and 'nearest'.

    Returns:
        tuple: ``(scanOut3M, maskOut4M, gridOutS)``

            - **scanOut3M** (*np.ndarray*): 3D resized scan.
            - **maskOut4M** (*np.ndarray*): 4D resized mask.
            - **gridOutS** (*tuple*): vectors (xV, yV, zV) of output coordinates (3D methods),
              or matrices of slice-wise coordinates (xM, yM, zM) (2D methods).

    Example:
        Pad a scan to 512 x 512 for a network and import the result so it lines up
        with the original::

            from cerr import plan_container as pc
            from cerr.utils import image_proc

            scan3M = planC.scan[0].getScanArray()
            gridS = planC.scan[0].getScanXYZVals()
            out3M, _, outGridS = image_proc.resizeScanAndMask(
                scan3M, None, gridS, [512, 512, scan3M.shape[2]], 'padorcrop3d')
            planC = pc.importScanArray(out3M, *outGridS, 'CT', 0, planC)
    """
    # Get input image size
    if scan3M is not None:
        origSizeV = [scan3M.shape[0], scan3M.shape[1], scan3M.shape[2]]
    else:
        origSizeV = [mask4M.shape[0], mask4M.shape[1], mask4M.shape[2]]

    # Get no. input labels
    if mask4M is None:
        numLabels = 0
    else:
        numLabels = mask4M.shape[3]

    # Get input grid and voxel dimensions
    xV = gridS[0]
    yV = gridS[1]
    zV = gridS[2]

    if xV.ndim == 1:
        voxSizeV = [np.median(np.diff(xV)),
                    np.median(np.diff(yV)),
                    np.median(np.diff(zV))]
    elif xV.ndim == 2:
        # 2D methods
        voxSizeV = [np.median(np.diff(xV[:, 0])),
                    np.median(np.diff(yV[:, 0])),
                    np.median(np.diff(zV))]

    # Resize image using selected method
    methodLower = method.lower()
    if methodLower in ['padorcrop3d', 'pad3d', 'unpad3d']:
        # Pad and/or crop rows and cols by the same amount across all slices.
        # 'pad3d' only pads and 'unpad3d' only crops; 'padorcrop3d' does either,
        # independently for rows and cols.
        if preserveAspectFlag and methodLower == 'padorcrop3d':
            # Preserve aspect ratio by padding to a square with the mean
            # background intensity (from a corner of the scan) first
            squareSize = max(origSizeV[:2])
            bgVal = np.mean(scan3M[:5, :5, :5]) if scan3M is not None else 0
            squareSizeV = [squareSize, squareSize, origSizeV[2]]
            scan3M, mask4M, gridS = _padOrCrop3d(scan3M, mask4M, gridS, squareSizeV, bgVal)
            origSizeV = [squareSize, squareSize, origSizeV[2]]

        rowDiff = outputImgSizeV[0] - origSizeV[0]
        colDiff = outputImgSizeV[1] - origSizeV[1]
        if methodLower == 'pad3d' and (rowDiff < 0 or colDiff < 0):
            raise ValueError("To resize by padding, output image dimensions "
                             "must be larger than (cropped) input image dimensions")
        if methodLower == 'unpad3d' and (rowDiff > 0 or colDiff > 0):
            raise ValueError("To resize by cropping, output image dimensions "
                             "must be smaller than input image dimensions")

        padVal = np.min(scan3M) if scan3M is not None else 0
        scanOut3M, maskOut4M, gridOutS = _padOrCrop3d(scan3M, mask4M, gridS,
                                                      outputImgSizeV, padVal)

    elif methodLower == 'pad2d':
        # Pad ROI by varying amounts across slices for identical output dimensions.

        # Initialize resized scan and mask
        if scan3M is None:
            scanOut3M = np.array([])
        else:
            minScanVal = np.min(scan3M)
            scanOut3M = np.full([outputImgSizeV[0], outputImgSizeV[1], \
                                 origSizeV[2]], minScanVal, dtype=scan3M.dtype)

        if mask4M is None:
            maskOut4M = None
        else:
            maskOut4M = np.zeros((outputImgSizeV[0], outputImgSizeV[1], \
                                  origSizeV[2], numLabels), dtype=np.uint32)

        xOutM = np.zeros((outputImgSizeV[1], origSizeV[2]))
        yOutM = np.zeros((outputImgSizeV[0], origSizeV[2]))

        # Min/max row and col limits for each slice
        for slcNum in range(origSizeV[2]):
            minr = limitsM[slcNum, 0]
            maxr = limitsM[slcNum, 1]
            minc = limitsM[slcNum, 2]
            maxc = limitsM[slcNum, 3]

            rowCenter = round((minr + maxr) / 2)
            colCenter = round((minc + maxc) / 2)

            rMin = rowCenter - outputImgSizeV[0] // 2
            cMin = colCenter - outputImgSizeV[1] // 2

            if rMin < 0:
                rMin = 0
            if cMin < 0:
                cMin = 0

            rMax = rMin + outputImgSizeV[0] - 1
            cMax = cMin + outputImgSizeV[1] - 1

            rMin = rMin.astype(int)
            cMin = cMin.astype(int)
            rMax = rMax.astype(int)
            cMax = cMax.astype(int)

            if rMax > origSizeV[0] - 1:
                rMax = origSizeV[0] - 1
            if cMax > origSizeV[1] - 1:
                cMax = origSizeV[1] - 1

            outRmin = 0
            outCmin = 0
            outRmax = rMax - rMin
            outCmax = cMax - cMin

            if scan3M is not None:
                scanOut3M[outRmin:outRmax + 1, outCmin:outCmax + 1, slcNum] = \
                    scan3M[rMin:rMax + 1, cMin:cMax + 1, slcNum]

            if mask4M is not None:
                maskOut4M[outRmin:outRmax + 1, outCmin:outCmax + 1, slcNum, :] = \
                    mask4M[rMin:rMax + 1, cMin:cMax + 1, slcNum, :]

            xOutM[:, slcNum] = np.arange(xV[cMin], xV[cMax] + voxSizeV[0], voxSizeV[0] + EPS)
            yOutM[:, slcNum] = np.arange(yV[rMin], yV[rMax] + voxSizeV[1], voxSizeV[1] - EPS)
            gridOutS = (xOutM, yOutM, zV)

    elif methodLower == 'unpad2d':

        # Initialize resized scan and mask
        if scan3M is None:
            scanOut3M = np.array([])
        else:
            minScanVal = np.min(scan3M)
            scanOut3M = np.full([outputImgSizeV[0], outputImgSizeV[1], \
                                 origSizeV[2]], minScanVal, dtype=scan3M.dtype)

        if mask4M is None:
            maskOut4M = np.array([])
        else:
            maskOut4M = np.zeros((outputImgSizeV[0], outputImgSizeV[1], \
                                  origSizeV[2], numLabels), dtype=np.uint32)

        xOutM = np.zeros((origSizeV[0], origSizeV[2]))
        yOutM = np.zeros((origSizeV[1], origSizeV[2]))

        # Min/max row and col limits for each slice
        for slcNum in range(origSizeV[2]):
            minr = limitsM[slcNum, 0]
            maxr = limitsM[slcNum, 1]
            minc = limitsM[slcNum, 2]
            maxc = limitsM[slcNum, 3]

            rowCenter = round((minr + maxr) / 2)
            colCenter = round((minc + maxc) / 2)

            rMin = rowCenter - origSizeV[0] // 2
            cMin = colCenter - origSizeV[1] // 2

            if rMin < 0:
                rMin = 0
            if cMin < 0:
                cMin = 0

            rMax = rMin + origSizeV[0] - 1
            cMax = cMin + origSizeV[1] - 1

            rMin = rMin.astype(int)
            cMin = cMin.astype(int)
            rMax = rMax.astype(int)
            cMax = cMax.astype(int)

            if rMax > outputImgSizeV[0] - 1:
                rMax = outputImgSizeV[0] - 1
            if cMax > outputImgSizeV[1] - 1:
                cMax = outputImgSizeV[1] - 1

            outRmin = 0
            outCmin = 0
            outRmax = rMax - rMin
            outCmax = cMax - cMin

            if scan3M is not None:
                scanOut3M[rMin:rMax + 1, cMin:cMax + 1, slcNum] = scan3M[outRmin:outRmax + 1, \
                                                                  outCmin:outCmax + 1, slcNum]
            if mask4M is not None:
                maskOut4M[rMin:rMax + 1, cMin:cMax + 1, slcNum, :] = mask4M[outRmin:outRmax + 1, \
                                                                     outCmin:outCmax + 1, slcNum, :]
            # Return co-ordinates of cropped region
            xOutM[:, slcNum] = np.arange(xV[outCmin, slcNum], xV[outCmax, slcNum] + voxSizeV[0], voxSizeV[0] + EPS)
            yOutM[:, slcNum] = np.arange(yV[outRmin, slcNum], yV[outRmax, slcNum] + voxSizeV[1], voxSizeV[1] - EPS)
            gridOutS = (xOutM, yOutM, zV)

    elif methodLower in ['bilinear', 'bicubic', 'nearest']:
        # In-plane resizing of each slice. Masks are always resized with
        # 'nearest' interpolation.
        orderDict = {'bilinear': 1, 'bicubic': 3, 'nearest': 0}  # Resizing order

        if preserveAspectFlag:
            # Preserve original aspect ratio by padding to a square with the
            # mean background intensity (from a corner of the scan) first
            squareSize = max(origSizeV[:2])
            bgVal = np.mean(scan3M[:5, :5, :5]) if scan3M is not None else 0
            squareSizeV = [squareSize, squareSize, origSizeV[2]]
            scan3M, mask4M, gridS = _padOrCrop3d(scan3M, mask4M, gridS, squareSizeV, bgVal)
            origSizeV = [squareSize, squareSize, origSizeV[2]]
            xV, yV = gridS[0], gridS[1]

        scanOut3M = None
        maskOut4M = None
        outRows, outCols = int(outputImgSizeV[0]), int(outputImgSizeV[1])
        if scan3M is not None:
            scanOut3M = np.empty((outRows, outCols, scan3M.shape[2]))
            for nSlc in range(scan3M.shape[2]):
                scanOut3M[:, :, nSlc] = resize(scan3M[:, :, nSlc], (outRows, outCols),
                                               anti_aliasing=True,
                                               order=orderDict[methodLower],
                                               preserve_range=True)
        if mask4M is not None:
            maskOut4M = np.zeros((outRows, outCols, mask4M.shape[2], mask4M.shape[3]),
                                 dtype=mask4M.dtype)
            for nSlc in range(mask4M.shape[2]):
                maskOut4M[:, :, nSlc, :] = resize(mask4M[:, :, nSlc, :],
                                                  (outRows, outCols, mask4M.shape[3]),
                                                  order=0,  # 'nearest' interpolation
                                                  anti_aliasing=False,
                                                  preserve_range=True)

        # skimage's resize maps the outer edges of the input pixels onto the
        # outer edges of the output pixels, so output pixel j is centred at
        # input index (j + 0.5) * nIn / nOut - 0.5.
        xOutV = _resampledCoords(xV, outCols)
        yOutV = _resampledCoords(yV, outRows)
        gridOutS = (xOutV, yOutV, zV)

    elif methodLower == 'padslices':
        numSlices = outputImgSizeV[2]
        zPad = numSlices - origSizeV[2]

        scanOut3M = None
        if scan3M is not None:
            scanOut3M = np.zeros((scan3M.shape[0], scan3M.shape[1], numSlices))
            scanOut3M[:, :, :origSizeV[2]] = scan3M

        maskOut4M = None
        if mask4M is not None:
            maskOut4M = np.zeros((scan3M.shape[0], scan3M.shape[1], numSlices, numLabels))
            maskOut4M[:, :, :origSizeV[2], :] = mask4M

        zOutV = np.arange(zV[0], zV[-1] + (zPad + 1) * voxSizeV[2], voxSizeV[2])
        gridOutS = (xV, yV, zOutV)

    elif methodLower == 'unpadslices':
        numSlices = outputImgSizeV[2]

        scanOut3M = None
        if scan3M is not None:
            scanOut3M = scan3M[:, :, :numSlices]

        maskOut4M = None
        if mask4M is not None:
            maskOut4M = mask4M[:, :, :numSlices, :]

        zOutV = np.arange(zV[0], zV[numSlices], voxSizeV[2])
        gridOutS = (xV, yV, zOutV)

    return scanOut3M, maskOut4M, gridOutS


def transformScan(scan3M, mask4M, gridS, orientation):
    """Transpose a scan (and optional mask/grid) into a target anatomical orientation.

    Reorders the axes of ``scan3M`` and ``mask4M`` so that the data is laid out
    in the requested viewing plane without resampling.

    Args:
        scan3M (np.ndarray): 3D input scan array with shape
            ``[nRows, nCols, nSlices]``.
        mask4M (np.ndarray or None): 4D mask array with shape
            ``[nRows, nCols, nSlices, nStructures]``, or ``None`` if no mask
            is needed.
        gridS (tuple or None): Coordinate tuple ``(xV, yV, zV)`` for the input
            scan/mask, or ``None`` if coordinates are not required.
        orientation (str): Target orientation. Supported values are
            ``'axial'`` (no change), ``'coronal'``, and ``'sagittal'``.

    Returns:
        tuple: A three-element tuple ``(outScan3M, outMask4M, outGridS)`` where

        - **outScan3M** (*np.ndarray*): Transposed scan array.
        - **outMask4M** (*np.ndarray or None*): Transposed mask array, or
          ``None`` if ``mask4M`` was ``None``.
        - **outGridS** (*tuple or None*): Reordered coordinate tuple, or
          ``None`` if ``gridS`` was ``None``.

    Raises:
        ValueError: If ``orientation`` is not one of the supported values.
    """
    outMask4M = None
    outGridS = None
    if orientation == 'coronal':
        outScan3M = np.transpose(scan3M, (2, 1, 0))
        if mask4M is not None:
            outMask4M = np.transpose(mask4M, (2, 1, 0, 3))
        if gridS is not None:
            outGridS = (gridS[2], gridS[1], gridS[0])

    elif orientation == 'sagittal':
        outScan3M = np.transpose(scan3M, (2, 0, 1))
        if mask4M is not None:
            outMask4M = np.transpose(mask4M, (2, 0, 1, 3))
        if gridS is not None:
            outGridS = (gridS[2], gridS[0], gridS[1])

    elif orientation == 'axial':
        # Do nothing
        outScan3M = scan3M
        outMask4M = mask4M
        if gridS is not None:
            outGridS = gridS
    else:
        raise ValueError(f'Transformation to view {orientation} is not supported')

    return outScan3M, outMask4M, outGridS


def affineDiffusion3d(arr, nSteps, dt=0.1, affFlag=True):
    """Affine-invariant mean-curvature-flow smoothing.

    A nonlinear, edge/shape-preserving 3-D smoothing filter (affine-invariant
    mean-curvature flow), with a plain isotropic heat-equation mode
    (``affFlag=False``) as an alternative. Evolves the image via mean/Gaussian
    curvature numerators of an affine flow.

    Args:
        arr (np.ndarray): 3-D image.
        nSteps (int): number of evolution steps (``n_t``).
        dt (float): evolution step size.
        affFlag (bool): True -> affine-invariant flow; False -> linear (heat).

    Returns:
        np.ndarray: smoothed image (float64), nonnegative.
    """
    phi = np.asarray(arr, dtype=np.float64).copy()
    if phi.ndim != 3:
        raise ValueError("affineDiffusion3d takes a 3-D array")
    if min(phi.shape) < 3 or nSteps <= 0:
        return phi
    c_ = (slice(1, -1), slice(1, -1), slice(1, -1))
    for _ in range(int(nSteps)):
        c = phi[c_]
        jp = phi[1:-1, 2:, 1:-1]; jm = phi[1:-1, :-2, 1:-1]   # col +/-
        kp = phi[2:, 1:-1, 1:-1]; km = phi[:-2, 1:-1, 1:-1]   # row +/-
        lp = phi[1:-1, 1:-1, 2:]; lm = phi[1:-1, 1:-1, :-2]   # slice +/-
        pXX = jp - 2 * c + jm
        pYY = kp - 2 * c + km
        pZZ = lp - 2 * c + lm
        if not affFlag:                          # linear (heat) smoothing
            phi[c_] = c + dt * (pXX + pYY + pZZ)
            continue
        pX = 0.5 * (jp - jm)
        pY = 0.5 * (km - kp)                      # note sign (MATLAB k-1 minus k+1)
        pZ = 0.5 * (lm - lp)
        pXY = 0.25 * (-phi[:-2, :-2, 1:-1] + phi[:-2, 2:, 1:-1]
                      + phi[2:, :-2, 1:-1] - phi[2:, 2:, 1:-1])
        pXZ = 0.25 * (phi[1:-1, :-2, 2:] + phi[1:-1, 2:, :-2]
                      - phi[1:-1, :-2, :-2] - phi[1:-1, 2:, 2:])
        pYZ = 0.25 * (phi[:-2, 1:-1, :-2] + phi[2:, 1:-1, 2:]
                      - phi[:-2, 1:-1, 2:] - phi[2:, 1:-1, :-2])
        meanCurvNum = (pX ** 2 * (pYY + pZZ) + pY ** 2 * (pXX + pZZ)
                       + pZ ** 2 * (pXX + pYY)
                       - 2 * (pX * pY * pXY + pX * pZ * pXZ + pY * pZ * pYZ))
        gausCurvNum = (pX ** 2 * (pYY * pZZ - pYZ ** 2)
                       + pY ** 2 * (pXX * pZZ - pXZ ** 2)
                       + pZ ** 2 * (pXX * pYY - pXY ** 2)
                       + 2 * pX * pY * (pXZ * pYZ - pXY * pZZ)
                       + 2 * pY * pZ * (pXY * pXZ - pYZ * pXX)
                       + 2 * pX * pZ * (pXY * pYZ - pXZ * pYY))
        upd = np.sign(meanCurvNum) * np.maximum(0.0, gausCurvNum) ** 0.25
        phi[c_] = np.maximum(0.0, c + dt * upd)
    return phi


def _axisSlices(offset, nIn, nOut):
    """Matching output and input index ranges for a pad (offset > 0) or a
    crop (offset < 0) along one axis."""
    outStart = max(offset, 0)
    inStart = max(-offset, 0)
    n = min(nIn - inStart, nOut - outStart)
    return slice(outStart, outStart + n), slice(inStart, inStart + n)


def _gridStep(coordV):
    """Signed spacing of a coordinate vector (y decreases down the rows)."""
    return np.median(np.diff(coordV)) if len(coordV) > 1 else 1.0


def _padOrCrop3d(scan3M, mask4M, gridS, outputImgSizeV, padVal):
    """Pad and/or crop rows and cols symmetrically, keeping the physical grid.

    Each axis is handled independently: a larger output size pads (with
    ``padVal`` for the scan and 0 for masks) and a smaller one crops. When the
    size difference is odd, the extra row or column goes at the end for padding
    and is removed from the start for cropping, so padding to a size and
    cropping back returns the original array. Slices are padded at the end or
    cropped from the end when ``outputImgSizeV[2]`` differs from the input.

    Returns:
        tuple: (scanOut3M, maskOut4M, (xOutV, yOutV, zOutV)).
    """
    ref = scan3M if scan3M is not None else mask4M
    inRows, inCols, inSlcs = ref.shape[:3]
    outRows, outCols = int(outputImgSizeV[0]), int(outputImgSizeV[1])
    outSlcs = int(outputImgSizeV[2]) if len(outputImgSizeV) > 2 else inSlcs

    # Index in the output of input row/col 0
    rowDiff, colDiff = outRows - inRows, outCols - inCols
    rowOffset = math.ceil(rowDiff / 2) if rowDiff >= 0 else math.floor(rowDiff / 2)
    colOffset = math.ceil(colDiff / 2) if colDiff >= 0 else math.floor(colDiff / 2)
    outR, inR = _axisSlices(rowOffset, inRows, outRows)
    outC, inC = _axisSlices(colOffset, inCols, outCols)
    nSlc = min(inSlcs, outSlcs)

    scanOut3M = None
    if scan3M is not None:
        scanOut3M = np.full((outRows, outCols, outSlcs), padVal, dtype=scan3M.dtype)
        scanOut3M[outR, outC, :nSlc] = scan3M[inR, inC, :nSlc]
    maskOut4M = None
    if mask4M is not None:
        maskOut4M = np.zeros((outRows, outCols, outSlcs, mask4M.shape[3]), dtype=mask4M.dtype)
        maskOut4M[outR, outC, :nSlc, :] = mask4M[inR, inC, :nSlc, :]

    xV, yV, zV = gridS
    xOutV = xV[0] + (np.arange(outCols) - colOffset) * _gridStep(xV)
    yOutV = yV[0] + (np.arange(outRows) - rowOffset) * _gridStep(yV)
    if outSlcs == inSlcs:
        zOutV = zV
    else:
        zOutV = zV[0] + np.arange(outSlcs) * _gridStep(zV)
    return scanOut3M, maskOut4M, (xOutV, yOutV, zOutV)


def _resampledCoords(coordV, nOut):
    """Pixel-centre coordinates after resizing an axis from len(coordV) to nOut
    pixels with skimage.transform.resize."""
    nIn = len(coordV)
    idxV = (np.arange(nOut) + 0.5) * nIn / nOut - 0.5
    return coordV[0] + idxV * _gridStep(coordV)
