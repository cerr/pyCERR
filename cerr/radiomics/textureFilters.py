"""
 This module contains definitions of image texture filters and a wrapper function to apply any of them.
 Supported filters include: "mean", "sobel", "LoG", "gabor", "gabor3d", "laws", "lawsEnergy"
 "rotationInvariantLaws", "rotationInvariantLawsEnergy"
"""
import pywt
import numpy as np
from scipy.signal import convolve2d
from scipy.signal import convolve
from scipy.ndimage import rotate
from cerr.radiomics.preprocess import padScan


def meanFilter(scan3M, kernelSize, absFlag=False):
    """meanFilter
    Returns  mean filter response.

    scan3M     : 3D scan (numpy) arr3ay
    kernelSize  : 1-D array specifying filter dimensions (2D/3D) [numRows, numCols, optional:numSlc]
    absFlag     : Set to true to return mean of absolute intensities (default:false)
    """

    # Generate mean filter kernel
    filt3M = np.ones(kernelSize, like=scan3M)
    filt3M = filt3M / np.sum(filt3M)

    # Support absolute mean (e.g. for energy calc.)
    if absFlag:
        scan3M = np.abs(scan3M)

    # Generate filter response
    if len(kernelSize) == 3 and kernelSize[2] != 0:  # 3d
        out3M = convolve(scan3M, filt3M, mode='same')
    elif len(kernelSize) == 2 or kernelSize[2] == 0:  # 2d
        out3M = np.empty_like(scan3M)
        for slc in range(scan3M.shape[2]):
            out3M[:, :, slc] = convolve2d(scan3M[:, :, slc], filt3M, mode='same', boundary='fill', fillvalue=0)

    return out3M


def sobelFilter(scan3M):
    """sobelFilter
    Returns  Sobel filter response.

    scan3M      : 3D scan (numpy) array
    """

    fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Convolve with Sobel Filters
    gx3M = convolve2d(scan3M, fx, mode='same')  # Horizontal gradient
    gy3M = convolve2d(scan3M, fy, mode='same')  # Vertical gradient

    # Calculate Gradient Magnitude
    outMag3M = np.sqrt(gx3M ** 2 + gy3M ** 2)

    # Calculate Gradient Direction
    outDiR3M = np.arctan2(gy3M, gx3M)

    return outMag3M, outDiR3M


def LoGFilter(scan3M, sigmaV, cutoffV, voxelSizeV):
    """LoGFilter
    Returns IBSI-compatible Laplacian of Gaussian filter response

    scan3M      : 3D scan (numpy) array
    sigmaV      : 1-D numpy array of Gaussian smoothing widths
                  [sigmaRows,sigmaCols,sigmaSlc] in mm.
    cutoffV     : 1-D numpy array of filter cutoffs [cutOffRow, cutOffCol cutOffSlc] in mm. Filter size = 2.*cutoffV+1
    voxelSizeV  : 1-D numpy array [dx, dy, dz] of scan voxel dimensions in mm.
    """

    # Convert from physical to voxel units
    nDims = len(cutoffV)  # Determnines if 2d or 3d
    cutoffV = np.round(cutoffV / voxelSizeV[:nDims]).astype(int)
    sigmaV = sigmaV / voxelSizeV[:nDims]
    filtSizeV = np.floor(2 * cutoffV + 1).astype(int)

    if len(sigmaV) == 3:  # 3D filter
        # Calculate filter weights
        x, y, z = np.meshgrid(
            np.arange(-cutoffV[1], cutoffV[1] + 1),
            np.arange(-cutoffV[0], cutoffV[0] + 1),
            np.arange(-cutoffV[2], cutoffV[2] + 1)
        )
        xSig2 = sigmaV[1] ** 2
        ySig2 = sigmaV[0] ** 2
        zSig2 = sigmaV[2] ** 2
        h = np.exp(-(x ** 2 / (2 * xSig2) + y ** 2 / (2 * ySig2) + z ** 2 / (2 * zSig2)))
        h[h < np.finfo(float).eps * h.max()] = 0
        sumH = h.sum()
        if sumH != 0:
            h = h / sumH
        h1 = h * (x ** 2 / xSig2 ** 2 + y ** 2 / ySig2 ** 2 + z ** 2 / zSig2 ** 2 - 1 / xSig2 - 1 / ySig2 - 1 / zSig2)
        h = h1 - h1.sum() / np.prod(filtSizeV)  # Shift to ensure sum result=0 on homogeneous regions
        # Apply LoG filter
        out3M = convolve(scan3M, h, mode='same')

    elif len(sigmaV) == 2:  # 2D
        # Calculate filter weights
        x, y = np.meshgrid(
            np.arange(-cutoffV[1], cutoffV[1] + 1),
            np.arange(-cutoffV[0], cutoffV[0] + 1)
        )
        xSig2 = sigmaV[1] ** 2
        ySig2 = sigmaV[0] ** 2
        h = np.exp(-(x ** 2 / (2 * xSig2) + y ** 2 / (2 * ySig2)))
        h[h < np.finfo(float).eps * h.max()] = 0
        sumH = h.sum()
        if sumH != 0:
            h = h / sumH
        h1 = h * (x ** 2 / xSig2 ** 2 + y ** 2 / ySig2 ** 2 - 1 / xSig2 - 1 / ySig2)
        h = h1 - h1.sum() / np.prod(filtSizeV)  # Shift to ensure sum result=0 on homogeneous regions
        # Apply LoG filter
        out3M = np.zeros_like(scan3M)
        for slcNum in range(scan3M.shape[2]):
            out3M[:, :, slcNum] = convolve(scan3M[:, :, slcNum], h, mode='same')

    return out3M


def gaborFilter(scan3M, sigma, wavelength, gamma, thetaV, aggS=None, radius=None, paddingV=None):
    """gaborFilter
    Returns 2D Gabor filter response (IBSI-compatible)

    scan3M     : 3D scan (numpy) array
    sigma       : Std. dev. of Gaussian envelope (no. voxels)
    lambda      : Wavelength (no. voxels)
    gamma       : Spatial aspect ratio
    thetaV     : Vector of orientations (degrees)
    --- Optional---
    aggS        : Parameter dictionary for averaging responses across orientations
    radius      : Kernel radius in voxels [nRows nCols]
    paddingV    : Amount of paddingV applied to scan3M in voxels [nRows nCols]
    """

    # Convert input orientation(s) to list
    if thetaV.ndim == 0:
        thetaV = (thetaV,)

    # Check for user-input radius
    scanSizeV = np.shape(scan3M)
    if radius is None:
        # Otherwise, use default suggestion (IBSI)
        inPlaneSizeV = np.array(scanSizeV[:2]) - 2 * np.array(paddingV[:2])
        evenIdxV = inPlaneSizeV % 2 == 0
        inPlaneSizeV[evenIdxV] = inPlaneSizeV[evenIdxV] + 1
        radius = np.floor(inPlaneSizeV / 2)

    # Get filter scale along x, y axes
    sigmaX = float(sigma)
    sigmaY = sigma / gamma

    # Define filter extents
    d = 4
    x, y = np.meshgrid(np.arange(-radius[1], radius[1] + 1), np.arange(-radius[0], radius[0] + 1))

    # Loop over input orientations
    outS = dict()
    gaborEvenFilters = dict()
    gaborOddFilters = dict()
    for theta in thetaV:
        # Orient grid
        xTheta = x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))
        yTheta = x * np.sin(np.deg2rad(theta)) - y * np.cos(np.deg2rad(theta))

        # Compute filter coefficients
        hGaussian = np.exp(-0.5 * (xTheta ** 2 / sigmaX ** 2 + yTheta ** 2 / sigmaY ** 2))
        hGaborEven = hGaussian * np.cos(2 * np.pi * xTheta / wavelength)
        hGaborOdd = hGaussian * np.sin(2 * np.pi * xTheta / wavelength)
        h = hGaborEven + 1j * hGaborOdd

        # Apply slice-wise
        out3M = np.zeros_like(scan3M)
        for slcNum in range(scanSizeV[2]):
            scanM = scan3M[:, :, slcNum]
            outM = convolve2d(scanM, h, mode='same', boundary='fill')
            # Return modulus
            out3M[:, :, slcNum] = np.abs(outM)
        fieldName = f'gabor_{str(theta)}'
        outS[fieldName] = out3M
        gaborEvenFilters[fieldName] = hGaborEven
        gaborOddFilters[fieldName] = hGaborOdd

    # Aggregate responses across orientations
    gaborThetas = dict()
    if len(thetaV) > 1:
        if 'OrientationAggregation' in aggS:
            gaborThetas4M = [filt_response3M for theta, filt_response3M in outS.items()]
            aggMethod = aggS['OrientationAggregation']
            if aggMethod == 'average':
                gaborAggThetaS3M = np.mean(gaborThetas4M, axis=0)
            elif aggMethod == 'max':
                gaborAggThetaS3M = np.max(gaborThetas4M, axis=0)
            elif aggMethod == 'std':
                gaborAggThetaS3M = np.std(gaborThetas4M, axis=0)

            angleStr = '_'.join(map(lambda x: str(x).replace('.', 'p').replace('-', 'M'), thetaV))
            fieldName = f'gabor_{angleStr}_{aggMethod}'
            # if len(fieldName) > 39:
            #    fieldName = fieldName[:39]
            gaborThetas[fieldName] = gaborAggThetaS3M
            fieldName = [fieldName]
    else:
        gaborThetas = outS

    # Return results as n-d arrays/dicts for single/multiple input orientations
    hGabor = dict()
    hGabor['even'] = gaborEvenFilters
    hGabor['odd'] = gaborOddFilters

    return gaborThetas, hGabor


def gaborFilter3d(scan3M, sigma, wavelength, gamma, thetaV, aggS, radius=None, paddingV=None):
    """gaborFilter3d
    Returns Gabor filter responses aggregated across the 3 orthogonal planes (IBSI-compatible)

    scan3M     : 3D scan (numpy) array
    sigma      : Std. dev. of Gaussian envelope (no. voxels)
    lambda     : Wavelength (no. voxels)
    gamma      : Spatial aspect ratio
    thetaV     : List of orientations (degrees)
    aggS       : Parameter dictionary for aggregation of responses across orientations and/or planes.
    --- Optional---
    radius      : Kernel radius in voxels [nRows nCols]
    paddingV     : Amount of paddingV applied to scan3M in voxels [nRows nCols]
    """

    if thetaV.ndim == 0:
        thetaV = (thetaV,)

    # Loop over image planes
    planes = ['axial', 'sagittal', 'coronal']
    gaborThetas = dict()
    hGaborPlane = dict()

    # Check for user-input radius
    scanSizeV = np.shape(scan3M)
    if radius is None:
        # Otherwise, use default suggestion (IBSI)
        inPlaneSizeV = np.array(scanSizeV[:2]) - 2 * np.array(paddingV[:2])
        evenIdxV = inPlaneSizeV % 2 == 0
        inPlaneSizeV[evenIdxV] = inPlaneSizeV[evenIdxV] + 1
        radius = np.floor(inPlaneSizeV / 2)

    gaborThetaPlanes = dict()
    for nPlane in range(len(planes)):
        plane = planes[nPlane].lower()
        # Flip scan
        if plane == 'axial':
            pass  # do nothing
        elif plane == 'sagittal':
            scan3M = np.transpose(scan3M, (2, 0, 1))
        elif plane == 'coronal':
            scan3M = np.transpose(scan3M, (2, 1, 0))

        # Apply filter
        gaborThetas, hGabor = gaborFilter(scan3M, sigma, wavelength, gamma, thetaV, aggS, radius, paddingV)
        for fieldName in gaborThetas.keys():
            gaborThetaPlanes[(plane + '_') + fieldName] = gaborThetas[fieldName]

        # Re-orient results for cross-plane aggregation
        for fieldName in gaborThetaPlanes.keys():
            gabor_plane3M = gaborThetaPlanes[fieldName]
            if plane == 'axial':
                # do nothing
                pass
            elif plane == 'sagittal':
                gabor_plane3M = np.transpose(gabor_plane3M, (1, 2, 0))
                scan3M = np.transpose(scan3M, (1, 2, 0))
            elif plane == 'coronal':
                gabor_plane3M = np.transpose(gabor_plane3M, (2, 1, 0))
                scan3M = np.transpose(scan3M, (2, 1, 0))

        gaborThetaPlanes[fieldName] = gabor_plane3M
        hGaborPlane[plane] = hGabor

    # Gather results from common settings
    settings = list(gaborThetaPlanes.keys())
    commonSettings = [val.replace('axial_', '').replace('sagittal_', '').replace('coronal_', '') for val in settings]
    uqSettings = np.unique(commonSettings)

    # Aggregate results across orthogonal planes
    aggMethod = None
    gaborOut = dict()
    if 'PlaneAggregation' in aggS:
        aggMethod = aggS['PlaneAggregation']

        for key in range(len(uqSettings)):
            matchIdxV = [idx for idx, val in enumerate(commonSettings) if val == uqSettings[key]]
            matchFields = [settings[match_idx] for match_idx in matchIdxV]
            if type(matchFields) is not list:
                matchFields = [matchFields]
            gaborPlanes4M = getMatchFields(gaborThetaPlanes, *matchFields)

            if aggMethod == 'average':
                gabor3M = np.mean(gaborPlanes4M, axis=0)
            elif aggMethod == 'max':
                gabor3M = np.max(gaborPlanes4M, axis=0)
            elif aggMethod == 'std':
                gabor3M = np.std(gaborPlanes4M, axis=0)

        if len(uqSettings) > 1:
            planesStr = '_'.join(planes)
            fieldName = f'{uqSettings[key]}_{planesStr}_{aggMethod}'
            gaborOut[fieldName] = gabor3M
        else:
            gaborOut[uqSettings[key]] = gabor3M
    else:
        gaborOut = gaborThetas

    return gaborOut, hGaborPlane


# Get 3D Laws' filter kernel (IBSI-compatible)
def get3dLawsMask(x, y, z):
    convedM = np.outer(y, x)
    numRows, numCols = convedM.shape
    conved3M = np.zeros((numRows, numCols, len(z)), dtype='float32')
    for i in range(numRows):
        conved3M[i, :, :] = np.outer(convedM[i, :], z)
    return conved3M


def getLawsMasks(direction='all', filterType='all', normFlag=False):
    """getLawsMasks
    Return Laws filter kernels

    direction   : '2d', '3d' or 'All'
    filterType  : '3', '5', 'all', or a combination of any 2 (if 2d) or 3 (if 3d) of E3, L3, S3, E5, L5, S5.
    normFlag   :  0 - no normalization (default) 1 - normalize  ( Normalization ensures average pixel in filtered image
                 is as bright as the average pixel in the original image)
    """

    filterType = filterType.upper()

    # Define 1-D filter kernels
    L3 = np.array([1, 2, 1], dtype=float)
    E3 = np.array([-1, 0, 1], dtype=float)
    S3 = np.array([-1, 2, -1], dtype=float)
    L5 = np.array([1, 4, 6, 4, 1], dtype=float)
    E5 = np.array([-1, -2, 0, 2, 1], dtype=float)
    S5 = np.array([-1, 0, 2, 0, -1], dtype=float)
    R5 = np.array([1, -4, 6, -4, 1], dtype=float)
    W5 = np.array([-1, 2, 0, -2, 1], dtype=float)

    if normFlag:
        L3 /= np.sqrt(6)
        E3 /= np.sqrt(2)
        S3 /= np.sqrt(6)
        L5 /= np.sqrt(70)
        E5 /= np.sqrt(10)
        S5 /= np.sqrt(6)
        R5 /= np.sqrt(70)
        W5 /= np.sqrt(10)

    lawsMasks = {}

    if filterType not in ['3', '5', 'all']:
        f1 = eval(filterType[0:2])
        f2 = eval(filterType[2:4])
        if len(filterType) == 4:  # 2d
            lawsMasks[filterType] = np.outer(f2, f1)
        elif len(filterType) == 6:
            f3 = eval(filterType[4:6])
            lawsMasks[filterType] = get3dLawsMask(f1, f2, f3)
    else:
        if filterType in ['3', 'all'] and direction in ['2d', 'all']:
            # 2-d (length 3)
            lawsMasks['L3E3'] = np.outer(L3, E3)
            lawsMasks['L3S3'] = np.outer(L3, S3)

            lawsMasks['E3E3'] = np.outer(E3, E3)
            lawsMasks['E3L3'] = np.outer(E3, L3)
            lawsMasks['E3S3'] = np.outer(E3, S3)

            lawsMasks['S3S3'] = np.outer(S3, S3)
            lawsMasks['S3L3'] = np.outer(S3, L3)
            lawsMasks['S3E3'] = np.outer(S3, E3)

        if type in ['5', 'all'] and direction in ['2d', 'all']:
            # 2-d (length 5)
            lawsMasks['L5L5'] = np.outer(L5, L5)
            lawsMasks['L5E5'] = np.outer(L5, E5)
            lawsMasks['L5S5'] = np.outer(L5, S5)
            lawsMasks['L5R5'] = np.outer(L5, R5)
            lawsMasks['L5W5'] = np.outer(L5, W5)

            lawsMasks['E5E5'] = np.outer(E5, E5)
            lawsMasks['E5L5'] = np.outer(E5, L5)
            lawsMasks['E5S5'] = np.outer(E5, S5)
            lawsMasks['E5R5'] = np.outer(E5, R5)
            lawsMasks['E5W5'] = np.outer(E5, W5)

            lawsMasks['S5S5'] = np.outer(S5, S5)
            lawsMasks['S5L5'] = np.outer(S5, L5)
            lawsMasks['S5E5'] = np.outer(S5, E5)
            lawsMasks['S5R5'] = np.outer(S5, R5)
            lawsMasks['S5W5'] = np.outer(S5, W5)

            lawsMasks['R5R5'] = np.outer(R5, R5)
            lawsMasks['R5S5'] = np.outer(R5, S5)
            lawsMasks['R5L5'] = np.outer(R5, L5)
            lawsMasks['R5E5'] = np.outer(R5, E5)
            lawsMasks['R5W5'] = np.outer(R5, W5)

            lawsMasks['W5W5'] = np.outer(W5, W5)
            lawsMasks['W5R5'] = np.outer(W5, R5)
            lawsMasks['W5S5'] = np.outer(W5, S5)
            lawsMasks['W5L5'] = np.outer(W5, L5)
            lawsMasks['W5E5'] = np.outer(W5, E5)

        if type in ['3', 'all'] and direction in ['3d', 'all']:
            # 3-d (length 3)
            lawsMasks['E3E3E3'] = get3dLawsMask(E3, E3, E3)
            lawsMasks['E3E3L3'] = get3dLawsMask(E3, E3, L3)
            lawsMasks['E3E3S3'] = get3dLawsMask(E3, E3, S3)
            lawsMasks['E3L3E3'] = get3dLawsMask(E3, L3, E3)
            lawsMasks['E3L3L3'] = get3dLawsMask(E3, L3, L3)
            lawsMasks['E3L3S3'] = get3dLawsMask(E3, L3, S3)
            lawsMasks['E3S3E3'] = get3dLawsMask(E3, S3, E3)
            lawsMasks['E3S3L3'] = get3dLawsMask(E3, S3, L3)
            lawsMasks['E3S3S3'] = get3dLawsMask(E3, S3, S3)
            lawsMasks['L3E3E3'] = get3dLawsMask(L3, E3, E3)
            lawsMasks['L3E3L3'] = get3dLawsMask(L3, E3, L3)
            lawsMasks['L3E3S3'] = get3dLawsMask(L3, E3, S3)
            lawsMasks['L3L3E3'] = get3dLawsMask(L3, L3, E3)
            lawsMasks['L3L3L3'] = get3dLawsMask(L3, L3, L3)
            lawsMasks['L3L3S3'] = get3dLawsMask(L3, L3, S3)
            lawsMasks['L3S3E3'] = get3dLawsMask(L3, S3, E3)
            lawsMasks['L3S3L3'] = get3dLawsMask(L3, S3, L3)
            lawsMasks['L3S3S3'] = get3dLawsMask(L3, S3, S3)
            lawsMasks['S3E3E3'] = get3dLawsMask(S3, E3, E3)
            lawsMasks['S3E3L3'] = get3dLawsMask(S3, E3, L3)
            lawsMasks['S3E3S3'] = get3dLawsMask(S3, E3, S3)
            lawsMasks['S3L3E3'] = get3dLawsMask(S3, L3, E3)
            lawsMasks['S3L3L3'] = get3dLawsMask(S3, L3, E3)
            lawsMasks['S3L3S3'] = get3dLawsMask(S3, L3, S3)
            lawsMasks['S3S3E3'] = get3dLawsMask(S3, S3, E3)
            lawsMasks['S3S3L3'] = get3dLawsMask(S3, S3, L3)
            lawsMasks['S3S3S3'] = get3dLawsMask(S3, S3, S3)

        if type in ['5', 'all'] and direction in ['3d', 'all']:
            # 3-d (Length 5)
            lawsMasks['L5L5L5'] = get3dLawsMask(L5, L5, L5)
            lawsMasks['L5L5E5'] = get3dLawsMask(L5, L5, E5)
            lawsMasks['L5L5S5'] = get3dLawsMask(L5, L5, S5)
            lawsMasks['L5L5R5'] = get3dLawsMask(L5, L5, R5)
            lawsMasks['L5L5W5'] = get3dLawsMask(L5, L5, W5)
            lawsMasks['L5E5L5'] = get3dLawsMask(L5, E5, L5)
            lawsMasks['L5E5E5'] = get3dLawsMask(L5, E5, E5)
            lawsMasks['L5E5S5'] = get3dLawsMask(L5, E5, S5)
            lawsMasks['L5E5R5'] = get3dLawsMask(L5, E5, R5)
            lawsMasks['L5E5W5'] = get3dLawsMask(L5, E5, W5)
            lawsMasks['L5S5L5'] = get3dLawsMask(L5, S5, L5)
            lawsMasks['L5S5E5'] = get3dLawsMask(L5, S5, E5)
            lawsMasks['L5S5S5'] = get3dLawsMask(L5, S5, S5)
            lawsMasks['L5S5R5'] = get3dLawsMask(L5, S5, R5)
            lawsMasks['L5S5W5'] = get3dLawsMask(L5, S5, W5)
            lawsMasks['L5R5L5'] = get3dLawsMask(L5, R5, L5)
            lawsMasks['L5R5E5'] = get3dLawsMask(L5, R5, E5)
            lawsMasks['L5R5S5'] = get3dLawsMask(L5, R5, S5)
            lawsMasks['L5R5R5'] = get3dLawsMask(L5, R5, R5)
            lawsMasks['L5R5W5'] = get3dLawsMask(L5, R5, W5)
            lawsMasks['L5W5L5'] = get3dLawsMask(L5, W5, L5)
            lawsMasks['L5W5E5'] = get3dLawsMask(L5, W5, E5)
            lawsMasks['L5W5S5'] = get3dLawsMask(L5, W5, S5)
            lawsMasks['L5W5R5'] = get3dLawsMask(L5, W5, R5)
            lawsMasks['L5W5W5'] = get3dLawsMask(L5, W5, W5)
            lawsMasks['E5L5L5'] = get3dLawsMask(E5, L5, L5)
            lawsMasks['E5L5E5'] = get3dLawsMask(E5, L5, E5)
            lawsMasks['E5L5S5'] = get3dLawsMask(E5, L5, S5)
            lawsMasks['E5L5R5'] = get3dLawsMask(E5, L5, R5)
            lawsMasks['E5L5W5'] = get3dLawsMask(E5, L5, W5)
            lawsMasks['E5E5L5'] = get3dLawsMask(E5, E5, L5)
            lawsMasks['E5E5E5'] = get3dLawsMask(E5, E5, E5)
            lawsMasks['E5E5S5'] = get3dLawsMask(E5, E5, S5)
            lawsMasks['E5E5R5'] = get3dLawsMask(E5, E5, R5)
            lawsMasks['E5E5W5'] = get3dLawsMask(E5, E5, W5)
            lawsMasks['E5S5L5'] = get3dLawsMask(E5, S5, L5)
            lawsMasks['E5S5E5'] = get3dLawsMask(E5, S5, E5)
            lawsMasks['E5S5S5'] = get3dLawsMask(E5, S5, S5)
            lawsMasks['E5S5R5'] = get3dLawsMask(E5, S5, R5)
            lawsMasks['E5S5W5'] = get3dLawsMask(E5, S5, W5)
            lawsMasks['E5R5L5'] = get3dLawsMask(E5, R5, L5)
            lawsMasks['E5R5E5'] = get3dLawsMask(E5, R5, E5)
            lawsMasks['E5R5S5'] = get3dLawsMask(E5, R5, S5)
            lawsMasks['E5R5R5'] = get3dLawsMask(E5, R5, R5)
            lawsMasks['E5R5W5'] = get3dLawsMask(E5, R5, W5)
            lawsMasks['E5W5L5'] = get3dLawsMask(E5, W5, L5)
            lawsMasks['E5W5E5'] = get3dLawsMask(E5, W5, E5)
            lawsMasks['E5W5S5'] = get3dLawsMask(E5, W5, S5)
            lawsMasks['E5W5R5'] = get3dLawsMask(E5, W5, R5)
            lawsMasks['E5W5W5'] = get3dLawsMask(E5, W5, W5)
            lawsMasks['S5L5L5'] = get3dLawsMask(S5, L5, L5)
            lawsMasks['S5L5E5'] = get3dLawsMask(S5, L5, E5)
            lawsMasks['S5L5S5'] = get3dLawsMask(S5, L5, S5)
            lawsMasks['S5L5R5'] = get3dLawsMask(S5, L5, R5)
            lawsMasks['S5L5W5'] = get3dLawsMask(S5, L5, W5)
            lawsMasks['S5E5L5'] = get3dLawsMask(S5, E5, L5)
            lawsMasks['S5E5E5'] = get3dLawsMask(S5, E5, E5)
            lawsMasks['S5E5S5'] = get3dLawsMask(S5, E5, S5)
            lawsMasks['S5E5R5'] = get3dLawsMask(S5, E5, R5)
            lawsMasks['S5E5W5'] = get3dLawsMask(S5, E5, W5)
            lawsMasks['S5S5L5'] = get3dLawsMask(S5, S5, L5)
            lawsMasks['S5S5E5'] = get3dLawsMask(S5, S5, E5)
            lawsMasks['S5S5S5'] = get3dLawsMask(S5, S5, S5)
            lawsMasks['S5S5R5'] = get3dLawsMask(S5, S5, R5)
            lawsMasks['S5S5W5'] = get3dLawsMask(S5, S5, W5)
            lawsMasks['S5R5L5'] = get3dLawsMask(S5, R5, L5)
            lawsMasks['S5R5E5'] = get3dLawsMask(S5, R5, E5)
            lawsMasks['S5R5S5'] = get3dLawsMask(S5, R5, S5)
            lawsMasks['S5R5R5'] = get3dLawsMask(S5, R5, R5)
            lawsMasks['S5R5W5'] = get3dLawsMask(S5, R5, W5)
            lawsMasks['S5W5L5'] = get3dLawsMask(S5, W5, L5)
            lawsMasks['S5W5E5'] = get3dLawsMask(S5, W5, E5)
            lawsMasks['S5W5S5'] = get3dLawsMask(S5, W5, S5)
            lawsMasks['S5W5R5'] = get3dLawsMask(S5, W5, R5)
            lawsMasks['S5W5W5'] = get3dLawsMask(S5, W5, W5)
            lawsMasks['R5L5L5'] = get3dLawsMask(R5, L5, L5)
            lawsMasks['R5L5E5'] = get3dLawsMask(R5, L5, E5)
            lawsMasks['R5L5S5'] = get3dLawsMask(R5, L5, S5)
            lawsMasks['R5L5R5'] = get3dLawsMask(R5, L5, R5)
            lawsMasks['R5L5W5'] = get3dLawsMask(R5, L5, W5)
            lawsMasks['R5E5L5'] = get3dLawsMask(R5, E5, L5)
            lawsMasks['R5E5E5'] = get3dLawsMask(R5, E5, E5)
            lawsMasks['R5E5S5'] = get3dLawsMask(R5, E5, S5)
            lawsMasks['R5E5R5'] = get3dLawsMask(R5, E5, R5)
            lawsMasks['R5E5W5'] = get3dLawsMask(R5, E5, W5)
            lawsMasks['R5S5L5'] = get3dLawsMask(R5, S5, L5)
            lawsMasks['R5S5E5'] = get3dLawsMask(R5, S5, E5)
            lawsMasks['R5S5S5'] = get3dLawsMask(R5, S5, S5)
            lawsMasks['R5S5R5'] = get3dLawsMask(R5, S5, R5)
            lawsMasks['R5S5W5'] = get3dLawsMask(R5, S5, W5)
            lawsMasks['R5R5L5'] = get3dLawsMask(R5, R5, L5)
            lawsMasks['R5R5E5'] = get3dLawsMask(R5, R5, E5)
            lawsMasks['R5R5S5'] = get3dLawsMask(R5, R5, S5)
            lawsMasks['R5R5R5'] = get3dLawsMask(R5, R5, R5)
            lawsMasks['R5R5W5'] = get3dLawsMask(R5, R5, W5)
            lawsMasks['R5W5L5'] = get3dLawsMask(R5, W5, L5)
            lawsMasks['R5W5E5'] = get3dLawsMask(R5, W5, E5)
            lawsMasks['R5W5S5'] = get3dLawsMask(R5, W5, S5)
            lawsMasks['R5W5R5'] = get3dLawsMask(R5, W5, R5)
            lawsMasks['R5W5W5'] = get3dLawsMask(R5, W5, W5)
            lawsMasks['W5L5L5'] = get3dLawsMask(W5, L5, L5)
            lawsMasks['W5L5E5'] = get3dLawsMask(W5, L5, E5)
            lawsMasks['W5L5S5'] = get3dLawsMask(W5, L5, S5)
            lawsMasks['W5L5R5'] = get3dLawsMask(W5, L5, R5)
            lawsMasks['W5L5W5'] = get3dLawsMask(W5, L5, W5)
            lawsMasks['W5E5L5'] = get3dLawsMask(W5, E5, L5)
            lawsMasks['W5E5E5'] = get3dLawsMask(W5, E5, E5)
            lawsMasks['W5E5S5'] = get3dLawsMask(W5, E5, S5)
            lawsMasks['W5E5R5'] = get3dLawsMask(W5, E5, R5)
            lawsMasks['W5E5W5'] = get3dLawsMask(W5, E5, W5)
            lawsMasks['W5S5L5'] = get3dLawsMask(W5, S5, L5)
            lawsMasks['W5S5E5'] = get3dLawsMask(W5, S5, E5)
            lawsMasks['W5S5S5'] = get3dLawsMask(W5, S5, S5)
            lawsMasks['W5S5R5'] = get3dLawsMask(W5, S5, R5)
            lawsMasks['W5S5W5'] = get3dLawsMask(W5, S5, W5)
            lawsMasks['W5R5L5'] = get3dLawsMask(W5, R5, L5)
            lawsMasks['W5R5E5'] = get3dLawsMask(W5, R5, E5)
            lawsMasks['W5R5S5'] = get3dLawsMask(W5, R5, S5)
            lawsMasks['W5R5R5'] = get3dLawsMask(W5, R5, R5)
            lawsMasks['W5R5W5'] = get3dLawsMask(W5, R5, W5)
            lawsMasks['W5W5L5'] = get3dLawsMask(W5, W5, L5)
            lawsMasks['W5W5E5'] = get3dLawsMask(W5, W5, E5)
            lawsMasks['W5W5S5'] = get3dLawsMask(W5, W5, S5)
            lawsMasks['W5W5R5'] = get3dLawsMask(W5, W5, R5)
            lawsMasks['W5W5W5'] = get3dLawsMask(W5, W5, W5)

    return lawsMasks


def lawsFilter(scan3M, direction, filterDim, normFlag):
    """lawsFilter
    Return Laws' filter response (IBSI-compatible)

    scan3M    : 3D scan (numpy) array
    direction  : '2d', '3d' or 'All'
    filterDim : '3', '5', 'all', or a combination of any 2 (if 2d) or 3 (if 3d) of E3, L3, S3, E5, L5, S5.
    normFlag  : False - no normalization (default) or True - normalize  ( Normalization ensures average pixel in filtered image
                 is as bright as the average pixel in the original image)
    """

    if isinstance(filterDim, (int, float)):
        filterDim = str(filterDim)

    # Get Laws kernel(s)
    lawsMasks = getLawsMasks(direction, filterDim, normFlag)

    # Compute features
    fieldNames = list(lawsMasks.keys())
    num_features = len(fieldNames)

    out_dict = {}

    for i in range(num_features):
        filter_weights = lawsMasks[fieldNames[i]]
        response3M = convolve(scan3M, lawsMasks[fieldNames[i]], mode='same')
        out_dict[fieldNames[i]] = response3M

    return out_dict


def energyFilter(tex3M, texPadSizeV, energyKernelSizeV, energyPadSizeV, energyPadMethod):
    """energyFilter
    Returns local mean of absolute values

    tex3M              : Response map
    texPadSizeV      : Padding applied in computing tex3M
    energyKernelSizeV: Patch size for mean filtering [numRows numCols num_slc] in voxels
    energyPadMethod   : Padding method for mean filter
    energyPadSizeV   : Padding for mean filter [numRows numCols num_slc] in voxels
    """

    origSizeV = tex3M.shape

    # Pad for mean filter
    calcMask3M = np.ones_like(tex3M, dtype=bool)
    calcMask3M[0:texPadSizeV[0], :, :] = False
    calcMask3M[origSizeV[0] - texPadSizeV[0]:origSizeV[0], :, :] = False
    calcMask3M[:, 0:texPadSizeV[1], :] = False
    calcMask3M[:, origSizeV[1] - texPadSizeV[1]:origSizeV[1], :] = False
    calcMask3M[:, :, 0:texPadSizeV[2]] = False
    calcMask3M[:, :, origSizeV[2] - texPadSizeV[2]:origSizeV[2]] = False
    padTexBbox3M, __, __ = padScan(tex3M, calcMask3M, energyPadMethod, energyPadSizeV, True)

    # Apply mean filter
    texEnergyPad3M = meanFilter(padTexBbox3M, energyKernelSizeV, True)
    padResponseSizeV = texEnergyPad3M.shape

    # Remove paddingV
    texEnergy3M = texEnergyPad3M[energyPadSizeV[0]: padResponseSizeV[0] - energyPadSizeV[0],
                  energyPadSizeV[1]: padResponseSizeV[1] - energyPadSizeV[1],
                  energyPadSizeV[2]: padResponseSizeV[2] - energyPadSizeV[2]]

    # Reapply original paddingV
    texEnergyPad3M = np.full(origSizeV, np.nan)
    texEnergyPad3M[texPadSizeV[0]: -texPadSizeV[0],
    texPadSizeV[1]: -texPadSizeV[1],
    texPadSizeV[2]: -texPadSizeV[2]] = texEnergy3M

    texEnergyPad3M[0: texPadSizeV[0], :, :] = tex3M[0: texPadSizeV[0], :, :]
    texEnergyPad3M[-texPadSizeV[0]:, :, :] = tex3M[-texPadSizeV[0]:, :, :]
    texEnergyPad3M[:, 0: texPadSizeV[1], :] = tex3M[:, 0: texPadSizeV[1], :]
    texEnergyPad3M[:, -texPadSizeV[1]:, :] = tex3M[:, -texPadSizeV[1]:, :]
    texEnergyPad3M[:, :, 0: texPadSizeV[2]] = tex3M[:, :, 0: texPadSizeV[2]]
    texEnergyPad3M[:, :, -texPadSizeV[2]:] = tex3M[:, :, -texPadSizeV[2]:]

    return texEnergyPad3M


def lawsEnergyFilter(scan3M, direction, filterDim, normFlag, lawsPadSizeV,
                     energyKernelSizeV, energyPadSizeV, energyPadMethod):
    """lawsEnergyFilter
    Returns local mean of absolute values of laws filter response

   scan3M             : 3D scan (numpy) array
   direction           : '2d', '3d' or 'All'
   filterDim          : '3', '5', 'all', or a combination of any 2 (if 2d) or 3 (if 3d) of E3, L3, S3, E5, L5, S5.
   normFlag           : False - no normalization (default) or True - normalize  ( Normalization ensures average pixel in filtered image
                         is as bright as the average pixel in the original image)
   lawsPadSizeV     :
   energyKernelSizeV:
   energyPadMethod   :
   energyPadSizeV   :
   """

    # Compute Laws filter(s) reponse(s)
    lawMaps = lawsFilter(scan3M, direction, filterDim, normFlag)

    outS = dict()
    # Loop over response maps
    for type in lawMaps.keys():
        # Compute energy
        lawsTex3M = lawMaps[type]
        #origSizeV = lawsTex3M.shape

        lawsEnergyPad3M = energyFilter(lawsTex3M, lawsPadSizeV, energyKernelSizeV, energyPadSizeV,
                                       energyPadMethod)

    out_field = f"{type}_energy"
    outS[out_field] = lawsEnergyPad3M

    return outS


def rotationInvariantLawsFilter(scan3M, direction, filterDim, normFlag, rotS):
    filter = {"lawsFilter": lawsFilter}
    mask3M = np.array([])
    response = rotationInvariantFilt(scan3M, [], filter, direction, filterDim, normFlag, rotS)
    out3M = response[filterDim]

    return out3M


def rotationInvariantLawsEnergyFilter(scan3M, direction, filterDim, normFlag, lawsPadSizeV,
                                      energyKernelSizeV, energyPadSizeV, energyPadMethod, rotS):
    # Compute rotation-invariant Laws response map
    lawsAggTex3m = rotationInvariantLawsFilter(scan3M, direction, filterDim, normFlag, rotS)
    origSizeV = lawsAggTex3m.shape

    # Compute energy on aggregated response
    lawsEnergyAggPad3M = energyFilter(lawsAggTex3m, lawsPadSizeV, energyKernelSizeV, energyPadSizeV,
                                      energyPadMethod)

    return lawsEnergyAggPad3M


def getWaveletSubbands(scan3M, waveletName, level=1, dim='3d'):
    """ getWaveletSubbands
    Copyright (C) 2017-2019 Martin Vallières
    All rights reserved.
    https://github.com/mvallieres/radiomics-develop
    ------------------------------------------------------------------------
    IMPORTANT:
    - THIS FUNCTION IS TEMPORARY AND NEEDS BENCHMARKING. ALSO, IT
    ONLY WORKS WITH AXIAL SCANS FOR NOW. USING DICOM CONVENTIONS(NOT MATLAB).
    - Strategy: 2D transform for each axial slice. Then 1D transform for each
    axial line. I need to find a faster way to do that with 3D convolutions
    of wavelet filters, this is too slow now. Using GPUs would be ideal.
    ------------------------------------------------------------------------
    """

    # Initialization
    if dim not in ['2d', '3d']:
        raise ValueError("Invalid 'dim' value. Supported values are '2d' and '3d'.")

    sizeV = scan3M.shape
    subbands = {}

    # Step 1: Making sure the volume has even size
    if sizeV[0] % 2 == 1:
        scan3M = np.concatenate((scan3M, scan3M[-1][np.newaxis, :, :]), axis=0)
    if sizeV[1] % 2 == 1:
        scan3M = np.concatenate((scan3M, scan3M[:, -1][:, np.newaxis, :]), axis=1)
    if dim == '3d' and sizeV[2] % 2 == 1:
        scan3M = np.concatenate((scan3M, scan3M[:, :, -1][:, :, np.newaxis]), axis=2)

    # Step 2: Compute all sub-bands
    names = []
    if dim == '2d':
        names = ['LL', 'LH', 'HL', 'HH']
    elif dim == '3d':
        names = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']

    # Ensure odd filter dimensions
    loFilt, hiFilt = pywt.Wavelet(waveletName).filter_bank[0]

    # First pass using 2D stationary wavelet transform in axial direction
    for k in range(sizeV[2]):
        coeffs = pywt.swt2(scan3M[:, :, k], wavelet=waveletName, level=level)
        for s, name in enumerate(names):
            subbands[name][:, :, k] = coeffs[s][0][:, :, level]

    # Second pass using 1D stationary wavelet transform for all axial lines
    if dim == '3d':
        for j in range(sizeV[1]):
            for i in range(sizeV[0]):
                for s, name in enumerate(names[:4]):
                    vector = subbands[name][i, j, :]
                    L, H = pywt.swt(vector, wavelet=waveletName, level=level)
                    subbands[name][i, j, :] = L[level]
                for s, name in enumerate(names[4:]):
                    vector = subbands[name][i, j, :]
                    L, H = pywt.swt(vector, wavelet=waveletName, level=level)
                    subbands[name][i, j, :] = L[level]

    # Removing unnecessary data added in step 1
    if sizeV[0] % 2 == 1:
        for name in names:
            subbands[name] = subbands[name][:-1, :, :]
    if sizeV[1] % 2 == 1:
        for name in names:
            subbands[name] = subbands[name][:, :-1, :]
    if dim == '3d' and sizeV[2] % 2 == 1:
        for name in names:
            subbands[name] = subbands[name][:, :, :-1]

    return subbands


def waveletFilter(vol3M, waveType, direction, level):
    if len(direction) == 3:
        dim = '3d'
    elif len(direction) == 2:
        dim = '2d'

    if dim == '3d':
        dir_list = ['All', 'HHH', 'LHH', 'HLH', 'HHL', 'LLH', 'LHL', 'HLL', 'LLL']
    elif dim == '2d':
        dir_list = ['All', 'HH', 'HL', 'LH', 'LL']

    outS = dict()
    if direction == 'All':
        for n in range(1, len(dir_list)):

            out_name = f"{waveType}_{dir_list[n]}".replace('.', '_').replace(' ', '_')
            subbandsS = getWaveletSubbands(vol3M, waveType, level, dim)

            if 'RotationInvariance' in paramS and paramS['RotationInvariance']:
                perm_dir_list = [''.join(p) for p in permutations(dir_list[n])]
                match_dir = f"{perm_dir_list[0]}_{waveType}"
                out3M = subbandsS[match_dir]

                for perm_dir in perm_dir_list[1:]:
                    match_dir = f"{perm_dir}_{waveType}"
                    out3M += subbandsS[match_dir]

                out3M /= len(perm_dir_list)
            else:
                match_dir = f"{dir_list[n]}_{waveType}"
                out3M = subbandsS[match_dir]

            outS[out_name] = out3M

    else:
        out_name = f"{waveType}_{direction}".replace('.', '_').replace(' ', '_')
        subbandsS = getWaveletSubbands(vol3M, waveType, level, dim)

        if 'RotationInvariance' in paramS and paramS['RotationInvariance']:
            perm_dir_list = [''.join(p) for p in permutations(direction)]
            match_dir = f"{perm_dir_list[0]}_{waveType}"
            out3M = subbandsS[match_dir]

            for perm_dir in perm_dir_list[1:]:
                match_dir = f"{perm_dir}_{waveType}"
                out3M += subbandsS[match_dir]

            out3M /= len(perm_dir_list)
        else:
            match_dir = f"{direction}_{waveType}"
            out3M = subbandsS[match_dir]
        outS[out_name] = out3M

    return outS


### Functions for rotation-invariant filtering and pooling

def getMatchFields(dictIn, *args):
    """Return vals in list of dictionaries matching input key"""

    return [dictIn[field] for field in args]


def aggregateRotatedResponses(rotTexture, aggregationMethod):
    """Aggregate textures from all orientations"""

    out_4m = np.stack(rotTexture, axis=0)
    if aggregationMethod == 'avg':
        agg3M = np.mean(out_4m, axis=0)
    elif aggregationMethod == 'max':
        agg3M = np.max(out_4m, axis=0)
    elif aggregationMethod == 'std':
        agg3M = np.std(out_4m, axis=0)

    return agg3M


def rot3d90(arr3M, axis, angle):
    """Rotate 3D array 90 degrees around a specific axis"""

    rotArr3M = rotate(arr3M, angle, axes=(axis % 3, (axis + 1) % 3), reshape=False)
    return rotArr3M


def rotate3dSequence(vol3M, index, sign):
    """Apply pre-defined sequence of right angle rotations to 2D/3D arrays"""

    signedAngle = sign * 90
    switch = {
        0: lambda x: x,
        1: lambda x: rot3d90(x, 2, signedAngle * 1),
        2: lambda x: rot3d90(x, 2, signedAngle * 2),
        3: lambda x: rot3d90(x, 2, signedAngle * 3),
        4: lambda x: rot3d90(rot3d90(x, 1, signedAngle * 1), 3, signedAngle * 1) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 1), 1, signedAngle * 1),
        5: lambda x: rot3d90(rot3d90(x, 1, signedAngle * 1), 3, signedAngle * 3) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 3), 1, signedAngle * 1),
        6: lambda x: rot3d90(x, 1, signedAngle * 1),
        7: lambda x: rot3d90(x, 1, signedAngle * 2),
        8: lambda x: rot3d90(x, 1, signedAngle * 3),
        9: lambda x: rot3d90(rot3d90(x, 2, signedAngle * 1), 3, signedAngle * 3) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 3), 2, signedAngle * 1),
        10: lambda x: rot3d90(rot3d90(x, 2, signedAngle * 1), 3, signedAngle * 2) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 2), 2, signedAngle * 1),
        11: lambda x: rot3d90(rot3d90(x, 2, signedAngle * 1), 3, signedAngle * 1) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 1), 2, signedAngle * 1),
        12: lambda x: rot3d90(rot3d90(x, 2, signedAngle * 1), 1, signedAngle * 2) if signedAngle > 0 else rot3d90(
            rot3d90(x, 1, signedAngle * 2), 2, signedAngle * 1),
        13: lambda x: rot3d90(rot3d90(x, 2, signedAngle * 2), 1, signedAngle * 2) if signedAngle > 0 else rot3d90(
            rot3d90(x, 1, signedAngle * 2), 2, signedAngle * 2),
        14: lambda x: rot3d90(rot3d90(x, 2, signedAngle * 3), 1, signedAngle * 2) if signedAngle > 0 else rot3d90(
            rot3d90(x, 1, signedAngle * 2), 2, signedAngle * 3),
        15: lambda x: rot3d90(rot3d90(x, 1, signedAngle * 3), 3, signedAngle * 1) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 1), 1, signedAngle * 3),
        16: lambda x: rot3d90(rot3d90(x, 1, signedAngle * 3), 3, signedAngle * 2) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 2), 1, signedAngle * 3),
        17: lambda x: rot3d90(rot3d90(x, 1, signedAngle * 3), 3, signedAngle * 3) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 3), 1, signedAngle * 3),
        18: lambda x: rot3d90(rot3d90(x, 2, signedAngle * 2), 3, signedAngle * 1) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 1), 2, signedAngle * 2),
        19: lambda x: rot3d90(rot3d90(x, 2, signedAngle * 3), 3, signedAngle * 1) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 1), 2, signedAngle * 3),
        20: lambda x: rot3d90(x, 3, signedAngle * 1),
        21: lambda x: rot3d90(rot3d90(x, 2, signedAngle * 2), 3, signedAngle * 3) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 3), 2, signedAngle * 2),
        22: lambda x: rot3d90(rot3d90(x, 2, signedAngle * 3), 3, signedAngle * 3) if signedAngle > 0 else rot3d90(
            rot3d90(x, 3, signedAngle * 3), 2, signedAngle * 3),
        23: lambda x: rot3d90(x, 3, signedAngle * 3),
    }

    return switch[index](vol3M)


def flipSequenceForWavelets():
    pass


def rotationInvariantFilt(scan3M, mask3M, filter, *params):
    """Apply filter over a range of orientations"""

    filterType = list(filter.keys())
    filterType = filterType[0].replace(' ', '')
    waveletFlag = 0
    if filterType == 'wavelets':
        waveletFlag = 1

    # Parameters for rotation invariance
    rot = params[-1]
    aggregationMethod = rot['AggregationMethod']
    dim = rot['Dim']
    if waveletFlag:
        numRotations = 4 if dim.lower() == '2d' else 8
    else:
        numRotations = 4 if dim.lower() == '2d' else 24

    # Handle S-I orientation flip for Wavelet filters
    if waveletFlag:
        scan3M = np.flip(scan3M, axis=2)  # FOR IBSI2 compatibility
        if len(mask3M) > 0:
            mask3M = np.flip(mask3M, axis=2)

    # Apply filter at specified orientations
    rotTextureTypes = [{} for _ in range(numRotations)]
    for index in range(1, numRotations + 1):
        if waveletFlag:
            rotScan3M = flipSequenceForWavelets(scan3M, index - 1, 1)
            rotMask3M = mask3M
            if len(mask3M) > 0:
                rotMask3M = flipSequenceForWavelets(mask3M, index - 1, 1)
        else:
            if dim.lower() == '2d':
                rotScan3M = np.rot90(scan3M, k=index - 1)
                rotMask3M = mask3M
                if len(mask3M) > 0:
                    rotMask3M = np.rot90(mask3M, k=index - 1)
            elif dim.lower() == '3d':
                rotScan3M = rotate3dSequence(scan3M, index - 1, 1)
                rotMask3M = mask3M
                if len(mask3M) > 0:
                    rotMask3M = rotate3dSequence(mask3M, index - 1, 1)

        # Apply filter
        filterHandle = filter[filterType]
        rotScan3M = rotScan3M.astype(float)
        filtParams = params[:-1]

        # Return result(s) to input orientation
        filterResult = filterHandle(rotScan3M, *filtParams)
        if isinstance(filterResult, dict):
            for key in filterResult.keys():
                texture3M = filterResult[key]

                if waveletFlag:
                    out3M = flipSequenceForWavelets(texture3M, index - 1, -1)
                    out3M = np.flip(out3M, axis=2)
                else:
                    if dim.lower() == '2d':
                        out3M = np.rot90(texture3M, k=-(index - 1))
                    elif dim.lower() == '3d':
                        out3M = rotate3dSequence(texture3M, index - 1, -1)

                filterResult[key] = out3M
            rotTextureTypes[index - 1] = filterResult
        else:
            if waveletFlag:
                out3M = flipSequenceForWavelets(filterResult, index - 1, -1)
                #out3M = np.flip(rotOut3M, axis=2)
            else:
                if dim.lower() == '2d':
                    filterResult = np.rot90(filterResult, k=-(index - 1))
                elif dim.lower() == '3d':
                    filterResult = rotate3dSequence(filterResult, index - 1, -1)
            rotTextureTypes[index - 1] = filterResult

    # Aggregate responses across orientations
    if isinstance(rotTextureTypes[0], dict):
        aggS = dict()
        for type in rotTextureTypes[0].keys():
            rotTextureType = [rotTextures[type] for rotTextures in rotTextureTypes]
            agg3M = aggregateRotatedResponses(rotTextureType, aggregationMethod)
        aggS[key] = agg3M
    else:
        aggS = aggregateRotatedResponses(rotTextureTypes, aggregationMethod)

    return aggS
