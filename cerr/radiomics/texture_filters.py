"""
 This module contains definitions of image texture filters and a wrapper function to apply any of them.
 Supported filters include: "mean", "sobel", "LoG", "gabor", "gabor3d", "laws", "lawsEnergy"
 "rotationInvariantLaws", "rotationInvariantLawsEnergy"
"""

import numpy as np
import pywt
from itertools import permutations
from scipy.signal import convolve2d
from scipy.signal import convolve
from scipy.ndimage import rotate
from cerr.radiomics.preprocess import dyadUp, padScan, wextend
from cerr.utils.mask import computeBoundingBox


def meanFilter(scan3M, kernelSize, absFlag=False):
    """
    Function to compute patchwise mean on input image using specified kernel size.

    Args:
        scan3M (np.ndarray): Input image matrix.
        kernelSize (np.array): Size of filter kernel [nRow, nCol, nSlc].
        absFlag (bool): [optional, default:False] Flag to use absolute intensities.
    Returns:
        np.ndarray(dtype=float): Mean filter response.

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
            out3M[:, :, slc] = convolve2d(scan3M[:, :, slc],\
                               filt3M, mode='same', boundary='fill', fillvalue=0)

    return out3M


def sobelFilter(scan3M):
    """
    Function to compute gradient magnitude and direction using Sobel filter.

    Args:
        scan3M (np.ndarray): 3D input scan matrix.
    Returns:
        np.ndarray(dtype=float): gradient magnitude
        np.ndarray(dtype=float): gradient direction

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
    """
    Function to apply IBSI standard Laplacian of Gaussian (LoG) filter

    Args:
        scan3M (np.ndarray): 3D scan matrix.
        sigmaV (np.array, 1-D): Gaussian smoothing widths, [sigmaRows,sigmaCols,sigmaSlc] in mm.
        cutoffV (np.array 1-D): Filter cutoffs [cutOffRow, cutOffCol cutOffSlc] in mm.
                                Note: Filter size = 2.*cutoffV+1
        voxelSizeV (np.array, 1-D): Scan voxel dimensions [dx, dy, dz]  in mm.

    Returns:
        np.ndarray(dtype=float): IBSI-compatible Laplacian of Gaussian filter response.
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
        h = np.exp(-(x ** 2 / (2 * xSig2) + y ** 2 / (2 * ySig2) + \
                     z ** 2 / (2 * zSig2)))
        h[h < np.finfo(float).eps * h.max()] = 0
        sumH = h.sum()
        if sumH != 0:
            h = h / sumH
        h1 = h * (x ** 2 / xSig2 ** 2 + y ** 2 / ySig2 ** 2 + \
                  z ** 2 / zSig2 ** 2 - 1 / xSig2 - 1 / ySig2 - 1 / zSig2)
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


def gaborFilter(scan3M, sigma, wavelength, gamma, thetaV,\
                aggS=None, radius=None, paddingV=None):
    """
    Function to apply IBSI standard 2D Gabor filter

    Args:
        scan3M (np.ndarray): 3D scan matrix.
        sigma (float): Std. deviation Gaussian envelope in no. voxels.
        lambda (float): wavelength in no. voxels.
        gamma (float): Spatial aspect ratio
        thetaV (np.array, 1D): Orientations in degrees
        aggS (dict): [optional, default=None] Parameters for averaging responses across orientations.
        radius (np.array(dtype=int)): [optional, default=None] Kernel radius in voxels [nRows nCols].
        paddingV (np.array, 1D): [optional, default=None] Amount of padding applied to scan in voxels [nRows nCols].

    Returns:
        np.ndarray(dtype=float): IBSI-compatible 2D Gabor filter response.

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
    if int(radius[0])==radius[0] and int(radius[1])==radius[1]:
        x, y = np.meshgrid(np.arange(-radius[1], radius[1]+1 ,1),\
                           np.arange(-radius[0], radius[0]+1 ,1))
    elif int(radius[0])==radius[0]:
        x, y = np.meshgrid(np.arange(-radius[1], radius[1] ,1),\
                           np.arange(-radius[0], radius[0]+1 ,1))
    elif int(radius[1])==radius[1]:
        x, y = np.meshgrid(np.arange(-radius[1], radius[1]+1 ,1),\
                           np.arange(-radius[0], radius[0] ,1))
    else:
        x, y = np.meshgrid(np.arange(-radius[1], radius[1] ,1),\
                           np.arange(-radius[0], radius[0] ,1))

    # Loop over input orientations
    outS = dict()
    gaborEvenFilters = dict()
    gaborOddFilters = dict()
    for theta in thetaV:
        # Orient grid
        xTheta = x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))
        yTheta = x * np.sin(np.deg2rad(theta)) - y * np.cos(np.deg2rad(theta))

        # Compute filter coefficients
        hGaussian = np.exp(-0.5 * (xTheta ** 2 / sigmaX ** 2 +\
                                   yTheta ** 2 / sigmaY ** 2))
        hGaborEven = hGaussian * np.cos(2 * np.pi * xTheta / wavelength)
        hGaborOdd = hGaussian * np.sin(2 * np.pi * xTheta / wavelength)
        h = hGaborEven + 1j * hGaborOdd

        # Apply slice-wise
        out3M = np.zeros_like(scan3M)
        for slcNum in range(scanSizeV[2]):
            scanM = scan3M[:, :, slcNum]
            outM = convolve2d(scanM, h, mode='same', boundary='fill', fillvalue=0)
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
            gaborThetas4M = [filt_response3M for theta,\
                             filt_response3M in outS.items()]
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


def gaborFilter3d(scan3M, sigma, wavelength, gamma, thetaV,\
                  aggS, radius=None, paddingV=None):
    """gaborFilter3d
    Function to return Gabor filter responses aggregated across the 3 orthogonal planes
    (IBSI-compatible)

    Args:
        scan3M (np.ndarray): 3D scan array.
        sigma (int): Std. dev. of Gaussian envelope in no. voxels.
        lambda (int): Wavelength in no. voxels.
        gamma (float): Spatial aspect ratio.
        thetaV (list(dtype=float)): Orientations in degrees.
        aggS (dict): Parameters for aggregation of responses across orientations
                     and/or planes.
        radius (np.array, 1D): [optional, default=None] Kernel radii in voxels [nRows, nCols].
        paddingV (np.aray, 1D): [optional, default=None] Amount of padding applied to scan in voxels [nRows nCols].

    Returns:
        gaborOut (dict): Gabor filter responses.
        hGaborPlane (dict): Gabor filter kernels for each plane.

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
        gaborThetas, hGabor = gaborFilter(scan3M, sigma, wavelength,\
                              gamma, thetaV, aggS, radius, paddingV)
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
            matchIdxV = [idx for idx, val in enumerate(commonSettings)\
                         if val == uqSettings[key]]
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


def get3dLawsMask(x, y, z):
    """
    Function to get 3D Laws' filter kernel (IBSI-compatible)

    Args:
        x (np.array, 1D): Supported Laws filter coefficients applied along rows.
        y (np.array, 1D): Supported Laws filter coefficients applied along cols.
        z (np.array, 1D): Supported Laws filter coefficients applied along slices.

    Returns:
        conved3M (np.ndarray): 3D Laws' kernel.
    """
    convedM = np.outer(y, x)
    numRows, numCols = convedM.shape
    conved3M = np.zeros((numRows, numCols, len(z)), dtype='float32')
    for i in range(numRows):
        conved3M[i, :, :] = np.outer(convedM[i, :], z)
    return conved3M


def getLawsMasks(direction='all', filterType='all', normFlag=False):
    """getLawsMasks
    Function to get Laws filter kernels.

    Args:
        direction (string): [optional, default='all'] specifying '2d', '3d' or 'All'
        filterType (string): [optional, default='all']  specifying '3', '5', 'all',
                    or a combination of any 2 (if 2d)
                    or 3 (if 3d) of E3, L3, S3, E5, L5, S5.
        normFlag (bool): [optional, default=False] Flag to normalize filter coefficients
                 when True. Normalization ensures average pixel in filtered image
                 is as bright as the average pixel in the original image.

    Returns:
        lawsMasks (dict): Laws kernels for specified filter types and directions.
    """

    filterType = filterType.upper()

    # Define 1-D filter kernels
    L3 = np.array([1, 2, 1], dtype=np.double)
    E3 = np.array([-1, 0, 1], dtype=np.double)
    S3 = np.array([-1, 2, -1], dtype=np.double)
    L5 = np.array([1, 4, 6, 4, 1], dtype=np.double)
    E5 = np.array([-1, -2, 0, 2, 1], dtype=np.double)
    S5 = np.array([-1, 0, 2, 0, -1], dtype=np.double)
    R5 = np.array([1, -4, 6, -4, 1], dtype=np.double)
    W5 = np.array([-1, 2, 0, -2, 1], dtype=np.double)

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
    """
    Function to compute Laws' filter response (IBSI-compatible)

    Args:
        scan3M (np.ndarray): 3D scan.
        direction (string): specifying '2d', '3d' or 'All'
        filterDim (string): specifying '3', '5', 'all', or a combination of
                    any 2 (if 2d) or 3 (if 3d) of E3, L3, S3, E5, L5, S5.
        normFlag (bool): Flag to normalize filter coefficients if True.
                  Normalization ensures average pixel in filtered image
                  is as bright as the average pixel in the original image)

    Returns:
       lawsOut (dict): Filter responses for specified filter types and directions.

    """

    if isinstance(filterDim, (int, float)):
        filterDim = str(filterDim)

    # Get Laws kernel(s)
    lawsMasks = getLawsMasks(direction, filterDim, normFlag)

    # Compute features
    fieldNames = list(lawsMasks.keys())
    numFeatures = len(fieldNames)

    lawsOut = {}

    for i in range(numFeatures):
        filterWeights = lawsMasks[fieldNames[i]]
        if direction.lower() == '3d':
            response3M = convolve(scan3M, filterWeights, mode='same')
        elif direction.lower() == '2d':
            response3M = np.empty_like(scan3M)
            for slc in range(scan3M.shape[2]):
                response3M[:, :, slc] = convolve2d(scan3M[:, :, slc],\
                                        filterWeights, mode='same')
        lawsOut[fieldNames[i]] = response3M

    return lawsOut


def energyFilter(tex3M, mask3M, texPadFlag, texPadSizeV, texPadMethod,\
                 energyKernelSizeV, energyPadSizeV, energyPadMethod):
    """
    Function to compute energy (local mean of absolute intensities)

    Args:
        tex3M (np.ndarray(dtype=float)): Input filter response map
        mask3M (np.ndarray(dtype=bool)): Processed mask returned by radiomics.preprocess.
        texPadFlag (bool): Flag to indicate if padding was applied to compute
                           Laws filter response.
        texPadSizeV (np.array(dtype=int)): Amount of padding applied to compute tex3M
        texPadMethod (string): Padding method applied prior to compute tex3M
        energyKernelSizeV (np.array(dtype=int)): Patch size used to calculate local energy
                           [numRows numCols num_slc] in voxels.
        energyPadMethod (string): Padding method applied to calculate local energy.
        energyPadSizeV: np.array(dtype=int) for amount padding applied to
                        calculate local energy [numRows numCols num_slc] in voxels.

    Returns:
        texEnergyPad3M (np.ndarray(dtype=float)): Energy filter response.

    """

    # Calc. padding applied
    origSizeV = tex3M.shape
    if not texPadFlag:
        valOrigPadV = [0,0,0,0,0,0]
    elif texPadMethod.lower=='expand':
        minr, maxr, minc, maxc, mins, maxs, __ = computeBoundingBox(mask3M)
        valOrigPadV = [np.min(texPadSizeV[0], minr), np.min(texPadSizeV[0], origSizeV[0]-maxr),\
                   np.min(texPadSizeV[1], minc), np.min(texPadSizeV[1], origSizeV[1]-maxc),\
                   np.min(texPadSizeV[2], mins), np.min(texPadSizeV[2], origSizeV[2]-maxs)]
    else:
        valOrigPadV = [texPadSizeV[0], texPadSizeV[0], texPadSizeV[1], texPadSizeV[1], \
                       texPadSizeV[2], texPadSizeV[2]]

    # Pad for mean filter
    calcMask3M = np.ones_like(tex3M, dtype=bool)
    calcMask3M[0:valOrigPadV[0], :, :] = False
    calcMask3M[origSizeV[0]-valOrigPadV[1]:origSizeV[0], :, :] = False
    calcMask3M[:, 0:valOrigPadV[2], :] = False
    calcMask3M[:, origSizeV[1]-valOrigPadV[3]:origSizeV[1], :] = False
    calcMask3M[:, :, 0:valOrigPadV[4]] = False
    calcMask3M[:, :, origSizeV[2]-valOrigPadV[5]:origSizeV[2]] = False
    padTexBbox3M, outMask3M, extentsV = padScan(tex3M, calcMask3M, energyPadMethod, energyPadSizeV, True)

    # Apply mean filter
    texEnergyPad3M = meanFilter(padTexBbox3M, energyKernelSizeV, True)
    padResponseSizeV = texEnergyPad3M.shape

    # Remove paddingV
    if energyPadMethod.lower=='expand':
        valEnergyPadV = [np.min(energyPadSizeV[0], extentsV[0]),\
                         np.min(energyPadSizeV[0], origSizeV[0]-extentsV[1]),\
                         np.min(energyPadSizeV[1], extentsV[2]),\
                         np.min(energyPadSizeV[1], origSizeV[1]-extentsV[3]),\
                         np.min(energyPadSizeV[2],extentsV[4]),\
                         np.min(energyPadSizeV[2], origSizeV[2]-extentsV[5])]
    else:
        valEnergyPadV = [energyPadSizeV[0],energyPadSizeV[0],energyPadSizeV[1],\
                         energyPadSizeV[1],energyPadSizeV[2],energyPadSizeV[2]]

    texEnergy3M = texEnergyPad3M[valEnergyPadV[0]:padResponseSizeV[0]-valEnergyPadV[1],\
                                 valEnergyPadV[2]:padResponseSizeV[1]-valEnergyPadV[3],\
                                 valEnergyPadV[4]:padResponseSizeV[2]-valEnergyPadV[5]]

    # Reapply original paddingV
    texEnergyPad3M = texEnergy3M
    if texPadFlag:
        texEnergyPad3M = tex3M
        if texPadSizeV[2] == 0:
            texEnergyPad3M[valOrigPadV[0]: origSizeV[0]-valOrigPadV[1],
                       valOrigPadV[2]: origSizeV[1]-valOrigPadV[3],:] = texEnergy3M
        else:
            texEnergyPad3M[valOrigPadV[0]: origSizeV[0]-valOrigPadV[1],
                       valOrigPadV[2]: origSizeV[1]-valOrigPadV[3],
                       valOrigPadV[4]: origSizeV[2]-valOrigPadV[5]] = texEnergy3M

    return texEnergyPad3M


def lawsEnergyFilter(scan3M, mask3M, direction, filterDim, normFlag,\
                     lawsPadFlag, lawsPadSizeV,lawsPadMethod, energyKernelSizeV,\
                     energyPadSizeV, energyPadMethod):
    """Function to compute local mean of absolute values of laws filter response

    Args:
        scan3M (np.ndarray): 3D scan.
        mask3M (np.ndarray(dtype=bool)): 3D mask.
        direction (string): Specifying '2d', '3d' or 'All'.
        filterDim (string: Specifying '3', '5', 'all', or a combination
                    of any 2 (if 2d) or 3 (if 3d) of E3, L3, S3, E5, L5, S5.
        normFlag (bool): Flag to normalize kernel coefficients if set to True.
                  Normalization ensures average pixel in filtered image
                  is as bright as the average pixel in the original image.
        lawsPadFlag (bool): Flag to indicate if padding was applied to compute
                     Laws filter response.
        lawsPadSizeV (np.array(dtype=int)): Amount of padding applied to compute Laws
                 filter response.
        lawsPadMethod (string): Padding method applied to compute Laws filter response.
        energyKernelSizeV (np.array(dtype=int)): Patch size used to calculate local energy
                      [numRows numCols num_slc] in voxels.
        energyPadMethod (np.array(dtype=int)): Padding method applied to
                   calculate local energy.
        energyPadSizeV (np.array(dtype=int)): Amount padding applied to
                   calculate local energy [numRows numCols num_slc] in voxels.

   Returns:
        outS (dict): Filter responses for specified directions and filter types.
   """

    # Compute Laws filter(s) reponse(s)
    lawMaps = lawsFilter(scan3M, direction, filterDim, normFlag)


    outS = dict()
    # Loop over response maps
    for type in lawMaps.keys():
        # Compute energy
        lawsTex3M = lawMaps[type]
        #origSizeV = lawsTex3M.shape

        lawsEnergyPad3M = energyFilter(lawsTex3M, mask3M, lawsPadFlag, lawsPadSizeV, lawsPadMethod, energyKernelSizeV,\
                                       energyPadSizeV, energyPadMethod)

    out_field = f"{type}_energy"
    outS[out_field] = lawsEnergyPad3M

    return outS


def rotationInvariantLawsFilter(scan3M, direction, filterDim, normFlag, rotS):
   """
   Function to return rotation-invariant Laws filter response.

   Args:
        scan3M (np.ndarray): 3D scan.
        direction (string): Specifying '2d', '3d' or 'All'.
        filterDim (string): Specifying '3', '5', 'all', or a combination
                    of any 2 (if 2d) or 3 (if 3d) of E3, L3, S3, E5, L5, S5.
        normFlag (bool): Flag to normalize kernel coefficients if set to True.
                  Normalization ensures average pixel in filtered image
                  is as bright as the average pixel in the original image
        rotS (dict): Parameters for aggregating filter response across
              different orientations

   Returns:
        out3M (np.ndarray(dtype=float)): Laws filter response aggregated
               across orientations as specified.
   """

   filter = {"lawsFilter": lawsFilter}
   response = rotationInvariantFilt(scan3M, [], filter,\
               direction, filterDim, normFlag, rotS)
   out3M = response[filterDim]

   return out3M


def rotationInvariantLawsEnergyFilter(scan3M, mask3M, direction, filterDim,\
                                      normFlag, lawsPadFlag, lawsPadSizeV,\
                                      lawsPadMethod, energyKernelSizeV,\
                                      energyPadSizeV, energyPadMethod, rotS):
    """
       Function to return rotation-invariant Laws energy response.

       Args:
            scan3M (np.ndarray): 3D scan.
            direction (string): Specifying '2d', '3d' or 'All'.
            filterDim (string): Specifying '3', '5', 'all', or a combination
                        of any 2 (if 2d) or 3 (if 3d) of E3, L3, S3, E5, L5, S5.
            normFlag (bool): Flag to normalize kernel coefficients if set to True.
                      Normalization ensures average pixel in filtered image
                      is as bright as the average pixel in the original image
            lawsPadFlag (bool): Flag to indicate if padding was applied to compute
                     Laws filter response.
            lawsPadSizeV (np.array(dtype=int)): Amount of padding applied to compute Laws
                 filter response.
            lawsPadMethod (string): Padding method applied to compute Laws filter response.
            energyKernelSizeV (np.array(dtype=int)): Patch size used to calculate local energy
                      [numRows numCols num_slc] in voxels.
            energyPadMethod (np.array(dtype=int)): Padding method applied to
                   calculate local energy.
            energyPadSizeV (np.array(dtype=int)): Amount padding applied to
                   calculate local energy [numRows numCols num_slc] in voxels.

            rotS (dict): Parameters for aggregating filter response across
                  different orientations

       Returns:
            lawsEnergyAggPad3M (np.ndarray(dtype=float)): Energy of Laws filter response
                                aggregated across orientations as specified.
       """
    # Compute rotation-invariant Laws response map
    lawsAggTex3m = rotationInvariantLawsFilter(scan3M, direction, filterDim, normFlag, rotS)

    # Compute energy on aggregated response
    lawsEnergyAggPad3M = energyFilter(lawsAggTex3m, mask3M, lawsPadFlag,\
                                      lawsPadSizeV, lawsPadMethod,energyKernelSizeV,\
                                      energyPadSizeV, energyPadMethod)

    return lawsEnergyAggPad3M

def wkeep(z, size, first):
    """Keep central segment of signal after convolution"""
    if z.ndim == 1: #1D
        last = first + size - 1
        zkeep = z[first-1:last]
    elif z.ndim == 2: #2D
        last = [first[i] + size[i] - 1 for i in range(2)]
        zkeep = z[first[0]-1:last[0], first[1]-1:last[1]]
    return zkeep


def decomposeLOC(x, lo, hi, first, sizeV):
    # Approximation
    y = convolve2d(x, lo[:,None], mode='full')
    z = convolve2d(y.T, lo[:,None], mode='full').T
    ca = wkeep(z, sizeV, first)

    # Horizontal
    z = convolve2d(y.T, hi[:,None], mode='full')
    ch = wkeep(z, sizeV, first)

    # Vertical
    y = convolve2d(x, hi[None,], mode='full')
    z = convolve2d(y.T, lo[None,], mode='full').T
    cv = wkeep(z, sizeV, first)

    # Diagonal
    z = convolve2d(y.T, hi[None,:], mode='full').T
    cd = wkeep(z, sizeV, first)

    return ca, ch, cv, cd


def swt(sigV, level, loD, hiD):
    """
    Replicates MATLAB's swt 1D behavior using custom filters and symmetric padding.
    Args:
        sigV (np.array): 1D signal
        level (int)    : No. of decomposition levels
        loD (np.array): Low-pass decomposition filter
        hiD(np.array) : High-pass decomposition filter

    Returns:
        L: list of approximation coefficients per level
        H: list of detail coefficients per level
    """
    origSigV = sigV.copy()
    N = len(sigV)

    L = np.zeros((level, N))
    H = np.zeros_like(L)

    tempLo = loD.copy()
    tempHi = hiD.copy()

    evenOdd = 0  #Matches default dyadup(x, 0)

    for l in range(level):
        lf = tempLo.size
        pad = (lf // 2,)

        sigExtV = wextend(origSigV, pad)

        # Convolve with filters
        cA = convolve(sigExtV, tempLo, mode='full')
        cD = convolve(sigExtV, tempHi, mode='full')


        # Crop to original length
        L[l, :] = wkeep(cA, N, lf+1)
        H[l, :] = wkeep(cD, N, lf+1)

        # Upsample filters
        tempLo = dyadUp(tempLo, evenOdd)
        tempHi = dyadUp(tempHi, evenOdd)

        # Update signal for next level
        origSigV = L[l, :]

    return L, H


def swt2(imgM, level, loD, hiD):
    """
    Args:
        imgM (np.float64): 2D image
        level (int)      : No. of decomposition levels
        loD (np.array)   : Low-pass decomposition filter
        hiD (np.array)   : High-pass decomposition filter
    Returns:
        a: list of approximation coefficients per level
        h: list of horizontal wavelet coefficients per level
        v: list of vertical wavelet coefficients per level
        d: list of diagonal wavelet coefficients per level
    """
    imgM = imgM.astype(np.float64)
    origSiz = imgM.shape
    lf = loD.size

    # Initialize output
    a = np.zeros((*origSiz, level))
    h = np.zeros_like(a)
    v = np.zeros_like(a)
    d = np.zeros_like(a)

    tempLo = loD.copy()
    tempHi = hiD.copy()
    tempImgM = imgM.copy()

    for l in range(level):
            first = [lf + 1, lf + 1]
            ext = (lf // 2, lf // 2)

            # Periodic extension
            imgExtM = wextend(tempImgM, ext)

            # Decompose
            ca, ch, cv, cd = decomposeLOC(imgExtM, tempLo, tempHi, first, origSiz)
            a[:, :, l] = ca
            h[:, :, l] = ch
            v[:, :, l] = cv
            d[:, :, l] = cd

            tempImgM = ca  # Next level input

            # Upsample filters for next level
            tempLo = dyadUp(dyadUp(tempLo, 1), 1)
            tempHi = dyadUp(dyadUp(tempHi, 1), 1)

    return a, h, v, d

def getWaveletSubbands(scan3M, waveletName, level=1, dim='3d'):
    """ getWaveletSubbands
    Copyright (C) 2017-2019 Martin Vallières
    All rights reserved.
    https://github.com/mvallieres/radiomics-develop
    ------------------------------------------------------------------------
    IMPORTANT:
    - THIS FUNCTION IS TEMPORARY AND NEEDS BENCHMARKING. ALSO, IT
    ONLY WORKS WITH AXIAL SCANS FOR NOW. USING DICOM CONVENTIONS(NOT MATLAB).
    - Strategy: 2D transform for each axial slice. Then 1D transform for each        r
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
    removeV = np.zeros((3,))
    if sizeV[0] % 2 == 1:
        scan3M = np.concatenate((scan3M, scan3M[-1][np.newaxis, :, :]), axis=0)
        removeV[0] = 1
    if sizeV[1] % 2 == 1:
        scan3M = np.concatenate((scan3M, scan3M[:, -1][:, np.newaxis, :]), axis=1)
        removeV[1] = 1
    if dim == '3d' and sizeV[2] % 2 == 1:
        scan3M = np.concatenate((scan3M, scan3M[:, :, -1][:, :, np.newaxis]), axis=2)
        removeV[2] = 1

    # Step 2: Compute all sub-bands
    names = []
    if dim == '2d':
        names = ['LL', 'LH', 'HL', 'HH']
    elif dim == '3d':
        names = ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']

    #Decomposition filters
    wavelet = pywt.Wavelet(waveletName)
    loD = np.asarray(wavelet.dec_lo).flatten()  # Low-pass decomposition filter
    hiD = np.asarray(wavelet.dec_hi).flatten()  # High-pass decomposition filter

    # First pass using 2D stationary wavelet transform in axial direction
    for k in range(sizeV[2]):
        #coeffs = pywt.swt2(scan3M[:, :, k], wavelet=wavelet, level=level, norm=False, trim_approx=False)
        #LL, (LH, HL, HH) = coeffs[-1]  #last level
        LL, LH, HL, HH = swt2(scan3M[:, :, k], level, loD, hiD)

        if dim.lower() == '2d':
            subbands.setdefault(f'LL_{waveletName}', np.zeros_like(scan3M))
            subbands.setdefault(f'LH_{waveletName}', np.zeros_like(scan3M))
            subbands.setdefault(f'HL_{waveletName}', np.zeros_like(scan3M))
            subbands.setdefault(f'HH_{waveletName}', np.zeros_like(scan3M))

            subbands[f'LL_{waveletName}'][:, :, k] = np.squeeze(LL[:,:,-1]) #Last index corresponds to input level
            subbands[f'LH_{waveletName}'][:, :, k] = np.squeeze(LH[:,:,-1])
            subbands[f'HL_{waveletName}'][:, :, k] = np.squeeze(HL[:,:,-1])
            subbands[f'HH_{waveletName}'][:, :, k] = np.squeeze(HH[:,:,-1])

        elif dim.lower() == '3d':
            for band in ['LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH']:
                subbands.setdefault(f'{band}_{waveletName}', np.zeros_like(scan3M))

            # Simple mapping (same coefficients used multiple times)
            subbands[f'LLL_{waveletName}'][:, :, k] = np.squeeze(LL[:,:,-1])
            subbands[f'LLH_{waveletName}'][:, :, k] = np.squeeze(LL[:,:,-1])
            subbands[f'LHL_{waveletName}'][:, :, k] = np.squeeze(LH[:,:,-1])
            subbands[f'LHH_{waveletName}'][:, :, k] = np.squeeze(LH[:,:,-1])
            subbands[f'HLL_{waveletName}'][:, :, k] = np.squeeze(HL[:,:,-1])
            subbands[f'HLH_{waveletName}'][:, :, k] = np.squeeze(HL[:,:,-1])
            subbands[f'HHL_{waveletName}'][:, :, k] = np.squeeze(HH[:,:,-1])
            subbands[f'HHH_{waveletName}'][:, :, k] = np.squeeze(HH[:,:,-1])


    # Second pass using 1D stationary wavelet transform for all axial lines
    if dim == '3d':
        keysToProcess = [('LLL', 'LLH'),('LHL', 'LHH'),('HLL', 'HLH'),('HHL', 'HHH')]
        wav = pywt.Wavelet(waveletName)
        loD = np.asarray(wav.dec_lo).flatten()  # Low-pass decomposition filter
        hiD = np.asarray(wavelet.dec_hi).flatten()
        shape = next(iter(subbands.values())).shape

        for bandLow, bandHigh in keysToProcess:
            lowKey = f'{bandLow}_{waveletName}'
            highKey = f'{bandHigh}_{waveletName}'

            for j in range(shape[1]):
                for i in range(shape[0]):
                    vec = np.asarray(subbands[lowKey][i, j, :]).flatten()
                    #coeffs = pywt.swt(vec, wavelet=wav, level=level)
                    #L, H = coeffs[-1]
                    L, H = swt(vec, level, loD, hiD)

                    subbands[lowKey][i, j, :] = L[-1]
                    subbands[highKey][i, j, :] = H[-1]

    # Removing unnecessary data added in step 1
    if removeV[0]:
        for name in names:
            subbands[name] = subbands[name][:-1, :, :]
    if removeV[1]:
        for name in names:
            subbands[name] = subbands[name][:, :-1, :]
    if removeV[2]:
        for name in names:
            subbands[name] = subbands[name][:, :, :-1]

    return subbands


def waveletFilter(vol3M, waveType, direction, level, rotInvFlag=False):
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


            if rotInvFlag:
                # Compute average of all permutations of selected decomposition
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

        if rotInvFlag:
            # ---- testing -----
            import warnings
            warnings.warn("This is a preliminary implementation of the rotationally invariant"
                          "wavelet filter.It has not been validated against IBSI results.")
            # -------------------
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

    return np.array([dictIn[field] for field in args])


def aggregateRotatedResponses(rotTexture, aggregationMethod):
    """
    Function to aggregate textures from all orientations.

    Args:
        rotTexture: list of filter response maps to be aggregated,
                    computed at different orientations.
        aggregationMethod: string for aggregation method. May be 'avg','max', or 'std'.

    Returns:
        Aggregated response map.

    """

    out4M = np.stack(rotTexture, axis=0)
    if aggregationMethod == 'avg':
        agg3M = np.mean(out4M, axis=0)
    elif aggregationMethod == 'max':
        agg3M = np.max(out4M, axis=0)
    elif aggregationMethod == 'std':
        agg3M = np.std(out4M, axis=0)

    return agg3M


def rot3d90(arr3M, axis, angle):
    """
    Function to rotate a 3D array 90 degrees around a specified axis.

    Args:
        arr3M (np.ndarray): 3D scan.
        axis (int): Axis around which to rotate the array.
        angle (float): Angle of rotation in degrees.

    Returns:
        rotArr3M (np.ndarray): Rotated scan.
    """

    rotArr3M = rotate(arr3M, angle, axes=(axis % 3, (axis + 1) % 3), reshape=False)
    return rotArr3M


def rotate3dSequence(vol3M, index, sign):
    """
    Function to apply pre-defined sequence of right angle rotations to 2D/3D arrays

    Args:
        vol3M (np.ndarray): 3D scan.
        index (int): Multiplicative factor of 90 degrees.
        sign (int): Indicate direction of rotation (+1 or -1).

    Returns:
        rotArr3M (np.ndarray): Rotated scan.

    """

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

    rotArr3M = switch[index](vol3M)

    return rotArr3M


def flipSequenceForWavelets(vol3M, index, sign):
    """
    Function to flip 2D or 3D arrays by 180 degrees around a specified axis.
    Args:
        vol3M (np.ndarray): 3D scan.
        index (int): Flip index from (0,1,2...,7)
        sign  : +1 or -1 (-1 to reverse order of flips).
    """

    if index == 0:
        volOut3M = vol3M
    elif index == 1:
        volOut3M = np.flip(vol3M, axis=1)  # Flip rows
    elif index == 2:
        volOut3M = np.flip(vol3M, axis=0)  # Flip cols
    elif index == 3:
        if sign > 0:
            volOut3M = np.flip(np.flip(vol3M, axis=1), axis=0)  # rows then cols
        else:
            volOut3M = np.flip(np.flip(vol3M, axis=0), axis=1)  # cols then rows
    elif index == 4:
        volOut3M = np.flip(vol3M, axis=2)  # Flip slices
    elif index == 5:
        if sign > 0:
            volOut3M = np.flip(np.flip(vol3M, axis=1), axis=2)  # rows then slices
        else:
            volOut3M = np.flip(np.flip(vol3M, axis=2), axis=1)  # slices then rows
    elif index == 6:
        if sign > 0:
            volOut3M = np.flip(np.flip(vol3M, axis=0), axis=2)  # cols then slices
        else:
            volOut3M = np.flip(np.flip(vol3M, axis=2), axis=0)  # slices then cols
    elif index == 7:
        if sign > 0:
            volOut3M = np.flip(np.flip(np.flip(vol3M, axis=1), axis=0), axis=2)  # rows, cols, slices
        else:
            volOut3M = np.flip(np.flip(np.flip(vol3M, axis=2), axis=0), axis=1)  # slices, cols, rows
    else:
        raise ValueError("index must be an integer from 0 to 7.")

    return volOut3M


def rotationInvariantFilt(scan3M, mask3M, filter, *params):
    """
    Function to apply filter over a range of orientations to approximate rotation
    invariant filter.

    Args:
        scan3M (np.ndarray): 3D scan.
        mask3M (np.ndarray(dtype=bool)): 3D mask.
        filter (dict): Filter names and associated parameters.
        params (dict)

    Returns:
        aggS (dict): Rotation -invariant filter responses.

    """

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
        rotMask3M = mask3M
        if waveletFlag:
            rotScan3M = flipSequenceForWavelets(scan3M, index - 1, 1)
            if len(mask3M) > 0:
                rotMask3M = flipSequenceForWavelets(mask3M, index - 1, 1)
        else:
            if dim.lower() == '2d':
                rotScan3M = np.rot90(scan3M, k=index - 1)
                if len(mask3M) > 0:
                    rotMask3M = np.rot90(mask3M, k=index - 1)
            elif dim.lower() == '3d':
                rotScan3M = rotate3dSequence(scan3M, index - 1, 1)
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
