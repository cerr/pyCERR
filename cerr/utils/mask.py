"""
Functions for processing of binary masks, including morphological
operations and custom routines mask generation.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import label, binary_opening, binary_fill_holes, uniform_filter
from skimage import exposure, filters, morphology, transform
from skimage.morphology import square, octagon
import cerr.utils.statistics as statUtil
import SimpleITK as sitk


def getDown2Mask(inM, sample):
    sV = inM.shape
    vec = list(range(0, sV[1], sample))

    maskM = np.zeros_like(inM, dtype=bool)

    vec2 = list(range(0, sV[0], sample))

    for ind in vec2:
        maskM[ind, vec] = True

    return maskM

def getDown3Mask(mask3M, sampleTrans, sampleAxis):
    sV = mask3M.shape
    sampleSlices = int(np.ceil(sV[2] / sampleAxis))

    outMask3M = np.zeros_like(mask3M, dtype=bool)

    indV = list(range(0, sV[2], sampleAxis))
    for i in indV:
        maskM = getDown2Mask(outMask3M[:, :, i], sampleTrans)
        outMask3M[:, :, i] = maskM

    return outMask3M


def getSurfacePoints(mask3M, sampleTrans=1, sampleAxis=1):
    """Routine to obtain sruface coordinates of the input mask

    Args:
        mask3M (numpy.ndarray): binary mask representing segmentation
        sampleTrans (int): optional, sample rate in transverse plane
        sampleAxis 9int): optional, sample rate along slices

    Returns:
        tuple: r,c,s coordinates of surface voxels
    """

    surfPoints = []

    r, c, s = np.where(mask3M)

    minR, maxR = np.min(r), np.max(r)
    minC, maxC = np.min(c), np.max(c)
    minS, maxS = np.min(s), np.max(s)

    croppedMask3M = mask3M[minR:maxR + 1, minC:maxC + 1, minS:maxS + 1]

    plusRowShift = croppedMask3M[2:, 1:-1, 1:-1]
    allNeighborsOn = plusRowShift.copy()

    minusRowShift = croppedMask3M[:-2, 1:-1, 1:-1]
    allNeighborsOn &= minusRowShift

    plusColShift = croppedMask3M[1:-1, 2:, 1:-1]
    allNeighborsOn &= plusColShift

    minusColShift = croppedMask3M[1:-1, :-2, 1:-1]
    allNeighborsOn &= minusColShift

    plusSlcShift = croppedMask3M[1:-1, 1:-1, 2:]
    allNeighborsOn &= plusSlcShift

    minusSlcShift = croppedMask3M[1:-1, 1:-1, :-2]
    allNeighborsOn &= minusSlcShift

    kernal = croppedMask3M[1:-1, 1:-1, 1:-1] & ~allNeighborsOn

    if sampleTrans is not None and sampleAxis is not None:
        if sampleTrans > 1 or sampleAxis > 1:
            kernal = kernal & getDown3Mask(kernal, sampleTrans, sampleAxis)

    if sampleTrans is not None and sampleAxis is not None:
        croppedMask3M = croppedMask3M & getDown3Mask(croppedMask3M,sampleTrans, sampleAxis)

    croppedMask3M[1:-1, 1:-1, 1:-1] = kernal

    r, c, s = np.where(croppedMask3M)

    r += minR
    c += minC
    s += minS

    #surfPoints = np.column_stack((r, c, s))

    return r,c,s

def surfaceExpand(mask3M, deltaRCSv):

    maskExpanded3M = mask3M.copy()

    # Define the edge kernel
    edge = np.zeros((3, 3, 3))
    edge[:, :, 0] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    edge[:, :, 1] = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    edge[:, :, 2] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    edge = edge / 7

    # Get surface points (assuming surfPoints is a list of coordinates)
    surfPoints = getSurfacePoints(maskExpanded3M)

    # Initialize edge3D as a logical array
    edge3D = np.zeros_like(maskExpanded3M, dtype=bool)

    # Mark surface points in edge3D
    edge3D[surfPoints[0],surfPoints[1],surfPoints[2]] = True

    # Optional: Convolution (commented out in MATLAB)
    # edge3D = convolve(maskDown3D, edge, mode='same')
    # edge3D = (edge3D < 0.999) & maskDown3D

    # Expand margin using convolution
    # Create margin ball
    #c1 = int(np.ceil(margin / delta_xy))
    #c2 = int(np.ceil(margin / delta_xy))
    #c3 = int(np.ceil(margin / sliceThickness))
    c1, c2, c3 = deltaRCSv

    uM, vM, wM = np.meshgrid(np.arange(-c1, c1 + 1), np.arange(-c2, c2 + 1), np.arange(-c3, c3 + 1), indexing='ij')

    #xM = uM * delta_xy
    #yM = vM * delta_xy
    #zM = wM * sliceThickness

    rM = uM**2 + vM**2 + wM**2
    marginSq = c1**2 + c2**2 + c3**2
    ball = rM <= marginSq

    # Find indices of the ball
    iBallV, jBallV, kBallV = np.where(ball)

    sR = rM.shape
    deltaV = (np.array(sR) - 1) / 2 + 1

    onesV = np.ones(len(iBallV), dtype=bool)

    # Find indices of edge3D
    iV, jV, kV = np.where(edge3D)
    sV = mask3M.shape

    # Convert subscripts to linear indices
    #ind_surfV = np.ravel_multi_index((iV, jV, kV), sV, order = 'F')
    ind_surfV = iV + jV * sV[0] + kV * sV[0] * sV[1]

    # Calculate ball offsets
    ball_offsetV = (iBallV - deltaV[0]) + sV[0] * (jBallV - deltaV[1]) + sV[0] * sV[1] * (kBallV - deltaV[2])

    # Apply the ball to maskDown3D
    for i in range(len(ind_surfV)):
        total_indV = ind_surfV[i] + ball_offsetV
        total_indV = np.clip(total_indV, 0, np.prod(sV) - 1)  # Python uses 0-based indexing
        #maskExpanded3M[total_indV.astype(int)] = onesV
        maskExpanded3M[np.unravel_index(total_indV.astype(int), sV, order='F')] = onesV

    return maskExpanded3M


def createStructuringElement(sizeCm, resolutionCmV, dimensions=3, shape='flat'):
    """
    Function to create structuring element for morphological operations given
    desired dimensions in cm.

    Args:
        sizeCm (np.float): Size of structuring element in cm.
        resolutionCmV (np.array): Image resolution in cm [dx, dy, dz].
        dimensions (int): [optional, default=3] Specify 3 for 3D or 2 for 2D.
        shape (string): [optional, default='flat'] Geometric neighborhood shape. Supported
                       values: 'flat', 'sphere', 'disk'.

    Returns:
        structuringElement (np.ndarray): Structuring element.
    """

    if np.ndim(sizeCm) == 0:
        sizeCmV = np.repeat(sizeCm, dimensions)
    elif np.ndim(sizeCm) == 1 and len(sizeCm)==len(resolutionCmV):
        sizeCmV = sizeCm
    else:
        raise ValueError('Invalid input sizeCm. Must be scalar or list [size_x_cm, size_y_cm, size_z_cm]. ')
    sizePixels = np.ceil(np.divide(sizeCmV, np.abs(resolutionCmV)))
    evenIdxV = sizePixels % 2 == 0
    if any(evenIdxV):
        sizePixels[evenIdxV] += 1  # Ensure odd size for symmetric structuring element

    if shape == 'flat':
        structuringElement = np.ones(tuple(sizePixels.astype(int)), dtype=np.uint8)
    elif shape == 'sphere':
        x, y, z = np.meshgrid(np.arange(-sizePixels[0], sizePixels[0] + 1),
                              np.arange(-sizePixels[1], sizePixels[1] + 1),
                              np.arange(-sizePixels[2], sizePixels[2] + 1),indexing='ij')
        structuringElement = ((x / sizePixels[0]) ** 2 +
                              (y / sizePixels[1]) ** 2 +
                              (z / sizePixels[2]) ** 2) <= 1
    elif shape == 'disk':
        x, y = np.meshgrid(np.arange(-sizePixels[0], sizePixels[0] + 1),
                              np.arange(-sizePixels[1], sizePixels[1] + 1))

        structuringElement = ((x / sizePixels[0]) ** 2 +
                              (y / sizePixels[1]) ** 2) <= sizePixels[0]**2

    else:
        raise ValueError('Structuring element type %s is not supported.' %(shape))

    return structuringElement

def fillHoles(binaryMask):
    """
    Function to fill small holes in input binary mask

    Args:
        binaryMask: np.ndarray(type=bool) for input mask.

    Returns:
        filledMask: np.ndarray(type=bool) for filled mask.
    """

    filledMask = ndimage.binary_fill_holes(binaryMask)
    return filledMask

def morphologicalClosing(binaryMask, structuringElement):
    """
    Function for morphological closing of input binary mask

    Args:
        binaryMask (np.ndarray(dtype=bool)): Input mask.
        structuringElement (np.array): Flat morphological structuring element.

    Returns:
        numpy.ndarray(dtype=bool): Closed mask using input structuring element.
    """
    # Apply padding
    padding = int(np.floor(structuringElement.shape[0]/2))
    binaryMaskPad = np.pad(binaryMask, padding, mode='constant', constant_values=0)

    # Closing
    if binaryMask.ndim == 3 and structuringElement.ndim == 2:
        closedMaskPad = np.zeros(binaryMaskPad.shape)
        for slc in range(binaryMaskPad.shape[2]):
            closedMaskPad[:, :, slc] = ndimage.binary_closing(binaryMaskPad[:, :, slc], structure=structuringElement)
    elif binaryMask.ndim == 3 and structuringElement.ndim == 3:
        binaryMaskPad = np.pad(binaryMask, padding, mode='constant', constant_values=0)
        closedMaskPad = ndimage.binary_closing(binaryMaskPad, structure=structuringElement)
    elif binaryMask.ndim == 2 and structuringElement.ndim == 2:
        binaryMaskPad = np.pad(binaryMask, padding, mode='constant', constant_values=0)
        closedMaskPad = ndimage.binary_closing(binaryMaskPad, structure=structuringElement)


    # Un-pad
    if structuringElement.ndim == 2:
        closedMask = closedMaskPad[padding:-padding, padding:-padding]
    elif structuringElement.ndim == 3:
        closedMask = closedMaskPad[padding:-padding, padding:-padding, padding:-padding]

    return closedMask


def blurring(binaryMask, sigmaVox, filtType='gaussian'):
    """
    Function for Gaussian blurring of input binary mask

    Args:
        binaryMask (numpy.array): Binary mask to blur.
        sigmaVox (float): Sigma for Gaussian OR filter radius for box filter (in units of voxels).
        filtType (string): [optional, default:'gaussian'] 'gaussian' or 'box' smoothing filter.

    Returns:
        numpy.ndarray(dtype=bool): Blurred mask using Gaussian blur with input sigma.
    """

    origSize = binaryMask.shape
    binaryMaskImg = sitk.GetImageFromArray(binaryMask.astype(int))
    padded = False

    if filtType == 'gaussian':
        filter = sitk.SmoothingRecursiveGaussianImageFilter()
        filter.SetSigma(sigmaVox)
    elif filtType == 'box':
        padded = True
        padMaskImg = sitk.MirrorPad(binaryMaskImg, [sigmaVox] * 3, [sigmaVox] * 3)
        padMaskArr = sitk.GetArrayFromImage(padMaskImg)
        filter = sitk.BoxMeanImageFilter()
        filter.SetRadius(sigmaVox)

    fullSize = padMaskArr.shape
    blurredMaskPad3M = np.zeros((binaryMask.shape[0], binaryMask.shape[1], fullSize[2]), dtype=float)
    for slc in range(fullSize[2]):
        img = sitk.GetImageFromArray(padMaskArr[:, :, slc].astype(float))
        blurImageFull = filter.Execute(img)
        if padded:
            startIdx = [(f - o) // 2 for f, o in zip(fullSize, origSize)]
            blurImage = sitk.Crop(blurImageFull, startIdx, startIdx)
        else:
            blurImage = blurImageFull
        blurredMaskPad3M[:, :, slc] = sitk.GetArrayFromImage(blurImage)
    if padded and sigmaVox > 0:
        blurredMask3M = blurredMaskPad3M[:, :, sigmaVox:-sigmaVox]
    else:
        blurredMask3M = blurredMaskPad3M
    return blurredMask3M


def computeBoundingBox(binaryMaskM, is2DFlag=False, maskFlag=0):
    """
    Function for finding extents of bounding box given a binary mask

    Args:
        binaryMaskM (np.ndarray(type=bool)): Input mask.
        is2DFlag (bool): [optional, default=False] Flag for computing
                  slice-wise extents if true.
        maskFlag (int): [optional, default=0] If maskFlag > 0, it is interpreted as a
                         padding parameter.
    Returns:
        minr (int): Start of mask along rows.
        maxr(int): End of mask along rows.
        minc(int): Start of mask along cols.
        maxc(int): End of mask along cols.
        mins(int): Start of mask along slices.
        maxs(int): End of mask along slices.
        bboxmask (np.ndarray(dtype=bool)): Mask of bounding box.
    """
    maskFlag = int(maskFlag)

    if is2DFlag:

        iV, jV = np.where(binaryMaskM)
        kV = []
        minr = np.min(iV).astype(int)
        maxr = np.max(iV).astype(int)
        minc = np.min(jV).astype(int)
        maxc = np.max(jV).astype(int)
        mins = []
        maxs = []
    else:
        iV, jV, kV = np.where(binaryMaskM)
        minr = np.min(iV).astype(int)
        maxr = np.max(iV).astype(int)
        minc = np.min(jV).astype(int)
        maxc = np.max(jV).astype(int)
        mins = np.min(kV).astype(int)
        maxs = np.max(kV).astype(int)

    bboxmask = None

    if maskFlag != 0:
        bboxmask = np.zeros_like(binaryMaskM)

        if maskFlag > 0:
            siz = binaryMaskM.shape
            minr -= maskFlag
            maxr += maskFlag
            if maxr >= siz[0]:
                maxr = int(siz[0] - 1)
            if minr < 0:
                minr = 0
            minc -= maskFlag
            maxc += maskFlag
            if maxc >= siz[1]:
                maxc = int(siz[1] - 1)
            if minc < 0:
                minc = 0

            else:
                mins -= maskFlag
                maxs += maskFlag
                if maxs >= siz[2]:
                    maxs = int(siz[2] - 1)
                if mins < 0:
                    mins = 0

        if is2DFlag:
            bboxmask[minr:maxr+1, minc:maxc+1] = 1
        else:
            bboxmask[minr:maxr+1, minc:maxc+1, mins:maxs+1] = 1

    return minr, maxr, minc, maxc, mins, maxs, bboxmask

def closeMask(mask3M, inputResV, structuringElementSizeCm, shape='flat'):

    """
    Function for morphological closing and hole-filling for binary masks

    Args:
        mask3M (np.ndarray): Binary mask to close and hole-fill.
        inputResV (np.array): Physical Resolution of the mask in cm.
        structuringElementSizeCm( float): Size of structuring element for closing in cm
        shape (str): Shape of structuring element. May be 'flat', 'disk', or 'sphere'.
    Returns:
        filledMask3M (np.ndarray(dtype=bool)): Filled mask.
    """
    # Create structuring element
    structuringElement = createStructuringElement(structuringElementSizeCm,\
                                                  inputResV, dimensions=3, shape=shape)

    # Apply morphological closing
    closedMask3M = morphologicalClosing(mask3M, structuringElement)

    # Fill any remaining holes
    filledMask3M = fillHoles(closedMask3M)

    return filledMask3M


def largestConnComps(mask3M, numConnComponents, minSize=0, dim=3):
    """
    Function to retain 'N' largest connected components in input binary mask

    Args:
        mask3M (np.ndarray(dtype=bool)): 3D binary segmentation mask
                   (OR)  3D binary mask.
        numConnComponents (int): number of largest components to retain.
        minSize (int): [optional, default=0] Min. size of connected component to retain.
        dim (int): [optional, default=3. Includes 26 neighbours in 3D ] 2 (2D) or 3 (3D).

    Returns:
        maskOut3M (np.ndarray(dtype=bool)): 3D mask with labels corresponding to components.

    """
    if dim == 2:
        structure = np.ones((3, 3))
    elif dim == 3:
        structure = np.ones((3, 3, 3))

    maskOut3M = np.zeros_like(mask3M, dtype=bool)

    if np.sum(mask3M) > 1:
        #Extract connected components
        labeledArray, numFeatures = label(mask3M, structure)

        # Sort by size
        ccSiz = np.array([len(labeledArray[labeledArray == i]) for i in range(1, numFeatures + 1)])

        # Filter min acceptable
        ccSiz[ccSiz < minSize] = 0
        if np.any(ccSiz):
            rankV = np.argsort(ccSiz)[::-1]
            if len(rankV) > numConnComponents:
                selV = rankV[:numConnComponents]
            else:
                selV = rankV[:]

            # Return N largest
            for n in selV:
                idxV = labeledArray == n + 1
                maskOut3M[idxV] = True

    return maskOut3M


def getCouchLocationHough(scan3M, minLengthOpt=None, retryOpt=False):
    """

    Function to identify location (row no.) of couch in input scan

    Args:
        scan3M (np.ndarray): Input scan.
        minLengthOpt (float): [optional, default=None] Minimum length
                              of couch expected (in no. voxels). If set to None,
                              min. length is taken to be 1/8th image size.
        retryOpt (bool): [optional, default=False] Flag to rerun search
                         with minLengthOpt halved if couch length is 0.

    Returns:
        yCouch (int): Row no. representing couch location.
        selectedLines (dict): Candidate lines representing couch.
    """
    if minLengthOpt is None:
        minLengthOpt = []

    scanSizeV = scan3M.shape
    midptS = np.floor(scanSizeV[0] / 2)
    numPeaks = 20

    # 3D max projection
    maxM = np.amax(scan3M, axis=2)
    histeqM = exposure.equalize_hist(maxM, nbins=64)

    # Detect edges
    edgeM1 = filters.sobel(histeqM)
    edgeM2 = morphology.dilation(edgeM1, footprint=np.ones((3,3)))
    bwThreshold = np.max(edgeM2)/4
    edgeM3 = edgeM2>=bwThreshold

    if not minLengthOpt:
        minLength = np.floor(edgeM3.shape[1] / 8).astype(int)  # couch covers 1/8th of image
    else:
        minLength = int(minLengthOpt)

    # Hough transform
    hspace, theta, dist = transform.hough_line(edgeM3)
    peakSpace, peakTheta, peakDist = transform.hough_line_peaks(hspace, theta,\
                                     dist, num_peaks=numPeaks)
    numDetectedPeaks = len(peakTheta)
    #probLines = transform.probabilistic_hough_line(edgeM2, threshold=100,
    # line_length=minLength, line_gap=5, theta=peakTheta)

    ## Find line segments in edge image corresponding to peak lines
    midV = np.arange(int(0.5 * midptS), int(0.5 * midptS) + midptS)
    tolp = 5

    selectedLines = []
    yi = []
    overlapFraction = []
    # Loop over peaks
    for peakIdx in range(numDetectedPeaks):

        # Convert normal form to slope-intercept form
        angle = peakTheta[peakIdx]
        distance = peakDist[peakIdx]
        peakSlope = -np.cos(angle) / np.sin(angle)
        peakIntercept = distance / np.sin(angle)

        # Detect line segments at this angle
        probLines = transform.probabilistic_hough_line(edgeM2, threshold=100,\
                                                       line_length=minLength, \
                                                       line_gap=1, theta=np.array([angle]))
        # Match intercepts
        for lineSegment in probLines:
            (x1, y1), (x2, y2) = lineSegment
            x1, y1 = map(float, (x1, y1))
            x2, y2 = map(float, (x2, y2))
            intercept = y1 - peakSlope * x1
            if np.abs(intercept - peakIntercept)< tolp:
                lineLength = np.linalg.norm(np.array(lineSegment[1]) -\
                                            np.array(lineSegment[0]))
                p1 = lineSegment[0]
                p2 = lineSegment[1]

                # Require couch lines to have same starting & ending points
                if lineLength > minLength and np.abs(p2[1] - p1[1])<tolp:
                    if p1[0] < p2[0]:
                        line = {'point1': p1, 'point2': p2}
                        lineV = np.arange(p1[0], p2[0]+1)
                    else:
                        line = {'point1': p2, 'point2': p1}
                        lineV = np.arange(p2[0], p1[0]+1)

                    # Record location
                    intx = np.intersect1d(lineV, midV).size
                    if line['point1'][1] > midptS and intx > 0:
                        yi.append(line['point2'][1])
                        overlapFraction.append(intx)
                        selectedLines.append(line)

    # Return couch location

    if len(yi) == 0:
        if retryOpt:
            yCouch, selectedLines = getCouchLocationHough(scan3M, minLength / 2)
        else:
            yCouch = scanSizeV[0]
    else:
        if np.any(overlapFraction):
            I = np.argmax(overlapFraction)
            yCouch = int(yi[I])
        else:
            yi = np.array(yi)
            yCouch = int(np.min(yi[yi > 0]))

    return yCouch, selectedLines


def getPatientOutline(scan3M, outThreshold=-400, slicesV=None,
                      minMaskSize=1500, normFlag=False):
    """
    Function to extract binary mask of patient outline on input scan.

    Args:
        scan3M (np.ndarray): 3D scan.
        outThreshold (float): [optional, default=-400] Intensity level representing air.
                              -400 HU is for CT scans. Users should input appropriate threshold
                              for other modalities.
        slicesV (np.array): [optional, default=None] Range of slices for
                            outline extraction. All slices are analyzed if set to None.
        minMaskSize (int): [optional, default=1500] Minimum acceptable size of mask
                            on any slice in no. voxels.
        normFlag (bool): [optional, default=False] Flag to normalize scan3M
                         before applying air threshold (recommended for MR images).

    Returns:
        conn3dPtMask3M (np.ndarray(dtype=bool)): Mask of patient outline.

    """
    # Define default values for optional inputs
    if slicesV is None:
        slicesV = np.arange(scan3M.shape[2])

    # Mask out couch
    couchStartIdx, __ = getCouchLocationHough(scan3M)
    couchMaskM = np.zeros((scan3M.shape[0], scan3M.shape[1]), dtype=bool)
    couchMaskM[couchStartIdx:, :] = True

    # Intensity threshold for air
    if normFlag:
        scan3M = scan3M / (np.max(scan3M) + np.finfo(float).eps)

    scanThreshV = scan3M[scan3M>outThreshold]
    adjustedThreshold = statUtil.prctile(scanThreshV, 5)
    minInt = np.min(scan3M)

    # Loop over slices
    ptMask3M = np.zeros_like(scan3M, dtype=bool)
    discardSize = 200
    for slc in slicesV:

        # Threshold image
        sliceM = scan3M[:, :, slc]
        threshM = sliceM > adjustedThreshold

        # Mask out couch
        binM = np.logical_and(threshM, np.logical_not(couchMaskM))

        # Separate pt outline from table
        binM = binary_opening(binM, octagon(5,2))

        # Fill holes in pt outline
        if np.any(binM):
            # Retain largest connected component if size exceeds discard threshold
            labeledArray, numFeatures = label(binM)
            sizes = np.bincount(labeledArray.ravel())
            maxLabel = np.argmax(sizes[1:]) + 1

            if sizes[maxLabel] >= minMaskSize:
                maskM = labeledArray == maxLabel

                # Fill holes
                rowMaxIdxV = np.argmax(np.flipud(maskM), axis=0)
                rowMaxValV = np.max(np.flipud(maskM), axis=0)
                rowMaxIdx = binM.shape[0] - np.min(rowMaxIdxV[rowMaxValV], axis=0)
                sliceM[rowMaxIdx:, :] = minInt
                thresh2M = sliceM > 1.5 * adjustedThreshold
                thresh2M = binary_fill_holes(thresh2M)

                # Remove small islands
                labeledArray, numFeatures = ndimage.label(thresh2M)
                componentSizes = np.bincount(labeledArray.ravel())
                tooSmall = componentSizes < discardSize
                tooSmallMask = tooSmall[labeledArray]
                thresh2M[tooSmallMask] = 0

                thresh2M = morphologicalClosing(thresh2M,square(5))
                smoothedLabelM = uniform_filter(thresh2M.astype(float), size=5)
                maskM = smoothedLabelM > 0.5

                ptMask3M[:, :, slc] = maskM

    # 3D connected component filter
    conn3dPtMask3M = largestConnComps(ptMask3M, 1)

    return conn3dPtMask3M
