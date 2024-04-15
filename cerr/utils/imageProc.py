"""
Pre- and post-processing transformations for AI models
"""

import math
import numpy as np
from scipy import ndimage
from scipy.ndimage import label, binary_opening, binary_fill_holes, uniform_filter
from skimage import exposure, filters, morphology, transform
from skimage.morphology import square, octagon

import cerr.plan_container as pc
import cerr.utils.statistics_utils as stat
from cerr.contour import rasterseg as rs
from cerr.dataclasses import scan as cerrScan


def resizeScanAndMask(scan3M, mask4M, gridS, outputImgSizeV, method, *argv):
    """
    Script to resize images and label maps for deep learning.
    """

    nArgs = 5 + len(argv)

    if nArgs > 5:
        limitsM = argv[0]
        if len(argv) > 1:
            preserveAspectFlag = argv[1]
        else:
            preserveAspectFlag = 0
    else:
        limitsM = []
        preserveAspectFlag = 0

    # Get input image size
    if scan3M.size != 0:
        origSizeV = [scan3M.shape[0], scan3M.shape[1], scan3M.shape[2]]
    else:
        origSizeV = [mask4M.shape[0], mask4M.shape[1], mask4M.shape[2]]
    numStr = mask4M.shape[3]

    # Get input grid
    xV = gridS[0]
    yV = gridS[1]
    zV = gridS[2]
    voxSizeV = [np.median(np.diff(xV)),np.median(np.diff(yV)),np.median(np.diff(zV))]

    # Resize image by method
    methodLower = method.lower()
    if methodLower == 'padorcrop3d':

        if preserveAspectFlag:
            corner_cube = scan3M[:5, :5, :5]
            bg_mean = np.mean(corner_cube)
            scanSizeV = scan3M.shape
            paddedSizeV = max(scanSizeV[:2])
            padded3M = bg_mean * np.ones((paddedSizeV, paddedSizeV, scan3M.shape[2]))
            idx11 = int(1 + (paddedSizeV - scanSizeV[0]) / 2)
            idx12 = int(idx11 + scanSizeV[0] - 1)
            idx21 = int(1 + (paddedSizeV - scanSizeV[1]) / 2)
            idx22 = int(idx21 + scanSizeV[1] - 1)
            padded3M[idx11:idx12, idx21:idx22, :] = scan3M
            scan3M = padded3M
            origSizeV = scan3M.shape

        xPad = int(np.floor((outputImgSizeV[0] - origSizeV[0]) / 2))
        if xPad < 0:
            resizeMethod = 'unpad3d'
        else:
            resizeMethod = 'pad3d'
        scanOut3M, maskOut4M, gridOutS = resizeScanAndMask(scan3M, mask4M, gridS, outputImgSizeV, resizeMethod, *argv)

    elif methodLower == 'pad3d':
        xPad = math.ceil((outputImgSizeV[0] - origSizeV[0]) / 2)
        yPad = math.ceil((outputImgSizeV[1] - origSizeV[1]) / 2)

        if xPad < 0 or yPad < 0:
            raise ValueError("To resize by padding, output image dimensions must be larger than (cropped) input image dimensions")

        # Pad scan
        if scan3M is None:
            scanOut3M = None
        else:
            minScanVal = np.min(scan3M)
            scanOut3M = np.full(outputImgSizeV, minScanVal, dtype=scan3M.dtype)
            scanOut3M[xPad:xPad + origSizeV[0], yPad:yPad + origSizeV[1], :origSizeV[2]] = scan3M
            xOutV = np.arange(xV[0] - (xPad/2)*voxSizeV[0], xV[-1]+(xPad/2)*voxSizeV[0], voxSizeV[0])
            yOutV = np.arange(yV[0] - (yPad/2)*voxSizeV[1], yV[-1]+(yPad/2)*voxSizeV[1], voxSizeV[1])
            zOutV = zV

        # Pad mask
        if mask4M.size == 0:
            maskOut4M = None
        else:
            maskOut4M = np.zeros(outputImgSizeV + [numStr])
            maskOut4M[xPad:xPad + origSizeV[0], yPad:yPad + origSizeV[1], :origSizeV[2], :] = mask4M

        gridOutS = (xOutV, yOutV, zOutV)

    elif methodLower == 'unpad3d':
        xPad = int(np.floor((outputImgSizeV[0] - origSizeV[0]) / 2))
        yPad = int(np.floor((outputImgSizeV[1] - origSizeV[1]) / 2))

        xPad = -xPad
        yPad = -yPad

        if scan3M.size == 0:
            scanOut3M = []
        else:
            scanOut3M = scan3M[xPad:xPad + outputImgSizeV[0],
                        yPad:yPad + outputImgSizeV[1], :]

            xOutV = xV[xPad/2:-xPad/2]
            yOutV = yV[yPad/2:-yPad/2]
            zOutV = zV

        if mask4M.size == 0:
            maskOut4M = []
        else:
            maskOut4M = mask4M[xPad:xPad + outputImgSizeV[0],
                        yPad:yPad + outputImgSizeV[1], :, :]
        gridOutS = (xOutV, yOutV, zOutV)


    return scanOut3M, maskOut4M, gridOutS


def getPatientOutline(scan3M, outThreshold, slicesV=None,
                      minMaskSize=None, normFlag=False):
    """
    Returns binary mask of patient outline on input scan
    :param scan3M: 3D scan array
    :param outThreshold: Intensity level representing air
    ---Optional---
    :param slicesV: Range of slices for outline extraction(default:all)
    :param minMaskSize: Minimum size of mask on any slice in no. xoels (default: 1500)
    :param normFlag: Set to True to normalize scan before thresholding (recommended for MR images)
    """
    # Define default values for optional inputs
    if slicesV is None:
        slicesV = np.arange(scan3M.shape[2])

    if minMaskSize is None:
        minMaskSize = 1500

    # Mask out couch
    couchStartIdx, __ = getCouchLocationHough(scan3M)
    couchMaskM = np.zeros((scan3M.shape[0], scan3M.shape[1]), dtype=bool)
    couchMaskM[couchStartIdx:, :] = True

    # Intensity threshold for air
    if normFlag:
        scan3M = scan3M / (np.max(scan3M) + np.finfo(float).eps)

    scanThreshV = scan3M[scan3M>outThreshold]
    adjustedThreshold = stat.prctile(scanThreshV,5)
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
                labeled_array, num_features = ndimage.label(thresh2M)
                component_sizes = np.bincount(labeled_array.ravel())
                too_small = component_sizes < discardSize
                too_small_mask = too_small[labeled_array]
                thresh2M[too_small_mask] = 0

                thresh2M = morphologicalClosing(thresh2M,square(5))
                smoothedLabelM = uniform_filter(thresh2M.astype(float), size=5)
                maskM = smoothedLabelM > 0.5

                ptMask3M[:, :, slc] = maskM

    # 3D connected component filter
    conn3dPtMask3M, __ = getLargestConnComps(ptMask3M, 1)

    return conn3dPtMask3M

def getCouchLocationHough(scan3M, minLengthOpt=None, retryOpt=0):
    """
    Return location (row no.) of couch in input scan
    """
    if minLengthOpt is None:
        minLengthOpt = []

    midptS = np.floor(scan3M.shape[0] / 2)

    maxM = np.amax(scan3M, axis=2)
    histeqM = exposure.equalize_hist(maxM)
    edgeM1 = filters.sobel(histeqM)
    edgeM2 = morphology.dilation(edgeM1)

    if not minLengthOpt:
        minLength = np.floor(edgeM2.shape[1] / 8).astype(int)  # couch covers 1/8th of image
    else:
        minLength = int(minLengthOpt)

    lines = transform.probabilistic_hough_line(edgeM2, threshold=0, line_length=minLength, line_gap=5)
    overlapFraction = np.zeros(len(lines))
    midV = np.arange(int(0.5 * midptS), int(0.5 * midptS) + midptS)
    yi = np.zeros(len(lines))

    for i, (p0, p1) in enumerate(lines):
        len_line = np.linalg.norm(np.array(p0) - np.array(p1))
        if p0[1] == p1[1] and len_line > minLength:
            if p0[0] < p1[0]:
                lineV = np.arange(p0[0], p1[0])
            else:
                lineV = np.arange(p1[0], p0[0])
            if p0[1] > midptS and np.intersect1d(lineV, midV).size != 0:
                yi[i] = p1[1]
                overlapFraction[i] = np.intersect1d(lineV, midV).size

    if np.any(overlapFraction):
        I = np.argmax(overlapFraction)
        yCouch = yi[I].astype(int)
    else:
        yCouch = min(yi[np.where(yi > 0)]).astype(int)

    if retryOpt and yCouch == 0:
        yCouch, lines = getCouchLocationHough(scan3M, minLength / 2)

    return yCouch, lines


def getLargestConnComps(structNum, numConnComponents, planC=None, saveFlag=None, replaceFlag=None, procSructName=None):
    """
    Returns 'N' largest connected components in input binary mask

    :param structNum : Index of structure in planC (OR) 3D binary mask
    :param structuringElementSizeCm : Desired size of structuring element for closing in cm
    :param planC
    :param saveFlag : Import filtered mask to planC structure
    :param replaceFlag : Set to true to replace input mask with processed mask to planC (Default:False)
    :param procSructName : Output structure name. Original structure name is used if empty
    :returns maskOut3M, planC
    """

    if np.isscalar(structNum):
        # Get binary mask of structure
        mask3M = rs.getStrMask(structNum,planC)
    else:
        # Input is binary structure mask
        mask3M = structNum

    if np.sum(mask3M) > 1:
        #Extract connected components
        labeledArray, numFeatures = label(mask3M, structure=np.ones((3, 3, 3)))

        # Sort by size
        ccSiz = [len(labeledArray[labeledArray == i]) for i in range(1, numFeatures + 1)]
        rankV = np.argsort(ccSiz)[::-1]
        if len(rankV) > numConnComponents:
            selV = rankV[:numConnComponents]
        else:
            selV = rankV[:]

        # Return N largest
        maskOut3M = np.zeros_like(mask3M, dtype=bool)
        for n in selV:
            idxV = labeledArray == n + 1
            maskOut3M[idxV] = True

    if planC is not None and saveFlag:
        if procSructName is None:
            procSructName = planC.structure[structNum].structureName
        planC = updateStructure(structNum, maskOut3M, procSructName, replaceFlag, planC)
    else:
        maskOut3M = mask3M

    return maskOut3M, planC

def closeMask(structNum, structuringElementSizeCm, planC, saveFlag=False, replaceFlag=None, procSructName=None):
    """
    Morphological closing and hole-filling for binary masks

    :param structNum : Index of structure in planC
    :param structuringElementSizeCm : Desired size of structuring element for closing in cm
    :param planC
    :param saveFlag : Set to true to save processed mask to planC (Default:False)
    :param replaceFlag: Set to true to replace input mask with processed mask to planC (Default:False)
    :param procSructName : Output structure name. Original structure name is used if empty
    :returns filledMask3M, planC
    """

    # Get binary mask of structure
    mask3M = rs.getStrMask(structNum,planC)

    # Get mask resolution
    assocScanNum = cerrScan.getScanNumFromUID(planC.structure[structNum].assocScanUID, planC)
    inputResV = planC.scan[assocScanNum].getScanSpacing()

    # Create structuring element
    structuringElement = createStructuringElement(structuringElementSizeCm, inputResV, dimensions=3)

    # Apply morphological closing
    closedMask3M = morphologicalClosing(mask3M, structuringElement)

    # Fill any remaining holes
    filledMask3M = fillHoles(closedMask3M)

    # Save to planC
    if saveFlag:
        if procSructName is None:
            procSructName = planC.structure[structNum].structureName
        planC = updateStructure(structNum, filledMask3M, procSructName, replaceFlag, planC)

    return filledMask3M, planC

def createStructuringElement(sizeCm, resolutionCmV, dimensions=3):
    """
    Create structuring element for morphological operations given desired dimensions in cm.
    """
    sizeCmV =  np.repeat(sizeCm, dimensions)
    sizePixels = np.ceil(np.divide(sizeCmV, resolutionCmV))
    evenIdxV = sizePixels % 2 == 0
    if any(evenIdxV):
        sizePixels[evenIdxV] += 1  # Ensure odd size for symmetric structuring element
    structuringElement = np.ones(tuple(sizePixels.astype(int)), dtype=np.uint8)

    return structuringElement

def fillHoles(binaryMask):
    """
    Fill small holes in input binary mask
    """
    filledMask = ndimage.binary_fill_holes(binaryMask)
    return filledMask

def morphologicalClosing(binaryMask, structuringElement):
    """
    Morphological closing of input binary mask
    """
    closedMask = ndimage.binary_closing(binaryMask, structure=structuringElement)
    return closedMask

def updateStructure(structNum, newMask3M, newStrName, replaceFlag, planC):
    """
    Save updated structure to planC
    """
    assocScanNum = cerrScan.getScanNumFromUID(planC.structure[structNum].assocScanUID, planC)

    if replaceFlag:
        # Delete structNum
        del planC.structure[structNum]

    pc.import_structure_mask(newMask3M, assocScanNum, newStrName, None, planC)

    return planC

