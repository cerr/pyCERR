"""
Pre- and post-processing transformations for AI models
"""

import numpy as np
import math
from scipy.ndimage import label


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

def getLargestConnComps(mask3M, numConnComponents):
    if np.sum(mask3M) > 1:
        labeledArray, numFeatures = label(mask3M, structure=np.ones((3, 3, 3)))
        ccSiz = [len(labeledArray[labeledArray == i]) for i in range(1, numFeatures + 1)]

        rankV = np.argsort(ccSiz)[::-1]
        if len(rankV) > numConnComponents:
            selV = rankV[:numConnComponents]
        else:
            selV = rankV[:]

        idxV = np.array([], dtype=int)
        for n in selV:
            idxV = np.concatenate((idxV, np.where(labeledArray == n + 1)[0]))

        maskOut3M = np.zeros_like(mask3M, dtype=bool)
        maskOut3M[idxV] = True
    else:
        maskOut3M = mask3M

    return maskOut3M
