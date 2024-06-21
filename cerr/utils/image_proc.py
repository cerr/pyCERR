"""
Pre- and post-processing transformations for AI models
"""
import math
import numpy as np
from skimage.transform import resize


def resizeScanAndMask(scan3M, mask4M, gridS, outputImgSizeV, method, \
                      limitsM=None, preserveAspectFlag=False):
    """
    Function to resize input scan and mask using specified coordinates, method, and output dimensions.
    Supports preserving aspect ratio and slice-wise resizing within bounding box limits.

    Args:
        scan3M (np.ndarray): 3D input scan.
        mask4M (np.ndarray): 4D input mask of dimension [nRows x nCols x nSlices x nStructures]
                            (stack of binary masks representing various structures).
        gridS (tuple): (xV, yV, zV) for coordinates of input scan/mask.
        outputImgSizeV (np.array): Output image dimensions [nRows, nCols, nSlices].
        method (string): Resizing method for input scan. Supported options include
                'padorcrop3d', 'pad3d', 'unpad3d', 'pad2d', 'unpad2d', 'padslices'
                'unpadslices', 'bilinear','bicubic', and 'nearest'.
                 Note: Masks are resized using 'nearest' for input methods 'bilinear','bicubic', and 'nearest'.
        limitsM (np.ndarray): [optional, default=None] Extents of bounding box on each slice.
                              minr = limitsM[slcNum, 0], maxr = limitsM[slcNum, 1],
                              minc = limitsM[slcNum, 2], maxc = limitsM[slcNum, 3]
        preserveAspectFlag (bool): Flag to preserve input aspect ratio by padding prior to resizing.

    Returns:
          scanOut3M (np.ndarray): 3D resized scan.
          maskOut4M (np.ndarray): 4D resized mask.
          gridOutS (tuple): (xV, yV, zV) for coordinates of output scan/mask.
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
    voxSizeV = [np.median(np.diff(xV)),np.median(np.diff(yV)), \
                np.median(np.diff(zV))]

    # Resize image using selected method
    methodLower = method.lower()
    if methodLower == 'padorcrop3d':
        # Pad or crop in 3D.
        # Decide based  on relative sizes of input and output dimensions.
        # Note: Padding or cropping is applied to ALL dimensions.

        if preserveAspectFlag:
            # Preserve aspect ratio by padding input image with mean background intensity
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

        # Determine resizing method
        xPad = int(np.floor((outputImgSizeV[0] - origSizeV[0]) / 2))
        if xPad < 0:
            resizeMethod = 'unpad3d'
        else:
            resizeMethod = 'pad3d'


        scanOut3M, maskOut4M, gridOutS = resizeScanAndMask(scan3M, mask4M, gridS, \
                                                           outputImgSizeV, resizeMethod, limitsM=None, \
                                                           preserveAspectFlag=False)

    elif methodLower == 'pad3d':
        # Pad rows and/or cols by same amount across all slices.

        # Determine padding amounts
        xPad = math.ceil((outputImgSizeV[0] - origSizeV[0]) / 2)
        yPad = math.ceil((outputImgSizeV[1] - origSizeV[1]) / 2)

        if xPad < 0 or yPad < 0:
            raise ValueError("To resize by padding, output image dimensions\
                  must be larger than (cropped) input image dimensions")

        # Pad scan
        if scan3M is None:
            scanOut3M = None
        else:
            minScanVal = np.min(scan3M)
            scanOut3M = np.full(outputImgSizeV, minScanVal, dtype=scan3M.dtype)
            scanOut3M[xPad:xPad + origSizeV[0], yPad:yPad + origSizeV[1], \
            :origSizeV[2]] = scan3M

        # Pad mask
        if mask4M is None:
            maskOut4M = None
        else:
            maskOut4M = np.zeros(outputImgSizeV + [numLabels])
            maskOut4M[xPad:xPad + origSizeV[0], yPad:yPad + origSizeV[1], \
            :origSizeV[2], :] = mask4M

        xOutV = np.arange(xV[0] - (xPad/2)*voxSizeV[0],
                          xV[-1]+(xPad/2)*voxSizeV[0], voxSizeV[0])
        yOutV = np.arange(yV[0] - (yPad/2)*voxSizeV[1], \
                          yV[-1]+(yPad/2)*voxSizeV[1], voxSizeV[1])
        zOutV = zV


        gridOutS = (xOutV, yOutV, zOutV)

    elif methodLower == 'unpad3d':
        # Crop rows and/or cols by same amount across all slices.

        #Determine crop extents
        xPad = - int(np.floor((outputImgSizeV[0] - origSizeV[0]) / 2))
        yPad = - int(np.floor((outputImgSizeV[1] - origSizeV[1]) / 2))

        if scan3M.size == 0:
            scanOut3M = []
        else:
            scanOut3M = scan3M[xPad:xPad + outputImgSizeV[0],
                        yPad:yPad + outputImgSizeV[1], :]

            xOutV = xV[xPad/2:-xPad/2]
            yOutV = yV[yPad/2:-yPad/2]
            zOutV = zV

        if mask4M is None:
            maskOut4M = None
        else:
            maskOut4M = mask4M[xPad:xPad + outputImgSizeV[0],
                        yPad:yPad + outputImgSizeV[1], :, :]
        gridOutS = (xOutV, yOutV, zOutV)

    elif methodLower=='pad2d':
        # Pad ROI by varying amounts across slices for identical output dimensions.

        # Initialize resized scan and mask
        if scan3M.size == 0:
            scanOut3M = np.array([])
        else:
            minScanVal = np.min(scan3M)
            scanOut3M = np.full([outputImgSizeV[0], outputImgSizeV[1], origSizeV[2]] \
                                , minScanVal, dtype=scan3M.dtype)

        if mask4M is None:
            maskOut4M = None
        else:
            maskOut4M = np.zeros((outputImgSizeV[0], outputImgSizeV[1],\
                                  numLabels), dtype=np.uint32)

        xOutM = np.zeros((outputImgSizeV[1],origSizeV[2]))
        yOutM = np.zeros((outputImgSizeV[0],origSizeV[2]))

        # Min/max row and col limits for each slice
        for slcNum in range(origSizeV[2]):
            minr = limitsM[slcNum, 0]
            maxr = limitsM[slcNum, 1]
            minc = limitsM[slcNum, 2]
            maxc = limitsM[slcNum, 3]

            rowCenter = (minr + maxr) / 2
            colCenter = (minc + maxc) / 2

            rMin = rowCenter - outputImgSizeV[0] // 2
            cMin = colCenter - outputImgSizeV[1] // 2

            if rMin < 0:
                rMin = 0
            if cMin < 0:
                cMin = 0

            rMax = rMin + outputImgSizeV[0] - 1
            cMax = cMin + outputImgSizeV[1] - 1

            rMin = np.ceil(rMin).astype(int)
            cMin = np.ceil(cMin).astype(int)
            rMax = np.ceil(rMax).astype(int)
            cMax = np.ceil(cMax).astype(int)

            if rMax > origSizeV[0]-1:
                rMax = origSizeV[0]-1
            if cMax > origSizeV[1]-1:
                cMax = origSizeV[1]-1

            outRmin = 0
            outCmin = 0
            outRmax = rMax - rMin
            outCmax = cMax - cMin

            if scan3M.size > 0:
                scanOut3M[outRmin:outRmax+1, outCmin:outCmax+1, slcNum] = \
                    scan3M[rMin:rMax+1, cMin:cMax+1, slcNum]

            if mask4M is not None:
                maskOut4M[outRmin:outRmax+1, outCmin:outCmax+1, slcNum, :] = \
                    mask4M[rMin:rMax+1, cMin:cMax+1, slcNum, :]

            xOutM[:,slcNum] = np.arange(xV[cMin], xV[cMax] + voxSizeV[0], voxSizeV[0])
            yOutM[:,slcNum] = np.arange(yV[rMin], yV[rMax] + voxSizeV[1], voxSizeV[1])
            gridOutS = (xOutM, yOutM, zV)

    elif methodLower=='unpad2d':
        # Initialize resized scan and mask
        if scan3M.size == 0:
            scanOut3M = np.array([])
        else:
            minScanVal = np.min(scan3M)
            scanOut3M = np.full([outputImgSizeV[0], outputImgSizeV[1], origSizeV[2]] \
                                , minScanVal, dtype=scan3M.dtype)

        if mask4M.size == 0:
            maskOut4M = np.array([])
        else:
            maskOut4M = np.zeros((outputImgSizeV[0], outputImgSizeV[1],\
                                  numLabels), dtype=np.uint32)

        xOutM = np.zeros((outputImgSizeV[1],origSizeV[2]))
        yOutM = np.zeros((outputImgSizeV[0],origSizeV[2]))

        # Min/max row and col limits for each slice
        for slcNum in range(origSizeV[2]):
            minr = limitsM[slcNum, 0]
            maxr = limitsM[slcNum, 1]
            minc = limitsM[slcNum, 2]
            maxc = limitsM[slcNum, 3]

            rowCenter = (minr + maxr) / 2
            colCenter = (minc + maxc) / 2

            rMin = rowCenter - origSizeV[0] // 2
            cMin = colCenter - origSizeV[1] // 2

            if rMin < 0:
                rMin = 0
            if cMin < 0:
                cMin = 0

            rMax = rMin + origSizeV[0]
            cMax = cMin + origSizeV[1]

            rMin = np.ceil(rMin).astype(int)
            cMin = np.ceil(cMin).astype(int)
            rMax = np.ceil(rMax).astype(int)
            cMax = np.ceil(cMax).astype(int)

            if rMax > outputImgSizeV[0] - 1:
                rMax = outputImgSizeV[0] - 1
            if cMax > outputImgSizeV[1] - 1:
                cMax = outputImgSizeV[1] - 1

            outRmin = 0
            outCmin = 0
            outRmax = rMax - rMin + 1
            outCmax = cMax - cMin + 1

            if scan3M.size != 0:
                scanOut3M[rMin:rMax, cMin:cMax, slcNum] = scan3M[outRmin:outRmax, outCmin:outCmax, slcNum]
            if mask4M.size != 0:
                maskOut4M[rMin:rMax, cMin:cMax, slcNum, :] = mask4M[outRmin:outRmax, outCmin:outCmax, slcNum, :]

            xOutM[:,slcNum] = np.arange(xV[cMin], xV[cMax], voxSizeV[0])
            yOutM[:,slcNum] = np.arange(yV[rMin], yV[rMax], voxSizeV[1])
            gridOutS = (xOutM, yOutM, zV)

    elif methodLower in ['bilinear','bicubic','nearest']:
        #3D resizing. TBD: 2D
        scanOut3M = None
        maskOut4M = None
        orderDict = {'bilinear': 1, 'bicubic': 3, 'nearest': 0} #Resizing order

        if preserveAspectFlag:
            # Preserve original aspect ratio by padding with background mean

            if scan3M is not None:
                # Resize scan
                cornerCube = scan3M[:5, :5, :5]
                bgMean = np.mean(cornerCube)
                scanSize = scan3M.shape
                padding = max(scanSize[:2])
                padded3M = bgMean * np.ones((padding, padding, scan3M.shape[2]))
                padSize = padded3M.shape()
                idx11 = 1 + (padding - scanSize[0]) // 2
                idx12 = idx11 + scanSize[0] - 1
                idx21 = 1 + (padding - scanSize[1]) // 2
                idx22 = idx21 + scanSize[1] - 1
                padded3M[idx11:idx12+1, idx21:idx22+1, :] = scan3M
                scanOut3M = np.empty((outputImgSizeV[0], outputImgSizeV[1], scan3M.shape[2]))
                for nSlc in range(padded3M.shape[2]):
                    scanOut3M[:, :, nSlc] = resize(padded3M[:, :, nSlc],
                                                   (outputImgSizeV[0], outputImgSizeV[1]),
                                                   anti_aliasing=True,
                                                   order=orderDict[methodLower])

            #Resize mask
            if mask4M is not None:
                minr = limitsM[0]
                maxr = limitsM[1]
                minc = limitsM[2]
                maxc = limitsM[3]
                cropDim = [maxr-minr+1, maxc-minc+1]
                padding = [max(cropDim[0:1])] * 2
                maskResize4M = np.zeros((*padding, mask4M.shape[2], mask4M.shape[3]))
                padSize = maskResize4M.shape()
                for nSlc in range(mask4M.shape[2]):
                    maskResize4M[:, :, nSlc, :] = resize(np.squeeze(mask4M[:, :, nSlc, :]),
                                                         padding,
                                                         order = 0,  # 'nearest' interpolation
                                                         anti_aliasing = False)

                idx11 = 1 + round((padding[0] - cropDim[0]) / 2)
                idx12 = idx11 + cropDim[0] - 1
                idx21 = 1 + round((padding[1] - cropDim[1]) / 2)
                idx22 = idx21 + cropDim[1] - 1
                maskOut4M = maskResize4M[idx11:idx12+1, idx21:idx22+1, :, :]

            #Padded coordinates
            xPadV = np.arange(xV[0] - (padding / 2) * voxSizeV[0],
                           xV[-1] + (padding / 2) * voxSizeV[0], voxSizeV[0])
            yPadV = np.arange(yV[0] - (padding / 2) * voxSizeV[1],
                           yV[-1] + (padding / 2) * voxSizeV[1], voxSizeV[1])
            # Rescaled coordinates
            scaleFactor = [outputImgSizeV[1] / padSize[1],
                           outputImgSizeV[0] / padSize[0]]
            xScaleV = xPadV * scaleFactor[0]
            yScaleV = yPadV * scaleFactor[1]
            xOutV = np.linspace(xScaleV.min(), xScaleV.max(), outputImgSizeV[1])
            yOutV = np.linspace(yScaleV.min(), yScaleV.max(), outputImgSizeV[0])
            gridOutS = (xOutV,yOutV,zV)

        else:
            # Resize scan
            if scan3M is not None:
                scanOut3M = np.empty((outputImgSizeV[0], outputImgSizeV[1], scan3M.shape[2]))
                for nSlc in range(scan3M.shape[2]):
                    scanOut3M[:, :, nSlc] = resize(scan3M[:, :, nSlc],
                                               (outputImgSizeV[:-1]),
                                               anti_aliasing=True,
                                               order=orderDict[methodLower])
            # Resize mask
            if mask4M is not None:
                maskOut4M = np.zeros((*outputImgSizeV[0:2], mask4M.shape[2], mask4M.shape[3]))
                for nSlc in range(mask4M.shape[2]):
                    maskOut4M[:, :, nSlc, :] = resize(np.squeeze(mask4M[:, :, nSlc, :]),
                                                      outputImgSizeV[:-1],
                                                      order=0,  # 'nearest' interpolation
                                                      anti_aliasing=False)

            # Rescaled coordinates
            scaleFactor = [outputImgSizeV[1] / origSizeV[1],\
                           outputImgSizeV[0] / origSizeV[0]]
            xScaleV = xV * scaleFactor[0]
            yScaleV = yV * scaleFactor[1]
            xOutV = np.linspace(xScaleV.min(), xScaleV.max(), outputImgSizeV[1])
            yOutV = np.linspace(yScaleV.min(), yScaleV.max(), outputImgSizeV[0])
            gridOutS = (xOutV,yOutV,zV)

    elif methodLower=='padslices':
        numSlices = outputImgSizeV[2]
        zPad = numSlices-origSizeV[2]

        scanOut3M = None
        if scan3M is not None:
            scanOut3M = np.zeros((scan3M.shape[0], scan3M.shape[1], numSlices))
            scanOut3M[:, :, :origSizeV[2]] = scan3M

        maskOut4M = None
        if mask4M is not None:
            maskOut4M = np.zeros((scan3M.shape[0], scan3M.shape[1], numSlices, numLabels))
            maskOut4M[:, :, :origSizeV[2], :] = mask4M

        zOutV = np.arange(zV[0], zV[-1]+(zPad+1)*voxSizeV[2], voxSizeV[2])
        gridOutS = (xV, yV, zOutV)

    elif methodLower=='unpadslices':
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
