import numpy as np
import cerr.contour.rasterseg as rs
from cerr.utils.bbox import compute_boundingbox
from scipy.interpolate import interpn
import SimpleITK as sitk
from scipy.ndimage import zoom
from scipy.ndimage import label
from scipy.ndimage import generate_binary_structure

def imquantize_cerr(x, num_level=None, xmin=None, xmax=None, binwidth=None):
    """
    Function to quantize an image based on the number of bins (num_level) or the bin width (bin_width).
    Args:
        x: Input image matrix.
        num_level: Number of quantization levels (optional).
        xmin: Minimum value for quantization (optional).
        xmax: Maximum value for quantization (optional).
        bin_width: Bin width for quantization (optional).
    Returns:
        q: Quantized image.
    Note: If xmin and xmax are not provided, they are computed from the input image x.
    """
    if xmin is not None:
        x[x < xmin] = xmin
    else:
        xmin = np.nanmin(x)

    if xmax is not None:
        x[x > xmax] = xmax
    else:
        xmax = np.nanmax(x)

    if num_level is not None:
        slope = (num_level - 1) / (xmax - xmin)
        intercept = 1 - (slope * xmin)
        q = np.round(slope * x + intercept)
        q[np.isnan(q)] = 0
        q = q.astype(np.int)
    elif binwidth is not None:
        q = (x - xmin) / binwidth
        q[np.isnan(q)] = -1
        q = q.astype(int) + 1
    else:
        # No quantization
        print('Returning input image. Specify the number of bins or the bin_width to quantize.')
        q = x

    return q


def getResampledGrid(resampResolutionV, xValsV, yValsV, zValsV, gridAlignMethod='center', *args):
    if not gridAlignMethod:
        gridAlignMethod = 'center'

    if args:
        # currently, perturbation of grid is not supported
        perturbV = args[0]
        perturbX, perturbY, perturbZ = perturbV[0], perturbV[1], perturbV[2]
    else:
        perturbX, perturbY, perturbZ = 0, 0, 0


    #Ensure ascending vals for interpolation
    flipY = 0
    if yValsV[0] > yValsV[1]:
        flipY = 1
        yValsV = np.flip(yValsV)

    # Input origin and resolution
    originV = [xValsV[0], yValsV[0], zValsV[-1]]
    dx = abs(np.median(np.diff(xValsV)))
    dy = abs(np.median(np.diff(yValsV)))
    dz = abs(np.median(np.diff(zValsV)))
    origResolutionV = [dx, dy, dz]

    origResolutionV[2] = -origResolutionV[2]
    resampResolutionV = resampResolutionV.copy()
    resampResolutionV[2] = -resampResolutionV[2]

    if len(resampResolutionV) == 3 and not np.isnan(resampResolutionV[2]):
        resamp3dFlag = True
    else:
        resamp3dFlag = False

    origSizeV = [len(xValsV), len(yValsV), len(zValsV)]
    resampSizeV = np.ceil(np.array(origSizeV) * np.array(origResolutionV) / np.array(resampResolutionV)).astype(int)

    if gridAlignMethod == 'center':
        resampOriginV = originV + (np.array(origResolutionV) * (np.array(origSizeV) - 1) \
                                   - np.array(resampResolutionV) * (np.array(resampSizeV) - 1)) / 2

        xResampleV = np.arange(resampOriginV[0], resampOriginV[0] \
                               + (resampSizeV[0] - 1) * resampResolutionV[0] \
                               + resampResolutionV[0], resampResolutionV[0])
        yResampleV = np.arange(resampOriginV[1], resampOriginV[1] \
                               + (resampSizeV[1] - 1) * resampResolutionV[1] \
                               + resampResolutionV[1], resampResolutionV[1])

        if resamp3dFlag:
            zResampleV = np.arange(resampOriginV[2], resampOriginV[2] \
                                   + (resampSizeV[2] - 1) * resampResolutionV[2] \
                                   + resampResolutionV[2], resampResolutionV[2])
            zResampleV = np.flip(zResampleV)
        else:
            zResampleV = zValsV
    else:
        raise ValueError(f'Unsupported grid alignment method {gridAlignMethod}')

    if flipY:
        yResampleV = np.flip(yResampleV)

    return xResampleV, yResampleV, zResampleV

def imgResample3D(img3M, xValsV, yValsV, zValsV, xResampleV, yResampleV, zResampleV, method, extrapVal=0):
    """

    Args:
        img3M: 3D numpy Array. e.g. planC.scan[scanNum].getScanArray()
        xValsV: 1D array of x-coordinates i.e. along columns of img3M. Must be monotonically increasing order as per CERR coordinate system.
        yValsV: 1D array of y-coordinates i.e. along rows of img3M. Must be monotonically decreasing order as per CERR coordinate system.
        zValsV: 1D array of z-coordinates i.e. along slices of img3M. Must be monotonically increasing order as per CERR coordinate system.
        xResampleV: 1D array of new x-coordinates i.e. along columns of resampled image
        yResampleV: 1D array of new y-coordinates i.e. along rows of resampled image
        zResampleV: 1D array of new z-coordinates i.e. along slices of resampled image
        method: string representing one of the supported methods from 'sitkNearestNeighbor', 'sitkLinear', 'sitkBSpline', 'sitkGaussian', 'sitkLabelGaussian',
                    'sitkHammingWindowedSinc', 'sitkCosineWindowedSinc', 'sitkWelchWindowedSinc',
                    'sitkLanczosWindowedSinc', 'sitkBlackmanWindowedSinc'

    Returns:
        3D numpy Array reampled at xResampleV, yResampleV, zResampleV
    """

    sitkMethods = {'sitkNearestNeighbor', 'sitkLinear', 'sitkBSpline', 'sitkGaussian', 'sitkLabelGaussian',
                    'sitkHammingWindowedSinc', 'sitkCosineWindowedSinc', 'sitkWelchWindowedSinc',
                    'sitkLanczosWindowedSinc', 'sitkBlackmanWindowedSinc'}

    img_from_sitk_3m = None
    if method in sitkMethods:
        # SimpleITK based interpolation
        # Flip along slices as CERR z slices increase from head to toe (opposite to DICOM)
        sitk_img = sitk.GetImageFromArray(np.flip(np.transpose(img3M, (2, 0, 1)), axis = 0))
        sitk_img.SetDirection((1,0,0,0,1,0,0,0,1))
        originV = xValsV[0],-yValsV[0],-zValsV[-1]
        spacing_v = xValsV[1]-xValsV[0], yValsV[0]-yValsV[1], zValsV[1]-zValsV[0]
        sitk_img.SetOrigin(originV)
        sitk_img.SetSpacing(spacing_v)

        resample_img_spacing_v = xResampleV[1]-xResampleV[0], \
                             yResampleV[0]-yResampleV[1], \
                             zResampleV[1]-zResampleV[0]
        resample_img_size_v = len(yResampleV),len(xResampleV),len(zResampleV)
        resample_orig_v = xResampleV[0],-yResampleV[0],-zResampleV[-1]
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(resample_img_spacing_v)
        resample.SetOutputDirection((1,0,0,0,1,0,0,0,1))
        resample.SetSize(resample_img_size_v)
        resample.SetOutputDirection(sitk_img.GetDirection())
        resample.SetOutputOrigin(resample_orig_v)
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(sitk_img.GetPixelIDValue())
        resample.SetUseNearestNeighborExtrapolator(True)

        # resample.SetInterpolator(sitk.sitkLinear)
        resample.SetInterpolator(getattr(sitk,method))

        sitk_resamp_img = resample.Execute(sitk_img)

        img_from_sitk_3m = sitk.GetArrayFromImage(sitk_resamp_img)
        img_from_sitk_3m = np.transpose(img_from_sitk_3m, (1, 2, 0))
        # flip slices in CERR z-slice order which increases from head to toe
        img_from_sitk_3m = np.flip(img_from_sitk_3m, axis=2)

    return img_from_sitk_3m

    # if method in {'linear', 'cubic', 'nearest', 'quintic', 'slinear', 'pchip'}:
    #     numRowsResamp = len(yResampleV)
    #     numColsResamp = len(xResampleV)
    #     numSlcsResamp = len(zResampleV)
    #     resamplingPtsM = np.meshgrid(yResampleV,xResampleV,zResampleV, indexing='ij')
    #     resamplingPtsM = np.array((resamplingPtsM[0].flatten(),\
    #                                resamplingPtsM[1].flatten(),\
    #                                resamplingPtsM[2].flatten()),dtype=float).T
    #     bounds_error = False
    #     resampimg3M = interpn((yValsV, xValsV, zValsV), img3M, resamplingPtsM, method, bounds_error, extrapVal)
    #     resampimg3M = np.reshape(resampimg3M,[numRowsResamp,numColsResamp,numSlcsResamp])


def padScan(scan3M, mask3M, method, marginV, cropFlag=True):
    """Return padded array using specified methosd and dimensions"""
    if cropFlag is None:
        cropFlag = True

    if method.lower() == 'none':
        marginV = [0, 0, 0]

    if len(marginV) == 2:
        marginV = np.append(marginV,[0])

    marginV = np.asarray(marginV, dtype = int)

    if cropFlag:
        minr, maxr, minc, maxc, mins, maxs, __ = compute_boundingbox(mask3M)
        croppedScan3M = scan3M[minr:maxr+1, minc:maxc+1, mins:maxs+1]
        croppedMask3M = mask3M[minr:maxr+1, minc:maxc+1, mins:maxs+1]
        minr = minr - marginV[0]
        maxr = maxr + marginV[0]
        minc = minc - marginV[1]
        maxc = maxc + marginV[1]
        mins = mins - marginV[2]
        maxs = maxs + marginV[2]
    else:
        inputSizeV = scan3M.shape
        minr, minc, mins = 0, 0, 0
        maxr = inputSizeV[0]
        maxc = inputSizeV[1]
        maxs = inputSizeV[2]
        minr = minr - marginV[0]
        maxr = maxr + marginV[0]
        minc = minc - marginV[1]
        maxc = maxc + marginV[1]
        mins = mins - marginV[2]
        maxs = maxs + marginV[2]
        croppedScan3M = scan3M
        croppedMask3M = mask3M

    outLimitsV = [minr, maxr, minc, maxc, mins, maxs]

    if method.lower() == 'expand':
        if not cropFlag:
            raise ValueError("Set crop_flag=True to use method 'expand'")
        minr = max(minr, 0)
        maxr = min(maxr, mask3M.shape[0] - 1)
        minc = max(minc, 0)
        maxc = min(maxc, mask3M.shape[1] - 1)
        mins = max(mins, 0)
        maxs = min(maxs, mask3M.shape[2] - 1)
        outLimitsV = [minr, maxr, minc, maxc, mins, maxs]
        outScan3M = scan3M[minr:maxr+1, minc:maxc+1, mins:maxs+1]
        outMask3M = mask3M[minr:maxr+1, minc:maxc+1, mins:maxs+1]
    elif method.lower() == 'padzeros':
        outScan3M = np.pad(croppedScan3M, ((marginV[0],marginV[0]),(marginV[1],marginV[1]),
                                               (marginV[2],marginV[2])), mode='constant')
        outMask3M = np.pad(croppedMask3M, ((marginV[0],marginV[0]),(marginV[1],marginV[1]),
                                               (marginV[2],marginV[2])), mode='constant')
    elif method.lower() in ['periodic', 'nearest', 'mirror']:
        mode = {'periodic': 'wrap', 'nearest': 'edge', 'mirror': 'symmetric'}[method.lower()]
        outScan3M = np.pad(croppedScan3M, ((marginV[0],marginV[0]),(marginV[1],marginV[1]),
                                               (marginV[2],marginV[2])), mode=mode)
        outMask3M = np.pad(croppedMask3M, ((marginV[0],marginV[0]),(marginV[1],marginV[1]),
                                               (marginV[2],marginV[2])), mode='constant')
    elif method.lower() == 'none':
        outScan3M = croppedScan3M
        outMask3M = croppedMask3M
    else:
        raise ValueError(f"Invalid method '{method}'. Supported methods include 'expand',"
                          f" 'padzeros', 'periodic', 'nearest', 'mirror', and 'none'.")

    return outScan3M, outMask3M, outLimitsV


def unpadScan(padScan3M, method, marginV):
    """Return original array after stripping off specified padding margin"""

    if len(marginV) == 2:
        marginV = np.append(marginV,[0])

    padScanSizeV = np.shape(padScan3M)

    scan3M = padScan3M[marginV[0] : padScanSizeV[0]-marginV[0],
                          marginV[1] : padScanSizeV[1]-marginV[1],
                          marginV[2] : padScanSizeV[2]-marginV[2]]
    return scan3M


def preProcessForRadiomics(scanNum, structNum, paramS, planC):

    diagS = {}
    scanArray3M = planC.scan[scanNum].getScanArray()

    if isinstance(structNum, (int, float)):
        # Get structure mask
        mask3M = rs.getStrMask(structNum,planC)
    else:
        #Input is structure mask
        mask3M = structNum
    xValsV, yValsV, zValsV = planC.scan[scanNum].getScanXYZVals()
    if yValsV[0] > yValsV[1]:
        yValsV = np.flip(yValsV)

    # Get pixelSpacing of the new grid
    cropForResamplingFlag = False
    if 'resample' in paramS["settings"] and len(paramS["settings"]['resample']) > 0:
        pixelSpacingX, pixelSpacingY, pixelSpacingZ = \
            paramS["settings"]['resample']['resolutionXCm'], \
            paramS["settings"]['resample']['resolutionYCm'], \
            paramS["settings"]['resample']['resolutionYCm']
        roiInterpMethod = 'sitkLinear' # always linear interp for mask
        scanInterpMethod = paramS["settings"]['resample']['interpMethod'] #'sitkLinear' #whichFeatS.resample.interpMethod
        if scanInterpMethod == "linear":
            scanInterpMethod = 'sitkLinear'
        if scanInterpMethod == "bspline":
            scanInterpMethod = 'sitkBSpline'
        if scanInterpMethod == "sinc":
            scanInterpMethod = 'sitkLanczosWindowedSinc'
        grid_resample_method = 'center'
        maskInterpTol = 1e-8 # err on the side of including a voxel within this value from 0.5.
        if 'cropForResampling' in paramS["settings"]['resample']:
           cropForResamplingFlag = paramS["settings"]['resample']['cropForResampling']
    else:
        pixelSpacingX = np.absolute(np.median(np.diff(xValsV)))
        pixelSpacingY = np.absolute(np.median(np.diff(yValsV)))
        pixelSpacingZ = np.absolute(np.median(np.diff(zValsV)))

    outputResV = np.array([pixelSpacingX, pixelSpacingY, pixelSpacingZ])

    #Crop and pad scan for resampling
    #Get padding settings
    padSizeV = []
    padMethod = 'none'
    if cropForResamplingFlag:
        padMethod = 'expand'     #Default:Pad by 5 voxels (from original image) before resampling
        padSizeV = [5,5,5]
        if 'padding' in paramS["settings"] and len(paramS["settings"]['padding']) > 0:
            padMethod = paramS["settings"]['padding'][0]['method']
            padSizeV = paramS["settings"]['padding'][0]['size']

            inputResV = [np.median(xValsV),np.median(yValsV),np.median(zValsV)]
            if inputResV[0] * padSizeV[1] < outputResV[0] * padSizeV[1]:
                padSizeV[1] = np.ceil(outputResV[0] * padSizeV[1] / inputResV[0])
            if inputResV[1] * padSizeV[0] < outputResV[1] * padSizeV[0]:
                padSizeV[0] = np.ceil(outputResV[1] * padSizeV[0] / inputResV[1])
            if inputResV[2] * padSizeV[2] < outputResV[2] * padSizeV[2]:
                padSizeV[1] = np.ceil(outputResV[2] * padSizeV[1] / inputResV[2])

    #Crop to ROI and pad
    (padScanBoundsForResamp3M,padMaskBoundsForResamp3M,outLimitsV) = \
            padScan(scanArray3M, mask3M, padMethod, padSizeV, cropForResamplingFlag)
    xValsV = xValsV[outLimitsV[2]:outLimitsV[3]+1]
    yValsV = yValsV[outLimitsV[0]:outLimitsV[1]+1]
    zValsV = zValsV[outLimitsV[4]:outLimitsV[5]+1]

    # Interpolate image as defined in settings file
    if 'resample' in paramS["settings"] and len(paramS["settings"]['resample']) > 0:

        #Get resampling grid
        [xResampleV,yResampleV,zResampleV] = getResampledGrid(outputResV,\
                                                 xValsV, yValsV, zValsV, grid_resample_method)
        #Resample scan
        resampScanBounds3M = imgResample3D(padScanBoundsForResamp3M, xValsV, yValsV, zValsV,\
                                xResampleV, yResampleV, zResampleV, scanInterpMethod)
        # Round image intensities
        if 'intensityRounding' in paramS["settings"]['resample'] and \
                paramS["settings"]['resample']['intensityRounding'].lower =='on':
            resampScanBounds3M = np.round(resampScanBounds3M)

        #Resample mask
        resampMaskBounds3M = imgResample3D(padMaskBoundsForResamp3M.astype(float),xValsV,yValsV,zValsV,\
            xResampleV,yResampleV,zResampleV,roiInterpMethod) >= (0.5 - maskInterpTol)
        #maskBoundingBox3M = maskBoundingBox3M.astype(bool)

    else:
        xResampleV = xValsV
        yResampleV = yValsV
        zResampleV = zValsV
        resampScanBounds3M = padScanBoundsForResamp3M
        resampMaskBounds3M = padMaskBoundsForResamp3M


    # Pad scan as required for convolutional filtering
    filtPadMethod = 'none'
    filtPadSizeV = [0,0,0]
    cropFlag = True
    if not cropForResamplingFlag:
        if 'padding' in paramS['settings']:
            if 'cropToMaskBounds' in  paramS['settings']['padding']:
                cropFlag = paramS['settings']['padding']['cropToMaskBounds'].lower() == 'yes'
            if 'method' in paramS['settings']['padding'] and paramS['settings']['padding']['method'].lower()!='none':
                filtPadMethod = paramS['settings']['padding']['method']
                filtPadSizeV = paramS['settings']['padding']['size']
                if len(filtPadSizeV)==2:
                    filtPadSizeV = [filtPadSizeV,0]
        [volToEval,maskBoundingBox3M,outLimitsV] = padScan(resampScanBounds3M,\
        resampMaskBounds3M,filtPadMethod,filtPadSizeV,cropFlag)

        # Extend resampling grid if padding original image (cropFlag: False)
        if outLimitsV[0]<0:
            numPad = -outLimitsV[0]
            padCountV = np.arange(numPad,0,-1)
            yExtendV = yResampleV[0]-padCountV*outputResV[1]
            yResampleV = np.concatenate((yExtendV,yResampleV))
            outLimitsV[0] = outLimitsV[0] + numPad
            outLimitsV[1] = outLimitsV[1] + numPad

        if outLimitsV[2]<0:
            numPad = -outLimitsV[2]
            padCountV = np.arange(numPad,0,-1)
            xExtendV = xResampleV[0]-padCountV*outputResV[0]
            xResampleV = np.concatenate((xExtendV,xResampleV))
            outLimitsV[2] = outLimitsV[2] + numPad
            outLimitsV[3] = outLimitsV[3] + numPad

        if outLimitsV[4]<0:
            numPad = -outLimitsV[4]
            padCountV = np.arange(numPad,0,-1)
            zExtendV = zResampleV[0]-padCountV*outputResV[2]
            zResampleV = np.concatenate((zExtendV,zResampleV))
            outLimitsV[4] = outLimitsV[4] + numPad
            outLimitsV[5] = outLimitsV[5] + numPad

        if outLimitsV[1]>len(yResampleV)-1:
            numPad = outLimitsV[1] - len(yResampleV) + 1
            padCountV = np.arange(1,numPad+1,1)
            yExtendV = yResampleV[-1] + padCountV*outputResV[1]
            yResampleV = np.concatenate((yResampleV,yExtendV))

        if outLimitsV[3]>len(xResampleV)-1:
            numPad = outLimitsV[3] - len(xResampleV) + 1
            padCountV = np.arange(1,numPad+1,1)
            xExtendV = xResampleV[-1] + padCountV*outputResV[0]
            xResampleV = np.concatenate((xResampleV,xExtendV))

        if outLimitsV[5]>len(zResampleV):
            numPad = outLimitsV[5] - len(zResampleV) + 1
            padCountV = np.arange(1,numPad+1,1)
            zExtendV = zResampleV[-1] + padCountV*outputResV[2]
            zResampleV = np.concatenate((zResampleV,zExtendV))

        xResampleV = xResampleV[outLimitsV[2]:outLimitsV[3]+1]
        yResampleV = yResampleV[outLimitsV[0]:outLimitsV[1]+1]
        zResampleV = zResampleV[outLimitsV[4]:outLimitsV[5]+1]

    else:
        volToEval = resampScanBounds3M
        maskBoundingBox3M = resampMaskBounds3M


    # Ignore voxels below and above cutoffs, if defined ----
    minSegThreshold = []
    maxSegThreshold = []
    if 'texture' in paramS['settings']:
        if 'minSegThreshold' in paramS['settings']['texture']:
            minSegThreshold = paramS['settings']['texture']['minSegThreshold']
        if 'maxSegThreshold' in paramS['settings']['texture']:
            maxSegThreshold = paramS['settings']['texture']['maxSegThreshold']
        if isinstance(minSegThreshold,int) or isinstance(minSegThreshold,float):
            maskBoundingBox3M[volToEval < minSegThreshold] = 0
        if isinstance(maxSegThreshold,int) or isinstance(maxSegThreshold,float):
            maskBoundingBox3M[volToEval > maxSegThreshold] = 0

    # Record diagnostic stats
    scanV = volToEval[maskBoundingBox3M]
    diagS['numVoxelsInterpReseg'] = maskBoundingBox3M.sum()
    diagS['meanIntensityInterpReseg'] = scanV.mean()
    diagS['maxIntensityInterpReseg'] = scanV.max()
    diagS['minIntensityInterpReseg'] = scanV.min()

    # Return output image grid
    gridS = {'xValsV': xResampleV,
             'yValsV': yResampleV,
             'zValsV': zResampleV,
             'PixelSpacingV': outputResV}

    return volToEval, maskBoundingBox3M, gridS, paramS, diagS
