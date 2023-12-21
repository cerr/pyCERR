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
        sitk_img = sitk.GetImageFromArray(np.moveaxis(np.flip(img3M,axis=2),[0,1,2],[2,1,0]))
        sitk_img.SetDirection((1,0,0,0,1,0,0,0,1))
        #img_from_sitk_3m = sitk.GetArrayFromImage(sitk_img)
        #img_from_sitk_3m = np.moveaxis(img_from_sitk_3m,[0,1,2],[2,1,0])
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
        img_from_sitk_3m = np.moveaxis(img_from_sitk_3m,[0,1,2],[2,1,0])
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

    if cropFlag:
        minr, maxr, minc, maxc, mins, maxs, __ = compute_boundingbox(mask3M)
        croppedScan3M = scan3M[minr:maxr+1, minc:maxc+1, mins:maxs+1]
        croppedMask3M = mask3M[minr:maxr+1, minc:maxc+1, mins:maxs+1]
        minr = max(minr - marginV[0], 0)
        maxr = min(maxr + marginV[0], mask3M.shape[0] - 1)
        minc = max(minc - marginV[1], 0)
        maxc = min(maxc + marginV[1], mask3M.shape[1] - 1)
        mins = max(mins - marginV[2], 0)
        maxs = min(maxs + marginV[2], mask3M.shape[2] - 1)
    else:
        inputSizeV = scan3M.shape
        minr, minc, mins = 0, 0, 0
        maxr = inputSizeV[0]
        maxc = inputSizeV[1]
        maxs = inputSizeV[2]
        minr = max(minr - marginV[0], 0)
        maxr = min(maxr + marginV[0], mask3M.shape[0] - 1)
        minc = max(minc - marginV[1], 0)
        maxc = min(maxc + marginV[1], mask3M.shape[1] - 1)
        mins = max(mins - marginV[2], 0)
        maxs = min(maxs + marginV[2], mask3M.shape[2] - 1)
        croppedScan3M = scan3M
        croppedMask3M = mask3M

    outLimitsV = [minr, maxr, minc, maxc, mins, maxs]

    if method.lower() == 'expand':
        if not cropFlag:
            raise ValueError("Set crop_flag=True to use method 'expand'")
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
    #if yValsV[0] > yValsV[1]:
    #    yValsV = np.flip(yValsV)

    # Minimum padding size if 5,5,5
    minPadSizeV = [5,5,5]
    cropForResamplingFlag = True

    # Get pixelSpacing of the new grid
    if 'resample' in paramS["settings"] and len(paramS["settings"]['resample']) > 0:
        pixel_spacing_x, pixel_spacing_y, pixel_spacing_z = \
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
    else:
        pixel_spacing_x = np.median(xValsV)
        pixel_spacing_y = np.median(yValsV)
        pixel_spacing_z = np.median(zValsV)
        xResampleV = xValsV
        yResampleV = yValsV
        zResampleV = zValsV

    outputResV = [pixel_spacing_x, pixel_spacing_y, pixel_spacing_z]

    # Pad and crop
    if 'padding' in paramS["settings"] and len(paramS["settings"]['padding']) > 0:
        pad_method = 'expand'
        pad_method = paramS["settings"]['padding'][0]['method']
        pad_siz_v = paramS["settings"]['padding'][0]['size']

        input_res_v = [np.median(xValsV),np.median(yValsV),np.median(zValsV)]
        if input_res_v[0] * pad_siz_v[1] < outputResV[0] * pad_siz_v[1]:
            pad_siz_v[1] = np.ceil(outputResV[0] * pad_siz_v[1] / input_res_v[0])
        if input_res_v[1] * pad_siz_v[0] < outputResV[1] * pad_siz_v[0]:
            pad_siz_v[0] = np.ceil(outputResV[1] * pad_siz_v[0] / input_res_v[1])
        if input_res_v[2] * pad_siz_v[2] < outputResV[2] * pad_siz_v[2]:
            pad_siz_v[1] = np.ceil(outputResV[2] * pad_siz_v[1] / input_res_v[2])

        (padScanBoundsForResamp3M,padMaskBoundsForResamp3M,outLimitsV) = \
            padScan(scanArray3M, mask3M, pad_method, pad_siz_v, cropForResamplingFlag)
        xValsV = xValsV[outLimitsV[2]:outLimitsV[3]+1]
        yValsV = yValsV[outLimitsV[0]:outLimitsV[1]+1]
        zValsV = zValsV[outLimitsV[4]:outLimitsV[5]+1]


    # Interpolate using the method defined in settings file
    if 'resample' in paramS["settings"] and len(paramS["settings"]['resample']) > 0:

        #Get resampling grid
        [xResampleV,yResampleV,zResampleV] = getResampledGrid(outputResV,\
                                                 xValsV, yValsV, zValsV, grid_resample_method)

        gridS = {'xValsV': xResampleV,
                 'yValsV': yResampleV,
                 'zValsV': zResampleV,
                 'pixelSpacingV': outputResV}

        #Resample scan
        resampScanBounds3M = imgResample3D(padScanBoundsForResamp3M, xValsV, yValsV, zValsV,\
                                xResampleV, yResampleV, zResampleV, scanInterpMethod)

        #Resample mask
        maskBoundingBox3M = imgResample3D(padMaskBoundsForResamp3M.astype(float),xValsV,yValsV,zValsV,\
            xResampleV,yResampleV,zResampleV,roiInterpMethod) >= (0.5 - maskInterpTol)
        #maskBoundingBox3M = maskBoundingBox3M.astype(bool)

    else:
        resampScanBounds3M = padScanBoundsForResamp3M
        maskBoundingBox3M = padMaskBoundsForResamp3M
        gridS = {'xValsV': xResampleV,
                 'yValsV': yResampleV,
                 'zValsV': zResampleV,
                 'pixelSpacingV': outputResV}


    # Ignore voxels below and above cutoffs, if defined ----
    minSegThreshold = []
    maxSegThreshold = []
    if 'texture' in paramS['settings']:
        if 'minSegThreshold' in paramS['settings']['texture']:
            minSegThreshold = paramS['settings']['texture']['minSegThreshold']
        if 'maxSegThreshold' in paramS['settings']['texture']:
            maxSegThreshold = paramS['settings']['texture']['maxSegThreshold']
        if isinstance(minSegThreshold,int) or isinstance(minSegThreshold,float):
            maskBoundingBox3M[resampScanBounds3M < minSegThreshold] = 0
        if isinstance(maxSegThreshold,int) or isinstance(maxSegThreshold,float):
            maskBoundingBox3M[resampScanBounds3M > maxSegThreshold] = 0

    scanV = resampScanBounds3M[maskBoundingBox3M]
    diagS['numVoxelsInterpReseg'] = maskBoundingBox3M.sum()
    diagS['meanIntensityInterpReseg'] = scanV.mean()
    diagS['maxIntensityInterpReseg'] = scanV.max()
    diagS['minIntensityInterpReseg'] = scanV.min()

    return resampScanBounds3M, maskBoundingBox3M, gridS, paramS, diagS
