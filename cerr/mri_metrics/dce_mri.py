import numpy as np
from scipy.signal import resample
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt
from cerr.contour.rasterseg import getStrMask
from cerr.utils.statistics import round


def loadTimeSeq(planC, structNum):
    """loadTimeSeq
    Function to extract 4D DCE scan array associated with input structure from planC

    Args:
        planC (plan_container.planC): pyCERR's plan container object
        structNum (int): Index of structure in planC

    Returns:
        scanArr4M (np.ndarray, 4D)  : DCE array (nRows x nCols x nROISlc x nTime)
        timePtsV (np.array, 1D)     : Acquisition times (min)
        maskSlc3M (np.ndarray, 3D)  : Mask of ROI (nRows x nCols x nROISlc)
        maskSlcV (np.array, 1D)     : Indices of slices in the ROI (1 x nROISlc)
    """
    numTimePts = len(planC.scan)
    mask3M = getStrMask(structNum, planC)
    maskSlcV = sum(sum(mask3M)) > 0
    maskSlc3M = mask3M[:, :, maskSlcV]
    numSlc = maskSlcV.sum()

    scanSizeV = planC.scan[0].getScanSize()
    scanArr4M = np.zeros((scanSizeV[0], scanSizeV[1], numSlc, numTimePts))
    timePtsV = np.array([planC.scan[scn].scanInfo[0].triggerTime for scn in range(len(planC.scan))])

    for slc in range(len(maskSlcV)):
        scanArr4M[:, :, slc, :] = np.array([planC.scan[scn].getScanArray()[:, :, slc] for scn in range(numTimePts)])

    return scanArr4M, timePtsV, maskSlc3M, maskSlcV


def getStartofUptake(slice3M, maskM):
    """getStartofUptake
    Function to plot sample uptake curve for interactive selection of baseline points

    Args:
        slice3M (np.ndarray, 3D)  : 3D array containing time sequence of scan slice (nRows x nCols x nTime)
        maskM (np.ndarray, 2D)    : Mask of ROI slice

    Returns:
        basePts (int) : Time point representing start of uptake
    """

    # Compute mean ROI intensity at each time point
    roiSize = maskM.sum()
    mask3M = np.repeat(maskM[:, :, np.newaxis], slice3M.shape[2], axis=2)
    slice3M[mask3M] = np.nan
    meanSigV = np.nansum(np.nansum(slice3M, axis=0), axis=0) / roiSize
    timePtsV = np.arange(0, len(meanSigV))

    # Interactive selection of baseline pts
    plt.plot(timePtsV, meanSigV, marker='o')
    for i, (x, y) in enumerate(zip(timePtsV, meanSigV)):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    plt.show(block=True)

    basePts = input("Enter timepoint representing start of uptake: ")
    # Try to convert input to float, handle invalid input
    try:
        basePts = int(basePts)
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return None

    return basePts


def normalizeToBaseline(scanArr4M, mask3M, timePtsV, basePts=None, imgSmoothDict=None):
    """normalizeToBaseline
    Function to normalize DCE signal to avg. baseline value

    Args:
        scanArr4M (np.ndarray, 4D) : DCE array (nRows x nCols x nROISlc x nTime)
        timePtsV (np.array, 1D)    : Acquisition times (min)
        maskSlc3M (np.ndarray, 3D) : Mask of ROI (nRows x nCols x nROISlc)
        basePts (int): [optional, default:None] Time pt. representing start of uptake.
                       By default, have user input value.
        imgSmoothDict (dict)       : [optional, default:None] Dictionary specifying whether to
                                     smooth image & associated filter parameters.
                                     Keys:  'smoothFlag', 'kernelSize', 'sigma'

    Returns:
        scanArr4M (np.ndarray) : DCE array (nRows x nCols x nROISlc x nTime)
        timePtsV (np.array)    : Acquisition times (min)
        normScan4M(np.ndarray) : Normalized scan array (nRows x nCols x nROISlc x nUptakeTime)
        uptakeTimeV            : Acquisition times for uptake (min) (1 x nUptakeTime)
    """

    smoothFlag = False
    if imgSmoothDict is not None:
        params = imgSmoothDict.keys()
        smoothFlag = imgSmoothDict['smoothFlag']
        if 'kernelSize' in params:
            fSize = imgSmoothDict['kernelSize']
        if 'sigma' in params:
            fSigma = imgSmoothDict['sigma']

    numSlc = scanArr4M.shape[2]
    nTimePts = scanArr4M.shape[3]

    normScan4M = np.zeros(scanArr4M.shape)
    for slc in range(numSlc):
        slcSeq3M = scanArr4M[:, :, slc, :]
        # if smoothFlag:
        # Smoothing
        maskSlc3M = np.repeat(mask3M[:, :, slc, np.newaxis], nTimePts, axis=2)
        slcSeq3M[~maskSlc3M] = np.nan
        if slc == 0 and basePts is None:
            # Get user - input shift to start of uptake curve
            midSlc = round(numSlc / 2)
            midSliceSeq3M = scanArr4M[:, :, midSlc, :]
            midSlcMaskM = mask3M[:, :, midSlc]
            basePts = getStartofUptake(midSliceSeq3M, midSlcMaskM)

        maskedSlcSeq3M = np.ma.masked_invalid(slcSeq3M[:, :, 0:basePts])  #Prevents RuntimeWarning: Mean of empty slice
        baselineM = np.mean(maskedSlcSeq3M, axis=2).filled(np.nan)
        baselineM[baselineM == 0] = np.finfo(float).eps
        normScan4M[:, :, slc, :] = scanArr4M[:, :, slc, :] / baselineM[:, :, np.newaxis]

    timePtsV = timePtsV - timePtsV[basePts]
    uptakeTimeV = timePtsV[basePts:]
    normScan4M = normScan4M[:, :, :, basePts:]

    return normScan4M, uptakeTimeV


def locatePeak(sigM):
    """locatePeak
    Function to locate peak of uptake curve

    Args:
        sigM (np.ndarray, 2D) : Uptake curves (nVox x nUptakeTime)

    Returns:
        peakIdxV (np.array)   : Indices corresponding to peak of uptake (1 x nVox)
    """

    nVox = sigM.shape[0]
    sigMax = 0.8 * np.max(sigM, axis=0)

    diffNextM = np.concatenate((np.zeros((nVox, 1)), np.diff(sigM, 1, 1)), axis=1)
    diffPrevM = np.concatenate((-np.diff(sigM, 1, 1), np.zeros((nVox, 1))), axis=1)
    localMaxIdxM = diffNextM >= 0 and diffPrevM >= 0
    highSigIdxM = sigM > sigMax

    allPeaksM = localMaxIdxM and highSigIdxM
    peakIdxV = np.argmax(allPeaksM, axis=1)
    return peakIdxV

def smoothResample(sigM, timeV, temporalSmoothDict=None, resampFlag=False):
    #TBD
    return 0


def computeFeatures(procSlcSigM, procTimeV):
    # TBD
    return 0


def calcSemiQuantFeatures(planC, structNum, basePts=None, temporalSmoothDict=None,
                          imgSmoothDict=None, resampFlag=False):

    # Load DCE series
    scanArr4M, timePtsV, mask3M, maskSlcV = loadTimeSeq(planC, structNum)
    numTimePts = len(timePtsV)
    scanSize = planC.scan[0].getScanSize()

    # Normalize to baseline
    normScan4M, selTimePtsV = normalizeToBaseline(scanArr4M, mask3M, timePtsV, basePts, imgSmoothDict=None)

    # Loop over ROI slices
    featureList = []
    for slc in range(len(maskSlcV)):
        # Reshape to 2D array (nVox x nTimePts)
        normSlc3M = normScan4M[:, :, slc, :]
        normSlcSigM = normSlc3M.reshape(-1, normSlc3M.shape[2], order='F')  # column major

        # Pre-process
        #procSlcSigM, procTimeV = smoothResample(normSlcSigM, selTimePtsV,
        #                                        temporalSmoothDict=temporalSmoothDict, resampFlag=resampFlag)
        #featureDict = computeFeatures(procSlcSigM, procTimeV)
        #featureList.append(featureDict)

    return featureList