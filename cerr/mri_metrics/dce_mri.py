import os
from math import degrees, atan
import numpy as np
from matplotlib import pyplot as plt

from scipy.signal import resample, savgol_filter, medfilt
from scipy.ndimage import gaussian_filter
from scipy.integrate import cumtrapz

from cerr import plan_container as pc
from cerr.contour.rasterseg import getStrMask
from cerr.utils.statistics import round

EPS = np.finfo(float).eps
rng = np.random.default_rng()


def loadTimeSeq(planC, structNum, userInputTime=None):
    """loadTimeSeq
    Function to extract 4D DCE scan array associated with input structure from planC

    Args:
        planC (plan_container.planC): pyCERR's plan container object
        structNum (int): Index of structure in planC
        userInputTime (np.array, float): [optional. default=None, read acquisitionTime]
                                         Set to True for user-input acquisition times

    Returns:
        scanArr4M (np.ndarray, 4D)  : DCE array (nRows x nCols x nROISlc x nTime)
        timePtsV (np.array, 1D)     : Acquisition times (min)
        maskSlc3M (np.ndarray, 3D)  : Mask of ROI (nRows x nCols x nROISlc)
        maskSlcV (np.array, 1D)     : Indices of slices in the ROI (1 x nROISlc)
    """
    numTimePts = len(planC.scan)
    mask3M = getStrMask(structNum, planC)
    maskSlcIdxV = np.sum(np.sum(mask3M, axis=0), axis=0) > 0
    maskSlcV = np.array(np.where(maskSlcIdxV)[0])
    maskSlc3M = mask3M[:, :, maskSlcIdxV]
    numSlc = len(maskSlcV)

    # Extract uptake curves for voxels in ROI
    scanSizeV = planC.scan[0].getScanSize()
    scanArr4M = np.zeros((scanSizeV[0], scanSizeV[1], numSlc, numTimePts))
    for slc in range(numSlc):
        scanSlc3M = np.array([scn.getScanArray()[:, :, maskSlcV[slc]] for scn in planC.scan])
        scanArr4M[:, :, slc, :] = np.moveaxis(scanSlc3M, 0, -1)

    timePtsV = userInputTime
    if userInputTime is None:
        timePtsV = np.array([planC.scan[scn].scanInfo[0].acquisitionTime for scn in range(numTimePts)])

    # Sort time pts
    indSortedV = np.argsort(timePtsV)
    timePtsV = timePtsV[indSortedV]
    scanArr4M = scanArr4M[:, :, :, indSortedV]

    return scanArr4M, timePtsV, maskSlc3M, maskSlcV


def intToConc(normSigM, concDict):
    """
    Converts DCE-MRI signal intensity into contrast agent concentration.

    Args:
        normSigM (tuple, float): Array of normalized intensities (S(t)/S(0))
        concDict (dict): Dictionary specifying
            clip_between (float array): Clip normalized intensities (intensity/baseline)
                                        between specified mon,max values
            T10 (float): Pre-contrast longitudinal relaxation time
            FA (float): Flip angle (degrees)
            TR (float): Repetition time (seconds)
            r1 (float): Relaxivity

    Returns:
        Concentration (C) in mmol/L and R1 map

    Ref.: Heilmann, M. et al. (2006) "Determination of pharmacokinetic parameters in DCE MRI:
          consequence of nonlinearity between contrast agent concentration and signal intensity."
          Investigative radiology 41.6: 536-543.
    """

    T10 = concDict['T10']
    TR = concDict['TR']
    FA = concDict['FA']
    r1 = concDict['r1']

    R10 = 1.0 / T10  # Relaxation rate before contrast

    # Apply threshold to normalized signal
    skipIdxV = np.nansum(normSigM, axis=1) == 0
    zeroIdxV = np.sum(normSigM, axis=1) == 0
    validNormSigM = normSigM[~skipIdxV, :]
    if 'clip_between' in concDict:
        normThreshV = concDict['clip_between']
        validNormSigM[validNormSigM < normThreshV[0]] = normThreshV[0]
        validNormSigM[validNormSigM > normThreshV[1]] = normThreshV[1]

    D = validNormSigM * (1 - np.exp(-TR * R10)) / (1 - np.exp(-TR * R10) * np.cos(np.radians(FA)) + EPS)
    with np.errstate(invalid='ignore'):
        Dr = (1 - D) / (1 - D * np.cos(np.radians(FA)) + EPS)
        R1 = -1 / TR * np.log(Dr)
        R1[Dr <= 0] = 0

    # Remove complex values
    #R1[np.iscomplex(R1)] = 0

    # Concentration
    C = np.full(normSigM.shape, np.nan)
    C[~skipIdxV, :] = 1 / r1 * (R1 - R10)
    C[np.iscomplex(C)] = 0
    C[zeroIdxV,:] = 0
    C[C < 0] = 0

    return C


def plotUptake(timePtsV, sigV, blockFlag, savePath=None):
    """plotUptake
    Function to plot sample uptake curve for interactive selection of baseline points
    """

    # Interactive selection of baseline pts
    plt.plot(timePtsV, sigV, marker='o')
    for i, (x, y) in enumerate(zip(timePtsV, sigV)):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    plt.xlabel('Time point')
    plt.ylabel('ROI mean signal intensity')
    plt.title('Select start of uptake')
    plt.show(block=blockFlag)

    if savePath is not None:
        plt.savefig(savePath)
        plt.close()

    return 0


def getStartofUptake(slice3M, maskM):
    """getStartofUptake
    Function for interactive selection of baseline points

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
    plt.xlabel('Time point')
    plt.ylabel('ROI mean signal intensity')
    plt.title('Select start of uptake')
    plt.show(block=True)
    basePts = input("Enter timepoint representing start of uptake: ")
    # Try to convert input to float, handle invalid input
    try:
        basePts = int(basePts)
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return None

    return basePts


def normalizeToBaseline(scanArr4M, mask3M, timePtsV, basePts=None, imgSmoothDict=None, enhThresh=None,
                        method='RSE', concDict=None):
    """normalizeToBaseline
    Function to normalize DCE signal to avg. baseline value

    Args:
        scanArr4M (np.ndarray, 4D) : DCE array (nRows x nCols x nROISlc x nTime)
        mask3M (np.ndarray, 3D) : Mask of ROI (nRows x nCols x nROISlc)
        timePtsV (np.array, 1D)    : Acquisition times (min)
        basePts (int): [optional, default:None] Time pt. representing start of uptake.
                       By default, have user input value.
        imgSmoothDict (dict)       : [optional, default:None] Dictionary specifying Gaussian
                                     smoothing filter parameters. If specified, keys
                                     'kernelSize' and 'sigma' must be present.
        enhThresh (float): [optional, default: None] Intensity threshold to identify enhancing voxels.
                           Voxels with peak intensity < thresh*baseline are excluded from analysis.
        method (string): [optional, default:'RSE'] Convert intensities to relative signal
                         enhancement ('RSE') or contrast concentration (CC)

        concDict (dict): [optional, default:None] Required if method='CC'. Dictionary of parameters
                         required to compute contrast agent concentration. Required keys: 'clip_above',
                         'T10', 'flipAngle', 'TR', 'r1'.

    Returns:
        scanArr4M (np.ndarray) : DCE array (nRows x nCols x nROISlc x nTime)
        timePtsV (np.array)    : Acquisition times (min)
        normScan4M(np.ndarray) : Normalized scan array (nRows x nCols x nROISlc x nUptakeTime)
        uptakeTimeV            : Acquisition times for uptake (min) (1 x nUptakeTime)
    """

    smoothFlag = False
    if imgSmoothDict is not None:
        smoothFlag = True
        fSize = imgSmoothDict['kernelSize']
        fSigma = imgSmoothDict['sigma']

    numSlc = scanArr4M.shape[2]
    nTimePts = scanArr4M.shape[3]

    normScan4M = np.zeros(scanArr4M.shape)
    for slc in range(numSlc):
        slcSeq3M = scanArr4M[:, :, slc, :].copy()
        # Smooth to calc. baseline signal only
        if smoothFlag:
            for t in range(slcSeq3M.shape[2]):
                slcSeq3M[:, :, t] = gaussian_filter(slcSeq3M[:, :, t], sigma=fSigma,
                                                    mode='nearest', truncate=fSize / fSigma)
        maskSlcM = mask3M[:, :, slc]
        maskSlc3M = np.repeat(maskSlcM[:, :, np.newaxis], nTimePts, axis=2)
        slcSeq3M[~maskSlc3M] = np.nan
        if slc == 0 and basePts is None:
            # Get user - input shift to start of uptake curve
            midSlc = round(numSlc / 2)
            midSliceSeq3M = scanArr4M[:, :, midSlc, :]
            midSlcMaskM = mask3M[:, :, midSlc]
            basePts = getStartofUptake(midSliceSeq3M, midSlcMaskM)
        maskedSlcSeq3M = np.ma.masked_invalid(slcSeq3M[:, :, 0:basePts])  # Prevents RuntimeWarning: Mean of empty slice
        baselineM = np.mean(maskedSlcSeq3M, axis=2).filled(np.nan)
        baselineM[baselineM == 0] = EPS

        normSig3M = scanArr4M[:, :, slc, :] / baselineM[:, :, np.newaxis]
        if enhThresh is not None:
              sizV =  normSig3M.shape
              normSigM = normSig3M.reshape(-1, normSig3M.shape[2], order='F')
              peakIdxV = locatePeak(normSigM)
              colIdxV = np.full_like(peakIdxV, fill_value=-1, dtype=np.int32).flatten()
              rowIdxV = np.arange(normSigM.shape[0]).flatten()
              enhMask = np.logical_or(colIdxV == -1, normSigM[rowIdxV, colIdxV] < enhThresh)
              normSigM[enhMask, :] = np.nan
              normSig3M = normSigM.reshape(sizV, order='F')

        if method == 'RSE':
            # Return normalized signal (S(t)/S(0))
            normScan4M[:, :, slc, :] = normSig3M
        elif method == 'CC':
            # Return contrast agent concentration
            for t in range(normScan4M.shape[3]):
                normScan4M[:, :, slc, t] = intToConc(normSig3M[:, :, t], concDict)
        elif method is None:
            normScan4M = scanArr4M


    timePtsV = timePtsV - timePtsV[basePts]
    uptakeTimeV = timePtsV[basePts:]
    normScanUptake4M = normScan4M[:, :, :, basePts:]
    
    return normScanUptake4M, uptakeTimeV, basePts


def locatePeak(sigM, smoothFlag=False):
    """locatePeak
    Function to locate peak of uptake curve

    Args:
        sigM (np.ndarray, 2D) : Uptake curves (nVox x nUptakeTime)
        smoothFlag (bool): [optional; default: False] Filter out noise if True.

    Returns:
        peakIdxV (np.array)   : Indices corresponding to peak of uptake (1 x nVox)
    """

    nVox = sigM.shape[0]
    nTime = sigM.shape[1]

    maxWin = 21
    minWin = 5  # Ensure reasonable window size
    calcNoiseLevel = lambda sigV: np.std(sigV - medfilt(sigV, kernel_size=3))
    getWindowSize = lambda sigV: max(min(2 * round(0.05 * calcNoiseLevel(sigV) * len(sigV) / 2) + 1,
                                         maxWin, len(sigV) - 1), minWin)
    sigMax = 0.8 * np.max(sigM, axis=1)

    if smoothFlag:
        filtSigM =  np.apply_along_axis(
                    lambda row: savgol_filter(row, window_length=getWindowSize(row), polyorder=3),
                    axis=1,
                    arr=sigM)
    else:
        filtSigM = sigM

    diffNextM = np.concatenate((np.zeros((nVox, 1)), np.diff(filtSigM, 1, 1)), axis=1)
    diffPrevM = np.concatenate((-np.diff(filtSigM, 1, 1), np.zeros((nVox, 1))), axis=1)
    localMaxIdxM = np.logical_and(diffNextM >= 0, diffPrevM >= 0)
    highSigIdxM = filtSigM > np.tile(sigMax, (nTime, 1)).transpose()

    # Correction for noisy signals
    if smoothFlag:
        hasPeakV = np.any(highSigIdxM, axis=1)
        if np.any(hasPeakV == 0):
            highSigIdxM[~hasPeakV, :] = filtSigM[~hasPeakV, :] > np.tile(0.6 * sigMax[~hasPeakV],
                                                                     (nTime, 1)).transpose()
            hasPeakV = np.any(highSigIdxM, axis=1)
            if np.any(hasPeakV == 0):
                # Correction where max of filt signal < 0
                selFiltSigM = filtSigM[~hasPeakV, :]
                shiftIdxV = np.max(selFiltSigM, axis=1) < 0
                selFiltSigM[shiftIdxV, :] = selFiltSigM[shiftIdxV, :] - np.tile(np.min(selFiltSigM[shiftIdxV,:], axis=1),
                                                              (nTime, 1)).transpose()
                filtSigM[~hasPeakV, :] = selFiltSigM
                highSigIdxM[~hasPeakV, :] = filtSigM[~hasPeakV, :] > np.tile(0.8 * np.max(filtSigM[~hasPeakV, :], axis=1),
                                                                         (nTime, 1)).transpose()

    allPeaksM = np.logical_and(localMaxIdxM, highSigIdxM)
    skipVoxV = ~np.any(allPeaksM, axis=1)
    peakIdxV = np.argmax(allPeaksM, axis=1).astype(float)
    if any(skipVoxV):
        peakIdxV[skipVoxV] = 0

    return peakIdxV


def smoothResample(sigM, timeV, temporalSmoothFlag=False, resampFlag=False):
    """smoothResample
    Function to process uptake curve prior to feature extraction

    Args:
        sigM (np.ndarray, 2D)      : Uptake curves (nVox x nUptakeTime)
        timeV (np.array, 1D)       : Acquisition times (1 x nUptakeTime)
        temporalSmoothFlag (bool)  : [optional, default:False] Smooth curves follg. peak
                                     using cubic splines.
        resampFlag (bool)          : [optional, default:False] Resample uptake curves to 0.1 min
                                     resolution if True.

    Returns:
        resampSigM  (np.ndarray, 2D)  : Processed uptake curves (nVox x nResampUptakeTime)
        timeOutV    (np.array, 1D)    : Resampled time pts (min) (1 x nResampUptakeTime)

    """

    # Smoothing settings
    # smoothing = 0.05  # Adjust this for more or less smoothing
    sigma = 1
    radius = 2

    # Resampling settings
    nPad = 100
    ts = 0.1
    tdiff = timeV[1] - timeV[0]

    # Pad signal
    padSigM = np.hstack((np.tile(sigM[:, 0], (nPad, 1)).transpose(), sigM,
                         np.tile(sigM[:, -1], (nPad, 1)).transpose()))
    padTimeV = np.hstack((np.linspace(timeV[0] - nPad * tdiff, timeV[0] - tdiff, num=nPad, endpoint=True), timeV,
                          np.linspace(timeV[-1] + tdiff, timeV[-1] + nPad * tdiff, num=nPad, endpoint=True)))

    maxWin = 21
    minWin = 5
    calcNoiseLevel = lambda sigV: np.std(sigV - medfilt(sigV, kernel_size=3))
    getWindowSize = lambda sigV: max(min(2 * round(0.05 * calcNoiseLevel(sigV) * len(sigV) / 2) + 1,
                                         maxWin, len(sigV) - 1), minWin)

    if not (resampFlag or temporalSmoothFlag):
        return sigM, timeV
    else:
        if temporalSmoothFlag:
            # Locate first peak
            peakIdxV = locatePeak(sigM, smoothFlag=True)
            # Smooth signal following first peak
            keepIdxV = np.nansum(padSigM, axis=1) != 0
            selPadSigM = padSigM[keepIdxV, :]
            peakIdxV = peakIdxV[keepIdxV]
            for vox in range(selPadSigM.shape[0]):
                smoothIdxV = np.arange(int(nPad + peakIdxV[vox] + 1), padSigM.shape[1])
                padSigM[vox, smoothIdxV] = savgol_filter(selPadSigM[vox, smoothIdxV],
                                                         window_length=getWindowSize(selPadSigM[vox, smoothIdxV]),
                                                         polyorder=3)

        if resampFlag:
            skipIdxV = np.nansum(padSigM, axis=1) == 0
            zeroIdxV = np.sum(padSigM, axis=1) == 0
            skipIdxV = np.logical_and(skipIdxV, ~zeroIdxV)
            padSubSigM = padSigM[~skipIdxV, :]
            numPts = int(padSubSigM.shape[1] * tdiff / ts)
            resampPadSigM = np.full((sigM.shape[0], numPts), np.nan)
            resampPadSigM[~skipIdxV, :], timePadV = resample(padSubSigM, numPts, t=padTimeV, axis=1)
        else:
            resampPadSigM = padSigM
            ts = tdiff
            timePadV = timeV

        # Un-pad
        tSkip = round(nPad * tdiff / ts)
        resampSigM = resampPadSigM[:, tSkip:-tSkip]
        timeOutV = timePadV[tSkip:-tSkip]

        return resampSigM, timeOutV


def semiQuantFeatures(procSlcSigM, procTimeV):
    """semiQuantFeatures
        Compute non-parametric features from pre-processed contrast uptake curve.
        Ref.: Lee, S.H., et al. (2017) "Correlation Between Tumor Metabolism and Semiquantitative Perfusion
               MRI Metrics in Non–small Cell Lung Cancer." IJROBP 99.2:S83-S84.

        Args:
            procSlcSigM (np.ndarray, 2D)   : Processed uptake curves (nVox x nResampleTime)
            procTimeV (np.array, 1D)       : Acquisition times (1 x nResampleTime) in min.

        Returns:
            featureDict (dict)             : Dictionary of non-parameteric features.

    """

    nVox = procSlcSigM.shape[0]

    # Peak value (enhancement if 'RSE' or  concentration if 'CC')
    # PEv = np.max(procSlcSigM, axis=1)
    # peakIdxV = np.argmax(procSlcSigM, axis=1)
    zeroIdxV = np.sum(procSlcSigM, axis=1) == 0
    skipIdxV = np.logical_and(np.nansum(procSlcSigM, axis=1) == 0, ~zeroIdxV)
    peakIdxV = np.zeros(nVox, dtype=int)
    peakIdxV[~skipIdxV] = (locatePeak(procSlcSigM[~skipIdxV,:], smoothFlag=True)).astype(int)
    PEv = procSlcSigM[np.arange(nVox), peakIdxV]
    TTPv = procTimeV[peakIdxV]  # Time-to-peak
    TTPv[skipIdxV] = np.nan

    # Half-peak
    halfMaxSig = .5 * PEv
    SHPcolIdx = np.zeros(nVox, dtype=int)
    for vox in range(nVox):
        SHPcolIdx[vox] = np.argmin(np.abs(procSlcSigM[vox, :peakIdxV[vox] + 1] - halfMaxSig[vox, np.newaxis]))
    SHPv = procSlcSigM[np.arange(nVox), SHPcolIdx]  # Value (relative enhancement or concentration) at half-peak
    TTHPv = procTimeV[SHPcolIdx]  # Time to half-peak

    # Wash-in slope
    WISv = PEv / (TTPv + EPS)  # Wash in slope, WIS = PE / TTP

    # Wash-out slope
    # WOS = (PE - RSE(Tend)) / (Tend - TTP), if PE does not occur at Tend (nan otherwise).
    Tend = procTimeV[-1]
    RSEendV = procSlcSigM[:, -1]
    peakAtEndIdx = TTPv == Tend
    with np.errstate(invalid='ignore'):
        WOSv = (PEv - RSEendV) / (TTPv + EPS - Tend)
        WOSv[peakAtEndIdx] = np.nan  # Not defined

    # Wash-in/out gradients
    # Initial gradient estimated by linear regression of RSE between 10 % and 70 % PE (occurring prior to peak)
    IGv = np.full((nVox,), fill_value=np.nan)
    for i in range(nVox):
        id_10 = np.argmin(np.abs(procSlcSigM[i, :peakIdxV[i] + 1] - .1 * PEv[i]))
        id_70 = np.argmin(np.abs(procSlcSigM[i, id_10:peakIdxV[i] + 1] - .7 * PEv[i]))
        if id_70 == 0:
            id_70 = peakIdxV[i]  # Handle case where no column exceeds 70%
        initialPts = np.arange(id_10, id_70 + 1)
        y = procSlcSigM[i, initialPts].T
        x = np.hstack((np.ones((len(initialPts), 1)), procTimeV[initialPts].T[:, np.newaxis]))
        # x = np.column_stack((np.ones(len(initialPts)), procTimeV[initialPts].T))  # Create the design matrix
        b, __, __, __ = np.linalg.lstsq(x, y, rcond=None)
        IGv[i] = b[1]

    # Wash-out gradient estimated by linear regression of RSE between  1 and 2 min elapsed from start of uptake
    WOGv = np.full((nVox,), fill_value=np.nan)
    for i in range(nVox):

        id_1 = np.argmax(procTimeV >= 1)
        id_2 = np.argmax(procTimeV > 2)
        if id_1 == 0 or id_2 == 0:
            WOGv[i] = np.nan
        else:
            washOutPts = np.arange(id_1, id_2)
            y = procSlcSigM[i, washOutPts].T
            x = np.hstack((np.ones((len(washOutPts), 1)), procTimeV[washOutPts].T[:, np.newaxis]))
            b, __, __, __ = np.linalg.lstsq(x, y, rcond=None)
            WOGv[i] = b[1]

    # Signal enhancement ratio
    # RSE at 0.5 min divided by RSE at 2.5 min, elapsed from start of uptake
    tse1 = np.nanargmax(procTimeV >= .5)
    tse2 = np.nanargmax(procTimeV >= 2.5)
    SERv = procSlcSigM[:, tse1] / (procSlcSigM[:, tse2] + EPS)

    # IAUC
    IAUCv = cumtrapz(y=procSlcSigM.T, x=procTimeV.T, axis=0, initial=0).T
    IAUCtthpV = np.full((nVox,), fill_value=np.nan)
    IAUCttpV = np.full((nVox,), fill_value=np.nan)
    for i in range(nVox):
        IAUCtthpV[i] = IAUCv[i, np.nanargmax(procTimeV >= TTHPv[i])]
        IAUCttpV[i] = IAUCv[i, np.nanargmax(procTimeV >= TTPv[i])]

    featureDict = {'PeakEnhancement': PEv,
                   'SignalAtHalfPeak': SHPv,
                   'TimeToPeak': TTPv,
                   'TimeToHalfPeak': TTHPv,
                   'SignalEnhancementRatio': SERv,
                   'WashInSlope': WISv,
                   'WashOutSlope': WOSv,
                   'InitialGradient': IGv,
                   'WashOutGradient': WOGv,
                   'AUCatPeak': IAUCttpV,
                   'AUCatHalfPeak': IAUCtthpV}

    return featureDict


def calcROIuptakeFeatures(planC, structNum, timeV=None, basePts=None, imgSmoothDict=None, enhThresh=None,
                          sigType='RSE', concDict={}, temporalSmoothFlag=False, resampFlag=False, plotDict={}):
    """calcROIuptakeFeatures
        Wrapper to compute non-parametric uptake characteristics for each slice of input ROI.

        Args:
            planC (plan_container.planC): pyCERR's plan container object
            structNum (int): Index of structure in planC
            timeV (np.array, float): [optional, default:None] User-input acquisition times
            basePts (int): [optional, default:None] Time pt. representing start of uptake.
                           By default, have user input value.
            imgSmoothDict (dict): [optional, default:None] Dictionary specifying whether to
                                  smooth image & associated filter parameters.
                                  Keys: 'kernelSize', 'sigma'.
            enhThresh (float): [optional, default:None] Intensity threshold to identify enhancing voxels.
                           Voxels with peak intensity < thresh*baseline are excluded from analysis.
            sigType (string): [optional, default:'RSE'] Convert intensities to relative signal
                         enhancement ('RSE') or contrast concentration ('CC')
            concDict (dict): [optional, default:{}] Required if method='CC'. Dictionary of parameters
                         required to compute contrast agent concentration. Must specify threshold,
                         T10, flipAngle, TR, r1.
            temporalSmoothFlag (bool) : [optional, default:False] Flag specifying whether to
                                     smooth curves follg. peak using cubic splines.
            resampFlag (bool): [optional, default:False] Resample uptake curves to 0.1 min
                                     resolution if True.
            plotDict (dict): [optional, default:{}] Display sample plots showing computed features (interactive)


        Returns:
            featureList: List of dictionaries (one per ROI slice) containing uptake features.

    """
    userInputTime = []
    if len(timeV) > 0:
        userInputTime = timeV

    # Load DCE series
    scanArr4M, timePtsV, mask3M, maskSlcV = loadTimeSeq(planC, structNum, userInputTime)

    # Transform signal intensity to
    # relative signal enhancement (signal over baseline intensity) if sigType is 'RSE'  or
    # contrast agent concentration if sigType is 'CC'
    normScan4M, selTimePtsV, basePts = normalizeToBaseline(scanArr4M, mask3M, timePtsV,
                                                           basePts=basePts, imgSmoothDict=imgSmoothDict,
                                                           enhThresh=enhThresh, method=sigType, concDict=concDict)

    # Loop over ROI slices
    featureList = []
    for slc in range(len(maskSlcV)):
        # Reshape to 2D array (nVox x nTimePts)
        normSlc3M = normScan4M[:, :, slc, :]
        normSlcSigM = normSlc3M.reshape(-1, normSlc3M.shape[2], order='F')  # column major

        # Pre-process
        ## Retain voxels in ROI
        zeroIdxV = np.sum(normSlcSigM, axis=1) == 0
        skipIdxV = np.logical_and(np.nansum(normSlcSigM, axis=1)==0, ~zeroIdxV)
        #skipIdxV = np.isnan(np.sum(normSlcSigM, axis=1))
        if np.all(skipIdxV): #No enhancing voxels
           continue
        else:
            normROISlcSigM = normSlcSigM[~skipIdxV, :]
            ## Smoothing + resampling
            procSlcSigM, procTimeV = smoothResample(normROISlcSigM, selTimePtsV,
                                                    temporalSmoothFlag=temporalSmoothFlag,
                                                    resampFlag=resampFlag)
            if sigType == 'RSE':
                # Calc. signal enhancement relative to baseline (assumed to be proportional to contrast agent concentration)
                convSlcSigM = procSlcSigM.copy() - 1  # S(t)/S(0) - 1
            else:
                convSlcSigM = procSlcSigM.copy()

            # Compute features
            featureDict = semiQuantFeatures(convSlcSigM, procTimeV)

            if 'display' in plotDict and plotDict['display']:
                plotSampleFeatures(procSlcSigM, procTimeV, featureDict, numPlots=1,
                                   savePath=plotDict['savepath'], prefix=plotDict['prefix'] + '_slc' + str(slc))

            featureList.append(featureDict)

    return featureList, basePts


def plotSampleFeatures(procSlcSigM, procTimeV, featureDict, numPlots=1, savePath=None, prefix=''):
    """plotSampleFeatures
    Function to plot sample uptake curves and indicate extracted features.

    Args:
        procSlcSigM  (np.ndarray, 2D)  : Processed uptake curves (nVox x nResampUptakeTime)
        sigType (string): [optional, default:'RSE'] Convert intensities to relative signal
                         enhancement ('RSE') or contrast concentration (CC)
        featureDict (dict): Dictionary of non-parameteric features
        numPlots (int): [optional, default = 1] No. sample plots to display per ROI slice.
    """
    voxIdxV = rng.integers(low=0, high=procSlcSigM.shape[0], size=numPlots)
    for idx in voxIdxV:
        plt.figure()
        plt.axis([0, procTimeV[-1], np.min(procSlcSigM[idx, :]) - 0.01, np.max(procSlcSigM[idx, :]) + 0.01])
        plt.plot(procTimeV, procSlcSigM[idx, :], color='black', linewidth=2)
        # plt.annotate('Peak', xy=(featureDict['TimeToPeak'][idx], featureDict['PeakEnhancement'][idx]))

        # TTP
        ttp = featureDict['TimeToPeak'][idx]
        match = np.argmin(np.abs(procTimeV - ttp))
        plt.annotate('TTP', xy=(ttp, min(procSlcSigM[idx, :])))
        plt.vlines(x=ttp, ymin=min(procSlcSigM[idx, :]), ymax=procSlcSigM[idx, match],
                   color='purple', linestyles='dashed', linewidth=1.5)

        # TTHP
        tthp = featureDict['TimeToHalfPeak'][idx]
        match = np.argmin(np.abs(procTimeV - tthp))
        plt.annotate('TTHP', xy=(tthp, min(procSlcSigM[idx, :])))
        plt.vlines(x=tthp, ymin=min(procSlcSigM[idx, :]), ymax=procSlcSigM[idx, match],
                   color='purple', linestyles='dashed', linewidth=1.5)

        # Wash-in slope
        ctr = np.argmin(np.abs(procTimeV - featureDict['TimeToPeak'][idx]))
        point_x1 = 0  # procTimeV[0]
        point_y1 = 0  # relSigM[idx, 0]
        point_x2 = procTimeV[ctr]
        point_y2 = procSlcSigM[idx, ctr]
        cptIdx = int(ctr/4)
        cpt = (procTimeV[cptIdx], procSlcSigM[idx,cptIdx])
        slope = featureDict['WashInSlope'][idx]
        angle = np.rad2deg(np.arctan2(point_y2 - point_y1, point_x2 - point_x1))
        # line_length = 0.5  # Adjust length of the line
        # dx = line_length / np.sqrt(1 + slope ** 2)
        # dy = slope * dx
        # xSlopeLine = [point_x - dx / 2, point_x + dx / 2]
        # ySlopeLine = [point_y - dy / 2, point_y + dy / 2]
        plt.plot([point_x1, point_x2], [point_y1, point_y2], '--', color='purple',
                 label='Wash-in slope', linewidth=1.5)
        plt.text(cpt[0], cpt[1], f'Wash-in slope: {slope:.2f}', ha='left', va='bottom',
                 transform_rotates_text=True, rotation=angle,
                 rotation_mode='anchor')


        # Wash-out slope
        point_x1 = procTimeV[ctr]
        point_y1 = procSlcSigM[idx, ctr]
        point_x2 = procTimeV[-1]
        point_y2 = procSlcSigM[idx, -1]
        midptIdx = int(ctr + (len(procTimeV) - ctr) / 2)
        midpt = (procTimeV[midptIdx], procSlcSigM[idx, midptIdx])
        slope = featureDict['WashOutSlope'][idx]
        angle = np.rad2deg(np.arctan2(point_y2 - point_y1, point_x2 - point_x1))
        plt.plot([point_x1, point_x2], [point_y1, point_y2], '--', color='purple',
                 label='Wash-out slope', linewidth=1.5)
        # plt.annotate(f'Wash-out slope: {slope:.2f}', xy=midpt, rotation=degrees(atan(slope)),
        #               fontsize=10, ha="center", color="black")
        plt.text(midpt[0], midpt[1], f'Wash-out slope: {slope:.2f}', ha='left', va='bottom',
                 transform_rotates_text=True, rotation=angle,
                 rotation_mode='anchor')

        # Initial gradient
        id_10 = np.argmin(np.abs(procSlcSigM[idx, :ctr + 1] - .1 * featureDict['PeakEnhancement'][idx]))
        id_70 = np.argmin(np.abs(procSlcSigM[idx, id_10:ctr + 1] - .7 * featureDict['PeakEnhancement'][idx]))
        x_mid = (procTimeV[id_10] + procTimeV[id_70]) / 2
        y_mid = (procSlcSigM[idx, id_10] + procSlcSigM[idx, id_70]) / 2
        slope = featureDict['InitialGradient'][idx]
        length = 1  # Length of the dotted line
        dx = length / 2 * np.sqrt(1 / (1 + slope ** 2))  # x-component of line length
        dy = slope * dx  # y-component of line length
        angle = np.rad2deg(np.arctan2(dy, dx))
        x_start, x_end = x_mid - dx, x_mid + dx
        y_start, y_end = y_mid - dy, y_mid + dy
        # plt.plot([x_start, x_end], [y_start, y_end], '--', color='purple', label="Slope Line", linewidth=1.5)
        # plt.text(x_mid, y_mid, f"Initial gradient: {slope:.2f}", rotation=angle,
        #           ha='center', va='center')

        # Wash-out gradient
        id_1 = np.argmax(procTimeV >= 1)
        id_2 = np.argmax(procTimeV > 2)
        x_mid = (procTimeV[id_1] + procTimeV[id_2]) / 2
        y_mid = (procSlcSigM[idx, id_1] + procSlcSigM[idx, id_2]) / 2
        slope = featureDict['WashOutGradient'][idx]
        length = 1  # Length of the dotted line
        dx = length / 2 * np.sqrt(1 / (1 + slope ** 2))  # x-component of line length
        dy = slope * dx  # y-component of line length
        angle = np.rad2deg(np.arctan2(dy, dx))
        x_start, x_end = x_mid - dx, x_mid + dx
        y_start, y_end = y_mid - dy, y_mid + dy
        # plt.plot([x_start, x_end], [y_start, y_end], '--', color='purple', label="Slope Line", linewidth=1.5)
        # plt.text(x_mid, y_mid, f"Wash-out gradient: {slope:.2f}", rotation=angle,
        #           ha='center', va='center')

        # AUC
        xFill = procTimeV[procTimeV <= tthp]
        yFill = procSlcSigM[idx, procTimeV <= tthp]
        plt.fill_between(xFill, 0, yFill, facecolor="none", color='coral', alpha=0.3,
                         hatch='//', label="AUC_{TTHP}")

        xFill = procTimeV[procTimeV <= ttp]
        yFill = procSlcSigM[idx, procTimeV <= ttp]
        plt.fill_between(xFill, 0, yFill, facecolor="none", color='skyblue', alpha=0.3,
                         hatch=r'\\', label="AUC_{TTP}}")

        if savePath is not None:
            figPath = os.path.join(savePath, prefix + '_vox' + str(idx) + '.jpg')
            plt.savefig(figPath)
            plt.close()
        else:
            plt.show(block=True)

    return 0


def createFeatureMaps(featureList, strNum, planC, importFlag=False, type='scan'):
    """createFeatureMaps
        Function to generate maps of non-parametric features.

        Args:
            featureList: List of dictionaries (one per ROI slice) containing uptake features.
            structNum (int): Index of structure in planC.
            planC (plan_container.planC): pyCERR's plan container object
            importFlag (bool): [optional, default:False] Import to planC as pseudo-dose.
            type (str): [optional, default:'scan'] Import to planC as pseudo-scan ('scan') or pseudo-dose ('dose').

        Returns:
            mapDict (dict) : Dictionary of features maps.
            planC (plan_container.planC): pyCERR's plan container object
    """

    # Get mask, associated scan and grid
    mask3M = getStrMask(strNum, planC)
    validSlcV = np.sum(np.sum(mask3M, axis=0), axis=0) > 0
    mask3M = mask3M[:, :, validSlcV]

    if importFlag:
        assocScan = planC.structure[strNum].getStructureAssociatedScan(planC)
        xV, yV, zV = planC.scan[assocScan].getScanXYZVals()
        zV = zV[validSlcV]

    # Extract list of available features
    feats = featureList[0].keys()
    numRow, numCol, numSlc = mask3M.shape

    mapDict = {f"{key}": np.zeros_like(mask3M, dtype=float) for key in feats}

    # Create 3D maps
    for key in feats:
        for s in range(numSlc):
            maskSlcM = mask3M[:, :, s]

            # Get voxel indices in column-first order
            rowIdxV, colIdxV = np.where(maskSlcM)
            colFirstIdxV = np.lexsort((rowIdxV, colIdxV))
            sortedRowIdxV = rowIdxV[colFirstIdxV]
            sortedColIdxV = colIdxV[colFirstIdxV]

            # Assign feature vals.
            featValV = featureList[s][key]
            sliceMap = mapDict[key][:, :, s]
            sliceMap[sortedRowIdxV, sortedColIdxV] = featValV
            mapDict[key][:, :, s] = sliceMap

        # Import as pseudo-dose array
        if importFlag:
            if type.lower() == 'scan':
                planC = pc.importScanArray(mapDict[key], xV, yV, zV, key, assocScan, planC)
            if type.lower() == 'dose':
                planC = pc.importDoseArray(mapDict[key], xV, yV, zV, planC, assocScan,
                                           doseInfo={'fractionGroupID': key})

    return mapDict, planC