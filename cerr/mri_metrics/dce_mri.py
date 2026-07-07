import os, random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.integrate import cumulative_trapezoid

from cerr import plan_container as pc
from cerr.dataclasses import structure as cerrStr
from cerr.contour.rasterseg import getStrMask
from cerr.utils.statistics import round, prctile
from cerr.dataclasses.scan import dcm_hhmmss

EPS = np.finfo(float).eps
rng = np.random.default_rng()


def getAcqTime(planC, timeKey, scanIdxV=None):
    """Extract timing information from DCE-MRI scans in planC.

    Args:
        planC (plan_container.planC): pyCERR's plan container object containing DCE scans and metadata.
        timeKey (str): [optional, default:None] scanInfo field-name from which to read timing.
                    Note: Assumes timing in min unless `acquisitionTime` or `triggerTime`.
                    If `None`, first uses``temporalPositionIndex`` (0020,0100), if available,
                    to establish canonical scan order.
                    Then examines ``acquisitionTime`` (0008,0032) (assumed to be in seconds).
                    Values must be unique across scans and ordering must match ``temporalPositionIndex``.
                    If not,``triggerTime`` (0018,1060) (assumed ot be in ms) is considered.
                    Returns error if both fail.
        scanIdxV (list): [optional, default:None] Indices into ``planC.scan`` identifying the DCE volumes to use,
                        in any order. Defaults to all scans.

    Returns:
        timeV (np.ndarray) : Timing of each scan (min))
    """

    if scanIdxV is None:
        scanIdxV = list(range(len(planC.scan)))
    scanIdxV = np.asarray(scanIdxV, dtype=int)
    nScans = len(scanIdxV)

    def _getTiming(planC, fieldName):
        """
        Extract timing from planC.scan.scanInfo[0].[fieldName] across scans.
        """
        keyTimesV = np.full(nScans, np.nan)
        for k, i in enumerate(scanIdxV):
            t = getattr(planC.scan[i].scanInfo[0], fieldName, None)
            if t not in ('', None):
                try:
                    keyTimesV[k] = t
                except Exception:
                    pass  # nan
        return keyTimesV
    sortOrder = np.arange(nScans)

    if timeKey is None or timeKey in ['acquisitionTime','triggerTime']:
        # Use acquisitionTime / triggerTime
        temporalPosV = _getTiming(planC, 'temporalPositionIndex')
        temporalPosV = np.array(temporalPosV, dtype=float)
        if not np.any(np.isnan(temporalPosV)):
            sortOrder = np.argsort(temporalPosV)
        else:
            print("WARNING: temporalPositionIndex not found in scanInfo. "
                  "Assuming scans are in temporal order.")

        # Extract acquisitionTime unless triggerTime is specified
        skipAcq = True if timeKey == 'triggerTime' else False
        acqTimesStrV = _getTiming(planC, 'acquisitionTime')
        acqTimesV = []
        for at in acqTimesStrV:
            if at in ('', None) or (isinstance(at, float) and np.isnan(at)):
                acqTimesV.append(np.nan)
            else:
                try:
                    acqTimesV.append(dcm_hhmmss(str(at))[0])
                except Exception:
                    acqTimesV.append(np.nan)
        acqTimesV = np.array(acqTimesV, dtype=float)

        # Check validity
        if not skipAcq:
            sortedAcqTimesV = acqTimesV[sortOrder]
            if len(np.unique(acqTimesV)) < nScans:
                skipAcq = True
                print(f"Acquisition time is not unique across scans. Skipping...")
            if not np.all(np.diff(sortedAcqTimesV) > 0):
                skipAcq = True
                print(f"Acquisition time ordering does not agree with temporalPositionIndex. Skipping...")
        if timeKey=='acquisitionTime':
                skipAcq = False

        # Extract triggerTime unless acquisitionTime is specified
        if skipAcq:
            skipTriggerTime = True if timeKey == 'acquisitionTime' else False

            if not skipTriggerTime:
                trigTimesV = _getTiming(planC, 'triggerTime')
                trigTimesV = np.array(trigTimesV, dtype=float)
                sortedTriggerTimesV = trigTimesV[sortOrder]
                if len(np.unique(trigTimesV)) < nScans:
                    skipTriggerTime = True
                    print(f"Trigger time is not unique across scans.")
                if not np.all(np.diff(sortedTriggerTimesV) > 0):
                    skipTriggerTime = True
                    print(f"Trigger time ordering does not agree with temporalPositionIndex.")
            if timeKey == 'triggerTime':
                skipTriggerTime = False

            # Check validity
            if skipTriggerTime:
                    raise ValueError(f"ERROR: Could not extract timing. Please supply timeV manually.")
            else:
                print("Timing extracted from triggerTime (0018,1060).")
                timeV = trigTimesV / 1000.0 / 60.0  # ms to min
                return timeV
        else:
            print("Timing extracted from acquisitionTime (0020,010).")
            timeV = acqTimesV / 60.0  # min
    else:
        # Extract from `timeKey` field
        timeV = _getTiming(planC, timeKey)
        timeV = np.array(timeV, dtype=float)

    return timeV


def loadTimeSeq(planC, structNum, userInputTime=None, scanNumV=None):
    """loadTimeSeq
    Function to extract 4D DCE scan array associated with input structure from planC

    Args:
        planC (plan_container.planC): pyCERR's plan container object
        structNum (int): Index of structure in planC
        userInputTime (np.array, float): [optional. default=None]
                                         or user-input acquisition times (min) as array.
                                         Must correspond to scan ordering.
        scanNumV (list): Indices into ``planC.scan`` identifying the DCE volumes to use,
                                         in any order. Defaults to all scans.

    Returns:
        scanArr4M (np.ndarray, 4D)  : DCE array (nRows x nCols x nROISlc x nTime)
        timePtsV (np.array, 1D)     : Acquisition times (min)
        maskSlc3M (np.ndarray, 3D)  : Mask of ROI (nRows x nCols x nROISlc)
        maskSlcV (np.array, 1D)     : Indices of slices in the ROI (1 x nROISlc)
    """

    mask3M = getStrMask(structNum, planC)
    maskSlcIdxV = np.sum(np.sum(mask3M, axis=0), axis=0) > 0
    maskSlcV = np.array(np.where(maskSlcIdxV)[0])
    maskSlc3M = mask3M[:, :, maskSlcIdxV]
    numSlc = len(maskSlcV)

    if scanNumV is not None:
        numTimePts = len(scanNumV)
    else:
        numTimePts = len(planC.scan)
        scanNumV = np.arange(0, numTimePts)

    # Extract uptake curves for voxels in ROI
    scanSizeV = planC.scan[scanNumV[0]].getScanSize()
    scanArr4M = np.zeros((scanSizeV[0], scanSizeV[1], numSlc, numTimePts))
    for slc in range(numSlc):
        scanSlc3M = np.array([planC.scan[scn].getScanArray()[:, :, maskSlcV[slc]] for scn in scanNumV])
        scanArr4M[:, :, slc, :] = np.moveaxis(scanSlc3M, 0, -1)

    timePtsV = userInputTime
    if userInputTime is None:
        timePtsV = np.array([planC.scan[scn].scanInfo[0].acquisitionTime for scn in scanNumV])

    # Sort time pts
    indSortedV = np.argsort(timePtsV)
    timePtsV = timePtsV[indSortedV]
    scanArr4M = scanArr4M[:, :, :, indSortedV]

    return scanArr4M, timePtsV, maskSlc3M, maskSlcV


def intToConc(normSigM, concDict):
    """
    Converts DCE-MRI signal intensity into contrast agent concentration.

    Args:
        normSigM (np.ndarray): 2-D array of intensity-normalized signal (S(t)/S(0)) (nVox, nTimePts)
        concDict (dict): Dictionary specifying the following keys:

            - clip_between (float array): Clip normalized intensities
              (intensity/baseline) between specified min, max values.
            - T10 (float): Pre-contrast longitudinal relaxation time (seconds).
            - FA (float): Flip angle (degrees).
            - TR (float): Repetition time (seconds).
            - r1 (float): Relaxivity.

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
    skipIdxM = np.isnan(normSigM)
    zeroIdxM = normSigM == 0
    validNormSigM = normSigM[~skipIdxM]
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
    C[~skipIdxM] = 1 / r1 * (R1 - R10)
    C[np.iscomplex(C)] = 0
    C[zeroIdxM] = 0
    C[C < 0] = 0

    return C


def plotUptake(timePtsV, sigV, blockFlag, savePath=None):
    """Plot a sample DCE uptake curve for interactive selection of baseline points.

    Displays a labeled time-series plot of ROI mean signal intensity with each
    time-point annotated by its index so the user can identify the start of uptake.

    Args:
        timePtsV (np.ndarray): 1-D array of acquisition time points.
        sigV (np.ndarray): 1-D array of ROI mean signal intensities corresponding
            to each time point in ``timePtsV``.
        blockFlag (bool): If ``True``, the plot blocks execution until it is
            closed; if ``False``, execution continues immediately.
        savePath (str, optional): File path at which to save the figure.
            When provided the figure is saved and closed rather than displayed
            interactively. Defaults to ``None``.

    Returns:
        int: Always returns 0.
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


def getStartofUptakeFromMeanSignal(meanSigV, plotDict=None):
    """getStartofUptakeFromMeanSignal
    Function for interactive selection of baseline points from a 1D mean signal curve.

    Args:
        meanSigV (np.ndarray, 1D) : 1D array containing the time sequence of the 3D ROI average signal.
        plotDict (dict): [optional, default:None] Dictionary for writing the annotated uptake plot (PNG)
                         with the selected start point highlighted.
                         'uptake_savepath': Path to output directory.
                         'prefix': Append to file name.
    Returns:
        basePts (int) : Time point representing start of uptake
    """
    timePtsV = np.arange(0, len(meanSigV))

    # Interactive selection of baseline pts
    plt.figure()
    plt.plot(timePtsV, meanSigV, marker='o', color='b')

    # Annotate indices so the user knows exactly what number to type
    for i, (x, y) in enumerate(zip(timePtsV, meanSigV)):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Time point')
    plt.ylabel('ROI mean signal intensity')
    plt.title('Select start of uptake (3D ROI Volume Average)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show(block=True)

    basePts = input("Enter timepoint index representing start of uptake: ")

    try:
        basePts = int(basePts)
        # Basic bounds checking
        if basePts < 0 or basePts >= len(meanSigV):
            print(f"Error: Index must be between 0 and {len(meanSigV) - 1}.")
            return None
    except ValueError:
        print("Invalid input. Please enter an integer value.")
        return None

    fig, ax = plt.subplots()
    ax.plot(timePtsV, meanSigV, marker='o', color='b')
    for i, (x, y) in enumerate(zip(timePtsV, meanSigV)):
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    # Overlay selected point in red
    ax.plot(timePtsV[basePts], meanSigV[basePts], marker='*',
            color='r', markersize=12, zorder=5, label=f'Start of uptake (index={basePts})')
    ax.annotate(str(basePts), (timePtsV[basePts], meanSigV[basePts]),
                xytext=(5, 5), textcoords='offset points', color='r', fontweight='bold')
    ax.set_xlabel('Time point')
    ax.set_ylabel('ROI mean signal intensity')
    ax.set_title('Select start of uptake (3D ROI Volume Average)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    if plotDict is not None and 'uptake_savepath' in plotDict:
        fileName = 'start_of_uptake.png'
        if 'prefix' in plotDict:
            fileName = plotDict['prefix'] + '_' + fileName
        figPath = os.path.join(plotDict['uptake_savepath'], fileName)
        fig.savefig(figPath, bbox_inches='tight')

    #plt.show(block=True)
    plt.close(fig)

    return basePts


def getStartofUptake(slice3M, maskM, plotDict=None):
    """getStartofUptake
    Function for interactive selection of baseline points

    Args:
        slice3M (np.ndarray, 3D)  : 3D array containing time sequence of scan slice (nRows x nCols x nTime)
        maskM (np.ndarray, 2D)    : Mask of ROI slice
        plotDict (dict): [optional, default:None] Dictionary for writing the annotated uptake plot (PNG)
                         with the selected start point highlighted.
                         'uptake_savepath': Path to output directory.
                         'prefix': Append to file name.
    Returns:
        basePts (int) : Time point representing start of uptake
    """

    # Compute mean ROI intensity at each time point
    meanSigV = np.mean(slice3M[maskM, :], axis=0)
    basePts = getStartofUptakeFromMeanSignal(meanSigV, plotDict)
    return basePts


def normalizeToBaseline(scanArr4M, mask3M, timePtsV, basePts=None, imgSmoothDict=None, enhThresh=None,
                        method='RSE', concDict=None, plotDict=None):
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
                         required to compute contrast agent concentration.
                         Required keys: 'T10','r1','TR','FA'. Optional keys:'clip_between' (float) [c1, c2].
        plotDict (dict): [optional, default:None] Dictionary for writing the annotated uptake plot (PNG)
                         with the selected start point highlighted.
                         'uptake_savepath': Path to output directory.
                         'prefix': Append to file name.
    Returns:
        normScan4M (np.ndarray)   : Normalized scan array (nRows x nCols x nROISlc x nUptakeTime)
        uptakeTimeV (np.array, 1D): Acquisition times for uptake (min) (1 x nUptakeTime)
        baseline3M (np.ndarray)   : Mean baseline signal intensity (post smoothing if specified) (nRows x nCols x nROISlc)
        basePts (int)             : No. of time points to the start of uptake.
    """

    smoothFlag = False
    if imgSmoothDict is not None:
        smoothFlag = True
        fSize = imgSmoothDict['kernelSize']
        fSigma = imgSmoothDict['sigma']

    numSlc = scanArr4M.shape[2]
    nTimePts = scanArr4M.shape[3]

    baseline3M = np.full((scanArr4M.shape[0], scanArr4M.shape[1], numSlc), np.nan)
    if method is None:
        # Input (scanArr4M) concentration map
        normScan4M = np.zeros(scanArr4M.shape)
        for slc in range(numSlc):
            slcSeq3M = scanArr4M[:, :, slc, :].copy()
            maskSlcM = mask3M[:, :, slc]
            maskSlc3M = np.repeat(maskSlcM[:, :, np.newaxis], nTimePts, axis=2)
            slcSeq3M[~maskSlc3M] = np.nan
            normScan4M[:, :, slc, :] = slcSeq3M
        basePts = 0
    else:
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
                midSlc = round(numSlc / 2) if numSlc > 1 else 0
                midSliceSeq3M = scanArr4M[:, :, midSlc, :]
                midSlcMaskM = mask3M[:, :, midSlc]
                basePts = getStartofUptake(midSliceSeq3M, midSlcMaskM, plotDict)
            maskedSlcSeq3M = np.ma.masked_invalid(
                slcSeq3M[:, :, 0:basePts])  # Prevents RuntimeWarning: Mean of empty slice
            baselineM = np.mean(maskedSlcSeq3M, axis=2).filled(np.nan)
            baselineM[baselineM == 0] = EPS

            baseline3M[:, :, slc] = baselineM

            normSig3M = scanArr4M[:, :, slc, :] / baselineM[:, :, np.newaxis]
            if enhThresh is not None:
                sizV = normSig3M.shape
                normSigM = normSig3M.reshape(-1, normSig3M.shape[2], order='F')
                peakIdxV = locatePeak(normSigM)
                colIdxV = np.full_like(peakIdxV, fill_value=-1, dtype=np.int32).flatten()
                rowIdxV = np.arange(normSigM.shape[0]).flatten()
                enhMask = normSigM[rowIdxV, colIdxV] < enhThresh
                normSigM[enhMask, :] = np.nan
                normSig3M = normSigM.reshape(sizV, order='F')

            if method == 'RSE':
                # Return normalized signal (S(t)/S(0))
                normScan4M[:, :, slc, :] = normSig3M
            elif method == 'CC':
                # Return contrast concentration
                nRows, nCols, nTimePts = normSig3M.shape
                flatSlcM = normSig3M.reshape(nRows * nCols, nTimePts)
                flatConcM = np.zeros_like(flatSlcM)
                for t in range(nTimePts):
                    flatConcM[:, t] = intToConc(flatSlcM[:, t], concDict)  # (nRows*nCols,) per time pt
                normScan4M[:, :, slc, :] = flatConcM.reshape(nRows, nCols, nTimePts)
            else:
                raise ValueError("Method {} not supported.".format(method))

    timePtsV = timePtsV - timePtsV[basePts]
    uptakeTimeV = timePtsV[basePts:]
    normScanUptake4M = normScan4M[:, :, :, basePts:]

    return normScanUptake4M, uptakeTimeV, baseline3M, basePts


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
    sigMax = 0.8 * np.nanmax(sigM, axis=1)

    if smoothFlag:
        filtSigM = np.apply_along_axis(
            lambda row: row if np.all(np.isnan(row)) else \
                savgol_filter(row, window_length=getWindowSize(row), polyorder=3),
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
                selFiltSigM[shiftIdxV, :] = selFiltSigM[shiftIdxV, :] - np.tile(
                    np.min(selFiltSigM[shiftIdxV, :], axis=1),
                    (nTime, 1)).transpose()
                filtSigM[~hasPeakV, :] = selFiltSigM
                highSigIdxM[~hasPeakV, :] = filtSigM[~hasPeakV, :] > np.tile(
                    0.8 * np.max(filtSigM[~hasPeakV, :], axis=1),
                    (nTime, 1)).transpose()

    allPeaksM = np.logical_and(localMaxIdxM, highSigIdxM)
    skipVoxV = ~np.any(allPeaksM, axis=1)
    peakIdxV = np.argmax(allPeaksM, axis=1).astype(float)
    if any(skipVoxV):
        peakIdxV[skipVoxV] = 0

    return peakIdxV


def smoothResample(sigM, timeV, temporalSmoothFlag=False, resampFlag=False, minWin=None, maxWin=None):
    """smoothResample
    Function to process uptake curve prior to feature extraction

    Args:
        sigM (np.ndarray, 2D)      : Uptake curves (nVox x nUptakeTime)
        timeV (np.array, 1D)       : Acquisition times (1 x nUptakeTime)
        temporalSmoothFlag (bool)  : [optional, default:False] Smooth curves follg. peak
                                     using cubic splines.
        resampFlag (bool)          : [optional, default:False] Resample uptake curves to 0.1 min
                                     resolution if True.
        minWin (int)               : [optional, default:7] Minimum length of smoothing window (must be odd).
        maxWin (int)               : [optional, default:25] Maximum length of smoothing window (must be odd).

    Returns:
        resampSigM  (np.ndarray, 2D)  : Processed uptake curves (nVox x nResampUptakeTime)
        timeOutV    (np.array, 1D)    : Resampled time pts (min) (1 x nResampUptakeTime)

    """

    origSigM = sigM.copy()

    # Resampling settings
    nPad = 100
    ts = 0.1
    tdiff = timeV[1] - timeV[0]

    padSigM = np.hstack((np.tile(origSigM[:, 0], (nPad, 1)).transpose(), sigM,
                         np.tile(origSigM[:, -1], (nPad, 1)).transpose()))
    padTimeV = np.hstack(
        (np.linspace(timeV[0] - nPad * tdiff, timeV[0] - tdiff, num=nPad, endpoint=True), timeV,
         np.linspace(timeV[-1] + tdiff, timeV[-1] + nPad * tdiff, num=nPad, endpoint=True)))

    # Smoothing settings
    if minWin is None:
        minWin = 7
    if maxWin is None:
        maxWin = 25
    maxWin = min(maxWin, origSigM.shape[1] - 1)
    calcNoiseLevel = lambda sigV: np.std(sigV - medfilt(sigV, kernel_size=3))
    #calcRelativeNoise = lambda sigV: calcNoiseLevel(sigV)/(np.ptp(sigV) + EPS)
    calcRelativeNoise = lambda sigV: calcNoiseLevel(sigV) / (prctile(sigV, 95) - prctile(sigV, 5) + EPS)
    getWindowSize = lambda sigV: max(min(2 * round(calcRelativeNoise(sigV) * len(sigV) / 2) + 1,
                                         maxWin, len(sigV) - 1), minWin)

    if not (resampFlag or temporalSmoothFlag):
        return origSigM, timeV
    else:
        if temporalSmoothFlag:
            # Locate first peak
            peakIdxV = locatePeak(sigM, smoothFlag=True)
            # Smooth signal following first peak
            keepIdxV = np.nansum(origSigM, axis=1) != 0
            selSigM = origSigM[keepIdxV, :]
            selPadSigM = padSigM[keepIdxV, :]
            peakIdxV = peakIdxV[keepIdxV]
            for vox in range(selSigM.shape[0]):
                smoothIdxV = np.arange(int(nPad + peakIdxV[vox] + 1), padSigM.shape[1])
                postPeakPadSigV = selPadSigM[vox, smoothIdxV]
                winSiz = getWindowSize(postPeakPadSigV)
                postPeakSigV = selSigM[vox, int(peakIdxV[vox]) + 1:]
                winSiz = min(winSiz, len(postPeakSigV))
                if winSiz % 2 == 0:
                    winSiz -= 1
                if len(postPeakSigV) < minWin:
                    continue  # skip smoothing
                selSigM[vox, int(peakIdxV[vox]) + 1:] = savgol_filter(postPeakSigV,
                                                                      window_length=winSiz,
                                                                      polyorder=3)
            origSigM[keepIdxV, :] = selSigM
        if resampFlag:
            # Pad signal
            padSigM = np.hstack((np.tile(origSigM[:, 0], (nPad, 1)).transpose(), origSigM,
                                 np.tile(origSigM[:, -1], (nPad, 1)).transpose()))
            padTimeV = np.hstack(
                (np.linspace(timeV[0] - nPad * tdiff, timeV[0] - tdiff, num=nPad, endpoint=True), timeV,
                 np.linspace(timeV[-1] + tdiff, timeV[-1] + nPad * tdiff, num=nPad, endpoint=True)))

            nanIdxV = np.nansum(padSigM, axis=1) == 0
            zeroIdxV = np.sum(padSigM, axis=1) == 0
            skipIdxV = np.logical_and(nanIdxV, ~zeroIdxV)
            padSubSigM = padSigM[~skipIdxV, :]
            numPts = int(padSubSigM.shape[1] * tdiff / ts)
            resampPadSigM = np.full((origSigM.shape[0], numPts), np.nan)
            #FFT
            #resampPadSigM[~skipIdxV, :], timePadV = resample(padSubSigM, numPts, t=padTimeV, axis=1)
            timePadV = np.linspace(padTimeV[0], padTimeV[-1], num=numPts, endpoint=True)
            resampler = CubicSpline(padTimeV, padSubSigM, axis=1, extrapolate=False)
            temp = resampler(timePadV[:numPts])
            resampPadSigM[~skipIdxV, :] = temp
        else:
            resampPadSigM = np.hstack((np.tile(origSigM[:, 0], (nPad, 1)).transpose(), origSigM,
                                       np.tile(origSigM[:, -1], (nPad, 1)).transpose()))
            timePadV = np.hstack(
                (np.linspace(timeV[0] - nPad * tdiff, timeV[0] - tdiff, num=nPad, endpoint=True), timeV,
                 np.linspace(timeV[-1] + tdiff, timeV[-1] + nPad * tdiff, num=nPad, endpoint=True)))
            ts = tdiff

        # Un-pad
        tSkip = round(nPad * tdiff / ts)
        resampSigM = resampPadSigM[:, tSkip:-tSkip]
        timeOutV = timePadV[tSkip:-tSkip]

        return resampSigM, timeOutV


def semiQuantFeatures(procSlcSigM, procTimeV, baselineV, sigType='RSE'):
    """Compute non-parametric features from a pre-processed contrast uptake curve.

    Ref.: Lee, S.H., et al. (2017) "Correlation Between Tumor Metabolism and
    Semiquantitative Perfusion MRI Metrics in Non-small Cell Lung Cancer."
    IJROBP 99.2:S83-S84.

    Args:
        procSlcSigM (np.ndarray, 2D): Processed uptake curves (nVox x nResampleTime).
        procTimeV (np.array, 1D): Acquisition times (1 x nResampleTime) in min.
        baselineV (np.array, 1D): Mean signal before BAT (nVox x 1).
        sigType (str): [optional, default:'RSE'] Convert intensities to relative
            signal enhancement ('RSE') or contrast concentration ('CC').

    Returns:
        featureDict (dict): Dictionary of non-parametric features.
    """

    nVox = procSlcSigM.shape[0]

    # Peak value (enhancement if 'RSE' or  concentration if 'CC')
    # PEv = np.max(procSlcSigM, axis=1)
    # peakIdxV = np.argmax(procSlcSigM, axis=1)
    zeroIdxV = np.sum(procSlcSigM, axis=1) == 0
    nanIdxV = np.logical_and(np.nansum(procSlcSigM, axis=1) == 0, ~zeroIdxV)
    skipIdxV = np.logical_or(nanIdxV, zeroIdxV)
    peakIdxV = np.zeros(nVox, dtype=int)
    peakIdxV[~skipIdxV] = (locatePeak(procSlcSigM[~skipIdxV, :], smoothFlag=True)).astype(int)
    PEv = procSlcSigM[np.arange(nVox), peakIdxV]
    TTPv = procTimeV[peakIdxV]  # Time-to-peak (min)
    TTPv[nanIdxV] = np.nan

    # Half-peak
    halfMaxSig = .5 * PEv
    SHPcolIdx = np.zeros(nVox, dtype=int)
    for vox in range(nVox):
        SHPcolIdx[vox] = np.argmin(np.abs(procSlcSigM[vox, :peakIdxV[vox] + 1] - halfMaxSig[vox, np.newaxis]))
    SHPv = procSlcSigM[np.arange(nVox), SHPcolIdx]  # Value (relative enhancement or concentration) at half-peak
    SHPv[nanIdxV] = np.nan
    TTHPv = procTimeV[SHPcolIdx]  # Time to half-peak (min)
    TTHPv[nanIdxV] = np.nan

    # Wash-in slope
    WISv = PEv / (TTPv + EPS)  # Wash in slope, WIS = PE / TTP
    WISv[nanIdxV] = np.nan

    # Wash-out slope
    # WOS = (PE - RSE(Tend)) / (Tend - TTP), if PE does not occur at Tend (nan otherwise).
    Tend = procTimeV[-1]
    RSEendV = procSlcSigM[:, -1]
    peakAtEndIdxV = TTPv == Tend
    with np.errstate(invalid='ignore'):
        WOSv = (PEv - RSEendV) / (TTPv + EPS - Tend)
        WOSv[peakAtEndIdxV] = np.nan  # Not defined
    WOSv[nanIdxV] = np.nan

    # Time halfway between peak and end of acquisition
    midTimeV = 0.5 * (TTPv + Tend)  # (nVox,) or scalar
    midTimeV = np.atleast_1d(np.asarray(midTimeV, dtype=float))
    TMWv = procTimeV[np.argmin(np.abs(procTimeV[np.newaxis, :] - midTimeV[:, np.newaxis]), axis=1)]  # (nVox,)
    TMWv[peakAtEndIdxV] = np.nan

    # Wash-in/out gradients
    # Initial gradient estimated by linear regression of RSE between 10 % and 70 % PE (occurring prior to peak)
    IGv = np.full((nVox,), fill_value=np.nan)
    for i in range(nVox):
        id_10 = np.argmin(np.abs(procSlcSigM[i, :peakIdxV[i] + 1] - .1 * PEv[i]))
        id_70_rel = np.argmin(np.abs(procSlcSigM[i, id_10:peakIdxV[i] + 1] - .7 * PEv[i]))
        id_70 = id_10 + id_70_rel
        if id_70_rel == 0:
            id_70 = peakIdxV[i]  # Handle case where no column exceeds 70%
        initialPts = np.arange(id_10, id_70 + 1)
        y = procSlcSigM[i, initialPts].T
        x = np.hstack((np.ones((len(initialPts), 1)), procTimeV[initialPts].T[:, np.newaxis]))
        # x = np.column_stack((np.ones(len(initialPts)), procTimeV[initialPts].T))  # Create the design matrix
        b, __, __, __ = np.linalg.lstsq(x, y, rcond=None)
        IGv[i] = b[1]
    IGv[nanIdxV] = np.nan

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
    WOGv[nanIdxV] = np.nan

    # Signal enhancement ratio
    # Defined as in https://doi.org/10.1117/1.JMI.5.1.011019 ; S2-S0/S1-S0
    if sigType == 'RSE':
        S2v = procSlcSigM[:, -1]  #S2/S0
        SERv = (S2v - 1) / (PEv - 1 + EPS)  #PEv: S1/S0
    elif sigType == 'CC':
        C0v = baselineV
        C2v = procSlcSigM[:, -1]
        SERv = (C2v - C0v) / (PEv - C0v + EPS)
    else:
        raise ValueError('Unknown signal type {sigType}.')
    SERv[nanIdxV] = np.nan

    # IAUC
    # IAUC
    IAUCv = cumulative_trapezoid(y=procSlcSigM.T, x=procTimeV.T, axis=0, initial=0).T
    IAUCtthpV = np.full((nVox,), fill_value=np.nan)
    IAUCttpV = np.full((nVox,), fill_value=np.nan)
    for i in range(nVox):
        IAUCtthpV[i] = IAUCv[i, np.nanargmax(procTimeV >= TTHPv[i])]
        IAUCttpV[i] = IAUCv[i, np.nanargmax(procTimeV >= TTPv[i])]
    IAUCtthpV[nanIdxV] = np.nan
    IAUCttpV[nanIdxV] = np.nan

    PEv[nanIdxV] = np.nan

    featureDict = {'PeakEnhancement': PEv,
                   'SignalAtHalfPeak': SHPv,
                   'TimeToPeak': TTPv,
                   'TimeToHalfPeak': TTHPv,
                   'MidWashoutTime': TMWv,
                   'SignalEnhancementRatio': SERv,
                   'WashInSlope': WISv,
                   'WashOutSlope': WOSv,
                   'InitialGradient': IGv,
                   'WashOutGradient': WOGv,
                   'AUCatPeak': IAUCttpV,
                   'AUCatHalfPeak': IAUCtthpV}

    return featureDict, skipIdxV


def calcROIuptakeFeatures(planC, structNum, timeV=None, basePts=None, imgSmoothDict=None, enhThresh=None,
                          sigType='RSE', concDict=None, temporalSmoothFlag=False, resampFlag=False, plotDict=None):
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
            concDict (dict): [optional, default:None] Required if method='CC'. Dictionary of parameters
                         required to compute contrast agent concentration.
                         Required keys: 'T10','r1','TR','FA'. Optional keys:'clip_between' (float) [c1, c2].
            temporalSmoothFlag (bool) : [optional, default:False] Flag specifying whether to
                                     smooth curves follg. peak using cubic splines.
            resampFlag (bool): [optional, default:False] Resample uptake curves to 0.1 min
                                     resolution if True.
            plotDict (dict): [optional, default:None] Display sample plots showing computed features (interactive)


        Returns:
            featureList: List of dictionaries (one per ROI slice) containing uptake features.

    """
    userInputTime = None
    if timeV is not None and len(timeV) > 0:
        userInputTime = timeV

    # Load DCE series
    if concDict is not None and 'concMapIdx' in concDict:
        scanIdxV = concDict['concMapIdx']
        scanArr4M, timePtsV, mask3M, maskSlcV = loadTimeSeq(planC, structNum, userInputTime, scanIdxV)
    else:
        scanArr4M, timePtsV, mask3M, maskSlcV = loadTimeSeq(planC, structNum, userInputTime)

    # Transform signal intensity to
    # relative signal enhancement (signal over baseline intensity) if sigType is 'RSE'  or
    # contrast agent concentration if sigType is 'CC'
    normScan4M, selTimePtsV, baseline3M, basePts = normalizeToBaseline(scanArr4M, mask3M, timePtsV,
                                                                       basePts=basePts,
                                                                       imgSmoothDict=imgSmoothDict,
                                                                       enhThresh=enhThresh,
                                                                       method=sigType,
                                                                       concDict=concDict,
                                                                       plotDict=plotDict)

    # Loop over ROI slices
    featureList = []
    for slc in range(len(maskSlcV)):
        # Reshape to 2D array (nVox x nTimePts)
        normSlc3M = normScan4M[:, :, slc, :]
        normSlcSigM = normSlc3M.reshape(-1, normSlc3M.shape[2], order='F')  # column major
        baselineSlcM = baseline3M[:, :, slc]
        baselineV = baselineSlcM.reshape(-1, order='F')

        # Pre-process
        ## Retain voxels in ROI
        zeroIdxV = np.sum(normSlcSigM, axis=1) == 0
        skipIdxV = np.logical_and(np.nansum(normSlcSigM, axis=1) == 0, ~zeroIdxV)
        #skipIdxV = np.isnan(np.sum(normSlcSigM, axis=1))
        if np.all(skipIdxV):  #No enhancing voxels
            continue
        else:
            normROISlcSigM = normSlcSigM[~skipIdxV, :]
            slcBaselineV = baselineV[~skipIdxV]
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
            featureDict, skipIdxV = semiQuantFeatures(convSlcSigM, procTimeV, slcBaselineV, sigType=sigType)

            origSigM = normROISlcSigM if sigType != 'RSE' else normROISlcSigM - 1
            origTimeV = selTimePtsV

            if plotDict is not None and 'display' in plotDict and plotDict['display']:
                plotSampleFeatures(origSigM, procSlcSigM, origTimeV, procTimeV, featureDict, skipIdxV,
                                   numPlots=1, savePath=plotDict['vox_savepath'],
                                   prefix=plotDict['prefix'] + '_slc' + str(slc))

            featureList.append(featureDict)

    return featureList, basePts


def calcROImeanUptakeFeatures(planC, structNum, timeV=None, basePts=None, imgSmoothDict=None,
                              sigType='RSE', concDict=None, temporalSmoothFlag=False, resampFlag=False,
                              plotDict=None):
    """calcROImeanUptakeFeatures
        Compute non-parametric uptake features from the ROI-average signal (RSE or CC).

        Args:
            planC (plan_container.planC): pyCERR's plan container object
            structNum (int): Index of structure in planC
            timeV (np.array, float): [optional, default:None] User-input acquisition times
            basePts (int): [optional, default:None] Time pt. representing start of uptake.
                           By default, have user input value.
            imgSmoothDict (dict): [optional, default:None] Dictionary specifying whether to
                                  smooth image & associated filter parameters.
                                  Keys: 'kernelSize', 'sigma'.
            sigType (string): [optional, default:'RSE'] Convert intensities to relative signal
                         enhancement ('RSE') or contrast concentration ('CC')
            concDict (dict): [optional, default:None] Required if method='CC'. Dictionary of parameters
                         required to compute contrast agent concentration.
                         Required keys: 'T10','r1','TR','FA'. Optional keys:'clip_between' (float) [c1, c2].
            temporalSmoothFlag (bool): [optional, default:False] Flag specifying whether to
                                     smooth curves following peak.
            resampFlag (bool): [optional, default:False] Resample uptake curves to 0.1 min
                                     resolution if True.
            plotDict (dict): [optional, default:None] Display sample plots showing computed features (interactive)

        Returns:
            featureDict (dict): Dictionary of semi-quantitative features computed on the ROI-mean curve.
            basePts (int): BAT index.
    """
    userInputTime = None
    usePrecomputedCC = False
    if timeV is not None and len(timeV) > 0:
        userInputTime = timeV

    # Load DCE series or pre-computed concentration maps
    if concDict is not None and 'concMapIdx' in concDict:
        usePrecomputedCC = True
        sigType = 'CC'
        concScanIdxV = concDict['concMapIdx']
        scanArr4M, timePtsV, mask3M, maskSlcV = loadTimeSeq(planC, structNum, userInputTime, concScanIdxV)
    else:
        scanArr4M, timePtsV, mask3M, maskSlcV = loadTimeSeq(planC, structNum, userInputTime)

    numSlc = scanArr4M.shape[2]
    nTimePts = scanArr4M.shape[3]

    # Optional spatial smoothing
    smoothScanArr4M = scanArr4M.copy()
    if imgSmoothDict is not None:
        fSize = imgSmoothDict['kernelSize']
        fSigma = imgSmoothDict['sigma']
        for slc in range(numSlc):
            for t in range(nTimePts):
                smoothScanArr4M[:, :, slc, t] = gaussian_filter(scanArr4M[:, :, slc, t], sigma=fSigma,
                                                          mode='nearest', truncate=fSize / fSigma)

    meanSigV = np.mean(scanArr4M[mask3M, :], axis=0)
    meanSmoothSigV = np.mean(smoothScanArr4M[mask3M, :], axis=0)

    # Interactively determine BAT if not provided
    if basePts is None:
        basePts = getStartofUptakeFromMeanSignal(meanSmoothSigV, plotDict)
        if basePts is None:
            raise ValueError("Start of uptake (basePts) could not be determined from user input.")
    if basePts == 0:
        raise ValueError("Please provide a basePts value > 0 that identifies the start of uptake.")

    # Normalize
    meanBaseline = np.mean(meanSmoothSigV[0:basePts])  # Mean signal pre-BAT
    if meanBaseline == 0:
        meanBaseline = EPS
    normMeanSigV = meanSigV / meanBaseline
    timePtsV = timePtsV - timePtsV[basePts]
    uptakeTimeV = timePtsV[basePts:]

    # Convert to RSE/CC
    if sigType == 'CC':
        if usePrecomputedCC:
            # Use pre-computed concentration map
            meanUptakeV = meanSigV[basePts:]
            procInputM = meanUptakeV[np.newaxis, :]
        else:
            # Convert to contrast concentration
            normMeanUptakeM = normMeanSigV[np.newaxis, :]  #reshape to (1, nUptakeTime)
            procInputM = intToConc(normMeanUptakeM, concDict)
            procInputM = procInputM[:, basePts:]
    elif sigType == 'RSE':
        normMeanUptakeV = normMeanSigV[basePts:]
        procInputM = normMeanUptakeV[np.newaxis, :]
    else:
        raise ValueError(f"sigType '{sigType}' not supported.")

    # Smoothing + resampling
    origSigM = procInputM.copy() if sigType != 'RSE' else procInputM.copy() - 1 # RSE = S/S0 - 1
    origTimeV = uptakeTimeV.copy()
    procSigM, procTimeV = smoothResample(procInputM, uptakeTimeV,
                                         temporalSmoothFlag=temporalSmoothFlag,
                                         resampFlag=resampFlag)
    if sigType == 'RSE':
        convSigM = procSigM.copy() - 1  # S(t)/S(0) - 1
    else:
        convSigM = procSigM.copy()

    # Compute features
    baselineV = np.array([meanBaseline])  # (1,)
    featureDict, _ = semiQuantFeatures(convSigM, procTimeV, baselineV, sigType=sigType)

    if plotDict is not None and 'display' in plotDict and plotDict['display']:
        plotSampleFeatures(origSigM, convSigM, origTimeV, procTimeV, featureDict, skipIdxV=None,
                          numPlots=1, savePath=plotDict['vox_savepath'],
                          prefix=plotDict['prefix']+'_roi_mean')

    return featureDict, basePts


def plotSampleFeatures(origSigM, procSlcSigM, origTimeV, procTimeV, featureDict, skipIdxV=None,
                       numPlots=1, savePath=None, prefix=''):
    """plotSampleFeatures
    Function to plot sample uptake curves and indicate extracted features.

    Args:
        origSigM (np.ndarray, 2D)       : Normalized uptake signal (RSE or CC) (nVox x nUptakeTime)
        procSlcSigM  (np.ndarray, 2D)   : Processed uptake curves (RSE or CC) (nVox x nResampUptakeTime)
        origTimeV (np.ndarray, 1D)      : Original acquisition uptake timestamps.
        procTimeV (np.ndarray, 1D)      : Resampled/smoothed acquisition uptake timestamps.
        featureDict (dict)              : Dictionary of semi-quantitative features
        skipIdxV (int)                  : [optional, default:None] Indices of voxels with nan or all-zero signals
        numPlots (int)                  : [optional, default = 1] No. sample plots to display per ROI slice.
        savePath (str, optional)        : Destination directory path to save images. If None, plots interactively.
        prefix (str, optional)          : Filename prefix string used when saving plot images.

    Returns:
        int: Returns 0 upon successful iteration.
    """
    #voxIdxV = rng.integers(low=0, high=procSlcSigM.shape[0], size=numPlots)
    allIdxV = set(range(0, procSlcSigM.shape[0]))
    if skipIdxV is None:
        skipIdxV = np.zeros(procSlcSigM.shape[0], dtype=bool)
    validIdxV = list(allIdxV - set(np.where(skipIdxV)[0]))
    voxIdxV = random.sample(validIdxV, numPlots)

    for idx in voxIdxV:
        plt.figure()

        plt.axis([0, procTimeV[-1], np.min(procSlcSigM[idx, :]) - 0.01, np.max(procSlcSigM[idx, :]) + 0.01])
        plt.plot(origTimeV, origSigM[idx, :], color='gray', alpha=0.7,
                 linewidth=2, linestyle='dashed', label='Original')
        plt.plot(procTimeV, procSlcSigM[idx, :], color='black', linewidth=1, label='SmoothResamp')
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
        point_x2 = featureDict['TimeToPeak'][idx]
        point_y2 = featureDict['PeakEnhancement'][idx]
        cptIdx = int(ctr / 4)
        cpt = (procTimeV[cptIdx], procSlcSigM[idx, cptIdx])
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
        point_x1 = featureDict['TimeToPeak'][idx]
        point_y1 = featureDict['PeakEnhancement'][idx]
        point_x2 = procTimeV[-1]
        point_y2 = procSlcSigM[idx, -1]
        midptIdx = int(ctr + (len(procTimeV) - ctr) / 2)
        midpt = (procTimeV[midptIdx], procSlcSigM[idx, midptIdx])
        slope = featureDict['WashOutSlope'][idx]
        angle = np.rad2deg(np.arctan2(point_y2 - point_y1, point_x2 - point_x1))
        plt.plot([point_x1, point_x2], [point_y1, point_y2], '--', color='purple',
                 label='Wash-out slope', linewidth=1.5)
        #plt.annotate(f'Wash-out slope: {slope:.2f}', xy=midpt, rotation=degrees(atan(slope)),
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
        #plt.plot([x_start, x_end], [y_start, y_end], '--', color='purple', label="Slope Line", linewidth=1.5)
        #plt.text(x_mid, y_mid, f"Initial gradient: {slope:.2f}", rotation=angle,
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
        #plt.plot([x_start, x_end], [y_start, y_end], '--', color='purple', label="Slope Line", linewidth=1.5)
        #plt.text(x_mid, y_mid, f"Wash-out gradient: {slope:.2f}", rotation=angle,
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


def createFeatureMaps(featureList, structNum, planC, importFlag=False, type='scan'):
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
    mask3M = getStrMask(structNum, planC)
    validSlcV = np.sum(np.sum(mask3M, axis=0), axis=0) > 0
    mask3M = mask3M[:, :, validSlcV]

    if importFlag:
        assocScan = planC.structure[structNum].getStructureAssociatedScan(planC)
        xV, yV, zV = planC.scan[assocScan].getScanXYZVals()
        zV = zV[validSlcV]

    # Extract list of available features
    feats = featureList[0].keys()
    numRow, numCol, numSlc = mask3M.shape

    mapDict = {f"{key}": np.full(mask3M.shape, np.nan) for key in feats}

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
            elif type.lower() == 'dose':
                planC = pc.importDoseArray(mapDict[key], xV, yV, zV, planC, assocScan,
                                           doseInfo={'fractionGroupID': key})

    return mapDict, planC


def collectUserInput(saveDir):
    """Display saved uptake-curve plots sequentially and collect user-entered start-of-uptake values.

    Iterates over all PNG files in ``saveDir``, displays each image, prompts the
    user to enter the start-of-uptake time point for that dataset, and saves all
    responses to an Excel file (``user_inputs.xlsx``) in the same directory.

    Args:
        saveDir (str): Path to a directory containing PNG plot files whose
            base-names are used as dataset identifiers.

    Returns:
        int: Always returns 0.
    """
    import pandas as pd

    userInputs = []

    # Read plots from the directory
    plotFiles = [f for f in os.listdir(saveDir) if f.endswith(".png")]

    for plotFile in plotFiles:
        datasetName = os.path.splitext(plotFile)[0]

        # Display plot
        plotPath = os.path.join(saveDir, plotFile)
        img = plt.imread(plotPath)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Dataset: {datasetName}")
        plt.show(block=True)

        # Get user input
        value = input(f"Enter value for {datasetName}: ")
        userInputs.append({"pt ID": datasetName, "Start of uptake": value})
        plt.close()

    # Create dataframe and save to Excel
    df = pd.DataFrame(userInputs)
    df.to_excel(os.path.join(saveDir, "user_inputs.xlsx"), index=False)
    print("User inputs saved to user_inputs.xlsx")

    return 0


def batchSelectStartOfUptake(baseDir, saveDir, timeV=None, strName=None, ):
    """Batch-process a cohort of DCE-MRI datasets to facilitate interactive start-of-uptake selection.

    For each patient sub-directory located under ``baseDir``, this function loads the corresponding DICOM
    image and segmentation into pyCERR's ``planC``, extracts the DCE time sequence, computes the mean ROI
    signal curve for the central slice and writes the uptake curve to a PNG file in ``saveDir``.
    It then invokes :func:`collectUserInput` which allows the user to review the scaved plots sequentially
    and input the BAT in one pass. Any exceptions encountered are recorded to ``exceptions.log`` (``saveDir``).

    Args:
        baseDir (str): Root directory whose immediate sub-directories each
            correspond to one patient / dataset.
        saveDir (str): Directory in which output PNG plots, the collected
            ``user_inputs.xlsx``, and any ``exceptions.log`` are written.
            Created automatically if it does not exist.
        timeV (np.array, float): [optional, default=None, read acquisitionTime]
            or array of user-input acquisition times.
        strName (str): [optional, default=None] Name of ROI in ``planC``; the
            first structure is used by default.

    Returns:
        file: The log-file handle for ``exceptions.log`` or ``None`` if no exceptions were encountered.
    """

    # Directories
    os.makedirs(saveDir, exist_ok=True)

    # Exception log
    exceptions = []

    ptList = os.listdir(baseDir)
    for pt in ptList:

        try:

            ptNum = pt.split("_")[0].split('BC')[-1]
            vNum = pt.split("_")[-1]
            uqID = f"BC{ptNum}{vNum}"

            ptDir = os.path.join(baseDir, pt)
            planC = pc.loadDcmDir(ptDir)
            strList = [structure.structureName for structure in planC.structure]
            if strName is not None:
                strNum = cerrStr.getMatchingIndex(strName, strList, matchCriteria='exact')
            else:
                strNum = 0
            mask = getStrMask(strNum, planC)

            figSavePath = os.path.join(saveDir, pt + '.png')

            # Load time sequence
            scanArr4M, timeOutV, mask3M, maskSlcV = loadTimeSeq(planC, strNum, timeV)

            # Save uptake curve for middle slice
            midSlc = round(len(maskSlcV) / 2) if len(maskSlcV) > 1 else 0
            midSliceSeq3M = scanArr4M[:, :, midSlc, :]
            midSlcMaskM = mask3M[:, :, midSlc]
            roiSize = midSlcMaskM.sum()
            midMask3M = np.repeat(midSlcMaskM[:, :, np.newaxis], midSliceSeq3M.shape[2], axis=2)
            midSliceSeq3M[~midMask3M] = np.nan
            meanSigV = np.nansum(np.nansum(midSliceSeq3M, axis=0), axis=0) / roiSize
            timePtsV = np.arange(0, len(meanSigV))
            plotUptake(timePtsV, meanSigV, blockFlag=False, savePath=figSavePath)

        except Exception as e:
            # Log any exceptions and continue
            exceptions.append((pt, str(e)))

    # Log exceptions to file
    logFile = None
    if exceptions:
        with open(os.path.join(saveDir, "exceptions.log"), "w") as logFile:
            for dataset, error in exceptions:
                logFile.write(f"Dataset: {dataset}\nError: {error}\n")

    collectUserInput(saveDir)

    return logFile
