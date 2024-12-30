import numpy as np
from matplotlib import pyplot as plt

from scipy.signal import resample
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumtrapz

from cerr import plan_container as pc
from cerr.contour.rasterseg import getStrMask
from cerr.utils.statistics import round

EPS = np.finfo(float).eps
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
        baselineM[baselineM == 0] = EPS
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
    nTime = sigM.shape[1]
    sigMax = 0.8 * np.max(sigM, axis=1)

    diffNextM = np.concatenate((np.zeros((nVox, 1)), np.diff(sigM, 1, 1)), axis=1)
    diffPrevM = np.concatenate((-np.diff(sigM, 1, 1), np.zeros((nVox, 1))), axis=1)
    localMaxIdxM = np.logical_and(diffNextM >= 0, diffPrevM >= 0)
    highSigIdxM = sigM > np.tile(sigMax,(nTime, 1)).transpose()

    allPeaksM = np.logical_and(localMaxIdxM, highSigIdxM)
    skipVoxV = ~np.any(allPeaksM, axis=1)
    peakIdxV = np.argmax(allPeaksM, axis=1).astype(float)
    peakIdxV[skipVoxV] = np.nan

    return peakIdxV

def smoothResample(sigM, timeV, temporalSmoothDict=None, resampFlag=False):
    """smoothResample
    Function to process uptake curve prior to feature extraction

    Args:
        sigM (np.ndarray, 2D)      : Uptake curves (nVox x nUptakeTime)
        timeV (np.array, 1D)       : Acquisition times (1 x nUptakeTime)
        temporalSmoothDict (dict)  : [optional, default:None] Dictionary specifying whether to
                                     smooth curves follg. peak & associated filter parameters.
                                     Keys:  'smoothFlag', 'kernelSize', 'sigma'
        resampFlag (bool)          : [optional, default:False] Resample uptake curves to 0.1 min
                                     resolution if True.

    Returns:
        resampSigM  (np.ndarray, 2D)  : Processed uptake curves (nVox x nResampUptakeTime)
        timeOutV    (np.array, 1D)    : Resampled time pts (min) (1 x nResampUptakeTime)

    """

    # Smoothing filter settings
    if temporalSmoothDict is None:
        smoothFlag = False
    else:
        smoothSettings = temporalSmoothDict.keys()
        smoothFlag = temporalSmoothDict['smooth']
        sigma = 1
        radius = 2
        if 'sigma' in smoothSettings:
            sigma = temporalSmoothDict['sigma']
        if 'radius' in smoothSettings:
            radius = temporalSmoothDict['radius']

    # Resampling settings
    nPad = 100
    ts = 0.01
    tdiff = timeV[1] - timeV[0]

    # Pad signal
    padSigM = np.hstack((np.tile(sigM[:, 0], (nPad, 1)).transpose(), sigM,
               np.tile(sigM[:, -1], (nPad, 1)).transpose()))
    padTimeV = np.hstack((np.linspace(timeV[0] - nPad * tdiff, timeV[0] - tdiff, num=nPad, endpoint=True), timeV,
                          np.linspace(timeV[-1] + tdiff, timeV[-1] + nPad * tdiff, num=nPad, endpoint=True)))

    if not (resampFlag or smoothFlag):
        return sigM, timeV
    else:
        if smoothFlag:
            # Locate first peak
            peakIdxV = locatePeak(sigM)
            # Smooth signal following first peak
            keepIdxV = np.nansum(padSigM, axis=1) != 0
            selPadSigM = padSigM[keepIdxV, :]
            peakIdxV = peakIdxV[keepIdxV]
            for vox in range(selPadSigM.shape[0]):
                padSigM[:, int(nPad + peakIdxV[vox] + 1):] = gaussian_filter1d(selPadSigM[vox, int(nPad + peakIdxV[vox] + 1):],
                                                                          sigma=sigma,radius=radius, axis=0)
        if resampFlag:
            skipIdxV = np.isnan(np.nansum(padSigM, axis=1))
            padSubSigM = padSigM[~skipIdxV, :]
            numPts = int(padSubSigM.shape[1] * tdiff /ts)
            resampPadSigM = np.full((sigM.shape[0], numPts), np.nan)
            resampPadSigM[~skipIdxV, :], __ = resample(padSubSigM, numPts, t=padTimeV, axis=1)
        else:
            resampPadSigM = padSigM
            ts = tdiff
        # Un-pad
        tSkip = round(nPad * tdiff / ts)
        resampSigM = resampPadSigM[:, tSkip:-tSkip]
        timeOutV = np.linspace(0, (resampSigM.shape[1] - 1) * ts, num=resampSigM.shape[1])

        return resampSigM, timeOutV


def computeFeatures(procSlcSigM, procTimeV):

    relEnhancementM = procSlcSigM - 1  # S(t)/S(0) - 1
    nVox = relEnhancementM.shape[0]

    # Peak enhancement
    PEv = np.max(relEnhancementM, axis=1)
    peakIdxV = np.argmax(relEnhancementM, axis=1)
    TTPv = procTimeV[peakIdxV]             #Time-to-peak

    # Half-peak
    halfMaxSig = (np.max(procSlcSigM, axis=1) - 1) / 2
    SHPcolIdx = np.argmax(relEnhancementM >= halfMaxSig[:, np.newaxis], axis=1)
    SHPv = procSlcSigM[np.arange(len(procSlcSigM)), SHPcolIdx] #Signal at half-peak
    TTHPv = procTimeV[SHPcolIdx]           #Time to half-peak

    # Wash-in / wash-out slopes
    WISv = PEv / (TTPv + EPS)              # Wash in slope, WIS = PE / TTP
    Tend = procTimeV[-1]
    RSEendV = relEnhancementM[:, -1]
    peakAtEndIdx = TTPv == Tend
    WOSv = (PEv - RSEendV)/ (TTPv - Tend)  # WOS = (PE - RSE(Tend)) / (TTP – Tend), if PE does not occur at Tend
    WOSv[peakAtEndIdx] = 0

    # Wash-in/out gradients
    ## Initial gradient estimated by linear regression of RSE between 20 % and 80 % PE
    id_20v = np.argmax(relEnhancementM >= .2 * PEv[:, np.newaxis], axis=1)
    id_80v = np.argmax(relEnhancementM > .8 * PEv[:, np.newaxis], axis=1)
    id_80v = id_80v - 1
    IGv = np.full((nVox, ), fill_value=np.nan)
    igIdxV = np.full(relEnhancementM.shape, fill_value=False)
    for i in range(nVox):
        idxV = np.arange(id_20v[i], id_80v[i]+1)
        y = relEnhancementM[i, idxV].T
        x = np.hstack((np.ones((len(idxV), 1)), procTimeV[idxV].T[:,np.newaxis]))
        b, __, __, __ = np.linalg.lstsq(x, y, rcond=None)
        IGv[i] = b[1]
        igIdxV[i, idxV] = True

    ## Wash-out gradient estimated by linear regression of RSE between PE and 1 min post-PE
    t0 = peakIdxV
    t1IdxM = procTimeV[:,None] >= (procTimeV[t0] + 1)
    skipRowV = ~np.any(t1IdxM, axis=0)
    t1 = np.argmax(t1IdxM, axis=0)
    WOGv = np.full((nVox, ), fill_value=np.nan)
    for i in range(nVox):
        if ~skipRowV[i]:
            x = np.hstack((np.ones((t1[i] - t0[i] + 1, 1)), procTimeV[t0[i]:t1[i]+1][:, np.newaxis]))
            y = relEnhancementM[i, t0[i]: t1[i]+1].T
            b, __, __, __ = np.linalg.lstsq(x, y, rcond=None)
            WOGv[i] = b[1]
        else:
            WOGv[i] = np.nan

    #Signal enhancement ratio
    tse1 = np.argmax(procTimeV >= .5)
    tse2 = np.argmax(procTimeV >= 2.5)
    SERv = relEnhancementM[:, tse1] / relEnhancementM[:, tse2]

    #IAUC
    IAUCv = cumtrapz(y=relEnhancementM.T, x=procTimeV.T, axis=0, initial=0).T
    IAUCtthpV = np.full((nVox,), fill_value=np.nan)
    IAUCttpV = np.full((nVox,), fill_value=np.nan)
    for i in range(nVox):
        IAUCtthpV[i] = IAUCv[i, np.argmax(procTimeV >= TTHPv[i])]
        IAUCttpV[i] = IAUCv[i, np.argmax(procTimeV >= TTPv[i])]

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
        ## Retain voxels in ROI
        skipIdxV = np.isnan(np.nansum(normSlcSigM, axis=1))
        normROISlcSigM = normSlcSigM[~skipIdxV, :]
        ## Smoothing + resampling
        procSlcSigM, procTimeV = smoothResample(normROISlcSigM, selTimePtsV,
                                               temporalSmoothDict=temporalSmoothDict, resampFlag=resampFlag)

        # Compute features
        featureDict = computeFeatures(procSlcSigM, procTimeV)
        featureList.append(featureDict)

    return featureList

def createFeatureMaps(featureList, strNum, planC, importFlag=False):

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
    numFeats = len(feats)
    numRow, numCol, numSlc = mask3M.shape

    mapDict = {f"{key}": np.zeros_like(mask3M, dtype=float) for key in feats}

    # Create 3D maps
    for key in feats:
        for s in range(numSlc):
            maskSlcM = mask3M[:, :, s]
            mapDict[key][...,s][maskSlcM] = featureList[s][key]
            # Import as pseudo-scan array
        if importFlag:
            planC = pc.importScanArray(mapDict[key], xV, yV, zV, key, assocScan, planC)

    return mapDict, planC