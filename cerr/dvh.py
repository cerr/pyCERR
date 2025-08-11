import numpy as np
from cerr.dataclasses import scan as scn

EPS = np.finfo(float).eps

def getDVH(structNum, doseNum, planC):
    """Routine to calculate Dose and Volume vectors to be used for Histogram calculation

    Args:
        structNum (int): Binary mask where 1s represent the segmentation
        doseNum (int): x-values i.e. coordinates of columns of input mask
        planC (cerr.plan_container.PlanC): pyCERR's plan container object

    Returns:
        List (dosesV): vector of dose values for voxels in structNum
        List (volsV): vector of volumes corresponding to voxels in dosesV
        int (isError): error flag. 0: No error, 1: error in calculation

    """


    # Get the scan number associated with the requested structure.
    assocScanUID = planC.structure[structNum].assocScanUID
    scanSet = scn.getScanNumFromUID(assocScanUID,planC)

    ROIImageSize = [
        planC.scan[scanSet].scanInfo[0].sizeOfDimension1,
        planC.scan[scanSet].scanInfo[0].sizeOfDimension2
    ]

    deltaY = planC.scan[scanSet].scanInfo[0].grid1Units

    # Get raster segments for structure.
    segmentsM = planC.structure[structNum].rasterSegments

    isError = 0
    if not np.any(segmentsM):
        isError = 1
    numSegs = segmentsM.shape[0]

    # Relative sampling of ROI voxels in this place, compared to CT spacing.
    # Set when rasterSegments are generated (usually on import).
    sampleRate = 1

    # Sample the rows
    indFullV = list(range(1, numSegs + 1))
    if sampleRate != 1:
        indFullV = [i for i in range(1, len(indFullV) + 1) if (i + sampleRate - 1) % sampleRate == 0]

    # Block process to avoid swamping on large structures
    DVHBlockSize = 50

    blocks = int(np.ceil(len(indFullV) / DVHBlockSize))
    volsV = []
    dosesV = []

    start = 0

    for b in range(blocks):

        # Build the interpolation points matrix

        dummy = np.zeros(DVHBlockSize * ROIImageSize[1])
        x1V = dummy.copy()
        y1V = dummy.copy()
        z1V = dummy.copy()
        volsSectionV = dummy.copy()

        if start + DVHBlockSize > len(indFullV):
            stop = len(indFullV)
        else:
            stop = start + DVHBlockSize

        indV = indFullV[start:stop]
        tol = 1e-5
        mark = 0
        for i in indV:

            tmpV = segmentsM[i - 1, 0:10]
            delta = tmpV[4] * sampleRate
            xV = np.arange(tmpV[2], tmpV[3]+tol, delta)
            len_x = len(xV)
            rangeV = np.ones(len_x)
            yV = tmpV[1] * rangeV
            zV = tmpV[0] * rangeV
            sliceThickness = tmpV[9]
            v = delta * (deltaY * sampleRate) * sliceThickness
            x1V[mark: (mark + len_x)] = xV
            y1V[mark: (mark + len_x)] = yV
            z1V[mark: (mark + len_x)] = zV
            volsSectionV[mark:mark + len_x] = v
            mark += len_x

        # Cut unused matrix elements
        x1V = x1V[:mark]
        y1V = y1V[:mark]
        z1V = z1V[:mark]
        volsSectionV = volsSectionV[:mark]

        # Interpolate.
        dosesSectionV = planC.dose[doseNum].getDoseAt(x1V, y1V, z1V)

        dosesV.extend(dosesSectionV)
        volsV.extend(volsSectionV)

        start = stop

    # volsV = volsV * sampleRate**2  # must account for sampling rate!
    dosesV = np.asarray(dosesV,dtype=float)
    volsV = np.asarray(volsV,dtype=float)

    return dosesV, volsV, isError

def accumulate(V1, V2, indV):
    for i in range(len(V2)):
        V1[indV[i]] += V2[i]
    return V1

def doseHist(doseV, volsV, binWidth):
    """Routine to calculate Dose Volume Histogram from input dose and volume vectors

    Args:
        doseV (List): vector of dose values for voxels in structNum
        volsV (List): vector of volumes corresponding to voxels in dosesV
        binWidth (float): Bin-width of dose bins.

    Returns:
        List (doseBinsV): vector of dose bin centers.
        List (volsHistV): vector of volumes accumulated in corresponding dose bins.

    """

    bufferNum = 1e-10
    if np.min(doseV) >= 0:
        maxD = np.max(doseV)

        indV = np.asarray(np.ceil(bufferNum + (doseV / binWidth)),dtype=int) - 1

        maxBin = np.ceil(bufferNum + (maxD / binWidth))

        doseBinsV = (np.arange(1, maxBin + 1) - 1) * binWidth + binWidth / 2

        volsHistV = np.zeros(int(maxBin), dtype=float)

        volsHistV = accumulate(volsHistV, volsV, indV)

    else:
        maxD = np.max(doseV)
        minD = np.min(doseV)

        indV = np.asarray(np.ceil(doseV / binWidth),dtype=int)
        indV = indV - np.min(indV)

        maxBin = np.ceil((maxD / binWidth) + bufferNum)
        minBin = np.ceil((minD / binWidth))

        doseBinsV = (np.arange(minBin, maxBin + 1) - 1) * binWidth + binWidth / 2

        volsHistV = np.zeros(int(maxBin - minBin + 1), dtype=float)

        volsHistV = accumulate(volsHistV, volsV, indV)

    return doseBinsV, volsHistV


#
# # Example usage
# doseV = np.array([1.2, 2.5, 3.5, 2.0, 1.5, 3.0])
# volsV = np.array([10, 20, 15, 25, 30, 40])
# binWidth = 1.0
#
# doseBinsV, volsHistV = doseHist(doseV, volsV, binWidth)
# print("Dose Bins:", doseBinsV)
# print("Volumes Histogram:", volsHistV)

def MOHx(doseBinsV,volsHistV,percent):
    """
    This routine computes the mean of hottest x% dose.

    Args:
        doseBinsV: (List): vector of dose bin centers.
        volsHistV (List): vector of volumes accumulated in corresponding dose bins.
        percent (float): cutoff in terms of percentage. (e.g. 90)

    Returns:
        Float: mean of the hottest x% dose
    """

    cumVolsV = np.cumsum(volsHistV)
    cumVols2V = cumVolsV[-1] - cumVolsV

    inds = np.where(cumVols2V / cumVolsV[-1] <= percent / 100)[0]

    if len(inds) == 0:
        moh = 0
    else:
        moh = float(np.sum(doseBinsV[inds] * volsHistV[inds]) / np.sum(volsHistV[inds]))
        # To get min dose: result = doseBinsV[inds[0]]

    return moh


def MOCx(doseBinsV,volsHistV,percent):
    """
    This routine computes the mean of coldest x% dose.

        doseBinsV: (List): vector of dose bin centers.
        volsHistV (List): vector of volumes accumulated in corresponding dose bins.
        percent (float): cutoff in terms of percentage. (e.g. 10)

    Returns:
        Float: mean of the coldest x% dose

    """

    cumVolsV = np.cumsum(volsHistV)
    cumVols2V = cumVolsV[-1] - cumVolsV

    inds = np.where(cumVols2V / cumVolsV[-1] >= (100-percent) / 100)[0]

    if len(inds) == 0:
        moc = 0
    else:
        moc = float(np.sum(doseBinsV[inds] * volsHistV[inds]) / np.sum(volsHistV[inds]))
        # To get min dose: result = doseBinsV[inds[0]]
    return moc


def Vx(doseBinsV,volsHistV,doseCutoff,volumeType):
    """
    This routine computes the volume receiving at least x dose.

    Args:
        doseBinsV: (List): vector of dose bin centers.
        volsHistV (List): vector of volumes accumulated in corresponding dose bins.
        doseCutoff: dose cutoff in Gy.
        volumeType (int): 0: Return output volume as absolute cc.
                          1: Return output volume as percentage.

    Returns:
        Float: Volume (absolute ot percentage)
    """

    # Add 0 to the beginning of volsHistV
    volsHistV = np.insert(volsHistV, 0, 0)

    cumVolsV = np.cumsum(volsHistV)
    cumVols2V = cumVolsV[-1] - cumVolsV
    inds = np.where(doseBinsV >= doseCutoff)
    if len(inds) == 0:
        vx = 0
    else:
        vx = cumVols2V[np.min(inds)]

    if volumeType == 1:
        vx = vx / cumVolsV[-1]
    else:
        # warning('Vx is being calculated in absolute terms.')
        pass

    return vx


def Dx(doseBinsV,volsHistV,volCutoff,volType):
    """"
    This routine computes the minimum dose to the hottest x% volume.

    Args:
        doseBinsV: (List): vector of dose bin centers.
        volsHistV (List): vector of volumes accumulated in corresponding dose bins.
        volCutoff: volume cutoff in cc or percentage.
        volumeType (int): 0: volume is input in absolute cc.
                          1: volume is input in percentage.

    Returns:
        Float: Volume (absolute ot percentage)
    """

    # Check if volType variable exists
    if 'volType' not in locals():
        # warning('Input volume assumed to be in percentage. Set volType=0 for absolute values.')
        pass
    else:
        if not volType:
            volCutoff = volCutoff / np.sum(volsHistV) * 100

    cumVolsV = np.cumsum(volsHistV)
    cumVols2V = cumVolsV[-1] - cumVolsV
    ind = np.where(np.array(cumVols2V) / cumVolsV[-1] < volCutoff / 100)
    if len(ind[0]) > 0:
        ind = np.min(ind)
        dx = doseBinsV[ind]
    else:
        dx = 0

    return dx

def meanDose(doseBinsV,volsHistV):
    """
    This routine computes the mean dose

    Args:
        doseBinsV: (List): vector of dose bin centers.
        volsHistV (List): vector of volumes accumulated in corresponding dose bins.

    Returns:
        Float: Mean dose

    """

    return np.sum(doseBinsV * volsHistV) / np.sum(volsHistV)

def minDose(doseBinsV,volsHistV):
    """
    This routine computes the minimum dose

    Args:
        doseBinsV: (List): vector of dose bin centers.
        volsHistV (List): vector of volumes accumulated in corresponding dose bins.

    Returns:
        Float: Minimum dose

    """

    ind = np.where(volsHistV != 0)[0][0]
    return doseBinsV[ind]

def maxDose(doseBinsV,volsHistV):
    """
    This routine computes the maximum dose

    Args:
        doseBinsV: (List): vector of dose bin centers.
        volsHistV (List): vector of volumes accumulated in corresponding dose bins.

    Returns:
        Float: Maximum dose

    """

    ind = np.where(volsHistV != 0)[0][-1]
    return doseBinsV[ind]

def medianDose(doseBinsV,volsHistV):
    """
    This routine computes the median dose

    Args:
        doseBinsV: (List): vector of dose bin centers.
        volsHistV (List): vector of volumes accumulated in corresponding dose bins.

    Returns:
        Float: Median dose

    """

    # Find the indices of non-zero elements in volsHistV
    non_zero_indices = np.where(volsHistV != 0)[0]

    # Calculate the median of these indices
    ind = np.median(non_zero_indices)

    # Calculate the floor and ceil indices for median
    ind1 = int(np.floor(ind))
    ind2 = int(np.ceil(ind))

    return (doseBinsV[ind1] + doseBinsV[ind2]) / 2

def eud(doseBinsV, volsHistV, exponent):
    """
    This routine computes the equivalent uniform dose given DVH and exponent.

    Args:
        doseBinsV: (List): vector of dose bin centers.
        volsHistV (List): vector of volumes accumulated in corresponding dose bins.
        exponent (int): exponent.

    Returns:
        Float: EUD

    """
    cumVolsV = np.cumsum(volsHistV)
    cumVols2V = cumVolsV[-1] - cumVolsV
    ind = np.max(np.where(volsHistV != 0)[0])
    totalVolume = np.sum(volsHistV)
    a = exponent + EPS

    result = np.sum((doseBinsV ** a) * (volsHistV / totalVolume)) ** (1 / a)
    return result