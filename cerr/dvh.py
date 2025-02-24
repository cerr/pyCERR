import numpy as np
from cerr.dataclasses import scan as scn

def getDVH(structNum, doseNum, planC):
    """
    Returns DVH vectors for a specified structure and dose set, where
    dosesV is a vector of dose values at a voxel and volsV is a vector of
    volumes of the corresponding voxel in dosesV.
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

    # Add 0 to the beginning of volsHistV
    volsHistV = np.insert(volsHistV, 0, 0)

    cumVolsV = np.cumsum(volsHistV)
    cumVols2V = cumVolsV[-1] - cumVolsV
    ind = np.argmax(doseBinsV >= doseCutoff)

    if ind is None:
        vx = 0
    else:
        vx = cumVols2V[ind]

    if volumeType == 1:
        vx = vx / cumVolsV[-1]
    else:
        # warning('Vx is being calculated in absolute terms.')
        pass

    return vx


def Dx(doseBinsV,volsHistV,x,volType):

    # Assuming volsHistV, cumVolsV, doseBinsV, volType, and x are defined elsewhere

    # Check if volType variable exists
    if 'volType' not in locals():
        # warning('Input volume assumed to be in percentage. Set volType=0 for absolute values.')
        pass
    else:
        if not volType:
            x = x / np.sum(volsHistV) * 100

    cumVolsV = np.cumsum(volsHistV)
    cumVols2V = cumVolsV[-1] - cumVolsV
    ind = np.argmin(np.array(cumVols2V) / cumVolsV[-1] < x / 100)

    if ind is None:
        dx = 0
    else:
        dx = doseBinsV[ind]
    return dx

def meanDose(doseBinsV,volsHistV):
    return np.sum(doseBinsV * volsHistV) / np.sum(volsHistV)

def minDose(doseBinsV,volsHistV):
    ind = np.where(volsHistV != 0)[0][0]
    return doseBinsV[ind]

def maxDose(doseBinsV,volsHistV):
    ind = np.where(volsHistV != 0)[0][-1]
    return doseBinsV[ind]

def medianDose(doseBinsV,volsHistV):
    # Find the indices of non-zero elements in volsHistV
    non_zero_indices = np.where(volsHistV != 0)[0]

    # Calculate the median of these indices
    ind = np.median(non_zero_indices)

    # Calculate the floor and ceil indices for median
    ind1 = int(np.floor(ind))
    ind2 = int(np.ceil(ind))

    return (doseBinsV[ind1] + doseBinsV[ind2]) / 2
