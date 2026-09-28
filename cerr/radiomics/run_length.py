"""
This module contains routines for calculation of Run Length texture features
"""

import numpy as np

def calcRLM(quantizedM, offsetsM, nL, rlmType=1):
    """

    This function calculates the run-length matrix for the passed quantized image based on
    IBSI definitions https://ibsi.readthedocs.io/en/latest/03_Image_features.html#grey-level-run-length-based-features

    Args:
        quantizedM (np.ndarray(dtype=int)): quantized 3d matrix obtained, for example, from radiomics.preprocess.imquantize_cerr
        offsetsM (np.ndarray(dtype=int)): Offsets for directionality/neighbors, obtained from radiomics.ibsi1.getDirectionOffsets
        nL (int): Number of gray levels. nL must be less than 65535.
        rlmType (int): flag, 1 or 2.
                          1: returns a single run length matrix by combining
                          contributions from all offsets into one cooccurrence
                          matrix.
                          2: returns a list of run length matrices, per row row of offsetsM.
    Returns:
        np.ndarray: run-length matrix of size (nL x L) for rlmType = 1,
                    list of size equal to the number of directions for rlmType = 2.

                    The output can be passed to rlmToScalarFeatures to get RLM texture features.

    """

    # Zero-pad so every step from a ROI voxel lands inside the array. 0 marks
    # voxels outside the ROI, so runs always end in the padding.
    maxRunLen = int(np.ceil((np.max(quantizedM.shape) + 2) * 2))
    pad = max(2, int(np.max(np.abs(offsetsM))) + 1)
    q = np.pad(quantizedM, pad, mode='constant', constant_values=0)
    q[np.isnan(q)] = 0
    q = q.astype(np.uint16)
    flatV = q.ravel()
    stridesV = np.array(q.strides) // q.itemsize

    # All ROI voxels (flat indices)
    roiIndV = np.flatnonzero(flatV)

    numOffsets = offsetsM.shape[0]
    rlmM = np.zeros((nL, maxRunLen))
    rlmOut = []
    for off in range(numOffsets):
        if rlmType == 2:
            rlmM = np.zeros((nL, maxRunLen))

        # A step along this offset is a fixed stride in the flattened array
        step = int(np.dot(offsetsM[off], stridesV))

        # Run starts: ROI voxels whose predecessor along -offset has a different level
        startIndV = roiIndV[flatV[roiIndV - step] != flatV[roiIndV]]
        levelV = flatV[startIndV].astype(np.int64)

        # Advance all open runs together, one step at a time. A run ends at the
        # first voxel with a different level; count it and drop it from the active set.
        # Each voxel is visited once per offset, independent of the number of levels.
        curIndV = startIndV.copy()
        lenV = np.ones(len(startIndV), dtype=np.int64)
        activeV = np.arange(len(startIndV))
        while activeV.size:
            nextIndV = curIndV[activeV] + step
            continuesV = flatV[nextIndV] == levelV[activeV]
            endedV = activeV[~continuesV]
            np.add.at(rlmM, (levelV[endedV] - 1, lenV[endedV] - 1), 1)
            activeV = activeV[continuesV]
            curIndV[activeV] = nextIndV[continuesV]
            lenV[activeV] += 1

        if rlmType == 2:
            rlmOut.append(rlmM)

    if rlmType == 1:
        rlmOut = rlmM

    return rlmOut


def rlmToScalarFeatures(rlmM, numVoxels):
    """

    This function calculates scalar texture features from run length matrix as per
    IBSI definitions https://ibsi.readthedocs.io/en/latest/03_Image_features.html#grey-level-run-length-based-features

    Args:
        rlmM (np.ndarray(dtype=int)): run-length matrix of size (nL x L) for rlmType = 1,
                    list of size equal to the number of directions for rlmType = 2.

    Returns:
        dict: dictionary with scalar texture features as its
             fields. Each field's value is a vector containing the feature values
             for each list element of rlmM.

    """

    featureS = {}
    numDirs = 1
    if isinstance(rlmM,list):
        numDirs = len(rlmM)
    else:
        rlmM = [rlmM]

    featureS['shortRunEmphasis'] = np.zeros(numDirs)
    featureS['longRunEmphasis'] = np.zeros(numDirs)
    featureS['grayLevelNonUniformity'] = np.zeros(numDirs)
    featureS['grayLevelNonUniformityNorm'] = np.zeros(numDirs)
    featureS['runLengthNonUniformity'] = np.zeros(numDirs)
    featureS['runLengthNonUniformityNorm'] = np.zeros(numDirs)
    featureS['runPercentage'] = np.zeros(numDirs)
    featureS['lowGrayLevelRunEmphasis'] = np.zeros(numDirs)
    featureS['highGrayLevelRunEmphasis'] = np.zeros(numDirs)
    featureS['shortRunLowGrayLevelEmphasis'] = np.zeros(numDirs)
    featureS['shortRunHighGrayLevelEmphasis'] = np.zeros(numDirs)
    featureS['longRunLowGrayLevelEmphasis'] = np.zeros(numDirs)
    featureS['longRunHighGrayLevelEmphasis'] = np.zeros(numDirs)
    featureS['grayLevelVariance'] = np.zeros(numDirs)
    featureS['runLengthVariance'] = np.zeros(numDirs)
    featureS['runEntropy'] = np.zeros(numDirs)

    for dirNum in range(numDirs):
        nL = rlmM[dirNum].shape[0]
        lenV = np.arange(1, rlmM[dirNum].shape[1] + 1, dtype = np.uint64)
        levV = np.arange(1, nL + 1, dtype = np.uint64)
        lenV = lenV[None,:]
        levV = levV[None,:]

        sreM = rlmM[dirNum] / lenV**2
        featureS['shortRunEmphasis'][dirNum] = np.sum(sreM) / np.sum(rlmM[dirNum])

        lreM = rlmM[dirNum] * lenV**2
        featureS['longRunEmphasis'][dirNum] = np.sum(lreM) / np.sum(rlmM[dirNum])

        featureS['grayLevelNonUniformity'][dirNum] = np.sum(np.sum(rlmM[dirNum], axis=1)**2) / np.sum(rlmM[dirNum])

        featureS['grayLevelNonUniformityNorm'][dirNum] = np.sum(np.sum(rlmM[dirNum], axis=1)**2) / np.square(np.sum(rlmM[dirNum]))

        featureS['runLengthNonUniformity'][dirNum] = np.sum(np.square(np.sum(rlmM[dirNum], axis=0))) / np.sum(rlmM[dirNum])

        featureS['runLengthNonUniformityNorm'][dirNum] = np.sum(np.sum(rlmM[dirNum], axis=0)**2) / np.square(np.sum(rlmM[dirNum]))

        if numVoxels is None:
            numVoxels = 1
        featureS['runPercentage'][dirNum] = np.sum(rlmM[dirNum]) / numVoxels

        lglreM = rlmM[dirNum] / levV.T**2
        featureS['lowGrayLevelRunEmphasis'][dirNum] = np.sum(lglreM) / np.sum(rlmM[dirNum])

        hglreM = rlmM[dirNum] * levV.T**2
        featureS['highGrayLevelRunEmphasis'][dirNum] = np.sum(hglreM) / np.sum(rlmM[dirNum])

        levLenM = levV.T**2 * lenV**2
        srlgleM = rlmM[dirNum] / levLenM
        featureS['shortRunLowGrayLevelEmphasis'][dirNum] = np.sum(srlgleM) / np.sum(rlmM[dirNum])

        levLenM = levV.T**2 / lenV**2
        srhgleM = rlmM[dirNum] * levLenM
        featureS['shortRunHighGrayLevelEmphasis'][dirNum] = np.sum(srhgleM) / np.sum(rlmM[dirNum])

        levLenM = 1/levV.T**2 * lenV**2
        lrlgleM = np.multiply(rlmM[dirNum], levLenM)
        featureS['longRunLowGrayLevelEmphasis'][dirNum] = np.sum(lrlgleM) / np.sum(rlmM[dirNum])

        levLenM = levV.T**2 * lenV**2
        lrhgleM = rlmM[dirNum] * levLenM
        featureS['longRunHighGrayLevelEmphasis'][dirNum] = np.sum(lrhgleM) / np.sum(rlmM[dirNum])

        iPij = rlmM[dirNum].T / np.sum(rlmM[dirNum]) * levV
        mu = np.sum(iPij)
        iMinusMuPij = rlmM[dirNum].T / np.sum(rlmM[dirNum]) * (levV - mu)**2
        featureS['grayLevelVariance'][dirNum] = np.sum(iMinusMuPij)

        jPij = rlmM[dirNum] / np.sum(rlmM[dirNum]) * lenV
        mu = np.sum(jPij)
        jMinusMuPij = rlmM[dirNum] / np.sum(rlmM[dirNum]) *  np.square(lenV - mu)
        featureS['runLengthVariance'][dirNum] = np.sum(jMinusMuPij)

        runSum = np.sum(rlmM[dirNum])
        featureS['runEntropy'][dirNum] = -np.sum(rlmM[dirNum] / runSum * np.log2(rlmM[dirNum] / runSum + np.finfo(float).eps))

    return featureS
