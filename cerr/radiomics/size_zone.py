"""
This module contains routines for claculation of Size Zone texture features
"""

import numpy as np
from scipy.ndimage import label

def calcSZM(quantized3M, nL, szmType):
    """

    This function calculates the Size Zone Matrix for the passed quantized image based on
    IBSI definitions https://ibsi.readthedocs.io/en/latest/03_Image_features.html#grey-level-size-zone-based-features

    Args:
        quantized3M (np.ndarray(dtype=int)): quantized 3d matrix obtained, for example, from radiomics.preprocess.imquantize_cerr
        nL (int): Number of gray levels.
        szmType: flag, 1 or 2.
                   1: 3D zones
                   2: 2D zones
    Returns:
        np.ndarray: size-zone matrix of size (nL x L)

        The output can be passed to szmToScalarFeatures to get SZM texture features.

    """

    if szmType == 1:
        s = np.ones((3,3,3))
    else:
        s = np.ones((3,3))

    sizeV = quantized3M.shape
    szmM = np.zeros((nL, quantized3M.size), dtype=int)
    maxSiz = 0
    for level in range(1, nL+1):
        if szmType == 1:
            connM, num_features = label(quantized3M == level, structure=s)
        else:
            connM = np.zeros(sizeV, dtype=int)
            featOffset = 0
            for slc in range(sizeV[2]):
                connSlcM, num_features = label(quantized3M[:,:,slc] == level, structure=s)
                indNoZero = connSlcM > 0
                connSlcM[indNoZero] += featOffset
                connM[:,:,slc] = connSlcM
                featOffset += num_features
        regionSizV = np.bincount(connM[connM > 0])
        if len(regionSizV) > 0:
            maxSiz = max(maxSiz, max(regionSizV))
        counts = np.bincount(regionSizV)[1:]
        szmM[level - 1, :len(counts)] = counts
    szmM = szmM[:, :maxSiz]
    return szmM


def szmToScalarFeatures(szmM, numVoxels):
    """

    This function calculates scalar texture features from Size Zone Matrix as per
    IBSI definitions https://ibsi.readthedocs.io/en/latest/03_Image_features.html#grey-level-size-zone-based-features

    Args:
        szmM (np.ndarray(dtype=int)): size-zone matrix of size (nL x L)
        numVoxels (int): number of voxels in the region of interest for szmM calculation

    Returns:
        dict: dictionary with scalar texture features as its
             fields. Each field's value is a vector containing the feature values
             for each list element of rlmM.

    """

    featureS = {}

    nL, maxLength = szmM.shape
    lenV = np.arange(1, maxLength + 1, dtype = np.uint64)
    levV = np.arange(1, nL + 1, dtype = np.uint64)
    lenV = lenV[None,:]
    levV = levV[None,:]

    szmM = szmM.astype(float)

    saeM = szmM / lenV**2
    featureS["smallAreaEmphasis"] = np.sum(saeM) / np.sum(szmM)

    laeM = szmM * lenV**2
    featureS["largeAreaEmphasis"] = np.sum(laeM) / np.sum(szmM)

    featureS["grayLevelNonUniformity"] = np.sum(np.sum(szmM, axis=1)**2) / np.sum(szmM)

    featureS["grayLevelNonUniformityNorm"] = np.sum(np.sum(szmM, axis=1)**2) / np.sum(szmM)**2

    featureS["sizeZoneNonUniformity"] = np.sum(np.sum(szmM, axis=0)**2) / np.sum(szmM)

    featureS["sizeZoneNonUniformityNorm"] = np.sum(np.sum(szmM, axis=0)**2) / np.sum(szmM)**2

    if numVoxels is None:
        numVoxels = 1
    featureS["zonePercentage"] = np.sum(szmM) / numVoxels

    lglzeM = szmM.T / levV**2
    featureS["lowGrayLevelZoneEmphasis"] = np.sum(lglzeM) / np.sum(szmM)

    hglzeM = szmM.T * levV**2
    featureS["highGrayLevelZoneEmphasis"] = np.sum(hglzeM) / np.sum(szmM)

    levLenM = levV.T**2 * lenV**2
    salgleM = szmM / levLenM
    featureS["smallAreaLowGrayLevelEmphasis"] = np.sum(salgleM) / np.sum(szmM)

    levLenM = levV.T**2 * lenV**2
    lahgleM = szmM * levLenM
    featureS["largeAreaHighGrayLevelEmphasis"] = np.sum(lahgleM) / np.sum(szmM)

    levLenM = levV.T**2 / lenV**2
    sahgleM = szmM * levLenM
    featureS["smallAreaHighGrayLevelEmphasis"] = np.sum(sahgleM) / np.sum(szmM)

    levLenM = (1/levV.T**2) * lenV**2
    lalgleM = szmM * levLenM
    featureS["largeAreaLowGrayLevelEmphasis"] = np.sum(lalgleM) / np.sum(szmM)


    iPij = szmM.T / szmM.sum() * levV
    mu = np.sum(iPij)
    iMinusMuPij = szmM.T / np.sum(szmM) * (levV - mu)**2
    featureS["grayLevelVariance"] = np.sum(iMinusMuPij)

    jPij = szmM / np.sum(szmM) * lenV
    mu = np.sum(jPij)
    jMinusMuPij = szmM / np.sum(szmM) * (lenV - mu)**2
    featureS["sizeZoneVariance"] = np.sum(jMinusMuPij)

    zoneSum = szmM.sum()
    featureS["zoneEntropy"] = -np.sum((szmM / zoneSum) * np.log2((szmM / zoneSum) + np.finfo(float).eps))

    return featureS
