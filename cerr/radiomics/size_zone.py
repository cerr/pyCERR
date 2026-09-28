"""
This module contains routines for claculation of Size Zone texture features
"""

import numpy as np
from skimage.measure import label

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

    # Zones are connected regions of equal grey level. skimage's label connects
    # neighbours with EQUAL values, so one pass labels the zones of every level
    # (connectivity 3 = 26-connected in 3D, 2 = 8-connected in 2D). 0 is outside the ROI.
    q = np.nan_to_num(np.asarray(quantized3M, dtype=float), nan=0).astype(np.int32)

    if szmType == 1:
        zoneSizeV, zoneLevelV = _zoneSizesAndLevels(q, connectivity=3)
    else:
        sizeL, levelL = [], []
        for slc in range(q.shape[2]):
            sizV, levV = _zoneSizesAndLevels(q[:, :, slc], connectivity=2)
            sizeL.append(sizV)
            levelL.append(levV)
        zoneSizeV = np.concatenate(sizeL)
        zoneLevelV = np.concatenate(levelL)

    maxSiz = int(zoneSizeV.max()) if zoneSizeV.size else 0
    szmM = np.zeros((nL, maxSiz), dtype=int)
    np.add.at(szmM, (zoneLevelV - 1, zoneSizeV - 1), 1)
    return szmM


def _zoneSizesAndLevels(q, connectivity):
    """Size and grey level of every zone (connected region of equal non-zero level) in q."""
    labelM = label(q, background=0, connectivity=connectivity)
    labelV = labelM.ravel()
    zoneSizeV = np.bincount(labelV)[1:]
    # Grey level of each zone, read from any one of its voxels
    roiIndV = np.flatnonzero(labelV)
    voxelOfZoneV = np.zeros(zoneSizeV.size + 1, dtype=np.int64)
    voxelOfZoneV[labelV[roiIndV]] = roiIndV
    zoneLevelV = q.ravel()[voxelOfZoneV[1:]].astype(np.int64)
    return zoneSizeV, zoneLevelV


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

    # Keep only zone sizes that occur. Empty columns add nothing to any feature,
    # and a single large zone would otherwise make every (nL x maxSize) temporary
    # below hundreds of MB.
    sizeIndV = np.flatnonzero(np.sum(szmM, axis=0))
    szmM = szmM[:, sizeIndV]
    nL = szmM.shape[0]
    lenV = (sizeIndV + 1).astype(np.uint64)
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
