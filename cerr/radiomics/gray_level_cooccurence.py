"""

This module contains routines for calculation of Gray Level Co-occurrence matrix and features.

"""

import numpy as np

def calcCooccur(quantizedM, offsetsM, nL, cooccurType=1):
    """This function calculates the cooccurrence matrix for the passed quantized image based on
    IBSI definitions https://ibsi.readthedocs.io/en/latest/03_Image_features.html#grey-level-co-occurrence-based-features

    Args:
        quantizedM (np.ndarray(dtype=int)): quantized 3d matrix obtained, for example, from radiomics.preprocess.imquantize_cerr
        offsetsM (np.ndarray(dtype=int)): Offsets for directionality/neighbors, obtained from radiomics.ibsi1.getDirectionOffsets
        nL (int): Number of gray levels. nL must be less than 65535.
        cooccurType (int): flag, 1 or 2.
                          1: returns a single cooccurrence matrix by combining
                          contributions from all offsets into one cooccurrence
                          matrix.
                          2: returns cooccurM with each column containing
                          cooccurrence matrix for the row of offsetsM.
    Returns:
        np.ndarray: cooccurrence matrix of size (nL*nL) x 1 for cooccurType = 1,
                                             or (nL*nL) x offsetsM.shape[0] for cooccurType = 2.
                    The output can be passed to cooccurToScalarFeatures to get GLCM texture features.
    """

    # Default to building cooccurrence by combining all offsets
    numColsPad = 1
    numRowsPad = 1
    numSlcsPad = 1

    # Get the size of the quantized image
    numRows, numCols, numSlices = quantizedM.shape

    # Pad quantizedM
    q = np.pad(quantizedM, ((numRowsPad, numRowsPad), (numColsPad, numColsPad), (numSlcsPad, numSlcsPad)), constant_values=0)
    q[q==0] = nL + 1  # Replace NaN values with lq

    q = q.astype(np.uint32)  # q is the quantized image
    if q.max() > 65535:
        raise Exception('Number of quantized levels greater than 65535. Increase binWidth to reduce discretized levels')

    # Number of offsets
    numOffsets = offsetsM.shape[0]

    # Indices of last level to filter out
    lq = nL + 1
    nanIndV = np.zeros(lq*lq, dtype=bool)
    nanIndV[lq-1:lq*lq-lq:lq] = True
    nanIndV[lq*lq-lq-1:lq*lq] = True

    # Build linear indices column/row-wise for Symmetry
    indRowV = np.zeros(lq*lq)
    for i in range(1, lq + 1):
        indRowV[(i-1)*lq:i*lq] = np.arange(i, lq*lq+1, lq)

    # Initialize cooccurrence matrix (vectorized for speed)
    numCoOcs = lq * lq
    # if cooccurType == 1:
    #     cooccurM = lil_matrix((numCoOcs, 1), dtype=np.float32)
    # else:
    #     cooccurM = lil_matrix((numCoOcs, numOffsets), dtype=np.float32)
    if cooccurType == 1:
        cooccurM = np.zeros((numCoOcs,1), dtype=np.float32)
    else:
        cooccurM = np.zeros((numCoOcs, numOffsets), dtype=np.float32)

    slc1M = q[numRowsPad:(numRowsPad+numRows+1), numColsPad:(numColsPad+numCols+1), numSlcsPad:(numSlcsPad+numSlices+1)]
    for off in range(numOffsets):
        offset = offsetsM[off, :]
        slc2M = np.roll(q, offset, axis=(0, 1, 2))
        slc2M = slc2M[numRowsPad:(numRowsPad+numRows+1), numColsPad:(numColsPad+numCols+1), numSlcsPad:(numSlcsPad+numSlices+1)] + \
                (slc1M - 1) * lq
        coccurForOffV = np.bincount(slc2M.flatten(), minlength=numCoOcs)[1:]
        if cooccurType == 1:
            cooccurM += np.array(np.reshape(coccurForOffV,(numCoOcs,1)))
        else:
            cooccurM[:,off] = coccurForOffV

    # Ensure symmetry
    cooccurM += cooccurM[indRowV.astype(int)-1, :]

    # Remove rows and columns with NaN
    #cooccurM = cooccurM.tocsr()
    cooccurM = cooccurM[~nanIndV, :]

    # Normalize the cooccurrence matrix
    #col_sum = np.asarray(cooccurM.sum(axis=0))
    col_sum = cooccurM.sum(axis=0)
    col_sum[col_sum == 0] = 1  # Avoid division by zero
    for off in range(cooccurM.shape[1]):
        cooccurM[:,off] /= col_sum[off]

    #return cooccurM.toarray()
    return cooccurM


def cooccurToScalarFeatures(cooccurM):
    """

    This function calculates scalar texture features from cooccurrence matrix as per
    IBSI definitions https://ibsi.readthedocs.io/en/latest/03_Image_features.html#grey-level-co-occurrence-based-features

    Args:
        cooccurM (np.ndarray(dtype=int)): cooccurrence matrix of size (nL*nL) x 1 for combined cooccurrrences from all the directions,
                                                             or (nL*nL) x offsetsM.shape[0] for individual directions or patches.

    Returns:
        dict: dictionary with scalar texture features as its
         fields. Each field's value is a vector containing the feature values
         for each column cooccurM.

    """

    # Vectorized: the sums over |i-j| = n and i+j = n are bincounts over the
    # flattened matrix, so no Python loop over gray levels or directions is needed.
    # Formulas, output shapes and key order are unchanged.
    P = cooccurM
    numCooccurs = P.shape[1]
    nL = int(np.sqrt(P.shape[0]))
    eps = np.finfo(float).eps

    # Level of each element of the flattened (nL*nL) matrix
    flatIndV = np.arange(nL * nL)
    levRowV = (flatIndV % nL + 1).astype(float)
    levColV = (flatIndV // nL + 1).astype(float)
    diffIndV = np.abs(flatIndV % nL - flatIndV // nL)   # |i - j|
    sumIndV = (flatIndV % nL + 1) + (flatIndV // nL + 1)  # i + j

    Pd = P.astype(float)
    px = np.stack([np.bincount(flatIndV // nL, weights=Pd[:, col], minlength=nL)
                   for col in range(numCooccurs)], axis=1)
    pXminusY = np.stack([np.bincount(diffIndV, weights=Pd[:, col], minlength=nL)
                         for col in range(numCooccurs)], axis=1)
    pXplusY = np.stack([np.bincount(sumIndV, weights=Pd[:, col], minlength=2 * nL + 1)[1:]
                        for col in range(numCooccurs)], axis=1)
    diffLevV = np.arange(nL, dtype=float)[:, None]          # n = |i - j|
    sumLevV = np.arange(1, 2 * nL + 1, dtype=float)[:, None]  # n = i + j

    featureS = {}

    # Angular Second Moment (Energy)
    featureS['energy'] = np.sum(P ** 2, axis=0)

    # Joint Entropy
    featureS['jointEntropy'] = -np.sum(P * np.log2(P + eps), axis=0)

    # Joint Max
    featureS['jointMax'] = np.max(P, axis=0)

    # Joint Average
    featureS['jointAvg'] = np.sum(P.T * levRowV[None, :], axis=1)

    # Joint Variance
    featureS['jointVar'] = np.sum((levRowV[:, None] - featureS['jointAvg'][None, :]) ** 2 * P, axis=0)

    # Sum Average, Sum Variance, Sum Entropy
    featureS['sumAvg'] = np.sum(sumLevV * pXplusY, axis=0)
    featureS['sumVar'] = np.sum((sumLevV - featureS['sumAvg'][None, :]) ** 2 * pXplusY, axis=0)
    featureS['sumEntropy'] = -np.sum(pXplusY * np.log2(pXplusY + eps), axis=0)

    # Contrast
    featureS['contrast'] = np.sum(diffLevV ** 2 * pXminusY, axis=0)

    # Inverse Difference Moment (and normalized)
    featureS['invDiffMom'] = np.sum(pXminusY / (1 + diffLevV ** 2), axis=0)
    featureS['invDiffMomNorm'] = np.sum(pXminusY / (1 + (diffLevV / nL) ** 2), axis=0)

    # Inverse Difference (and normalized)
    featureS['invDiff'] = np.sum(pXminusY / (1 + diffLevV), axis=0)
    featureS['invDiffNorm'] = np.sum(pXminusY / (1 + diffLevV / nL), axis=0)

    # Inverse Variance
    featureS['invVar'] = np.sum(pXminusY[1:] / diffLevV[1:] ** 2, axis=0)

    # Dissimilarity
    featureS['dissimilarity'] = np.sum(diffLevV * pXminusY, axis=0)

    # Difference Entropy, Difference Variance, Difference Average
    featureS['diffEntropy'] = -np.sum(pXminusY * np.log2(pXminusY + eps), axis=0)
    featureS['diffVar'] = np.sum((diffLevV - featureS['dissimilarity'][None, :]) ** 2 * pXminusY, axis=0)
    featureS['diffAvg'] = featureS['dissimilarity']

    # Weighted Pixel Average (mu), Weighted Pixel Variance (sig)
    levV = np.arange(1, nL + 1, dtype=float)[:, None]
    mu = np.sum(levV * px, axis=0)[None, :]
    sig = np.sum((levV - mu) ** 2 * px, axis=0)[:, None]

    # Correlation
    levIMinusMu = levRowV[:, None] - mu
    levJMinusMu = levColV[:, None] - mu
    featureS['corr'] = (np.sum(levIMinusMu * levJMinusMu * P, axis=0)[:, None] / (sig + eps)).T

    # Cluster Tendency, Shade, Prominence
    clstrV = levIMinusMu + levJMinusMu
    featureS['clustTendency'] = np.sum(clstrV ** 2 * P, axis=0)
    featureS['clustShade'] = np.sum(clstrV ** 3 * P, axis=0)
    featureS['clustPromin'] = np.sum(clstrV ** 4 * P, axis=0)

    # Auto Correlation (elementwise sum rather than matmul, which emits spurious
    # divide/overflow warnings with some BLAS builds, e.g. Apple Accelerate)
    featureS['autoCorr'] = np.sum((levRowV * levColV)[:, None] * Pd, axis=0)[None, :]

    # Haralick Correlation
    muX = 1 / nL
    sigX = np.sum((px - muX) ** 2, axis=0) / nL
    featureS['haralickCorr'] = (featureS['autoCorr'] - muX ** 2) / (sigX + eps)

    # First and Second Measures of Information Correlation
    pxI = px[flatIndV % nL, :]
    pxJ = px[flatIndV // nL, :]
    logTerm = np.log2((pxI + eps) * (pxJ + eps))
    HXY1 = -np.sum(P * logTerm, axis=0)[None, :]
    HX = -np.sum(px * np.log2(px + eps), axis=0)
    featureS['firstInfCorr'] = (featureS['jointEntropy'] - HXY1) / HX
    HXY2 = -np.sum(pxI * pxJ * logTerm, axis=0)[None, :]
    secondInfCorr = 1 - np.exp(-2 * (HXY2 - featureS['jointEntropy']))
    secondInfCorr[secondInfCorr <= 0] = 0
    featureS['secondInfCorr'] = np.sqrt(secondInfCorr)

    # Same key order as before
    keyOrder = ['energy', 'jointEntropy', 'jointMax', 'jointAvg', 'jointVar', 'sumAvg', 'sumVar',
                'sumEntropy', 'contrast', 'invDiffMom', 'invDiffMomNorm', 'invDiff', 'invDiffNorm',
                'invVar', 'dissimilarity', 'diffEntropy', 'diffVar', 'diffAvg', 'corr',
                'clustTendency', 'clustShade', 'clustPromin', 'haralickCorr', 'autoCorr',
                'firstInfCorr', 'secondInfCorr']
    return {key: featureS[key] for key in keyOrder}
