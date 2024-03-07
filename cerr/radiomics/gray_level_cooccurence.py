import numpy as np
from scipy.sparse import lil_matrix


def calcCooccur(quantizedM, offsetsM, nL, cooccurType=1):
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
        cooccurM = np.empty((numCoOcs, 1), dtype=np.float32)
    else:
        cooccurM = np.empty((numCoOcs, numOffsets), dtype=np.float32)

    for off in range(numOffsets):
        offset = offsetsM[off, :]
        slc1M = q[numRowsPad:(numRowsPad+numRows), numColsPad:(numColsPad+numCols), numSlcsPad:(numSlcsPad+numSlices)]
        slc2M = np.roll(q, offset, axis=(0, 1, 2))
        slc2M = slc2M[numRowsPad:(numRowsPad+numRows), numColsPad:(numColsPad+numCols), numSlcsPad:(numSlcsPad+numSlices)] + (slc1M - 1) * lq
        coccurForOffV = np.histogram(slc2M.flatten(), bins=numCoOcs-1)[1:]
        if cooccurType == 1:
            cooccurM += coccurForOffV
        else:
            cooccurM[:,off] = np.array(coccurForOffV)

    # Ensure symmetry
    cooccurM += cooccurM[np.ix_(indRowV.astype(int)-1, range(numOffsets))]

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
    ''''
    Calculate scalar texture features from cooccurM
    '''''

    # Initialize featureS dictionaries
    # Calculate the number of cooccur matrices (number of columns in cooccurM)
    numCooccurs = cooccurM.shape[1]
    featureS = {
        'energy': np.zeros(numCooccurs),
        'jointEntropy': np.zeros(numCooccurs),
        'jointMax': np.zeros(numCooccurs),
        'jointAvg': np.zeros(numCooccurs),
        'jointVar': np.zeros(numCooccurs),
        'sumAvg': np.zeros(numCooccurs),
        'sumVar': np.zeros(numCooccurs),
        'sumEntropy': np.zeros(numCooccurs),
        'contrast': np.zeros(numCooccurs),
        'invDiffMom': np.zeros(numCooccurs),
        'invDiffMomNorm': np.zeros(numCooccurs),
        'invDiff': np.zeros(numCooccurs),
        'invDiffNorm': np.zeros(numCooccurs),
        'invVar': np.zeros(numCooccurs),
        'dissimilarity': np.zeros(numCooccurs),
        'diffEntropy': np.zeros(numCooccurs),
        'diffVar': np.zeros(numCooccurs),
        'diffAvg': np.zeros(numCooccurs),
        'sumAvg': np.zeros(numCooccurs),
        'sumVar': np.zeros(numCooccurs),
        'sumEntropy': np.zeros(numCooccurs),
        'corr': np.zeros(numCooccurs),
        'clustTendency': np.zeros(numCooccurs),
        'clustShade': np.zeros(numCooccurs),
        'clustPromin': np.zeros(numCooccurs),
        'haralickCorr': np.zeros(numCooccurs),
        'autoCorr': np.zeros(numCooccurs),
        'firstInfCorr': np.zeros(numCooccurs),
        'secondInfCorr': np.zeros(numCooccurs),
    }

    nL = int(np.sqrt(cooccurM.shape[0]))

    # Build levels vector for mu, sig
    levRowV = np.tile(np.arange(1, nL + 1), (1,nL))
    levColV = np.tile(np.arange(1, nL + 1), (nL,1))
    levColV = levColV.flatten(order = "F")
    levColV = levColV[None,:]

    # Build list of indices for px and contrast calculation
    indCtrstC = {}
    indPxC = {}
    px = np.zeros((nL, cooccurM.shape[1]))
    pXminusY = np.zeros((nL, cooccurM.shape[1]))
    pXminusYlogPXminusY = np.zeros((nL, cooccurM.shape[1]))

    for n in range(nL):
        # indices for p(x-y), contrast
        indCtrst1V = np.arange(0, nL - n, dtype = np.uint64)
        indCtrst2V = np.arange(0 + n, nL, dtype = np.uint64)
        indCtrstTmpV = np.concatenate((indCtrst1V + indCtrst2V * nL, indCtrst2V + indCtrst1V * nL))
        indCtrstC[n] = np.unique(indCtrstTmpV)

        # indices for px
        indPxC[n] = np.arange(nL * n, nL * (n + 1), dtype = np.uint64)

        for col in range(cooccurM.shape[1]):
            px[n, col] = np.sum(cooccurM[indPxC[n], col])
            pXminusY[n, col] = np.sum(cooccurM[indCtrstC[n], col])

            if np.any(indCtrstC[n]):
                pXminusYlogPXminusY[n, col] = pXminusY[n, col] * np.log2(pXminusY[n, col] + np.finfo(float).eps)
            else:
                pXminusYlogPXminusY[n, col] = 0

    levRowColSum = levRowV + levColV
    indPxPlusYc = [np.where(levRowColSum == n)[1] for n in range(1, 2 * nL + 1)]


    # Angular Second Moment (Energy)
    featureS['energy'] = np.sum(cooccurM**2, axis=0)

    # Joint Entropy
    featureS['jointEntropy'] = -np.sum(cooccurM * np.log2(cooccurM + np.finfo(float).eps), axis=0)

    # Joint Max
    featureS['jointMax'] = np.max(cooccurM, axis=0)

    # Joint Average
    # featureS['jointAvg'] = np.zeros(cooccurM.shape[1])
    featureS['jointAvg'] = np.sum(cooccurM.T * levRowV, axis = 1)
    # for off in range(cooccurM.shape[1]):
    #     featureS['jointAvg'][off] = np.sum(cooccurM[:,off,None] * levRowV[:,None].reshape((levRowV.shape[1],1)))

    # Joint Variance
    xMinusMu = np.zeros(cooccurM.shape)
    for off in range(len(featureS['jointAvg'])):
        xMinusMu[:,off] = levRowV - featureS['jointAvg'][off]
    featureS['jointVar'] = np.sum(xMinusMu**2 * cooccurM, axis=0)

    # Contrast
    featureS['contrast'] = np.zeros(cooccurM.shape[1])
    for off in range(cooccurM.shape[1]):
        for n in range(nL):
            featureS['contrast'][off] += np.sum(n**2 * cooccurM[indCtrstC[n], off], axis=0)

    # Dissimilarity
    featureS['dissimilarity'] = np.zeros(cooccurM.shape[1])
    for off in range(cooccurM.shape[1]):
        for n in range(nL):
            featureS['dissimilarity'][off] += np.sum(n * cooccurM[indCtrstC[n], off], axis=0)

    # Inverse Difference Moment
    featureS['invDiffMom'] = np.zeros(cooccurM.shape[1])
    for off in range(cooccurM.shape[1]):
        for n in range(nL):
            featureS['invDiffMom'][off] += np.sum((1 / (1 + n**2)) * cooccurM[indCtrstC[n], off], axis=0)

    # Inverse Difference Moment (Normalized)
    featureS['invDiffMomNorm'] = np.zeros(cooccurM.shape[1])
    for off in range(cooccurM.shape[1]):
        for n in range(nL):
            featureS['invDiffMomNorm'][off] += np.sum((1 / (1 + (n / nL)**2)) * cooccurM[indCtrstC[n], off], axis=0)

    # Inverse Difference
    featureS['invDiff'] = np.zeros(cooccurM.shape[1])
    for off in range(cooccurM.shape[1]):
        for n in range(nL):
            featureS['invDiff'][off] += np.sum((1 / (1 + n)) * cooccurM[indCtrstC[n], off], axis=0)

    # Inverse Difference (Normalized)
    featureS['invDiffNorm'] = np.zeros(cooccurM.shape[1])
    for off in range(cooccurM.shape[1]):
        for n in range(nL):
            featureS['invDiffNorm'][off] += np.sum((1 / (1 + n / nL)) * cooccurM[indCtrstC[n], off], axis=0)

    # Inverse Variance
    featureS['invVar'] = np.zeros(cooccurM.shape[1])
    for off in range(cooccurM.shape[1]):
        for n in range(1, nL):
            featureS['invVar'][off] += np.sum((1 / n**2) * cooccurM[indCtrstC[n], off], axis=0)

    # Difference Entropy
    featureS['diffEntropy'] = - np.sum(pXminusYlogPXminusY, axis=0)

    # Difference Variance
    featureS['diffVar'] = np.zeros(cooccurM.shape[1])
    for off in range(cooccurM.shape[1]):
        for n in range(nL):
            featureS['diffVar'][off] += (n - featureS['dissimilarity'][off])**2 * pXminusY[n, off]

    # Difference Average
    featureS['diffAvg'] = featureS['dissimilarity']

    pXplusY = np.zeros((2 * nL, numCooccurs))
    pXplusYlogPXplusY = np.zeros((2 * nL, numCooccurs))

    featureS['sumAvg'] = np.zeros(cooccurM.shape[1])
    featureS['sumEntropy'] = np.zeros(cooccurM.shape[1])
    for off in range(cooccurM.shape[1]):
        for n in range(1, 2 * nL + 1):
            # Calculate p(x+y)
            pXplusY[n - 1, off] = np.sum(cooccurM[indPxPlusYc[n - 1], off], axis=0)

            # Calculate p(x+y) log2(p(x+y))
            if np.any(indPxPlusYc[n - 1]):
                pXplusYlogPXplusY[n - 1, off] = pXplusY[n - 1, off] * np.log2(pXplusY[n - 1, off] + np.finfo(float).eps)
            else:
                pXplusYlogPXplusY[n - 1, off] = 0 #np.zeros(numCooccurs)

            # Calculate Sum Average
            featureS['sumAvg'][off] += n * pXplusY[n - 1, off]

            # Calculate Sum Entropy
            featureS['sumEntropy'][off] -= pXplusYlogPXplusY[n - 1, off]

    # Calculate Sum Variance
    featureS['sumVar'] = np.zeros(cooccurM.shape[1])
    for off in range(cooccurM.shape[1]):
        for n in range(1, 2 * nL + 1):
            featureS['sumVar'][off] += (n - featureS['sumAvg'][off])**2 * pXplusY[n - 1, off]

    # Weighted Pixel Average (mu), Weighted Pixel Variance (sig)
    mu = np.matmul(np.reshape(np.arange(1,nL+1, dtype = np.uint64),(1,nL)),px)
    sig = np.reshape(np.arange(1,nL+1, dtype = np.uint64),(nL,1)) - mu
    sig = np.sum(sig * sig * px, axis = 0)
    sig = sig[:,None]

    # Correlation
    levIMinusMu = levRowV.T - mu
    levJMinusMu = levColV.T - mu
    featureS['corr'] = np.reshape(np.sum(levIMinusMu * levJMinusMu * cooccurM, axis=0), sig.shape) / (sig + np.finfo(float).eps)
    featureS['corr'] = featureS['corr'].T

    clstrV = levIMinusMu + levJMinusMu

    # Cluster Tendency
    featureS['clustTendency'] = np.sum(clstrV**2 * cooccurM, axis=0)

    # Cluster Shade
    featureS['clustShade'] = np.sum(clstrV**3 * cooccurM, axis=0)

    # Cluster Prominence
    featureS['clustPromin'] = np.sum(clstrV**4 * cooccurM, axis=0)

    # Haralick Correlation
    muX = 1 / nL
    sigX = px - muX
    sigX = np.sum(sigX**2, axis=0) / nL
    featureS['haralickCorr'] = (np.matmul(levRowV * levColV,
                                   cooccurM) - muX**2) / (sigX + np.finfo(float).eps)

    # Auto Correlation
    featureS['autoCorr'] =  np.matmul((levRowV * levColV),cooccurM)

    # First Measure of Information Correlation
    logTerm = np.log2((px[levRowV-1, :] + np.finfo(float).eps) * (px[levColV-1, :] + np.finfo(float).eps))
    HXY1 = -np.sum(cooccurM * logTerm, axis=1)
    HX = -np.sum(px * np.log2(px + np.finfo(float).eps), axis=0)
    featureS['firstInfCorr'] = (featureS['jointEntropy'] - HXY1) / HX

    # Second Measure of Information Correlation
    HXY2 = -np.sum(px[levRowV-1, :] * px[levColV-1, :] * np.log2((px[levRowV-1, :]
                + np.finfo(float).eps) * (px[levColV-1, :] + np.finfo(float).eps)), axis=1)
    featureS['secondInfCorr'] = 1 - np.exp(-2 * (HXY2 - featureS['jointEntropy']))
    indZerosV = featureS['secondInfCorr'] <= 0
    featureS['secondInfCorr'][indZerosV] = 0
    featureS['secondInfCorr'] = np.sqrt(featureS['secondInfCorr'])

    return featureS
