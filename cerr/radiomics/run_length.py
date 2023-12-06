import numpy as np

####### ========= New implementation
import pydantic.schema


def calcRLM(quantizedM, offsetsM, nL, rlmType=1):
    if rlmType != 1:
        rlmOut = []

    pad_width = [(1, 1)] * quantizedM.ndim
    quantizedM = np.pad(quantizedM, pad_width, mode='constant', constant_values=0)

    numRowsPad, numColsPad, numSlcsPad = 1, 1, 1
    q = np.pad(quantizedM, ((numRowsPad, numRowsPad), (numColsPad, numColsPad), (numSlcsPad, numSlcsPad)),
               mode='constant', constant_values=0)
    q[np.isnan(q)] = 0
    q = q.astype(np.uint16)

    numOffsets = offsetsM.shape[0]
    maxRunLen = int(np.ceil(np.max(quantizedM.shape)*2))
    #maxRunLen = 1000
    #print('Max Run Length = ' + str(maxRunLen))

    rlmM = np.zeros((nL, maxRunLen))
    if rlmType == 2:
        rlmOut = [np.zeros((nL, maxRunLen)) for i in range(numOffsets)]

    siz = np.array(q.shape)
    rowV = np.arange(siz[0],dtype = np.uint16)[:,None]
    colV = np.arange(siz[1],dtype = np.uint16)[None,:]

    rowIndM = np.repeat(rowV,siz[1], axis = 1)[:,:,None]
    colIndM = np.repeat(colV,siz[0], axis = 0)[:,:,None]
    rowIndM = np.repeat(rowIndM, siz[2], axis = 2)
    colIndM = np.repeat(colIndM, siz[2], axis = 2)
    slcIndM = np.empty(siz,dtype = np.uint16)
    for slc in range(siz[2]):
        slcIndM[:,:,slc] = slc


    for off in range(numOffsets):
        if rlmType == 2:
            rlmM = rlmOut[off]

        offset = offsetsM[off]
        rolled_q = np.roll(q, offset, axis=(0, 1, 2))
        for level in range(1, nL + 1):
            #t = time.time()
            prevM = (q == level).astype(int)
            # diffM = prevM - np.roll(prevM, offset, axis=(0, 1, 2))
            diffM = prevM - (rolled_q == level).astype(int)
            startM = diffM == 1
            #startIndV = np.where(startM)
            #convergedV = np.full(len(startIndV[0]),False,dtype=bool)
            rowStartV = rowIndM[startM]
            colStartV = colIndM[startM]
            slcStartV = slcIndM[startM]
            startIndV = (rowStartV, colStartV, slcStartV)
            stopM = diffM == -1
            convergedV = np.full(len(rowStartV),False,dtype=bool)
            count = 0
            while not np.all(convergedV):
                count += 1
                nextIndV = startIndV + count * offset[:,None]
                nextIndV[nextIndV < 0] = 0
                nextIndV[nextIndV >= siz[:,None]] = 0 # assign 0 as it will have value of 0
                newConvergedV = stopM[nextIndV[0,:],nextIndV[1,:],nextIndV[2,:]]
                numConverged = (~convergedV[newConvergedV]).sum()
                convergedV[newConvergedV] = True
                rlmM[level - 1, count-1] = numConverged

        if rlmType == 2:
            #rlmOut.append(rlmM)
            rlmOut[off] = rlmM

    if rlmType == 1:
        rlmOut = rlmM

    return rlmOut


##### ======= Matlab implementation that uses np.roll (circshift)
def calcRLM_old(quantizedM, offsetsM, nL, rlmType=1):
    if rlmType != 1:
        rlmOut = []

    pad_width = [(1, 1)] * quantizedM.ndim
    quantizedM = np.pad(quantizedM, pad_width, mode='constant', constant_values=0)

    numRowsPad, numColsPad, numSlcsPad = 1, 1, 1
    q = np.pad(quantizedM, ((numRowsPad, numRowsPad), (numColsPad, numColsPad), (numSlcsPad, numSlcsPad)),
               mode='constant', constant_values=0)
    q[np.isnan(q)] = 0
    q = q.astype(np.uint16)

    numOffsets = offsetsM.shape[0]
    maxRunLen = int(np.ceil(np.max(quantizedM.shape)*2))
    #maxRunLen = 1000
    print('Max Run Length = ' + str(maxRunLen))

    rlmM = np.zeros((nL, maxRunLen))
    if rlmType == 2:
        rlmOut = [np.zeros((nL, maxRunLen)) for i in range(numOffsets)]

    #lenV = np.zeros(np.prod(q.shape), dtype = np.uint16)

    siz = q.shape

    for off in range(numOffsets):
        if rlmType == 2:
            rlmM = rlmOut[off]

        offset = offsetsM[off]

        for level in range(1, nL + 1):
            #t = time.time()
            prevM = (q == level).astype(int)
            #diffM = (q == level).astype(int) - (np.roll(q, offset, axis=(0, 1, 2)) == level).astype(int)
            diffM = prevM - np.roll(prevM, offset, axis=(0, 1, 2))
            startM = diffM == 1
            startIndV = np.where(startM)

            #prevM = (q == level).astype(np.uint16)
            prevM = prevM.astype(np.uint16)
            convergedM = ~startM

            #elapsed = time.time() - t
            #print('Level start = ' + str(level) + '. Time = ' + str(elapsed))
            #t = time.time()

            #lenV *= 0
            #start = 0
            while not np.all(convergedM):
                nextM = np.roll(prevM, -offset, axis=(0, 1, 2))
                addM = prevM + nextM
                newConvergedM = (addM == prevM)
                toUpdateM = ~convergedM & newConvergedM
                prevM = nextM
                prevM[startIndV] = addM[startIndV]
                lenV = addM[toUpdateM]
                # if len(lenV) > 0:
                #     countsV = np.bincount(lenV)
                #     numCounts = len(countsV) - 1
                #     rlmM[level - 1, :numCounts] = rlmM[level - 1, :numCounts] + countsV[1:]
                if len(lenV) > 0:
                    rlmM[level - 1, :] = rlmM[level - 1, :] + np.bincount(lenV, minlength=maxRunLen+1)[1:]
                #stop = start + len(addM[toUpdateM])
                #lenV[start:stop] = addM[toUpdateM]
                #start = stop
                convergedM = convergedM | newConvergedM
            #rlmM[level - 1, :] = np.bincount(lenV[:stop], minlength=maxRunLen+1)[1:]
            #elapsed = time.time() - t
            #print('Level end = ' + str(level) + '. Time = ' + str(elapsed))

        if rlmType == 2:
            #rlmOut.append(rlmM)
            rlmOut[off] = rlmM

    if rlmType == 1:
        rlmOut = rlmM

    return rlmOut


def rlmToScalarFeatures(rlmM, numVoxels):
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
        lenV = np.arange(1, rlmM[dirNum].shape[1] + 1)
        levV = np.arange(1, nL + 1)
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
