import numpy as np

def getDown2Mask(inM, sample):
    sV = inM.shape
    vec = list(range(0, sV[1], sample))

    maskM = np.zeros_like(inM, dtype=bool)

    vec2 = list(range(0, sV[0], sample))

    for ind in vec2:
        maskM[ind, vec] = True

    return maskM

def getDown3Mask(mask3M, sampleTrans, sampleAxis):
    sV = mask3M.shape
    sampleSlices = int(np.ceil(sV[2] / sampleAxis))

    outMask3M = np.zeros_like(mask3M, dtype=bool)

    indV = list(range(0, sV[2], sampleAxis))
    for i in indV:
        maskM = getDown2Mask(outMask3M[:, :, i], sampleTrans)
        outMask3M[:, :, i] = maskM

    return outMask3M


def getSurfacePoints(mask3M, sampleTrans=1, sampleAxis=1):
    surfPoints = []

    r, c, s = np.where(mask3M)

    minR, maxR = np.min(r), np.max(r)
    minC, maxC = np.min(c), np.max(c)
    minS, maxS = np.min(s), np.max(s)

    croppedMask3M = mask3M[minR:maxR + 1, minC:maxC + 1, minS:maxS + 1]

    plusRowShift = croppedMask3M[2:, 1:-1, 1:-1]
    allNeighborsOn = plusRowShift.copy()

    minusRowShift = croppedMask3M[:-2, 1:-1, 1:-1]
    allNeighborsOn &= minusRowShift

    plusColShift = croppedMask3M[1:-1, 2:, 1:-1]
    allNeighborsOn &= plusColShift

    minusColShift = croppedMask3M[1:-1, :-2, 1:-1]
    allNeighborsOn &= minusColShift

    plusSlcShift = croppedMask3M[1:-1, 1:-1, 2:]
    allNeighborsOn &= plusSlcShift

    minusSlcShift = croppedMask3M[1:-1, 1:-1, :-2]
    allNeighborsOn &= minusSlcShift

    kernal = croppedMask3M[1:-1, 1:-1, 1:-1] & ~allNeighborsOn

    if sampleTrans is not None and sampleAxis is not None:
        if sampleTrans > 1 or sampleAxis > 1:
            kernal = kernal & getDown3Mask(kernal, sampleTrans, sampleAxis)

    if sampleTrans is not None and sampleAxis is not None:
        croppedMask3M = croppedMask3M & getDown3Mask(croppedMask3M,sampleTrans, sampleAxis)

    croppedMask3M[1:-1, 1:-1, 1:-1] = kernal

    r, c, s = np.where(croppedMask3M)

    r += minR
    c += minC
    s += minS

    surfPoints = np.column_stack((r, c, s))

    return surfPoints
