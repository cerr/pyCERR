import numpy as np

def compute_boundingbox(x3D, maskFlag=0):
    # Finding the bounding box parameters
    # If maskFlag > 1, it is interpreted as a padding parameter

    maskFlag = int(maskFlag)

    iV, jV, kV = np.where(x3D)
    minr = np.min(iV)
    maxr = np.max(iV)
    minc = np.min(jV)
    maxc = np.max(jV)
    mins = np.min(kV)
    maxs = np.max(kV)

    bboxmask = None

    if maskFlag != 0:
        bboxmask = np.zeros_like(x3D)

        if maskFlag > 0:
            minr -= maskFlag
            maxr += maskFlag
            if maxr > x3D.shape[0]:
                maxr = x3D.shape[0]
            if minr < 0:
                minr = 0
            minc -= maskFlag
            maxc += maskFlag
            if maxc > x3D.shape[1]:
                maxc = x3D.shape[1]
            if minc < 0:
                minc = 0
            mins -= maskFlag
            maxs += maskFlag
            if maxs > x3D.shape[2]:
                maxs = x3D.shape[2]
            if mins < 0:
                mins = 0

        maxarr = [maxr, maxc, maxs]
        minarr = [minr, minc, mins]
        minarr = [max(1, val) for val in minarr]
        minr, minc, mins = minarr[0], minarr[1], minarr[2]
        bboxmask[minarr[0]:maxarr[0], minarr[1]:maxarr[1], minarr[2]:maxarr[2]] = 1

    return minr, maxr, minc, maxc, mins, maxs, bboxmask
