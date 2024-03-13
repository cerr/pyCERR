import numpy as np

def compute_boundingbox(x3D, maskFlag=0):
    # Finding the bounding box parameters
    # If maskFlag > 0, it is interpreted as a padding parameter

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
            siz = x3D.shape
            minr -= maskFlag
            maxr += maskFlag
            if maxr >= siz[0]:
                maxr = siz[0] - 1
            if minr < 0:
                minr = 0
            minc -= maskFlag
            maxc += maskFlag
            if maxc >= siz[1]:
                maxc = siz[1] - 1
            if minc < 0:
                minc = 0
            mins -= maskFlag
            maxs += maskFlag
            if maxs >= siz[2]:
                maxs = siz[2] - 1
            if mins < 0:
                mins = 0

        bboxmask[minr:maxr+1, minc:maxc+1, mins:maxs+1] = 1

    return minr, maxr, minc, maxc, mins, maxs, bboxmask
