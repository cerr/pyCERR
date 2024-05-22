import numpy as np

def compute_boundingbox(binaryMaskM, is2DFlag=False, maskFlag=0):
    # Finding extents of bounding box given a binary mask
    # If maskFlag > 0, it is interpreted as a padding parameter
    # If sliceWiseFlag is True slicewise extents are returned


    maskFlag = int(maskFlag)

    if is2DFlag:

        iV, jV = np.where(binaryMaskM)
        kV = []
        minr = np.min(iV)
        maxr = np.max(iV)
        minc = np.min(jV)
        maxc = np.max(jV)
        mins = []
        maxs = []
    else:
        iV, jV, kV = np.where(binaryMaskM)
        minr = np.min(iV)
        maxr = np.max(iV)
        minc = np.min(jV)
        maxc = np.max(jV)
        mins = np.min(kV)
        maxs = np.max(kV)

    bboxmask = None

    if maskFlag != 0:
        bboxmask = np.zeros_like(binaryMaskM)

        if maskFlag > 0:
            siz = binaryMaskM.shape
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

            else:
                mins -= maskFlag
                maxs += maskFlag
                if maxs >= siz[2]:
                    maxs = siz[2] - 1
                if mins < 0:
                    mins = 0

        if is2DFlag:
            bboxmask[minr:maxr+1, minc:maxc+1] = 1
        else:
            bboxmask[minr:maxr+1, minc:maxc+1, mins:maxs+1] = 1

    return minr, maxr, minc, maxc, mins, maxs, bboxmask
