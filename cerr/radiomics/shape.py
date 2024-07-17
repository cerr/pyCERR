"""shape module.

The shape module contains routines for calculation of shape features.

"""

import numpy as np
from cerr.utils.mask import getSurfacePoints
from scipy.spatial import distance
from skimage import measure
from cerr.utils.mask import computeBoundingBox

def trimeshSurfaceArea(v,f):
    """Routine to calculate surface area from vertices and faces of triangular mesh

    Args:
        v (numpy.array): (numPoints x 3) vertices of triangular mesh
        f (numpy.array): (numFaces x 3) faces of triangular mesh

    Returns:
        float: Surface area
    """

    v1 = (v[f[:, 1], :] - v[f[:, 0], :])
    v2 = (v[f[:, 2], :] - v[f[:, 0], :])

    # Calculate the cross product and its norm
    cross_product = np.cross(v1, v2)
    cross_product_norm = np.linalg.norm(cross_product, axis=1)

    # Calculate the area of each triangle
    area = np.sum(cross_product_norm) / 2

    return area

def vectorNorm3d(v):
    return np.linalg.norm(v, axis=1)

def eig(a):
    return np.sort(np.linalg.eig(a)[0])

def sepsq(a, b):
    return np.sum((a - b)**2, axis=0)

def calcMaxDistBetweenPts(ptsM, distType):
    """This routine calculates the maximum distance between the input points

    Args:
        ptsM (numpy.array): (nunPoints x 3) coordinates of points
        distType (str or Callable): Type of distance. E.g. 'euclidean' as supported by scipy.spatial.distance.cdist

    Returns:
        float: Maximum distance between the input points
    """

    dmax = 0
    numPts = ptsM.shape[0]
    step = 1000
    numSteps = numPts // step
    if numSteps > 0:
        remPts = numPts % step
        startV = np.arange(numSteps) * step
        stopV = startV + step
        if remPts > 0:
            startV = np.append(startV,stopV[-1])
            stopV = np.append(stopV,numPts)
    else:
        startV = np.arange(1)
        stopV = startV + numPts

    for i in range(len(startV)):
        iStart = startV[i]
        iStop = stopV[i]
        for j in range(len(startV)):
            jStart = startV[j]
            jStop = stopV[j]
            distM = distance.cdist(ptsM[iStart:iStop,:], ptsM[jStart:jStop,:], distType)
            dmax = max(dmax, np.max(distM))

    return dmax

def calcShapeFeatures(mask3M, xValsV, yValsV, zValsV):
    """Routine to calculate shape features for the inout mask and grid

    Args:
        mask3M (numpy.nparray): Binary mask where 1s represent the segmentation
        xValsV (numpy.nparray): x-values i.e. coordinates of columns of input mask
        yValsV (numpy.nparray): y-values i.e. coordinates of rows of input mask
        zValsV (numpy.nparray): z-values i.e. coordinates of slices of input mask

    Returns:
        dict: Dictionary containing shape features

    """

    # Convert grid from cm to mm
    xValsV = xValsV * 10
    yValsV = yValsV * 10
    zValsV = zValsV * 10

    maskForShape3M = mask3M.copy()
    voxel_siz = [abs(yValsV[1] - yValsV[0]), abs(xValsV[1] - xValsV[0]), abs(zValsV[1] - zValsV[0])]
    voxel_volume = np.prod(voxel_siz)

    volume = voxel_volume * np.sum(maskForShape3M)

    # Fill holes
    rmin,rmax,cmin,cmax,smin,smax,_ = computeBoundingBox(maskForShape3M, 0, 1)
    #struct3D = np.ones((3,3,3))
    #maskForShape3M = ndimage.binary_fill_holes(maskForShape3M[rmin:rmax+1,cmin:cmax+1,smin:smax+1])
    maskForShape3M = maskForShape3M[rmin:rmax+1,cmin:cmax+1,smin:smax+1]

    filled_volume = voxel_volume * np.sum(maskForShape3M)

    # Axis Aligned bounding Box (AABB) volume
    volumeAABB = (rmax-rmin+1) * (cmax-cmin+1) * (smax-smin+1) * voxel_volume

    # Get x/y/z coordinates of all the voxels
    indM = np.argwhere(maskForShape3M)

    xV = xValsV[indM[:, 1]]
    yV = yValsV[indM[:, 0]]
    zV = zValsV[indM[:, 2]]
    xyzM = np.column_stack((xV, yV, zV))
    meanV = np.mean(xyzM, axis=0)
    xyzM = (xyzM - meanV) / np.sqrt(xyzM.shape[0])
    eig_valV = eig(np.dot(xyzM.T, xyzM))
    shapeS = {}
    shapeS['majorAxis'] = 4 * np.sqrt(eig_valV[2])
    shapeS['minorAxis'] = 4 * np.sqrt(eig_valV[1])
    shapeS['leastAxis'] = 4 * np.sqrt(eig_valV[0])
    shapeS['flatness'] = np.sqrt(eig_valV[0] / eig_valV[2])
    shapeS['elongation'] = np.sqrt(eig_valV[1] / eig_valV[2])

    # Get the surface points for the structure mask
    rowV, colV, slcV = getSurfacePoints(maskForShape3M)

    # Downsample surface points
    # sample_rate = 1
    # dx = abs(np.median(np.diff(xValsV)))
    # dz = abs(np.median(np.diff(zValsV)))
    # while surf_points.shape[0] > 50000:
    #     sample_rate += 1
    #     if dz / dx < 2:
    #         surf_points = getSurfacePoints(maskForShape3M, sample_rate, sample_rate)
    #     else:
    #         surf_points = getSurfacePoints(maskForShape3M, sample_rate, 1)

    xSurfV = xValsV[colV]
    ySurfV = yValsV[rowV]
    zSurfV = zValsV[slcV]
    #distM = sepsq(np.column_stack((xSurfV, ySurfV, zSurfV)), np.column_stack((xSurfV, ySurfV, zSurfV)))
    ptsM = np.column_stack((xSurfV, ySurfV, zSurfV))
    #distM = distance.cdist(ptsM, ptsM, 'euclidean')
    #shapeS['max3dDiameter'] = np.max(distM)

    dmaxAxial = 0
    dmaxCor = 0
    dmaxSag = 0

    uniqRowV = np.unique(rowV)
    uniqColV = np.unique(colV)
    uniqSlcV = np.unique(slcV)

    # Max diameter along slices
    for i in range(len(uniqSlcV)):
        slc = uniqSlcV[i]
        indV = slcV == slc
        distM = distance.cdist(ptsM[indV,:], ptsM[indV,:], 'euclidean')
        dmaxAxial = max(dmaxAxial, np.max(distM))

    # Max diameter along cols
    for i in range(len(uniqColV)):
        col = uniqColV[i]
        indV = colV == col
        distM = distance.cdist(ptsM[indV,:], ptsM[indV,:], 'euclidean')
        dmaxSag = max(dmaxSag, np.max(distM))

    # Max diameter along rows
    for i in range(len(uniqRowV)):
        row = uniqRowV[i]
        indV = rowV == row
        distM = distance.cdist(ptsM[indV,:], ptsM[indV,:], 'euclidean')
        dmaxCor = max(dmaxCor, np.max(distM))

    shapeS['max2dDiameterAxialPlane'] = dmaxAxial
    shapeS['max2dDiameterSagittalPlane'] = dmaxSag
    shapeS['max2dDiameterCoronalPlane'] = dmaxCor

    # Surface Area
    # Pad mask to account for contribution from edge slices
    maskForShape3M = np.pad(maskForShape3M, ((1,1),(1,1),(1,1)),
                            mode='constant', constant_values=((0, 0),))
    verts, faces, normals, values = measure.marching_cubes(maskForShape3M, level=0.5, spacing=voxel_siz)
    shapeS['surfArea'] = trimeshSurfaceArea(verts,faces)

    #distSurfM = distance.cdist(verts, verts, 'euclidean')

    shapeS['max3dDiameter'] = calcMaxDistBetweenPts(verts, 'euclidean')

    shapeS['volume'] = volume
    shapeS['filledVolume'] = filled_volume

    shapeS['volumeDensityAABB'] = volume / volumeAABB

    # Compactness 1 (V/(pi*A^(3/2))
    shapeS['Compactness1'] = shapeS['volume'] / (np.pi**0.5 * shapeS['surfArea']**1.5)

    # Compactness 2 (36*pi*V^2/A^3)
    shapeS['Compactness2'] = 36 * np.pi * shapeS['volume']**2 / shapeS['surfArea']**3

    # Spherical disproportion (A/(4*pi*R^2)
    R = (shapeS['volume']*3/4/np.pi)**(1/3)
    shapeS['spherDisprop'] = shapeS['surfArea'] / (4*np.pi*R**2)

    # Sphericity
    shapeS['sphericity'] = np.pi**(1/3) * (6*shapeS['volume'])**(2/3) / shapeS['surfArea']

    # Surface to volume ratio
    shapeS['surfToVolRatio'] = shapeS['surfArea'] / shapeS['volume']

    return shapeS
