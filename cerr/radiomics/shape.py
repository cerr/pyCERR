import numpy as np
from scipy import ndimage
from cerr.utils.mask import getSurfacePoints
from scipy.spatial import distance
from skimage import measure
from cerr.utils import bbox

def trimeshSurfaceArea(v,f):

    v1 = v[f[:, 1], :] - v[f[:, 0], :]
    v2 = v[f[:, 2], :] - v[f[:, 0], :]

    # Calculate the cross product and its norm
    cross_product = np.cross(v1, v2)
    cross_product_norm = np.linalg.norm(cross_product, axis=1)

    # Calculate the area of each triangle
    area = np.sum(cross_product_norm) / 2

    return area

def vector_norm3d(v):
    return np.linalg.norm(v, axis=1)

def eig(a):
    return np.sort(np.linalg.eig(a)[0])

def sepsq(a, b):
    return np.sum((a - b)**2, axis=0)

def compute_shape_features(mask3M, xValsV, yValsV, zValsV):

    maskForShape3M = mask3M.copy()
    voxel_siz = [abs(yValsV[1] - yValsV[0]), abs(xValsV[1] - xValsV[0]), abs(zValsV[1] - zValsV[0])]
    voxel_volume = np.prod(voxel_siz)

    volume = voxel_volume * np.sum(maskForShape3M)

    # Fill holes
    rmin,rmax,cmin,cmax,smin,smax,_ = bbox.compute_boundingbox(maskForShape3M,1)
    _ = ndimage.binary_fill_holes(maskForShape3M[rmin:rmax,cmin:cmax,smin:smax],
                                  output=maskForShape3M[rmin:rmax,cmin:cmax,smin:smax])

    filled_volume = voxel_volume * np.sum(maskForShape3M)

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
    surf_points = getSurfacePoints(maskForShape3M)
    sample_rate = 1
    dx = abs(np.median(np.diff(xValsV)))
    dz = abs(np.median(np.diff(zValsV)))
    while surf_points.shape[0] > 20000:
        sample_rate += 1
        if dz / dx < 2:
            surf_points = getSurfacePoints(maskForShape3M, sample_rate, sample_rate)
        else:
            surf_points = getSurfacePoints(maskForShape3M, sample_rate, 1)

    xSurfV = xValsV[surf_points[:, 1]]
    ySurfV = yValsV[surf_points[:, 0]]
    zSurfV = zValsV[surf_points[:, 2]]
    #distM = sepsq(np.column_stack((xSurfV, ySurfV, zSurfV)), np.column_stack((xSurfV, ySurfV, zSurfV)))
    ptsM = np.column_stack((xSurfV, ySurfV, zSurfV))
    distM = distance.cdist(ptsM, ptsM, 'euclidean')
    shapeS['max3dDiameter'] = np.max(distM)

    rowV = np.unique(surf_points[:, 0])
    colV = np.unique(surf_points[:, 1])
    slcV = np.unique(surf_points[:, 2])

    # Max diameter along slices
    dmax = 0
    for i in range(len(slcV)):
        slc = slcV[i]
        indV = surf_points[:, 2] == slc
        distSlcM = distM[indV][:, indV]
        dmax = max(dmax, np.max(distSlcM))
    shapeS['max2dDiameterAxialPlane'] = dmax

    # Max diameter along cols
    dmax = 0
    for i in range(len(colV)):
        col = colV[i]
        indV = surf_points[:, 1] == col
        distColM = distM[indV][:, indV]
        dmax = max(dmax, np.max(distColM))
    shapeS['max2dDiameterSagittalPlane'] = dmax

    # Max diameter along rows
    dmax = 0
    for i in range(len(rowV)):
        row = rowV[i]
        indV = surf_points[:, 0] == row
        distRowM = distM[indV][:, indV]
        dmax = max(dmax, np.max(distRowM))
    shapeS['max2dDiameterCoronalPlane'] = dmax

    # Surface Area
    # Pad mask to account for contribution from edge slices
    maskForShape3M = np.pad(maskForShape3M, ((1,1),(1,1),(1,1)),
                            mode='constant', constant_values=((0, 0),))
    xPre = 2 * xValsV[0] - xValsV[1]
    yPre = 2 * yValsV[0] - yValsV[1]
    zPre = 2 * zValsV[0] - zValsV[1]
    xPost = 2 * xValsV[-1] - xValsV[-2]
    yPost = 2 * yValsV[-1] - yValsV[-2]
    zPost = 2 * zValsV[-1] - zValsV[-2]
    xValsPadV = np.pad(xValsV,(1,1),mode='constant',constant_values=(xPre, xPost))
    yValsPadV = np.pad(yValsV,(1,1),mode='constant',constant_values=(yPre, yPost))
    zValsPadV = np.pad(zValsV,(1,1),mode='constant',constant_values=(zPre, zPost))
    verts, faces, normals, values = measure.marching_cubes(maskForShape3M, level=0.5)
    # scale vcertices to physical cooordinates
    xRange = xValsPadV[-1] - xValsPadV[0]
    yRange = yValsPadV[-1] - yValsPadV[0]
    zRange = zValsPadV[-1] - zValsPadV[0]
    ranges = [yRange, xRange, zRange]
    mins = [yValsPadV[0], xValsPadV[0], zValsPadV[0]]
    verts_phys = verts * ranges / np.array(maskForShape3M.shape) + mins
    shapeS['surfArea'] = trimeshSurfaceArea(verts_phys,faces)

    shapeS['volume'] = volume
    shapeS['filledVolume'] = filled_volume

    # Compactness 1 (V/(pi*A^(3/2))
    shapeS['Compactness1'] = shapeS['volume'] / (np.pi**0.5 * shapeS['surfArea']**1.5)

    # Compactness 2 (36*pi*V^2/A^3)
    shapeS['Compactness2'] = 36 * np.pi * shapeS['volume']**2 / shapeS['surfArea']**3

    # max3dDiameter , eq. (17) Aerts Nature suppl.

    # Spherical disproportion (A/(4*pi*R^2)
    R = (shapeS['volume']*3/4/np.pi)**(1/3)
    shapeS['spherDisprop'] = shapeS['surfArea'] / (4*np.pi*R**2)

    # Sphericity
    shapeS['sphericity'] = np.pi**(1/3) * (6*shapeS['volume'])**(2/3) / shapeS['surfArea']

    # Surface to volume ratio
    shapeS['surfToVolRatio'] = shapeS['surfArea'] / shapeS['volume']

    return shapeS
