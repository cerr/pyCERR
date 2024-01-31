import numpy as np
from cerr import plan_container as pc
from cerr.utils import uid
import cerr.contour.rasterseg as rs
from shapelysmooth import chaikin_smooth
from shapelysmooth import catmull_rom_smooth
from shapelysmooth import taubin_smooth
import copy

'''
smooth_structure(planC, struct_num, replace_original = True, tol = 4, taubin_mu = 0.8, taubin_factor = 0.8, catmull_alpha = 1)
Description: Get structure object from planC/container. Apply 2D smoothing. Option: replace original structure (default=True), or insert new structure
Returns: updated planC
'''

def smooth_structure(planC, struct_idx, replace_original = True, tol = 4, taubin_mu = 0.8, taubin_factor = 0.8, catmull_alpha = 1):
    if replace_original:
        struct_obj = copy.deepcopy(planC.structure[struct_idx])
    else:
        struct_obj = planC.structure[struct_idx]
    for contour_orig in struct_obj.contour:
        if contour_orig != []:
            for seg in contour_orig.segments:
                C = seg.points
                z_coord = C[0][2]
                print(C.shape)
                X, tr = smooth_2D_contour(C, tol, taubin_mu, taubin_factor,catmull_alpha)
                Z = z_coord * np.ones((X.shape[0],1))
                print(X.shape)
                seg.points = np.hstack((X,Z))
    struct_obj.strUID = uid.createUID("structure")
    struct_obj.rasterSegments = rs.generate_rastersegs(struct_obj,planC)
    struct_obj.structureName = struct_obj.structureName + ' smoothed'
    if not replace_original:
        planC.structure.append(struct_obj)
    return planC


'''
smooth_2D_contour(C, tol = 4, taubin_mu = 0.8, taubin_factor = 0.8, catmull_alpha = 1)

Description: Function will piecewise-smooth a closed contour C.
Returns: Smoothed Contour X, range indices of jagged regions in original contour, taubin_range
'''

def smooth_2D_contour(C, tol = 4, taubin_mu = 0.8, taubin_factor = 0.8, catmull_alpha = 1):
    if C.shape[1] == 3:
        Cxy = np.delete(C,2,1)
    elif C.shape[2] == 2:
        Cxy = C
    N = C.shape[0]
    C_sinT = np.ndarray(shape = (N-1,1))
    C_jagg = []
    for i in range(1,N-1):
        x1 = C[i,:2] - C[i-1,:2]
        x2 = C[i+1,:2] - C[i,:2]
        sinT = np.dot(x1,x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        C_sinT[i-1,0] = sinT
        if sinT < 0.005 or sinT > 0.995:
            C_jagg.append(i)
    range_lists = []
    if C_jagg == []:
        print('No jagged segments')
        return C, []

    idx0 = C_jagg[0]
    idxf = 0

    for i in range(len(C_jagg)-1):
        if np.absolute(C_jagg[i+1] - C_jagg[i]) > tol and np.absolute(C_jagg[i+1] - C_jagg[i]) != 0:
            idxf = C_jagg[i]
            if idxf != idx0:
                range_lists.append([idx0,idxf])
            idx0 = C_jagg[i+1]
    if idxf < idx0:
        idxf = C_jagg[-1]
        range_lists.append([idx0,idxf])
    taubin_range = range_lists

    if taubin_range[0][0] > 0:
        cf = taubin_range[0][0]
        if taubin_range[-1][1] != N:
            c0 = taubin_range[-1][1]
            C1 = C[c0:N-1,0:2]
            C2 = C[0:cf+1,0:2]
            C_tmp = np.concatenate((C1,C2))
        else:
            C_tmp = C[0:cf,0:2]
        geom = list(map(tuple,C_tmp))
        C_catmull = np.asarray(catmull_rom_smooth(geom,alpha=catmull_alpha))
        piecewise_segs = C_catmull
    else:
        piecewise_segs = C[0,0:2]

    for x in range(len(taubin_range)):
        t = taubin_range[x]
        t0 = t[0]
        tf = t[1]
        if tf - t0 > 1:
            geom = list(map(tuple, C[t0:tf+1,0:2]))
            C_taubin = np.asarray(taubin_smooth(geom,mu = taubin_mu, factor = taubin_factor))
            piecewise_segs = np.concatenate((piecewise_segs, C_taubin))
        c0 = tf + 1
        if x + 1 < len(taubin_range):
            t2 = taubin_range[x+1]
            cf = t2[0]
            geom = list(map(tuple, C[c0:cf + 1,0:2]))
            C_catmull = np.asarray(catmull_rom_smooth(geom,alpha=catmull_alpha))
            piecewise_segs = np.concatenate((piecewise_segs, C_catmull))

    return piecewise_segs, taubin_range
