import numpy as np
from cerr import plan_container as pc
from cerr.utils import uid
import cerr.contour.rasterseg as rs
from shapelysmooth import chaikin_smooth
from shapelysmooth import catmull_rom_smooth
from shapelysmooth import taubin_smooth
import copy

'''
smooth_structure(planC, struct_num, replace_original = True, name_suffix = "", tol = 4, taubin_mu = 0.8, taubin_factor = 0.8, catmull_alpha = 1)
Description: Get structure object from planC/container. Apply 2D smoothing. Option: replace original structure (default=True), or insert new structure
Returns: updated planC
'''

def smoothStructure(planC, struct_idx, replace_original = True, name_suffix ="", tol = 4, taubin_mu = 0.8, taubin_factor = 0.8, catmull_alpha = 1):
    """Apply 2-D contour smoothing to every segment of a structure in planC.

    Deep-copies the structure at ``struct_idx``, applies piecewise smoothing to
    each 2-D contour segment via :func:`smooth2DContour`, regenerates raster
    segments, and either replaces the original structure in ``planC`` or appends
    a new one depending on ``replace_original``.

    Args:
        planC (cerr.plan_container.PlanC): pyCERR plan container object.
        struct_idx (int): Index of the structure to smooth in
            ``planC.structure``.
        replace_original (bool, optional): When ``True`` (default) the
            smoothed structure overwrites the original entry.  When ``False``
            the smoothed structure is appended as a new entry.
        name_suffix (str, optional): String appended to the structure name of
            the smoothed copy.  Defaults to ``""``.
        tol (int, optional): Minimum gap (in vertices) between jagged regions
            for them to be treated as separate segments.  Defaults to ``4``.
        taubin_mu (float, optional): Mu parameter for Taubin smoothing of
            jagged regions.  Defaults to ``0.8``.
        taubin_factor (float, optional): Factor (lambda) for Taubin smoothing.
            Defaults to ``0.8``.
        catmull_alpha (float, optional): Alpha parameter for Catmull-Rom
            interpolation of smooth regions.  Defaults to ``1``.

    Returns:
        cerr.plan_container.PlanC: The updated plan container with the smoothed
        structure in place (or appended).
    """
    struct_obj = copy.deepcopy(planC.structure[struct_idx])
    for contour_orig in struct_obj.contour:
        if contour_orig != []:
            for seg in contour_orig.segments:
                C = seg.points
                z_coord = C[0][2]
                X, tr = smooth2DContour(C, tol, taubin_mu, taubin_factor, catmull_alpha)
                if X.shape[1] == 2:
                    Z = z_coord * np.ones((X.shape[0],1))
                    seg.points = np.hstack((X,Z))
    struct_obj.strUID = uid.createUID("structure")
    struct_obj.rasterSegments = rs.generateRastersegs(struct_obj, planC)
    struct_obj.structureName = struct_obj.structureName + name_suffix
    if not replace_original:
        planC.structure.append(struct_obj)
    else:
        planC.structure[struct_idx] = struct_obj
    return planC


'''
smooth_2D_contour(C, tol = 4, taubin_mu = 0.8, taubin_factor = 0.8, catmull_alpha = 1)

Description: Function will piecewise-smooth a closed contour C.
Returns: Smoothed Contour X, range indices of jagged regions in original contour, taubin_range
'''

def smooth2DContour(C, tol = 4, taubin_mu = 0.8, taubin_factor = 0.8, catmull_alpha = 1):
    """Piecewise-smooth a closed 2-D contour in place.

    Identifies "jagged" vertices — those where the cosine of the interior
    angle (``dot(v1, v2) / (|v1| |v2|)``) is either very small (near-perpendicular
    turns) or very close to 1 (near-collinear, i.e. micro-steps) — groups them
    into contiguous jagged regions, and applies Taubin smoothing to each jagged
    region while using Catmull-Rom interpolation on the smooth regions in
    between.  If no jagged vertices are detected the original 2-D coordinates
    are returned unchanged.

    Args:
        C (np.ndarray): Contour vertex array of shape ``(N, 3)`` or ``(N, 2)``.
            When 3 columns are present the third (z) column is stripped before
            processing and not included in the output.
        tol (int, optional): Maximum vertex-index gap allowed within a single
            jagged region.  Adjacent jagged vertices separated by more than
            ``tol`` indices are split into separate regions.  Defaults to ``4``.
        taubin_mu (float, optional): Mu parameter for Taubin smoothing.
            Defaults to ``0.8``.
        taubin_factor (float, optional): Factor (lambda) parameter for Taubin
            smoothing.  Defaults to ``0.8``.
        catmull_alpha (float, optional): Alpha parameter for Catmull-Rom
            interpolation.  Defaults to ``1``.

    Returns:
        tuple:
            - **piecewise_segs** (np.ndarray): Smoothed contour vertices as an
              array of shape ``(M, 2)`` containing only the x/y coordinates.
              Returns the original ``(N, 2)`` array unchanged when no jagged
              segments are found.
            - **taubin_range** (list[list[int]] | list): List of ``[start, end]``
              vertex-index pairs identifying each jagged region that was
              Taubin-smoothed.  Empty list when no jagged segments were found.
    """
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
        return Cxy, []

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
