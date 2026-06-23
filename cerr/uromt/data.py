"""urOMT Part 1 - load data & mask.

Port of "Part 1" of the MATLAB ``driver_RatBrain.m``: build the ROI mask and
the preprocessed longitudinal frame stack. Here the frames are the
co-registered DCE-MRI scans held in ``planC.scan`` (one per time point) and the
ROI is a ``planC.structure``; everything else mirrors the MATLAB steps
(binarize + fill the mask, optional resize, per-frame smoothing, masking).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from cerr.contour import rasterseg as rs
from cerr.dataclasses import scan as scn
from cerr.utils.mask import fillHoles, computeBoundingBox


def _frameArray(planC, scanNum):
    return np.asarray(planC.scan[scanNum].getScanArray(), dtype=np.float64)


def _voxelSpacingCm(planC, scanNum):
    xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
    dx = abs(float(xV[1] - xV[0])) if len(xV) > 1 else 1.0
    dy = abs(float(yV[1] - yV[0])) if len(yV) > 1 else 1.0
    dz = abs(float(zV[1] - zV[0])) if len(zV) > 1 else 1.0
    return [dx, dy, dz]


def prepareData(cfg, planC):
    """Part 1: build ``cfg.mask`` and ``cfg.vol`` (preprocessed frame stack).

    The selected scans (``cfg.scanNumV`` filtered by the ``time`` setting) must
    be co-registered onto a common grid - they are the longitudinal time
    points. The ROI structure (``cfg.structNum``) defines the working domain;
    frames are cropped to its bounding box, smoothed, optionally resized, and
    masked, exactly as the MATLAB Part 1 loop fills ``cfg.vol(j).data``.

    Args:
        cfg (UROMTConfig): configuration (model settings + scan/struct refs).
        planC (cerr.plan_container.PlanC): plan container.

    Returns:
        UROMTConfig: the same cfg with ``mask``, ``vol``, ``trueSize``,
        ``spacing`` and ``bbox`` populated.
    """
    # ---- select the time-point scans -------------------------------------
    sel = cfg.selectedTimeIndices(len(cfg.scanNumV))
    frameScanNums = [cfg.scanNumV[i] for i in sel]
    if len(frameScanNums) < 2:
        raise ValueError("urOMT needs at least 2 time-point scans; got %d "
                         "after applying the 'time' selection."
                         % len(frameScanNums))
    cfg.frameScanNums = frameScanNums

    refScan = frameScanNums[0]
    refShape = tuple(int(v) for v in planC.scan[refScan].getScanArray().shape)
    for s in frameScanNums[1:]:
        if tuple(int(v) for v in planC.scan[s].getScanArray().shape) != refShape:
            raise ValueError("All time-point scans must share one grid "
                             "(co-register them first). Scan %d differs from "
                             "the reference scan %d." % (s, refScan))

    # ---- ROI mask from the structure (MATLAB: load_rhon + bwmorph3 fill)---
    if cfg.structNum is not None:
        roi = rs.getStrMask(cfg.structNum, planC)
        if tuple(int(v) for v in roi.shape) != refShape:
            raise ValueError("Structure %d mask shape %s does not match the "
                             "scan grid %s." % (cfg.structNum, roi.shape,
                                                refShape))
        mask = np.zeros(refShape, dtype=np.uint8)
        mask[roi > 0] = 1
        mask = fillHoles(mask.astype(bool)).astype(np.uint8)
        minr, maxr, minc, maxc, mins, maxs, _ = computeBoundingBox(mask > 0)
    else:                                   # whole scan
        mask = np.ones(refShape, dtype=np.uint8)
        minr, maxr = 0, refShape[0] - 1
        minc, maxc = 0, refShape[1] - 1
        mins, maxs = 0, refShape[2] - 1

    rs_, re_ = int(minr), int(maxr) + 1
    cs_, ce_ = int(minc), int(maxc) + 1
    ss_, se_ = int(mins), int(maxs) + 1
    cfg.bbox = (rs_, re_, cs_, ce_, ss_, se_)

    mask = mask[rs_:re_, cs_:ce_, ss_:se_]

    # optional resize of the mask (MATLAB: resizeMatrix + threshold back to 0/1)
    if int(cfg.do_resize):
        mask = (_resize(mask.astype(float), cfg.size_factor) >= 0.5
                ).astype(np.uint8)
    cfg.mask = mask.astype(np.uint8)
    cfg.trueSize = list(cfg.mask.shape)

    # ---- voxel spacing (cm) from planC unless overridden -----------------
    if cfg.spacing is None:
        cfg.spacing = _voxelSpacingCm(planC, refScan)

    # ---- per-frame: crop, smooth, resize, mask (MATLAB cfg.vol(j).data) --
    smooth = float(cfg.smooth)
    vol = []
    maskBool = cfg.mask == 0
    for s in frameScanNums:
        frm = _frameArray(planC, s)[rs_:re_, cs_:ce_, ss_:se_]
        if smooth > 0:
            # affine_diffusion_3d analog: light edge-preserving-ish smoothing.
            # A Gaussian is used as a first pass; iterate/replace as needed.
            frm = gaussian_filter(frm, sigma=0.1 * smooth)
        if int(cfg.do_resize):
            frm = _resize(frm, cfg.size_factor)
        frm[maskBool] = 0.0
        vol.append(frm)
    cfg.vol = vol
    return cfg


def _resize(arr, factor):
    factor = float(factor)
    if factor == 1.0:
        return arr
    return zoom(arr, factor, order=1)
