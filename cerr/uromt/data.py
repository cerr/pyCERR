"""urOMT- load data & mask.

Port of "Part 1" of the MATLAB ``driver_RatBrain.m``: build the ROI mask and
the preprocessed longitudinal frame stack. Here the frames are the
co-registered DCE-MRI scans held in ``planC.scan`` (one per time point) and the
ROI is a ``planC.structure``; everything else mirrors the MATLAB steps
(binarize + fill the mask, optional resize, per-frame smoothing, masking).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from cerr.contour import rasterseg as rs
from cerr.utils.mask import fillHoles, computeBoundingBox
from cerr.utils.image_proc import affineDiffusion3d
from cerr.mri_metrics.dce_mri import normalizeToBaseline, getScanOrder, buildConcDict

def _frameArray(planC, scanNum):
    return np.asarray(planC.scan[scanNum].getScanArray(), dtype=np.float64)


def scanTimeLabel(planC, scanNum):
    """Short acquisition-time string for a scan (for timepoint labels)."""
    si = planC.scan[scanNum].scanInfo[0]
    return (getattr(si, "acquisitionTime", "") or
            getattr(si, "seriesTime", "") or "")


def externalBaselineCount(normMethod, baselineFrames, basePts, firstSelPos):
    """How many leading scans to use as an EXTERNAL (non-consumed) pre-contrast
    baseline for concentration/RSE normalization.

    Returns ``>0`` when the whole selected window should be transported (the
    baseline is the mean of the first N temporally-ordered scans); ``0`` when the
    leading ``basePts`` selected frames are consumed in-sequence as the baseline.

    An external baseline is used when ``baselineFrames`` is set explicitly, or
    when the selected window starts at/after the first ``basePts`` frames (so the
    pre-contrast frames lie before the window) - this lets e.g. first=20:2:22
    transport both selected scans instead of eating the first as a baseline.
    Only matters for ``CC``/``RSE`` normalization."""
    if str(normMethod).upper() not in ("CC", "RSE"):
        return 0
    if int(baselineFrames) > 0:
        return int(baselineFrames)
    if int(basePts) > 0 and int(firstSelPos) >= int(basePts):
        return int(basePts)
    return 0


def prepareData(cfg, planC):
    """Part 1: build ``cfg.mask`` and ``cfg.vol`` (preprocessed frame stack).

    The selected scans (``cfg.scanNumV`` filtered by the ``time`` setting) must
    be co-registered onto a common grid - they are the longitudinal time
    points. When ``cfg.convertToConc`` is set, the DCE signal frames are first
    converted to contrast-agent concentration maps (the first ``basePts`` frames
    define the pre-contrast baseline). The ROI structure (``cfg.structNum``)
    defines the working domain; frames are cropped to its bounding box,
    smoothed, optionally resized, and masked, exactly as the MATLAB Part 1 loop
    fills ``cfg.vol(j).data``.

    Args:
        cfg (UROMTConfig): configuration (model settings + scan/struct refs).
        planC (cerr.plan_container.PlanC): plan container.

    Returns:
        UROMTConfig: the same cfg with ``mask``, ``vol``, ``trueSize``,
        ``spacing`` and ``bbox`` populated.
    """

    # Infer order if the caller didn't supply one
    if not cfg.scanNumV:
        cfg.scanNumV = getScanOrder(planC)

    # Select time-points for analysis
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

    # Extract ROI mask
    if cfg.structNum is not None:
        mask = rs.getStrMask(cfg.structNum, planC)
        if tuple(int(v) for v in mask.shape) != refShape:
            raise ValueError("Structure %d mask shape %s does not match the "
                             "scan grid %s." % (cfg.structNum, mask.shape,
                                                refShape))
        mask = fillHoles(mask.astype(bool)).astype(np.uint8)
        minr, maxr, minc, maxc, mins, maxs, _ = computeBoundingBox(mask > 0)
    else: # use whole scan
        mask = np.ones(refShape, dtype=np.uint8)
        minr, maxr = 0, refShape[0] - 1
        minc, maxc = 0, refShape[1] - 1
        mins, maxs = 0, refShape[2] - 1

    # ---- crop the ROI mask to its bounding box --------------------------
    # Optional bbox padding (pad in-plane bounding box by `bbox_pad` voxels each side,
    # optionally use the full z).
    pad = int(getattr(cfg, "bbox_pad", 0))
    if pad > 0:
        minr, maxr = max(0, minr - pad), min(refShape[0] - 1, maxr + pad)
        minc, maxc = max(0, minc - pad), min(refShape[1] - 1, maxc + pad)
    if int(getattr(cfg, "bbox_full_z", 0)):
        mins, maxs = 0, refShape[2] - 1

    rs_, re_ = int(minr), int(maxr) + 1
    cs_, ce_ = int(minc), int(maxc) + 1
    ss_, se_ = int(mins), int(maxs) + 1
    cfg.bbox = (rs_, re_, cs_, ce_, ss_, se_)

    croppedMask = mask[rs_:re_, cs_:ce_, ss_:se_]      # pre-resize (uint8)
    # Optional resizing
    if int(cfg.do_resize):
        modelMask = (_resize(croppedMask.astype(float), cfg.size_factor) >= 0.5
                     ).astype(np.uint8)
    else:
        modelMask = croppedMask.astype(np.uint8)
    cfg.mask = modelMask
    cfg.trueSize = list(cfg.mask.shape)

    # Get voxel spacing (mm)
    #mask.shape = (rows, cols, slices),so reorder to [dy(row), dx(col), dz(slice)].
    dCol, dRow, dSlc = (10.0*float(v) for v in
                        planC.scan[refScan].getScanSpacing()) #in mm
    cfg.spacing = [dRow, dCol, dSlc]

    # ---- cropped raw frames (crop first -> cheap concentration conversion)
    croppedFrames = [_frameArray(planC, s)[rs_:re_, cs_:ce_, ss_:se_]
                     for s in frameScanNums]

    # Normalize the DCE signal (RSE/CC)
    # First, identify the normalization method.
    # 'CC' -> contrast-agent concentration (SPGR model, needs TR/FA);
    # 'RSE' -> relative signal enhancement S(t)/S(0);
    # 'none' -> raw signal.
    # The first `basePts` frames define the baseline and are consumed by RSE/CC.
    # The default is taken from `normMethod`, or `convertToConc` (1->'CC', 0->'none').
    normMethod = getattr(cfg, "normMethod", None)
    if normMethod is None:
        normMethod = "CC" if int(getattr(cfg, "convertToConc", 1)) else "none"
    normMethod = str(normMethod).upper()

    # Next, determine the pre-contrast baseline source.
    # An EXTERNAL baseline (mean of the first N temporally-ordered scans, NOT part of the
    # transport sequence, so no frames are consumed) is used when either:
    #   (a) ``baselineFrames`` is set explicitly, or
    #   (b) the selected transport window starts AFTER the first ``basePts``
    #       frames - i.e. the pre-contrast frames lie before the window, so the
    #       whole selected window is transported instead of eating its first
    #       frame(s) as a baseline (e.g. first=20:2:22 -> transport scans 20 & 22
    #       with the early frames as S0, rather than failing with 1 uptake frame).
    # For an external baseline the N baseline scans (cropped to the same bbox) are prepended ahead of the transport
    # window and passed as the leading `basePts` entries. Only when the window starts at the very beginning are the leading frames
    # consumed in-sequence.
    # ``normalizeToBaseline`` averages them as S0. Thy are stripped off the returned array for external baselines.
    # For an in-sequence baseline the transport window's own leading frames serve directly as basePts.
    extBaseN = externalBaselineCount(
        normMethod, int(getattr(cfg, "baselineFrames", 0) or 0),
        int(getattr(cfg, "basePts", 1)), sel[0])
    if normMethod in ("CC", "RSE"):
        clip = getattr(cfg, "conc_clip", None)
        concDict = None
        if normMethod == "CC":
            concDict = buildConcDict(planC, refScan, getattr(cfg, "T10"),
                                     getattr(cfg, "r1"), TR=getattr(cfg, "TR", None),
                                     FA=getattr(cfg, "FA", None), clip=clip)

        if extBaseN > 0:
            baseScans = list(cfg.scanNumV)[:extBaseN]
            baseFrames = [_frameArray(planC, s)[rs_:re_, cs_:ce_, ss_:se_]
                         for s in baseScans]
            combinedFrames = baseFrames + croppedFrames   # baseline first
            basePtsToUse = extBaseN
        else:
            combinedFrames = croppedFrames
            basePtsToUse = int(getattr(cfg, "basePts", 1))

        # Normalize data
        scanArr4M = np.stack([np.asarray(f, dtype=np.float64)
                              for f in combinedFrames], axis=3)
        timePtsV = np.arange(scanArr4M.shape[3], dtype=float)
        normUptake4M, _t, _b, basePtsUsed = normalizeToBaseline(
            scanArr4M, croppedMask > 0, timePtsV, basePts=basePtsToUse,
            method=normMethod, concDict=concDict)
        normUptake4M = np.nan_to_num(normUptake4M, nan=0.0)
        if normMethod == "RSE" and clip is not None:
            normUptake4M = np.clip(normUptake4M, float(clip[0]), float(clip[1]))
        croppedFrames = [normUptake4M[:, :, :, j]
                         for j in range(normUptake4M.shape[3])]

        if extBaseN == 0:               # in-sequence baseline: frames consumed
            frameScanNums = frameScanNums[basePtsUsed:]
            cfg.frameScanNums = frameScanNums
            if len(croppedFrames) < 2:
                raise ValueError(
                    "After %s normalization only %d uptake frame(s) remain: the "
                    "selection has %d frame(s) and the first %d are consumed as "
                    "the pre-contrast baseline. Select >= %d frames, start the "
                    "window later (so earlier frames serve as the baseline), or "
                    "set basePts=0 / an external baselineFrames."
                    % (normMethod, len(croppedFrames), len(sel), basePtsUsed,
                       basePtsUsed + 2))
        # else: external baseline - frameScanNums unchanged

    # Optional source-indicator chi (K) from a structure
    if getattr(cfg, "chiStructNum", None) is not None:
        chiMask = rs.getStrMask(cfg.chiStructNum, planC)
        if tuple(int(v) for v in chiMask.shape) != refShape:
            raise ValueError("chi structure %d mask shape %s does not match the "
                             "scan grid %s." % (cfg.chiStructNum, chiMask.shape,
                                                refShape))
        chiArr = (chiMask > 0).astype(np.float64)[rs_:re_, cs_:ce_, ss_:se_]
        if int(cfg.do_resize):
            chiArr = (_resize(chiArr, cfg.size_factor) >= 0.5).astype(np.float64)
        chiArr[cfg.mask == 0] = 0.0
        cfg.chi = chiArr.ravel(order="F")
    else:
        cfg.chi = None

    # Optional smoothing and resizing of the mask (frames already cropped/converted)
    smooth = float(cfg.smooth)
    method = str(getattr(cfg, "smooth_method", "affine")).lower()
    smoothDt = float(getattr(cfg, "smooth_dt", 0.1))
    vol = []
    maskBool = cfg.mask == 0
    for frm0 in croppedFrames:
        frm = np.array(frm0, dtype=np.float64)      # copy (avoid mutating views)
        if smooth > 0:
            if method == "gaussian":
                frm = gaussian_filter(frm, sigma=0.1 * smooth)
            else:               # affine-invariant mean-curvature flow (MATLAB)
                frm = affineDiffusion3d(frm, nSteps=int(round(smooth)),
                                        dt=smoothDt, affFlag=(method != "linear"))
        if int(cfg.do_resize):
            frm = _resize(frm, cfg.size_factor)
        frm[maskBool] = 0.0
        vol.append(frm)

    #  Optional global rho rescaling
    # urOMT's velocity/source are invariant to a global scale of rho (the
    # continuity PDE is linear in rho), but a very small rho (e.g. contrast
    # concentration ~0.04 mmol/L) shrinks the Gauss-Newton Hessian and can
    # over-damp the solver. Rescaling all frames by one constant restores
    # conditioning without changing the recovered velocity. 'auto' -> the ROI
    # mean of the first frame becomes 1; a number -> multiply by it.
    rhoScale = getattr(cfg, "rhoScale", None)
    cfg.rhoScaleFactor = 1.0
    if rhoScale not in (None, "", 0, "none"):
        m = cfg.mask > 0
        if isinstance(rhoScale, str) and rhoScale.lower() == "auto":
            ref = float(np.abs(vol[0][m]).mean()) if m.any() else 0.0
            factor = (1.0 / ref) if ref > 1e-12 else 1.0
        else:
            factor = float(rhoScale)
        vol = [v * factor for v in vol]
        cfg.rhoScaleFactor = factor

    cfg.vol = vol
    return cfg


def _resize(arr, factor):
    factor = float(factor)
    if factor == 1.0:
        return arr
    return zoom(arr, factor, order=1)
