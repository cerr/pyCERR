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
from cerr.mri_metrics.dce_mri import normalizeToBaseline

# Defaults for the DCE signal -> contrast-agent-concentration conversion
# (overridable from the urOMT settings file).
DEFAULT_T10 = 1.0 / 0.6        # pre-contrast longitudinal relaxation time
DEFAULT_R1 = 3.8              # relaxivity


def affineDiffusion3d(arr, nSteps, dt=0.1, affFlag=True):
    """Affine-invariant mean-curvature-flow smoothing.

    Vectorized port of the MATLAB ``affine_diffusion_3d.m`` (Chen, urOMT).
    Interior voxels are updated by central differences; the one-voxel border is
    held fixed (matching the MATLAB ``2:N-1`` loops). Array axes are
    ``(row, col, slice)`` = MATLAB ``(k, j, l)``.

    Args:
        arr (np.ndarray): 3-D image.
        nSteps (int): number of evolution steps (``n_t``).
        dt (float): evolution step size.
        affFlag (bool): True -> affine-invariant flow; False -> linear (heat).

    Returns:
        np.ndarray: smoothed image (float64), nonnegative.
    """
    phi = np.asarray(arr, dtype=np.float64).copy()
    if phi.ndim != 3:
        raise ValueError("affineDiffusion3d takes a 3-D array")
    if min(phi.shape) < 3 or nSteps <= 0:
        return phi
    c_ = (slice(1, -1), slice(1, -1), slice(1, -1))
    for _ in range(int(nSteps)):
        c = phi[c_]
        jp = phi[1:-1, 2:, 1:-1]; jm = phi[1:-1, :-2, 1:-1]   # col +/-
        kp = phi[2:, 1:-1, 1:-1]; km = phi[:-2, 1:-1, 1:-1]   # row +/-
        lp = phi[1:-1, 1:-1, 2:]; lm = phi[1:-1, 1:-1, :-2]   # slice +/-
        pXX = jp - 2 * c + jm
        pYY = kp - 2 * c + km
        pZZ = lp - 2 * c + lm
        if not affFlag:                          # linear (heat) smoothing
            phi[c_] = c + dt * (pXX + pYY + pZZ)
            continue
        pX = 0.5 * (jp - jm)
        pY = 0.5 * (km - kp)                      # note sign (MATLAB k-1 minus k+1)
        pZ = 0.5 * (lm - lp)
        pXY = 0.25 * (-phi[:-2, :-2, 1:-1] + phi[:-2, 2:, 1:-1]
                      + phi[2:, :-2, 1:-1] - phi[2:, 2:, 1:-1])
        pXZ = 0.25 * (phi[1:-1, :-2, 2:] + phi[1:-1, 2:, :-2]
                      - phi[1:-1, :-2, :-2] - phi[1:-1, 2:, 2:])
        pYZ = 0.25 * (phi[:-2, 1:-1, :-2] + phi[2:, 1:-1, 2:]
                      - phi[:-2, 1:-1, 2:] - phi[2:, 1:-1, :-2])
        meanCurvNum = (pX ** 2 * (pYY + pZZ) + pY ** 2 * (pXX + pZZ)
                       + pZ ** 2 * (pXX + pYY)
                       - 2 * (pX * pY * pXY + pX * pZ * pXZ + pY * pZ * pYZ))
        gausCurvNum = (pX ** 2 * (pYY * pZZ - pYZ ** 2)
                       + pY ** 2 * (pXX * pZZ - pXZ ** 2)
                       + pZ ** 2 * (pXX * pYY - pXY ** 2)
                       + 2 * pX * pY * (pXZ * pYZ - pXY * pZZ)
                       + 2 * pY * pZ * (pXY * pXZ - pYZ * pXX)
                       + 2 * pX * pZ * (pXY * pYZ - pXZ * pYY))
        upd = np.sign(meanCurvNum) * np.maximum(0.0, gausCurvNum) ** 0.25
        phi[c_] = np.maximum(0.0, c + dt * upd)
    return phi


def framesToConcentration(frameArrays, mask3M, method="CC", T10=DEFAULT_T10,
                          r1=DEFAULT_R1, TR=None, FA=None, basePts=1, clip=None,
                          timePtsV=None):
    """Normalize a stack of DCE-MRI signal frames to the baseline S(0).

    Wrapper around :func:`cerr.mri_metrics.dce_mri.normalizeToBaseline`. The
    first ``basePts`` frames define the pre-contrast baseline (and are consumed),
    so the returned sequence has ``len(frameArrays) - basePts`` uptake frames.

    Args:
        frameArrays (list[np.ndarray]): 3-D signal frames (row, col, slice).
        mask3M (np.ndarray): ROI mask (row, col, slice) on the same grid.
        method (str): 'CC' -> contrast-agent concentration via the SPGR model
            (needs ``TR``/``FA``); 'RSE' -> relative signal enhancement
            S(t)/S(0) (no SPGR parameters).
        T10, r1, TR, FA: SPGR parameters (``method='CC'`` only). TR in seconds.
        basePts (int): number of pre-contrast baseline frames.
        clip (list[float]): optional [min, max] clip on the output.
        timePtsV (np.ndarray): optional acquisition times; defaults to indices.

    Returns:
        (list[np.ndarray], int): normalized frames and the baseline count used.
    """
    method = str(method).upper()
    scanArr4M = np.stack([np.asarray(f, dtype=np.float64) for f in frameArrays],
                         axis=3)
    nTime = scanArr4M.shape[3]
    if timePtsV is None:
        timePtsV = np.arange(nTime, dtype=float)
    maskBool = np.asarray(mask3M).astype(bool)
    if method == "CC":
        if TR is None or FA is None:
            raise ValueError("framesToConcentration(method='CC') needs TR and "
                             "FA (flip angle).")
        concDict = {"T10": float(T10), "r1": float(r1),
                    "TR": float(TR), "FA": float(FA)}
        if clip is not None:
            concDict["clip_between"] = [float(clip[0]), float(clip[1])]
        normUptake4M, _t, _b, basePtsUsed = normalizeToBaseline(
            scanArr4M, maskBool, timePtsV, basePts=basePts, method="CC",
            concDict=concDict)
    elif method == "RSE":                  # relative signal enhancement S(t)/S0
        normUptake4M, _t, _b, basePtsUsed = normalizeToBaseline(
            scanArr4M, maskBool, timePtsV, basePts=basePts, method="RSE")
        if clip is not None:
            normUptake4M = np.clip(normUptake4M, float(clip[0]), float(clip[1]))
    else:
        raise ValueError("method must be 'CC' or 'RSE', got %r" % method)
    frames = [np.nan_to_num(normUptake4M[:, :, :, j], nan=0.0)
              for j in range(normUptake4M.shape[3])]
    return frames, int(basePtsUsed)


def _ccTRFA(cfg, planC, refScan):
    """Resolve (TR seconds, FA degrees) for the SPGR concentration model from
    the settings (``TR``/``FA``) or the scan metadata. scanInfo stores the DICOM
    RepetitionTime (0018,0080) in milliseconds, so it is converted to seconds; a
    settings ``TR`` is taken as-is (already seconds)."""
    si = planC.scan[refScan].scanInfo[0]
    TR = getattr(cfg, "TR", None)
    if TR is None:
        trMs = getattr(si, "repetitionTime", None)
        TR = (float(trMs) / 1000.0) if trMs is not None else None
    FA = cfg.FA if getattr(cfg, "FA", None) is not None \
        else getattr(si, "flipAngle", None)
    if TR is None or FA is None:
        raise ValueError(
            "urOMT 'CC' concentration needs the repetition time and flip angle;"
            " scan %d has none (set 'TR'/'FA', or use normMethod='RSE')."
            % refScan)
    return float(TR), float(FA)


def scanTimeKey(planC, scanNum):
    """Sortable acquisition-time key for a scan. Scans in a planC may not be
    stored in temporal order (e.g. lexical SeriesInstanceUID order), so urOMT
    needs the true timepoint sequence. Prefers DICOM AcquisitionDate/Time
    (0008,0022 / 0008,0032), then Series date/time, then TriggerTime, finally
    the scan index. Returns a tuple that sorts chronologically."""
    si = planC.scan[scanNum].scanInfo[0]
    # The leading group int keeps heterogeneous key types from being compared
    # against each other (within one dynamic series all scans use the same key).
    tpi = getattr(si, "temporalPositionIndex", None)   # dynamic-phase index
    if tpi not in (None, ""):
        try:
            return (0, float(tpi), scanNum)
        except (TypeError, ValueError):
            pass
    for dateF, timeF in (("acquisitionDate", "acquisitionTime"),
                         ("seriesDate", "seriesTime")):
        t = getattr(si, timeF, None)
        if t not in (None, ""):
            return (1, "%s %s" % (getattr(si, dateF, "") or "", t), scanNum)
    tt = getattr(si, "triggerTime", None)
    if tt not in (None, "", 0):
        return (2, float(tt), scanNum)
    return (3, scanNum, scanNum)        # fallback: preserve original order


def scanTimeLabel(planC, scanNum):
    """Short acquisition-time string for a scan (for timepoint labels)."""
    si = planC.scan[scanNum].scanInfo[0]
    return (getattr(si, "acquisitionTime", "") or
            getattr(si, "seriesTime", "") or "")


def scanTimeOrder(planC, scanNumV=None):
    """Return the given scan indices (or all scans) ordered by acquisition
    time. The basis for both the urOMT timepoint sequence and the GUI
    timepoint->scan-index mapping."""
    nums = (list(range(len(planC.scan))) if scanNumV is None
            else list(scanNumV))
    return sorted(nums, key=lambda s: scanTimeKey(planC, s))


def _frameArray(planC, scanNum):
    return np.asarray(planC.scan[scanNum].getScanArray(), dtype=np.float64)


def _voxelSpacingMm(planC, scanNum):
    """Voxel spacing in MILLIMETRES, ordered to match the scan-array axes
    (row, col, slice). urOMT runs its entire calculation in mm (so sigma,
    velocities, Peclet and flux are on the same physical scale as the reference
    MATLAB urOMT, whose grid is in mm).

    ``getScanSpacing()`` returns [dx(col), dy(row), dz(slice)] in **cm**; urOMT
    pairs ``spacing`` element-wise with ``trueSize`` = ``mask.shape`` =
    (rows, cols, slices), so we reorder to [dy(row), dx(col), dz(slice)] and
    convert cm -> mm (x10). The spacing is always read from planC - there is no
    user override."""
    dCol, dRow, dSlc = (float(v) for v in
                        planC.scan[scanNum].getScanSpacing())   # cm, [x,y,z]
    return [dRow * 10.0, dCol * 10.0, dSlc * 10.0]              # mm, [row,col,slc]


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

    # optional bbox padding (matches the MATLAB getRange: pad the in-plane
    # bounding box by `bbox_pad` voxels each side, optionally use the full z).
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

    # ---- crop the ROI mask to its bounding box --------------------------
    croppedMask = mask[rs_:re_, cs_:ce_, ss_:se_]      # pre-resize (uint8)
    # optional resize of the mask (MATLAB: resizeMatrix + threshold back to 0/1)
    if int(cfg.do_resize):
        modelMask = (_resize(croppedMask.astype(float), cfg.size_factor) >= 0.5
                     ).astype(np.uint8)
    else:
        modelMask = croppedMask.astype(np.uint8)
    cfg.mask = modelMask
    cfg.trueSize = list(cfg.mask.shape)

    # ---- voxel spacing (mm) - always from planC, no user override --------
    cfg.spacing = _voxelSpacingMm(planC, refScan)

    # ---- cropped raw frames (crop first -> cheap concentration conversion)
    croppedFrames = [_frameArray(planC, s)[rs_:re_, cs_:ce_, ss_:se_]
                     for s in frameScanNums]

    # ---- STEP 1: normalize the DCE signal --------------------------------
    # Done first (within the ROI bbox, before smoothing/resize). 'CC' ->
    # contrast-agent concentration (SPGR model, needs TR/FA); 'RSE' -> relative
    # signal enhancement S(t)/S(0); 'none' -> raw signal. The first `basePts`
    # frames define the baseline and are consumed by RSE/CC. The default is
    # taken from `normMethod`, or `convertToConc` (1->'CC', 0->'none').
    normMethod = getattr(cfg, "normMethod", None)
    if normMethod is None:
        normMethod = "CC" if int(getattr(cfg, "convertToConc", 1)) else "none"
    normMethod = str(normMethod).upper()

    # Pre-contrast baseline source. An EXTERNAL baseline (mean of the first N
    # temporally-ordered scans, NOT part of the transport sequence, so no frames
    # are consumed) is used when either:
    #   (a) ``baselineFrames`` is set explicitly, or
    #   (b) the selected transport window starts AFTER the first ``basePts``
    #       frames - i.e. the pre-contrast frames lie before the window, so the
    #       whole selected window is transported instead of eating its first
    #       frame(s) as a baseline (e.g. first=20:2:22 -> transport scans 20 & 22
    #       with the early frames as S0, rather than failing with 1 uptake frame).
    # Only when the window starts at the very beginning are the leading frames
    # consumed in-sequence. This matches the MATLAB workflow ('RSE' keeps
    # S(t)/S0; 'CC' applies the SPGR model intToConc to S(t)/S0).
    extBaseN = externalBaselineCount(
        normMethod, int(getattr(cfg, "baselineFrames", 0) or 0),
        int(getattr(cfg, "basePts", 1)), sel[0])
    if normMethod in ("CC", "RSE") and extBaseN > 0:
        baseScans = list(cfg.scanNumV)[:extBaseN]
        S0 = np.mean([_frameArray(planC, s)[rs_:re_, cs_:ce_, ss_:se_]
                      for s in baseScans], axis=0)
        S0 = np.where(np.abs(S0) < 1e-8, 1e-8, S0)
        clip = getattr(cfg, "conc_clip", None)
        concDict = None
        if normMethod == "CC":
            TR, FA = _ccTRFA(cfg, planC, refScan)
            concDict = {"T10": float(getattr(cfg, "T10", DEFAULT_T10)),
                        "r1": float(getattr(cfg, "r1", DEFAULT_R1)),
                        "TR": TR, "FA": FA}
            if clip is not None:
                concDict["clip_between"] = [float(clip[0]), float(clip[1])]
        norm = []
        for f in croppedFrames:
            ns = np.nan_to_num(np.asarray(f, dtype=np.float64) / S0, nan=0.0)
            if normMethod == "CC":
                from cerr.mri_metrics.dce_mri import intToConc
                g = np.nan_to_num(intToConc(ns, concDict), nan=0.0)
            else:                       # RSE
                g = (np.clip(ns, float(clip[0]), float(clip[1]))
                     if clip is not None else ns)
            norm.append(g)
        croppedFrames = norm            # frameScanNums unchanged (none consumed)
    elif normMethod in ("CC", "RSE"):
        kw = dict(method=normMethod, basePts=int(getattr(cfg, "basePts", 1)),
                  clip=getattr(cfg, "conc_clip", None))
        if normMethod == "CC":
            TR, FA = _ccTRFA(cfg, planC, refScan)
            kw.update(T10=float(getattr(cfg, "T10", DEFAULT_T10)),
                      r1=float(getattr(cfg, "r1", DEFAULT_R1)), TR=TR, FA=FA)
        croppedFrames, basePtsUsed = framesToConcentration(
            croppedFrames, croppedMask > 0, **kw)
        frameScanNums = frameScanNums[basePtsUsed:]
        cfg.frameScanNums = frameScanNums
        if len(croppedFrames) < 2:
            raise ValueError(
                "After %s normalization only %d uptake frame(s) remain: the "
                "selection has %d frame(s) and the first %d are consumed as the "
                "pre-contrast baseline. Select >= %d frames, start the window "
                "later (so earlier frames serve as the baseline), or set "
                "basePts=0 / an external baselineFrames."
                % (normMethod, len(croppedFrames), len(sel), basePtsUsed,
                   basePtsUsed + 2))

    # ---- optional source-indicator chi (MATLAB K) from a structure -------
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

    # ---- per-frame: smooth, resize, mask (frames already cropped/converted)
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

    # ---- optional global rho rescaling -----------------------------------
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
