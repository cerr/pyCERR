"""urOMT- load data & mask.

Port of "Part 1" of the MATLAB ``driver_RatBrain.m``: build the ROI mask and
the preprocessed longitudinal frame stack. Here the frames are the
co-registered DCE-MRI scans held in ``planC.scan`` (one per time point) and the
ROI is a ``planC.structure``; everything else mirrors the MATLAB steps
(binarize + fill the mask, optional resize, per-frame smoothing, masking).
"""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, zoom

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


def smoothSize(m, cap, radices=(2, 3, 5)):
    """Smallest length >= ``m`` (and <= ``cap``) whose only prime factors are
    ``radices``. Returns ``m`` unchanged when no such length fits under the cap.
    """
    m, cap = int(m), int(cap)
    if m >= cap:
        return cap
    for v in range(m, cap + 1):
        t = v
        for p in radices:
            while t % p == 0:
                t //= p
        if t == 1:
            return v
    return m


def fftFriendlyRange(lo, hi, extent, radices=(2, 3, 5)):
    """Grow the half-open voxel range ``[lo, hi)`` to an FFT-friendly length,
    staying inside ``[0, extent)`` and keeping the original range covered.

    The urOMT diffusion solve inverts ``B = I + dt*sigma*Grad'Grad`` with a 3-D
    DCT, whose cost is governed by the *prime factorization* of each grid
    dimension rather than by the voxel count: a 61x59x46 box (61 and 59 prime)
    costs ~15.7 ms per solve, while 64x60x48 - 11% more voxels - costs ~5.9 ms.
    Since the diffusion solve runs twice per sub-step in every CG matvec, this
    is a ~2.7x on the dominant term for a few percent more voxels.

    Growing (never shrinking) the box only moves the Neumann boundary further
    from the structure and brings in real neighbouring image data, so it does
    not clip the ROI; it does change results slightly. Disable with the
    ``fft_pad`` setting.
    """
    lo, hi, extent = int(lo), int(hi), int(extent)
    want = smoothSize(hi - lo, extent, radices)
    grow = want - (hi - lo)
    if grow <= 0:
        return lo, hi
    lo2 = max(0, lo - grow // 2)
    hi2 = min(extent, lo2 + want)
    lo2 = max(0, hi2 - want)          # ran into the top: take the rest below
    return lo2, hi2


def _bboxPad(pad):
    """Normalize the ``bbox_pad`` setting to ``(row, col, slice)`` voxels.

    A scalar keeps the historical meaning - pad the two **in-plane** axes and
    leave z alone. A 3-element sequence pads each axis independently, which is
    what reproducing the reference ``getRange`` needs (it pads all three).
    """
    if pad is None:
        return 0, 0, 0
    if np.isscalar(pad):
        p = int(pad)
        return p, p, 0
    vals = [int(v) for v in pad]
    if len(vals) != 3:
        raise ValueError("bbox_pad must be a scalar (in-plane) or a 3-element "
                         "[row, col, slice]; got %r" % (pad,))
    return tuple(vals)


def dilateMask(mask, dilate):
    """Grow a binary ROI mask by ``dilate`` voxels (the MATLAB ``cfg.dilate``).

    Uses the reference's ellipsoidal structuring element - the voxels of
    ``-d:d`` in each axis satisfying ``(x/d)^2 + (y/d)^2 + (z/d)^2 <= 1``, i.e.
    a ball of radius ``d`` in *index* space (not physical space, so it is
    anisotropic on anisotropic voxels - matching the reference).

    ``dilate <= 0`` returns the mask unchanged. The urOMT solve always runs on
    the whole ROI box; the mask decides which voxels the Eulerian/Lagrangian
    maps report, so dilating it widens that reporting support. It is required
    to reproduce the reference maps voxel-for-voxel.
    """
    d = int(dilate)
    if d <= 0:
        return mask
    from scipy.ndimage import binary_dilation
    ax = np.arange(-d, d + 1)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    strel = (X / d) ** 2 + (Y / d) ** 2 + (Z / d) ** 2 <= 1.0
    return binary_dilation(np.asarray(mask) > 0,
                           structure=strel).astype(np.uint8)


def toftsPostProcess(im, cfg):
    """Reference ``tofts`` post-steps applied after the signal->concentration
    conversion, in the reference implementation's order.

    ``concScale`` -> replace values above ``highValueThreshold`` by a local
    ``highValueKernel``^3 box mean -> clip to ``outputClip``.

    urOMT is invariant to a global scale of rho, so ``concScale`` only
    conditions the solver - but it must match the reference for objective values
    to be comparable, and it sets the scale that ``highValueThreshold`` and
    ``outputClip`` are expressed in, so the three go together.

    Note the reference runs the box mean on the *full* volume before cropping to
    the ROI, whereas this runs after the crop, so voxels within
    ``highValueKernel`` of the ROI face see a slightly different neighbourhood.
    On the reference breast data that is a handful of voxels.
    """
    im = np.asarray(im, dtype=np.float64)
    scale = float(getattr(cfg, "concScale", 1.0) or 1.0)
    if scale != 1.0:
        im = im * scale
    thr = getattr(cfg, "highValueThreshold", None)
    if thr is not None:
        k = int(getattr(cfg, "highValueKernel", 2) or 2)
        hi = im > float(thr)
        if hi.any():
            # MATLAB convn(im, ones(k,k,k)/k^3, 'same'); origin=-1 aligns an
            # even-sized window the same way.
            im = np.where(hi, uniform_filter(im, size=k, mode="constant",
                                             origin=-1), im)
    clip = getattr(cfg, "outputClip", None)
    if clip:
        lo, hi_ = (clip + [None, None])[:2] if isinstance(clip, list) else clip
        im = np.clip(im, lo, hi_)
    return im


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
    # Optional bbox padding. A scalar pads the two IN-PLANE axes only; a
    # 3-element [row, col, slice] pads each axis independently (the reference
    # getRange pads all three, e.g. [3, 3, 3] on the breast data).
    padR, padC, padS = _bboxPad(getattr(cfg, "bbox_pad", 0))
    if padR > 0:
        minr, maxr = max(0, minr - padR), min(refShape[0] - 1, maxr + padR)
    if padC > 0:
        minc, maxc = max(0, minc - padC), min(refShape[1] - 1, maxc + padC)
    if padS > 0:
        mins, maxs = max(0, mins - padS), min(refShape[2] - 1, maxs + padS)
    if int(getattr(cfg, "bbox_full_z", 0)):
        mins, maxs = 0, refShape[2] - 1

    rs_, re_ = int(minr), int(maxr) + 1
    cs_, ce_ = int(minc), int(maxc) + 1
    ss_, se_ = int(mins), int(maxs) + 1
    if int(getattr(cfg, "fft_pad", 1)):
        rs_, re_ = fftFriendlyRange(rs_, re_, refShape[0])
        cs_, ce_ = fftFriendlyRange(cs_, ce_, refShape[1])
        ss_, se_ = fftFriendlyRange(ss_, se_, refShape[2])
    cfg.bbox = (rs_, re_, cs_, ce_, ss_, se_)

    # Optional mask dilation (MATLAB cfg.dilate). Applied to the FULL-size mask
    # *after* the bounding box is fixed, because the reference computes its
    # ROI range from the undilated mask file and only then dilates - so this
    # widens the ROI coverage without moving the solve domain.
    mask = dilateMask(mask, int(getattr(cfg, "mask_dilate", 0)))

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

        # Normalize data.
        #
        # The conversion runs over the WHOLE cropped box, not just the ROI mask:
        # the mask is applied at the very end, after the smoothing flow. Masking
        # here instead would zero the tissue surrounding the ROI, and the
        # edge-preserving flow would then smooth every boundary voxel against an
        # artificial zero background - measurably pulling them down (on the
        # reference breast data: 35357 in-mask voxels changed, max 32.4, a
        # through-origin slope of 0.979 against the reference density). The
        # reference implementation converts and smooths the full box and masks
        # last, and doing the same here reproduces its density to ~5e-9.
        #
        # `normalizeToBaseline` NaNs whatever the mask excludes, so the mask
        # passed here is "has a usable pre-contrast baseline". That both keeps
        # the surrounding tissue and reproduces the reference's guard against
        # dividing by a zero baseline (an infinite ratio would otherwise clip to
        # conc_clip and fabricate a large concentration out of background).
        scanArr4M = np.stack([np.asarray(f, dtype=np.float64)
                              for f in combinedFrames], axis=3)
        timePtsV = np.arange(scanArr4M.shape[3], dtype=float)
        baseline3M = np.mean(scanArr4M[:, :, :, :basePtsToUse], axis=3)
        normUptake4M, _t, _b, basePtsUsed = normalizeToBaseline(
            scanArr4M, baseline3M > 0, timePtsV, basePts=basePtsToUse,
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
        # reference tofts post-steps (scale / despike / clip) run on the
        # concentration BEFORE the smoothing flow
        frm = toftsPostProcess(frm, cfg)
        if smooth > 0:
            if method == "gaussian":
                frm = gaussian_filter(frm, sigma=0.1 * smooth)
            else:               # affine-invariant mean-curvature flow (MATLAB)
                # MATLAB affine_diffusion_3d(A, t_tot, dt, aff) integrates for a
                # total evolution TIME t_tot, taking n_t = round(t_tot/dt) steps.
                # cfg.smooth is that t_tot (MATLAB cfg.smooth), NOT a step count
                # -- passing it straight in as nSteps under-smooths by 1/dt
                # (e.g. smooth=1, dt=0.1 -> 1 step instead of 10). Verified
                # against MATLAB reference output: reading `smooth` as t_tot
                # reproduces its smoothed density exactly (for any dt), while
                # treating it as a step count leaves the peaks under-smoothed.
                nSteps = max(1, int(round(smooth / smoothDt)))
                frm = affineDiffusion3d(frm, nSteps=nSteps,
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
