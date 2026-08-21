"""Reproduce the reference MATLAB urOMT calculation and compare pyCERR against it.

Everything data-specific (paths, ROI ranges, frame indices) is supplied on the
command line, so no dataset path or patient identifier is baked into the repo.

The reference layout this expects:

  <frames-dir>/  <prefix>N.mat            raw signal frames, variable ``img``
  <mask>         .nii/.nii.gz or .mat     ROI segmentation on the full grid
  <ref-dir>/     rho_*_t_0.mat            initial density
                 u0_*_t_N.mat             converged velocity, interval N
                 r0_*_t_N.mat             converged source,   interval N
                 rhoNe_*_t_N.mat          evolved final density, interval N
                 record.txt               per-interval gamma table
                 EULA_*/SpeedR/*.mat      EulerS / EulerR / EulerPe / EulerFlux

Three modes, cheapest first:

  --mode preprocess   Build the concentration and compare against the reference
                      initial density. Seconds. Validates Part 1 only.
  --mode objective    Additionally load the reference's own converged (u, r) and
                      evaluate pyCERR's objective on them. Minutes. Validates the
                      forward model and objective with no optimizer involved, so
                      any gap is pure discretization.
  --mode solve        Additionally run the optimizer from zero and compare the
                      Eulerian maps, including the in/out surface flux. Hours
                      on CPU; use --gpu.
  --mode surfflux     Rebuild the reference's own flux VECTOR field from its
                      converged (u, r) and report the influx / outflux / net
                      through the ROI surface. Minutes, no optimizer. The
                      reference stores EulerFlux as a MAGNITUDE, so the split
                      cannot be read off its saved maps - this reconstructs the
                      vector and checks |reconstruction| against that saved
                      magnitude before splitting it.

Example::

    python -m cerr.scripts.uromt_matlab_compare \\
        --frames-dir /data/REF/MAT/CONCAT --frame-prefix V1_frame \\
        --mask /data/REF/segmented.nii.gz --mask-layout yxz \\
        --ref-dir /data/REF/diff_2e3_..._rreinit0 \\
        --roi 62:122,210:268,34:79 --spacing 1,1,1.4 \\
        --first 5 --jump 2 --intervals 3 --mode objective
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import scipy.io as sio
from scipy.ndimage import uniform_filter

from cerr.mri_metrics.dce_mri import intToConc
from cerr.uromt.config import loadModelSettings, UROMTConfig
from cerr.uromt.data import dilateMask
from cerr.utils.image_proc import affineDiffusion3d


# --------------------------------------------------------------------------- #
#  settings / IO helpers
# --------------------------------------------------------------------------- #
def loadConcSettings(path=None):
    """Load the DCE concentration settings JSON (defaults to the bundled one)."""
    import json
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "uromt", "settings",
            "dce_concentration_settings.json")
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def parseRange(text):
    """'62:122,210:268,34:79' (1-based, inclusive) -> three 0-based slices."""
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError("--roi needs three colon ranges, e.g. 1:10,1:10,1:5")
    out = []
    for p in parts:
        lo, hi = (int(v) for v in p.split(":"))
        out.append(slice(lo - 1, hi))
    return tuple(out)


def loadMask(path, layout="xyz"):
    """Load a full-grid ROI mask from NIfTI or .mat as a boolean (x, y, z)."""
    if path.endswith((".nii", ".nii.gz")):
        import SimpleITK as sitk
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path))   # (z, y, x)
        arr = arr.transpose(2, 1, 0)                          # -> (x, y, z)
    else:
        d = sio.loadmat(path)
        key = [k for k in d if not k.startswith("__")][0]
        arr = np.asarray(d[key])
    if layout == "yxz":
        # stored (y, x, z): swap the first two axes to reach (x, y, z)
        arr = arr.transpose(1, 0, 2)
    return np.asarray(arr) > 0


def loadFrame(framesDir, prefix, idx):
    """Load one raw signal frame (variable ``img``, or the sole variable)."""
    path = os.path.join(framesDir, "%s%d.mat" % (prefix, idx))
    d = sio.loadmat(path)
    key = "img" if "img" in d else [k for k in d if not k.startswith("__")][0]
    return np.asarray(d[key], dtype=np.float64)


def findRef(refDir, pattern):
    """First file under refDir (recursively) whose name matches `pattern`."""
    rx = re.compile(pattern)
    for root, _dirs, files in os.walk(refDir):
        for f in sorted(files):
            if rx.search(f):
                return os.path.join(root, f)
    return None


def loadRefArray(path):
    d = sio.loadmat(path)
    key = [k for k in d if not k.startswith("__")][0]
    return np.asarray(d[key], dtype=np.float64)


def parseRecord(refDir):
    """Parse the per-interval gamma table from record.txt.
    Columns: time-ind, ti, tf, gamma, gamma1, gamma2, gamma3, max(u), ...
    Returns {interval_index (0-based): dict}."""
    path = os.path.join(refDir, "record.txt")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            bits = line.strip().split(",")
            if len(bits) < 8:
                continue
            try:
                vals = [float(b) for b in bits[:8]]
            except ValueError:
                continue                       # header or prose
            ind = int(vals[0])
            if ind < 1:
                continue                       # the 0,0,0,... sentinel rows
            out[ind - 1] = dict(ti=vals[1], tf=vals[2], gamma=vals[3],
                                gamma1=vals[4], gamma2=vals[5],
                                gamma3=vals[6], maxu=vals[7])
    return out


# --------------------------------------------------------------------------- #
#  Part 1: concentration, reproducing the reference 'tofts' preprocessing
# --------------------------------------------------------------------------- #
def toftsConcentration(signal, baseline, cs):
    """Raw signal -> contrast concentration, per the reference preprocessing.

    ``cs`` is the concentration settings dict. Steps, in the reference's order:
    ratio -> clip -> SPGR inversion -> scale -> replace high values by a local
    box mean -> clip.

    This deliberately calls pyCERR's own :func:`intToConc` rather than a private
    copy of the SPGR algebra, so the comparison actually exercises the shipped
    code path. One consequence: ``intToConc`` clamps negative concentrations to
    zero immediately, whereas the reference keeps them until its final clip, so
    they take part in the high-value box mean. Only voxels above
    ``highValueThreshold`` see a different neighbourhood, which on the breast
    reference is a handful of voxels and bounds the disagreement at ~2e-5
    absolute (mean 3e-10, corr 1.00000000) -- negligible, but it is why this is
    not bit-exact.
    """
    # Voxels with no pre-contrast signal have no defined enhancement; treat them
    # as zero rather than letting inf/nan through (an inf ratio would otherwise
    # clip to signalRatioClip and fabricate a large concentration there).
    ratio = np.divide(signal, baseline, out=np.zeros_like(signal, dtype=float),
                      where=baseline > 0)
    clip = cs.get("signalRatioClip")
    if clip is not None:
        ratio = np.minimum(ratio, float(clip))

    if str(cs.get("method", "tofts")).lower() == "rse":
        im = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    elif str(cs.get("method", "tofts")).lower() == "none":
        im = signal.copy()
    else:
        concDict = {"T10": float(cs["T10"]), "r1": float(cs["r1"]),
                    "TR": float(cs["TR"]), "FA": float(cs["FA"])}
        conc = intToConc(ratio.reshape(-1, 1), concDict).reshape(signal.shape)
        im = np.nan_to_num(conc, nan=0.0, posinf=0.0, neginf=0.0)

    im = im * float(cs.get("concScale", 1.0) or 1.0)

    thr = cs.get("highValueThreshold")
    if thr is not None:
        k = int(cs.get("highValueKernel", 2))
        hi = im > float(thr)
        if hi.any():
            # MATLAB convn(im, ones(k,k,k)/k^3, 'same'); origin=-1 aligns the
            # even-sized window the same way.
            sm = uniform_filter(im, size=k, mode="constant", origin=-1)
            im = np.where(hi, sm, im)
    return im


def buildFrames(args, cs, roi):
    """Concentration frames for the selected time points, cropped to the ROI."""
    nBase = int(cs.get("baselineFrames", 1) or 1)
    baseline = np.mean([loadFrame(args.frames_dir, args.frame_prefix, j)
                        for j in range(1, nBase + 1)], axis=0)
    idx = [args.first + args.jump * k for k in range(args.intervals + 1)]
    lo, hi = cs.get("outputClip", [None, None])
    nSteps = max(1, int(round(float(cs.get("smooth", 0) or 0)
                              / float(cs.get("smooth_dt", 0.1)))))
    frames = []
    for i in idx:
        im = toftsConcentration(loadFrame(args.frames_dir, args.frame_prefix, i),
                                baseline, cs)
        im = im[roi]
        if lo is not None or hi is not None:
            im = np.clip(im, lo, hi)
        if float(cs.get("smooth", 0) or 0) > 0:
            im = affineDiffusion3d(
                im, nSteps=nSteps, dt=float(cs.get("smooth_dt", 0.1)),
                affFlag=str(cs.get("smooth_method", "affine")) != "linear")
        frames.append(im)
    return frames, idx


# --------------------------------------------------------------------------- #
#  comparison helpers
# --------------------------------------------------------------------------- #
def compareMaps(py, ref, mask):
    """Voxel-wise agreement inside `mask`: corr, through-origin slope, median
    ratio, and the fraction within a factor of two."""
    a = np.asarray(py)[mask].ravel()
    b = np.asarray(ref)[mask].ravel()
    if a.size == 0 or a.std() < 1e-30 or b.std() < 1e-30:
        return dict(corr=np.nan, slope=np.nan, medR=np.nan, w2=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = a / np.where(b != 0, b, np.nan)
    rr = rr[np.isfinite(rr)]
    return dict(corr=float(np.corrcoef(a, b)[0, 1]),
                slope=float(np.dot(a, b) / max(np.dot(b, b), 1e-30)),
                medR=float(np.median(rr)) if rr.size else np.nan,
                w2=float(np.mean((rr > 0.5) & (rr < 2))) if rr.size else np.nan,
                relL2=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)))


def _row(label, st):
    print("  %-12s corr=%+.6f  slope=%.4f  medRatio=%.4f  within2x=%.3f  "
          "relL2=%.3e" % (label, st["corr"], st["slope"], st["medR"],
                          st["w2"], st["relL2"]), flush=True)


def pcSave(planC, path):
    """Write a planC as a pickle, readable by ``loadPlanCFromPkl``.

    Pickle rather than HDF5 deliberately: ``saveToH5`` serializes only
    scans/structures/doses/deforms, so it would silently drop ``planC.urOMT`` -
    which is the whole point of saving here. ``plan_container`` exposes no
    pickle writer, hence the direct dump.
    """
    import pickle
    if str(path).lower().endswith((".h5", ".hdf5")):
        raise SystemExit("--save-planc must be a .pkl: saveToH5 does not "
                         "serialize planC.urOMT, so the run would be lost.")
    with open(path, "wb") as fh:
        pickle.dump(planC, fh)


def concToModelSettings(cs, settings):
    """Fold the DCE concentration settings into the urOMT model settings.

    The .mat-frame path applies the concentration recipe itself (see
    :func:`toftsConcentration`); the planC path leaves it to
    ``cerr.uromt.data.prepareData``, which reads these keys. Keeping the mapping
    in one place is what makes the two paths comparable.
    """
    method = str(cs.get("method", "tofts")).lower()
    settings["normMethod"] = {"tofts": "CC", "rse": "RSE"}.get(method, "none")
    for src, dst in (("T10", "T10"), ("r1", "r1"), ("TR", "TR"), ("FA", "FA"),
                     ("baselineFrames", "baselineFrames"),
                     ("concScale", "concScale"),
                     ("highValueThreshold", "highValueThreshold"),
                     ("highValueKernel", "highValueKernel"),
                     ("outputClip", "outputClip"),
                     ("smooth", "smooth"), ("smooth_dt", "smooth_dt"),
                     ("smooth_method", "smooth_method"),
                     ("mask_dilate", "mask_dilate")):
        if src in cs:
            settings[dst] = cs[src]
    ratioClip = cs.get("signalRatioClip")
    if ratioClip is not None:
        settings["conc_clip"] = [0.0, float(ratioClip)]
    return settings


def runFromDicom(args, cs, settings, roi):
    """Load the series from DICOM, run the shipped pipeline, store on planC.

    Returns ``(planC, index)`` where ``index`` indexes ``planC.urOMT``.

    Unlike the .mat-frame path this does NOT force the reference's ROI box:
    ``prepareData`` derives the bounding box from the structure, so the grid can
    differ from ``--roi`` and objective values are not directly comparable to
    ``record.txt``. The point of this path is a run the pyCERR viewer can open.
    """
    import cerr.plan_container as pc
    from cerr.mri_metrics.dce_mri import getScanOrder
    from cerr.uromt import runUROMT

    t0 = time.time()
    print("   loading DICOM (this is the slow part) ...", flush=True)
    planC = pc.loadDcmDir(args.dicom_dir)
    print("   %d scan(s) in %.0f s" % (len(planC.scan), time.time() - t0),
          flush=True)
    if len(planC.scan) < 2:
        raise SystemExit("expected one scan per time point, got %d - the "
                         "series may not be splitting correctly."
                         % len(planC.scan))

    planC = pc.loadNiiStructure(args.mask, 0, planC)
    structNum = len(planC.structure) - 1
    print("   mask -> structure %d (%s)"
          % (structNum, planC.structure[structNum].structureName), flush=True)

    # Pass the FULL time-ordered scan list and let the `time` setting pick the
    # transport window. That matters because `prepareData` takes the external
    # baseline from the HEAD of scanNumV (the temporally-first frames), not from
    # the head of the selected window - handing it only the window would use
    # frames 5 and 7 as the baseline instead of 1 and 2. It also mirrors the
    # reference config (data_index_E = 1:32 with first/jump/last on top).
    # Note the scans are not stored in acquisition order, hence getScanOrder.
    order = list(getScanOrder(planC))
    last = args.first + args.jump * args.intervals
    settings["time"] = {"first_time": args.first, "time_jump": args.jump,
                        "last_time": last}
    print("   %d scans time-ordered; transport window %d:%d:%d (baseline = "
          "first %s)" % (len(order), args.first, args.jump, last,
                         settings.get("baselineFrames")), flush=True)
    scanNumV = order

    t0 = time.time()
    index = runUROMT(planC, structNum=structNum, scanNumV=scanNumV,
                             analyze=True, saveToPlanC=True, **settings)
    print("   pipeline done in %.0f s -> planC.urOMT[%d]"
          % (time.time() - t0, index), flush=True)
    return planC, index


# --------------------------------------------------------------------------- #
def referenceEulerian(args, cfg, settings, vol, nIvl):
    """Per-interval Eulerian maps built from the REFERENCE's converged (u, r).

    The reference saves ``EulerFlux`` as a magnitude, so its in/out split cannot
    be recovered from the saved maps. It does save the converged velocity and
    source per interval, though, so the flux VECTOR can be rebuilt: evolve the
    density with pyCERR's own forward model (chaining off the reference's
    ``rhoNe`` when ``reinitR=0``, as :func:`main`'s objective mode does) and run
    the same :func:`runEULAIntervals` both sides use. Returns ``None`` when the
    reference fields are missing.
    """
    from cerr.uromt.numerics import paramInit, sourceAdvecDiff
    from cerr.uromt.analyze import runEULAIntervals
    par = paramInit(cfg)
    # `par` may live on the GPU (--gpu); the reference arrays come off disk as
    # numpy, and runEULAIntervals is host-side, so move each way explicitly.
    xp = par.get("xp", np)

    def _dev(a):
        return xp.asarray(np.asarray(a, dtype=float))

    def _host(a):
        return np.asarray(a.get() if hasattr(a, "get") else a, dtype=float)

    reinit = bool(int(settings.get("reinitR", 0)))
    uL, rL, rhoL = [], [], []
    for k in range(nIvl):
        pu = findRef(args.ref_dir, r"^u0_.*_t_%d\.mat$" % (k + 1))
        pr = findRef(args.ref_dir, r"^r0_.*_t_%d\.mat$" % (k + 1))
        if not (pu and pr):
            return None
        uM = loadRefArray(pu).ravel()
        rM = loadRefArray(pr).ravel()
        rho0 = vol[k].ravel(order="F")
        if k > 0 and not reinit:
            prev = findRef(args.ref_dir, r"^rhoNe_.*_t_%d\.mat$" % k)
            if prev:
                rho0 = loadRefArray(prev).ravel()
        rho = _host(sourceAdvecDiff(_dev(rho0), _dev(uM), _dev(rM), par))
        N, nt = rho.shape
        # The reference stores u flat; getGamma reads it as (3N, nt) Fortran
        # order with the three components stacked, so unstack it the same way
        # into the (3, N, nt) layout solveUROMT produces.
        U = np.asarray(uM, dtype=float).reshape(3 * N, nt, order="F")
        uL.append(np.stack([U[0:N], U[N:2 * N], U[2 * N:3 * N]]))
        rL.append(np.asarray(rM, dtype=float).reshape(N, nt, order="F"))
        rhoL.append(rho)
    res = dict(u=uL, r=rL, rho=rhoL, n=cfg.trueSize, spacing=cfg.spacing,
               mask=cfg.mask, bbox=cfg.bbox, nt=int(settings["nt"]),
               sigma=float(settings["sigma"]),
               pecletFloor=settings.get("peclet_floor"),
               frameScanNums=cfg.frameScanNums)
    return runEULAIntervals(res)


def surfaceFluxRows(ei, label, mask=None, ref=None):
    """Print influx / outflux / net through the ROI surface, per interval.

    Sign convention is outward-positive, so ``net > 0`` means the region loses
    tracer over that interval. ``ref`` is another such dict to ratio against.
    """
    from cerr.uromt.analyze import intervalSurfaceFlux
    sf = intervalSurfaceFlux(ei, mask=mask)
    print("  %-10s %10s %10s %10s%s"
          % (label, "influx", "outflux", "net", "" if ref is None
             else "     ratio in / out / net"))
    for k in range(len(sf["net"])):
        row = ("    ivl %-4d %10.4g %10.4g %10.4g"
               % (k, sf["influx"][k], sf["outflux"][k], sf["net"][k]))
        if ref is not None and k < len(ref["net"]):
            def _r(a, b):
                return a / b if abs(b) > 1e-30 else float("nan")
            row += ("   %.3f / %.3f / %.3f"
                    % (_r(sf["influx"][k], ref["influx"][k]),
                       _r(sf["outflux"][k], ref["outflux"][k]),
                       _r(sf["net"][k], ref["net"][k])))
        print(row, flush=True)
    return sf


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare pyCERR urOMT against a reference MATLAB run.")
    ap.add_argument("--frames-dir", required=True,
                    help="directory of raw signal frame .mat files")
    ap.add_argument("--frame-prefix", default="frame",
                    help="filename prefix before the frame number")
    ap.add_argument("--mask", required=True, help="ROI mask (.nii/.nii.gz/.mat)")
    ap.add_argument("--mask-layout", default="xyz", choices=["xyz", "yxz"],
                    help="axis order the mask is stored in")
    ap.add_argument("--ref-dir", help="reference results directory "
                                      "(omit for --mode preprocess)")
    ap.add_argument("--settings", help="DCE concentration settings JSON")
    ap.add_argument("--model-settings", help="urOMT model settings JSON")
    ap.add_argument("--roi", required=True,
                    help="1-based inclusive ROI, e.g. 62:122,210:268,34:79")
    ap.add_argument("--spacing", default="1,1,1",
                    help="voxel spacing in mm as x,y,z")
    ap.add_argument("--first", type=int, default=1, help="first frame index")
    ap.add_argument("--jump", type=int, default=1, help="frame step")
    ap.add_argument("--intervals", type=int, default=1,
                    help="number of transport intervals to compare")
    ap.add_argument("--mode", default="preprocess",
                    choices=["preprocess", "objective", "solve", "lagrangian",
                             "surfflux"])
    ap.add_argument("--path-spfs", type=int, default=1,
                    help="pathline seeding: every Nth voxel per AXIS "
                         "(thins as N**3; default 1 = every ROI voxel)")
    ap.add_argument("--path-neuler", type=int, default=5,
                    help="Euler sub-steps per urOMT sub-step for pathlines")
    ap.add_argument("--path-sltol", type=float, default=0.0,
                    help="drop pathlines with net displacement below this many "
                         "voxels. Default 0 = keep all; 1.0 discards the ~88%% "
                         "that move sub-voxel and biases the maps high")
    ap.add_argument("--gpu", action="store_true", help="run the solver on GPU")
    ap.add_argument("--maxUiter", type=int, help="override outer iterations")
    ap.add_argument("--dicom-dir",
                    help="load the time series from DICOM and run the SHIPPED "
                         "planC pipeline instead of the .mat frames. Slower, "
                         "but it exercises loadDcmDir + prepareData, and is "
                         "what --save-planc needs.")
    ap.add_argument("--save-planc", metavar="FILE",
                    help="write the planC (with the run stored on planC.urOMT) "
                         "to this .pkl so it can be opened in the pyCERR viewer "
                         "and the velocity vectors overlaid. Requires "
                         "--dicom-dir.")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a model setting, repeatable. The reference "
                         "weights are usually NOT pyCERR's defaults, e.g. "
                         "--set alpha=30000 --set beta=1000")
    args = ap.parse_args(argv)

    if args.save_planc and not args.dicom_dir:
        raise SystemExit("--save-planc requires --dicom-dir: the planC pipeline "
                         "is what populates planC.urOMT.")

    cs = loadConcSettings(args.settings)
    roi = parseRange(args.roi)
    spacing = [float(v) for v in args.spacing.split(",")]

    print("== Part 1: concentration ==", flush=True)
    frames, idx = buildFrames(args, cs, roi)
    maskFull = loadMask(args.mask, args.mask_layout)
    mask = maskFull[roi]
    dil = int(cs.get("mask_dilate", 0) or 0)
    # Dilate on the FULL grid and then crop, as the reference does: dilating the
    # cropped mask would drop the contribution of voxels just outside the box.
    # Coerce to bool -- dilateMask may return a non-boolean array, and `~` on an
    # integer array is a bitwise negation, not a logical one.
    maskDil = np.asarray(dilateMask(maskFull, dil) if dil else maskFull) > 0
    maskDil = maskDil[roi]
    print("   ROI %s = %d voxels | mask %d -> dilated %d | frames %s"
          % ("x".join(str(s.stop - s.start) for s in roi),
             int(np.prod([s.stop - s.start for s in roi])),
             int(mask.sum()), int(maskDil.sum()), idx), flush=True)

    vol = []
    for f in frames:
        g = f.copy()
        g[~maskDil] = 0.0
        vol.append(g)

    if args.ref_dir:
        p = findRef(args.ref_dir, r"^rho_.*_t_0\.mat$")
        if p:
            ref0 = loadRefArray(p).ravel()
            got = vol[0].ravel(order="F")
            d = np.abs(got - ref0)
            print("   vs reference initial density: max|diff|=%.3e  "
                  "mean|diff|=%.3e  corr=%.8f"
                  % (d.max(), d.mean(),
                     np.corrcoef(got[maskDil.ravel(order='F')],
                                 ref0[maskDil.ravel(order='F')])[0, 1]),
                  flush=True)
        else:
            print("   (no rho_*_t_0.mat found under --ref-dir)", flush=True)
    if args.mode == "preprocess":
        return 0

    # ---- model config ------------------------------------------------------
    record = parseRecord(args.ref_dir) if args.ref_dir else {}
    settings = loadModelSettings(args.model_settings)
    settings["fft_pad"] = 0            # stay on the reference's prescribed grid
    settings["mask_dilate"] = 0        # already applied above
    if args.gpu:
        # `loadModelSettings` has already renamed the legacy `gpu` key to
        # `useGPU`, and `getUseGPU` reads `useGPU` first - writing `gpu` here
        # left `useGPU: "no"` in place and silently ran on the CPU.
        settings["useGPU"] = "yes"
    if args.maxUiter:
        settings["maxUiter"] = args.maxUiter
    for kv in args.set:
        key, _, val = kv.partition("=")
        if key not in settings:
            raise SystemExit("unknown model setting: %s" % key)
        try:
            settings[key] = int(val)
        except ValueError:
            try:
                settings[key] = float(val)
            except ValueError:
                settings[key] = val
    print("   model: alpha=%s beta=%s sigma=%s dt=%s nt=%s reinitR=%s "
          "warm_start=%s"
          % (settings["alpha"], settings["beta"], settings["sigma"],
             settings["dt"], settings["nt"], settings["reinitR"],
             settings.get("warm_start")), flush=True)

    if args.dicom_dir:
        print("\n== planC pipeline from DICOM ==", flush=True)
        settings = concToModelSettings(cs, settings)
        settings["fft_pad"] = 0
        planC, index = runFromDicom(args, cs, settings, roi)
        res = planC.urOMT[index].UROMTResult
        print("   ROI grid %s | %d interval(s)"
              % (res["n"], len(res["u"])), flush=True)
        for k, g in enumerate(res["gamma"]):
            ref = record.get(k, {}) if record else {}
            extra = ("  ref Gamma %.5g ratio %.4f"
                     % (ref["gamma"], g["Gamma"] / ref["gamma"])
                     if ref.get("gamma") else "")
            print("   ivl %d: Gamma=%.5g G1=%.5g G2=%.5g G3=%.5g%s"
                  % (k, g["Gamma"], g["Gamma1"], g["Gamma2"], g["Gamma3"],
                     extra), flush=True)
        if args.save_planc:
            t0 = time.time()
            pcSave(planC, args.save_planc)
            print("\n   saved planC -> %s  (%.0f s)"
                  % (args.save_planc, time.time() - t0), flush=True)
            print("   open it in the viewer, then Tools -> urOMT -> "
                  "Existing runs -> Overlay 'Velocity vectors' -> Show on scan",
                  flush=True)
        return 0
    cfg = UROMTConfig(settings, scanNumV=list(range(len(vol))), structNum=0)
    cfg.mask = maskDil.astype(np.uint8)
    cfg.trueSize = list(vol[0].shape)
    cfg.spacing = spacing
    cfg.bbox = (roi[0].start, roi[0].stop, roi[1].start, roi[1].stop,
                roi[2].start, roi[2].stop)
    cfg.vol = vol
    cfg.chi = None
    cfg.frameScanNums = idx

    from cerr.uromt.numerics import paramInit, getGamma
    from cerr.uromt.solver import solveUROMT
    from cerr.uromt.analyze import runEULAIntervals

    if args.mode == "objective":
        print("\n== Objective at the reference's own (u, r) ==", flush=True)
        par = paramInit(cfg)
        reinit = bool(int(settings.get("reinitR", 0)))
        for k in range(args.intervals):
            pu = findRef(args.ref_dir, r"^u0_.*_t_%d\.mat$" % (k + 1))
            pr = findRef(args.ref_dir, r"^r0_.*_t_%d\.mat$" % (k + 1))
            if not (pu and pr):
                print("  interval %d: reference (u, r) not found" % k)
                continue
            uM = loadRefArray(pu).ravel()
            rM = loadRefArray(pr).ravel()
            # With reinitR=0 an interval starts from the PREVIOUS interval's
            # evolved density, not from the measured frame. Chain off the
            # reference's own rhoNe so this stays a pure per-interval check of
            # the forward model rather than accumulating our own drift.
            rho0 = vol[k].ravel(order="F")
            if k > 0 and not reinit:
                prev = findRef(args.ref_dir, r"^rhoNe_.*_t_%d\.mat$" % k)
                if prev:
                    rho0 = loadRefArray(prev).ravel()
                else:
                    print("    (no reference rhoNe for interval %d; starting "
                          "from the measured frame, expect a mismatch)" % (k - 1))
            G, comps, _rho = getGamma(rho0, uM, rM, par,
                                      vol[k + 1].ravel(order="F"))
            ref = record.get(k, {})
            print("  interval %d" % k, flush=True)
            for name, val, refval in (("Gamma", G, ref.get("gamma")),
                                      ("Gamma1", comps[0], ref.get("gamma1")),
                                      ("Gamma2", comps[1], ref.get("gamma2")),
                                      ("Gamma3", comps[2], ref.get("gamma3"))):
                if refval:
                    print("    %-7s %12.5g  ref %12.5g  ratio %.4f"
                          % (name, val, refval, val / refval), flush=True)
                else:
                    print("    %-7s %12.5g" % (name, val), flush=True)
        return 0

    if args.mode == "surfflux":
        print("\n== Surface flux from the reference's own (u, r) ==",
              flush=True)
        refEi = referenceEulerian(args, cfg, settings, vol, args.intervals)
        if refEi is None:
            print("   reference (u, r) not found - nothing to reconstruct")
            return 1
        # Sanity-check the reconstruction against the reference's OWN saved
        # EulerFlux magnitude before trusting the split derived from it.
        print("  |reconstructed flux| vs the saved EulerFlux map:", flush=True)
        for k in range(args.intervals):
            tag = r"EulerFlux_.*_T_%02d_%02d\.mat$" % (idx[k], idx[k + 1])
            pth = findRef(args.ref_dir, tag)
            if not pth:
                continue
            mag = np.sqrt(np.sum(np.asarray(refEi["flux"][k]) ** 2, axis=0))
            _row("EulerFlux", compareMaps(mag, loadRefArray(pth), maskDil))
        surfaceFluxRows(refEi, "reference")
        return 0

    # ---- full solve --------------------------------------------------------
    print("\n== Solving %d interval(s) ==" % args.intervals, flush=True)
    t0 = time.time()
    res = solveUROMT(cfg)
    eul = runEULAIntervals(res)
    print("   solved in %.0f s" % (time.time() - t0), flush=True)
    for k, g in enumerate(res["gamma"]):
        ref = record.get(k, {})
        extra = ("  ref Gamma %.5g ratio %.4f" % (ref["gamma"],
                                                  g["Gamma"] / ref["gamma"])
                 if ref.get("gamma") else "")
        print("   ivl %d: Gamma=%.5g G1=%.5g G2=%.5g G3=%.5g nfev=%d%s"
              % (k, g["Gamma"], g["Gamma1"], g["Gamma2"], g["Gamma3"],
                 g["nfev"], extra), flush=True)

    print("\n== Eulerian maps vs reference ==", flush=True)
    pairs = (("EulerS", "speed"), ("EulerR", "rate"), ("EulerPe", "peclet"),
             ("EulerFlux", "flux"))
    for k in range(min(args.intervals, len(res["gamma"]))):
        print("  interval %d" % k, flush=True)
        for refName, key in pairs:
            tag = r"%s_.*_T_%02d_%02d\.mat$" % (refName, idx[k], idx[k + 1])
            p = findRef(args.ref_dir, tag)
            if not p:
                continue
            arr = np.asarray(eul[key][k])
            if key == "flux":
                arr = np.sqrt(np.sum(arr ** 2, axis=0))
            _row(refName, compareMaps(arr, loadRefArray(p), maskDil))

    # In/out flux through the ROI surface. The reference's EulerFlux is a
    # MAGNITUDE, so the split has no saved counterpart - it is derived from the
    # reference's own converged (u, r) the same way, and ratioed against.
    print("\n== Surface flux (outward-positive; net > 0 = ROI losing) ==",
          flush=True)
    refEi = referenceEulerian(args, cfg, settings, vol, args.intervals)
    if refEi is not None:
        refSf = surfaceFluxRows(refEi, "reference")
        surfaceFluxRows(eul, "pyCERR", ref=refSf)
    else:
        surfaceFluxRows(eul, "pyCERR")

    # ---- Part 4: Lagrangian maps vs the reference LPPA_*/SpeedR/ -----------
    if args.mode == "lagrangian":
        from cerr.uromt.analyze import runGLAD, lagrangianMaps
        print("\n== Lagrangian (Part 4) vs reference ==", flush=True)
        t0 = time.time()
        Lag = runGLAD(res, spfs=args.path_spfs, nEuler=args.path_neuler,
                      slTolVox=args.path_sltol)
        maps = lagrangianMaps(Lag)
        print("   %d pathlines (spfs=%d, nEuler=%d, slTolVox=%.1f) in %.0f s"
              % (len(Lag["SL"]), args.path_spfs, args.path_neuler,
                 args.path_sltol, time.time() - t0), flush=True)
        # The reference writes ONE whole-run map per metric (not per interval),
        # in a plain and a `_full` variant; `_full` is the one that matches.
        for refName, key in (("LagS", "speed"), ("LagPe", "peclet")):
            for variant in ("_full", ""):
                p = findRef(args.ref_dir,
                            r"%s%s_.*\.mat$" % (refName, variant))
                if not p:
                    continue
                _row("%s%s" % (refName, variant or "(plain)"),
                     compareMaps(np.asarray(maps[key]), loadRefArray(p),
                                 maskDil))
        missing = [m for m in ("LagR", "LagFlux", "LagRFlux", "LagAdv",
                               "LagDiff") if findRef(args.ref_dir,
                                                     r"%s_.*\.mat$" % m)]
        if missing:
            print("   (no pyCERR counterpart for %s: runGLAD samples only "
                  "speed and Peclet along the trajectory)" % ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
