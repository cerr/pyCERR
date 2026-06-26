"""Compare the pyCERR urOMT implementation against the MATLAB urOMT output.

Test case: brain perfusion DCE-MRI (patient 209764).
  Inputs : DICOM scans (40 dynamic phases) + a NIfTI ROI segmentation.
  MATLAB : per-interval Eulerian maps (EulerS=speed, EulerR=source rate,
           EulerPe=Peclet, EulerRho=density, EulerFlux) on the ROI bbox grid,
           one .mat (variable ``rho_n``) per time interval.

The pyCERR Gauss-Newton solver is *not* identical to the MATLAB exact GN block,
and the ROI bbox / smoothing / orientation differ, so this is a **similarity**
comparison (magnitudes, spatial pattern, temporal trend), not a bit-exact one.
It reports, per interval and per quantity:
  * ROI statistics (mean / median / p90) for pyCERR vs MATLAB,
  * the best-aligned spatial Pearson correlation (orientation auto-detected),
and the correlation of the per-interval ROI-mean time series. A summary figure
is written next to the data.

Usage (run in the pycerr env, repo root on PYTHONPATH):
    python -m cerr.scripts.compare_uromt_matlab [--intervals N]
"""
import argparse
import os

# Single-threaded BLAS: the ROI is small (~34k voxels), so multithreaded numpy/
# scipy oversubscribe the cores with more sync overhead than benefit. Must be
# set before numpy is imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

DATA_DIR = r"C:\software\urOMT\test_data\brain_pt_209764"
MAT_DIR = os.path.join(DATA_DIR, "SpeedR_outout_from_matlab")
SEG_NII = os.path.join(DATA_DIR, "segmentation_mask_XC.nii")
DICOM_DIR = os.path.join(DATA_DIR, "DICOM")

# MATLAB settings_from_matlab.txt (FA/TR commented out -> no SPGR concentration;
# MATLAB rho mean ~2.2 / max ~151 indicates relative signal enhancement
# S(t)/S0 -> normMethod='RSE'). S0 is an EXTERNAL pre-contrast baseline (mean of
# the first BASELINE_FRAMES temporally-ordered phases), not consumed from the
# transport sequence, so we transport phases 14,16,...,40 = 13 intervals
# (T_14_16 ... T_38_40), matching MATLAB. bbox padded +2 in-plane with full-z to
# match the MATLAB getRange grid (70x49x10; pyCERR is the transpose 49x70x10).
BASELINE_FRAMES = 3       # pre-contrast phases averaged as S0 (try 1..10)
MATLAB_SETTINGS = dict(
    sigma=2e-3, dt=0.4, nt=10, alpha=30000.0, beta=1000.0, niter_pcg=60,
    maxUiter=6, solver="gn", gnLambda0=0.02, dTri=1, reinitR=0,
    smooth=1, smooth_method="affine", smooth_dt=0.1,   # MATLAB smooth=0.5 (light)
    do_resize=0, size_factor=1.0,
    normMethod="CC", baselineFrames=BASELINE_FRAMES,    # concentration via SPGR
    T10=2.1, r1=3.9,                                     # (TR=3.524ms, FA=25 from DICOM)
    rhoScale="auto",                                    # rescale tiny conc -> O(1) (velocity-invariant)
    conc_clip=None,
    bbox_pad=2, bbox_full_z=1,                          # match MATLAB getRange grid
    spacing=None, chiStructNum=None,
    time={"first_time": 14, "time_jump": 2, "last_time": 40})

# quantity -> (pyCERR runEULAIntervals key, MATLAB file prefix)
QUANTITIES = [("effSpeed", "EulerS"), ("rate", "EulerR"),
              ("peclet", "EulerPe"), ("rho", "EulerRho")]


def _interval_tag(t):
    """pyCERR interval t (0-based) -> MATLAB T_{start}_{end} tag."""
    start = 14 + 2 * t
    return "%d_%d" % (start, start + 2)


def _load_mat(prefix, t):
    import scipy.io as sio
    f = os.path.join(MAT_DIR, "BRR01_%s_E14_40_T_%s.mat" % (prefix,
                                                            _interval_tag(t)))
    if not os.path.exists(f):
        return None
    return sio.loadmat(f)["rho_n"].astype(np.float64)


def _stats(a, mask):
    v = a[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 0.0, 0.0)
    return (float(v.mean()), float(np.median(v)), float(np.percentile(v, 90)))


def _best_corr(a, b):
    """Best Pearson correlation of pyCERR map ``a`` with MATLAB map ``b`` over
    the MATLAB ROI (b!=0), searching in-plane transpose + axis flips to absorb
    the DICOM<->NIfTI orientation difference. ``a`` is resized to ``b``'s shape."""
    from scipy.ndimage import zoom

    def corr(x, y):
        roi = (np.abs(y) > 0) & np.isfinite(x) & np.isfinite(y)
        if roi.sum() < 10:
            return -2.0
        xa, ya = x[roi], y[roi]
        if xa.std() < 1e-12 or ya.std() < 1e-12:
            return -2.0
        return float(np.corrcoef(xa, ya)[0, 1])

    best, bestT = -2.0, None
    for tr in ((0, 1, 2), (1, 0, 2)):
        at = np.transpose(a, tr)
        az = (at if at.shape == b.shape
              else zoom(at, [nb / na for nb, na in zip(b.shape, at.shape)],
                        order=1))
        for fx in (1, -1):
            for fy in (1, -1):
                for fz in (1, -1):
                    c = corr(az[::fx, ::fy, ::fz], b)
                    if c > best:
                        best, bestT = c, (tr, fx, fy, fz)
    return best, bestT


def voxel_compare(py, mat):
    """Best-aligned VOXEL-WISE comparison of pyCERR map ``py`` vs MATLAB ``mat``
    over the MATLAB ROI (mat != 0). Returns dict with ``corr`` (Pearson spatial
    pattern), ``slope`` (through-origin regression py ~ slope*mat = per-voxel
    magnitude), ``medR`` (median per-voxel ratio py/mat), ``w2`` (fraction of
    voxels with 0.5 < py/mat < 2) and ``n``. Searches the in-plane transpose +
    axis flips that absorb the DICOM<->NIfTI orientation difference."""
    from scipy.ndimage import zoom
    best = None
    for tr in ((0, 1, 2), (1, 0, 2)):
        at = np.transpose(py, tr)
        az = (at if at.shape == mat.shape
              else zoom(at, [m / a for m, a in zip(mat.shape, at.shape)],
                        order=1))
        for fx in (1, -1):
            for fy in (1, -1):
                for fz in (1, -1):
                    cand = az[::fx, ::fy, ::fz]
                    roi = (np.abs(mat) > 0) & np.isfinite(cand) & np.isfinite(mat)
                    if roi.sum() < 10:
                        continue
                    x, y = cand[roi], mat[roi]
                    if x.std() < 1e-12 or y.std() < 1e-12:
                        continue
                    c = float(np.corrcoef(x, y)[0, 1])
                    if best is None or c > best[0]:
                        best = (c, x, y)
    if best is None:
        return None
    c, x, y = best
    ratio = x / np.where(np.abs(y) < 1e-12, np.nan, y)
    return dict(corr=c, slope=float(np.sum(x * y) / (np.sum(y * y) + 1e-12)),
                medR=float(np.nanmedian(ratio)),
                w2=float(np.mean((ratio > 0.5) & (ratio < 2.0))), n=len(x))


def run_pycerr(maxIntervals=None, fast=False):
    """Load the data, run pyCERR urOMT with MATLAB-matched settings, return the
    per-interval Eulerian maps (runEULAIntervals output)."""
    import time as _time
    import cerr.plan_container as pc
    from cerr.uromt.config import UROMTConfig
    from cerr.uromt.data import prepareData, scanTimeOrder
    from cerr.uromt.solver import runUROMT
    from cerr.uromt.analyze import runEULAIntervals

    print("Loading DICOM (40 phases) ...")
    planC = pc.loadDcmDir(DICOM_DIR)
    print("  scans:", len(planC.scan))
    pc.loadNiiStructure(SEG_NII, 0, planC)             # ROI on the scan grid
    print("  loaded segmentation as structure %d" % (len(planC.structure) - 1))

    settings = dict(MATLAB_SETTINGS)
    if fast:
        # shrink the grid (8x fewer voxels) + fewer GN steps. Do NOT lower
        # niter_pcg: under-solving the CG makes the GN step fail the line search,
        # triggering Levenberg retries that re-solve CG and run *slower*.
        settings.update(do_resize=1, size_factor=0.5, maxUiter=4)
        print("  FAST mode: do_resize=0.5, maxUiter=4")
    if maxIntervals:                                   # fewer frames for a quick run
        last = 14 + 2 * maxIntervals                   # external baseline -> N intervals
        settings["time"] = {"first_time": 14, "time_jump": 2, "last_time": last}
    scanNumV = scanTimeOrder(planC)                    # order by temporalPositionIndex
    cfg = UROMTConfig(settings, scanNumV, structNum=len(planC.structure) - 1)
    cfg = prepareData(cfg, planC)
    print("  ROI grid (pyCERR):", cfg.trueSize, " frames:", len(cfg.vol),
          " spacing(cm):", [round(s, 4) for s in cfg.spacing])
    print("Running urOMT (%d intervals) ..." % (len(cfg.vol) - 1))
    t0 = _time.time()
    res = runUROMT(cfg)
    print("  urOMT solve: %.0fs (%d intervals)"
          % (_time.time() - t0, len(cfg.vol) - 1))
    return runEULAIntervals(res)


def compare(maxIntervals=None, fast=False):
    eul = run_pycerr(maxIntervals, fast=fast)
    nInt = len(eul["effSpeed"])
    mask3 = np.asarray(eul["mask"]) > 0

    print("\n%-4s %-8s | %-26s | %-26s | corr" %
          ("intv", "quant", "pyCERR mean/med/p90", "MATLAB mean/med/p90"))
    print("-" * 86)
    series = {q: {"py": [], "mat": []} for q, _ in QUANTITIES}
    corrs = {q: [] for q, _ in QUANTITIES}
    for t in range(nInt):
        for key, prefix in QUANTITIES:
            pyMap = eul[key][t]
            matMap = _load_mat(prefix, t)
            if matMap is None:
                continue
            pm, pmed, pp90 = _stats(pyMap, mask3)
            mm, mmed, mp90 = _stats(matMap, np.abs(matMap) > 0)
            c, _ = _best_corr(pyMap, matMap)
            series[key]["py"].append(pm)
            series[key]["mat"].append(mm)
            corrs[key].append(c)
            print("%-4d %-8s | %8.3g %8.3g %8.3g | %8.3g %8.3g %8.3g | %+.2f"
                  % (t, prefix, pm, pmed, pp90, mm, mmed, mp90, c))

    print("\n=== summary (over %d intervals) ===" % nInt)
    for key, prefix in QUANTITIES:
        if not series[key]["py"]:
            continue
        py = np.array(series[key]["py"])
        mat = np.array(series[key]["mat"])
        ratio = float(np.median(py / (mat + 1e-12)))
        tcorr = (float(np.corrcoef(py, mat)[0, 1]) if len(py) > 2 else float("nan"))
        sc = np.array(corrs[key])
        print("  %-8s ROI-mean ratio(py/mat)=%.2f  time-trend corr=%+.2f  "
              "spatial corr mean=%+.2f" % (prefix, ratio, tcorr, sc.mean()))

    _save_figure(series, nInt)
    return eul, series, corrs


def _save_figure(series, nInt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.arange(nInt)
    fig, axes = plt.subplots(1, len(QUANTITIES), figsize=(4 * len(QUANTITIES), 3.4))
    for ax, (key, prefix) in zip(np.atleast_1d(axes), QUANTITIES):
        if not series[key]["py"]:
            continue
        ax.plot(xs[:len(series[key]["py"])], series[key]["py"], "o-",
                label="pyCERR")
        ax.plot(xs[:len(series[key]["mat"])], series[key]["mat"], "s--",
                label="MATLAB")
        ax.set_title("%s  ROI mean / interval" % prefix)
        ax.set_xlabel("interval"); ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(DATA_DIR, "pycerr_vs_matlab_summary.png")
    fig.savefig(out, dpi=110)
    print("\nsaved figure:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", type=int, default=None,
                    help="run only the first N intervals (quick check)")
    ap.add_argument("--fast", action="store_true",
                    help="cheaper CG (niter_pcg=20) and fewer GN steps "
                         "(maxUiter=4), full resolution")
    args = ap.parse_args()
    compare(args.intervals, fast=args.fast)
