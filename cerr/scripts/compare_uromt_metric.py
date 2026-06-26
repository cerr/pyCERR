"""Compare pyCERR urOMT Eulerian metrics against the MATLAB output for ONE time
interval, using the prepared planC pickle. Runs only that interval (cheap) with
the concentration baseline = the 1st timepoint (frame 0).

Usage (pycerr env, repo root on PYTHONPATH):
    python -m cerr.scripts.compare_uromt_metric --tag 14_16
"""
import argparse
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import scipy.io as sio

from cerr.scripts.compare_uromt_matlab import (MATLAB_SETTINGS, MAT_DIR,
                                               _stats, voxel_compare)
import cerr.plan_container as pc
from cerr.uromt.config import UROMTConfig
from cerr.uromt.data import prepareData, scanTimeOrder
from cerr.uromt.solver import runUROMT
from cerr.uromt.analyze import runEULAIntervals

PKL = r"C:\software\urOMT\test_data\brain_pt_209764\planC_with_seg.pkl"
FIRST, JUMP = 14, 2

# MATLAB prefix -> pyCERR runEULAIntervals key
METRICS = [("EulerS", "effSpeed"), ("EulerR", "rate"), ("EulerPe", "peclet"),
           ("EulerRho", "rho"), ("EulerFlux", "flux")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="14_16", help="interval tag, e.g. 14_16")
    ap.add_argument("--rho-scale", default="none",
                    help="'none' (raw concentration, default), 'auto', or a number")
    ap.add_argument("--norm", default=None,
                    help="override normMethod: CC | RSE | none")
    args = ap.parse_args()
    start = int(args.tag.split("_")[0])
    last = start + JUMP
    interval = (start - FIRST) // JUMP

    import time as _time
    print("loading planC pickle ...", flush=True)
    planC = pc.loadPlanCFromPkl(PKL)
    scanNumV = scanTimeOrder(planC)
    structNum = len(planC.structure) - 1

    settings = dict(MATLAB_SETTINGS)
    settings["time"] = {"first_time": FIRST, "time_jump": JUMP,
                        "last_time": last}
    settings["baselineFrames"] = 1          # baseline = 1st timepoint (frame 0)
    if args.norm:
        settings["normMethod"] = args.norm
    try:
        settings["rhoScale"] = (None if args.rho_scale == "none"
                                else ("auto" if args.rho_scale == "auto"
                                      else float(args.rho_scale)))
    except ValueError:
        settings["rhoScale"] = None
    cfg = UROMTConfig(settings, scanNumV, structNum=structNum)
    cfg = prepareData(cfg, planC)
    print("  ROI grid:", cfg.trueSize, " frames:", len(cfg.vol),
          " spacing(mm):", [round(s, 3) for s in cfg.spacing],
          " norm=%s rhoScale=%s" % (cfg.normMethod, settings["rhoScale"]),
          flush=True)

    m = cfg.mask > 0
    print("  input concentration (ROI) per frame: mean / p90 / max")
    for j, v in enumerate(cfg.vol):
        roi = np.asarray(v)[m]
        print("    frame %d (tp %d): %.4g / %.4g / %.4g"
              % (j, FIRST + JUMP * j, roi.mean(),
                 np.percentile(roi, 90), roi.max()), flush=True)

    t0 = _time.time()
    res = runUROMT(cfg)
    eul = runEULAIntervals(res)
    g = res["gamma"][interval]
    print("  solved %d interval(s) in %.0fs  | interval %d: Gamma=%.4g "
          "(fit G3=%.4g) nfev=%d\n"
          % (len(cfg.vol) - 1, _time.time() - t0, interval, g["Gamma"],
             g["Gamma3"], g["nfev"]), flush=True)

    mask3 = np.asarray(eul["mask"]) > 0
    print("=== T_%s (interval %d)  pyCERR vs MATLAB (VOXEL-WISE) ===" %
          (args.tag, interval))
    print("  corr=spatial Pearson | slope=through-origin regression py~slope*mat"
          " | medR=median voxel ratio | w/in2x=frac 0.5<py/mat<2")
    print("%-9s %8s %8s | %6s %7s %7s %7s %6s"
          % ("metric", "py mean", "mat mean", "corr", "slope", "medR",
             "w/in2x", "nVox"))
    print("-" * 78)
    for prefix, key in METRICS:
        matFile = os.path.join(MAT_DIR,
                               "BRR01_%s_E14_40_T_%s.mat" % (prefix, args.tag))
        if not os.path.exists(matFile):
            continue
        arr = np.asarray(eul[key][interval])
        if key == "flux":
            arr = np.sqrt(np.sum(arr ** 2, axis=0))
        matMap = sio.loadmat(matFile)["rho_n"].astype(np.float64)
        v = voxel_compare(arr, matMap)
        pm = _stats(arr, mask3)[0]
        mm = _stats(matMap, np.abs(matMap) > 0)[0]
        if v is None:
            continue
        print("%-9s %8.3g %8.3g | %+6.2f %7.2f %7.2f %7.2f %6d"
              % (prefix, pm, mm, v["corr"], v["slope"], v["medR"],
                 v["w2"], v["n"]))


if __name__ == "__main__":
    main()
