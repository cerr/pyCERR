"""Sweep sigma and report a VOXEL-WISE comparison of pyCERR vs MATLAB urOMT for
one interval. Loads the planC and runs Part-1 once, then re-solves the urOMT
optimization for each sigma (Part 1 is sigma-independent), so the sweep is cheap.

Voxel metrics (on the MATLAB ROI, best-aligned orientation):
  corr   = Pearson correlation (spatial pattern)
  slope  = through-origin regression py ~ slope*mat (per-voxel magnitude)
  medR   = median per-voxel ratio py/mat
  w/in2x = fraction of voxels with 0.5 < py/mat < 2

Usage: python -m cerr.scripts.sweep_uromt_sigma [--tag 14_16]
"""
import argparse
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import scipy.io as sio

from cerr.scripts.compare_uromt_matlab import (MATLAB_SETTINGS, MAT_DIR,
                                               voxel_compare)
import cerr.plan_container as pc
from cerr.uromt.config import UROMTConfig
from cerr.uromt.data import prepareData, scanTimeOrder
from cerr.uromt.solver import runUROMT
from cerr.uromt.analyze import runEULAIntervals

PKL = r"C:\software\urOMT\test_data\brain_pt_209764\planC_with_seg.pkl"
FIRST, JUMP = 14, 2
SIGMAS = [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2]
METRICS = [("EulerPe", "peclet"), ("EulerS", "effSpeed"), ("EulerRho", "rho")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="14_16")
    args = ap.parse_args()
    start = int(args.tag.split("_")[0])
    last = start + JUMP
    interval = (start - FIRST) // JUMP

    print("loading planC pickle + Part 1 ...", flush=True)
    planC = pc.loadPlanCFromPkl(PKL)
    scanNumV = scanTimeOrder(planC)
    structNum = len(planC.structure) - 1
    settings = dict(MATLAB_SETTINGS)
    settings["time"] = {"first_time": FIRST, "time_jump": JUMP, "last_time": last}
    settings["baselineFrames"] = 1
    settings["normMethod"] = "RSE"
    settings["rhoScale"] = None
    cfg = UROMTConfig(settings, scanNumV, structNum=structNum)
    cfg = prepareData(cfg, planC)
    print("  grid %s  spacing(mm) %s  interval %d (T_%s)\n"
          % (cfg.trueSize, [round(s, 3) for s in cfg.spacing], interval,
             args.tag), flush=True)

    mats = {p: sio.loadmat(os.path.join(
        MAT_DIR, "BRR01_%s_E14_40_T_%s.mat" % (p, args.tag)))["rho_n"]
        .astype(np.float64) for p, _ in METRICS}

    import time as _t
    hdr = "%-9s | %-28s | %-28s | %-28s" % (
        "sigma", "EulerPe (corr/slope/medR/w2)",
        "EulerS  (corr/slope/medR/w2)", "EulerRho(corr/slope/medR/w2)")
    print(hdr); print("-" * len(hdr))
    for sigma in SIGMAS:
        cfg.sigma = sigma
        t0 = _t.time()
        eul = runEULAIntervals(runUROMT(cfg))
        cells = []
        for prefix, key in METRICS:
            arr = np.asarray(eul[key][interval])
            v = voxel_compare(arr, mats[prefix])
            cells.append("%+.2f/%5.2f/%5.2f/%.2f"
                         % (v["corr"], v["slope"], v["medR"], v["w2"]))
        print("%-9.4g | %-28s | %-28s | %-28s   (%.0fs)"
              % (sigma, cells[0], cells[1], cells[2], _t.time() - t0),
              flush=True)


if __name__ == "__main__":
    main()
