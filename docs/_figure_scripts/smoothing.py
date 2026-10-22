"""Figures and printed output for docs/user_guide/smoothing.rst."""
import matplotlib.pyplot as plt
import numpy as np

from common import loadPhantom, save, showAxial
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.contour import polyline_smoothing as ps
from cerr.contour.contour_processing import smoothStructure
from cerr.dataclasses import structure as cerrStr
from cerr.utils import mask as maskUtils

TRUE_C, INPUT_C, OUT_C = "#40ff60", "#ffd000", "#40e0ff"   # on-CT overlays
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"      # on white

planC = loadPhantom()
scanNum, gtvNum = 0, 0
gtv3M = rs.getStrMask(gtvNum, planC)
slc = int(gtv3M.sum(axis=(0, 1)).argmax())
voxCc = np.prod(planC.scan[scanNum].getScanSpacing())


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def segmentsOn(structNum, slc):
    cont = planC.structure[structNum].contour[slc]
    return [seg.points for seg in cont.segments] if cont else []


def perimeter(structNum, slc):
    return sum(np.linalg.norm(np.diff(np.vstack((P, P[:1]))[:, :2], axis=0), axis=1).sum()
               for P in segmentsOn(structNum, slc))


# ---- input: a coarse, jagged GTV -------------------------------------------
# As if a model ran on a 3x coarser in-plane grid and its mask was mapped back
# with nearest-neighbour interpolation: 0.3 cm steps along the boundary.
f = 3
coarse3M = gtv3M[::f, ::f, :].repeat(f, axis=0).repeat(f, axis=1)
coarse3M = coarse3M[:gtv3M.shape[0], :gtv3M.shape[1], :]
planC = pc.importStructureMask(coarse3M, scanNum, "GTV_coarse", planC)
inNum = len(planC.structure) - 1

# ---- the four structure-level methods --------------------------------------
results = {}
planC = smoothStructure(planC, inNum, replace_original=False, name_suffix="_taubinCR")
results["smoothStructure"] = len(planC.structure) - 1

planC = cerrStr.getBsplineSmoothing(inNum, 1.0, 2.0, planC, replaceFlag=False,
                                    procSructName="GTV_bspline")
results["getBsplineSmoothing"] = len(planC.structure) - 1

_, planC = cerrStr.getGaussianBlurredMask(inNum, 1.5, planC, saveFlag=True,
                                          procSructName="GTV_gauss")
results["getGaussianBlurredMask"] = len(planC.structure) - 1

box3M = maskUtils.blurring(rs.getStrMask(inNum, planC), 2, filtType="box") >= 0.5
planC = pc.importStructureMask(box3M, scanNum, "GTV_box", planC)
results["blurring (box)"] = len(planC.structure) - 1

rows = [("GTV (reference)", gtvNum), ("GTV_coarse (input)", inNum)] + list(results.items())
print("%-24s %8s %10s %14s" % ("structure", "vol (cc)", "Dice vs GTV", "perimeter (cm)"))
for name, n in rows:
    m = rs.getStrMask(n, planC)
    print("%-24s %8.1f %10.3f %14.1f" % (name, m.sum() * voxCc, dice(m, gtv3M), perimeter(n, slc)))

# ---- figure 1: structure-level methods on CT --------------------------------
xs = np.vstack(segmentsOn(gtvNum, slc))
x0, y0 = xs[:, 0].max() - 3.2, xs[:, 1].mean() - 1.0
win = [x0, x0 + 4.0, y0 - 2.0, y0 + 2.0]

fig, axs = plt.subplots(1, 4, figsize=(15, 4.3))
for ax, (title, n) in zip(axs, results.items()):
    showAxial(ax, planC, scanNum, slc, window=(-600, 300))
    for P in segmentsOn(gtvNum, slc):
        ax.plot(*np.vstack((P, P[:1]))[:, :2].T, color=TRUE_C, lw=1, ls="--")
    for P in segmentsOn(inNum, slc):
        ax.plot(*np.vstack((P, P[:1]))[:, :2].T, color=INPUT_C, lw=1.2)
    for P in segmentsOn(n, slc):
        ax.plot(*np.vstack((P, P[:1]))[:, :2].T, color=OUT_C, lw=2)
    m = rs.getStrMask(n, planC)
    ax.set_title("%s\nDice vs GTV %.3f" % (title, dice(m, gtv3M)))
    ax.set_xlim(win[0], win[1])
    ax.set_ylim(win[3], win[2])
axs[0].plot([], [], color=TRUE_C, ls="--", label="GTV (reference)")
axs[0].plot([], [], color=INPUT_C, label="GTV_coarse (input)")
axs[0].plot([], [], color=OUT_C, lw=2, label="smoothed")
axs[0].legend(loc="lower left", fontsize=7, framealpha=0.85)
for ax in axs[1:]:
    ax.set_ylabel("")
fig.tight_layout()
save(fig, "smoothing_structure")

# ---- figure 2: polyline functions on one contour ---------------------------


def polylineCases(P, res):
    return [
        ("taubin_smooth", [("factor=0.5, mu=-0.5 (default)", ps.taubin_smooth(P)),
                           ("factor=0.8, mu=0.8 (smoothStructure)",
                            ps.taubin_smooth(P, factor=0.8, mu=0.8))]),
        ("chaikin_smooth", [("iters=1", ps.chaikin_smooth(P, iters=1)),
                            ("iters=5 (default)", ps.chaikin_smooth(P))]),
        ("catmull_rom_smooth", [("alpha=0.5 (default)", ps.catmull_rom_smooth(P)),
                                ("alpha=1.0", ps.catmull_rom_smooth(P, alpha=1.0))]),
        ("bspline_smooth", [("smoothing_factor=0.2", ps.bspline_smooth(P, res, 0.2)),
                            ("smoothing_factor=2.0", ps.bspline_smooth(P, res, 2.0))]),
    ]


dense = max(segmentsOn(inNum, slc), key=len)
dense = np.vstack((dense, dense[:1]))[:, :2]            # closed: last point = first
# A coarse polygon, like one clicked by hand: the reference GTV outline kept
# at one point every ~0.7 cm.
ref = max(segmentsOn(gtvNum, slc), key=len)[:, :2]
ref = ref[np.r_[True, np.any(np.diff(ref, axis=0) != 0, axis=1)]]
clicked = ref[::8]
clicked = np.vstack((clicked, clicked[:1]))
inputs = [("staircase, point at every voxel edge", dense, 1.0),
          ("coarse polygon, ~0.7 cm between points", clicked, 10.0)]
print("staircase %d points; clicked polygon %d points, mean spacing %.2f cm" % (
    len(dense), len(clicked), np.linalg.norm(np.diff(clicked, axis=0), axis=1).mean()))

fig, axs = plt.subplots(2, 4, figsize=(15, 8.4), sharex=True, sharey=True)
for row, (rowLabel, P, res) in zip(axs, inputs):
    for ax, (title, outs) in zip(row, polylineCases(P, res)):
        ax.plot(*P.T, color=GREY, lw=1, marker="o", ms=2.5, label="input points")
        for (lab, X), col in zip(outs, (BLUE, ORANGE)):
            ax.plot(*np.asarray(X).T, color=col, lw=2, label=lab)
        ax.set_title(title + (" (resolution_factor=%g)" % res if title == "bspline_smooth" else ""))
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[3], win[2])
        ax.set_aspect("equal")
        ax.grid(color="#e4e3dd", lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    row[0].set_ylabel("Input: %s\n\ny (cm)" % rowLabel)
for ax in axs[1]:
    ax.set_xlabel("x (cm)")
fig.tight_layout()
save(fig, "smoothing_polyline")
for rowLabel, P, res in inputs:
    print("%s -> taubin %d, chaikin(5) %d, catmull %d, bspline %d points" % (
        rowLabel, len(ps.taubin_smooth(P)), len(ps.chaikin_smooth(P)),
        len(ps.catmull_rom_smooth(P)), len(ps.bspline_smooth(P, res, 2.0))))

# ---- figure 3: strength of B-spline and Gaussian smoothing -----------------
sweeps = [("getBsplineSmoothing, smoothing factor", [0.2, 2.0, 10.0]),
          ("getGaussianBlurredMask, sigma (voxels)", [0.75, 1.5, 3.0])]
fig, axs = plt.subplots(2, 3, figsize=(11.5, 7.6))
print("\n%-40s %6s %8s %10s" % ("sweep", "value", "vol (cc)", "Dice vs GTV"))
for row, (label, values) in zip(axs, sweeps):
    for ax, v in zip(row, values):
        if label.startswith("getBspline"):
            planC = cerrStr.getBsplineSmoothing(inNum, 1.0, v, planC, replaceFlag=False,
                                                procSructName="sweep")
        else:
            _, planC = cerrStr.getGaussianBlurredMask(inNum, v, planC, saveFlag=True,
                                                      procSructName="sweep")
        n = len(planC.structure) - 1
        m = rs.getStrMask(n, planC)
        print("%-40s %6.2f %8.1f %10.3f" % (label, v, m.sum() * voxCc, dice(m, gtv3M)))
        showAxial(ax, planC, scanNum, slc, window=(-600, 300))
        for Q in segmentsOn(inNum, slc):
            ax.plot(*np.vstack((Q, Q[:1]))[:, :2].T, color=INPUT_C, lw=1.2)
        for Q in segmentsOn(n, slc):
            ax.plot(*np.vstack((Q, Q[:1]))[:, :2].T, color=OUT_C, lw=2)
        ax.set_title("%s = %g\nDice vs GTV %.3f" % (label.split(", ")[1], v, dice(m, gtv3M)))
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[3], win[2])
        del planC.structure[n]
    row[0].set_ylabel(label.split(",")[0] + "\n\ny (cm)")
for ax in axs[:, 1:].ravel():
    ax.set_ylabel("")
fig.tight_layout()
save(fig, "smoothing_parameters")
