"""Figures and printed output for docs/user_guide/segmentation_metrics.rst."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from common import contourMask, loadPhantom, save, showAxial
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.contour import structure_metrics as sm
from cerr.dataclasses import structure as cerrStr


planC = loadPhantom()
scanNum, refNum = 0, 0
ref3M = rs.getStrMask(refNum, planC)

# Test contours: the reference grown by 3 mm, shrunk by 3 mm and shifted
planC = cerrStr.getSurfaceExpand(refNum, 0.3, planC)
planC = cerrStr.getSurfaceExpand(refNum, -0.3, planC)
planC = pc.importStructureMask(np.roll(ref3M, 5, axis=1), scanNum, "GTV shifted 5 mm", planC)
testNums = {"GTV + 3 mm": 1, "GTV - 3 mm": 2, "GTV shifted 5 mm": 3}
tests = {name: rs.getStrMask(n, planC) for name, n in testNums.items()}

rows = []
for name, n in testNums.items():
    m = sm.compareStructurePair(refNum, n, planC)
    rows.append((name, m))
keys = list(rows[0][1])
print("%-18s" % "" + "".join("%13s" % k for k in keys))
for name, m in rows:
    print("%-18s" % name + "".join("%13.3f" % m[k] for k in keys))

# ---- metrics against a pure shift ----------------------------------------------
dxMm = planC.scan[scanNum].getScanSpacing()[0] * 10
shiftsVox = np.arange(0, 41, 4)
curve = []
for s in shiftsVox:
    planC = pc.importStructureMask(np.roll(ref3M, int(s), axis=1), scanNum, "shift", planC)
    curve.append(sm.compareStructurePair(refNum, len(planC.structure) - 1, planC))
    del planC.structure[-1]
shiftsMm = shiftsVox * dxMm
for s, m in zip(shiftsMm, curve):
    print("shift %.1f mm  Dice %.3f  HD95 %.2f  MSD %.2f  SurfDice %.3f  APL %.0f mm" %
          (s, m["Dice"], m["HD95_mm"], m["MSD_mm"], m["SurfaceDice_2mm"], m["APL_mm"]))

# ---- figure ------------------------------------------------------------------
slc = int(np.round(np.where(ref3M)[2].mean()))
fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.15]})
test3M = tests["GTV + 3 mm"]
showAxial(axs[0], planC, scanNum, slc)
contourMask(axs[0], planC, scanNum, ref3M, slc, "#ff4040", "reference (GTV)")
contourMask(axs[0], planC, scanNum, test3M, slc, "#40e0ff", "test (GTV + 3 mm)")
axs[0].legend(loc="lower left", fontsize=7)
axs[0].set_title("Reference and test contour")

# Distance from every test-surface voxel to the reference surface, for display
dx, dy, dz = planC.scan[scanNum].getScanSpacing() * 10
refSurf3M = ref3M & ~ndimage.binary_erosion(ref3M)
dist3M = ndimage.distance_transform_edt(~refSurf3M, sampling=(dy, dx, dz))
shift3M = tests["GTV shifted 5 mm"]
testSurf3M = shift3M & ~ndimage.binary_erosion(shift3M)
showAxial(axs[1], planC, scanNum, slc)
xV, yV, _ = planC.scan[scanNum].getScanXYZVals()
rr, cc = np.where(testSurf3M[:, :, slc])
sc_ = axs[1].scatter(xV[cc], yV[rr], c=dist3M[:, :, slc][rr, cc], s=3, cmap="plasma",
                     vmin=0, vmax=8)
contourMask(axs[1], planC, scanNum, ref3M, slc, "#ff4040", "reference")
fig.colorbar(sc_, ax=axs[1], fraction=0.046, pad=0.03, label="distance to reference (mm)")
axs[1].legend(loc="lower left", fontsize=7)
axs[1].set_title("Surface distances, test shifted 5 mm")
for ax in axs[:2]:
    ax.set_xlim(-13.5, 0)
    ax.set_ylim(-8, 4.5)

ax2 = axs[2]
ax2.plot(shiftsMm, [m["Dice"] for m in curve], "o-", color="#1f77b4", label="Dice")
ax2.plot(shiftsMm, [m["SurfaceDice_2mm"] for m in curve], "s-", color="#2ca02c",
         label="surface Dice (2 mm)")
ax2.set_xlabel("Shift of the test contour (mm)")
ax2.set_ylabel("Overlap")
ax2.set_ylim(0, 1.02)
ax3 = ax2.twinx()
ax3.plot(shiftsMm, [m["HD95_mm"] for m in curve], "^--", color="#d62728", label="HD95")
ax3.plot(shiftsMm, [m["MSD_mm"] for m in curve], "v--", color="#ff7f0e", label="mean surface distance")
ax3.set_ylabel("Distance (mm)")
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax3.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
ax2.grid(alpha=0.3)
ax2.set_title("Metrics as the contour is shifted")
fig.tight_layout()
save(fig, "segmentation_metrics")
