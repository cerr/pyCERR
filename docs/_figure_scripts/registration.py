"""Figures and printed output for docs/user_guide/registration.rst.

Needs the ``ants`` extra (``pip install "pycerr[ants]"``).

The bundled data has no second scan of the same patient, so a "moving" scan is
made by warping the phantom with a known, smooth deformation plus a shift. The
registration then has to recover it.
"""
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from common import contourMask, loadPhantom, save, showAxial
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.registration import register


def warpKnown(vol3M, order):
    """Resample ``vol3M`` through a fixed shift + sinusoidal deformation."""
    nR, nC, nS = vol3M.shape
    rM, cM, sM = np.meshgrid(np.arange(nR), np.arange(nC), np.arange(nS), indexing="ij")
    rQ = rM + 7 + 5 * np.sin(2 * np.pi * cM / nC)
    cQ = cM - 9 + 5 * np.sin(2 * np.pi * rM / nR)
    sQ = sM + 1.0
    return map_coordinates(vol3M, [rQ, cQ, sQ], order=order, mode="nearest")


planC = loadPhantom()
baseScanNum = 0
xV, yV, zV = planC.scan[baseScanNum].getScanXYZVals()
fixed3M = planC.scan[baseScanNum].getScanArray().astype(float)
gtvFixed3M = rs.getStrMask(0, planC)

# Moving scan and its GTV, in the same planC (scan 1, structure 1)
planC = pc.importScanArray(warpKnown(fixed3M, 1), xV, yV, zV, "CT", baseScanNum, planC)
movScanNum = len(planC.scan) - 1
planC = pc.importStructureMask(warpKnown(gtvFixed3M.astype(float), 0) > 0.5,
                               movScanNum, "GTV-1 (moving)", planC)
movStructNum = len(planC.structure) - 1
gtvMov3M = rs.getStrMask(movStructNum, planC)

# ---- register ---------------------------------------------------------------
transformDir = tempfile.mkdtemp()
planC = register.registerScansAnts(planC, baseScanNum, planC, movScanNum,
                                   transformSaveDir=transformDir,
                                   typeOfTransform="antsRegistrationSyNQuick[s]")
warpedScanNum = len(planC.scan) - 1
deformS = planC.deform[-1]
print("scans", len(planC.scan), "deform", len(planC.deform))
print(deformS.registrationTool, deformS.algorithm, deformS.deformOutFileType)
print(sorted(deformS.deformParams))

# ---- warp the moving structure ---------------------------------------------
planC = register.warpStructuresAnts(planC, baseScanNum, planC, [movStructNum], deformS)
warpedStructNum = len(planC.structure) - 1
gtvWarped3M = rs.getStrMask(warpedStructNum, planC)
print([s.structureName for s in planC.structure])


def dice(a, b):
    return 2.0 * (a & b).sum() / (a.sum() + b.sum())


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


moving3M = planC.scan[movScanNum].getScanArray().astype(float)
warped3M = planC.scan[warpedScanNum].getScanArray().astype(float)
body = fixed3M > -500
print("RMSE HU before %.1f after %.1f" % (rmse(fixed3M[body], moving3M[body]),
                                          rmse(fixed3M[body], warped3M[body])))
print("Dice before %.3f after %.3f" % (dice(gtvFixed3M, gtvMov3M),
                                       dice(gtvFixed3M, gtvWarped3M)))

# ---- figure ----------------------------------------------------------------
slc = int(np.round(np.where(gtvFixed3M)[2].mean()))
fig, axs = plt.subplots(2, 3, figsize=(12, 7.4))
titles = ["Fixed scan (scan 0)", "Moving scan (scan 1)", "Moving scan warped to fixed (scan 2)"]
for ax, scanNum, title in zip(axs[0], (baseScanNum, movScanNum, warpedScanNum), titles):
    showAxial(ax, planC, scanNum, slc)
    ax.set_title(title)
contourMask(axs[0, 0], planC, 0, gtvFixed3M, slc, "#ffd000", "GTV-1 (fixed)")
contourMask(axs[0, 1], planC, 0, gtvFixed3M, slc, "#ffd000", "GTV-1 (fixed)")
contourMask(axs[0, 1], planC, 0, gtvMov3M, slc, "#ff4040", "GTV-1 (moving)")
contourMask(axs[0, 2], planC, 0, gtvFixed3M, slc, "#ffd000", "GTV-1 (fixed)")
contourMask(axs[0, 2], planC, 0, gtvWarped3M, slc, "#40e0ff", "GTV-1 (warped)")
for ax in axs[0]:
    ax.legend(loc="lower right", fontsize=7)

extent = axs[0, 0].images[0].get_extent()
for ax, img3M, label in ((axs[1, 0], moving3M, "before"), (axs[1, 1], warped3M, "after")):
    im = ax.imshow(img3M[:, :, slc] - fixed3M[:, :, slc], cmap="RdBu_r", vmin=-600,
                   vmax=600, extent=extent)
    ax.set_title("Difference from fixed, %s registration (HU)" % label)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

# checkerboard of fixed and warped
tile = 24
rI, cI = np.indices(fixed3M.shape[:2])
checker = ((rI // tile + cI // tile) % 2).astype(bool)
board = np.where(checker, fixed3M[:, :, slc], warped3M[:, :, slc])
axs[1, 2].imshow(board, cmap="gray", vmin=-160, vmax=240, extent=extent)
axs[1, 2].set_title("Checkerboard: fixed / warped")
axs[1, 2].set_xlabel("x (cm)")
axs[1, 2].set_ylabel("y (cm)")
fig.tight_layout()
save(fig, "registration_before_after")
