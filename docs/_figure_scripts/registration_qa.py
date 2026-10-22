"""Figures and printed output for docs/user_guide/registration_qa.rst.

Needs the ``ants`` and ``viewer`` extras. The viewer screenshots are taken
offscreen (QT_QPA_PLATFORM=offscreen is set below).

As on the registration page, the moving scan is the phantom warped by a known
deformation, so the registration error can be measured against the truth.
"""
import os
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from common import (addMovingScan, knownInverseIndices, loadPhantom,  # noqa: E402
                    save, sliceExtent)
from cerr.contour import rasterseg as rs                                      # noqa: E402
from cerr.contour import structure_metrics as sm                              # noqa: E402
from cerr.registration import register                                        # noqa: E402

planC = loadPhantom()
baseScanNum = 0
planC, movScanNum, movStructNum = addMovingScan(planC, baseScanNum, 0)
planC = register.registerScans(planC, baseScanNum, planC, movScanNum, tempfile.mkdtemp())
warpedScanNum = len(planC.scan) - 1
deformS = planC.deform[-1]
planC = register.warpStructures(planC, baseScanNum, planC, [movStructNum], deformS)
warpedStructNum = len(planC.structure) - 1
planC.structure[warpedStructNum].structureName = "GTV-1 (warped)"
print([s.structureName for s in planC.structure])

# ---- Level 1 and 2 metrics: contours ------------------------------------------
# The moving GTV is on the moving scan; copy it to the fixed scan to compare
from cerr.dataclasses import structure as cerrStr      # noqa: E402
planC = cerrStr.copyToScan(movStructNum, baseScanNum, planC)
movOnFixedNum = len(planC.structure) - 1
fixedGtv3M = rs.getStrMask(0, planC)
for label, n in (("before", movOnFixedNum), ("after", warpedStructNum)):
    m = sm.compareStructurePair(0, n, planC)
    print("%-6s Dice %.3f  HD95 %.2f mm  MSD %.2f mm  volume ratio %.3f  APL %.0f mm" % (
        label, m["Dice"], m["HD95_mm"], m["MSD_mm"], m["VolumeRatio"], m["APL_mm"]))

# ---- The deformation vector field ----------------------------------------------
# ANTs stores a transform chain; sample it as a DVF on the fixed scan's grid
planC = register.calcVectorField(deformS, planC, baseScanNum, tempfile.mkdtemp())
dvfDeform = planC.deform[-1]
(dRow3M, dCol3M, dSlc3M), (xR, yR, zR), info = register.getDvfGrid(
    dvfDeform, planC, baseScanNum, outputResV=[0, 0, 0])
valid3M = info["valid"]
print("DVF", dvfDeform.deformOutFileType, dRow3M.shape, "valid", int(valid3M.sum()))

# Error against the known deformation, in mm
shape = planC.scan[baseScanNum].getScanArray().shape
dx, dy, dz = planC.scan[baseScanNum].getScanSpacing()          # cm
rTrue3M, cTrue3M, sTrue3M = knownInverseIndices(shape)
rI, cI, sI = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
trueX3M, trueY3M, trueZ3M = (cTrue3M - cI) * dx, -(rTrue3M - rI) * dy, (sTrue3M - sI) * dz
err3M = 10 * np.sqrt((dCol3M - trueX3M) ** 2 + (dRow3M - trueY3M) ** 2 + (dSlc3M - trueZ3M) ** 2)

fixed3M = planC.scan[baseScanNum].getScanArray()
body3M = fixed3M > -500
interior = np.zeros_like(body3M)
interior[10:-10, 10:-10, 3:-3] = True                  # ignore the field-of-view edge
inMoving = ((rTrue3M > 1) & (rTrue3M < shape[0] - 2) & (cTrue3M > 1)
            & (cTrue3M < shape[1] - 2) & (sTrue3M > 1) & (sTrue3M < shape[2] - 2))
evalMask3M = body3M & interior & inMoving & valid3M
print("registration error, voxels above -500 HU (mm): median %.2f  95th pct %.2f  max %.2f" % (
    np.median(err3M[evalMask3M]), np.percentile(err3M[evalMask3M], 95),
    err3M[evalMask3M].max()))

# Jacobian determinant, computed by ANTs and added as a scan
planC = register.calcJacobian(deformS, planC)
jac3M = planC.scan[-1].getScanArray()
del planC.scan[-1]
print("Jacobian, same voxels: min %.2f  median %.2f  max %.2f  folding voxels %d" % (
    jac3M[evalMask3M].min(), np.median(jac3M[evalMask3M]), jac3M[evalMask3M].max(),
    int((jac3M[evalMask3M] <= 0).sum())))

# ---- figure: quantitative QA ---------------------------------------------------
slc = int(np.round(np.where(fixedGtv3M)[2].mean()))
ext = sliceExtent(planC.scan[0].getScanXYZVals()[0], planC.scan[0].getScanXYZVals()[1])
fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
step = 12
rr, cc = np.mgrid[5:shape[0] - 5:step, 5:shape[1] - 5:step]
keep = (valid3M & interior)[rr, cc, slc]
rr, cc = rr[keep], cc[keep]
xF, yF = planC.scan[0].getScanXYZVals()[:2]
axs[0].imshow(fixed3M[:, :, slc], cmap="gray", vmin=-160, vmax=240, extent=ext)
axs[0].quiver(xF[cc], yF[rr], dCol3M[rr, cc, slc], dRow3M[rr, cc, slc], color="#ffd000",
              angles="xy", scale_units="xy", scale=1, width=0.004)
axs[0].set_title("Displacement found by ANTs (cm, to scale)")
im = axs[1].imshow(np.ma.masked_where(~evalMask3M[:, :, slc], err3M[:, :, slc]),
                   cmap="magma", vmin=0, vmax=3, extent=ext)
fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03, label="mm")
axs[1].set_title("Error against the known deformation\n(voxels above -500 HU, inside both fields of view)", fontsize=9)
im = axs[2].imshow(np.ma.masked_where(~evalMask3M[:, :, slc], jac3M[:, :, slc]),
                   cmap="RdBu_r", vmin=0.6, vmax=1.4, extent=ext)
fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.03)
axs[2].set_title("Jacobian determinant")
for ax in axs:
    ax.set_xlabel("x (cm)")
    ax.set_facecolor("black")
axs[0].set_ylabel("y (cm)")
fig.tight_layout()
save(fig, "registration_qa_quantitative")

# ---- figure: viewer QA modes, before and after ---------------------------------
from cerr.viewer.pycerr_gui import show                  # noqa: E402
from cerr.viewer.pycerr_gui.app import _get_app          # noqa: E402

app = _get_app()
viewer = show(planC)
viewer.resize(1300, 900)
# Window the moving and warped scans like the fixed scan, in a different colormap
for scanIdx in (movScanNum, warpedScanNum, baseScanNum):
    viewer.set_scan(scanIdx)
    viewer.set_window_level(40, 400)
    viewer.set_scan_colormap("gray" if scanIdx == baseScanNum else "copper")
viewer.set_structures_visible([0])
viewer.goto_structure(0)
shotDir = tempfile.mkdtemp()
shots = {}
for movingNum, label in ((movScanNum, "before"), (warpedScanNum, "after")):
    for mode in ("Mirrorscope", "Sidebyside", "AlternateGrid", "Toggle"):
        kw = {"base_frac": 0.5} if mode == "Toggle" else {"size": 3.0}
        viewer.start_reg_qa(base=baseScanNum, moving=movingNum, mode=mode, **kw)
        viewer.refresh_views()
        for _ in range(10):
            app.processEvents()
            time.sleep(0.05)
        path = os.path.join(shotDir, "%s_%s.png" % (label, mode))
        shots[(label, mode)] = viewer.save_screenshot(path, target="Axial")
viewer.stop_reg_qa()
viewer.close()

modes = ("Mirrorscope", "Sidebyside", "AlternateGrid", "Toggle")
fig, axs = plt.subplots(2, 4, figsize=(15, 7.6))
for i, label in enumerate(("before", "after")):
    for j, mode in enumerate(modes):
        img = plt.imread(shots[(label, mode)])
        axs[i, j].imshow(img)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        if i == 0:
            axs[i, j].set_title(mode)
    axs[i, 0].set_ylabel("%s registration" % label.capitalize(), fontsize=11)
fig.tight_layout()
save(fig, "registration_qa_modes")
