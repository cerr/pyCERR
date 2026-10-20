"""Figures and printed output for docs/user_guide/ai_inference.rst.

No model is run here: model weights are not bundled. The pre-processing figure
uses the head and neck CT; the post-processing figure stands a noisy copy of
the phantom GTV in for a model's output.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from common import loadPhantom, save, sliceExtent
from cerr import datasets
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.dataclasses import structure as cerrStr
from cerr.utils import image_proc
from cerr.utils import mask as maskUtils

warnings.filterwarnings("ignore")

# ============================================================================
# Pre-processing
# ============================================================================
planC = pc.loadDcmDir(datasets.fetch_sample_data("head_and_neck"))
scanNum = 0
scan3M = planC.scan[scanNum].getScanArray()
gridS = planC.scan[scanNum].getScanXYZVals()
print("original", scan3M.shape)

# a) pad or crop to a fixed matrix size
pad3M, _, padGridS = image_proc.resizeScanAndMask(scan3M, None, gridS,
                                                  [512, 512, scan3M.shape[2]], "padorcrop3d")
print("padorcrop3d", pad3M.shape)

# b) crop to the body, then resize to the network's input size
body3M = maskUtils.getPatientOutline(scan3M, outThreshold=-400)
rMin, rMax, cMin, cMax, sMin, sMax, _ = maskUtils.computeBoundingBox(body3M, maskFlag=5)
crop3M = scan3M[rMin:rMax + 1, cMin:cMax + 1, sMin:sMax + 1]
cropGridS = (gridS[0][cMin:cMax + 1], gridS[1][rMin:rMax + 1], gridS[2][sMin:sMax + 1])
print("crop to body", crop3M.shape)
net3M, _, netGridS = image_proc.resizeScanAndMask(crop3M, None, cropGridS,
                                                  [256, 256, crop3M.shape[2]], "bilinear",
                                                  preserveAspectFlag=True)
print("bilinear 256 x 256", net3M.shape)

# Imported back with their grids, both line up with the original scan
planC = pc.importScanArray(net3M, *netGridS, "CT", scanNum, planC)
netScanNum = len(planC.scan) - 1
print("imported as scan", netScanNum, planC.scan[netScanNum].getScanSize())

# c) present sagittal slices to a 2D network
sag3M, _, sagGridS = image_proc.transformScan(net3M, None, netGridS, "sagittal")
print("sagittal stack", sag3M.shape)

# ---- figure: pre-processing -------------------------------------------------
slc = 45
nSlc = slc - sMin
fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
kw = dict(cmap="gray", vmin=-400, vmax=600)
axs[0].imshow(scan3M[:, :, slc], extent=sliceExtent(gridS[0], gridS[1]), **kw)
axs[0].set_title("Original, %d x %d" % scan3M.shape[:2])
axs[1].imshow(pad3M[:, :, slc], extent=sliceExtent(padGridS[0], padGridS[1]), **kw)
axs[1].set_title("padorcrop3d to 512 x 512")
axs[2].imshow(net3M[:, :, nSlc], extent=sliceExtent(netGridS[0], netGridS[1]), **kw)
axs[2].set_title("Crop to body, bilinear to 256 x 256")
for ax in axs[:3]:
    ax.contour(gridS[0], gridS[1], body3M[:, :, slc].astype(float), levels=[0.5],
               colors=["#40ff60"], linewidths=0.8)
    ax.set_xlim(gridS[0][0], gridS[0][-1])
    ax.set_ylim(gridS[1][-1], gridS[1][0])
    ax.set_xlabel("x (cm)")
axs[0].set_ylabel("y (cm)")
axs[3].imshow(sag3M[:, :, sag3M.shape[2] // 2], cmap="gray", vmin=-400, vmax=600,
              aspect="auto")
axs[3].set_title("transformScan(..., 'sagittal')")
axs[3].set_xticks([])
axs[3].set_yticks([])
fig.tight_layout()
save(fig, "ai_preprocessing")

# ============================================================================
# Post-processing
# ============================================================================
planC = loadPhantom()
scanNum, gtvNum = 0, 0
scan3M = planC.scan[scanNum].getScanArray()
gridS = planC.scan[scanNum].getScanXYZVals()
gtv3M = rs.getStrMask(gtvNum, planC)

# The model sees a 128 x 128 version of the scan
net3M, netMask4M, netGridS = image_proc.resizeScanAndMask(scan3M, gtv3M[..., None], gridS,
                                                          [128, 128, scan3M.shape[2]],
                                                          "bilinear")
planC = pc.importScanArray(net3M, *netGridS, "CT", scanNum, planC)
netScanNum = len(planC.scan) - 1

# Stand-in for the model output: the GTV on that grid, with false positives
# elsewhere and a hole in the middle
rng = np.random.default_rng(1)
pred3M = netMask4M[..., 0] > 0
for _ in range(8):
    r, c, k = rng.integers(8, 120), rng.integers(8, 120), rng.integers(2, 58)
    pred3M[r - 2:r + 3, c - 2:c + 3, k - 1:k + 2] = True
rowV, colV, slcV = np.where(pred3M)
rC, cC, sC = int(rowV.mean()), int(colV.mean()), int(slcV.mean())
pred3M[rC - 4:rC + 4, cC - 4:cC + 4, sC - 2:sC + 3] = False

planC = pc.importStructureMask(pred3M, netScanNum, "AI_GTV", planC)
aiNum = len(planC.structure) - 1
planC = cerrStr.copyToScan(aiNum, scanNum, planC)              # back to the original grid
copyNum = len(planC.structure) - 1
_, planC = cerrStr.getLargestConnComps(copyNum, 1, planC=planC, saveFlag=True,
                                       procSructName="AI_GTV_largest")
lccNum = len(planC.structure) - 1
_, planC = cerrStr.getClosedMask(lccNum, 1.0, planC, saveFlag=True,
                                 procSructName="AI_GTV_final")
finalNum = len(planC.structure) - 1
print([s.structureName for s in planC.structure])


def dice(a3M, b3M):
    return 2 * (a3M & b3M).sum() / (a3M.sum() + b3M.sum())


for n in (copyNum, lccNum, finalNum):
    m3M = rs.getStrMask(n, planC)
    print("%-16s Dice vs GTV %.3f  components %d" % (
        planC.structure[n].structureName, dice(m3M, gtv3M), ndimage.label(m3M)[1]))

# ---- figure: post-processing ------------------------------------------------
slc = sC
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
ext = sliceExtent(gridS[0], gridS[1])
for ax, n, title in ((axs[0], copyNum, "Model output on the original grid"),
                     (axs[1], lccNum, "getLargestConnComps(..., 1)"),
                     (axs[2], finalNum, "getClosedMask(..., 1.0 cm)")):
    m3M = rs.getStrMask(n, planC)
    ax.imshow(scan3M[:, :, slc], cmap="gray", vmin=-160, vmax=240, extent=ext)
    ax.imshow(np.ma.masked_equal(m3M[:, :, slc], 0), cmap="autumn", alpha=0.5, extent=ext)
    ax.contour(gridS[0], gridS[1], gtv3M[:, :, slc].astype(float), levels=[0.5],
               colors=["#40e0ff"], linewidths=1)
    nComp = ndimage.label(m3M)[1]
    ax.set_title("%s\nDice %.3f, %d component%s" % (title, dice(m3M, gtv3M), nComp,
                                                    "" if nComp == 1 else "s"), fontsize=9)
    ax.set_xlabel("x (cm)")
axs[0].set_ylabel("y (cm)")
fig.tight_layout()
save(fig, "ai_postprocessing")
