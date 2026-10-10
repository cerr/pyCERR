"""Figures and printed output for docs/user_guide/segmentation.rst."""
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from common import contourMask, loadPhantom, save, showAxial
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.dataclasses import structure as cerrStr
from cerr.dcm_export import rtstruct_iod
from cerr.utils import mask as maskUtils

planC = loadPhantom()
scanNum, gtvNum = 0, 0
scan3M = planC.scan[scanNum].getScanArray()
gtv3M = rs.getStrMask(gtvNum, planC)
slc = int(np.round(np.where(gtv3M)[2].mean()))

# ---- 1. bring in masks made by "any algorithm" -----------------------------
# Here: a simple threshold segmentation of bone, cleaned up by keeping the
# largest connected components.
bone3M = maskUtils.largestConnComps(scan3M > 250, 8, minSize=500)
planC = pc.importStructureMask(bone3M, scanNum, "Bone", planC)
boneNum = len(planC.structure) - 1
print([s.structureName for s in planC.structure])
print("fileFormat", planC.structure[boneNum].structureFileFormat, "voxels", bone3M.sum())

# ---- 2. structure operations ------------------------------------------------
planC = cerrStr.getSurfaceExpand(gtvNum, 1.0, planC)          # GTV + 1 cm
expNum = len(planC.structure) - 1
planC = cerrStr.getSurfaceExpand(gtvNum, -0.5, planC)         # GTV - 0.5 cm
shrinkNum = len(planC.structure) - 1
planC = cerrStr.structDiff(expNum, gtvNum, planC, "Ring_1cm")
ringNum = len(planC.structure) - 1
planC = cerrStr.structIntersect([expNum, boneNum], planC, "Bone_near_GTV")
interNum = len(planC.structure) - 1
planC = cerrStr.structUnion([gtvNum, boneNum], planC, "GTV_or_bone")
unionNum = len(planC.structure) - 1
names = [s.structureName for s in planC.structure]
print(names)
voxVol = np.prod(planC.scan[scanNum].getScanSpacing())
for n in range(len(names)):
    print("%d %-22s %8.1f cc" % (n, names[n], rs.getStrMask(n, planC).sum() * voxVol))

# ---- 3. label map and export ------------------------------------------------
labelDict = {"GTV-1": 1, "Ring_1cm": 2, "Bone": 3}
labelMap3M, strNumV = cerrStr.getLabelMap(planC, labelDict)
print(labelMap3M.shape, labelMap3M.dtype, np.unique(labelMap3M), strNumV)
with tempfile.TemporaryDirectory() as d:
    pc.saveNiiStructure(os.path.join(d, "labels.nii.gz"), labelDict, planC,
                        strNumV=[gtvNum, ringNum, boneNum])
    planC.structure[ringNum].saveNii(os.path.join(d, "ring.nii.gz"), planC)
    rtstruct_iod.create([gtvNum, ringNum], os.path.join(d, "rtstruct.dcm"), planC,
                        {"seriesDescription": "pyCERR docs example"})
    print(sorted(os.listdir(d)))

# ---- figures ----------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.1))
showAxial(axs[0], planC, scanNum, slc, window=(-400, 600))
contourMask(axs[0], planC, scanNum, bone3M, slc, "#40e0ff", "Bone (threshold > 250 HU)")
contourMask(axs[0], planC, scanNum, gtv3M, slc, "#ff4040", "GTV-1 (RTSTRUCT)")
axs[0].legend(loc="lower left", fontsize=7)
axs[0].set_title("Mask imported with importStructureMask")

showAxial(axs[1], planC, scanNum, slc)
for n, color in ((expNum, "#40e0ff"), (gtvNum, "#ff4040"), (shrinkNum, "#ffd000")):
    contourMask(axs[1], planC, scanNum, rs.getStrMask(n, planC), slc, color, names[n])
axs[1].legend(loc="lower left", fontsize=7)
axs[1].set_title("getSurfaceExpand: +1 cm and −0.5 cm")

showAxial(axs[2], planC, scanNum, slc)
extent = axs[2].images[0].get_extent()
for n, color in ((ringNum, "#40e0ff"), (interNum, "#ffd000")):
    m = rs.getStrMask(n, planC)[:, :, slc]
    axs[2].imshow(np.ma.masked_equal(m, 0), cmap=ListedColormap([color]), alpha=0.55,
                  extent=extent)
    axs[2].plot([], [], color=color, lw=5, alpha=0.6, label=names[n])
contourMask(axs[2], planC, scanNum, gtv3M, slc, "#ff4040", "GTV-1")
axs[2].legend(loc="lower left", fontsize=7)
axs[2].set_title("structDiff and structIntersect")
for ax in axs[1:]:
    ax.set_xlim(-14.5, -0.5)
    ax.set_ylim(-9, 5.5)
fig.tight_layout()
save(fig, "segmentation_structure_ops")

fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(labelMap3M[:, :, slc], cmap=ListedColormap(["black", "#d1495b", "#3b6ea5", "#f2c14e"]),
               vmin=-0.5, vmax=3.5, extent=extent, interpolation="nearest")
cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.03)
cb.ax.set_yticklabels(["0 background", "1 GTV-1", "2 Ring_1cm", "3 Bone"], fontsize=8)
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_title("getLabelMap(planC, labelDict)")
fig.tight_layout()
save(fig, "segmentation_label_map")
