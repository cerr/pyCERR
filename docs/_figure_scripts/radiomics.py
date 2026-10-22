"""Figures and printed output for docs/user_guide/radiomics.rst."""
import collections
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from common import DATA_DIR, contourMask, loadPhantom, save, showAxial
from cerr.contour import rasterseg as rs
from cerr.radiomics import ibsi1, texture_utils

planC = loadPhantom()
scanNum, structNum = 0, 0
settingsFile = os.path.join(DATA_DIR, "radiomics_settings", "original_settings.json")

featDict, diagDict = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)
print(len(featDict))
print(diagDict)
classes = collections.Counter(k.split("_")[1] for k in featDict if k.startswith("original_"))
print(classes)
for key in ["original_shape_volume", "original_shape_surfArea", "original_shape_sphericity",
            "original_firstOrder_mean", "original_firstOrder_std", "original_firstOrder_entropy"]:
    print(key, featDict.get(key))
print([k for k in featDict if "shape" in k])
print([k for k in featDict if "firstOrder" in k])
print([k for k in featDict if "glcm" in k][:6])
print([k for k in featDict if "glszm" in k][:4])

with tempfile.TemporaryDirectory() as d:
    csvFile = os.path.join(d, "features.csv")
    ibsi1.writeFeaturesToFile(featDict, csvFile)
    print(open(csvFile).read()[:160])

# ---- Figure 1: ROI, intensities and what the settings do to them -----------
scan3M = planC.scan[scanNum].getScanArray()
mask3M = rs.getStrMask(structNum, planC)
slc = int(np.round(np.where(mask3M)[2].mean()))
roiV = scan3M[mask3M]
lo, hi, binWidth = -1000, 300, 5

fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1, 1.15, 1.1]})
showAxial(axs[0], planC, scanNum, slc, window=(-1000, 400))
contourMask(axs[0], planC, scanNum, mask3M, slc, "#ff5050", "GTV-1")
axs[0].legend(loc="lower right", fontsize=8)
axs[0].set_title("ROI on the CT scan")

axs[1].hist(roiV, bins=np.arange(-1050, 500, 25), color="#9ecae1", ec="white", lw=0.2)
axs[1].axvspan(lo, hi, color="#2ca02c", alpha=0.10, label="kept: [%d, %d] HU" % (lo, hi))
axs[1].axvline(lo, color="#2ca02c", lw=1)
axs[1].axvline(hi, color="#2ca02c", lw=1)
axs[1].set_yscale("log")
axs[1].set_xlabel("Intensity (HU)")
axs[1].set_ylabel("Voxels in ROI")
axs[1].set_title("Re-segmentation range")
axs[1].legend(fontsize=8, loc="upper left")

names = ["shape", "firstOrder", "glcm", "glrlm", "glszm", "gldm", "gtdm"]
axs[2].barh(names[::-1], [classes[n] for n in names[::-1]], color="#6baed6")
for i, n in enumerate(names[::-1]):
    axs[2].text(classes[n] + 2, i, str(classes[n]), va="center", fontsize=8)
axs[2].set_xlim(0, max(classes.values()) * 1.15)
axs[2].set_xlabel("Number of values returned")
axs[2].set_title("Features per class (original image)")
fig.tight_layout()
save(fig, "radiomics_roi_and_features")

# ---- Figure 2: filter-response (texture) maps ------------------------------
filters = [("LoG_filter.json", "LoG  (σ = 1.5 mm)"),
           ("mean_filter.json", "Mean  (5×5×5)"),
           ("Rot_inv_laws_energy_filter.json", "Laws energy  (S5S5S5)"),
           ("gabor_filter.json", "Gabor  (λ = 1.4 mm)")]
maps = []
for fileName, label in filters:
    cfg = os.path.join(DATA_DIR, "convolutional_filter_settings", fileName)
    planC = texture_utils.generateTextureMapFromPlanC(planC, scanNum, structNum, cfg)
    texScanNum = len(planC.scan) - 1
    tex3M = planC.scan[texScanNum].getScanArray()
    print(label, texScanNum, planC.scan[texScanNum].scanInfo[0].imageType, tex3M.shape,
          float(tex3M.min()), float(tex3M.max()))
    maps.append((texScanNum, label))

zV = planC.scan[scanNum].getScanXYZVals()[2]
fig, axs = plt.subplots(1, 5, figsize=(14, 3.3))
texX, texY, texZ = planC.scan[maps[0][0]].getScanXYZVals()
showAxial(axs[0], planC, scanNum, slc)
contourMask(axs[0], planC, scanNum, mask3M, slc, "#ff5050")
axs[0].set_xlim(texX.min(), texX.max())
axs[0].set_ylim(texY.min(), texY.max())
axs[0].set_title("CT (cropped to ROI + margin)")
for ax, (texScanNum, label) in zip(axs[1:], maps):
    tex3M = planC.scan[texScanNum].getScanArray()
    tzV = planC.scan[texScanNum].getScanXYZVals()[2]
    tSlc = int(np.argmin(abs(tzV - zV[slc])))
    vals = tex3M[:, :, tSlc]
    lim = np.percentile(vals, [2, 98])
    im = showAxial(ax, planC, texScanNum, tSlc, window=lim, cmap="magma")
    contourMask(ax, planC, scanNum, mask3M, slc, "#40e0ff", lw=1)
    ax.set_xlim(texX.min(), texX.max())
    ax.set_ylim(texY.min(), texY.max())
    ax.set_title(label)
    ax.set_ylabel("")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
fig.tight_layout()
save(fig, "radiomics_texture_maps")
