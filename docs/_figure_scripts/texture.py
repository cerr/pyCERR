"""Figures and printed output for docs/user_guide/texture.rst."""
import json
import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np

from common import loadPhantom, save
from cerr.contour import rasterseg as rs
from cerr.radiomics import ibsi1, texture_utils
from cerr.utils.mask import computeBoundingBox

planC = loadPhantom()
scanNum, structNum = 0, 0
scan3M = planC.scan[scanNum].getScanArray().astype(float)
mask3M = rs.getStrMask(structNum, planC)
voxelSizeMm = planC.scan[scanNum].getScanSpacing() * 10        # filters work in mm

# Crop to the structure plus a margin to keep the filtering fast
rMin, rMax, cMin, cMax, sMin, sMax, _ = computeBoundingBox(mask3M)
pad = 10
crop = (slice(rMin - pad, rMax + pad), slice(cMin - pad, cMax + pad), slice(sMin - 3, sMax + 3))
cropScan3M, cropMask3M = scan3M[crop], mask3M[crop]
print(cropScan3M.shape, voxelSizeMm)

filters = {
    "Mean": {"KernelSize": [5, 5, 5]},
    "LoG": {"Sigma_mm": [3, 3, 3], "CutOff_mm": [12, 12, 12],
            "VoxelSize_mm": voxelSizeMm},
    "Laws": {"Direction": "3d", "Type": "E5L5S5", "Normalize": "yes"},
    "RotationInvariantLawsEnergy": {
        "Direction": "3d", "Type": "E5L5S5", "Normalize": "yes",
        "RotationInvariance": {"Dim": "3D", "AggregationMethod": "max"},
        "EnergyKernelSize": [7, 7, 7], "EnergyPadMethod": "mirror", "EnergyPadSize": [3, 3, 3],
        "Padding": {"Flag": True, "Size": [2, 2, 2], "Method": "mirror"}},
    "Gabor": {"Sigma_mm": 2.1, "Wavlength_mm": 1.4, "SpatialAspectRatio": 1.5,
              "Orientation": [60, 120, 180, 240, 300, 360], "OrientationAggregation": "average",
              "Radius_mm": 9.1, "VoxelSize_mm": voxelSizeMm},
    "Wavelets": {"Direction": "LHL", "Wavelets": "db", "Index": "2", "Level": 1},
    "RotationInvariantWavelets": {
        "Direction": "HHL", "Wavelets": "coif", "Index": "1", "Level": 1,
        "RotationInvariance": {"Dim": "3D", "AggregationMethod": "avg"}},
}
responses = {}
for filterType, paramS in filters.items():
    t0 = time.time()
    outS = texture_utils.processImage(filterType, cropScan3M, cropMask3M, paramS)
    for key, resp3M in outS.items():
        responses["%s: %s" % (filterType, key)] = resp3M
        print("%-28s %-16s %s  %.1f s" % (filterType, key, resp3M.shape, time.time() - t0))

# ---- features from a filtered image ------------------------------------------
settings = {
    "imageType": {"LoG": {"Sigma_mm": [3, 3, 3], "CutOff_mm": [12, 12, 12]}},
    "settings": {"padding": {"method": "mirror", "size": [6, 6, 6]},
                 "firstOrder": {"offsetForEnergy": 0}},
    "featureClass": {"firstOrder": {"featureList": ["all"]}},
}
with tempfile.TemporaryDirectory() as d:
    settingsFile = os.path.join(d, "log_features.json")
    with open(settingsFile, "w") as f:
        json.dump(settings, f)
    featDict, _ = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)
logKeys = [k for k in featDict if "firstOrder" in k]
print(len(logKeys), logKeys[:3])
print(logKeys[0].rsplit("_firstOrder", 1)[0])
for k in ("mean", "std", "skewness", "kurtosis"):
    key = [x for x in logKeys if x.endswith("_firstOrder_" + k)][0]
    print(key, featDict[key])

# ---- figure ------------------------------------------------------------------
mid = int(np.round(np.where(cropMask3M)[2].mean()))
panels = [("CT", cropScan3M)] + list(responses.items())
fig, axs = plt.subplots(2, 4, figsize=(14, 7.2))
for ax, (title, vol3M) in zip(axs.ravel(), panels):
    img = vol3M[:, :, mid]
    if title == "CT":
        im = ax.imshow(img, cmap="gray", vmin=-160, vmax=240)
    else:
        lo, hi = np.percentile(img, [2, 98])
        im = ax.imshow(img, cmap="magma", vmin=lo, vmax=hi)
    ax.contour(cropMask3M[:, :, mid], levels=[0.5], colors=["#40e0ff"], linewidths=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
fig.tight_layout()
save(fig, "texture_filter_gallery")
