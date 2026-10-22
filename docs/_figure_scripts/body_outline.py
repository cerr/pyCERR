"""Figures and printed output for docs/user_guide/body_outline.rst."""
import time

import matplotlib.pyplot as plt
import numpy as np

from common import save
from cerr import datasets
from cerr import plan_container as pc
from cerr.utils import mask as maskUtils

planC = pc.loadDcmDir(datasets.fetch_sample_data("head_and_neck"))
scanNum = 0
scan3M = planC.scan[scanNum].getScanArray()
print(scan3M.shape, planC.scan[scanNum].getScanSpacing())

# Step 1: couch detection on the maximum-intensity projection
couchRow, couchLines = maskUtils.getCouchLocationHough(scan3M)
print("couch row", couchRow, "candidate lines", len(couchLines))

# Step 2: the outline
t0 = time.time()
outline3M = maskUtils.getPatientOutline(scan3M, outThreshold=-400)
print("outline seconds %.1f" % (time.time() - t0))
planC = pc.importStructureMask(outline3M, scanNum, "Body", planC)
voxVol = np.prod(planC.scan[scanNum].getScanSpacing())
print("Body volume cc %.0f" % (outline3M.sum() * voxVol))

air3M = scan3M > -400
print("threshold-only voxels %d, outline voxels %d" % (air3M.sum(), outline3M.sum()))

# ---- figure ------------------------------------------------------------------
slc = 45
col = scan3M.shape[1] // 2
mip = scan3M.max(axis=2)
fig, axs = plt.subplots(1, 4, figsize=(15, 4.4),
                        gridspec_kw={"width_ratios": [1, 1, 1, 0.55]})
kw = dict(cmap="gray", vmin=-1000, vmax=1000)
axs[0].imshow(mip, **kw)
for ln in couchLines:
    axs[0].plot([ln["point1"][0], ln["point2"][0]], [ln["point1"][1], ln["point2"][1]],
                color="#ffd000", lw=1)
axs[0].axhline(couchRow, color="#ff4040", ls="--", lw=1.5)
axs[0].plot([], [], color="#ffd000", label="Hough line candidates")
axs[0].plot([], [], color="#ff4040", ls="--", label="couch row %d" % couchRow)
axs[0].legend(loc="upper right", fontsize=7)
axs[0].set_title("1. Couch search on the projection")

axs[1].imshow(scan3M[:, :, slc], **kw)
axs[1].contour(air3M[:, :, slc], levels=[0.5], colors=["#40e0ff"], linewidths=1)
axs[1].set_title("Threshold only (> -400 HU)")

axs[2].imshow(scan3M[:, :, slc], **kw)
axs[2].contour(outline3M[:, :, slc], levels=[0.5], colors=["#40ff60"], linewidths=1.5)
axs[2].set_title("2. getPatientOutline")

sag = scan3M[:, col, :].T                      # slices down, rows across
axs[3].imshow(sag, aspect=3.0, **kw)
axs[3].contour(outline3M[:, col, :].T, levels=[0.5], colors=["#40ff60"],
               linewidths=1.2)
axs[3].set_title("Outline, sagittal")
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
save(fig, "body_outline_couch")
