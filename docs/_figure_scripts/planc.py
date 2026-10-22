"""Figures for docs/user_guide/planc.rst: data-model diagram and coordinates."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import contourMask, loadPhantom, save, showAxial
from cerr.contour import rasterseg as rs

# --------------------------------------------------------------------------
# 1. planC data-model diagram
# --------------------------------------------------------------------------
BLUE, GREEN, ORANGE, GREY, PURPLE = "#dbe9f6", "#dff0d8", "#fde8cf", "#eeeeee", "#e8def8"


def box(ax, xy, w, h, title, lines, color):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#444444", lw=1))
    ax.text(xy[0] + w / 2, xy[1] + h - 0.22, title, ha="center", va="top",
            fontsize=10, fontweight="bold", family="monospace")
    ax.text(xy[0] + 0.12, xy[1] + h - 0.62, "\n".join(lines), ha="left", va="top",
            fontsize=7.5, family="monospace", linespacing=1.35)


def arrow(ax, p0, p1, text=None, rad=0.0, color="#444444", ls="-", textOffset=(0, 0.1)):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=1.1,
                                 color=color, linestyle=ls,
                                 connectionstyle="arc3,rad=%g" % rad))
    if text:
        ax.text((p0[0] + p1[0]) / 2 + textOffset[0], (p0[1] + p1[1]) / 2 + textOffset[1],
                text, ha="center", va="bottom", fontsize=7, color=color, style="italic")


fig, ax = plt.subplots(figsize=(10.5, 5.6))
ax.set_xlim(0, 15)
ax.set_ylim(-0.9, 8)
ax.axis("off")

box(ax, (5.9, 6.3), 3.2, 1.5, "PlanC",
    ["header   Header", "one list per object type"], GREY)

w, h, y = 2.75, 3.6, 1.6
xs = [0.1, 3.1, 6.1, 9.1, 12.1]
box(ax, (xs[0], y), w, h, "planC.scan[i]",
    ["Scan", "scanArray (r,c,s)", "scanInfo[slice]", "scanUID",
     "Image2PhysicalTransM", "getScanArray()", "getScanXYZVals()", "saveNii()"], BLUE)
box(ax, (xs[1], y), w, h, "planC.structure[i]",
    ["Structure", "structureName", "contour[slice]", "rasterSegments", "strUID",
     "assocScanUID", "getContourPolygons()", "saveNii()"], GREEN)
box(ax, (xs[2], y), w, h, "planC.dose[i]",
    ["Dose", "doseArray (r,c,s)", "doseUnits", "fractionGroupID", "doseUID",
     "assocScanUID", "getDoseXYZVals()", "getDoseAt(x,y,z)"], ORANGE)
box(ax, (xs[3], y), w, h, "planC.beams[i]",
    ["Beams (RTPLAN)", "RTPlanLabel", "BeamSequence", "FractionGroup-", "  Sequence",
     "SOPInstanceUID"], GREY)
box(ax, (xs[4], y), w, h, "planC.deform[i]",
    ["Deform", "baseScanUID", "movScanUID", "registrationTool", "algorithm",
     "deformOutFilePath", "dvfMatrix", "deformUID"], PURPLE)

for x in xs:
    arrow(ax, (7.5, 6.3), (x + w / 2, y + h))

# UID associations (drawn below the boxes)
link = "#b03030"
for src, label, rad, dx in ((1, "assocScanUID", -0.5, 0.45), (2, "assocScanUID", -0.4, 0.0),
                            (4, "baseScanUID, movScanUID", -0.3, -0.45)):
    xSrc = xs[src] + w / 2
    ax.text(xSrc, y - 0.12, label, ha="center", va="top", fontsize=7.5, color=link,
            style="italic")
    arrow(ax, (xSrc, y - 0.42), (xs[0] + w / 2 + dx, y), rad=rad, color=link, ls="--")
save(fig, "planc_data_model")

# --------------------------------------------------------------------------
# 2. Array indices versus virtual coordinates
# --------------------------------------------------------------------------
planC = loadPhantom()
scan3M = planC.scan[0].getScanArray()
xV, yV, zV = planC.scan[0].getScanXYZVals()
mask3M = rs.getStrMask(0, planC)
slc = int(np.round(np.where(mask3M)[2].mean()))
print("slice", slc, "x", xV[[0, -1]], "y", yV[[0, -1]], "z", zV[[0, -1]])

fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
axs[0].imshow(scan3M[:, :, slc], cmap="gray", vmin=-1000, vmax=400)
axs[0].contour(mask3M[:, :, slc].astype(float), levels=[0.5], colors=["#ff5050"])
axs[0].set_xlabel("column index  →  x")
axs[0].set_ylabel("row index  →  −y")
axs[0].set_title("Array indices: scan3M[row, col, slice]")

showAxial(axs[1], planC, 0, slc, window=(-1000, 400))
contourMask(axs[1], planC, 0, mask3M, slc, "#ff5050", "GTV-1")
axs[1].set_title("Virtual coordinates (cm): xV, yV from getScanXYZVals()")
axs[1].legend(loc="lower right", fontsize=8)
fig.tight_layout()
save(fig, "planc_coordinates")
