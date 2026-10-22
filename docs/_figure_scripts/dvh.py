"""Figures and printed output for docs/user_guide/dvh.rst."""
import matplotlib.pyplot as plt
import numpy as np

from common import addSyntheticDose, contourMask, loadPhantom, save, showAxial
from cerr import dvh
from cerr.contour import rasterseg as rs
from cerr.dataclasses import structure as cerrStr

planC = loadPhantom()
# A 1 cm shell of tissue around the GTV serves as a stand-in organ at risk.
planC = cerrStr.getSurfaceExpand(0, 1.0, planC)
planC = cerrStr.structDiff(1, 0, planC, "Ring_1cm")
planC = addSyntheticDose(planC)
names = [s.structureName for s in planC.structure]
print(names)

doseNum = 0
binWidth = 0.05
colors = {0: "#d62728", 2: "#1f77b4"}
curves = {}
for structNum in (0, 2):
    dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)
    doseBinsV, volsHistV = dvh.doseHist(dosesV, volsV, binWidth)
    curves[structNum] = (doseBinsV, volsHistV)
    print(names[structNum], "isErr", isErr, "nvox", len(dosesV),
          "volume cc %.1f" % volsV.sum())
    print("  mean %.2f  min %.2f  max %.2f  median %.2f" % (
        dvh.meanDose(doseBinsV, volsHistV), dvh.minDose(doseBinsV, volsHistV),
        dvh.maxDose(doseBinsV, volsHistV), dvh.medianDose(doseBinsV, volsHistV)))
    print("  D95 %.2f  D2 %.2f  V50 %.1f%%  V50 %.1f cc" % (
        dvh.Dx(doseBinsV, volsHistV, 95, 1), dvh.Dx(doseBinsV, volsHistV, 2, 1),
        100 * dvh.Vx(doseBinsV, volsHistV, 50, 1), dvh.Vx(doseBinsV, volsHistV, 50, 0)))
    print("  MOH5 %.2f  MOC5 %.2f  gEUD(a=-10) %.2f  gEUD(a=8) %.2f" % (
        dvh.MOHx(doseBinsV, volsHistV, 5), dvh.MOCx(doseBinsV, volsHistV, 5),
        dvh.eud(doseBinsV, volsHistV, -10), dvh.eud(doseBinsV, volsHistV, 8)))

# ---- dose colourwash + DVH -------------------------------------------------
gtv3M = rs.getStrMask(0, planC)
ring3M = rs.getStrMask(2, planC)
slc = int(np.round(np.where(gtv3M)[2].mean()))
xV, yV, zV = planC.scan[0].getScanXYZVals()
dose3M = planC.dose[doseNum].doseArray

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})
showAxial(axs[0], planC, 0, slc)
dxV, dyV, dzV = planC.dose[doseNum].getDoseXYZVals()
dSlc = int(np.argmin(abs(dzV - zV[slc])))
im = axs[0].imshow(np.ma.masked_less(dose3M[:, :, dSlc], 3), cmap="jet", alpha=0.45,
                   vmin=0, vmax=60, extent=axs[0].images[0].get_extent())
contourMask(axs[0], planC, 0, gtv3M, slc, colors[0], "GTV-1")
contourMask(axs[0], planC, 0, ring3M, slc, colors[2], "Ring_1cm")
axs[0].legend(loc="lower right", fontsize=8)
axs[0].set_title("Synthetic dose (Gy), axial slice %d" % slc)
fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03)

for structNum, (doseBinsV, volsHistV) in curves.items():
    cumVolsV = np.cumsum(volsHistV[::-1])[::-1] / volsHistV.sum() * 100
    axs[1].plot(doseBinsV, cumVolsV, color=colors[structNum], lw=2, label=names[structNum])
doseBinsV, volsHistV = curves[0]
d95 = dvh.Dx(doseBinsV, volsHistV, 95, 1)
axs[1].plot([0, d95, d95], [95, 95, 0], color=colors[0], ls=":", lw=1)
axs[1].annotate("D95 = %.1f Gy" % d95, (d95, 95), xytext=(d95 - 24, 70), fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
doseBinsV, volsHistV = curves[2]
v50 = 100 * dvh.Vx(doseBinsV, volsHistV, 50, 1)
axs[1].plot([50, 50, 0], [0, v50, v50], color=colors[2], ls=":", lw=1)
axs[1].annotate("V50 = %.0f%%" % v50, (50, v50), xytext=(22, v50 + 14), fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
axs[1].set_xlabel("Dose (Gy)")
axs[1].set_ylabel("Volume (%)")
axs[1].set_xlim(0, 65)
axs[1].set_ylim(0, 102)
axs[1].grid(alpha=0.3)
axs[1].legend(loc="lower left", fontsize=8)
axs[1].set_title("Cumulative DVH")
fig.tight_layout()
save(fig, "dvh_dose_and_curves")

# ---- differential histogram + MOH / MOC ------------------------------------
doseBinsV, volsHistV = curves[2]
coarseBinsV, coarseHistV = dvh.doseHist(*dvh.getDVH(2, doseNum, planC)[:2], 1.0)
moh = dvh.MOHx(doseBinsV, volsHistV, 10)
moc = dvh.MOCx(doseBinsV, volsHistV, 10)
d10 = dvh.Dx(doseBinsV, volsHistV, 10, 1)
d90 = dvh.Dx(doseBinsV, volsHistV, 90, 1)
print("ring MOH10 %.2f MOC10 %.2f D10 %.2f D90 %.2f" % (moh, moc, d10, d90))
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.bar(coarseBinsV, coarseHistV, width=1.0, color="#9ecae1", ec="white", lw=0.3)
ax.bar(coarseBinsV[coarseBinsV >= d10], coarseHistV[coarseBinsV >= d10], width=1.0,
       color="#d62728", ec="white", lw=0.3, label="hottest 10%% → MOH10 = %.1f Gy" % moh)
ax.bar(coarseBinsV[coarseBinsV <= d90], coarseHistV[coarseBinsV <= d90], width=1.0,
       color="#08519c", ec="white", lw=0.3, label="coldest 10%% → MOC10 = %.1f Gy" % moc)
ax.axvline(dvh.meanDose(doseBinsV, volsHistV), color="k", ls="--", lw=1,
           label="mean = %.1f Gy" % dvh.meanDose(doseBinsV, volsHistV))
ax.set_xlabel("Dose (Gy)")
ax.set_ylabel("Volume per 1 Gy bin (cc)")
ax.set_title("Differential DVH of Ring_1cm (doseHist, binWidth = 1 Gy)")
ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "dvh_differential")
