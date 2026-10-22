"""Figures and printed output for docs/user_guide/structure_consensus.rst.

The bundled phantom has a single GTV contour, so four "observers" are derived
from it: copies that are shifted by a few millimetres and drawn 3 mm larger or
smaller, and one copy shifted by about 1.4 cm that plays the outlier.
"""
import matplotlib.pyplot as plt
import numpy as np

from common import contourMask, loadPhantom, save, showAxial
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.contour import structure_consensus as sc
from cerr.dataclasses import structure as cerrStr

planC = loadPhantom()
scanNum, gtvNum = 0, 0
gtv3M = rs.getStrMask(gtvNum, planC)
planC = cerrStr.getSurfaceExpand(gtvNum, 0.3, planC)        # GTV + 3 mm
large3M = rs.getStrMask(len(planC.structure) - 1, planC)
planC = cerrStr.getSurfaceExpand(gtvNum, -0.3, planC)       # GTV - 3 mm
small3M = rs.getStrMask(len(planC.structure) - 1, planC)
del planC.structure[1:]                                     # keep only the GTV

# Each observer differs from the GTV in size and position
observers = {"Observer_A": np.roll(gtv3M, (-4, 3), axis=(0, 1)),
             "Observer_B": np.roll(large3M, (2, -3), axis=(0, 1)),
             "Observer_C": np.roll(small3M, (3, 2), axis=(0, 1)),
             "Observer_D": np.roll(gtv3M, (10, 10), axis=(0, 1))}   # outlier
for name, mask3M in observers.items():
    planC = pc.importStructureMask(mask3M, scanNum, name, planC)
del planC.structure[0]                                      # drop the original GTV
observerNums = [0, 1, 2, 3]
print([s.structureName for s in planC.structure])

result = sc.compareStructures(observerNums, planC)
print(sc.summaryText(result))
for name, se, sp in zip(result["structNames"], result["sensitivity"], result["specificity"]):
    print("  %-11s sensitivity %.3f  specificity %.4f" % (name, se, sp))

consensusNums = {}
for method, thr in (("staple", 0.5), ("majority", None), ("agreement", 0.75),
                    ("union", None), ("intersection", None)):
    kw = {} if thr is None else {"threshold": thr}
    planC, newNum = sc.createConsensusStructure(observerNums, planC, method=method,
                                                result=result, **kw)
    consensusNums[method] = newNum
    vol = rs.getStrMask(newNum, planC).sum() * result["voxelVolume_cc"]
    name = planC.structure[newNum].structureName
    print("%-12s -> %d %-24s %7.1f cc" % (method, newNum, name, vol))

# ---- figure 1: observers and consensus ---------------------------------------
masks = {n: rs.getStrMask(n, planC) for n in observerNums}
slc = int(np.round(np.where(masks[0])[2].mean()))
colors = ["#ff4040", "#ffd000", "#40e0ff", "#ff60ff"]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
showAxial(axs[0], planC, scanNum, slc)
for n, c in zip(observerNums, colors):
    contourMask(axs[0], planC, scanNum, masks[n], slc, c, planC.structure[n].structureName)
axs[0].set_title("Four observers")
axs[0].legend(loc="lower left", fontsize=7)

xV, yV, _ = planC.scan[scanNum].getScanXYZVals()
showAxial(axs[1], planC, scanNum, slc)
extent = axs[1].images[0].get_extent()
im = axs[1].imshow(np.ma.masked_less(result["staple3M"][:, :, slc], 0.01), cmap="viridis",
                   vmin=0, vmax=1, alpha=0.8, extent=extent)
fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03)
axs[1].set_title("STAPLE probability")

showAxial(axs[2], planC, scanNum, slc)
for method, c in (("union", "#888888"), ("staple", "#40ff60"), ("intersection", "#ff4040")):
    contourMask(axs[2], planC, scanNum, rs.getStrMask(consensusNums[method], planC), slc, c,
                planC.structure[consensusNums[method]].structureName)
axs[2].set_title("Consensus contours")
axs[2].legend(loc="lower left", fontsize=7)
for ax in axs:
    ax.set_xlim(-13.5, 0)
    ax.set_ylim(-8, 4.5)
fig.tight_layout()
save(fig, "consensus_observers")

# ---- figure 2: volume against agreement threshold ------------------------------
fig, ax = plt.subplots(figsize=(5.8, 3.4))
thr = result["agreementThresholds"]
ax.plot(thr, result["volumeByAgreement_cc"], "o-", color="#1f77b4",
        label="agreement fraction ≥ threshold")
ax.plot(thr, result["stapleVolumeByThreshold_cc"], "s--", color="#2ca02c",
        label="STAPLE probability ≥ threshold")
ax.set_xticks(thr)
ax.set_xticklabels(["%d/4" % k for k in range(1, 5)])
ax.set_xlabel("Threshold")
ax.set_ylabel("Volume (cc)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Consensus volume by threshold")
fig.tight_layout()
save(fig, "consensus_volume_by_threshold")
