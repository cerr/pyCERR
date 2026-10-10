"""Figures and printed output for docs/user_guide/roe.rst."""
import copy
import json

import matplotlib.pyplot as plt
import numpy as np

from common import addSyntheticDose, loadPhantom, save
from cerr import plan_container as pc
from cerr.dataclasses import structure as cerrStr
from cerr.roe import dosimetric_models as roe

planC = loadPhantom()
planC = cerrStr.getSurfaceExpand(0, 1.0, planC)
planC = cerrStr.structDiff(1, 0, planC, "Ring_1cm")
planC = addSyntheticDose(planC)
doseNum = 0
numFractions = 30                      # the synthetic plan: 30 x 2 Gy

print(roe.listModels())

# ---- one model, structure name mapped to a structure in planC --------------
with open(roe.mapModelToFile("Esophagitis (Huang)")) as f:
    huang = json.load(f)
huang["parameters"]["structures"] = {
    "Ring_1cm": huang["parameters"]["structures"]["Esophagus"]}
huang["parameters"]["concurrentChemo"]["val"] = 1
ntcp = roe.run(huang, doseNum, planC, fNumIn=numFractions)
print("Huang, chemo=1: %.4f" % ntcp)
noChemo = copy.deepcopy(huang)
noChemo["parameters"]["concurrentChemo"]["val"] = 0
print("Huang, chemo=0: %.4f" % roe.run(noChemo, doseNum, planC, fNumIn=numFractions))

# ---- LKB model -------------------------------------------------------------
with open(roe.mapModelToFile("Rectal bleeding (grade 2+)")) as f:
    lkb = json.load(f)
lkb["parameters"]["structures"] = {
    "Ring_1cm": lkb["parameters"]["structures"]["Rectum"]}
print("LKB: %.4f" % roe.run(lkb, doseNum, planC, fSizeIn=2.0))

# ---- from pre-computed predictors ------------------------------------------
print("fromPredictors: %.4f" % roe.runFromPredictors(
    "Esophagitis (Huang)", {"Esophagus meanDose": 30.0, "concurrentChemo": 1}))

# ---- NTCP versus plan scaling ----------------------------------------------
xV, yV, zV = planC.dose[doseNum].getDoseXYZVals()
dose3M = planC.dose[doseNum].doseArray
scaleV = np.linspace(0.5, 1.5, 11)
curves = {"Esophagitis (Huang), concurrent chemo": (huang, {"fNumIn": numFractions}),
          "Esophagitis (Huang), no chemo": (noChemo, {"fNumIn": numFractions}),
          "Rectal bleeding (LKB)": (lkb, None)}
ntcpD = {name: [] for name in curves}
for scale in scaleV:
    planC = pc.importDoseArray(dose3M * scale, xV, yV, zV, planC, 0,
                               {"fractionGroupID": "scaled"})
    for name, (model, kw) in curves.items():
        kw = kw if kw is not None else {"fSizeIn": 2.0 * scale}
        ntcpD[name].append(roe.run(model, len(planC.dose) - 1, planC, **kw))
    del planC.dose[-1]
for name in ntcpD:
    print(name, np.round(ntcpD[name], 3))

fig, ax = plt.subplots(figsize=(6.6, 3.9))
styles = [("#d62728", "-"), ("#d62728", "--"), ("#1f77b4", "-")]
for (name, vals), (color, ls) in zip(ntcpD.items(), styles):
    ax.plot(scaleV * 60, vals, color=color, ls=ls, lw=2, marker="o", ms=3.5, label=name)
ax.axvline(60, color="k", lw=0.8, ls=":")
ax.text(60.4, 0.03, "as planned", fontsize=8)
ax.set_xlabel("Prescription dose after scaling (Gy in 30 fractions)")
ax.set_ylabel("NTCP")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("NTCP as the plan is scaled (structure: Ring_1cm)")
fig.tight_layout()
save(fig, "roe_ntcp_vs_scale")

# ---- dose-response curves of the two functional forms ----------------------
meanV = np.linspace(0, 80, 161)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
for chemo, ls in ((1, "-"), (0, "--")):
    vals = [roe.runFromPredictors("Esophagitis (Huang)",
                                  {"Esophagus meanDose": d, "concurrentChemo": chemo})
            for d in meanV]
    axs[0].plot(meanV, vals, color="#d62728", ls=ls, lw=2,
                label="concurrentChemo = %d" % chemo)
axs[0].set_title("logitFn: Esophagitis (Huang)")
axs[0].set_xlabel("Esophagus mean dose (Gy, 35-fraction equivalent)")
axs[0].set_ylabel("NTCP")
axs[0].legend(fontsize=8)
D50, m = lkb["parameters"]["D50"]["val"], lkb["parameters"]["m"]["val"]
vals = [roe.runFromPredictors("Rectal bleeding (grade 2+)", {"Rectum gEUD": d})
        for d in meanV + 20]
axs[1].plot(meanV + 20, vals, color="#1f77b4", lw=2)
axs[1].axvline(D50, color="k", ls=":", lw=0.8)
axs[1].text(D50 + 0.7, 0.05, "D50 = %.1f Gy" % D50, fontsize=8)
axs[1].set_title("LKBFn: Rectal bleeding (grade 2+)")
axs[1].set_xlabel("Rectum gEUD, n = 0.09 (Gy, 2 Gy-fraction equivalent)")
for ax in axs:
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
fig.tight_layout()
save(fig, "roe_dose_response")
