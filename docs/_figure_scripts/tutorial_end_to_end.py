"""Figures and printed output for docs/tutorials/end_to_end.rst."""
import json
import os
import tempfile

import numpy as np

from common import DATA_DIR, IMG_DIR, addSyntheticDose, loadPhantom
from cerr import dvh
from cerr import plan_container as pc
from cerr.dataclasses import structure as cerrStr
from cerr.radiomics import ibsi1
from cerr.roe import dosimetric_models as roe
from cerr.viewer import pycerr_nbviewer

# 1. import
planC = loadPhantom()
# 2. derive structures
planC = cerrStr.getSurfaceExpand(0, 1.0, planC)
planC = cerrStr.structDiff(1, 0, planC, "Ring_1cm")
# 3. dose
planC = addSyntheticDose(planC)
print([s.structureName for s in planC.structure], len(planC.dose))

# 4. view
viewer = pycerr_nbviewer.NbViewer(planC, scanNum=0, autoDisplay=False)
viewer.set_structures_visible([0, 2])
viewer.set_dose(0)
viewer.goto_structure(0)
viewer.save_screenshot(os.path.join(IMG_DIR, "tutorial_viewer.png"), dpi=80)
doseAxisV, dvhTable = viewer.compute_dvh(doseNum=0, structNums=[0, 2])
print(type(doseAxisV), list(dvhTable))

# 5. DVH metrics
rows = []
for structNum in (0, 2):
    dosesV, volsV, _ = dvh.getDVH(structNum, 0, planC)
    doseBinsV, volsHistV = dvh.doseHist(dosesV, volsV, 0.05)
    rows.append((planC.structure[structNum].structureName, volsV.sum(),
                 dvh.meanDose(doseBinsV, volsHistV), dvh.Dx(doseBinsV, volsHistV, 95, 1),
                 100 * dvh.Vx(doseBinsV, volsHistV, 50, 1)))
for r in rows:
    print("%-10s vol %.1f cc  mean %.1f  D95 %.1f  V50 %.1f%%" % r)

# 6. outcome model
with open(roe.mapModelToFile("Esophagitis (Huang)")) as f:
    model = json.load(f)
model["parameters"]["structures"] = {"Ring_1cm": model["parameters"]["structures"]["Esophagus"]}
model["parameters"]["concurrentChemo"]["val"] = 1
print("NTCP %.3f" % roe.run(model, 0, planC, fNumIn=30))

# 7. radiomics
settingsFile = os.path.join(DATA_DIR, "radiomics_settings", "original_settings.json")
featDict, _ = ibsi1.computeScalarFeatures(0, 0, settingsFile, planC)
print(len(featDict), featDict["original_firstOrder_mean"], featDict["original_firstOrder_entropy"])

# 8. save
with tempfile.TemporaryDirectory() as d:
    h5File = os.path.join(d, "phantom.h5")
    pc.saveToH5(planC, h5File)
    planC.structure[2].saveNii(os.path.join(d, "ring.nii.gz"), planC)
    planC2 = pc.loadFromH5(h5File)
    print(len(planC2.scan), [s.structureName for s in planC2.structure], len(planC2.dose))
    ibsi1.writeFeaturesToFile(featDict, os.path.join(d, "features.csv"))
    print(sorted(os.listdir(d)))
