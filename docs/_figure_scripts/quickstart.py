"""Figures and printed output for docs/quickstart.rst."""
import os
import tempfile

import numpy as np

from common import IMG_DIR, loadPhantom
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.viewer import pycerr_nbviewer

planC = loadPhantom()
print(len(planC.scan), len(planC.structure), len(planC.dose))

scan3M = planC.scan[0].getScanArray()
print(scan3M.shape, planC.scan[0].getScanSpacing(), planC.scan[0].getScanOrientation())
print([s.structureName for s in planC.structure])

mask3M = rs.getStrMask(0, planC)
voxVol = np.prod(planC.scan[0].getScanSpacing())
print(mask3M.shape, mask3M.dtype, mask3M.sum(), mask3M.sum() * voxVol)
print(scan3M[mask3M].mean())

viewer = pycerr_nbviewer.NbViewer(planC, scanNum=0, autoDisplay=False)
viewer.set_window_level(-400, 1500)
viewer.goto_structure(0)
viewer.save_screenshot(os.path.join(IMG_DIR, "quickstart_viewer.png"), dpi=80)

with tempfile.TemporaryDirectory() as d:
    planC.scan[0].saveNii(os.path.join(d, "scan.nii.gz"))
    planC.structure[0].saveNii(os.path.join(d, "gtv.nii.gz"), planC)
    pc.saveToH5(planC, os.path.join(d, "planC.h5"))
    planC2 = pc.loadFromH5(os.path.join(d, "planC.h5"))
    print(len(planC2.scan), len(planC2.structure), sorted(os.listdir(d)))
