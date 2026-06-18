"""Demo: open the pyCERR IMRTP GUI.

Option A - your own data:
    python demo_imrtp_gui.py C:/path/to/dicom_dir

Option B - no arguments: builds a synthetic CT with a PTV and a Cord so you
can try the GUI immediately.

In the GUI:
    1. 'Structures' panel: pick PTV from the Add dropdown, tick isTarg;
       add Cord as a non-target.
    2. 'Beams' panel: press Equispaced, enter e.g. 5 beams starting at 0.
       Drag inside the Geometry Preview to rotate the selected beam.
    3. 'IM Parameters': algorithm 'PB-Demo' (built-in demonstration engine).
    4. 'File' panel: select 'Recompute & add dosimetry' and press Go.
       The dose lands in planC.dose; press Show to open it in napari.
"""

import sys
import numpy as np

from cerr import plan_container as pc
from cerr.imrtp import IMRTPGui


def syntheticPlan():
    import nibabel as nib
    import tempfile, os
    ct = (np.zeros((128, 128, 40)) - 1000).astype(np.int16)
    rr, cc = np.meshgrid(np.arange(128), np.arange(128), indexing='ij')
    body = ((rr - 64) ** 2 / 3600 + (cc - 64) ** 2 / 2500) < 1
    for s in range(40):
        ct[:, :, s][body] = 0
    f = os.path.join(tempfile.gettempdir(), 'imrtp_demo_ct.nii.gz')
    nib.save(nib.Nifti1Image(ct, np.diag([2.5, 2.5, 2.5, 1.0])), f)
    planC = pc.loadNiiScan(f, imageType='CT SCAN')

    rr, cc, ss = np.meshgrid(np.arange(128), np.arange(128), np.arange(40),
                             indexing='ij')
    ptv = (((rr - 64) ** 2 + (cc - 64) ** 2 + ((ss - 20) * 2.0) ** 2)
           < 100).astype(float)
    cord = (((rr - 95) ** 2 + (cc - 64) ** 2) < 16).astype(float) * body[..., None]
    planC = pc.importStructureMask(ptv, 0, 'PTV', planC)
    planC = pc.importStructureMask(cord, 0, 'Cord', planC)
    return planC


if __name__ == '__main__':
    if len(sys.argv) > 1:
        planC = pc.loadDcmDir(sys.argv[1])
    else:
        planC = syntheticPlan()
    IMRTPGui(planC)
