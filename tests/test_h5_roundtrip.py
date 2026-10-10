"""Round-trip test for HDF5 serialization (``saveToH5`` / ``loadFromH5``).

Loads the bundled phantom (scan + RTSTRUCT), adds a synthetic non-uniform dose,
serializes the whole planC to HDF5, reloads it, and checks the scan pixels, the
structure mask and the dose array all survive the round-trip. Fully offline.

Note: saveToH5 writes every scan, structure, dose and deformation by default;
index lists select a subset (an empty list writes none of that type).
"""
import os
import numpy as np

from cerr import datasets
from cerr import plan_container as pc
from cerr.contour.rasterseg import getStrMask

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def _planC_with_dose():
    planC = pc.loadDcmDir(phantom_dir)
    nRows, nCols, nSlc = planC.scan[0].getScanSize()
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    # Non-uniform dose: linear ramp across columns (exercises array fidelity).
    ramp = np.linspace(0.0, 60.0, nCols, dtype=float)
    dose3M = np.broadcast_to(ramp[None, :, None], (nRows, nCols, nSlc)).copy()
    planC = pc.importDoseArray(dose3M, xV, yV, zV, planC, 0)
    return planC


def test_h5_roundtrip(tmp_path):
    planC = _planC_with_dose()
    h5File = str(tmp_path / 'plan.h5')

    pc.saveToH5(planC, h5File,
                scanNumV=list(range(len(planC.scan))),
                structNumV=list(range(len(planC.structure))),
                doseNumV=list(range(len(planC.dose))))
    assert os.path.exists(h5File)

    planC2 = pc.loadFromH5(h5File)

    # Same object counts.
    assert len(planC2.scan) == len(planC.scan)
    assert len(planC2.structure) == len(planC.structure)
    assert len(planC2.dose) == len(planC.dose)

    # Scan pixels preserved.
    np.testing.assert_array_equal(planC2.scan[0].getScanArray(),
                                  planC.scan[0].getScanArray())

    # Structure mask preserved.
    np.testing.assert_array_equal(getStrMask(0, planC2), getStrMask(0, planC))

    # Dose array preserved.
    np.testing.assert_allclose(planC2.dose[0].doseArray,
                               planC.dose[0].doseArray, rtol=1e-6, atol=1e-6)


def test_h5_roundtrip_multiple_contours_per_slice(tmp_path):
    # A ring (outer + inner contour) and a two-part structure have more than
    # one contour per slice. saveToH5 used to fail on these with
    # "Unable to ... create group (name already exists)".
    planC = pc.loadDcmDir(phantom_dir)
    nRows, nCols, nSlc = planC.scan[0].getScanSize()
    rowM, colM = np.ogrid[:nRows, :nCols]
    distM = np.hypot(rowM - nRows / 2, colM - nCols / 2)
    ring2M = (distM < 40) & (distM > 20)
    blobs2M = (np.hypot(rowM - 60, colM - 60) < 12) | (np.hypot(rowM - 140, colM - 140) < 12)

    firstNew = len(planC.structure)
    for name, mask2M in (('ring', ring2M), ('two_blobs', blobs2M)):
        mask3M = np.zeros((nRows, nCols, nSlc), dtype=bool)
        mask3M[:, :, nSlc // 2 - 2:nSlc // 2 + 3] = mask2M[:, :, None]
        planC = pc.importStructureMask(mask3M, 0, name, planC)
    newStructs = list(range(firstNew, len(planC.structure)))
    for structNum in newStructs:
        assert max(len(ctr.segments) for ctr in planC.structure[structNum].contour if ctr) > 1

    h5File = str(tmp_path / 'plan_multi_contour.h5')
    pc.saveToH5(planC, h5File, scanNumV=[0],
                structNumV=list(range(len(planC.structure))))
    planC2 = pc.loadFromH5(h5File)

    assert len(planC2.structure) == len(planC.structure)
    for structNum in range(len(planC.structure)):
        assert planC2.structure[structNum].structureName == planC.structure[structNum].structureName
        np.testing.assert_array_equal(getStrMask(structNum, planC2), getStrMask(structNum, planC))
        # every contour on every slice survives
        assert [len(c.segments) if c else 0 for c in planC2.structure[structNum].contour] == \
               [len(c.segments) if c else 0 for c in planC.structure[structNum].contour]


def test_h5_default_saves_everything(tmp_path):
    # With no index lists, saveToH5 writes the whole planC (it used to write
    # an empty file). Explicit lists still select a subset.
    planC = _planC_with_dose()
    h5File = str(tmp_path / 'plan_all.h5')
    pc.saveToH5(planC, h5File)
    planC2 = pc.loadFromH5(h5File)
    assert (len(planC2.scan), len(planC2.structure), len(planC2.dose)) == \
           (len(planC.scan), len(planC.structure), len(planC.dose))
    np.testing.assert_array_equal(getStrMask(0, planC2), getStrMask(0, planC))

    h5Subset = str(tmp_path / 'plan_subset.h5')
    pc.saveToH5(planC, h5Subset, scanNumV=[0], structNumV=[], doseNumV=[])
    planC3 = pc.loadFromH5(h5Subset)
    assert (len(planC3.scan), len(planC3.structure), len(planC3.dose)) == (1, 0, 0)
