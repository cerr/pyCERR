"""Round-trip test for HDF5 serialization (``saveToH5`` / ``loadFromH5``).

Loads the bundled phantom (scan + RTSTRUCT), adds a synthetic non-uniform dose,
serializes the whole planC to HDF5, reloads it, and checks the scan pixels, the
structure mask and the dose array all survive the round-trip. Fully offline.

Note: saveToH5 selects objects by explicit index lists; an empty list (the
default) writes nothing, so the indices are passed in here.
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
