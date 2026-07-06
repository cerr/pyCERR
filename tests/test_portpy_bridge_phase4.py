"""Phase 4 tests for cerr.imrtp.portpy_bridge (clinical-criteria mapping).

Exercises the full PortPy optimization path: a QIB influence matrix exported
to a PortPy dataset is optimized with PortPy's own Optimization class driven
by an opt_params/ClinicalCriteria built from a prescription, and the
resulting dose is imported back into planC.

Requires portpy (and cvxpy); pyRadPlan is NOT required.
"""

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("portpy")
pytest.importorskip("cvxpy")

import nibabel as nib  # noqa: E402

import cerr.plan_container as pc  # noqa: E402
from cerr.contour import rasterseg as rs  # noqa: E402
from cerr.imrtp import portpy_bridge as ppb  # noqa: E402


PRESC_GY = 6.0
NUM_FX = 3


def _phantom():
    N, NS = 28, 12
    ct = (np.zeros((N, N, NS), dtype=np.int16) - 1000)
    rr, cc = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    body = ((rr - N // 2) ** 2 + (cc - N // 2) ** 2) < (N * 0.42) ** 2
    for s in range(NS):
        ct[:, :, s][body] = 0
    tmp = tempfile.mkdtemp()
    f = os.path.join(tmp, 'ct.nii.gz')
    nib.save(nib.Nifti1Image(ct, np.diag([3., 3., 3., 1.])), f)
    planC = pc.loadNiiScan(f, imageType='CT SCAN')

    rr, cc, ss = np.meshgrid(np.arange(N), np.arange(N), np.arange(NS),
                             indexing='ij')
    bodyMask = np.zeros((N, N, NS), dtype=np.uint8)
    for s in range(NS):
        bodyMask[:, :, s][body] = 1
    ptv = ((rr - N // 2) ** 2 + (cc - N // 2) ** 2
           + ((ss - NS // 2) * 2) ** 2) < 6 ** 2
    pc.importStructureMask(bodyMask, 0, 'BODY', planC)
    pc.importStructureMask(ptv.astype(np.uint8), 0, 'PTV', planC)
    return planC, tmp


def _buildQIB(planC):
    from cerr.imrtp import imrtp_problem as imp
    from cerr.imrtp.dosecalc import generateQIBInfluence
    im = imp.initIMRTProblem(planC)
    ptvRel = [i for i, s in enumerate(planC.structure)
              if s.structureName == 'PTV'][0]
    g = imp.addGoal(im, ptvRel, planC)
    g.isTarget, g.xySampleRate = 'yes', 1
    imp.addEquispacedBeams(im, 3, 0.0, planC)
    im.params.algorithm = 'QIB'
    for b in im.beams:
        imp.conditionBeam(b, im, planC)
    generateQIBInfluence(im, planC)
    return im


def _sidx(planC, name):
    return [i for i, s in enumerate(planC.structure) if s.structureName == name][0]


@pytest.fixture(scope="module")
def phase4():
    planC, tmp = _phantom()
    im = _buildQIB(planC)
    ptv, body = _sidx(planC, "PTV"), _sidx(planC, "BODY")
    ppb.writePortpyFromQIB(planC, im, outDir=tmp, patientId="Phantom_4",
                           scanNum=0, structNums=[body, ptv],
                           bodyStructNum=body)
    sol, doseNum, planC = ppb.optimizeAndImport(
        planC, dataDir=tmp, patientId="Phantom_4",
        prescriptionGy=PRESC_GY, numFractions=NUM_FX, scanNum=0,
        targetName="PTV", oarObjectives={"BODY": 0.0}, solver="SCS")
    return {"planC": planC, "sol": sol, "doseNum": doseNum, "ptv": ptv}


def test_solution_produced(phase4):
    sol = phase4["sol"]
    x = sol["optimal_intensity"]
    assert x is not None and np.all(x >= -1e-6) and x.max() > 0


def test_imported_dose_on_scan_grid(phase4):
    planC, doseNum = phase4["planC"], phase4["doseNum"]
    dArr = planC.dose[doseNum].doseArray
    assert dArr.shape == tuple(planC.scan[0].getScanSize())
    assert np.all(np.isfinite(dArr)) and dArr.max() > 0


def test_prescription_delivered_to_target(phase4):
    planC, doseNum, ptv = phase4["planC"], phase4["doseNum"], phase4["ptv"]
    dArr = planC.dose[doseNum].doseArray
    ptvMask = rs.getStrMask(ptv, planC).astype(bool)
    ptvMean = dArr[ptvMask].mean()
    # total prescription (PRESC_GY over NUM_FX fractions) delivered to target
    assert abs(ptvMean - PRESC_GY) / PRESC_GY < 0.25
    # target hotter than the surrounding body
    from cerr.contour import rasterseg as rs2
    bodyMask = rs2.getStrMask(_sidx(planC, "BODY"), planC).astype(bool)
    outside = dArr[bodyMask & ~ptvMask]
    assert outside.mean() < ptvMean
