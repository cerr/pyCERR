"""Phase 3 tests for cerr.imrtp.portpy_bridge (native QIB influence path).

Builds a water-cylinder phantom with a spherical PTV, computes the beamlet
influence matrix with pyCERR's own QIB engine (no pyRadPlan), exports it as
a PortPy dataset, loads it back and builds PortPy's InfluenceMatrix, then
verifies unit-weight dose lands on the target and a direct CVXPy
optimization delivers the prescription.

Requires portpy (and cvxpy); pyRadPlan is NOT required.
"""

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("portpy")
cp = pytest.importorskip("cvxpy")

import nibabel as nib  # noqa: E402
import portpy.photon as pp  # noqa: E402

import cerr.plan_container as pc  # noqa: E402
from cerr.imrtp import portpy_bridge as ppb  # noqa: E402


PRESC_GY = 2.0


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
    g.isTarget = 'yes'
    g.xySampleRate = 1
    imp.addEquispacedBeams(im, 2, 0.0, planC)
    im.params.algorithm = 'QIB'
    for b in im.beams:
        imp.conditionBeam(b, im, planC)
    generateQIBInfluence(im, planC)
    return im


def _sidx(planC, name):
    return [i for i, s in enumerate(planC.structure) if s.structureName == name][0]


@pytest.fixture(scope="module")
def phase3():
    planC, tmp = _phantom()
    im = _buildQIB(planC)
    ptv, body = _sidx(planC, "PTV"), _sidx(planC, "BODY")
    ppb.writePortpyFromQIB(planC, im, outDir=tmp, patientId="Phantom_3",
                           scanNum=0, structNums=[body, ptv],
                           bodyStructNum=body)
    data = pp.DataExplorer(data_dir=tmp, patient_id="Phantom_3")
    ct_pp = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)
    inf = pp.InfluenceMatrix(structs, beams, ct_pp, target_structure="PTV",
                             is_bev=True)
    return {"planC": planC, "structs": structs, "inf": inf}


def test_qib_influence_matrix_nonempty(phase3):
    inf = phase3["inf"]
    assert inf.A.shape[0] > 0 and inf.A.shape[1] > 1
    assert inf.A.nnz > 0


def test_qib_unit_weight_dose_on_target(phase3):
    inf, structs = phase3["inf"], phase3["structs"]
    dose_1d = inf.A @ np.ones(inf.A.shape[1])
    ptv_idx = structs.structures_dict["name"].index("PTV")
    ptv_vox = np.asarray(structs.opt_voxels_dict["voxel_idx"][ptv_idx])
    assert ptv_vox.size > 0
    assert dose_1d[ptv_vox].mean() > 0


def test_qib_optimize_delivers_prescription(phase3):
    inf, structs = phase3["inf"], phase3["structs"]
    A = inf.A.tocsr()
    ptv_idx = structs.structures_dict["name"].index("PTV")
    ptv_vox = np.asarray(structs.opt_voxels_dict["voxel_idx"][ptv_idx])

    x = cp.Variable(A.shape[1], nonneg=True)
    dPtv = A[ptv_vox, :] @ x
    cp.Problem(cp.Minimize(cp.sum_squares(dPtv - PRESC_GY))).solve(
        solver=cp.SCS, verbose=False)
    assert x.value is not None
    dose_1d = A @ x.value
    ptv_mean = dose_1d[ptv_vox].mean()
    assert abs(ptv_mean - PRESC_GY) / PRESC_GY < 0.25
