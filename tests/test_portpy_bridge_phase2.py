"""Phase 2 tests for cerr.imrtp.portpy_bridge.

Computes a beamlet dose-influence matrix with the pyRadPlan bridge on a
water-box phantom, exports it as a PortPy dataset, loads it back and builds
PortPy's InfluenceMatrix, then verifies:
  * unit-weight dose maps onto the target (matrix rows/cols are consistent);
  * a direct CVXPy optimization on PortPy's A delivers the prescription to
    the target and spares the periphery;
  * the optimized dose imports back into planC.

Requires both pyRadPlan and portpy; skipped when either is unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("pyRadPlan")
pytest.importorskip("portpy")
cp = pytest.importorskip("cvxpy")

import SimpleITK as sitk  # noqa: E402
import portpy.photon as pp  # noqa: E402

from cerr import plan_container as pc  # noqa: E402
from cerr.imrtp import pyradplan_bridge as prp  # noqa: E402
from cerr.imrtp import portpy_bridge as ppb  # noqa: E402


VOXEL_MM = 4.0
SHAPE_ZYX = (28, 44, 44)
SPHERE_RADIUS_MM = 14.0
PRESC_GY = 2.0


def _makePhantomPlanC(tmp_path):
    nz, ny, nx = SHAPE_ZYX
    hu = np.full(SHAPE_ZYX, -1000.0, dtype=np.float32)
    hu[3:-3, 6:-6, 6:-6] = 0.0
    img = sitk.GetImageFromArray(hu)
    img.SetSpacing([VOXEL_MM] * 3)
    img.SetOrigin([0.0, 0.0, 0.0])
    niiFile = str(tmp_path / "phantom.nii.gz")
    sitk.WriteImage(img, niiFile)
    planC = pc.loadNiiScan(niiFile, imageType="CT SCAN")
    scanImg = planC.scan[0].getSitkImage()

    def _imp(mask_zyx, name):
        mImg = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
        mImg.CopyInformation(scanImg)
        m3 = prp.doseArrayFromSitk(mImg, planC, 0).astype(bool)
        pc.importStructureMask(m3, 0, name, planC)

    body = np.zeros(SHAPE_ZYX, dtype=bool)
    body[3:-3, 6:-6, 6:-6] = True
    _imp(body, "BODY")
    nz, ny, nx = SHAPE_ZYX
    cz, cy, cx = (nz - 1) / 2, (ny - 1) / 2, (nx - 1) / 2
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    dist = VOXEL_MM * np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    _imp(dist <= SPHERE_RADIUS_MM, "PTV")
    return planC


def _sidx(planC, name):
    return [i for i, s in enumerate(planC.structure) if s.structureName == name][0]


@pytest.fixture(scope="module")
def phase2(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("portpy_phase2")
    planC = _makePhantomPlanC(tmp_path)
    ptv, body = _sidx(planC, "PTV"), _sidx(planC, "BODY")
    iso = prp.targetCentroidMm(planC, [ptv])
    ct, cst, pln = prp.planFromPlanC(
        planC, scanNum=0, beamsNum=None,
        objectives={ptv: [prp.squaredDeviation(PRESC_GY)]},
        structNums=[ptv, body], targetStructNums=[ptv],
        gantryAngles=[0.0, 90.0], isoCenter=iso, bixelWidth=5.0,
        doseGridResolution={"x": 4.0, "y": 4.0, "z": 4.0})
    stf, dij = prp.calcDoseInfluence(ct, cst, pln)

    patDir = ppb.writePortpyFromPyRadPlan(
        planC, dij, stf, outDir=str(tmp_path), patientId="Phantom_2",
        scanNum=0, structNums=[body, ptv], bodyStructNum=body,
        isoCenter=iso)

    data = pp.DataExplorer(data_dir=str(tmp_path), patient_id="Phantom_2")
    ct_pp = pp.CT(data)
    structs = pp.Structures(data)
    beams = pp.Beams(data)
    inf = pp.InfluenceMatrix(structs, beams, ct_pp, target_structure="PTV",
                             is_bev=True)
    return {"planC": planC, "dij": dij, "structs": structs,
            "inf": inf, "refImg": ppb.doseGridRefImage(dij)}


def test_influence_matrix_shape(phase2):
    inf = phase2["inf"]
    assert inf.A.shape[0] > 0 and inf.A.shape[1] > 1
    assert inf.A.nnz > 0


def test_unit_weight_dose_on_target(phase2):
    inf, structs = phase2["inf"], phase2["structs"]
    x = np.ones(inf.A.shape[1])
    dose_1d = inf.A @ x
    dose_3d = inf.dose_1d_to_3d(dose_1d=dose_1d)
    ptv_idx = structs.structures_dict["name"].index("PTV")
    ptv_vox = structs.opt_voxels_dict["voxel_idx"][ptv_idx]
    assert dose_1d[ptv_vox].mean() > 0.5 * dose_1d.max()


def test_optimize_delivers_prescription(phase2):
    inf, structs = phase2["inf"], phase2["structs"]
    A = inf.A.tocsr()
    ptv_idx = structs.structures_dict["name"].index("PTV")
    body_idx = structs.structures_dict["name"].index("BODY")
    ptv_vox = np.asarray(structs.opt_voxels_dict["voxel_idx"][ptv_idx])
    body_vox = np.asarray(structs.opt_voxels_dict["voxel_idx"][body_idx])

    x = cp.Variable(A.shape[1], nonneg=True)
    dPtv = A[ptv_vox, :] @ x
    dBody = A[body_vox, :] @ x
    obj = cp.Minimize(cp.sum_squares(dPtv - PRESC_GY)
                      + 0.05 * cp.sum_squares(dBody))
    cp.Problem(obj).solve(solver=cp.SCS, verbose=False)
    assert x.value is not None

    dose_1d = A @ x.value
    ptv_mean = dose_1d[ptv_vox].mean()
    assert abs(ptv_mean - PRESC_GY) / PRESC_GY < 0.2
    # periphery (body minus target) should be cooler than the target
    body_only = np.setdiff1d(body_vox, ptv_vox)
    assert dose_1d[body_only].mean() < ptv_mean

    # import optimized dose back into planC (dose grid -> scan grid)
    dose_3d = inf.dose_1d_to_3d(dose_1d=dose_1d)          # (z, y, x) dose grid
    doseImg = sitk.GetImageFromArray(dose_3d.astype(np.float64))
    doseImg.CopyInformation(phase2["refImg"])
    doseNum = prp.importDoseToPlanC(phase2["planC"], doseImg, scanNum=0,
                                    fractionGroupID="portpy")
    dArr = phase2["planC"].dose[doseNum].doseArray
    assert dArr.shape == tuple(phase2["planC"].scan[0].getScanSize())
    assert dArr.max() > 0
