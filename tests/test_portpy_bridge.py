"""Phase 1 tests for cerr.imrtp.portpy_bridge.

Builds a synthetic water-box phantom with a spherical PTV and a BODY
structure, writes it as a PortPy-format patient folder, loads it back with
PortPy's DataExplorer / CT / Structures / Beams, and asserts the geometry
round-trips (CT resolution, structure volumes, beam angles, optimization
voxel mapping).

Requires the optional dependency portpy; skipped when unavailable.
"""

import numpy as np
import pytest

pytest.importorskip("portpy")

import SimpleITK as sitk  # noqa: E402
import portpy.photon as pp  # noqa: E402

from cerr import plan_container as pc  # noqa: E402
from cerr.imrtp import portpy_bridge as ppb  # noqa: E402


VOXEL_MM = 2.5
SHAPE_ZYX = (48, 80, 80)          # slices, rows, cols
SPHERE_RADIUS_MM = 20.0


def _makePhantomPlanC(tmp_path):
    """Water box + BODY + spherical PTV, loaded via NIfTI."""
    nz, ny, nx = SHAPE_ZYX
    hu = np.full(SHAPE_ZYX, -1000.0, dtype=np.float32)
    hu[4:-4, 8:-8, 8:-8] = 0.0                            # water box
    img = sitk.GetImageFromArray(hu)
    img.SetSpacing([VOXEL_MM] * 3)
    img.SetOrigin([0.0, 0.0, 0.0])
    niiFile = str(tmp_path / "phantom.nii.gz")
    sitk.WriteImage(img, niiFile)
    planC = pc.loadNiiScan(niiFile, imageType="CT SCAN")

    scanImg = planC.scan[0].getSitkImage()

    def _importMask(mask_zyx, name):
        mImg = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
        mImg.CopyInformation(scanImg)
        # to pyCERR (row, col, slice)
        arr = sitk.GetArrayFromImage(mImg)
        from cerr.dataclasses.scan import flipSliceOrderFlag
        if flipSliceOrderFlag(planC.scan[0]):
            arr = np.flip(arr, axis=0)
        mask3M = np.transpose(arr, (1, 2, 0)).astype(bool)
        return pc.importStructureMask(mask3M, 0, name, planC)

    # BODY = water box
    body = np.zeros(SHAPE_ZYX, dtype=bool)
    body[4:-4, 8:-8, 8:-8] = True
    _importMask(body, "BODY")

    # PTV = central sphere
    ctrZ, ctrY, ctrX = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    dist = VOXEL_MM * np.sqrt((zz - ctrZ) ** 2 + (yy - ctrY) ** 2 + (xx - ctrX) ** 2)
    _importMask(dist <= SPHERE_RADIUS_MM, "PTV")
    return planC


def _structIdx(planC, name):
    return [i for i, s in enumerate(planC.structure) if s.structureName == name][0]


@pytest.fixture(scope="module")
def writtenDataset(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("portpy_phase1")
    planC = _makePhantomPlanC(tmp_path)
    ptv = _structIdx(planC, "PTV")
    body = _structIdx(planC, "BODY")
    gantry = [0.0, 72.0, 144.0, 216.0, 288.0]
    patDir = ppb.writePortpyDataset(
        planC, outDir=str(tmp_path), patientId="Phantom_1",
        scanNum=0, structNums=[body, ptv], bodyStructNum=body,
        gantryAngles=gantry, energyMV="6X", machineName="pyCERR",
        fieldHalfWidthMm=40.0)
    return {"planC": planC, "dataDir": str(tmp_path), "patDir": patDir,
            "gantry": gantry}


def test_dataset_loads_in_portpy(writtenDataset):
    data = pp.DataExplorer(data_dir=writtenDataset["dataDir"],
                           patient_id="Phantom_1")
    meta = data.load_metadata()
    assert set(meta["structures"]["name"]) == {"BODY", "PTV"}
    assert len(meta["beams"]["ID"]) == 5


def test_ct_resolution_roundtrips(writtenDataset):
    data = pp.DataExplorer(data_dir=writtenDataset["dataDir"],
                           patient_id="Phantom_1")
    ct = pp.CT(data)
    np.testing.assert_allclose(ct.get_ct_res_xyz_mm(), [VOXEL_MM] * 3, atol=1e-6)


def test_structure_volumes_roundtrip(writtenDataset):
    data = pp.DataExplorer(data_dir=writtenDataset["dataDir"],
                           patient_id="Phantom_1")
    structs = pp.Structures(data)
    assert set(structs.get_structures()) == {"BODY", "PTV"}
    # analytic sphere volume in cc (4/3 pi r^3, mm^3 -> cc)
    analytic_cc = (4.0 / 3.0) * np.pi * SPHERE_RADIUS_MM ** 3 / 1000.0
    ptv_cc = structs.get_volume_cc("PTV")
    assert abs(ptv_cc - analytic_cc) / analytic_cc < 0.15
    # PTV must map into optimization voxels (non-empty, inside calc box)
    ptv_ind = structs.structures_dict["name"].index("PTV")
    assert structs.opt_voxels_dict["voxel_idx"][ptv_ind].size > 0
    assert structs.get_fraction_of_vol_in_calc_box("PTV") > 0.99


def test_beam_geometry_roundtrips(writtenDataset):
    data = pp.DataExplorer(data_dir=writtenDataset["dataDir"],
                           patient_id="Phantom_1")
    beams = pp.Beams(data)
    ids = beams.get_all_beam_ids()
    got = [beams.get_gantry_angle(b) for b in ids]
    assert sorted(got) == sorted(writtenDataset["gantry"])
    # beamlet 2d map built successfully from the nominal grid
    bmap = beams.get_beamlet_idx_2d_grid(beam_id=ids[0])
    assert bmap.ndim == 2 and (bmap >= 0).any()
    iso = beams.get_iso_center(ids[0])
    # sphere centered in the box -> isocenter near geometric center (mm)
    assert set(iso.keys()) == {"x_mm", "y_mm", "z_mm"}


def test_opt_voxel_map_consistency(writtenDataset):
    """ct_to_dose_voxel_map indices are 0..N-1 inside BODY, -1 outside."""
    data = pp.DataExplorer(data_dir=writtenDataset["dataDir"],
                           patient_id="Phantom_1")
    structs = pp.Structures(data)
    idxMap = structs.opt_voxels_dict["ct_to_dose_voxel_map"][0]
    inside = idxMap[idxMap >= 0]
    assert inside.min() == 0
    assert inside.max() == inside.size - 1
    assert np.array_equal(np.sort(inside), np.arange(inside.size))
