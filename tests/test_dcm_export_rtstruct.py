"""Round-trip test for RTSTRUCT DICOM export (``cerr.dcm_export.rtstruct_iod``).

Exports a structure to an RTSTRUCT DICOM, reloads it alongside the original CT
slices (the RTSTRUCT references their SOP instances), and verifies the
rasterized mask, structure name, slice coverage, and scan association are
preserved exactly. Fully offline using the bundled phantom.
"""
import os
import glob
import shutil
import numpy as np

from cerr import datasets
from cerr import plan_container as pc
from cerr.contour.rasterseg import getStrMask
from cerr.dcm_export import rtstruct_iod

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def test_rtstruct_export_import_roundtrip(tmp_path):
    planC = pc.loadDcmDir(phantom_dir)
    assert len(planC.structure) >= 1
    origName = planC.structure[0].structureName
    origMask = getStrMask(0, planC)
    assert origMask.any()

    # Export structure 0 into a fresh directory that also holds the CT slices,
    # so the RTSTRUCT's referenced SOP instances resolve on re-import.
    exportDir = tmp_path / 'rtstruct_roundtrip'
    exportDir.mkdir()
    for f in glob.glob(os.path.join(phantom_dir, 'DCM_IMG_*.dcm')):
        shutil.copy(f, str(exportDir))
    rsFile = os.path.join(str(exportDir), 'RS_export.dcm')

    rtstruct_iod.create([0], rsFile, planC,
                        {'SeriesDescription': 'pyCERR RTSTRUCT export test'})
    assert os.path.exists(rsFile)
    assert os.path.getsize(rsFile) > 0

    # Reload the exported RTSTRUCT alongside the CT.
    planC2 = pc.loadDcmDir(str(exportDir))
    assert len(planC2.structure) == 1
    assert planC2.structure[0].structureName == origName
    assert planC2.structure[0].assocScanUID == planC2.scan[0].scanUID

    rtMask = getStrMask(0, planC2)
    assert rtMask.shape == origMask.shape
    # Same slice coverage.
    np.testing.assert_array_equal(np.where(origMask.any(axis=(0, 1)))[0],
                                  np.where(rtMask.any(axis=(0, 1)))[0])
    # Exported-then-reimported segmentation is bit-for-bit identical.
    assert int(rtMask.sum()) == int(origMask.sum())
    np.testing.assert_array_equal(rtMask, origMask)
