"""Regression tests for DICOM import robustness (cerr.plan_container.loadDcmDir).

Covers three fixes:

* **single-slice scans** load without the "index 1 is out of bounds" crash and
  get finite coordinate transforms (spacing derived from the slice thickness);
* **duplicate objects are skipped** on re-import (by stable DICOM UID), with a
  warning, instead of corrupting associations;
* importing a **list of files** brings in only those files (e.g. a single
  RTSTRUCT) rather than their whole folder.

Uses the bundled radiomics phantom (CT + RTSTRUCT); no network.
"""
import glob
import os
import shutil
import tempfile
import warnings

import numpy as np
import pydicom

from cerr import datasets
from cerr import plan_container as pc

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def _classify():
    """Split the phantom files into (scan files, RTSTRUCT files) by Modality."""
    scanFiles, rtFiles = [], []
    for f in glob.glob(os.path.join(phantom_dir, '*')):
        if not os.path.isfile(f):
            continue
        try:
            mod = pydicom.dcmread(f, stop_before_pixels=True,
                                  force=True).Modality
        except Exception:        # noqa: BLE001 - not a readable DICOM
            continue
        if mod in ('CT', 'MR', 'PT', 'NM'):
            scanFiles.append(f)
        elif mod in ('RTSTRUCT', 'SEG'):
            rtFiles.append(f)
    return scanFiles, rtFiles


def test_single_slice_scan_loads():
    scanFiles, _ = _classify()
    assert scanFiles, 'phantom should contain scan slices'
    tmp = tempfile.mkdtemp()
    try:
        shutil.copy(scanFiles[0], tmp)          # one slice in its own folder
        planC = pc.loadDcmDir(tmp)
        assert len(planC.scan) == 1
        assert int(planC.scan[0].getScanSize()[2]) == 1
        # transforms are finite (single-slice spacing from slice thickness)
        assert np.all(np.isfinite(planC.scan[0].Image2VirtualPhysicalTransM))
        assert np.all(np.isfinite(planC.scan[0].Image2PhysicalTransM))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_duplicate_reimport_is_skipped():
    planC = pc.loadDcmDir(phantom_dir)
    n0 = (len(planC.scan), len(planC.structure))
    assert n0[0] >= 1 and n0[1] >= 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        planC = pc.loadDcmDir(phantom_dir, initplanC=planC)
    # nothing duplicated
    assert (len(planC.scan), len(planC.structure)) == n0
    msgs = [str(w.message) for w in caught
            if 'already exists in planC' in str(w.message)]
    assert any(m.startswith('Scan') for m in msgs)
    assert any(m.startswith('Structure') for m in msgs)


def test_import_only_listed_files_no_extra_scans():
    scanFiles, rtFiles = _classify()
    assert rtFiles, 'phantom should contain an RTSTRUCT'
    tmp = tempfile.mkdtemp()
    try:
        for f in scanFiles:
            shutil.copy(f, tmp)
        planC = pc.loadDcmDir(tmp)              # scan only
        nScan, nStruct = len(planC.scan), len(planC.structure)
        assert nScan >= 1
        # import ONLY the RTSTRUCT file(s): must add the structure but pull in
        # no extra scan series (the file-list path, not the whole folder).
        planC = pc.loadDcmDir(rtFiles, initplanC=planC)
        assert len(planC.scan) == nScan
        assert len(planC.structure) > nStruct
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
