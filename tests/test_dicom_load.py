"""Sanity tests for DICOM import (``cerr.plan_container.loadDcmDir``).

Loads the bundled radiomics phantom (scan + RTSTRUCT, no network) and checks
the most-used, least-tested ingestion path: scan pixels, grid geometry, the
coordinate vectors, and the scan<->structure association.
"""
import os
import numpy as np

from cerr import datasets
from cerr import plan_container as pc

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def test_loadDcmDir_scan_geometry():
    planC = pc.loadDcmDir(phantom_dir)
    assert len(planC.scan) >= 1

    scan = planC.scan[0]
    arr = scan.getScanArray()
    nRows, nCols, nSlc = scan.getScanSize()
    assert nSlc >= 2
    assert arr.shape == (nRows, nCols, nSlc)

    # Coordinate vectors line up with the array axes (cols=x, rows=y, slc=z).
    xV, yV, zV = scan.getScanXYZVals()
    assert len(xV) == nCols
    assert len(yV) == nRows
    assert len(zV) == nSlc

    # z is monotonic (slices ordered).
    dz = np.diff(zV)
    assert np.all(dz > 0) or np.all(dz < 0)


def test_loadDcmDir_structure_association():
    planC = pc.loadDcmDir(phantom_dir)
    assert len(planC.structure) >= 1
    # The bundled RTSTRUCT is associated with the loaded scan.
    assocUIDs = {s.assocScanUID for s in planC.structure}
    assert planC.scan[0].scanUID in assocUIDs
