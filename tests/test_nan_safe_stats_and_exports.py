"""Regression tests for three silent-failure bugs.

1. First-order statistics on filtered images: ``ibsi1.computeScalarFeatures``
   sets voxels outside the ROI to NaN, and ``range``/``meanAbsDev`` came back
   NaN while ``rms``/``medianAbsDev`` were divided by the total voxel count.
2. RTSTRUCT export after ``loadFromH5``: ``structureColor`` comes back as a
   numpy array, which pydicom 3 rejects as an IS value for ROIDisplayColor.
3. ``finterp3`` given full x/y coordinate vectors instead of
   ``[start, delta, end]`` returned OOBV everywhere instead of raising.

Fully offline using the bundled phantom.
"""
import os

import numpy as np
import pydicom
import pytest

from cerr import datasets
from cerr import plan_container as pc
from cerr.dcm_export import rtstruct_iod
from cerr.radiomics import first_order
from cerr.utils.interp import finterp3

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def test_first_order_ignores_nan_padding():
    """Stats of a NaN-padded array equal the stats of its valid voxels."""
    rng = np.random.default_rng(0)
    validV = rng.normal(40, 15, 500)
    paddedV = np.concatenate([validV, np.full(300, np.nan)])
    padded = first_order.stats(paddedV, 1.0, offsetForEnergy=0, binWidth=25)
    valid = first_order.stats(validV, 1.0, offsetForEnergy=0, binWidth=25)
    for name in ['range', 'rms', 'meanAbsDev', 'medianAbsDev', 'energy',
                 'mean', 'median', 'P10', 'P90', 'robustMeanAbsDev']:
        assert np.isfinite(padded[name]), name
        assert np.isclose(padded[name], valid[name]), name


def test_first_order_definitions():
    """IBSI definitions on a small hand-checkable array."""
    xV = np.array([1.0, 2.0, 3.0, 10.0, np.nan])
    s = first_order.stats(xV, 1.0, offsetForEnergy=0, binWidth=1)
    assert s['range'] == pytest.approx(9.0)
    assert s['meanAbsDev'] == pytest.approx(np.mean(np.abs([1, 2, 3, 10] - np.mean([1, 2, 3, 10]))))
    assert s['medianAbsDev'] == pytest.approx(np.mean(np.abs(np.array([1, 2, 3, 10]) - 2.5)))
    assert s['rms'] == pytest.approx(np.sqrt(np.mean(np.square([1, 2, 3, 10]))))


def test_rtstruct_export_after_h5_roundtrip(tmp_path):
    planC = pc.loadDcmDir(phantom_dir)
    h5File = str(tmp_path / 'plan.h5')
    pc.saveToH5(planC, h5File)
    planH5 = pc.loadFromH5(h5File)
    color = planH5.structure[0].structureColor

    rsFile = str(tmp_path / 'RS_from_h5.dcm')
    rtstruct_iod.create([0], rsFile, planH5, {})
    ds = pydicom.dcmread(rsFile)
    written = [int(c) for c in ds.ROIContourSequence[0].ROIDisplayColor]
    assert written == [int(c) for c in np.ravel(color)]


def test_finterp3_rejects_full_coordinate_vectors():
    field3M = np.arange(5 * 6 * 4, dtype=float).reshape(5, 6, 4) + 1
    xV = 10 + np.arange(6) * 0.2          # cols
    yV = 4 - np.arange(5) * 0.2           # rows, decreasing as in pyCERR
    zV = -3 + np.arange(4) * 0.3
    xQ, yQ, zQ = np.array([10.4]), np.array([3.6]), np.array([-2.7])

    with pytest.raises(ValueError, match=r'\[start, delta, end\]'):
        finterp3(xQ, yQ, zQ, field3M, xV, yV, zV, OOBV=0)

    xFieldV = np.array([xV[0], xV[1] - xV[0], xV[-1]])
    yFieldV = np.array([yV[0], yV[1] - yV[0], yV[-1]])
    val = finterp3(xQ, yQ, zQ, field3M, xFieldV, yFieldV, zV, OOBV=0)
    assert val[0] == pytest.approx(field3M[2, 2, 1])
