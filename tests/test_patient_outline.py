"""Tests for patient outline extraction and couch detection on a synthetic CT."""
import numpy as np

from cerr.utils import mask as maskUtils


def _ctWithCouch(nRows=200, nCols=200, nSlcs=6):
    """Air background, an elliptical 'patient' and a couch slab below it."""
    rng = np.random.default_rng(0)
    scan3M = rng.normal(-1000.0, 10.0, (nRows, nCols, nSlcs))
    rM, cM = np.mgrid[:nRows, :nCols]
    body = ((rM - 85) / 45.0) ** 2 + ((cM - 100) / 70.0) ** 2 <= 1
    core = ((rM - 85) / 37.0) ** 2 + ((cM - 100) / 62.0) ** 2 <= 1
    # A fat layer around soft tissue, as in CT: the outline threshold sits in the fat
    scan3M[body, :] = -100.0
    scan3M[core, :] = 40.0
    scan3M[body, :] += rng.normal(0.0, 5.0, (body.sum(), nSlcs))
    scan3M[136:146, 15:185, :] = 200.0          # couch, 6 rows below the patient
    return scan3M, body


def test_outline_excludes_couch_and_keeps_input():
    scan3M, body = _ctWithCouch()
    original = scan3M.copy()
    outline3M = maskUtils.getPatientOutline(scan3M, outThreshold=-400, minMaskSize=500)
    np.testing.assert_array_equal(scan3M, original)          # input untouched
    assert not outline3M[136:146, :, :].any()                # no couch
    dice = 2 * (outline3M[:, :, 2] & body).sum() / (outline3M[:, :, 2].sum() + body.sum())
    assert dice > 0.97


def test_couch_detection_is_reproducible():
    scan3M, _ = _ctWithCouch()
    rows = {maskUtils.getCouchLocationHough(scan3M)[0] for _ in range(5)}
    assert len(rows) == 1
    seeded = {maskUtils.getCouchLocationHough(scan3M, seed=3)[0] for _ in range(3)}
    assert len(seeded) == 1
