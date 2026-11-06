"""``ibsi1.getQuantizedVolume`` must not write into the array it is given.

For the 'Original' image type, ``computeScalarFeatures`` passes a view of the
shared preprocessed scan. Writing NaN outside the ROI into it made every filter
applied afterwards see NaN, which spread through the filter support into the
ROI, so filtered features depended on whether 'Original' was in the settings.
"""
import numpy as np

from cerr.radiomics import ibsi1


def _vol_and_mask():
    rng = np.random.default_rng(1)
    vol = rng.normal(100, 20, (12, 12, 6))
    mask = np.zeros(vol.shape, dtype=bool)
    mask[3:9, 3:9, 1:5] = True
    return vol, mask


def test_quantize_leaves_input_untouched_with_texture():
    vol, mask = _vol_and_mask()
    before = vol.copy()
    view = vol[:, :, :]                       # a view, as computeScalarFeatures passes
    paramS = {'settings': {'texture': {}}}
    q = ibsi1.getQuantizedVolume(view, paramS, mask, None, None, None, 25)
    np.testing.assert_array_equal(vol, before)
    assert np.all(np.isfinite(vol))
    assert q.shape == vol.shape


def test_quantize_leaves_input_untouched_without_texture():
    vol, mask = _vol_and_mask()
    before = vol.copy()
    out = ibsi1.getQuantizedVolume(vol[:, :, :], {'settings': {}}, mask, None, None, None, None)
    np.testing.assert_array_equal(vol, before)
    assert np.all(np.isnan(out[~mask])) and np.array_equal(out[mask], vol[mask])
