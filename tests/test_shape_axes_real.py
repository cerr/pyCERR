"""Shape axis-length features are real numbers, not complex ``x+0j``.

``shape.eig`` used ``np.linalg.eig``, which can return complex eigenvalues for
the (symmetric) covariance of the ROI coordinates; it now uses ``eigvalsh``.
"""
import numpy as np

from cerr.radiomics import shape

AXIS_FEATURES = ['majorAxis', 'minorAxis', 'leastAxis', 'flatness', 'elongation']


def _ellipsoid(a=8, b=5, c=3):
    n = 2 * a + 3
    g = np.arange(n) - n // 2
    yM, xM, zM = np.meshgrid(g, g, g, indexing='ij')
    mask = (xM / a) ** 2 + (yM / b) ** 2 + (zM / c) ** 2 <= 1
    xV = np.arange(n) * 0.1
    yV = -np.arange(n) * 0.1
    zV = np.arange(n) * 0.1
    return mask, xV, yV, zV


def test_eig_is_real_and_ascending():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3)) * [3.0, 2.0, 1.0]
    vals = shape.eig(pts.T @ pts)
    assert vals.dtype.kind == 'f'
    assert np.all(np.diff(vals) >= 0)
    np.testing.assert_allclose(vals, np.sort(np.linalg.eigvals(pts.T @ pts).real))


def test_axis_features_are_real():
    mask, xV, yV, zV = _ellipsoid()
    s = shape.calcShapeFeatures(mask, xV, yV, zV, 'LPS')
    for name in AXIS_FEATURES:
        assert not np.iscomplexobj(s[name]), name
        assert np.isfinite(s[name]), name
    assert s['majorAxis'] > s['minorAxis'] > s['leastAxis'] > 0
    assert 0 < s['flatness'] <= s['elongation'] <= 1


def test_flat_roi_gives_real_zero_axis():
    """A single-slice ROI has a zero eigenvalue; rounding must not make sqrt complex or NaN."""
    mask, xV, yV, zV = _ellipsoid()
    flat = np.zeros_like(mask)
    flat[:, :, mask.shape[2] // 2] = mask[:, :, mask.shape[2] // 2]
    s = shape.calcShapeFeatures(flat, xV, yV, zV, 'LPS')
    assert not np.iscomplexobj(s['leastAxis'])
    assert s['leastAxis'] >= 0 and np.isfinite(s['leastAxis'])
