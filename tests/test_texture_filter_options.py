"""Tests for texture-filter options outside the IBSI reference cases:
Sobel on 3D scans, the wavelet decomposition level, and Laws energy without
a Padding entry."""
import numpy as np
import pytest
from scipy.signal import convolve2d

from cerr.radiomics import texture_filters, texture_utils

rng = np.random.default_rng(0)
SCAN3M = rng.normal(size=(24, 26, 8))
MASK3M = np.zeros(SCAN3M.shape, dtype=bool)
MASK3M[6:18, 6:20, 2:6] = True


def test_sobel_filters_each_slice_of_a_3d_scan():
    outS = texture_utils.processImage('Sobel', SCAN3M, MASK3M, {})
    mag3M, dir3M = outS['SobelMag'], outS['SobelDir']
    assert mag3M.shape == SCAN3M.shape and dir3M.shape == SCAN3M.shape
    fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    for slc in (0, 5):
        gx = convolve2d(SCAN3M[:, :, slc], fx, mode='same')
        gy = convolve2d(SCAN3M[:, :, slc], fy, mode='same')
        np.testing.assert_allclose(mag3M[:, :, slc], np.hypot(gx, gy))
    # 2D input still works
    mag2M, _ = texture_filters.sobelFilter(SCAN3M[:, :, 0])
    np.testing.assert_allclose(mag2M, mag3M[:, :, 0])


def test_wavelet_level_is_read():
    """The decomposition level in the settings is used, with either spelling."""
    base = {'Wavelets': 'db', 'Index': '2', 'Direction': 'LLL'}
    out1 = texture_utils.processImage('Wavelets', SCAN3M, MASK3M, {**base, 'Level': 1})
    out2 = texture_utils.processImage('Wavelets', SCAN3M, MASK3M, {**base, 'Level': 2})
    outLower = texture_utils.processImage('Wavelets', SCAN3M, MASK3M, {**base, 'level': 2})
    key = list(out1)[0]
    np.testing.assert_allclose(out2[key], outLower[key])
    assert not np.allclose(out1[key], out2[key])
    # The level-2 approximation averages over a wider neighbourhood, so
    # neighbouring voxels are more alike
    def neighbourCorr(vol3M):
        return np.corrcoef(vol3M[:-1].ravel(), vol3M[1:].ravel())[0, 1]
    assert neighbourCorr(out2[key]) > neighbourCorr(out1[key])
    with pytest.raises(ValueError, match='level'):
        texture_utils.processImage('Wavelets', SCAN3M, MASK3M, {**base, 'Level': 0})


def _bestCircularShift(a, b):
    """Smallest max-abs difference between b and circular shifts of a, and the shift."""
    best = (np.inf, None)
    shifts = [(s,) for s in range(a.shape[0])] if a.ndim == 1 else \
        [(s, s) for s in range(a.shape[0])]
    for sh in shifts:
        err = np.abs(np.roll(a, sh, axis=tuple(range(len(sh)))) - b).max()
        if err < best[0]:
            best = (err, sh[0])
    return best


@pytest.mark.parametrize("name", ["haar", "db2", "coif1"])
def test_stationary_wavelet_transform_matches_pywavelets(name):
    """swt and swt2 agree with PyWavelets' stationary transform at every level,
    up to one circular shift per level that is the same in 1D and 2D, so the
    axial (2D) and slice (1D) passes of the 3D filter stay aligned."""
    pywt = pytest.importorskip("pywt")
    wav = pywt.Wavelet(name)
    loD, hiD = np.asarray(wav.dec_lo), np.asarray(wav.dec_hi)
    x = rng.normal(size=32)
    img = rng.normal(size=(32, 32))
    L, H = texture_filters.swt(x, 3, loD, hiD)
    a, h, v, d = texture_filters.swt2(img, 3, loD, hiD)
    ref1 = pywt.swt(x, wav, level=3, trim_approx=False, norm=False)[::-1]
    ref2 = pywt.swt2(img, wav, level=3, trim_approx=False, norm=False)[::-1]
    for lev in range(3):
        err1, shift1 = _bestCircularShift(L[lev], ref1[lev][0])
        errD, _ = _bestCircularShift(H[lev], ref1[lev][1])
        err2, shift2 = _bestCircularShift(a[:, :, lev], ref2[lev][0])
        assert max(err1, errD, err2) < 1e-10, (name, lev)
        assert shift1 % 32 == shift2 % 32
        detail = [_bestCircularShift(c[:, :, lev], r)[0]
                  for c in (h, v, d) for r in ref2[lev][1]]
        assert sorted(detail)[:3] == pytest.approx([0, 0, 0], abs=1e-10)


def test_laws_energy_without_padding_entry():
    paramS = {'Direction': '3d', 'Type': 'E5L5S5', 'Normalize': 'yes',
              'EnergyKernelSize': [5, 5, 5], 'EnergyPadSize': [2, 2, 2],
              'EnergyPadMethod': 'mirror'}
    outS = texture_utils.processImage('LawsEnergy', SCAN3M, MASK3M, paramS)
    resp3M = list(outS.values())[0]
    assert resp3M.shape == SCAN3M.shape
    assert np.all(np.isfinite(resp3M)) and np.all(resp3M >= 0)
