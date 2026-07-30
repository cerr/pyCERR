"""Headless tests for the View > 'Upsample display (sinc)' helper.

Covers ``PyCerrViewer._upsample_for_display``, which sinc-upsamples a 2D slice
to the finest in-plane voxel spacing divided by ``upsampleFactor`` (1.0 =
finest; 1.5x / 2x go finer) for display. Pure array logic, exercised with a
lightweight stand-in ``self`` (no QApplication).
"""
import types
import numpy as np
import pytest

mw = pytest.importorskip("cerr.viewer.pycerr_gui.main_window")
PyCerrViewer = mw.PyCerrViewer


def _fake(on=True, factor=1.0):
    return types.SimpleNamespace(upsampleDisplay=on, upsampleFactor=factor,
                                 _upsampleCache={})


def _up(fake, img, hV, vV, key=("k",)):
    return PyCerrViewer._upsample_for_display(fake, img, hV, vV, key)


def test_disabled_returns_input_unchanged():
    img = np.random.rand(20, 100).astype(np.float32)
    hV = np.linspace(0, 99 * 0.05, 100)
    vV = np.linspace(0, 19 * 0.40, 20)
    out = _up(_fake(False), img, hV, vV)
    assert out is img


def test_coarse_axis_upsampled_to_finer_spacing():
    # rows (vertical) 8x coarser than cols -> rows grow ~8x, cols unchanged.
    img = np.random.rand(20, 100).astype(np.float32)
    hV = np.linspace(0, 99 * 0.05, 100)     # fine (0.5 mm)
    vV = np.linspace(0, 19 * 0.40, 20)      # coarse (4 mm)
    out = _up(_fake(True), img, hV, vV)
    assert out.shape[1] == 100              # fine axis untouched
    assert 7 * 20 <= out.shape[0] <= 9 * 20  # ~8x on the coarse axis


def test_isotropic_slice_unchanged():
    img = np.random.rand(40, 40).astype(np.float32)
    hV = np.linspace(0, 39 * 0.1, 40)
    vV = np.linspace(0, 39 * 0.1, 40)
    out = _up(_fake(True), img, hV, vV)
    assert out.shape == img.shape


def test_factor_upsamples_finer_than_finest():
    # An isotropic slice is untouched at factor 1.0 (target = finest spacing),
    # but a 2x factor targets finest/2 so both axes double.
    img = np.random.rand(40, 40).astype(np.float32)
    hV = np.linspace(0, 39 * 0.1, 40)
    vV = np.linspace(0, 39 * 0.1, 40)
    assert _up(_fake(True, 1.0), img, hV, vV).shape == (40, 40)
    out2 = _up(_fake(True, 2.0), img, hV, vV, ("f2",))
    assert 2 * 40 - 2 <= out2.shape[0] <= 2 * 40 + 2
    assert 2 * 40 - 2 <= out2.shape[1] <= 2 * 40 + 2


def test_both_axes_coarser_are_both_upsampled():
    # A 3rd, unrelated fine reference is not needed: min spacing here is dh;
    # dv is 3x coarser, so only rows grow. Use a case where cols are coarser.
    img = np.random.rand(60, 15).astype(np.float32)
    hV = np.linspace(0, 14 * 0.30, 15)      # cols coarse (3 mm)
    vV = np.linspace(0, 59 * 0.10, 60)      # rows fine (1 mm)
    out = _up(_fake(True), img, hV, vV)
    assert out.shape[0] == 60               # fine axis untouched
    assert out.shape[1] > 15                # coarse axis upsampled


def test_result_is_cached():
    fake = _fake(True)
    img = np.random.rand(20, 100).astype(np.float32)
    hV = np.linspace(0, 99 * 0.05, 100)
    vV = np.linspace(0, 19 * 0.40, 20)
    a = _up(fake, img, hV, vV, ("same",))
    b = _up(fake, img, hV, vV, ("same",))
    assert a is b


def test_upsample_factor_capped():
    # Extreme anisotropy (50x) is capped so the array stays sane.
    img = np.random.rand(10, 50).astype(np.float32)
    hV = np.linspace(0, 49 * 0.02, 50)      # very fine
    vV = np.linspace(0, 9 * 1.0, 10)        # very coarse (50x)
    out = _up(_fake(True), img, hV, vV)
    assert out.shape[0] <= 8 * 10           # MAXFAC cap


# --- 3D-view plane upsampling -----------------------------------------------
def _fake_plane():
    fake = _fake(True)
    fake._upsample_for_display = (
        lambda img, hV, vV, key: PyCerrViewer._upsample_for_display(
            fake, img, hV, vV, key))
    return fake


def test_plane_upsample_preserves_extent():
    fake = _fake_plane()
    img = np.random.rand(20, 100).astype(np.float32)   # rows coarse, cols fine
    rowVals = np.linspace(0, 19 * 0.40, 20)
    colVals = np.linspace(0, 99 * 0.05, 100)
    out, dRow, dCol = PyCerrViewer._upsample_plane(
        fake, img, rowVals, colVals, ("p",))
    assert out.shape[1] == 100                          # fine axis untouched
    assert out.shape[0] > 20                            # coarse axis upsampled
    # New spacings still span the original physical extent.
    assert dRow * (out.shape[0] - 1) == pytest.approx(abs(rowVals[-1] - rowVals[0]))
    assert dCol * (out.shape[1] - 1) == pytest.approx(abs(colVals[-1] - colVals[0]))


def test_plane_upsample_off_is_noop():
    fake = _fake(False)
    fake._upsample_for_display = (
        lambda img, hV, vV, key: PyCerrViewer._upsample_for_display(
            fake, img, hV, vV, key))
    img = np.random.rand(20, 100).astype(np.float32)
    rowVals = np.linspace(0, 19 * 0.40, 20)
    colVals = np.linspace(0, 99 * 0.05, 100)
    out, dRow, dCol = PyCerrViewer._upsample_plane(
        fake, img, rowVals, colVals, ("p",))
    assert out is img
    assert dRow == pytest.approx(0.40)
    assert dCol == pytest.approx(0.05)


# --- smallest native spacing ------------------------------------------------
def test_smallest_spacing():
    xA = np.linspace(0, 15 * 0.1, 16)     # 0.1
    yA = np.linspace(0, 15 * 0.1, 16)     # 0.1
    zA = np.linspace(0, 3 * 0.4, 4)       # 0.4
    s = PyCerrViewer._smallest_spacing(xA, yA, zA, (16, 16, 4))
    assert s == pytest.approx(0.1)


# --- 3D volume isotropic resampling -----------------------------------------
def test_volume_iso_full_uses_smallest_spacing():
    # dy=dx=0.1, dz=0.4; target = smallest (0.1) -> z (extent 1.2) -> 13 samples.
    vol = np.random.rand(16, 16, 4).astype(np.float32)
    xA = np.linspace(0, 15 * 0.1, 16)
    yA = np.linspace(0, 15 * 0.1, 16)
    zA = np.linspace(0, 3 * 0.4, 4)
    out, xo, yo, zo = PyCerrViewer._resample_volume_isotropic(
        vol, xA, yA, zA, targetSpacing=0.1)
    assert out.shape[0] == 16 and out.shape[1] == 16     # already at 0.1
    assert out.shape[2] == 13                            # 1.2/0.1 + 1
    assert (zo[0], zo[-1]) == pytest.approx((zA[0], zA[-1]))


def test_volume_iso_half_resolution_is_coarser():
    # target = 2 * smallest (0.2): fewer samples on every axis than 'full'.
    vol = np.random.rand(16, 16, 4).astype(np.float32)
    xA = np.linspace(0, 15 * 0.1, 16)
    yA = np.linspace(0, 15 * 0.1, 16)
    zA = np.linspace(0, 3 * 0.4, 4)
    out, _, _, _ = PyCerrViewer._resample_volume_isotropic(
        vol, xA, yA, zA, targetSpacing=0.2)
    assert out.shape[0] == 9 and out.shape[1] == 9       # 1.5/0.2 + 1
    assert out.shape[2] == 7                             # 1.2/0.2 + 1


def test_volume_iso_noop_when_already_at_target():
    vol = np.random.rand(12, 12, 12).astype(np.float32)
    a = np.linspace(0, 11 * 0.1, 12)                     # spacing 0.1
    out, _, _, _ = PyCerrViewer._resample_volume_isotropic(
        vol, a, a, a, targetSpacing=0.1)
    assert out.shape == vol.shape


def test_volume_iso_capped_by_maxdim():
    vol = np.random.rand(8, 8, 4).astype(np.float32)
    a = np.linspace(0, 7 * 0.1, 8)
    zA = np.linspace(0, 3 * 0.4, 4)
    out, _, _, _ = PyCerrViewer._resample_volume_isotropic(
        vol, a, a, zA, targetSpacing=0.001, maxDim=64)
    assert max(out.shape) <= 64                          # maxDim cap
