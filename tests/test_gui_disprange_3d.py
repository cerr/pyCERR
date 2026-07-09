"""Headless tests for honoring the display (cyan) range in 3D.

Covers the scan/dose display-range helpers and the Volume3D scan opacity ramp
(zeroed outside the scan display range). Pure logic, lightweight stand-ins - no
QApplication.
"""
import types
import numpy as np
import pytest

mw = pytest.importorskip("cerr.viewer.pycerr_gui.main_window")
v3 = pytest.importorskip("cerr.viewer.pycerr_gui.volume3d")
PyCerrViewer = mw.PyCerrViewer
# the Volume3D dialog class (has apply_style / _scan_opacity_ramp)
Vol3D = next(c for c in vars(v3).values()
             if isinstance(c, type) and hasattr(c, "_scan_opacity_ramp"))


def _cb(disp, lo=0.0, hi=100.0):
    return types.SimpleNamespace(dispRange=disp, axisMin=lo, axisMax=hi,
                                 _span=lambda: hi - lo)


def test_scan_disp_range_full_is_infinite():
    # no stored range for the base scan -> no masking
    v = types.SimpleNamespace(scanNum=0, dispRangeByScan={0: None})
    assert PyCerrViewer._scan_disp_range(v) == (float("-inf"), float("inf"))
    v = types.SimpleNamespace(scanNum=0, dispRangeByScan={})
    assert PyCerrViewer._scan_disp_range(v) == (float("-inf"), float("inf"))


def test_scan_disp_range_partial():
    # a stored (lo, hi) for the base scan is returned as-is
    v = types.SimpleNamespace(scanNum=0, dispRangeByScan={0: (20.0, 60.0)})
    assert PyCerrViewer._scan_disp_range(v) == (20.0, 60.0)


def test_scan_disp_range_overlay_does_not_affect_base():
    # an overlay's (narrow) range must not mask the base scan
    v = types.SimpleNamespace(scanNum=0,
                              dispRangeByScan={0: None, 2: (40.0, 60.0)})
    assert PyCerrViewer._scan_disp_range(v) == (float("-inf"), float("inf"))


def test_apply_dispmask_per_scan():
    # base (0) has no mask -> unchanged; overlay (2) masks outside [40, 60]
    v = types.SimpleNamespace(dispRangeByScan={0: None, 2: (40.0, 60.0)})
    arr = np.array([[10.0, 50.0, 90.0]])
    out0 = PyCerrViewer._apply_dispmask(v, arr, 0)
    assert not np.ma.is_masked(out0)
    out2 = PyCerrViewer._apply_dispmask(v, arr, 2)
    assert np.ma.getmaskarray(out2).tolist() == [[True, False, True]]


def test_dose_disp_range_full_and_partial():
    v = types.SimpleNamespace(colorbar=_cb((0.0, 100.0)))
    assert PyCerrViewer._dose_disp_range(v) == (float("-inf"), float("inf"))
    v = types.SimpleNamespace(colorbar=_cb((10.0, 40.0)))
    assert PyCerrViewer._dose_disp_range(v) == (10.0, 40.0)


def _fakeDialog(dispRange):
    viewer = types.SimpleNamespace(
        windowCenter=50.0, windowWidth=100.0,          # clim = (0, 100)
        _scan_disp_range=lambda: dispRange)
    d = types.SimpleNamespace(scanOpacity=1.0, viewer=viewer)
    d._clim = lambda: Vol3D._clim(d)
    return d


def test_scan_opacity_ramp_full_range_all_visible():
    d = _fakeDialog((float("-inf"), float("inf")))
    ramp = Vol3D._scan_opacity_ramp(d)
    assert ramp.shape == (256,)
    assert ramp.max() > 0                     # some opacity present
    # no forced-zero band: the ramp equals the sigmoid x opacity everywhere
    assert np.count_nonzero(ramp == 0.0) < 128


def test_scan_opacity_ramp_zeroed_outside_display_range():
    # clim (0,100); display range (40,60) -> samples outside [40,60] are 0.
    d = _fakeDialog((40.0, 60.0))
    ramp = Vol3D._scan_opacity_ramp(d)
    scal = np.linspace(0.0, 100.0, 256)
    assert np.all(ramp[scal < 40.0] == 0.0)
    assert np.all(ramp[scal > 60.0] == 0.0)
    assert np.any(ramp[(scal >= 40.0) & (scal <= 60.0)] >= 0.0)


# --- graded dose iso-surface opacity ----------------------------------------
def test_dose_surface_opacity_grading():
    levels = [14.0, 28.0, 42.0, 56.0, 70.0]     # 5 evenly spaced shells
    base = 0.6
    op = [PyCerrViewer._dose_surface_opacity(lv, levels, base) for lv in levels]
    assert op[-1] == pytest.approx(base)         # highest level -> full opacity
    assert op[0] == pytest.approx(0.2 * base)    # lowest level -> 0.2x
    assert np.all(np.diff(op) > 0)               # monotonically increasing
    # linear interpolation by level
    assert op[2] == pytest.approx(base * (0.2 + 0.8 * 0.5))


def test_dose_surface_opacity_single_level():
    # one surface -> gets the full (highest) opacity
    assert PyCerrViewer._dose_surface_opacity(50.0, [50.0], 0.5) == \
        pytest.approx(0.5)
