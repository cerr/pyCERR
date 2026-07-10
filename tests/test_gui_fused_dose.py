"""Headless tests for the 3D dose transfer functions (Volume3D dialog).

The dialog renders the dose as its own GPU volume (composited over the scan
volume), styled by ``Volume3DDialog._dose_color_tf`` / ``_dose_opacity_tf``.
These tests exercise those transfer functions with lightweight stand-ins - no
QApplication, no GL context.
"""
import types
import pytest

v3 = pytest.importorskip("cerr.viewer.pycerr_gui.volume3d")

Vol3D = v3.Volume3DDialog


# --- multi-volume dose transfer functions -----------------------------------
def _tf_dialog(doseOpacity=0.8, cbar=(0.0, 40.0),
               doseDisp=(float("-inf"), float("inf"))):
    viewer = types.SimpleNamespace(
        colorbar=types.SimpleNamespace(cbarRange=cbar, cmapName="jet"),
        _dose_disp_range=lambda: doseDisp)
    d = types.SimpleNamespace(doseOpacity=doseOpacity, viewer=viewer)
    d._dose_clim = lambda: Vol3D._dose_clim(d)
    d._color_tf = Vol3D._color_tf          # staticmethod
    return d


def test_dose_opacity_tf_scales_with_selected_opacity():
    pytest.importorskip("vtk")
    d = _tf_dialog(doseOpacity=0.8)
    otf = Vol3D._dose_opacity_tf(d)
    assert otf.GetValue(40.0) == pytest.approx(0.8, abs=0.02)  # max dose
    assert otf.GetValue(0.0) == pytest.approx(0.0, abs=0.02)   # no dose
    assert 0.3 < otf.GetValue(20.0) < 0.5                      # ~linear middle


def test_dose_opacity_tf_zeroed_outside_display_range():
    pytest.importorskip("vtk")
    d = _tf_dialog(doseOpacity=0.8, doseDisp=(15.0, 30.0))
    otf = Vol3D._dose_opacity_tf(d)
    assert otf.GetValue(10.0) == pytest.approx(0.0, abs=0.02)  # below range
    assert otf.GetValue(35.0) == pytest.approx(0.0, abs=0.02)  # above range
    assert otf.GetValue(22.0) > 0.3                            # inside range


def test_dose_color_tf_spans_cbar_range():
    pytest.importorskip("vtk")
    d = _tf_dialog(cbar=(0.0, 40.0))
    ctf = Vol3D._dose_color_tf(d)
    lowRGB = [ctf.GetRedValue(0.5), ctf.GetGreenValue(0.5), ctf.GetBlueValue(0.5)]
    hiRGB = [ctf.GetRedValue(39.5), ctf.GetGreenValue(39.5), ctf.GetBlueValue(39.5)]
    assert lowRGB[2] > lowRGB[0]     # jet low end is blue-dominant
    assert hiRGB[0] > hiRGB[2]       # jet high end is red-dominant
