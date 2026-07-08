"""Headless tests for the GUI's isodose-level resolution.

These cover the logic that feeds both the 2D isodose lines and the 3D /
cut-plane isodose surfaces (``PyCerrViewer._isodose_abs_levels`` and the
level-resolution / cache-key portion of ``_pv_dose_iso``). They call the
methods with a lightweight stand-in ``self`` so no QApplication is created -
importing the module is enough. The VTK render path itself needs a live GUI
and is not exercised here.
"""
import types
import pytest

# Importing the GUI module does not create a QApplication; skip cleanly if the
# optional GUI stack (PyQt/pyvista) is unavailable.
mw = pytest.importorskip("cerr.viewer.pycerr_gui.main_window")
PyCerrViewer = mw.PyCerrViewer


def _fakeViewer(units, levels, prescription=0.0):
    dose = types.SimpleNamespace(prescriptionDose=prescription)
    planC = types.SimpleNamespace(dose=[dose])
    return types.SimpleNamespace(
        isodoseUnits=units, isodoseLevels=levels, planC=planC)


def test_abs_levels_gy_passthrough():
    v = _fakeViewer("Gy", [20, 50, 70])
    out = PyCerrViewer._isodose_abs_levels(v, 0, doseMax=80.0)
    assert out == [20, 50, 70]           # absolute, dose max irrelevant


def test_abs_levels_percent_of_max():
    v = _fakeViewer("% of max", [30, 50, 100])
    out = PyCerrViewer._isodose_abs_levels(v, 0, doseMax=60.0)
    assert out == pytest.approx([18.0, 30.0, 60.0])   # % of 60 Gy


def test_abs_levels_percent_of_rx_uses_prescription():
    v = _fakeViewer("% of Rx", [50, 100], prescription=45.0)
    out = PyCerrViewer._isodose_abs_levels(v, 0, doseMax=60.0)
    assert out == pytest.approx([22.5, 45.0])         # % of 45 Gy Rx


def test_abs_levels_percent_of_rx_falls_back_to_max():
    # No prescription stored -> falls back to % of dose max.
    v = _fakeViewer("% of Rx", [50, 100], prescription=0.0)
    out = PyCerrViewer._isodose_abs_levels(v, 0, doseMax=60.0)
    assert out == pytest.approx([30.0, 60.0])


def test_abs_levels_track_edited_levels():
    """Editing the level list changes the resolved output (the property the
    3D/cut-plane cache key now depends on)."""
    v = _fakeViewer("Gy", [30, 60])
    first = PyCerrViewer._isodose_abs_levels(v, 0, doseMax=70.0)
    v.isodoseLevels = [10, 30, 60, 66]
    second = PyCerrViewer._isodose_abs_levels(v, 0, doseMax=70.0)
    assert first != second
    assert second == [10, 30, 60, 66]
