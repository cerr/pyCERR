"""Headless tests for the 2D in-place style fast path.

Covers ``PyCerrViewer._fast_layer_style``: persistent scan/dose images get an
in-place colormap/window(clim)/opacity update + canvas redraw, while views
without a reusable image (isodose / per-axis override) or 3D views fall back to
a per-window refresh. Exercised with lightweight stand-ins - no QApplication.
"""
import types
import numpy as np
import pytest

mw = pytest.importorskip("cerr.viewer.pycerr_gui.main_window")
PyCerrViewer = mw.PyCerrViewer


class _Img:
    def __init__(self):
        self.alpha = self.cmap = self.clim = self.data = None

    def set_alpha(self, a):
        self.alpha = a

    def set_cmap(self, c):
        self.cmap = c

    def set_clim(self, lo, hi):
        self.clim = (lo, hi)

    def set_data(self, d):
        self.data = d


class _Canvas:
    def __init__(self):
        self.draws = 0

    def draw_idle(self):
        self.draws += 1


def _view(is3d=False, scanIm=None, doseIm=None):
    return types.SimpleNamespace(is3d=is3d, _scanIm=scanIm, _doseIm=doseIm,
                                 orientation="Axial", canvas=_Canvas())


def _viewer(views, axisScan=0, scanNum=0):
    v = types.SimpleNamespace(
        planC=types.SimpleNamespace(scan=[object()]),
        activeWins=list(views.keys()), views=views,
        slices={w: 0 for w in views},
        scanNum=scanNum, scanAlpha=0.7, doseAlpha=0.3,
        scanCmap="gray", windowCenter=40.0, windowWidth=400.0,
        colorbar=types.SimpleNamespace(cbarRange=(0.0, 70.0), mplCmap="jet",
                                       dispRange=(0.0, 100.0)),
        refreshed=[])
    v._axis_scan = lambda winId: axisScan
    v.refresh_views = lambda only=None: v.refreshed.append(only)
    v._fast_3d_style = lambda *a: False       # force 3D fallback in tests
    v._notify_volume3d_style = lambda layer: None
    # scan/dose re-mask helpers (cheap stand-ins)
    v._slice_data = lambda winId: (np.zeros((4, 4), dtype=float), None,
                                   [0, 1, 2, 3], [0, 1, 2, 3], None)
    v._upsample_for_display = lambda img, hV, vV, key: img
    v._apply_scan_dispmask = lambda img: img
    v._axis_dose = lambda winId: 0
    v._dose_interp = lambda idx: (object(), 1.0)
    v._resample_slice2d = lambda kind, idx, interp, winId, hV, vV: \
        np.ones((4, 4), dtype=float)
    return v


def test_scan_style_updates_images_in_place():
    views = {"A": _view(scanIm=_Img()), "B": _view(scanIm=_Img())}
    v = _viewer(views)
    PyCerrViewer._fast_layer_style(v, "scan")
    for w in ("A", "B"):
        im = views[w]._scanIm
        assert im.alpha == 0.7
        assert im.cmap == "gray"
        assert im.clim == (40.0 - 200.0, 40.0 + 200.0)   # center +/- width/2
        assert im.data is not None                       # slice re-masked
        assert views[w].canvas.draws == 1
    assert v.refreshed == []


def test_dose_style_updates_images_in_place():
    views = {"A": _view(doseIm=_Img())}
    v = _viewer(views)
    PyCerrViewer._fast_layer_style(v, "dose")
    im = views["A"]._doseIm
    assert im.alpha == 0.3
    assert im.cmap == "jet"
    assert im.clim == (0.0, 70.0)
    assert im.data is not None                        # colorwash re-masked
    assert v.refreshed == []


def test_missing_image_falls_back_per_window():
    views = {"A": _view(doseIm=None)}
    v = _viewer(views)
    PyCerrViewer._fast_layer_style(v, "dose")
    assert v.refreshed == ["A"]


def test_scan_override_skipped_not_refreshed():
    # base-scan style change does not affect a per-axis override view
    views = {"A": _view(scanIm=_Img())}
    v = _viewer(views, axisScan=2, scanNum=0)
    PyCerrViewer._fast_layer_style(v, "scan")
    assert v.refreshed == []
    assert views["A"]._scanIm.alpha is None           # untouched


def test_3d_view_falls_back():
    views = {"A": _view(is3d=True)}
    v = _viewer(views)
    PyCerrViewer._fast_layer_style(v, "scan")
    assert v.refreshed == ["A"]


def test_mixed_2d_and_3d():
    views = {"A": _view(scanIm=_Img()), "B": _view(is3d=True)}
    v = _viewer(views)
    PyCerrViewer._fast_layer_style(v, "scan")
    assert views["A"]._scanIm.cmap == "gray"          # 2D in place
    assert v.refreshed == ["B"]                        # only the 3D window
