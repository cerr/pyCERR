"""Import smoke tests for the viewer sub-package and the IMRTP package.

These large modules carry no functional unit tests; a plain import already
catches syntax errors, bad relative imports, and package-shadowing regressions
(e.g. a stray ``cerr/viewer.py`` masking the ``cerr/viewer/`` package). Heavy
optional GUI dependencies are skipped when absent so the suite stays portable;
where the dependency IS installed the module must import cleanly.
"""
import importlib
import pytest


def _import_with(mod, *deps):
    for dep in deps:
        pytest.importorskip(dep)
    return importlib.import_module(mod)


def test_import_pycerr_napari():
    _import_with('cerr.viewer.pycerr_napari', 'napari')


def test_import_pycerr_gui():
    _import_with('cerr.viewer.pycerr_gui', 'PyQt5')


def test_import_pycerr_nbviewer():
    _import_with('cerr.viewer.pycerr_nbviewer', 'ipywidgets')


def test_import_imrtp():
    # Pure numeric module; no GUI toolkit required.
    importlib.import_module('cerr.imrtp.imrtp')


def test_viewer_package_lazy_napari_api():
    """cerr.viewer proxies the napari API lazily (cerr.viewer.showNapari)."""
    pytest.importorskip('napari')
    import cerr.viewer as viewer
    assert hasattr(viewer, 'showNapari')


def test_viewer_package_unknown_attr_raises():
    """The lazy proxy still raises AttributeError for genuinely missing names."""
    import cerr.viewer as viewer
    with pytest.raises(AttributeError):
        _ = viewer.this_attribute_does_not_exist
