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


def _require_qt():
    """Skip unless qtpy is installed and finds a Qt binding (Qt5 or Qt6)."""
    # qtpy raises QtBindingsNotFoundError, an ImportError, without a binding
    pytest.importorskip('qtpy')


def test_import_pycerr_gui():
    _require_qt()
    importlib.import_module('cerr.viewer.pycerr_gui')


def test_import_roe_gui():
    _require_qt()
    importlib.import_module('cerr.roe.roe_gui')


def test_import_imrtp_gui():
    _require_qt()
    importlib.import_module('cerr.imrtp.imrtp_gui')


def test_gui_sources_are_binding_neutral():
    """The Qt GUIs import Qt only through qtpy and avoid Qt5-only idioms, so
    they keep running under both Qt5 and Qt6."""
    import pathlib
    import re
    root = pathlib.Path(importlib.import_module('cerr').__file__).parent
    files = list((root / 'viewer' / 'pycerr_gui').glob('*.py')) + [
        root / 'roe' / 'roe_gui.py', root / 'imrtp' / 'imrtp_gui.py',
        root / 'viewer' / 'pycerr_napari.py', root / 'multiViewHelper.py']
    banned = re.compile(r'^\s*(from|import)\s+(PyQt[56]|PySide[26])\b'
                        r'|backend_qt5agg|["\']Qt5Agg["\']|\.exec_\(|pyqtSignal',
                        re.MULTILINE)
    hits = [f'{f.name}: {m.group(0).strip()}'
            for f in files for m in banned.finditer(f.read_text(encoding='utf-8'))]
    assert not hits, hits


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


def test_viewer_getattr_defers_submodules():
    """Submodule names must NOT be served by the lazy __getattr__ proxy.

    If they are, ``from cerr.viewer import pycerr_nbviewer`` makes the import
    system probe ``hasattr(cerr.viewer, 'pycerr_nbviewer')`` -> __getattr__ ->
    import pycerr_napari -> probe again -> infinite recursion. The proxy must
    raise AttributeError for submodules so normal import handles them.
    (Regression guard; note `from ... import` is what triggers the probe, not
    importlib.import_module of the full dotted path.)
    """
    import cerr.viewer as viewer
    for sub in ('pycerr_napari', 'pycerr_gui', 'pycerr_nbviewer', 'cerr_colormaps'):
        with pytest.raises(AttributeError):
            viewer.__getattr__(sub)


def test_import_roe_without_gui():
    """cerr.roe must import without Qt: the GUI is loaded only by launch()."""
    import subprocess, sys
    code = ("import sys\n"
            "for m in ('qtpy', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6'):\n"
            "    sys.modules[m] = None        # make every Qt import fail\n"
            "import cerr.roe\n"
            "from cerr.roe import dosimetric_models\n"
            "assert callable(cerr.roe.launch)\n"
            "assert 'cerr.roe.roe_gui' not in sys.modules\n")
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_import_ai_models():
    # needs only base dependencies (PyYAML for the model run specifications)
    importlib.import_module('cerr.ai_models')
