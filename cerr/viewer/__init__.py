"""pyCERR viewer sub-package.

Modules:
    pycerr_napari   - napari-based viewer + widgets (showNapari, captureToFile)
    pycerr_gui      - PyQt5 desktop viewer (PyCerrViewer, show, launch, capture)
    pycerr_nbviewer - Jupyter/Colab viewer (NbViewer, showNB)

For backward compatibility, the public napari API is also reachable directly on
the package, e.g. ``cerr.viewer.showNapari(...)``. This is resolved lazily so
that importing ``cerr.viewer.pycerr_gui`` / ``cerr.viewer.pycerr_nbviewer`` does
not pull in napari.
"""

import importlib

# Real submodules - let the normal import machinery handle these. Without this
# guard, `from cerr.viewer import pycerr_nbviewer` makes the import system probe
# ``hasattr(cerr.viewer, 'pycerr_nbviewer')``, which calls __getattr__, which
# would try to import pycerr_napari and recurse infinitely.
_SUBMODULES = frozenset(
    {"pycerr_napari", "pycerr_gui", "pycerr_nbviewer", "cerr_colormaps"})


def __getattr__(name):
    # Defer submodule and dunder lookups to the import system / default behavior.
    if name in _SUBMODULES or name.startswith("__"):
        raise AttributeError(f"module 'cerr.viewer' has no attribute {name!r}")
    # Lazily expose the napari viewer API as cerr.viewer.<name>. import_module
    # (rather than `from ... import`) avoids re-entering this __getattr__.
    pycerr_napari = importlib.import_module("cerr.viewer.pycerr_napari")
    try:
        return getattr(pycerr_napari, name)
    except AttributeError:
        raise AttributeError(
            f"module 'cerr.viewer' has no attribute {name!r}") from None
