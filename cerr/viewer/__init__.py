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


def __getattr__(name):
    # Lazily expose the napari viewer API as cerr.viewer.<name>.
    from cerr.viewer import pycerr_napari
    try:
        return getattr(pycerr_napari, name)
    except AttributeError:
        raise AttributeError(
            f"module 'cerr.viewer' has no attribute {name!r}") from None
