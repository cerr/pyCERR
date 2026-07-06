"""Application theme, QApplication management and public entry points."""
from cerr.viewer.pycerr_gui.common import *  # noqa: F401,F403
from cerr.viewer.pycerr_gui.main_window import PyCerrViewer  # noqa: E402

_APP = None  # keep a reference so the QApplication is never garbage-collected


def _get_app():
    global _APP
    if sys.platform == "win32":
        # give the process its own taskbar identity so Windows shows the
        # pyCERR logo (not the python.exe icon) in the taskbar / task manager
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "pyCERR.Viewer")
        except Exception:  # noqa: BLE001
            pass
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        _apply_theme_palette(app)
    app.setWindowIcon(pycerr_icon())
    _APP = app
    return app


def launch(planC=None, dicomDir=None):
    """Open the viewer with an existing planC and BLOCK until it is closed.

    Anything imported or modified in the GUI (scans, structures, doses) is
    reflected in the returned plan container.

    Example:
        import cerr.plan_container as pc
        from pycerr_gui import launch

        planC = pc.loadDcmDir(r"C:/data/pat1")
        planC = launch(planC)          # interact, then close the window
        print(len(planC.structure))    # includes anything added in the GUI

    Args:
        planC: optional cerr.plan_container.PlanC to display on startup.
        dicomDir: optional DICOM directory to import on startup.

    Returns:
        cerr.plan_container.PlanC: the updated plan container.
    """
    app = _get_app()
    win = PyCerrViewer(planC)
    win.show()
    if dicomDir and os.path.isdir(dicomDir):
        QtCore.QTimer.singleShot(200, lambda: win.import_dicom(dicomDir))
    app.exec_()
    return win.planC


def show(planC=None):
    """Open the viewer WITHOUT blocking and return the viewer object.

    Intended for interactive sessions (IPython/Jupyter: run `%gui qt` first).
    The viewer keeps a live handle to the plan container:

        viewer = show(planC)
        # ... interact with the GUI ...
        planC = viewer.planC           # always the current state
        viewer.setPlanC(otherPlanC)    # swap plans programmatically
        viewer.refresh_views()         # redraw after external planC edits

    Returns:
        PyCerrViewer: the viewer window (access .planC for current state).
    """
    _get_app()
    win = PyCerrViewer(planC)
    win.show()
    return win


def capture(planC, out_path, target="window", setup=None, size=(1480, 920)):
    """Render ``planC`` in the viewer and save a screenshot to ``out_path``,
    without requiring manual interaction. Returns the path written.

    ``setup`` is an optional callable ``setup(viewer)`` that drives the
    scripting API before the screenshot is taken, e.g.::

        from cerr.viewer.pycerr_gui import capture
        def prep(v):
            v.set_layout("grid")
            v.set_window_preset("Lung")
            v.set_dose(0); v.set_dose_alpha(0.4)
            v.goto_structure(0)
        capture(planC, "shot.png", target="window", setup=prep)

    For headless/batch use, set the Qt platform to offscreen first
    (``os.environ['QT_QPA_PLATFORM'] = 'offscreen'``) - note the pyvista 3D
    view needs a real GL context, so 2D targets are most reliable headless.
    """
    app = _get_app()
    win = PyCerrViewer(planC)
    win.resize(*size)
    win.show()
    app.processEvents()
    if setup is not None:
        setup(win)
    app.processEvents()
    path = win.save_screenshot(out_path, target=target)
    win.close()
    app.processEvents()
    return path


def main():
    dicomDir = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) \
        else None
    launch(dicomDir=dicomDir)


if __name__ == "__main__":
    main()
