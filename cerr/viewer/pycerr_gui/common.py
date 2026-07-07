"""Shared imports, constants and small helpers for the pyCERR viewer GUI."""

import os
import sys
import time

# ---------------------------------------------------------------------------
# If pyCERR is not pip-installed, point this at your local checkout, e.g.
# r"C:\software\pyCERR_master\pyCERR"  (the folder that CONTAINS the "cerr" pkg)
# ---------------------------------------------------------------------------
PYCERR_PATH = r"C:\software\pyCERR_master\pyCERR"
if os.path.isdir(PYCERR_PATH) and PYCERR_PATH not in sys.path:
    sys.path.insert(0, PYCERR_PATH)

import numpy as np

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt
except ImportError:  # pragma: no cover - PySide fallback
    from PySide2 import QtCore, QtGui, QtWidgets  # type: ignore
    from PySide2.QtCore import Qt  # type: ignore

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (3d projection)
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

# Optional GPU-accelerated 3D view (falls back to matplotlib if missing):
#   pip install pyvista pyvistaqt
try:
    import pyvista as pv  # noqa: E402
    from pyvistaqt import QtInteractor  # noqa: E402
    HAS_PYVISTA = True
except Exception:  # noqa: BLE001  (VTK can fail on import, not just missing)
    HAS_PYVISTA = False

import cerr.plan_container as pc  # noqa: E402
import cerr.contour.rasterseg as rs  # noqa: E402
import cerr.dataclasses.scan as scn  # noqa: E402
from cerr import dvh as cerrDvh  # noqa: E402

# CERR dose colormaps (ported from MATLAB CERR's CERRColorMap.m).
# Keep cerr_colormaps.py next to this file; falls back to jet if missing.
try:
    from cerr.viewer.cerr_colormaps import CERR_COLORMAP_NAMES, \
        get_cmap as _cerr_get_cmap, get_lut as _cerr_get_lut
except ImportError:  # pragma: no cover
    CERR_COLORMAP_NAMES = []
    _cerr_get_cmap = _cerr_get_lut = None

# Familiar matplotlib colormaps offered for dose alongside CERR's own maps.
_MPL_DOSE_CMAPS = ["jet", "turbo", "rainbow", "viridis", "hot", "cool"]
DOSE_CMAP_NAMES = _MPL_DOSE_CMAPS + [c for c in CERR_COLORMAP_NAMES
                                     if c not in _MPL_DOSE_CMAPS]
# CERR's default dose colormap, falling back to jet if it is unavailable.
DEFAULT_DOSE_CMAP = "starinterp" if "starinterp" in CERR_COLORMAP_NAMES \
    else "jet"


def cerr_get_cmap(name):
    """Colormap by name: a CERR colormap when available, else matplotlib."""
    if _cerr_get_cmap is not None and name in CERR_COLORMAP_NAMES:
        return _cerr_get_cmap(name)
    return plt.get_cmap(name)


def cerr_get_lut(name, n=256):
    if _cerr_get_lut is not None and name in CERR_COLORMAP_NAMES:
        return _cerr_get_lut(name, n)
    return (plt.get_cmap(name)(np.linspace(0, 1, n))[:, :3] * 255
            ).astype(np.uint8)


# ---------------------------------------------------------------------------#
#  CT window presets (center, width) - same spirit as CERR's CTWindow menu
# ---------------------------------------------------------------------------#
CT_WINDOW_PRESETS = {
    "--- Manual ---": None,
    "Abd/Med": (-10, 330),
    "Head": (45, 125),
    "Liver": (80, 305),
    "Lung": (-500, 1500),
    "Spine": (30, 300),
    "Vrt/Bone": (400, 1500),
    "Soft Tissue": (40, 400),
    "Brain": (40, 80),
    "PET SUV": (2.5, 5),
}

VIEW_AXIAL, VIEW_SAGITTAL, VIEW_CORONAL = "Axial", "Sagittal", "Coronal"
VIEW_3D = "3D"
# user-facing display name for an orientation (the constants double as keys)
VIEW_DISPLAY = {VIEW_3D: "3D Cut Planes"}

# colors of the plane outlines in the 3D view, per orientation
PLANE_COLORS = {VIEW_AXIAL: "#e8c542", VIEW_SAGITTAL: "#3ad6e0",
                VIEW_CORONAL: "#ff66ff"}

# patient-orientation labels along pyCERR virtual axes:
# +x = patient Left, +y = Anterior, +z = Inferior (see scan.py: y and z are
# negated relative to DICOM LPS when building virtual coordinates)
ORIENT_POS = {"x": "L", "y": "A", "z": "I"}
ORIENT_NEG = {"x": "R", "y": "P", "z": "S"}
AXES_2D = {VIEW_AXIAL: ("x", "y"), VIEW_SAGITTAL: ("y", "z"),
           VIEW_CORONAL: ("x", "z")}   # (horizontal, vertical) per view
# which orientation a view's horizontal/vertical crosshair line navigates
CROSS_TARGET = {
    (VIEW_AXIAL, "h"): VIEW_SAGITTAL, (VIEW_AXIAL, "v"): VIEW_CORONAL,
    (VIEW_SAGITTAL, "h"): VIEW_CORONAL, (VIEW_SAGITTAL, "v"): VIEW_AXIAL,
    (VIEW_CORONAL, "h"): VIEW_SAGITTAL, (VIEW_CORONAL, "v"): VIEW_AXIAL,
}
N3D = 72   # max samples per dimension for the textured planes in the 3D view
# scan-array axes (0=row/y, 1=col/x, 2=slice/z) mapping to a view's
# (horizontal, vertical, through-plane) directions - for urOMT overlays
UROMT_AXES = {VIEW_AXIAL: (1, 0, 2), VIEW_SAGITTAL: (0, 2, 1),
              VIEW_CORONAL: (1, 2, 0)}

# Colormaps offered for fused scan overlays
OVERLAY_CMAPS = ["hot", "jet", "cool", "spring", "winter", "copper",
                 "viridis", "gray"]
# Colormaps offered for the base scan / registration-QA images (gray default)
SCAN_CMAPS = ["gray", "bone", "hot", "jet", "viridis", "magma", "plasma",
              "copper", "cool", "Greens", "Reds", "Blues"]


def ascending(coords, arr, axis):
    """Return coords sorted ascending and array flipped along axis to match."""
    coords = np.asarray(coords, dtype=float)
    if coords.size > 1 and coords[0] > coords[-1]:
        return coords[::-1], np.flip(arr, axis=axis)
    return coords, arr


if HAS_PYVISTA:
    class _VtkView(QtInteractor):
        """pyvista interactor that adds a right-CLICK signal (right-drag
        still zooms via VTK's trackball style) and lets the owner take
        over left-drags for plane dragging via the pick/drag hooks."""
        rightClicked = QtCore.pyqtSignal()
        doubleClicked = QtCore.pyqtSignal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self._rpos = None
            self._plane_dragging = False
            # hooks installed by the viewer: pick(pos)->bool starts a plane
            # drag and consumes the event; drag(pos)/end() continue/finish
            self.pick_plane = None
            self.drag_plane = None
            self.end_plane_drag = None

        def mousePressEvent(self, ev):
            if ev.button() == Qt.LeftButton and self.pick_plane is not None \
                    and self.pick_plane(ev.pos()):
                self._plane_dragging = True
                self.setCursor(Qt.ClosedHandCursor)
                return            # consumed: no camera rotation
            if ev.button() == Qt.RightButton:
                self._rpos = ev.pos()
            super().mousePressEvent(ev)

        def mouseDoubleClickEvent(self, ev):
            # double-click resets the 3D camera to the default framing
            if ev.button() == Qt.LeftButton and not self._plane_dragging:
                self.doubleClicked.emit()
                return
            super().mouseDoubleClickEvent(ev)

        def mouseMoveEvent(self, ev):
            if self._plane_dragging:
                if self.drag_plane is not None:
                    self.drag_plane(ev.pos())
                return
            super().mouseMoveEvent(ev)

        def mouseReleaseEvent(self, ev):
            if self._plane_dragging and ev.button() == Qt.LeftButton:
                self._plane_dragging = False
                self.setCursor(Qt.ArrowCursor)
                if self.end_plane_drag is not None:
                    self.end_plane_drag()
                return
            if ev.button() == Qt.RightButton and self._rpos is not None \
                    and (ev.pos() - self._rpos).manhattanLength() < 8:
                self.rightClicked.emit()
            self._rpos = None
            super().mouseReleaseEvent(ev)


# ---------------------------------------------------------------------------#
#  One orthogonal slice view (matplotlib canvas inside a Qt widget)
# ---------------------------------------------------------------------------#

def _pycerr_version():
    """Best-effort pyCERR version string for the About dialog."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version("pycerr")
        except PackageNotFoundError:
            pass
    except Exception:  # noqa: BLE001
        pass
    try:
        from cerr._version import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


_CONTOUR_CURSORS = {}


def _contour_cursor(kind):
    """Cached custom contouring cursor: 'pen' (freehand / polygon) or 'brush'
    (disk/ball mode). Drawn with a white halo for visibility; the hotspot is at
    the drawing tip (lower-left)."""
    cur = _CONTOUR_CURSORS.get(kind)
    if cur is not None:
        return cur
    pm = QtGui.QPixmap(24, 24)
    pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    halo = QtGui.QPen(QtGui.QColor("white"), 3.4, Qt.SolidLine, Qt.RoundCap)
    ink = QtGui.QPen(QtGui.QColor("black"), 1.6, Qt.SolidLine, Qt.RoundCap)
    for pen in (halo, ink):                       # the shaft (handle / body)
        p.setPen(pen)
        p.drawLine(7, 17, 20, 4)
    p.setPen(QtGui.QPen(QtGui.QColor("white"), 0.8))
    if kind == "brush":                           # blunt, flared bristle head
        p.setBrush(QtGui.QColor("#3a3a3a"))
        p.drawPolygon(QtGui.QPolygon([
            QtCore.QPoint(2, 22), QtCore.QPoint(8, 14),
            QtCore.QPoint(11, 17), QtCore.QPoint(5, 23)]))
    else:                                         # pointed pen nib
        p.setBrush(QtGui.QColor("black"))
        p.drawPolygon(QtGui.QPolygon([
            QtCore.QPoint(3, 21), QtCore.QPoint(9, 15),
            QtCore.QPoint(6, 13)]))
    p.end()
    cur = QtGui.QCursor(pm, 4, 20)                 # hotspot at the tip
    _CONTOUR_CURSORS[kind] = cur
    return cur


def _nonmodal_box(parent, title, text, icon, rich=False):
    """Build and show() a non-modal QMessageBox. Modal message boxes can hang
    or fail to appear when the viewer runs inside an integrated event loop
    (show() / IPython %gui qt). The Qt parent keeps the box alive; it deletes
    itself on close."""
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    if rich:
        box.setTextFormat(Qt.RichText)
    box.setText(text)
    box.setModal(False)
    box.setAttribute(Qt.WA_DeleteOnClose, True)
    box.show()
    box.raise_()
    box.activateWindow()
    return box


def _show_info(parent, title, text, rich=False):
    return _nonmodal_box(parent, title, text,
                         QtWidgets.QMessageBox.Information, rich)


def _show_warning(parent, title, text):
    return _nonmodal_box(parent, title, text, QtWidgets.QMessageBox.Warning)


def _show_error(parent, title, text):
    return _nonmodal_box(parent, title, text, QtWidgets.QMessageBox.Critical)


_LOGO_DIR = os.path.dirname(__file__)


def pycerr_icon():
    """The pyCERR logo as a QIcon, using the platform-native icon format:
    multi-resolution .ico on Windows (16-256 px, so the taskbar, title bar
    and Explorer each pick the right size), .icns on macOS, and .png
    elsewhere (Linux desktops use PNGs directly). Falls back to the PNG if
    the native file is missing."""
    if sys.platform == "win32":
        name = "CERR_logo.ico"
    elif sys.platform == "darwin":
        name = "CERR_logo.icns"
    else:
        name = "CERR_logo.png"
    path = os.path.join(_LOGO_DIR, name)
    if not os.path.isfile(path):
        path = os.path.join(_LOGO_DIR, "CERR_logo.png")
    return QtGui.QIcon(path)


_THEME_STYLESHEET = (
    "QMainWindow, QMenuBar, QStatusBar, QDialog { background: #d4d4d4; }"
    "QMenuBar::item:selected, QMenu::item:selected { background: #b0c4de; }")


def _theme_palette():
    """Light-gray panel palette (dark text, blue highlight). The image views
    themselves stay black (set on each matplotlib canvas)."""
    pal = QtGui.QPalette()
    gray = QtGui.QColor(212, 212, 212)
    pal.setColor(QtGui.QPalette.Window, gray)
    pal.setColor(QtGui.QPalette.WindowText, Qt.black)
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(245, 245, 245))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(228, 228, 228))
    pal.setColor(QtGui.QPalette.Text, Qt.black)
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(224, 224, 224))
    pal.setColor(QtGui.QPalette.ButtonText, Qt.black)
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(245, 245, 245))
    pal.setColor(QtGui.QPalette.ToolTipText, Qt.black)
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(120, 170, 230))
    pal.setColor(QtGui.QPalette.HighlightedText, Qt.white)
    return pal


def _apply_theme_palette(app):
    """Apply the light-gray theme application-wide (used when we own the app)."""
    app.setStyle("Fusion")
    app.setPalette(_theme_palette())
    app.setStyleSheet(_THEME_STYLESHEET)


# Names re-exported to the sibling modules via `from .common import *`
__all__ = [_n for _n in dir() if not _n.startswith('__')]
