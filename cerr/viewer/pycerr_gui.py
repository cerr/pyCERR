"""
pyCERR Viewer GUI
=================
A QT-based desktop viewer built on pyCERR.

Features (mirroring the classic CERR slice viewer):
  * Import DICOM directory / NIfTI scan, dose & structure files / planC pickle
  * Axial, sagittal and coronal linked views with slice sliders & mouse wheel
  * Pan (drag) and zoom (right-drag up/down) per view; double-click / R to reset
  * Right-click menu per view: choose the orientation (Axial/Sagittal/Coronal)
    plus the scan, dose & structures it displays (CERR-style per-axis
    View/ScanSet/DoseSet/StructSet selection)
  * Lock slices across matching views (View menu or L): windows showing the
    same orientation scroll together
  * View > Layout: one view, two side-by-side, one large + two stacked,
    three equal columns, or four equal views in a 2x2 grid
  * 3D view (default 4th window of the 2x2 layout, or right-click > View >
    3D): shows the three orthogonal planes selected in the 2D views,
    textured with the scan and outlined in the plane-locator colors.
    GPU-accelerated via pyvista/VTK when installed (pip install pyvista
    pyvistaqt; falls back to matplotlib otherwise); rotate with left-drag,
    zoom with right-drag or wheel, pan with middle-drag. The planes are
    DRAGGABLE: grab one with the left button to scroll through slices -
    the linked 2D views follow live. Also shows structure surfaces (from
    the Structures checklist) and 30/50/70/90% isodose surfaces (from the
    Dose combo, colored by the dose colorbar)
  * Ruler tool per view (right-click > Draw Ruler): drag to measure distances
    in cm, CERR-style; toggle again to clear
  * Contouring tools (Tools > Contouring): CERR-style draw/erase of structure
    contours on the axial view - freehand, point-by-point polygon (line
    segments, right/double-click to close) and disk brush modes - with
    per-slice delete/copy, undo, and save back to planC via
    importStructureMask (new or edited structures)
  * Crosshairs linked across views (View menu or X to toggle)
  * Patient-orientation labels (L/R/A/P/S/I) at the edges of each 2D view
    and an orientation triad in the 3D view (View menu or O to toggle)
  * Multi-scan fusion: overlay any loaded scan with per-scan opacity & colormap
  * CT window/level presets (Soft Tissue, Lung, Bone, Brain, ...) + manual W/L
  * Structure contour overlays with per-structure colors & visibility toggles,
    optional contour vertex dots ("Points"), adjustable contour line width, and
    double-click a structure to center all three views on it
  * Dose colorwash overlay with adjustable transparency, colorbar & dose units
  * Crosshair readout: position (cm), scan value (HU/SUV), dose (Gy)
  * DVH tool (cumulative dose-volume histograms, CERR-style)
  * Registration QA tool (Tools menu): compare a base and a moving scan via
    Mirrorscope (moving mirrored about a draggable line), Side-by-side,
    Alternate grid (checkerboard) and Toggle/blend (base-moving cross-fade
    slider) displays in every 2D view; each scan uses its own colormap
  * Tools menu launchers for the IMRTP GUI (beamlet dose calculation) and the
    ROE GUI (Radiotherapy Outcomes Explorer), sharing this viewer's planC.
    IMRTP shows its computed dose in this viewer and draws beam's-eye-view
    overlays (checked beams): the per-slice field cross-section in each 2D
    view (updates as you scroll) and the full field pyramid in the 3D view

Requirements:
  pip install pyCERR PyQt5 matplotlib scipy
  -- or add your local pyCERR checkout to sys.path (see PYCERR_PATH below).

Run:
  python pycerr_gui.py [optional: path to DICOM directory]
"""

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
    from cerr_colormaps import CERR_COLORMAP_NAMES, get_cmap as cerr_get_cmap, \
        get_lut as cerr_get_lut
except ImportError:  # pragma: no cover
    CERR_COLORMAP_NAMES = ["jet"]

    def cerr_get_cmap(_name):
        return plt.get_cmap("jet")

    def cerr_get_lut(_name, n=256):
        return (plt.get_cmap("jet")(np.linspace(0, 1, n))[:, :3] * 255
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
class SliceView(QtWidgets.QWidget):
    # all signals carry the window id (stable), not the view orientation
    # (mutable via the per-axis View menu)
    sliceChanged = QtCore.pyqtSignal(str, int)
    cursorMoved = QtCore.pyqtSignal(str, float, float)
    viewReset = QtCore.pyqtSignal(str)
    contextRequested = QtCore.pyqtSignal(str)
    rulerChanged = QtCore.pyqtSignal(str)
    strokeFinished = QtCore.pyqtSignal(str, object)  # winId, [(x, y)..]
    brushStroke = QtCore.pyqtSignal(str, object, bool)  # winId, pts, isStart
    brushDone = QtCore.pyqtSignal(str)               # brush drag finished
    crosshairDragged = QtCore.pyqtSignal(str, float, float, bool, bool)

    CLICK_PX = 6   # right press+release within this distance = click, not drag

    def __init__(self, winId, orientation, parent=None):
        super().__init__(parent)
        self.winId = winId
        self.orientation = orientation   # current view type, may be changed
        self.fig = Figure(facecolor="black", layout="tight")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("black")
        self.ax.set_xticks([]), self.ax.set_yticks([])

        self.user_limits = None      # (xlim, ylim) while panned/zoomed
        self.xline = None            # crosshair artists (vertical, horizontal)
        self.yline = None
        self._xhair_drag = None      # (dragVertical, dragHorizontal) while held
        self._pan = None             # (x0px, y0px, xlim0, ylim0) during drag
        self._zoom_drag = None       # (y0px, anchorX, anchorY, xlim0, ylim0)
        self._rclick = None          # (x0px, y0px, user_limits) at right press
        self.ruler_mode = False      # left-drag measures instead of panning
        self.ruler = None            # ((x0, y0), (x1, y1)) in data coords (cm)
        self._ruler_drag = False
        self._ruler_line = None      # artists, recreated after ax.clear()
        self._ruler_text = None
        self.draw_mode = False       # contouring active on this view
        self.draw_tool = "freehand"  # "freehand" | "polygon" | "brush"
        self.brush_radius = 0.5      # cm (data units)
        self._stroke = None          # [(x, y), ...] while drag-drawing
        self._stroke_line = None
        self._poly = None            # clicked vertices in polygon mode
        self._poly_line = None
        self._brush_circle = None    # brush-size cursor
        self.vtk_widget = None       # pyvista QtInteractor for the 3D view
        self.qa_split_cb = None      # registration-QA split drag hook
        self._qa_drag = False

        self.label = QtWidgets.QLabel(orientation)
        self.label.setStyleSheet(
            "color:#e8c542; background:#1e1e1e; font-weight:bold; padding:2px;")
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._on_slider)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(1)
        lay.addWidget(self.label)
        lay.addWidget(self.canvas, stretch=1)
        lay.addWidget(self.slider)

        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    @property
    def is3d(self):
        return self.orientation == VIEW_3D

    @property
    def uses_vtk(self):
        return self.is3d and self.vtk_widget is not None

    def set_projection(self, to3d):
        """Swap between the 2D matplotlib canvas and the 3D display.
        The 3D display is a GPU-accelerated pyvista/VTK widget when
        available, else a matplotlib Axes3D fallback."""
        if to3d and HAS_PYVISTA:
            if self.vtk_widget is None:
                self.vtk_widget = _VtkView(self)
                self.vtk_widget.set_background("black")
                self.vtk_widget.rightClicked.connect(
                    lambda: self.contextRequested.emit(self.winId))
                self.layout().insertWidget(1, self.vtk_widget, stretch=1)
            self.canvas.hide()
            self.vtk_widget.show()
            self.slider.setVisible(False)
            return
        if self.vtk_widget is not None:
            self.vtk_widget.hide()
        self.canvas.show()
        self.fig.clf()
        if to3d:    # matplotlib fallback when pyvista is unavailable
            self.ax = self.fig.add_subplot(111, projection="3d")
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xticks([]), self.ax.set_yticks([])
        try:
            self.ax.set_facecolor("black")
        except Exception:  # noqa: BLE001
            pass
        self.slider.setVisible(not to3d)
        # artists from the old axes are gone
        self.xline = self.yline = None
        self._ruler_line = self._ruler_text = None
        self._stroke_line = self._poly_line = self._brush_circle = None
        self.user_limits = None
        self.canvas.draw_idle()

    def _on_slider(self, val):
        self.sliceChanged.emit(self.winId, val)

    def _on_scroll(self, event):
        if self.is3d:
            return
        step = 1 if event.button == "up" else -1
        self.slider.setValue(int(np.clip(self.slider.value() + step,
                                         self.slider.minimum(),
                                         self.slider.maximum())))

    # ------------------------------------------------------- pan & zoom -----
    def _on_press(self, event):
        if event.inaxes is not self.ax:
            return
        if self.is3d:
            # Axes3D rotates (left) / zooms (right) itself; we only watch
            # for a plain right-click to open the context menu
            if event.button == 3:
                self._rclick = (event.x, event.y, None)
            return
        if self.draw_mode and self._handle_draw_press(event):
            return
        if event.dblclick:
            self.reset_view()
            return
        if self.ruler_mode and event.button == 1:
            if event.xdata is not None:   # start a new ruler (replaces old)
                p = (event.xdata, event.ydata)
                self.ruler = (p, p)
                self._ruler_drag = True
                self._update_ruler_artists()
                self.canvas.draw_idle()
            return
        if self.qa_split_cb is not None and event.button == 1 \
                and event.xdata is not None \
                and self.qa_split_cb(self.winId, event.xdata, event.ydata,
                                     False):
            self._qa_drag = True     # dragging the registration-QA split
            return
        if event.button == 1 and event.xdata is not None:
            hit = self._xhair_hit(event)   # grab a crosshair line?
            if hit is not None:
                self._xhair_drag = hit
                self.canvas.setCursor(Qt.SizeAllCursor)
                self.crosshairDragged.emit(self.winId, event.xdata,
                                           event.ydata, hit[0], hit[1])
                return
        if event.button in (1, 2):   # left or middle drag pans
            self._pan = (event.x, event.y,
                         self.ax.get_xlim(), self.ax.get_ylim())
            self.canvas.setCursor(Qt.ClosedHandCursor)
        elif event.button == 3:      # right: drag zooms, plain click = menu
            self._zoom_drag = (event.y, event.xdata, event.ydata,
                               self.ax.get_xlim(), self.ax.get_ylim())
            self._rclick = (event.x, event.y, self.user_limits)
            self.canvas.setCursor(Qt.SizeVerCursor)

    def _xhair_hit(self, event, tol=7):
        """If the press is near a visible crosshair line, return
        (nearVertical, nearHorizontal); else None."""
        if self.xline is None or self.yline is None \
                or not self.xline.get_visible():
            return None
        xv = self.xline.get_xdata()[0]
        yv = self.yline.get_ydata()[0]
        try:
            ix, iy = self.ax.transData.transform((xv, yv))
        except Exception:  # noqa: BLE001
            return None
        # Native bool: numpy.bool_ (from the numpy comparison) is rejected by
        # the PyQt5 crosshairDragged signal's bool arguments.
        nearV = bool(abs(event.x - ix) < tol)   # near the vertical line
        nearH = bool(abs(event.y - iy) < tol)   # near the horizontal line
        return (nearV, nearH) if (nearV or nearH) else None

    # ------------------------------------------------- contour draw tools ---
    def _handle_draw_press(self, event):
        """Dispatch a press while contouring; True if the event was consumed."""
        if self.draw_tool == "polygon":
            # CERR's point-by-point DRAW mode: left-click adds a vertex
            # (line segments between clicks); right- or double-click closes.
            if self._poly and (event.button == 3
                               or (event.button == 1 and event.dblclick)):
                self._finish_polygon()
                return True
            if event.button == 1 and event.xdata is not None:
                if self._poly is None:
                    self._poly = []
                self._poly.append((event.xdata, event.ydata))
                self._update_poly_preview(event.xdata, event.ydata)
                return True
            return False     # middle-pan / right zoom-menu still available
        if event.button == 1:    # freehand & brush: drag collects points
            if event.xdata is not None:
                p = (event.xdata, event.ydata)
                self._stroke = [p]
                if self.draw_tool == "brush":
                    # CERR drawBall: paint immediately - the mask itself is
                    # the live feedback, so no trail line is drawn
                    self._update_brush_cursor(*p)
                    self.brushStroke.emit(self.winId, [p], True)
                else:
                    if self._stroke_line is None \
                            or self._stroke_line.axes is not self.ax:
                        self._stroke_line, = self.ax.plot(
                            [p[0]], [p[1]], color="#ff66ff",
                            lw=1.5, zorder=12)
                    else:
                        self._stroke_line.set_data([p[0]], [p[1]])
                    self.canvas.draw_idle()
            return True
        return False

    def _update_poly_preview(self, cx, cy):
        xs = [p[0] for p in self._poly] + [cx]
        ys = [p[1] for p in self._poly] + [cy]
        if self._poly_line is None or self._poly_line.axes is not self.ax:
            self._poly_line, = self.ax.plot(
                xs, ys, color="#ff66ff", lw=1.2, ls="--", marker="o",
                markersize=3, zorder=12)
        else:
            self._poly_line.set_data(xs, ys)
        self.canvas.draw_idle()

    def _finish_polygon(self):
        pts = self._poly or []
        self.cancel_polygon()
        if len(pts) >= 3:
            self.strokeFinished.emit(self.winId, pts)

    def cancel_polygon(self):
        self._poly = None
        if self._poly_line is not None:
            try:
                self._poly_line.remove()
            except Exception:  # noqa: BLE001
                pass
            self._poly_line = None
            self.canvas.draw_idle()

    def _update_brush_cursor(self, x, y):
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        if self._brush_circle is None or self._brush_circle.axes is not self.ax:
            self._brush_circle = mpatches.Circle(
                (x, y), self.brush_radius, fill=False, color="#ff66ff",
                lw=1.0, ls=":", zorder=12)
            self.ax.add_patch(self._brush_circle)
        else:
            self._brush_circle.center = (x, y)
            self._brush_circle.set_radius(self.brush_radius)
        self.ax.set_xlim(xlim)        # keep the view fixed while brushing
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def clear_draw_artists(self):
        """Remove transient contouring previews (polygon, brush cursor)."""
        self.cancel_polygon()
        if self._brush_circle is not None:
            try:
                self._brush_circle.remove()
            except Exception:  # noqa: BLE001
                pass
            self._brush_circle = None
            self.canvas.draw_idle()

    def _default_cursor(self):
        """Resting cursor for the current mode, restored after pan/zoom so
        contouring keeps its pen/brush cursor."""
        if self.draw_mode:
            return _contour_cursor(
                "brush" if self.draw_tool == "brush" else "pen")
        if self.ruler_mode:
            return Qt.CrossCursor
        return Qt.ArrowCursor

    def _on_release(self, event):
        if self._xhair_drag is not None:
            self._xhair_drag = None
            self.canvas.setCursor(self._default_cursor())
            return
        if self._qa_drag:
            self._qa_drag = False
            return
        if self.is3d:
            if event.button == 3 and self._rclick is not None \
                    and abs(event.x - self._rclick[0]) < self.CLICK_PX \
                    and abs(event.y - self._rclick[1]) < self.CLICK_PX:
                self._rclick = None
                self.contextRequested.emit(self.winId)
            else:
                self._rclick = None
            return
        if self._stroke is not None:
            pts = self._stroke
            self._stroke = None
            if self._stroke_line is not None:
                try:
                    self._stroke_line.remove()
                except Exception:  # noqa: BLE001
                    pass
                self._stroke_line = None
                self.canvas.draw_idle()
            if self.draw_tool == "brush":
                # painting already happened live; signal end of the drag
                self.brushDone.emit(self.winId)
            elif pts and len(pts) >= 3:   # freehand needs an enclosed region
                self.strokeFinished.emit(self.winId, pts)
            return
        if self._ruler_drag:
            self._ruler_drag = False
            self.rulerChanged.emit(self.winId)
            return
        if event.button == 3 and self._rclick is not None \
                and self._zoom_drag is not None \
                and abs(event.x - self._rclick[0]) < self.CLICK_PX \
                and abs(event.y - self._rclick[1]) < self.CLICK_PX:
            # a click, not a zoom drag: undo any tiny zoom, open the menu
            _, _, _, xlim0, ylim0 = self._zoom_drag
            self.ax.set_xlim(xlim0)
            self.ax.set_ylim(ylim0)
            self.user_limits = self._rclick[2]
            self.canvas.draw_idle()
            self._pan = self._zoom_drag = self._rclick = None
            self.canvas.setCursor(self._default_cursor())
            self.contextRequested.emit(self.winId)
            return
        self._rclick = None
        if self._pan is not None or self._zoom_drag is not None:
            self._pan = None
            self._zoom_drag = None
            self.canvas.setCursor(self._default_cursor())

    def _do_zoom_drag(self, event):
        y0, ax_x, ax_y, xlim0, ylim0 = self._zoom_drag
        # drag up zooms in, drag down zooms out; ~120 px doubles/halves
        factor = float(np.clip(2.0 ** ((event.y - y0) / 120.0), 0.05, 20.0))
        self.ax.set_xlim(ax_x + (xlim0[0] - ax_x) / factor,
                         ax_x + (xlim0[1] - ax_x) / factor)
        self.ax.set_ylim(ax_y + (ylim0[0] - ax_y) / factor,
                         ax_y + (ylim0[1] - ax_y) / factor)
        self.user_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self.canvas.draw_idle()

    def _do_pan(self, event):
        x0, y0, xlim0, ylim0 = self._pan
        bbox = self.ax.get_window_extent()
        sx = (xlim0[1] - xlim0[0]) / max(bbox.width, 1)
        sy = (ylim0[1] - ylim0[0]) / max(bbox.height, 1)
        dx = (event.x - x0) * sx
        dy = (event.y - y0) * sy
        self.ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
        self.ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
        self.user_limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self.canvas.draw_idle()

    def reset_view(self):
        """Clear pan/zoom and ask the main window to re-render at full extent."""
        self.user_limits = None
        self._pan = None
        self._zoom_drag = None
        self.viewReset.emit(self.winId)

    def _on_motion(self, event):
        if self.is3d:
            return    # Axes3D handles rotation/zoom motion itself
        if self._xhair_drag is not None:
            if event.inaxes is self.ax and event.xdata is not None:
                self.crosshairDragged.emit(self.winId, event.xdata,
                                           event.ydata, self._xhair_drag[0],
                                           self._xhair_drag[1])
            return
        if self._qa_drag:
            if event.inaxes is self.ax and event.xdata is not None:
                self.qa_split_cb(self.winId, event.xdata, event.ydata, True)
            return
        if self._stroke is not None:
            if event.inaxes is self.ax and event.xdata is not None:
                p = (event.xdata, event.ydata)
                prev = self._stroke[-1]
                self._stroke.append(p)
                if self.draw_tool == "brush":
                    # paint the new segment in real time
                    self._update_brush_cursor(*p)
                    self.brushStroke.emit(self.winId, [prev, p], False)
                else:
                    xs, ys = zip(*self._stroke)
                    self._stroke_line.set_data(xs, ys)
                    self.canvas.draw_idle()
            return
        if self.draw_mode and event.inaxes is self.ax \
                and event.xdata is not None:
            if self.draw_tool == "polygon" and self._poly:
                self._update_poly_preview(event.xdata, event.ydata)
            elif self.draw_tool == "brush":
                self._update_brush_cursor(event.xdata, event.ydata)
        if self._ruler_drag:
            if event.inaxes is self.ax and event.xdata is not None:
                self.ruler = (self.ruler[0], (event.xdata, event.ydata))
                self._update_ruler_artists()
                self.canvas.draw_idle()
                self.rulerChanged.emit(self.winId)
            return
        if self._zoom_drag is not None:
            self._do_zoom_drag(event)
            return
        if self._pan is not None:
            self._do_pan(event)
            return
        if event.inaxes is self.ax and event.xdata is not None:
            self.cursorMoved.emit(self.winId, event.xdata, event.ydata)

    # ------------------------------------------------------------- ruler ----
    def ruler_length(self):
        """In-plane length of the current ruler in cm (data units are cm)."""
        if self.ruler is None:
            return 0.0
        (x0, y0), (x1, y1) = self.ruler
        return float(np.hypot(x1 - x0, y1 - y0))

    def _update_ruler_artists(self):
        if self.ruler is None:
            return
        (x0, y0), (x1, y1) = self.ruler
        if self._ruler_line is None or self._ruler_line.axes is not self.ax:
            self._ruler_line, = self.ax.plot(
                [x0, x1], [y0, y1], color="0.85", lw=1.2, marker="+",
                markersize=9, markeredgewidth=1.4, zorder=10)
            self._ruler_text = self.ax.text(
                x1, y1, "", color="0.9", fontsize=8, zorder=11,
                ha="left", va="bottom",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none",
                          pad=1.5))
        else:
            self._ruler_line.set_data([x0, x1], [y0, y1])
        self._ruler_text.set_position((x1, y1))
        self._ruler_text.set_text(f" {self.ruler_length():.3g} cm")

    def redraw_ruler(self):
        """Recreate ruler artists after the axes were cleared (refresh)."""
        self._ruler_line = self._ruler_text = None
        if self.ruler is not None:
            self._update_ruler_artists()

    def clear_ruler(self):
        for art in (self._ruler_line, self._ruler_text):
            if art is not None:
                try:
                    art.remove()
                except Exception:  # noqa: BLE001
                    pass
        self._ruler_line = self._ruler_text = None
        self.ruler = None
        self._ruler_drag = False
        self.canvas.draw_idle()

    def set_range(self, nslices, current):
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(nslices - 1, 0))
        self.slider.setValue(current)
        self.slider.blockSignals(False)


# ---------------------------------------------------------------------------#
#  Standalone dose colorbar with draggable range markers (CERR-style)
#    * LEFT handles (yellow)  : colorbar/colormap mapping range
#    * RIGHT handles (cyan)   : dose display range (doses outside are hidden)
#  Double-click resets both ranges to [0, doseMax].
# ---------------------------------------------------------------------------#
class DoseColorbarWidget(QtWidgets.QWidget):
    rangesChanged = QtCore.pyqtSignal()

    GRAB_PX = 9          # vertical grab tolerance for handles
    TOP, BOT = 16, 16    # margins

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(132)
        self.setMinimumHeight(240)
        self.axisMax = 1.0
        self.cbarRange = [0.0, 1.0]      # colormap mapping range
        self.dispRange = [0.0, 1.0]      # dose display (mask) range
        self._drag = None                # ("cbar"|"disp", 0|1) while dragging
        self.cmapName = "starinterp" if "starinterp" in CERR_COLORMAP_NAMES \
            else CERR_COLORMAP_NAMES[0]  # CERR's default doseColormap
        self._set_cmap(self.cmapName)
        self.setToolTip("Dose colorbar\n"
                        "Yellow (left) handles: colorbar/colormap range\n"
                        "Cyan (right) handles: dose display range\n"
                        "Right-click: colormap & exact ranges\n"
                        "Double-click: reset ranges")

    def _set_cmap(self, name):
        self.cmapName = name
        self.mplCmap = cerr_get_cmap(name)
        self._lut = cerr_get_lut(name, 256)

    # ------------------------------------------------------------ public ----
    def setDose(self, doseMax):
        self.axisMax = max(float(doseMax), 1e-6)
        self.cbarRange = [0.0, self.axisMax]
        self.dispRange = [0.0, self.axisMax]
        self.update()

    # ---------------------------------------------------------- geometry ----
    def _bar_rect(self):
        return QtCore.QRect(44, self.TOP, 24, self.height() - self.TOP - self.BOT)

    def _val2y(self, v):
        r = self._bar_rect()
        return int(r.bottom() - (v / self.axisMax) * r.height())

    def _y2val(self, y):
        r = self._bar_rect()
        return float(np.clip((r.bottom() - y) / max(r.height(), 1), 0, 1)) \
            * self.axisMax

    # ------------------------------------------------------------ paint -----
    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = self._bar_rect()
        cbLo, cbHi = self.cbarRange
        dLo, dHi = self.dispRange
        span = max(cbHi - cbLo, 1e-9)

        # gradient bar (row by row), dimming values outside the display range
        for y in range(r.top(), r.bottom() + 1):
            val = self._y2val(y)
            idx = int(np.clip((val - cbLo) / span, 0, 1) * 255)
            c = self._lut[idx]
            if val < dLo or val > dHi:
                c = (c * 0.22).astype(np.uint8)
            p.setPen(QtGui.QColor(int(c[0]), int(c[1]), int(c[2])))
            p.drawLine(r.left(), y, r.right(), y)
        p.setPen(QtGui.QPen(QtGui.QColor("#aaaaaa"), 1))
        p.drawRect(r)

        font = p.font()
        font.setPointSize(9)
        p.setFont(font)

        # scale tick marks + numeric labels (right of the cyan handles)
        tickX = r.right() + 18
        for frac in np.linspace(0, 1, 6):
            v = frac * self.axisMax
            y = self._val2y(v)
            p.setPen(QtGui.QPen(QtGui.QColor("#888888"), 1))
            p.drawLine(tickX, y, tickX + 4, y)
            p.setPen(QtGui.QColor("black"))
            p.drawText(QtCore.QRect(tickX + 7, y - 9, 36, 18),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{v:.4g}")

        # handles, each with its value always shown (left=cbar, right=disp)
        for which, (lo, hi), color, side in (
                ("cbar", self.cbarRange, QtGui.QColor("#e8c542"), "left"),
                ("disp", self.dispRange, QtGui.QColor("#3ad6e0"), "right")):
            for j, v in enumerate((lo, hi)):
                y = self._val2y(v)
                tri = QtGui.QPolygonF()
                if side == "left":
                    x0 = r.left() - 3
                    tri << QtCore.QPointF(x0, y) \
                        << QtCore.QPointF(x0 - 11, y - 6) \
                        << QtCore.QPointF(x0 - 11, y + 6)
                    txtRect = QtCore.QRect(0, y - 9, r.left() - 16, 18)
                    align = Qt.AlignRight | Qt.AlignVCenter
                else:
                    x0 = r.right() + 4
                    tri << QtCore.QPointF(x0, y) \
                        << QtCore.QPointF(x0 + 11, y - 6) \
                        << QtCore.QPointF(x0 + 11, y + 6)
                    txtRect = QtCore.QRect(self.width() - 44, y - 9, 42, 18)
                    align = Qt.AlignRight | Qt.AlignVCenter
                p.setBrush(color)
                p.setPen(QtGui.QPen(Qt.black, 1))
                p.drawPolygon(tri)
                p.setPen(QtGui.QColor("black"))   # persistent value label
                p.drawText(txtRect, align, f"{v:.3g}")
        p.end()

    # ------------------------------------------------------------ mouse -----
    def _hit_test(self, pos):
        r = self._bar_rect()
        which = "cbar" if pos.x() < r.center().x() else "disp"
        rng = self.cbarRange if which == "cbar" else self.dispRange
        best, bestDist = None, self.GRAB_PX + 1
        for j in (0, 1):
            d = abs(pos.y() - self._val2y(rng[j]))
            if d < bestDist:
                best, bestDist = (which, j), d
        return best

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        self._drag = self._hit_test(ev.pos())
        if self._drag:
            self._move_to(ev.pos().y())

    # ----------------------------------------------------- context menu -----
    def contextMenuEvent(self, ev):
        menu = QtWidgets.QMenu(self)

        cmapMenu = menu.addMenu("Colormap")
        grp = QtWidgets.QActionGroup(cmapMenu)
        for name in CERR_COLORMAP_NAMES:
            act = cmapMenu.addAction(name)
            act.setCheckable(True)
            act.setChecked(name == self.cmapName)
            act.setIcon(self._cmap_icon(name))
            grp.addAction(act)
            act.triggered.connect(
                lambda _=False, n=name: self._on_cmap_selected(n))

        menu.addSeparator()
        menu.addAction("Set ranges...", self._edit_ranges)
        menu.addAction("Display range = colorbar range",
                       self._sync_disp_to_cbar)
        menu.addAction("Reset ranges",
                       lambda: self.mouseDoubleClickEvent(None))
        menu.exec_(ev.globalPos())

    def _cmap_icon(self, name, w=48, h=12):
        lut = cerr_get_lut(name, w)
        img = QtGui.QImage(w, h, QtGui.QImage.Format_RGB32)
        for x in range(w):
            c = QtGui.QColor(int(lut[x, 0]), int(lut[x, 1]), int(lut[x, 2]))
            for y in range(h):
                img.setPixelColor(x, y, c)
        return QtGui.QIcon(QtGui.QPixmap.fromImage(img))

    def _on_cmap_selected(self, name):
        self._set_cmap(name)
        self.update()
        self.rangesChanged.emit()

    def _edit_ranges(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Colorbar / display ranges")
        form = QtWidgets.QFormLayout(dlg)
        spins = []
        for label, val in (("Colorbar min:", self.cbarRange[0]),
                           ("Colorbar max:", self.cbarRange[1]),
                           ("Display min:", self.dispRange[0]),
                           ("Display max:", self.dispRange[1])):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setDecimals(3)
            sp.setRange(0.0, self.axisMax)
            sp.setSingleStep(self.axisMax / 100.0)
            sp.setValue(val)
            form.addRow(label, sp)
            spins.append(sp)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        def _apply():
            eps = self.axisMax * 1e-3
            cbLo, cbHi = spins[0].value(), spins[1].value()
            dLo, dHi = spins[2].value(), spins[3].value()
            self.cbarRange = [min(cbLo, cbHi - eps), max(cbHi, cbLo + eps)]
            self.dispRange = [min(dLo, dHi - eps), max(dHi, dLo + eps)]
            self.update()
            self.rangesChanged.emit()
        # Non-modal (a modal exec_ hangs in an integrated event loop): apply on OK.
        dlg.accepted.connect(_apply)
        dlg.setModal(False)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.show()
        dlg.raise_()

    def _sync_disp_to_cbar(self):
        self.dispRange = list(self.cbarRange)
        self.update()
        self.rangesChanged.emit()

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._move_to(ev.pos().y())
        else:
            self.setCursor(Qt.PointingHandCursor if self._hit_test(ev.pos())
                           else Qt.ArrowCursor)

    def mouseReleaseEvent(self, _ev):
        if self._drag:
            self._drag = None
            self.update()
            self.rangesChanged.emit()

    def mouseDoubleClickEvent(self, _ev):
        self.cbarRange = [0.0, self.axisMax]
        self.dispRange = [0.0, self.axisMax]
        self.update()
        self.rangesChanged.emit()

    def _move_to(self, y):
        which, j = self._drag
        rng = self.cbarRange if which == "cbar" else self.dispRange
        v = self._y2val(y)
        eps = self.axisMax * 1e-3
        rng[j] = min(v, rng[1] - eps) if j == 0 else max(v, rng[0] + eps)
        rng[j] = float(np.clip(rng[j], 0, self.axisMax))
        self.update()
        self.rangesChanged.emit()    # live update while dragging


# ---------------------------------------------------------------------------#
#  Main window
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


class PyCerrViewer(QtWidgets.QMainWindow):
    def __init__(self, planC=None):
        super().__init__()
        self.setWindowTitle("pyCERR Viewer")
        self.resize(1480, 920)
        # light-gray theme on this window (also covers the embedded case where
        # a host owns the QApplication and set its own palette)
        self.setPalette(_theme_palette())
        self.setStyleSheet(_THEME_STYLESHEET)
        self.setAcceptDrops(True)    # drag-drop DICOM dirs / NIfTI / .pkl

        self.planC = None
        self.scanNum = 0
        self.doseNum = -1            # -1 = no dose displayed
        self.doseAlpha = 0.45
        self.doseInterp = None       # RegularGridInterpolator
        self.maskCache = {}          # structNum -> 3D mask
        self.slices = {w: 0 for w in ("A", "B", "C", "D")}  # winId -> slice
        self.activeWins = ["A", "B", "C"]   # windows shown by current layout
        # last slice per view type, for crosshairs & newly switched windows
        self.lastSlice = {VIEW_AXIAL: 0, VIEW_SAGITTAL: 0, VIEW_CORONAL: 0}
        self.lockViews = False       # sync slices across same-view windows
        self.windowCenter, self.windowWidth = 40.0, 400.0
        self.scanCmap = "gray"       # base-scan colormap
        self.scanAlpha = 1.0         # base-scan opacity
        self.showCrosshairs = True
        self.showOrientation = True  # L/R/A/P/S/I edge labels
        self.showStructDots = False  # contour vertex dots (Alaly dots)
        self.structLineWidth = 1.4   # contour line width
        self.overlayState = {}       # scanIdx -> {"on", "alpha", "cmap"}
        self.overlayCache = {}       # scanIdx -> (interp, vmin, vmax) | None
        self.wlByScan = {}           # scanIdx -> (center, width)
        self.dispByScan = {}         # scanIdx -> (cmapName, alpha)
        self.doseCache = {}          # doseIdx -> (interp, doseMax) | None
        self._pvStructCache = {}     # structNum -> pyvista surface | None
        self._pvDoseCache = {}       # doseIdx -> (isosurface, doseMax) | None
        # per-axis overrides (CERR-style axis menu); None = "Auto" -> follow
        # the global panel selection. "dose" may also be -1 (no dose).
        self.axisSel = {w: {"scan": None, "dose": None, "structs": None}
                        for w in ("A", "B", "C", "D")}
        self._toolWindows = []       # keep IMRTP/ROE windows alive
        self.contourCtl = None       # active ContourDialog (or None)
        self.regCtl = None           # active RegQaDialog (or None)
        # IMRTP beam overlays: list of {"polylines": [Nx3,...], "color": rgb}
        self.beams = []
        # re-render 3D views shortly after slice scrolling stops
        self._timer3d = QtCore.QTimer(self)
        self._timer3d.setSingleShot(True)
        self._timer3d.timeout.connect(self._refresh_3d_views)

        self._build_menus()
        self._build_ui()
        self.statusBar().showMessage("Import a DICOM directory or NIfTI file to begin "
                                     "(File menu).")
        if planC is not None:
            self.setPlanC(planC)

    def setPlanC(self, planC):
        """Attach an existing pyCERR plan container and display it."""
        self.planC = planC
        self.wlByScan.clear()        # scan indices refer to a new plan now
        self.dispByScan.clear()
        self.overlayState.clear()
        self._reset_axis_sel()
        if planC is not None and planC.scan:
            self.after_load()
            self.statusBar().showMessage(
                f"planC loaded: {len(planC.scan)} scan(s), "
                f"{len(planC.structure)} structure(s), {len(planC.dose)} dose(s)")

    def getPlanC(self):
        """Return the current (possibly updated) plan container."""
        return self.planC

    # ----------------------------------------------- external (IMRTP) API ---
    def display_dose(self, doseNum):
        """Refresh from planC and show dose ``doseNum`` (called by IMRTP)."""
        self.after_load(keep_view=True)          # pick up newly added dose
        if self.planC is not None and 0 <= doseNum < len(self.planC.dose):
            self.doseCombo.setCurrentIndex(doseNum + 1)   # +1 for "None"
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def setBeams(self, beams):
        """Display IMRTP beam overlays (2D + 3D). ``beams`` is a list of
        {"polylines": [Nx3 arrays], "color": (r, g, b)} dicts; [] clears."""
        self.beams = list(beams) if beams else []
        self.refresh_views()

    # ------------------------------------------------ scripting/control API --
    # Public methods to drive the GUI programmatically. Each sets the relevant
    # widget(s) so the on-screen controls also reflect the change. Obtain the
    # viewer object from show()/launch(), e.g.:
    #     viewer = show(planC); viewer.set_layout("grid")
    def _window_for(self, orientation):
        """winId of the (first) window showing an orientation, or None."""
        for wid in self.activeWins:
            if self.views[wid].orientation == orientation:
                return wid
        return None

    def set_scan(self, scanNum):
        """Select the base scan (index into planC.scan)."""
        self.scanCombo.setCurrentIndex(int(scanNum))

    def set_dose(self, doseNum):
        """Select the displayed dose; doseNum < 0 (or None) hides dose."""
        idx = 0 if doseNum is None or doseNum < 0 else int(doseNum) + 1
        self.doseCombo.setCurrentIndex(idx)

    def set_dose_alpha(self, alpha):
        """Dose colorwash opacity, 0..1."""
        self.alphaSlider.setValue(int(round(float(alpha) * 100)))

    def set_window_level(self, center, width):
        """Manual window center/width for the base scan."""
        self.centerSpin.setValue(float(center))
        self.widthSpin.setValue(float(width))

    def set_window_preset(self, name):
        """Apply a named CT window preset (see CT_WINDOW_PRESETS)."""
        self.presetCombo.setCurrentText(name)

    def set_scan_colormap(self, name):
        self.scanCmapCombo.setCurrentText(name)

    def set_scan_opacity(self, alpha):
        self.scanAlphaSlider.setValue(int(round(float(alpha) * 100)))

    def set_orientation(self, winId, orientation):
        """Set which orientation a window (A/B/C/D) displays (or VIEW_3D)."""
        self._set_axis_view(winId, orientation)

    def set_slice(self, orientation, k):
        """Move the window showing ``orientation`` to slice index ``k``."""
        wid = self._window_for(orientation)
        if wid is not None:
            n = self._slice_count(orientation)
            self.views[wid].slider.setValue(int(np.clip(k, 0, n - 1)))

    def set_structures_visible(self, which):
        """Show structures: 'all', 'none', or a list of structure indices."""
        if which == "all":
            self._set_all_structs(True)
            return
        if which == "none":
            self._set_all_structs(False)
            return
        want = set(int(i) for i in which)
        self.structList.blockSignals(True)
        for i in range(self.structList.count()):
            it = self.structList.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in want
                             else Qt.Unchecked)
        self.structList.blockSignals(False)
        self.refresh_views()

    def goto_structure(self, strNum):
        """Center all views on a structure's center of mass."""
        self.goto_struct_center(int(strNum))

    def _view_under_cursor(self):
        """Return this window's SliceView currently under the mouse, or None."""
        w = QtWidgets.QApplication.widgetAt(QtGui.QCursor.pos())
        while w is not None and not isinstance(w, SliceView):
            w = w.parentWidget()
        if isinstance(w, SliceView) and self.views.get(w.winId) is w:
            return w
        return None

    def eventFilter(self, obj, event):
        # Arrow keys step the slice of the 2D view under the mouse (hover-based
        # navigation, independent of which widget has keyboard focus).
        if event.type() == QtCore.QEvent.KeyPress and event.key() in (
                Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            view = self._view_under_cursor()
            if view is not None and not view.is3d:
                step = 1 if event.key() in (Qt.Key_Up, Qt.Key_Right) else -1
                sl = view.slider
                sl.setValue(int(np.clip(sl.value() + step,
                                        sl.minimum(), sl.maximum())))
                return True   # consume so a focused slider/widget doesn't also move
        return super().eventFilter(obj, event)

    def set_crosshairs(self, on):
        self.actXhair.setChecked(bool(on))

    def set_orientation_labels(self, on):
        self.actOrient.setChecked(bool(on))

    def set_structure_dots(self, on):
        self.dotsChk.setChecked(bool(on))

    def set_contour_linewidth(self, width):
        self.lineWidthSpin.setValue(float(width))

    def set_lock_views(self, on):
        self.actLock.setChecked(bool(on))

    # -------------------------------------------------------- screenshots ----
    def save_screenshot(self, path, target="window", dpi=150):
        """Save a screenshot to ``path`` (PNG by file extension).

        target:
          "window" - the whole GUI window
          "views"  - just the panel of view windows
          "A"/"B"/"C"/"D" - a single view window (2D figure or 3D render)
          an orientation ("Axial"/"Sagittal"/"Coronal"/"3D") - that view
          "3d"     - the 3D view (whichever window shows it)
        Returns the path written.
        """
        target = str(target)
        QtWidgets.QApplication.processEvents()
        if target == "window":
            self.grab().save(path)
        elif target == "views":
            self.viewContainer.grab().save(path)
        else:
            wid = target if target in self.views else self._window_for(
                VIEW_3D if target.lower() == "3d" else target)
            if wid is None:
                raise ValueError(f"No view window for target {target!r}")
            v = self.views[wid]
            if v.is3d and v.vtk_widget is not None:
                v.vtk_widget.screenshot(path)
            elif v.is3d:
                v.fig.savefig(path, dpi=dpi, facecolor="black")
            else:
                v.fig.savefig(path, dpi=dpi, facecolor="black",
                              bbox_inches="tight", pad_inches=0)
        return path

    # ----------------------------------------------- registration QA / DVH --
    def start_reg_qa(self, base=None, moving=None, mode="Mirrorscope",
                     size=None, base_frac=None):
        """Open and configure the registration QA tool programmatically.

        base/moving: scan indices; mode: 'Mirrorscope'|'Sidebyside'|
        'AlternateGrid'|'Toggle'; size: mirror-box/tile size (cm);
        base_frac: Toggle-mode base weight (0..1). Returns the RegQaDialog.
        """
        if self.planC is None or len(self.planC.scan) < 2:
            raise ValueError("Registration QA needs at least two scans.")
        if self.regCtl is None or not self.regCtl.isVisible():
            RegQaDialog(self).show()       # sets self.regCtl
        self.regCtl.configure(base=base, moving=moving, mode=mode, size=size,
                              base_frac=base_frac)
        return self.regCtl

    def stop_reg_qa(self):
        """Close the registration QA tool if open."""
        if self.regCtl is not None:
            self.regCtl.close()

    def compute_dvh(self, doseNum=None, structNums=None, num_bins=400):
        """Compute cumulative DVHs. Returns (doseAxis_Gy, {name: volPct}),
        each structure interpolated onto a shared dose axis (0..global max)."""
        planC = self.planC
        if planC is None or not planC.dose:
            raise ValueError("A dose is required to compute DVHs.")
        if doseNum is None:
            doseNum = self.doseNum if self.doseNum >= 0 else 0
        if not 0 <= doseNum < len(planC.dose):
            raise ValueError(f"doseNum {doseNum} out of range.")
        if structNums is None:
            structNums = list(range(len(planC.structure)))
        elif isinstance(structNums, (int, np.integer)):
            structNums = [int(structNums)]

        raw, gmax = {}, 0.0
        for strNum in structNums:
            try:
                dosesV, volsV, isErr = cerrDvh.getDVH(strNum, doseNum, planC)
            except Exception:  # noqa: BLE001
                continue
            if isErr or dosesV is None or len(dosesV) == 0:
                continue
            binWidth = max(float(np.max(dosesV)) / 400.0, 1e-3)
            doseBinsV, volsHistV = cerrDvh.doseHist(dosesV, volsV, binWidth)
            cumVols = np.flip(np.cumsum(np.flip(volsHistV)))
            if cumVols[0] <= 0:
                continue
            raw[strNum] = (np.asarray(doseBinsV, dtype=float),
                           100.0 * cumVols / cumVols[0])
            gmax = max(gmax, float(doseBinsV[-1]))
        if not raw:
            raise ValueError("No DVH could be computed for the given "
                             "structures/dose.")
        axis = np.linspace(0.0, gmax, int(num_bins))
        table = {}
        for strNum, (db, cp) in raw.items():
            name = planC.structure[strNum].structureName
            table[name] = np.interp(axis, db, cp, left=cp[0], right=0.0)
        return axis, table

    def export_dvh(self, path, doseNum=None, structNums=None, num_bins=400):
        """Compute cumulative DVHs and write them to a CSV (wide format:
        Dose(Gy), <struct>, ...). Returns (doseAxis, {name: volPct})."""
        import csv
        axis, table = self.compute_dvh(doseNum=doseNum, structNums=structNums,
                                       num_bins=num_bins)
        names = list(table.keys())
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Dose(Gy)"] + names)
            for i in range(len(axis)):
                w.writerow([f"{axis[i]:.5g}"]
                           + [f"{table[n][i]:.4f}" for n in names])
        return axis, table

    def closeEvent(self, event):
        # release VTK render windows cleanly to avoid shutdown crashes
        for v in self.views.values():
            if v.vtk_widget is not None:
                try:
                    v.vtk_widget.close()
                except Exception:  # noqa: BLE001
                    pass
                v.vtk_widget = None
        super().closeEvent(event)

    # ------------------------------------------------------------------ UI --
    def _build_menus(self):
        m = self.menuBar()
        fileM = m.addMenu("&File")
        fileM.addAction("Import &DICOM directory...", self.import_dicom)
        fileM.addAction("Import &NIfTI scan...", self.import_nii_scan)
        self.actNiiDose = fileM.addAction("Import NIfTI d&ose...", self.import_nii_dose)
        self.actNiiStr = fileM.addAction("Import NIfTI &structure(s)...",
                                         self.import_nii_struct)
        fileM.addSeparator()
        fileM.addAction("&Open planC (.pkl)...", self.open_pkl)
        fileM.addAction("&Save planC (.pkl)...", self.save_pkl)
        fileM.addSeparator()
        fileM.addAction("E&xit", self.close)

        viewM = m.addMenu("&View")
        layoutM = viewM.addMenu("&Layout")
        lgrp = QtWidgets.QActionGroup(layoutM)
        for key, label in (("single", "&One view"),
                           ("two", "&Two views side-by-side"),
                           ("default", "One large + two stacke&d"),
                           ("columns", "Three &columns"),
                           ("grid", "&Four views (2x2)")):
            act = layoutM.addAction(label)
            act.setCheckable(True)
            act.setChecked(key == "default")
            lgrp.addAction(act)
            act.triggered.connect(lambda _=False, k=key: self.set_layout(k))
        viewM.addSeparator()
        self.actXhair = viewM.addAction("Show &crosshairs")
        self.actXhair.setCheckable(True)
        self.actXhair.setChecked(True)
        self.actXhair.setShortcut("X")
        self.actXhair.toggled.connect(self.on_crosshair_toggled)
        self.actOrient = viewM.addAction("Show &orientation labels (L/R/A/P/S/I)")
        self.actOrient.setCheckable(True)
        self.actOrient.setChecked(True)
        self.actOrient.setShortcut("O")
        self.actOrient.toggled.connect(self.on_orientation_toggled)
        self.actLock = viewM.addAction("&Lock slices across matching views")
        self.actLock.setCheckable(True)
        self.actLock.setShortcut("L")
        self.actLock.toggled.connect(self.on_lock_toggled)
        actReset = viewM.addAction("&Reset pan/zoom", self.reset_all_views)
        actReset.setShortcut("R")

        toolsM = m.addMenu("&Tools")
        toolsM.addAction("&Contouring (draw/edit structures)...",
                         self.show_contour_dialog)
        toolsM.addAction("&DVH...", self.show_dvh_dialog)
        toolsM.addAction("Registration &QA (compare scans)...",
                         self.show_reg_dialog)
        toolsM.addSeparator()
        toolsM.addAction("&IMRTP (beamlet dose calculation)...",
                         self.show_imrtp_gui)
        toolsM.addAction("&ROE (Radiotherapy Outcomes Explorer)...",
                         self.show_roe_gui)
        toolsM.addSeparator()
        toolsM.addAction("Re&fresh from planC",
                         lambda: self.after_load(keep_view=True))

        helpM = m.addMenu("&Help")
        helpM.addAction("&Controls", self.show_controls)
        helpM.addAction("&About", self.show_about)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)
        h.setContentsMargins(2, 2, 2, 2)

        # ---------------- left control panel (CERR-style) ----------------
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(290)
        pl = QtWidgets.QVBoxLayout(panel)

        grpScan = QtWidgets.QGroupBox("Scan")
        gl = QtWidgets.QVBoxLayout(grpScan)
        self.scanCombo = QtWidgets.QComboBox()
        self.scanCombo.currentIndexChanged.connect(self.on_scan_changed)
        gl.addWidget(self.scanCombo)

        wlRow = QtWidgets.QHBoxLayout()
        self.presetCombo = QtWidgets.QComboBox()
        self.presetCombo.addItems(CT_WINDOW_PRESETS.keys())
        self.presetCombo.setCurrentText("Soft Tissue")
        self.presetCombo.currentTextChanged.connect(self.on_preset)
        wlRow.addWidget(QtWidgets.QLabel("Window:"))
        wlRow.addWidget(self.presetCombo, 1)
        gl.addLayout(wlRow)

        cwRow = QtWidgets.QHBoxLayout()
        self.centerSpin = QtWidgets.QDoubleSpinBox()
        self.centerSpin.setRange(-5000, 50000)
        self.centerSpin.setValue(self.windowCenter)
        self.widthSpin = QtWidgets.QDoubleSpinBox()
        self.widthSpin.setRange(0.01, 100000)
        self.widthSpin.setValue(self.windowWidth)
        for s in (self.centerSpin, self.widthSpin):
            s.valueChanged.connect(self.on_manual_wl)
        cwRow.addWidget(QtWidgets.QLabel("C:"))
        cwRow.addWidget(self.centerSpin)
        cwRow.addWidget(QtWidgets.QLabel("W:"))
        cwRow.addWidget(self.widthSpin)
        gl.addLayout(cwRow)

        cmRow = QtWidgets.QHBoxLayout()
        cmRow.addWidget(QtWidgets.QLabel("Colormap:"))
        self.scanCmapCombo = QtWidgets.QComboBox()
        self.scanCmapCombo.addItems(SCAN_CMAPS)
        self.scanCmapCombo.setCurrentText(self.scanCmap)
        self.scanCmapCombo.currentTextChanged.connect(self.on_scan_cmap)
        cmRow.addWidget(self.scanCmapCombo, 1)
        gl.addLayout(cmRow)

        opRow = QtWidgets.QHBoxLayout()
        opRow.addWidget(QtWidgets.QLabel("Opacity:"))
        self.scanAlphaSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.scanAlphaSlider.setRange(0, 100)
        self.scanAlphaSlider.setValue(int(self.scanAlpha * 100))
        self.scanAlphaSlider.valueChanged.connect(self.on_scan_alpha)
        opRow.addWidget(self.scanAlphaSlider)
        gl.addLayout(opRow)
        pl.addWidget(grpScan)

        # fused scan overlays (visible only when >1 scan is loaded)
        self.grpOverlay = QtWidgets.QGroupBox("Scan overlays (fusion)")
        ovl = QtWidgets.QVBoxLayout(self.grpOverlay)
        ovl.setContentsMargins(4, 4, 4, 4)
        overlayHost = QtWidgets.QWidget()
        self.overlayLayout = QtWidgets.QVBoxLayout(overlayHost)
        self.overlayLayout.setContentsMargins(0, 0, 0, 0)
        self.overlayLayout.setSpacing(2)
        self.overlayLayout.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setMaximumHeight(150)
        scroll.setWidget(overlayHost)
        ovl.addWidget(scroll)
        self.grpOverlay.setVisible(False)
        pl.addWidget(self.grpOverlay)

        grpStr = QtWidgets.QGroupBox("Structures")
        sl = QtWidgets.QVBoxLayout(grpStr)
        btnRow = QtWidgets.QHBoxLayout()
        allBtn = QtWidgets.QPushButton("All")
        noneBtn = QtWidgets.QPushButton("None")
        allBtn.clicked.connect(lambda: self._set_all_structs(True))
        noneBtn.clicked.connect(lambda: self._set_all_structs(False))
        btnRow.addWidget(allBtn), btnRow.addWidget(noneBtn)
        sl.addLayout(btnRow)
        self.structList = QtWidgets.QListWidget()
        self.structList.itemChanged.connect(lambda *_: self.refresh_views())
        self.structList.itemDoubleClicked.connect(self.on_struct_double_click)
        self.structList.setToolTip(
            "Double-click a structure to center all views on it")
        # black list background so the per-structure colored names all show
        self.structList.setStyleSheet(
            "QListWidget { background: #000; }"
            "QListWidget::item:selected { background: #3a6ea5; }")
        sl.addWidget(self.structList)

        optRow = QtWidgets.QHBoxLayout()
        self.dotsChk = QtWidgets.QCheckBox("Dots")
        self.dotsChk.setToolTip("Show contour vertex dots (Alaly dots)")
        self.dotsChk.toggled.connect(self.on_struct_dots)
        optRow.addWidget(self.dotsChk)
        optRow.addWidget(QtWidgets.QLabel("Line:"))
        self.lineWidthSpin = QtWidgets.QDoubleSpinBox()
        self.lineWidthSpin.setRange(0.2, 6.0)
        self.lineWidthSpin.setSingleStep(0.2)
        self.lineWidthSpin.setValue(self.structLineWidth)
        self.lineWidthSpin.setToolTip("Contour line width")
        self.lineWidthSpin.valueChanged.connect(self.on_struct_linewidth)
        optRow.addWidget(self.lineWidthSpin)
        optRow.addStretch(1)
        sl.addLayout(optRow)
        pl.addWidget(grpStr, stretch=1)

        grpDose = QtWidgets.QGroupBox("Dose")
        dl = QtWidgets.QVBoxLayout(grpDose)
        self.doseCombo = QtWidgets.QComboBox()
        self.doseCombo.currentIndexChanged.connect(self.on_dose_changed)
        dl.addWidget(self.doseCombo)
        aRow = QtWidgets.QHBoxLayout()
        aRow.addWidget(QtWidgets.QLabel("Colorwash alpha:"))
        self.alphaSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.alphaSlider.setRange(0, 100)
        self.alphaSlider.setValue(int(self.doseAlpha * 100))
        self.alphaSlider.valueChanged.connect(self.on_alpha)
        aRow.addWidget(self.alphaSlider)
        dl.addLayout(aRow)
        pl.addWidget(grpDose)

        h.addWidget(panel)

        # ------- right: view windows (each can show any orientation) -------
        self.views = {
            "A": SliceView("A", VIEW_AXIAL),
            "B": SliceView("B", VIEW_SAGITTAL),
            "C": SliceView("C", VIEW_CORONAL),
            "D": SliceView("D", VIEW_3D),       # 4th window: 3D by default
        }
        self.views["D"].set_projection(True)
        for v in self.views.values():
            v.sliceChanged.connect(self.on_slice_changed)
            v.cursorMoved.connect(self.on_cursor_moved)
            v.viewReset.connect(lambda winId: self.refresh_views(only=winId))
            v.contextRequested.connect(self._show_axis_menu)
            v.rulerChanged.connect(self.on_ruler_changed)
            v.crosshairDragged.connect(self.on_crosshair_dragged)

        # Arrow keys (Up/Down/Left/Right) change the slice of whichever 2D view
        # is under the mouse, regardless of focus.
        QtWidgets.QApplication.instance().installEventFilter(self)

        # container whose contents are rebuilt by _apply_layout()
        self.viewContainer = QtWidgets.QWidget()
        self.viewLay = QtWidgets.QVBoxLayout(self.viewContainer)
        self.viewLay.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.viewContainer, stretch=1)
        self._apply_layout("default")

        # standalone dose colorbar (hidden until a dose is selected)
        self.colorbar = DoseColorbarWidget()
        self.colorbar.rangesChanged.connect(lambda: self.refresh_views())
        self.colorbar.setVisible(False)
        h.addWidget(self.colorbar)

    # --------------------------------------------------------------- I/O ----
    def _busy(self, msg):
        self.statusBar().showMessage(msg)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()

    def _done(self, msg="Ready."):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(msg)

    def import_dicom(self, path=None):
        if not path:
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select DICOM directory")
        if not path:
            return
        self._do_dicom_import(path, f"Importing DICOM from {path} ...",
                              f"Imported {path}")

    def import_dicom_files(self, fileList):
        """Import only the given DICOM file(s), not their whole directory -
        e.g. a dropped RTSTRUCT, so we don't pull in the rest of the folder."""
        fileList = [f for f in fileList if f]
        if not fileList:
            return
        n = len(fileList)
        self._do_dicom_import(list(fileList),
                              f"Importing {n} DICOM file(s) ...",
                              f"Imported {n} DICOM file(s)")

    def _do_dicom_import(self, source, busyMsg, doneMsg):
        """Shared DICOM import: source is a directory path or a list of file
        paths (both accepted by loadDcmDir). Reports skipped duplicates."""
        try:
            self._busy(busyMsg)
            before = ((len(self.planC.scan), len(self.planC.structure),
                       len(self.planC.dose)) if self.planC else (0, 0, 0))
            import warnings as _warnings
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                self.planC = pc.loadDcmDir(source, initplanC=self.planC or "")
            self.after_load()
            self._done(doneMsg)
            self._report_skipped_dups(caught, before)
        except Exception as e:  # noqa: BLE001
            self._done()
            _show_error(self, "Import error", str(e))

    def _report_skipped_dups(self, caught, before):
        """Surface duplicates skipped by loadDcmDir (per-category, accurate for
        both fully- and partially-duplicate imports)."""
        msgs = [str(w.message) for w in caught
                if "already exists in planC" in str(w.message)]
        if not msgs:
            return
        after = (len(self.planC.scan), len(self.planC.structure),
                 len(self.planC.dose))
        lines = []
        for i, (label, prefix) in enumerate(
                (("scan", "Scan"), ("structure", "Structure"),
                 ("dose", "Dose"))):
            skipped = sum(1 for m in msgs if m.startswith(prefix))
            if skipped == 0:
                continue
            if after[i] - before[i] == 0:
                lines.append(f"No {label}s were imported as they already "
                             f"exist in planC.")
            else:
                lines.append(f"{skipped} {label}{'' if skipped == 1 else 's'} "
                             f"already exist in planC and "
                             f"{'was' if skipped == 1 else 'were'} not "
                             f"imported.")
        if lines:
            _show_info(self, "Import", "\n".join(lines))

    # ----------------------------------------------------- drag & drop ------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.toLocalFile()]
        event.acceptProposedAction()
        # Dropped DICOM *files* are imported as exactly those files (e.g. a
        # single RTSTRUCT), NOT their whole folder - dropping one structure
        # file must not pull in the scan series sitting next to it. Dropped
        # *directories* still import the whole directory.
        dcmFiles, dirs, others = [], [], []
        for p in paths:
            if os.path.isdir(p):
                dirs.append(p)
            elif os.path.isfile(p) and p.lower().endswith(".dcm"):
                dcmFiles.append(p)
            else:
                others.append(p)
        # directories (scans) first, then the dropped DICOM files (structures/
        # doses that reference them), then dose/structure NIfTIs
        for d in dirs:
            self.load_path(d)
        if dcmFiles:
            self.import_dicom_files(dcmFiles)
        for p in sorted(others, key=self._drop_priority):
            self.load_path(p)

    @staticmethod
    def _drop_priority(path):
        low = os.path.basename(path).lower()
        if low.endswith((".nii", ".nii.gz")) and any(
                k in low for k in ("dose", "mask", "label", "seg", "struct",
                                   "roi")):
            return 1                  # dose/structure NIfTIs load last
        return 0

    def load_path(self, path):
        """Load a dropped/opened path: a DICOM directory, a NIfTI file
        (.nii/.nii.gz - scan, or dose/structure by name heuristic when a scan
        is loaded), a single DICOM file (only that file is imported), or a
        .pkl plan container."""
        path = str(path)
        low = path.lower()
        try:
            if os.path.isdir(path):
                self.import_dicom(path)
            elif low.endswith(".pkl"):
                self._busy("Loading planC ...")
                self.planC = pc.loadPlanCFromPkl(path)
                self.wlByScan.clear()
                self.dispByScan.clear()
                self.overlayState.clear()
                self._reset_axis_sel()
                self.after_load()
                self._done(f"Loaded {path}")
            elif low.endswith((".nii", ".nii.gz")):
                self._load_nii(path)
            elif low.endswith(".dcm"):
                self.import_dicom_files([path])
            else:
                _show_info(
                    self, "Open", f"Unsupported file type:\n{path}")
        except Exception as e:  # noqa: BLE001
            self._done()
            _show_error(self, "Load error", str(e))

    def _load_nii(self, path):
        """Load a NIfTI as scan / dose / structure (by filename heuristic)."""
        name = os.path.basename(path).lower()
        hasScan = self.planC is not None and bool(self.planC.scan)
        if hasScan and "dose" in name:
            self._busy("Loading NIfTI dose ...")
            self.planC = pc.loadNiiDose(path, self.scanNum, self.planC)
            self.after_load(keep_view=True)
            self._done(f"Loaded dose {path}")
        elif hasScan and any(k in name for k in ("mask", "label", "seg",
                                                 "struct", "roi")):
            self._busy("Loading NIfTI structure(s) ...")
            self.planC = pc.loadNiiStructure(path, self.scanNum, self.planC)
            self.after_load(keep_view=True)
            self._done(f"Loaded structures from {path}")
        else:
            self._busy("Loading NIfTI scan ...")
            self.planC = pc.loadNiiScan(path, imageType="CT SCAN",
                                        initplanC=self.planC or "")
            self.after_load()
            self._done(f"Loaded {path}")

    def import_nii_scan(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select NIfTI scan", filter="NIfTI (*.nii *.nii.gz)")
        if not f:
            return
        try:
            self._busy("Loading NIfTI scan ...")
            self.planC = pc.loadNiiScan(f, imageType="CT SCAN",
                                        initplanC=self.planC or "")
            self.after_load()
            self._done(f"Loaded {f}")
        except Exception as e:  # noqa: BLE001
            self._done()
            _show_error(self, "Import error", str(e))

    def import_nii_dose(self):
        if self.planC is None or not self.planC.scan:
            _show_info(self, "pyCERR",
                                              "Load a scan first.")
            return
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select NIfTI dose", filter="NIfTI (*.nii *.nii.gz)")
        if not f:
            return
        try:
            self._busy("Loading NIfTI dose ...")
            self.planC = pc.loadNiiDose(f, self.scanNum, self.planC)
            self.after_load(keep_view=True)
            self._done(f"Loaded dose {f}")
        except Exception as e:  # noqa: BLE001
            self._done()
            _show_error(self, "Import error", str(e))

    def import_nii_struct(self):
        if self.planC is None or not self.planC.scan:
            _show_info(self, "pyCERR",
                                              "Load a scan first.")
            return
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select NIfTI label mask", filter="NIfTI (*.nii *.nii.gz)")
        if not f:
            return
        try:
            self._busy("Loading NIfTI structure(s) ...")
            self.planC = pc.loadNiiStructure(f, self.scanNum, self.planC)
            self.after_load(keep_view=True)
            self._done(f"Loaded structures from {f}")
        except Exception as e:  # noqa: BLE001
            self._done()
            _show_error(self, "Import error", str(e))

    def open_pkl(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open planC pickle", filter="Pickle (*.pkl)")
        if not f:
            return
        try:
            self._busy("Loading planC ...")
            self.planC = pc.loadPlanCFromPkl(f)
            self.wlByScan.clear()    # scan indices refer to a new plan now
            self.dispByScan.clear()
            self.overlayState.clear()
            self._reset_axis_sel()
            self.after_load()
            self._done(f"Loaded {f}")
        except Exception as e:  # noqa: BLE001
            self._done()
            _show_error(self, "Load error", str(e))

    def save_pkl(self):
        if self.planC is None:
            return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save planC pickle", filter="Pickle (*.pkl)")
        if not f:
            return
        try:
            import pickle
            with open(f, "wb") as fh:
                pickle.dump(self.planC, fh)
            self.statusBar().showMessage(f"Saved {f}")
        except Exception as e:  # noqa: BLE001
            _show_error(self, "Save error", str(e))

    # ----------------------------------------------------------- loading ----
    def after_load(self, keep_view=False):
        """Refresh combo boxes / lists after planC changes."""
        if self.planC is None or not self.planC.scan:
            return
        self.maskCache.clear()
        self.overlayCache.clear()
        self.doseCache.clear()
        self._pvStructCache.clear()
        self._pvDoseCache.clear()
        prevScan = self.scanNum if keep_view else 0

        self.scanCombo.blockSignals(True)
        self.scanCombo.clear()
        for i, s in enumerate(self.planC.scan):
            mod = getattr(s.scanInfo[0], "imageType", "scan")
            self.scanCombo.addItem(f"{i}: {mod}")
        self.scanCombo.setCurrentIndex(min(prevScan, len(self.planC.scan) - 1))
        self.scanCombo.blockSignals(False)

        self.doseCombo.blockSignals(True)
        self.doseCombo.clear()
        self.doseCombo.addItem("None")
        for i, d in enumerate(self.planC.dose):
            self.doseCombo.addItem(f"{i}: {getattr(d, 'fractionGroupID', 'dose')}")
        if self.planC.dose and not keep_view:
            self.doseCombo.setCurrentIndex(1)
        self.doseCombo.blockSignals(False)

        self.scanNum = self.scanCombo.currentIndex()
        self.doseNum = self.doseCombo.currentIndex() - 1
        self._load_scan_geometry(reset_slices=not keep_view)
        self._populate_struct_list()
        self._populate_overlay_rows()
        self._build_dose_interp()
        self.refresh_views()

    def _slice_count(self, orientation):
        nR, nC, nS = self.scan3M.shape
        return {VIEW_AXIAL: nS, VIEW_SAGITTAL: nC, VIEW_CORONAL: nR}[orientation]

    def _load_scan_geometry(self, reset_slices=True):
        scanObj = self.planC.scan[self.scanNum]
        self.scan3M = scanObj.getScanArray().astype(np.float32)
        self.xV, self.yV, self.zV = scanObj.getScanXYZVals()
        nR, nC, nS = self.scan3M.shape
        if reset_slices:
            self.lastSlice = {VIEW_AXIAL: nS // 2,
                              VIEW_SAGITTAL: nC // 2,
                              VIEW_CORONAL: nR // 2}
            for wid, v in self.views.items():
                if not v.is3d:
                    self.slices[wid] = self.lastSlice[v.orientation]
                v.user_limits = None    # new geometry -> drop pan/zoom
                v._pan = None
                v._zoom_drag = None
        for orient in (VIEW_AXIAL, VIEW_SAGITTAL, VIEW_CORONAL):
            self.lastSlice[orient] = min(self.lastSlice[orient],
                                         self._slice_count(orient) - 1)
        for wid, v in self.views.items():
            if v.is3d:
                continue
            n = self._slice_count(v.orientation)
            self.slices[wid] = min(self.slices[wid], n - 1)
            v.set_range(n, self.slices[wid])

        # restore this scan's saved window; auto-window non-CT on first view
        if self.scanNum in self.wlByScan:
            self.windowCenter, self.windowWidth = self.wlByScan[self.scanNum]
        else:
            mod = str(getattr(scanObj.scanInfo[0], "imageType", "")).upper()
            if "CT" not in mod:
                lo, hi = np.percentile(self.scan3M, [2, 98])
                self.windowCenter, self.windowWidth = \
                    (lo + hi) / 2, max(hi - lo, 1)
            self.wlByScan[self.scanNum] = (self.windowCenter, self.windowWidth)
        for sp, val in ((self.centerSpin, self.windowCenter),
                        (self.widthSpin, self.windowWidth)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)

        # restore this scan's saved colormap & opacity (default gray / 1.0)
        self.scanCmap, self.scanAlpha = self.dispByScan.setdefault(
            self.scanNum, ("gray", 1.0))
        self.scanCmapCombo.blockSignals(True)
        self.scanCmapCombo.setCurrentText(self.scanCmap)
        self.scanCmapCombo.blockSignals(False)
        self.scanAlphaSlider.blockSignals(True)
        self.scanAlphaSlider.setValue(int(round(self.scanAlpha * 100)))
        self.scanAlphaSlider.blockSignals(False)

    def _populate_struct_list(self):
        self.structList.blockSignals(True)
        self.structList.clear()
        for i, st in enumerate(self.planC.structure):
            item = QtWidgets.QListWidgetItem(f"{i}: {st.structureName}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, i)
            col = np.asarray(st.structureColor, dtype=float).ravel()
            rgb = (col / 255.0 if col.size == 3 and col.max() > 1
                   else col if col.size == 3 else np.array([1, 0, 0]))
            item.setForeground(QtGui.QColor.fromRgbF(*np.clip(rgb, 0, 1)))
            self.structList.addItem(item)
        self.structList.blockSignals(False)

    def _set_all_structs(self, on):
        self.structList.blockSignals(True)
        for i in range(self.structList.count()):
            self.structList.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.structList.blockSignals(False)
        self.refresh_views()

    # ------------------------------------------------------ scan overlays ---
    def _populate_overlay_rows(self):
        """One row per non-base scan: visibility, colormap & opacity."""
        while self.overlayLayout.count() > 1:      # keep trailing stretch
            w = self.overlayLayout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        nScans = len(self.planC.scan) if self.planC else 0
        self.grpOverlay.setVisible(nScans > 1)
        if nScans < 2:
            return
        for i, s in enumerate(self.planC.scan):
            if i == self.scanNum:
                continue
            st = self.overlayState.setdefault(
                i, {"on": False, "alpha": 0.5, "cmap": "hot"})
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(3)
            mod = getattr(s.scanInfo[0], "imageType", "scan")
            cb = QtWidgets.QCheckBox(f"{i}: {mod}")
            cb.setChecked(st["on"])
            cb.setToolTip("Overlay this scan on the base scan")
            cmapC = QtWidgets.QComboBox()
            cmapC.addItems(OVERLAY_CMAPS)
            cmapC.setCurrentText(st["cmap"])
            cmapC.setFixedWidth(62)
            cmapC.setToolTip("Overlay colormap")
            sld = QtWidgets.QSlider(Qt.Horizontal)
            sld.setRange(0, 100)
            sld.setValue(int(st["alpha"] * 100))
            sld.setFixedWidth(68)
            sld.setToolTip("Overlay opacity")
            cb.toggled.connect(
                lambda on, k=i: self._on_overlay_changed(k, "on", on))
            cmapC.currentTextChanged.connect(
                lambda t, k=i: self._on_overlay_changed(k, "cmap", t))
            sld.valueChanged.connect(
                lambda v, k=i: self._on_overlay_changed(k, "alpha", v / 100.0))
            rl.addWidget(cb, 1)
            rl.addWidget(cmapC)
            rl.addWidget(sld)
            self.overlayLayout.insertWidget(self.overlayLayout.count() - 1, row)

    def _on_overlay_changed(self, scanIdx, key, val):
        self.overlayState.setdefault(
            scanIdx, {"on": False, "alpha": 0.5, "cmap": "hot"})[key] = val
        self.refresh_views()

    def _overlay_interp(self, scanIdx):
        """Interpolator + auto 2-98 percentile window for an overlay scan."""
        if scanIdx not in self.overlayCache:
            try:
                sObj = self.planC.scan[scanIdx]
                a3M = sObj.getScanArray().astype(np.float32)
                xs, ys, zs = sObj.getScanXYZVals()
                ys, a3M = ascending(ys, a3M, axis=0)
                xs, a3M = ascending(xs, a3M, axis=1)
                zs, a3M = ascending(zs, a3M, axis=2)
                lo, hi = np.percentile(a3M, [2, 98])
                interp = RegularGridInterpolator(
                    (ys, xs, zs), a3M, bounds_error=False, fill_value=np.nan)
                self.overlayCache[scanIdx] = (interp, float(lo),
                                              float(max(hi, lo + 1e-3)))
            except Exception:  # noqa: BLE001
                self.overlayCache[scanIdx] = None
        return self.overlayCache[scanIdx]

    def _dose_interp(self, doseIdx):
        """Cached (interpolator, doseMax) for a dose index, or None."""
        if self.planC is None or doseIdx is None or doseIdx < 0 \
                or doseIdx >= len(self.planC.dose):
            return None
        if doseIdx not in self.doseCache:
            try:
                d = self.planC.dose[doseIdx]
                dose3M = np.asarray(d.doseArray, dtype=np.float32)
                xD, yD, zD = d.getDoseXYZVals()
                yD, dose3M = ascending(yD, dose3M, axis=0)
                xD, dose3M = ascending(xD, dose3M, axis=1)
                zD, dose3M = ascending(zD, dose3M, axis=2)
                interp = RegularGridInterpolator(
                    (yD, xD, zD), dose3M, bounds_error=False, fill_value=0.0)
                self.doseCache[doseIdx] = (interp, float(dose3M.max()))
            except Exception:  # noqa: BLE001
                self.doseCache[doseIdx] = None
        return self.doseCache[doseIdx]

    def _build_dose_interp(self):
        """Point the colorbar/status readout at the globally selected dose."""
        res = self._dose_interp(self.doseNum)
        if res is None:
            self.doseInterp = None
            self.doseMax = 0.0
            self.colorbar.setVisible(False)
            return
        self.doseInterp, self.doseMax = res
        self.colorbar.setDose(self.doseMax)
        self.colorbar.setVisible(True)

    # ----------------------------------------------------------- callbacks --
    def on_scan_changed(self, idx):
        if self.planC is None or idx < 0:
            return
        self.scanNum = idx
        self.maskCache.clear()
        self._pvStructCache.clear()   # surfaces live on the scan grid
        self._load_scan_geometry()
        self._populate_overlay_rows()   # base scan is excluded from overlays
        if self.regCtl is not None:     # keep QA base in sync with the scan
            self.regCtl.sync_base(idx)
        self.refresh_views()

    def on_dose_changed(self, idx):
        self.doseNum = idx - 1
        self._build_dose_interp()
        self.refresh_views()

    def on_alpha(self, val):
        self.doseAlpha = val / 100.0
        self.refresh_views()

    def on_scan_cmap(self, name):
        self.scanCmap = name
        self.dispByScan[self.scanNum] = (self.scanCmap, self.scanAlpha)
        self.refresh_views()

    def on_scan_alpha(self, val):
        self.scanAlpha = val / 100.0
        self.dispByScan[self.scanNum] = (self.scanCmap, self.scanAlpha)
        self.refresh_views()

    def on_preset(self, name):
        preset = CT_WINDOW_PRESETS.get(name)
        if preset is None:
            return
        self.windowCenter, self.windowWidth = preset
        self.wlByScan[self.scanNum] = (self.windowCenter, self.windowWidth)
        self.centerSpin.blockSignals(True)
        self.widthSpin.blockSignals(True)
        self.centerSpin.setValue(self.windowCenter)
        self.widthSpin.setValue(self.windowWidth)
        self.centerSpin.blockSignals(False)
        self.widthSpin.blockSignals(False)
        self.refresh_views()

    def on_manual_wl(self, *_):
        self.windowCenter = self.centerSpin.value()
        self.windowWidth = self.widthSpin.value()
        self.wlByScan[self.scanNum] = (self.windowCenter, self.windowWidth)
        self.presetCombo.blockSignals(True)
        self.presetCombo.setCurrentText("--- Manual ---")
        self.presetCombo.blockSignals(False)
        self.refresh_views()

    def on_slice_changed(self, winId, val):
        orientation = self.views[winId].orientation
        self.slices[winId] = val
        self.lastSlice[orientation] = val
        refreshed = [winId]
        if self.lockViews:   # follow with every other window of the same view
            for wid in self.activeWins:
                v = self.views[wid]
                if wid != winId and v.orientation == orientation \
                        and self.slices[wid] != val:
                    self.slices[wid] = val
                    v.set_range(v.slider.maximum() + 1, val)
                    refreshed.append(wid)
        for wid in refreshed:
            self.refresh_views(only=wid)
        # other views' crosshairs depend on this slice -> reposition
        for wid in self.activeWins:
            if wid not in refreshed and not self.views[wid].is3d:
                self._position_crosshair(self.views[wid])
                self.views[wid].canvas.draw_idle()
        # re-render 3D plane views shortly after scrolling settles
        if any(self.views[w].is3d for w in self.activeWins):
            self._timer3d.start(150)

    def on_lock_toggled(self, on):
        self.lockViews = on
        if not on or self.planC is None or not self.planC.scan:
            return
        # align all windows to the last-touched slice of their view type
        for wid, v in self.views.items():
            if v.is3d:
                continue
            k = self.lastSlice[v.orientation]
            if self.slices[wid] != k:
                self.slices[wid] = k
                v.set_range(v.slider.maximum() + 1, k)
        self.refresh_views()

    def on_crosshair_toggled(self, on):
        self.showCrosshairs = on
        for view in self.views.values():
            for ln in (view.xline, view.yline):
                if ln is not None:
                    ln.set_visible(on)
            view.canvas.draw_idle()

    def on_orientation_toggled(self, on):
        self.showOrientation = on
        self.refresh_views()

    def reset_all_views(self):
        for view in self.views.values():
            view.user_limits = None
            view._pan = None
            view._zoom_drag = None
        self.refresh_views()

    # ----------------------------------------------------------- layouts ----
    def set_layout(self, name):
        """Switch the arrangement of view windows (View > Layout)."""
        self._apply_layout(name)
        # Keep the contouring tool working across the layout change instead of
        # blocking it: re-bind to the new layout's axial view.
        if self.contourCtl is not None and self.contourCtl.isVisible():
            self.contourCtl.rebind_after_layout()

    def _apply_layout(self, name):
        prevActive = set(self.activeWins)
        # detach all views so the old splitters can be deleted safely
        for v in self.views.values():
            v.setParent(None)
        while self.viewLay.count():
            item = self.viewLay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        V = self.views
        if name == "single":
            self.activeWins = ["A"]
            self.viewLay.addWidget(V["A"])
        elif name == "two":
            # equal-stretch halves: always the same size
            self.activeWins = ["A", "B"]
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(2)
            for wid in self.activeWins:
                rl.addWidget(V[wid], 1)
            self.viewLay.addWidget(row)
        elif name == "columns":
            # equal-stretch columns: always the same size
            self.activeWins = ["A", "B", "C"]
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(2)
            for wid in self.activeWins:
                rl.addWidget(V[wid], 1)
            self.viewLay.addWidget(row)
        elif name == "grid":
            # equal-stretch 2x2 grid: four same-size quadrants
            self.activeWins = ["A", "B", "C", "D"]
            gw = QtWidgets.QWidget()
            gl = QtWidgets.QGridLayout(gw)
            gl.setContentsMargins(0, 0, 0, 0)
            gl.setSpacing(2)
            gl.addWidget(V["A"], 0, 0)
            gl.addWidget(V["B"], 0, 1)
            gl.addWidget(V["C"], 1, 0)
            gl.addWidget(V["D"], 1, 1)
            for i in (0, 1):
                gl.setRowStretch(i, 1)
                gl.setColumnStretch(i, 1)
            self.viewLay.addWidget(gw)
        else:   # "default": one large + two stacked
            self.activeWins = ["A", "B", "C"]
            right = QtWidgets.QSplitter(Qt.Vertical)
            right.addWidget(V["B"])
            right.addWidget(V["C"])
            main = QtWidgets.QSplitter(Qt.Horizontal)
            main.addWidget(V["A"])
            main.addWidget(right)
            main.setStretchFactor(0, 3)
            main.setStretchFactor(1, 2)
            self.viewLay.addWidget(main)

        for wid, v in self.views.items():
            v.setVisible(wid in self.activeWins)

        if self.planC is not None and self.planC.scan:
            # newly shown windows join at the last slice of their view type
            for wid in set(self.activeWins) - prevActive:
                if self.views[wid].is3d:
                    continue
                self.slices[wid] = min(
                    self.lastSlice[self.views[wid].orientation],
                    self._slice_count(self.views[wid].orientation) - 1)
            self._load_scan_geometry(reset_slices=False)
            self.refresh_views()

    # ------------------------------------------- per-axis menu (CERR-style) -
    def _reset_axis_sel(self):
        for sel in self.axisSel.values():
            sel["scan"] = sel["dose"] = sel["structs"] = None

    def _axis_scan(self, orient):
        """Effective base scan index for a view (manual override or global)."""
        sel = self.axisSel[orient]["scan"]
        if sel is not None and 0 <= sel < len(self.planC.scan):
            return sel
        return self.scanNum

    def _axis_dose(self, orient):
        """Effective dose index for a view (manual, 'none', or global)."""
        sel = self.axisSel[orient]["dose"]
        if sel is None:
            return self.doseNum
        return sel    # may be -1 = no dose

    def _axis_structs(self, orient):
        """Effective visible structures for a view."""
        sel = self.axisSel[orient]["structs"]
        if sel is None:
            return self._checked_structs()
        return [i for i in sorted(sel) if i < len(self.planC.structure)]

    def _show_axis_menu(self, orient):
        """Right-click menu: choose view/scan/dose/structures for this window
        only (mirrors MATLAB CERR's per-axis View/ScanSet/DoseSet/StructSet
        menu)."""
        if self.planC is None or not self.planC.scan:
            return
        sel = self.axisSel[orient]
        menu = QtWidgets.QMenu(self)

        viewM = menu.addMenu("View")
        vgrp = QtWidgets.QActionGroup(viewM)
        for o in (VIEW_AXIAL, VIEW_SAGITTAL, VIEW_CORONAL, VIEW_3D):
            a = viewM.addAction(o)
            a.setCheckable(True)
            a.setChecked(self.views[orient].orientation == o)
            vgrp.addAction(a)
            a.triggered.connect(
                lambda _=False, w=orient, oo=o: self._set_axis_view(w, oo))

        if self.views[orient].is3d:
            # slice-plane selections don't apply to the 3D view
            menu.addSeparator()
            menu.addAction(self.actLock)
            menu.exec_(QtGui.QCursor.pos())
            return

        scanM = menu.addMenu("ScanSet")
        grp = QtWidgets.QActionGroup(scanM)
        act = scanM.addAction("Auto (panel selection)")
        act.setCheckable(True)
        act.setChecked(sel["scan"] is None)
        grp.addAction(act)
        act.triggered.connect(
            lambda _=False, o=orient: self._set_axis_scan(o, None))
        scanM.addSeparator()
        for i, s in enumerate(self.planC.scan):
            mod = getattr(s.scanInfo[0], "imageType", "scan")
            a = scanM.addAction(f"{i}: {mod}")
            a.setCheckable(True)
            a.setChecked(sel["scan"] == i)
            grp.addAction(a)
            a.triggered.connect(
                lambda _=False, o=orient, k=i: self._set_axis_scan(o, k))

        doseM = menu.addMenu("DoseSet")
        dgrp = QtWidgets.QActionGroup(doseM)
        for label, val in (("Auto (panel selection)", None), ("None", -1)):
            a = doseM.addAction(label)
            a.setCheckable(True)
            a.setChecked(sel["dose"] is val if val is None
                         else sel["dose"] == val)
            dgrp.addAction(a)
            a.triggered.connect(
                lambda _=False, o=orient, v=val: self._set_axis_dose(o, v))
        doseM.addSeparator()
        for i, d in enumerate(self.planC.dose):
            a = doseM.addAction(
                f"{i}: {getattr(d, 'fractionGroupID', 'dose')}")
            a.setCheckable(True)
            a.setChecked(sel["dose"] == i)
            dgrp.addAction(a)
            a.triggered.connect(
                lambda _=False, o=orient, k=i: self._set_axis_dose(o, k))

        structM = menu.addMenu("StructSet")
        a = structM.addAction("Auto (panel selection)")
        a.setCheckable(True)
        a.setChecked(sel["structs"] is None)
        a.triggered.connect(
            lambda _=False, o=orient: self._toggle_axis_struct(o, None))
        structM.addSeparator()
        shown = set(self._axis_structs(orient))
        for i, st in enumerate(self.planC.structure):
            a = structM.addAction(f"{i}: {st.structureName}")
            a.setCheckable(True)
            a.setChecked(i in shown)
            a.triggered.connect(
                lambda _=False, o=orient, k=i: self._toggle_axis_struct(o, k))

        menu.addSeparator()
        rulerAct = menu.addAction("Draw Ruler")
        rulerAct.setCheckable(True)
        rulerAct.setChecked(self.views[orient].ruler_mode)
        rulerAct.triggered.connect(
            lambda _=False, o=orient: self._toggle_ruler(o))
        menu.addAction(self.actLock)   # lock slices across matching views
        menu.addAction("Reset pan/zoom (this view)",
                       self.views[orient].reset_view)
        menu.exec_(QtGui.QCursor.pos())

    def _set_axis_view(self, winId, orientation):
        """Switch which orientation a view window displays (CERR SET_VIEW)."""
        view = self.views[winId]
        if view.orientation == orientation:
            return
        if self.contourCtl is not None and self.contourCtl.isVisible() \
                and self.contourCtl.axView is view:
            self.statusBar().showMessage(
                "Close the Contouring panel before changing this view.")
            return
        was3d = view.is3d
        view.orientation = orientation
        if orientation == VIEW_3D:
            view.set_projection(True)
            self.refresh_views(only=winId)
            return
        if was3d:
            view.set_projection(False)
        view.clear_ruler()             # in-plane coords no longer apply
        view.user_limits = None
        n = self._slice_count(orientation)
        k = min(self.lastSlice[orientation], n - 1)
        self.slices[winId] = k
        view.set_range(n, k)
        self.refresh_views(only=winId)

    def _set_axis_scan(self, orient, scanIdx):
        self.axisSel[orient]["scan"] = scanIdx
        self.refresh_views(only=orient)

    def _set_axis_dose(self, orient, doseIdx):
        self.axisSel[orient]["dose"] = doseIdx
        self.refresh_views(only=orient)

    def _toggle_ruler(self, orient):
        """CERR's TOGGLERULER: arm ruler mode / clear the ruler and disarm."""
        view = self.views[orient]
        if view.ruler_mode:
            view.ruler_mode = False
            view.clear_ruler()
            view.canvas.setCursor(Qt.ArrowCursor)
            self.statusBar().showMessage("Ruler off.")
        else:
            view.ruler_mode = True
            view.canvas.setCursor(Qt.CrossCursor)
            self.statusBar().showMessage(
                f"Ruler: click and drag in the {view.orientation} view to "
                "measure; right-click > Draw Ruler again to clear.")

    def on_ruler_changed(self, orient):
        view = self.views[orient]
        if view.ruler is None:
            return
        (x0, y0), (x1, y1) = view.ruler
        self.statusBar().showMessage(
            f"({x0:.2f}, {y0:.2f}) to ({x1:.2f}, {y1:.2f})   "
            f"Dist: {view.ruler_length():.3g} cm")

    def _toggle_axis_struct(self, orient, strNum):
        sel = self.axisSel[orient]
        if strNum is None:                      # back to Auto
            sel["structs"] = None
        else:
            # first manual toggle starts from the currently shown set
            cur = set(sel["structs"]) if sel["structs"] is not None \
                else set(self._checked_structs())
            cur.symmetric_difference_update({strNum})
            sel["structs"] = cur
        self.refresh_views(only=orient)

    def on_cursor_moved(self, winId, xd, yd):
        if self.planC is None:
            return
        orientation = self.views[winId].orientation
        k = self.slices[winId]
        if orientation == VIEW_AXIAL:
            x, y, z = xd, yd, self.zV[k]
        elif orientation == VIEW_SAGITTAL:
            x, y, z = self.xV[k], xd, yd
        else:
            x, y, z = xd, self.yV[k], yd
        msg = f"x={x:.2f} cm  y={y:.2f} cm  z={z:.2f} cm"
        # scan value (of the scan shown in this view)
        try:
            baseIdx = self._axis_scan(winId)
            if baseIdx == self.scanNum:
                r = int(np.argmin(np.abs(self.yV - y)))
                c = int(np.argmin(np.abs(self.xV - x)))
                s = int(np.argmin(np.abs(self.zV - z)))
                msg += f"   scan={self.scan3M[r, c, s]:.1f}"
            else:
                res = self._overlay_interp(baseIdx)
                if res is not None:
                    msg += f"   scan={float(res[0]((y, x, z))):.1f}"
        except Exception:  # noqa: BLE001
            pass
        doseRes = self._dose_interp(self._axis_dose(winId))
        if doseRes is not None:
            try:
                dv = float(doseRes[0]((y, x, z)))
                msg += f"   dose={dv:.2f}"
            except Exception:  # noqa: BLE001
                pass
        self.statusBar().showMessage(msg)

    # ----------------------------------------------------------- rendering --
    def _checked_structs(self):
        out = []
        for i in range(self.structList.count()):
            it = self.structList.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _struct_mask(self, strNum):
        if strNum not in self.maskCache:
            try:
                self.maskCache[strNum] = rs.getStrMask(strNum, self.planC)
            except Exception:  # noqa: BLE001
                self.maskCache[strNum] = None
        return self.maskCache[strNum]

    def _struct_color(self, strNum):
        col = np.asarray(self.planC.structure[strNum].structureColor,
                         dtype=float).ravel()
        if col.size != 3:
            return (1.0, 0.0, 0.0)
        if col.max() > 1:
            col = col / 255.0
        return tuple(np.clip(col, 0, 1))

    @staticmethod
    def _draw_contour_dots(ax, contourSet, color):
        """Plot Alaly-style vertex dots along a structure's contour paths.
        Dots are drawn white on dark contour colors and black on light ones
        (sum of the contour RGB < 1.5 -> white) so they stay visible."""
        dotColor = "white" if sum(color[:3]) < 1.5 else "black"
        for seg in contourSet.allsegs[0]:        # level 0.5
            if len(seg) == 0:
                continue
            # small, fairly fine dots: ~every 3rd vertex, capped at ~150/contour
            step = max(len(seg) // 150, 3)
            pts = seg[::step]
            ax.plot(pts[:, 0], pts[:, 1], linestyle="none", marker="o",
                    markersize=1.0, markerfacecolor=dotColor,
                    markeredgecolor=dotColor, markeredgewidth=0.0, zorder=12)

    # --------------------------------------------------- structure options --
    def on_struct_dots(self, on):
        self.showStructDots = on
        self.refresh_views()

    def on_struct_linewidth(self, val):
        self.structLineWidth = float(val)
        self.refresh_views()

    def on_struct_double_click(self, item):
        """Center all views on the double-clicked structure's centroid."""
        strNum = item.data(Qt.UserRole)
        if strNum is None:
            return
        self.goto_struct_center(strNum)

    def goto_struct_center(self, strNum):
        """Navigate the three orthogonal views to the structure's center of
        mass (its isocenter slice in each plane)."""
        mask = self._struct_mask(strNum)
        if mask is None or mask.shape != self.scan3M.shape or not mask.any():
            self.statusBar().showMessage(
                "Structure has no voxels on the current scan grid.")
            return
        rows, cols, slcs = np.where(mask)       # (y, x, z)
        centers = {VIEW_AXIAL: int(round(slcs.mean())),
                   VIEW_SAGITTAL: int(round(cols.mean())),
                   VIEW_CORONAL: int(round(rows.mean()))}
        for orient, k in centers.items():
            n = self._slice_count(orient)
            self.lastSlice[orient] = int(np.clip(k, 0, n - 1))
        for wid, v in self.views.items():
            if v.is3d:
                continue
            k = self.lastSlice[v.orientation]
            self.slices[wid] = k
            v.set_range(self._slice_count(v.orientation), k)
        self.refresh_views()
        name = self.planC.structure[strNum].structureName
        self.statusBar().showMessage(
            f"Centered views on '{name}' (axial {centers[VIEW_AXIAL] + 1}, "
            f"sagittal {centers[VIEW_SAGITTAL] + 1}, "
            f"coronal {centers[VIEW_CORONAL] + 1}).")

    def _slice_data(self, winId):
        """Return (img2D, extent, aspect, (hAxisVals, vAxisVals), maskSlicer)."""
        orientation = self.views[winId].orientation
        k = self.slices[winId]
        if orientation == VIEW_AXIAL:
            img = self.scan3M[:, :, k]
            hV, vV = self.xV, self.yV
            slicer = lambda m: m[:, :, k]               # noqa: E731
        elif orientation == VIEW_SAGITTAL:
            img = self.scan3M[:, k, :].T                # rows=z, cols=y
            hV, vV = self.yV, self.zV
            slicer = lambda m: m[:, k, :].T             # noqa: E731
        else:  # coronal
            img = self.scan3M[k, :, :].T                # rows=z, cols=x
            hV, vV = self.xV, self.zV
            slicer = lambda m: m[k, :, :].T             # noqa: E731
        extent = [hV[0], hV[-1], vV[-1], vV[0]]
        return img, extent, hV, vV, slicer

    def _grid_points(self, winId, hV, vV):
        """(N,3) physical (y,x,z) points covering this view's slice plane."""
        H, V = np.meshgrid(hV, vV)
        orient = self.views[winId].orientation
        k = self.slices[winId]
        if orient == VIEW_AXIAL:
            pts = (V, H, np.full_like(H, self.zV[k]))
        elif orient == VIEW_SAGITTAL:
            pts = (H, np.full_like(H, self.xV[k]), V)
        else:
            pts = (np.full_like(H, self.yV[k]), H, V)
        return np.stack([p.ravel() for p in pts], axis=-1), H.shape

    def refresh_views(self, only=None):
        if self.planC is None or not self.planC.scan:
            return
        targets = [only] if only else list(self.activeWins)
        vmin = self.windowCenter - self.windowWidth / 2.0
        vmax = self.windowCenter + self.windowWidth / 2.0

        for orient in targets:
            view = self.views[orient]
            if view.is3d:
                self._render_3d(view)
                continue
            ax = view.ax
            ax.clear()
            ax.set_facecolor("black")
            img, extent, hV, vV, slicer = self._slice_data(orient)
            baseIdx = self._axis_scan(orient)
            regComp = None
            if self.regCtl is not None:
                regComp = self.regCtl.compose_slice(orient, img, hV, vV)
            if regComp is not None:    # RGB composite from the QA tool
                ax.imshow(regComp, extent=extent, interpolation="nearest",
                          aspect="equal")
            elif baseIdx == self.scanNum:
                ax.imshow(img, cmap=self.scanCmap, vmin=vmin, vmax=vmax,
                          extent=extent, interpolation="nearest",
                          aspect="equal", alpha=self.scanAlpha)
            else:
                # per-axis scan override: resample onto the reference grid
                res = self._overlay_interp(baseIdx)
                if res is not None:
                    interp, oLo, oHi = res
                    wl = self.wlByScan.get(baseIdx)
                    if wl is not None:
                        bvmin = wl[0] - wl[1] / 2.0
                        bvmax = wl[0] + wl[1] / 2.0
                    else:
                        bvmin, bvmax = oLo, oHi
                    pts, shp = self._grid_points(orient, hV, vV)
                    bSlc = interp(pts).reshape(shp)
                    ax.imshow(np.ma.masked_invalid(bSlc), cmap=self.scanCmap,
                              vmin=bvmin, vmax=bvmax, extent=extent,
                              interpolation="bilinear", aspect="equal",
                              alpha=self.scanAlpha)

            # ---- fused scan overlays ----
            for ovIdx, st in self.overlayState.items():
                if not st["on"] or st["alpha"] <= 0 \
                        or ovIdx == baseIdx \
                        or ovIdx >= len(self.planC.scan):
                    continue
                res = self._overlay_interp(ovIdx)
                if res is None:
                    continue
                interp, oLo, oHi = res
                pts, shp = self._grid_points(orient, hV, vV)
                ovSlc = interp(pts).reshape(shp)
                ovMasked = np.ma.masked_invalid(ovSlc)  # hide outside its FOV
                ax.imshow(ovMasked, cmap=st["cmap"], extent=extent,
                          vmin=oLo, vmax=oHi, alpha=st["alpha"],
                          interpolation="bilinear", aspect="equal")

            # ---- dose colorwash ----
            doseRes = self._dose_interp(self._axis_dose(orient)) \
                if self.doseAlpha > 0 else None
            if doseRes is not None and doseRes[1] > 0:
                doseInterp, _doseMax = doseRes
                pts, shp = self._grid_points(orient, hV, vV)
                doseSlc = doseInterp(pts).reshape(shp)
                cbLo, cbHi = self.colorbar.cbarRange     # colormap mapping
                dLo, dHi = self.colorbar.dispRange       # display (mask) range
                doseMasked = np.ma.masked_where(
                    (doseSlc < max(dLo, 1e-3)) | (doseSlc > dHi), doseSlc)
                ax.imshow(doseMasked, cmap=self.colorbar.mplCmap, extent=extent,
                          vmin=cbLo, vmax=max(cbHi, cbLo + 1e-6),
                          alpha=self.doseAlpha, interpolation="bilinear",
                          aspect="equal")

            # ---- structure contours ----
            ctl = self.contourCtl
            editStrNum = (ctl.structNum if ctl is not None and ctl.isVisible()
                          else None)
            for strNum in self._axis_structs(orient):
                if strNum == editStrNum:
                    continue   # being edited: shown via the live overlay below
                mask = self._struct_mask(strNum)
                if mask is None or mask.shape != self.scan3M.shape:
                    continue
                mslc = slicer(mask)
                if not np.any(mslc):
                    continue
                color = self._struct_color(strNum)
                cs = ax.contour(hV, vV, mslc.astype(float), levels=[0.5],
                                colors=[color],
                                linewidths=self.structLineWidth)
                if self.showStructDots:
                    self._draw_contour_dots(ax, cs, color)

            # ---- live contouring overlay (working mask being edited) ----
            if ctl is not None and ctl.isVisible() and ctl.mask3M is not None \
                    and ctl.mask3M.shape == self.scan3M.shape:
                cslc = slicer(ctl.mask3M)
                im = cs = None
                if np.any(cslc):
                    im = ax.imshow(
                        np.ma.masked_where(~cslc, cslc.astype(float)),
                        cmap=ListedColormap([ctl.color]), extent=extent,
                        alpha=0.35, vmin=0, vmax=1,
                        interpolation="nearest", aspect="equal")
                    cs = ax.contour(hV, vV, cslc.astype(float), levels=[0.5],
                                    colors=[ctl.color], linewidths=1.6,
                                    linestyles="--")
                if view is ctl.axView:
                    # the live updater owns the axial overlay, so each brush
                    # step removes and redraws it (no stale boundary on erase)
                    ctl._liveIm = im
                    ctl._liveContour = cs

            # ---- registration QA split/lens guides ----
            if regComp is not None:
                self.regCtl.draw_guides(view)

            # ---- IMRTP beam overlays (projected onto this view plane) ----
            if self.beams:
                self._draw_beams_2d(view)

            # ---- crosshairs from the other view types ----
            self._draw_crosshair(view)

            # ---- ruler (axes were cleared, recreate its artists) ----
            view.redraw_ruler()

            ax.set_xticks([]), ax.set_yticks([])
            if view.user_limits is not None:    # keep user pan/zoom
                ax.set_xlim(view.user_limits[0])
                ax.set_ylim(view.user_limits[1])

            # ---- patient orientation labels (L/R/A/P/S/I) ----
            if self.showOrientation:
                self._draw_orientation_labels(view)

            view.label.setText(self._view_label_text(view))
            view.canvas.draw_idle()

    def _view_label_text(self, view):
        """View title: slice-plane coordinate (cm), scan #/modality and dose
        # shown in this view, plus any per-axis overrides / lock state."""
        winId, orient = view.winId, view.orientation
        if view.is3d:
            return view.label.text()        # 3D label set by its renderer
        k = self.slices[winId]
        if orient == VIEW_AXIAL:
            axis, val = "z", self.zV[k]
        elif orient == VIEW_SAGITTAL:
            axis, val = "x", self.xV[k]
        else:
            axis, val = "y", self.yV[k]

        baseIdx = self._axis_scan(winId)
        mod = getattr(self.planC.scan[baseIdx].scanInfo[0], "imageType",
                      "scan")
        doseIdx = self._axis_dose(winId)
        doseStr = "no dose" if (doseIdx is None or doseIdx < 0) \
            else f"dose {doseIdx}"

        flags = []
        if self.axisSel[winId]["structs"] is not None:
            flags.append("structs*")
        if self.lockViews:
            flags.append("locked")
        extra = "  [" + ", ".join(flags) + "]" if flags else ""
        return (f"{orient}   {axis}={val:.2f} cm   |   "
                f"scan {baseIdx} ({mod})   |   {doseStr}{extra}")

    @staticmethod
    def _project_points(orient, pts):
        """Project (N,3) world (x,y,z) points to a view's (h, v) coords."""
        pts = np.asarray(pts, dtype=float)
        hAxis, vAxis = AXES_2D[orient]
        col = {"x": 0, "y": 1, "z": 2}
        return pts[:, col[hAxis]], pts[:, col[vAxis]]

    def _beam_slice_polygon(self, view, apex, corners):
        """Cross-section of the beam pyramid (apex + 4 far corners) with this
        view's slice plane, returned as an ordered (M,2) closed polygon in the
        view's (h, v) coords - or None if the slice misses the beam. Changes
        as you scroll, since the slice plane moves."""
        orient = view.orientation
        axisIdx, coords = {VIEW_AXIAL: (2, self.zV),
                           VIEW_SAGITTAL: (0, self.xV),
                           VIEW_CORONAL: (1, self.yV)}[orient]
        sv = float(coords[self.slices[view.winId]])
        edges = [(apex, corners[i]) for i in range(4)]          # lateral
        edges += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]  # base
        eps = 1e-6
        pts = []
        for a, b in edges:
            da, db = a[axisIdx] - sv, b[axisIdx] - sv
            aon, bon = abs(da) < eps, abs(db) < eps
            if aon:
                pts.append(a)
            if bon:
                pts.append(b)
            if not aon and not bon and da * db < 0:    # strict crossing
                t = da / (da - db)
                pts.append(a + t * (b - a))
        if len(pts) < 2:
            return None
        h, v = self._project_points(orient, np.asarray(pts))
        hv = np.column_stack([h, v])
        # dedupe near-identical points
        keep = [hv[0]]
        for p in hv[1:]:
            if min(np.hypot(*(p - q)) for q in keep) > 1e-4:
                keep.append(p)
        hv = np.asarray(keep)
        if len(hv) < 2:
            return None
        # order as a convex polygon (cross-section of a convex solid)
        c = hv.mean(axis=0)
        order = np.argsort(np.arctan2(hv[:, 1] - c[1], hv[:, 0] - c[0]))
        ring = hv[order]
        return np.vstack([ring, ring[0]])      # close it

    def _draw_beams_2d(self, view):
        """Draw beams on this 2D view as a single LineCollection. When a beam
        carries pyramid geometry ("apex"/"corners") its per-slice cross-section
        is drawn (changes with the slice); otherwise the polylines are
        projected. Zoom is preserved (the far source never autoscales)."""
        segs, colors = [], []
        for beam in self.beams:
            color = beam.get("color", (0.2, 0.85, 0.9))
            apex = beam.get("apex")
            corners = beam.get("corners")
            if apex is not None and corners is not None:
                ring = self._beam_slice_polygon(view, np.asarray(apex),
                                                np.asarray(corners))
                if ring is not None:
                    segs.append(ring)
                    colors.append(color)
            else:
                for poly in beam["polylines"]:
                    h, v = self._project_points(view.orientation, poly)
                    segs.append(np.column_stack([h, v]))
                    colors.append(color)
        if not segs:
            return
        xl, yl = view.ax.get_xlim(), view.ax.get_ylim()
        lc = LineCollection(segs, colors=colors, linewidths=1.2, alpha=0.9,
                            zorder=13)
        view.ax.add_collection(lc, autolim=False)
        view.ax.set_xlim(xl)            # keep the scan-extent / user zoom
        view.ax.set_ylim(yl)

    def _draw_orientation_labels(self, view):
        """L/R/A/P/S/I markers at the edge midpoints of a 2D view, derived
        from the displayed coordinate directions (pyCERR virtual coords:
        +x = Left, +y = Anterior, +z = Inferior)."""
        ax = view.ax
        hAxis, vAxis = AXES_2D[view.orientation]
        x0, x1 = ax.get_xlim()
        left, right = ((ORIENT_NEG[hAxis], ORIENT_POS[hAxis]) if x1 >= x0
                       else (ORIENT_POS[hAxis], ORIENT_NEG[hAxis]))
        y0, y1 = ax.get_ylim()
        bottom, top = ((ORIENT_NEG[vAxis], ORIENT_POS[vAxis]) if y1 >= y0
                       else (ORIENT_POS[vAxis], ORIENT_NEG[vAxis]))
        kw = dict(transform=ax.transAxes, color="#e8e8e8", fontsize=9,
                  fontweight="bold", ha="center", va="center", zorder=15,
                  bbox=dict(facecolor="black", alpha=0.45, edgecolor="none",
                            pad=1.5))
        ax.text(0.03, 0.5, left, **kw)
        ax.text(0.97, 0.5, right, **kw)
        ax.text(0.5, 0.96, top, **kw)
        ax.text(0.5, 0.04, bottom, **kw)

    def _refresh_3d_views(self):
        for wid in self.activeWins:
            if self.views[wid].is3d:
                self.refresh_views(only=wid)

    def _render_3d(self, view):
        if view.uses_vtk:
            self._render_3d_vtk(view)
        else:
            self._render_3d_mpl(view)

    def _plane_slices_3d(self):
        """Current plane indices, window range and plane label text."""
        kA = self.lastSlice[VIEW_AXIAL]
        kS = self.lastSlice[VIEW_SAGITTAL]
        kC = self.lastSlice[VIEW_CORONAL]
        label = (f"3D  -  planes: axial {kA + 1}, sagittal {kS + 1}, "
                 f"coronal {kC + 1}")
        return kA, kS, kC, label

    def _scan_grid_geometry(self):
        """Canonical ascending scan-grid axes and the flips applied to get
        them (pyCERR arrays are (row=y, col=x, slice=z))."""
        flipR = self.yV[0] > self.yV[-1]
        flipC = self.xV[0] > self.xV[-1]
        flipS = len(self.zV) > 1 and self.zV[0] > self.zV[-1]
        xA = self.xV[::-1] if flipC else self.xV
        yA = self.yV[::-1] if flipR else self.yV
        zA = self.zV[::-1] if flipS else self.zV
        return xA, yA, zA, flipR, flipC, flipS

    @staticmethod
    def _pv_volume(arr3, xA, yA, zA):
        """pyvista ImageData from a (y, x, z) array on ascending axes."""
        nR, nC, nS = arr3.shape
        grid = pv.ImageData(
            dimensions=(nC, nR, nS),
            spacing=(float(xA[1] - xA[0]) if nC > 1 else 1.0,
                     float(yA[1] - yA[0]) if nR > 1 else 1.0,
                     float(zA[1] - zA[0]) if nS > 1 else 1.0),
            origin=(float(xA[0]), float(yA[0]), float(zA[0])))
        grid.point_data["v"] = np.transpose(
            arr3, (1, 0, 2)).ravel(order="F")     # x fastest, then y, z
        return grid

    def _pv_struct_mesh(self, strNum):
        """Cached smoothed surface mesh of a structure mask, or None."""
        if strNum not in self._pvStructCache:
            surf = None
            try:
                mask = self._struct_mask(strNum)
                if mask is not None and mask.shape == self.scan3M.shape \
                        and np.any(mask):
                    xA, yA, zA, flipR, flipC, flipS = \
                        self._scan_grid_geometry()
                    m = mask.astype(np.float32)
                    if flipR:
                        m = m[::-1, :, :]
                    if flipC:
                        m = m[:, ::-1, :]
                    if flipS:
                        m = m[:, :, ::-1]
                    grid = self._pv_volume(m, xA, yA, zA)
                    surf = grid.contour([0.5])
                    if surf.n_points:
                        surf = surf.smooth(n_iter=30, relaxation_factor=0.1)
                    else:
                        surf = None
            except Exception:  # noqa: BLE001
                surf = None
            self._pvStructCache[strNum] = surf
        return self._pvStructCache[strNum]

    def _pv_dose_iso(self, doseIdx):
        """Cached (isodose surfaces, doseMax) for a dose index, or None.
        Levels at 30/50/70/90% of the dose maximum."""
        if doseIdx not in self._pvDoseCache:
            res = None
            try:
                d = self.planC.dose[doseIdx]
                dose3M = np.asarray(d.doseArray, dtype=np.float32)
                xD, yD, zD = d.getDoseXYZVals()
                yD, dose3M = ascending(yD, dose3M, axis=0)
                xD, dose3M = ascending(xD, dose3M, axis=1)
                zD, dose3M = ascending(zD, dose3M, axis=2)
                dmax = float(dose3M.max())
                if dmax > 0:
                    grid = self._pv_volume(dose3M, xD, yD, zD)
                    levels = [f * dmax for f in (0.3, 0.5, 0.7, 0.9)]
                    iso = grid.contour(levels)
                    if iso.n_points:
                        res = (iso, dmax)
            except Exception:  # noqa: BLE001
                res = None
            self._pvDoseCache[doseIdx] = res
        return self._pvDoseCache[doseIdx]

    def _render_3d_vtk(self, view):
        """GPU (pyvista/VTK) 3D view: full-resolution textured orthogonal
        planes at the slices selected in the 2D views + locator outlines."""
        pl = view.vtk_widget
        kA, kS, kC, label = self._plane_slices_3d()
        vmin = self.windowCenter - self.windowWidth / 2.0
        vmax = self.windowCenter + self.windowWidth / 2.0
        nR, nC, nS = self.scan3M.shape

        # canonical ascending axes; flip slice images to match
        flipR = self.yV[0] > self.yV[-1]
        flipC = self.xV[0] > self.xV[-1]
        flipS = nS > 1 and self.zV[0] > self.zV[-1]
        xA = self.xV[::-1] if flipC else self.xV
        yA = self.yV[::-1] if flipR else self.yV
        zA = self.zV[::-1] if flipS else self.zV
        dx = float(xA[1] - xA[0]) if nC > 1 else 1.0
        dy = float(yA[1] - yA[0]) if nR > 1 else 1.0
        dz = float(zA[1] - zA[0]) if nS > 1 else 1.0

        def canon(img2, flip0, flip1):
            if flip0:
                img2 = img2[::-1, :]
            if flip1:
                img2 = img2[:, ::-1]
            return np.ascontiguousarray(img2)

        axial = pv.ImageData(dimensions=(nC, nR, 1), spacing=(dx, dy, 1.0),
                             origin=(float(xA[0]), float(yA[0]),
                                     float(self.zV[kA])))
        aImg = canon(self.scan3M[:, :, kA], flipR, flipC)   # (y, x)
        axial.point_data["v"] = aImg.ravel()                # x fastest

        sag = pv.ImageData(dimensions=(1, nR, nS), spacing=(1.0, dy, dz),
                           origin=(float(self.xV[kS]), float(yA[0]),
                                   float(zA[0])))
        sImg = canon(self.scan3M[:, kS, :], flipR, flipS)   # (y, z)
        sag.point_data["v"] = sImg.ravel(order="F")         # y fastest

        cor = pv.ImageData(dimensions=(nC, 1, nS), spacing=(dx, 1.0, dz),
                           origin=(float(xA[0]), float(self.yV[kC]),
                                   float(zA[0])))
        cImg = canon(self.scan3M[kC, :, :], flipC, flipS)   # (x, z)
        cor.point_data["v"] = cImg.ravel(order="F")         # x fastest

        pl.clear()      # actors only; the camera is preserved
        view._plane_actors = {}
        for orient, mesh in ((VIEW_AXIAL, axial), (VIEW_SAGITTAL, sag),
                             (VIEW_CORONAL, cor)):
            view._plane_actors[orient] = pl.add_mesh(
                mesh, cmap="gray", clim=(vmin, max(vmax, vmin + 1e-6)),
                show_scalar_bar=False)

        # colored plane outlines (locators)
        x0, x1 = float(xA[0]), float(xA[-1])
        y0, y1 = float(yA[0]), float(yA[-1])
        z0, z1 = float(zA[0]), float(zA[-1])
        zP, xP, yP = float(self.zV[kA]), float(self.xV[kS]), float(self.yV[kC])
        edges = {
            VIEW_AXIAL: [(x0, y0, zP), (x1, y0, zP), (x1, y1, zP),
                         (x0, y1, zP), (x0, y0, zP)],
            VIEW_SAGITTAL: [(xP, y0, z0), (xP, y1, z0), (xP, y1, z1),
                            (xP, y0, z1), (xP, y0, z0)],
            VIEW_CORONAL: [(x0, yP, z0), (x1, yP, z0), (x1, yP, z1),
                           (x0, yP, z1), (x0, yP, z0)],
        }
        view._outline_actors = {}
        for orient, ptsList in edges.items():
            view._outline_actors[orient] = pl.add_lines(
                np.asarray(ptsList, dtype=float),
                color=PLANE_COLORS[orient], width=2, connected=True)

        # ---- structure surfaces (follow the Structures checklist) ----
        for strNum in self._axis_structs(view.winId):
            surf = self._pv_struct_mesh(strNum)
            if surf is not None:
                pl.add_mesh(surf, color=self._struct_color(strNum),
                            opacity=0.45, smooth_shading=True,
                            pickable=False, show_scalar_bar=False,
                            name=f"struct{strNum}")

        # ---- isodose surfaces (follow the Dose combo & alpha slider) ----
        doseIdx = self._axis_dose(view.winId)
        if doseIdx is not None and doseIdx >= 0 and self.doseAlpha > 0:
            res = self._pv_dose_iso(doseIdx)
            if res is not None:
                iso, _dmax = res
                cbLo, cbHi = self.colorbar.cbarRange
                pl.add_mesh(iso, cmap=self.colorbar.mplCmap,
                            clim=(cbLo, max(cbHi, cbLo + 1e-6)),
                            opacity=min(max(self.doseAlpha, 0.0), 0.6),
                            pickable=False, show_scalar_bar=False,
                            name=f"isodose{doseIdx}")

        # ---- IMRTP beam overlays: ONE combined polyline actor (fast) ----
        if self.beams:
            pts_list, conn, cellRGB, off = [], [], [], 0
            for beam in self.beams:
                rgb = (np.asarray(beam.get("color", (0.2, 0.85, 0.9)))
                       * 255).astype(np.uint8)
                for poly in beam["polylines"]:
                    p = np.asarray(poly, dtype=float)
                    n = len(p)
                    if n < 2:
                        continue
                    pts_list.append(p)
                    conn.append(np.concatenate(([n], np.arange(off, off + n))))
                    cellRGB.append(rgb)
                    off += n
            if pts_list:
                pd = pv.PolyData()
                pd.points = np.vstack(pts_list)
                pd.lines = np.concatenate(conn).astype(np.int64)
                pd.cell_data["rgb"] = np.asarray(cellRGB, dtype=np.uint8)
                pl.add_mesh(pd, scalars="rgb", rgb=True, line_width=2,
                            pickable=False, show_scalar_bar=False,
                            name="beams")

        # wire the plane-drag hooks once per widget
        if pl.pick_plane is None:
            pl.pick_plane = lambda pos, v=view: self._vtk_pick_plane(v, pos)
            pl.drag_plane = lambda pos, v=view: self._vtk_drag_plane(v, pos)
            pl.end_plane_drag = lambda v=view: self._vtk_end_plane_drag(v)

        # patient-orientation triad: (re)enable after the window is realized
        # (enabling it at widget-creation time leaves it invisible)
        if self.showOrientation:
            try:
                if getattr(pl.renderer, "axes_widget", None) is None:
                    pl.add_axes(xlabel="L", ylabel="A", zlabel="I")
                    # caption text defaults to black: match each label to
                    # its axis shaft color so it reads on dark backgrounds
                    axes = pl.renderer.axes_actor
                    for cap, shaft in (
                            (axes.GetXAxisCaptionActor2D(),
                             axes.GetXAxisShaftProperty()),
                            (axes.GetYAxisCaptionActor2D(),
                             axes.GetYAxisShaftProperty()),
                            (axes.GetZAxisCaptionActor2D(),
                             axes.GetZAxisShaftProperty())):
                        tp = cap.GetCaptionTextProperty()
                        tp.SetColor(*shaft.GetColor())
                        tp.SetShadow(0)
                        tp.BoldOn()
                pl.renderer.axes_widget.SetEnabled(1)
            except Exception:  # noqa: BLE001
                pass
        elif getattr(pl.renderer, "axes_widget", None) is not None:
            try:
                pl.renderer.axes_widget.SetEnabled(0)
            except Exception:  # noqa: BLE001
                pass

        if not getattr(view, "_vtk_cam_set", False):
            # medical default: superior up (-z in pyCERR virtual coords),
            # viewed from the patient's anterior-superior-right
            cx, cy, cz = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
            diag = float(np.linalg.norm([x1 - x0, y1 - y0, z1 - z0])) or 1.0
            pos = (cx - 0.9 * diag,    # -x = patient right
                   cy + 1.2 * diag,    # +y = anterior
                   cz - 0.9 * diag)    # -z = superior
            pl.camera_position = [pos, (cx, cy, cz), (0.0, 0.0, -1.0)]
            pl.reset_camera()          # refit distance, keep direction/up
            view._vtk_cam_set = True
        pl.render()
        view.label.setText(label)

    # --------------------------------------------- draggable 3D planes ------
    def _vtk_display_pos(self, view, qpos):
        """Qt widget position -> VTK display coordinates (origin bottom-left,
        physical pixels)."""
        pl = view.vtk_widget
        ratio = float(pl.devicePixelRatioF())
        x = qpos.x() * ratio
        y = pl.height() * ratio - qpos.y() * ratio - 1
        return x, y

    def _vtk_pick_plane(self, view, qpos):
        """Left press in the 3D view: True (consume) if a plane was grabbed."""
        if self.planC is None or not self.planC.scan:
            return False
        planeActors = getattr(view, "_plane_actors", {})
        outlineActors = getattr(view, "_outline_actors", {})
        if not planeActors:
            return False
        try:
            import vtk
            picker = vtk.vtkCellPicker()   # geometric ray cast: reliable
            picker.SetTolerance(0.002)
        except Exception:  # noqa: BLE001
            return False
        pl = view.vtk_widget
        x, y = self._vtk_display_pos(view, qpos)
        if not picker.Pick(x, y, 0, pl.renderer):
            return False
        picked = picker.GetActor()
        # plane and outline actor maps share orientation keys: check both
        orient = next((o for o, a in planeActors.items() if a is picked),
                      None)
        if orient is None:
            orient = next((o for o, a in outlineActors.items()
                           if a is picked), None)
        if orient is None:
            return False

        axisIdx = {VIEW_AXIAL: 2, VIEW_SAGITTAL: 0, VIEW_CORONAL: 1}[orient]
        n = np.zeros(3)
        n[axisIdx] = 1.0
        wp = np.asarray(picker.GetPickPosition(), dtype=float)
        ren = pl.renderer

        def w2d(p):
            ren.SetWorldPoint(p[0], p[1], p[2], 1.0)
            ren.WorldToDisplay()
            return np.asarray(ren.GetDisplayPoint()[:2], dtype=float)

        base = w2d(wp)
        dird = w2d(wp + n) - base    # screen direction of the plane normal
        if float(np.dot(dird, dird)) < 1e-9:
            return False             # normal is perpendicular to the screen
        coords = {VIEW_AXIAL: self.zV, VIEW_SAGITTAL: self.xV,
                  VIEW_CORONAL: self.yV}[orient]
        view._drag3d = {
            "orient": orient,
            "coords": np.asarray(coords, dtype=float),
            "coord0": float(coords[self.lastSlice[orient]]),
            "base": base, "dird": dird, "tRender": 0.0,
        }
        return True

    def _vtk_drag_plane(self, view, qpos):
        st = getattr(view, "_drag3d", None)
        if st is None:
            return
        x, y = self._vtk_display_pos(view, qpos)
        d = np.array([x, y]) - st["base"]
        t = float(np.dot(d, st["dird"]) / np.dot(st["dird"], st["dird"]))
        coord = st["coord0"] + t
        k = int(np.argmin(np.abs(st["coords"] - coord)))
        orient = st["orient"]
        if k == self.lastSlice[orient]:
            return
        self._set_orientation_slice(orient, k)
        # live plane motion: re-render the 3D scene, throttled to ~25 fps
        now = time.monotonic()
        if now - st["tRender"] > 0.04:
            st["tRender"] = now
            self._timer3d.stop()
            self._refresh_3d_views()

    def _vtk_end_plane_drag(self, view):
        view._drag3d = None
        self._timer3d.stop()
        self._refresh_3d_views()

    def _set_orientation_slice(self, orient, k):
        """Move the given view type to slice k, driving the first matching
        2D window's slider (so crosshairs, lock, etc. all follow)."""
        for wid in self.activeWins:
            v = self.views[wid]
            if not v.is3d and v.orientation == orient:
                v.slider.setValue(k)     # triggers on_slice_changed
                return
        self.lastSlice[orient] = k       # no 2D window shows it: update state

    def on_crosshair_dragged(self, winId, xdata, ydata, dragV, dragH):
        """Dragging a crosshair line in one view scrolls the perpendicular
        views to the dragged in-plane location."""
        if self.planC is None or not self.planC.scan:
            return
        orient = self.views[winId].orientation
        coordArr = {VIEW_AXIAL: self.zV, VIEW_SAGITTAL: self.xV,
                    VIEW_CORONAL: self.yV}
        if dragV:                        # vertical line -> h-axis target
            tgt = CROSS_TARGET[(orient, "h")]
            arr = np.asarray(coordArr[tgt])
            self._set_orientation_slice(tgt, int(np.argmin(np.abs(arr - xdata))))
        if dragH:                        # horizontal line -> v-axis target
            tgt = CROSS_TARGET[(orient, "v")]
            arr = np.asarray(coordArr[tgt])
            self._set_orientation_slice(tgt, int(np.argmin(np.abs(arr - ydata))))
        # move the dragged view's own crosshair live (its slice is unchanged)
        view = self.views[winId]
        self._position_crosshair(view)
        view.canvas.draw_idle()

    def _render_3d_mpl(self, view):
        """3D view: the three orthogonal planes (at the slices selected in
        the 2D views) textured with the scan, plus colored plane outlines -
        the analog of CERR's 3D plane locators."""
        ax = view.ax
        ax.clear()
        kA = self.lastSlice[VIEW_AXIAL]
        kS = self.lastSlice[VIEW_SAGITTAL]
        kC = self.lastSlice[VIEW_CORONAL]
        vmin = self.windowCenter - self.windowWidth / 2.0
        vmax = self.windowCenter + self.windowWidth / 2.0
        gray = plt.get_cmap("gray")

        def norm(a):
            return np.clip((a - vmin) / max(vmax - vmin, 1e-6), 0, 1)

        nR, nC, nS = self.scan3M.shape
        ir = slice(None, None, max(1, nR // N3D))
        ic = slice(None, None, max(1, nC // N3D))
        is_ = slice(None, None, max(1, nS // N3D))
        xs, ys, zs = self.xV[ic], self.yV[ir], self.zV[is_]
        surf_kw = dict(shade=False, rstride=1, cstride=1, antialiased=False)

        # axial plane (constant z)
        X, Y = np.meshgrid(xs, ys)
        ax.plot_surface(X, Y, np.full_like(X, self.zV[kA]),
                        facecolors=gray(norm(self.scan3M[ir, ic, kA])),
                        **surf_kw)
        # sagittal plane (constant x)
        Yg, Zg = np.meshgrid(ys, zs, indexing="ij")
        ax.plot_surface(np.full_like(Yg, self.xV[kS]), Yg, Zg,
                        facecolors=gray(norm(self.scan3M[ir, kS, is_])),
                        **surf_kw)
        # coronal plane (constant y)
        Xg, Zg2 = np.meshgrid(xs, zs, indexing="ij")
        ax.plot_surface(Xg, np.full_like(Xg, self.yV[kC]), Zg2,
                        facecolors=gray(norm(self.scan3M[kC, ic, is_])),
                        **surf_kw)

        # colored plane outlines (locators)
        x0, x1 = self.xV[0], self.xV[-1]
        y0, y1 = self.yV[0], self.yV[-1]
        z0, z1 = self.zV[0], self.zV[-1]
        zA, xS, yC = self.zV[kA], self.xV[kS], self.yV[kC]
        edges = {
            VIEW_AXIAL: ([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                         [zA] * 5),
            VIEW_SAGITTAL: ([xS] * 5, [y0, y1, y1, y0, y0],
                            [z0, z0, z1, z1, z0]),
            VIEW_CORONAL: ([x0, x1, x1, x0, x0], [yC] * 5,
                           [z0, z0, z1, z1, z0]),
        }
        for orient, (ex, ey, ez) in edges.items():
            ax.plot3D(ex, ey, ez, color=PLANE_COLORS[orient], lw=1.6)

        for beam in self.beams:        # IMRTP beam overlays
            color = beam.get("color", (0.2, 0.85, 0.9))
            for poly in beam["polylines"]:
                p = np.asarray(poly, dtype=float)
                ax.plot3D(p[:, 0], p[:, 1], p[:, 2], color=color, lw=1.0)

        ax.set_box_aspect((abs(x1 - x0) or 1, abs(y1 - y0) or 1,
                           abs(z1 - z0) or 1))
        ax.set_axis_off()
        view.label.setText(
            f"3D  -  planes: axial {kA + 1}, sagittal {kS + 1}, "
            f"coronal {kC + 1}")
        view.canvas.draw_idle()

    def _draw_crosshair(self, view):
        """(Re)create the crosshair artists for a freshly drawn view."""
        kw = dict(color="#e8c542", lw=0.6, ls="--", alpha=0.7,
                  visible=self.showCrosshairs)
        view.xline = view.ax.axvline(0, **kw)
        view.yline = view.ax.axhline(0, **kw)
        self._position_crosshair(view)

    def _position_crosshair(self, view):
        """Move a view's crosshair to the other view types' last slices."""
        if view.xline is None or self.planC is None:
            return
        if view.orientation == VIEW_AXIAL:
            xv = self.xV[self.lastSlice[VIEW_SAGITTAL]]
            yv = self.yV[self.lastSlice[VIEW_CORONAL]]
        elif view.orientation == VIEW_SAGITTAL:
            xv = self.yV[self.lastSlice[VIEW_CORONAL]]
            yv = self.zV[self.lastSlice[VIEW_AXIAL]]
        else:
            xv = self.xV[self.lastSlice[VIEW_SAGITTAL]]
            yv = self.zV[self.lastSlice[VIEW_AXIAL]]
        view.xline.set_xdata([xv, xv])
        view.yline.set_ydata([yv, yv])

    # ----------------------------------------------------------- DVH tool ---
    def show_dvh_dialog(self):
        if self.planC is None or not self.planC.structure or not self.planC.dose:
            _show_info(
                self, "DVH", "A dose and at least one structure are required.")
            return
        # Non-modal (like the other tools): a modal exec_() hangs when the
        # viewer runs inside an integrated event loop (show() / %gui qt).
        dlg = DvhDialog(self.planC, self)
        self._toolWindows.append(dlg)
        dlg.show()

    def _view_with_orientation(self, orientation):
        """First *visible* view window showing the given orientation."""
        for wid in self.activeWins:
            if self.views[wid].orientation == orientation:
                return self.views[wid]
        return None

    def show_contour_dialog(self):
        """Open the CERR-style contouring tools (draw on the axial view)."""
        if self.planC is None or not self.planC.scan:
            _show_info(self, "Contouring",
                                              "Load a scan first.")
            return
        if self._view_with_orientation(VIEW_AXIAL) is None:
            _show_info(
                self, "Contouring",
                "Set one of the views to Axial first (right-click > View).")
            return
        if self.contourCtl is not None and self.contourCtl.isVisible():
            self.contourCtl.raise_()
            self.contourCtl.activateWindow()
            return
        dlg = ContourDialog(self)
        dlg.show()

    def show_reg_dialog(self):
        """Open the registration QA tool (Mirrorscope / Side-by-side /
        Alternate grid / Toggle), cf. the napari QA modes in cerr.viewer."""
        if self.planC is None or len(self.planC.scan) < 2:
            _show_info(
                self, "Registration QA",
                "At least two scans are required to compare.")
            return
        if self.regCtl is not None and self.regCtl.isVisible():
            self.regCtl.raise_()
            self.regCtl.activateWindow()
            return
        dlg = RegQaDialog(self)
        dlg.show()

    def show_imrtp_gui(self):
        """Open the IMRTP beamlet-dose GUI (cerr.imrtp.imrtp_gui) non-blocking."""
        if self.planC is None or not self.planC.scan:
            _show_info(self, "IMRTP",
                                              "Load a scan first.")
            return
        try:
            from cerr.imrtp.imrtp_gui import IMRTPGui
            win = IMRTPGui(self.planC, block=False, viewer=self)
        except Exception as e:  # noqa: BLE001
            _show_error(
                self, "IMRTP", f"Could not open the IMRTP GUI:\n{e}")
            return
        self._toolWindows.append(win)
        self.statusBar().showMessage(
            "IMRTP opened (shares this viewer's planC). After computing dose, "
            "use Tools > Refresh from planC to see it here.")

    def show_roe_gui(self):
        """Open the ROE outcomes-explorer GUI (cerr.roe.roe_gui) non-blocking."""
        try:
            from cerr.roe.roe_gui import launch as roe_launch
            win = roe_launch(self.planC)
        except Exception as e:  # noqa: BLE001
            _show_error(
                self, "ROE", f"Could not open the ROE GUI:\n{e}")
            return
        self._toolWindows.append(win)
        self.statusBar().showMessage(
            "ROE opened (shares this viewer's planC).")

    def show_controls(self):
        _show_info(
            self, "Controls",
            "Mouse:\n"
            "  Scroll: change slice\n"
            "  Left / middle drag: pan\n"
            "  Right-drag up/down: zoom in/out\n"
            "  Drag a crosshair line: scroll the other views\n"
            "  Right-click: choose scan / dose / structures, draw ruler\n"
            "  Double-click: reset pan/zoom\n\n"
            "Keyboard:\n"
            "  Arrow keys (hovered view): change slice\n"
            "  X: toggle crosshairs     O: orientation labels\n"
            "  L: lock slices across views     R: reset pan/zoom\n\n"
            "Tools > DVH for dose-volume histograms.")

    def show_about(self):
        _show_info(
            self, "About pyCERR Viewer",
            "<b>pyCERR Viewer</b><br>"
            f"version {_pycerr_version()}<br><br>"
            "A CERR-style slice viewer built on pyCERR for visualizing "
            "scans, segmentations and radiotherapy dose.<br><br>"
            '<a href="https://github.com/cerr/pyCERR">'
            "https://github.com/cerr/pyCERR</a>",
            rich=True)


# ---------------------------------------------------------------------------#
#  DVH dialog
# ---------------------------------------------------------------------------#
class DvhDialog(QtWidgets.QDialog):
    def __init__(self, planC, parent=None):
        super().__init__(parent)
        self.planC = planC
        self.setWindowTitle("Dose-Volume Histogram")
        self.resize(820, 560)
        lay = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Dose:"))
        self.doseCombo = QtWidgets.QComboBox()
        for i, d in enumerate(planC.dose):
            self.doseCombo.addItem(f"{i}: {getattr(d, 'fractionGroupID', 'dose')}")
        left.addWidget(self.doseCombo)
        left.addWidget(QtWidgets.QLabel("Structures:"))
        self.strList = QtWidgets.QListWidget()
        for i, st in enumerate(planC.structure):
            it = QtWidgets.QListWidgetItem(f"{i}: {st.structureName}")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            it.setData(Qt.UserRole, i)
            self.strList.addItem(it)
        left.addWidget(self.strList, 1)
        btn = QtWidgets.QPushButton("Plot DVH")
        btn.clicked.connect(self.plot)
        left.addWidget(btn)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(240)
        lay.addWidget(lw)

        self.fig = Figure(layout="tight")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        lay.addWidget(self.canvas, 1)

    def plot(self):
        doseNum = self.doseCombo.currentIndex()
        self.ax.clear()
        plotted = False
        for i in range(self.strList.count()):
            it = self.strList.item(i)
            if it.checkState() != Qt.Checked:
                continue
            strNum = it.data(Qt.UserRole)
            try:
                dosesV, volsV, isErr = cerrDvh.getDVH(strNum, doseNum, self.planC)
                if isErr or dosesV is None or len(dosesV) == 0:
                    continue
                binWidth = max(float(np.max(dosesV)) / 400.0, 1e-3)
                doseBinsV, volsHistV = cerrDvh.doseHist(dosesV, volsV, binWidth)
                cumVols = np.flip(np.cumsum(np.flip(volsHistV)))
                cumPct = 100.0 * cumVols / cumVols[0]
                st = self.planC.structure[strNum]
                col = np.asarray(st.structureColor, dtype=float).ravel()
                if col.size == 3 and col.max() > 1:
                    col = col / 255.0
                col = tuple(np.clip(col, 0, 1)) if col.size == 3 else None
                self.ax.plot(doseBinsV, cumPct, label=st.structureName,
                             color=col, lw=1.8)
                plotted = True
            except Exception as e:  # noqa: BLE001
                print(f"DVH failed for structure {strNum}: {e}")
        if plotted:
            self.ax.set_xlabel("Dose (Gy)")
            self.ax.set_ylabel("Volume (%)")
            self.ax.set_ylim(0, 105)
            self.ax.grid(alpha=0.3)
            self.ax.legend(fontsize=8)
        else:
            self.ax.text(0.5, 0.5, "No DVH could be computed.",
                         ha="center", va="center")
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------#
#  Contouring tools (CERR-style: draw on the transverse/axial view)
#  Edits a working 3D mask; saving converts it back to structure contours via
#  cerr.dataclasses.structure.importStructureMask (new or replaced structure).
# ---------------------------------------------------------------------------#
class ContourDialog(QtWidgets.QDialog):
    NEW_ITEM = "<New structure>"

    def __init__(self, viewer):
        super().__init__(viewer)
        self.setWindowTitle("Contouring")
        self.setModal(False)
        self.viewer = viewer
        self.scanNum = viewer.scanNum    # session is locked to this scan
        self.mask3M = None               # working bool mask (scan shape)
        self.structNum = None            # None = creating a new structure
        self.color = (0.91, 0.77, 0.26)
        self._undo = []                  # [(sliceIdx, previous 2D mask), ...]
        self._dirty = False
        self._liveIm = None              # axial overlay artist while brushing
        self._liveContour = None         # live dashed boundary while brushing

        lay = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            f"Editing on scan {self.scanNum}, in the AXIAL view.\n"
            "Freehand/Brush: left-drag.  Polygon: left-click points,\n"
            "right- or double-click to close.  Scroll changes slice.")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        form = QtWidgets.QFormLayout()
        self.structCombo = QtWidgets.QComboBox()
        form.addRow("Structure:", self.structCombo)
        self.nameEdit = QtWidgets.QLineEdit()
        form.addRow("Name:", self.nameEdit)
        lay.addLayout(form)

        toolRow = QtWidgets.QHBoxLayout()
        self.drawBtn = QtWidgets.QRadioButton("Draw (add)")
        self.eraseBtn = QtWidgets.QRadioButton("Erase")
        self.drawBtn.setChecked(True)
        self._actionGrp = QtWidgets.QButtonGroup(self)
        self._actionGrp.addButton(self.drawBtn)
        self._actionGrp.addButton(self.eraseBtn)
        toolRow.addWidget(self.drawBtn)
        toolRow.addWidget(self.eraseBtn)
        lay.addLayout(toolRow)

        # drawing modes (cf. CERR drawContour: drawMode / drawing / drawBall)
        modeBox = QtWidgets.QGroupBox("Drawing mode")
        modeLay = QtWidgets.QVBoxLayout(modeBox)
        self.freehandBtn = QtWidgets.QRadioButton("Freehand (drag)")
        self.polyBtn = QtWidgets.QRadioButton(
            "Polygon - click points / line segments\n"
            "(right- or double-click to close)")
        self.brushBtn = QtWidgets.QRadioButton("Brush (drag paints disks)")
        self.freehandBtn.setChecked(True)
        self._modeGrp = QtWidgets.QButtonGroup(self)
        for b in (self.freehandBtn, self.polyBtn, self.brushBtn):
            self._modeGrp.addButton(b)
            modeLay.addWidget(b)
            b.toggled.connect(self._on_mode_changed)
        brushRow = QtWidgets.QHBoxLayout()
        brushRow.addWidget(QtWidgets.QLabel("Brush radius (cm):"))
        self.brushSpin = QtWidgets.QDoubleSpinBox()
        self.brushSpin.setRange(0.05, 5.0)
        self.brushSpin.setSingleStep(0.1)
        self.brushSpin.setValue(0.5)
        self.brushSpin.valueChanged.connect(self._on_mode_changed)
        brushRow.addWidget(self.brushSpin)
        brushRow.addStretch(1)
        modeLay.addLayout(brushRow)
        lay.addWidget(modeBox)

        grid = QtWidgets.QGridLayout()
        bDel = QtWidgets.QPushButton("Delete slice contour")
        bUndo = QtWidgets.QPushButton("Undo")
        bSup = QtWidgets.QPushButton("Copy from superior slice")
        bInf = QtWidgets.QPushButton("Copy from inferior slice")
        bDel.clicked.connect(self.delete_slice)
        bUndo.clicked.connect(self.undo)
        bSup.clicked.connect(lambda: self.copy_adjacent(+1))
        bInf.clicked.connect(lambda: self.copy_adjacent(-1))
        grid.addWidget(bDel, 0, 0)
        grid.addWidget(bUndo, 0, 1)
        grid.addWidget(bSup, 1, 0)
        grid.addWidget(bInf, 1, 1)
        lay.addLayout(grid)

        bb = QtWidgets.QDialogButtonBox()
        saveBtn = bb.addButton("Save to planC",
                               QtWidgets.QDialogButtonBox.ApplyRole)
        bb.addButton(QtWidgets.QDialogButtonBox.Close)
        saveBtn.clicked.connect(self.save)
        bb.rejected.connect(self.close)
        bb.button(QtWidgets.QDialogButtonBox.Close).clicked.connect(self.close)
        lay.addWidget(bb)

        self._populate_structs()
        self.structCombo.currentIndexChanged.connect(self._on_struct_selected)
        self._on_struct_selected(self.structCombo.currentIndex())

        # bind to the window currently showing the axial view
        self.axView = viewer._view_with_orientation(VIEW_AXIAL)
        self._attach(self.axView)
        viewer.contourCtl = self
        self._on_mode_changed()

    def _attach(self, axView):
        """Start contouring on an axial view (cursor + draw signals)."""
        if axView is None:
            return
        axView.draw_mode = True
        axView.canvas.setCursor(Qt.CrossCursor)
        axView.strokeFinished.connect(self._on_stroke)
        axView.brushStroke.connect(self._on_brush_stroke)
        axView.brushDone.connect(self._on_brush_done)
        # changing slice mid-polygon would mix slices: drop the open polygon
        axView.sliceChanged.connect(self._on_axial_slice_changed)

    def _detach(self, axView):
        """Stop contouring on a view (undo what _attach did)."""
        if axView is None:
            return
        axView.draw_mode = False
        axView.clear_draw_artists()
        axView.canvas.setCursor(Qt.ArrowCursor)
        for sig, slot in ((axView.strokeFinished, self._on_stroke),
                          (axView.brushStroke, self._on_brush_stroke),
                          (axView.brushDone, self._on_brush_done),
                          (axView.sliceChanged, self._on_axial_slice_changed)):
            try:
                sig.disconnect(slot)
            except Exception:  # noqa: BLE001
                pass

    def rebind_after_layout(self):
        """Re-attach to the current layout's axial view after a layout change,
        so contouring keeps working instead of blocking the layout change."""
        newAx = self.viewer._view_with_orientation(VIEW_AXIAL)
        if newAx is self.axView:
            if newAx is not None:        # same view, just re-parented
                newAx.draw_mode = True
                newAx.canvas.setCursor(Qt.CrossCursor)
                self._on_mode_changed()
            return
        self._detach(self.axView)
        self.axView = newAx
        if newAx is None:
            self.viewer.statusBar().showMessage(
                "No axial view in this layout - set a view to Axial to "
                "continue contouring.")
            return
        self._attach(newAx)
        self._on_mode_changed()

    def _on_axial_slice_changed(self, *_):
        self.axView.cancel_polygon()

    def _on_mode_changed(self, *_):
        axView = self.axView
        axView.cancel_polygon()
        if not self.brushBtn.isChecked():
            axView.clear_draw_artists()
        axView.brush_radius = self.brushSpin.value()
        axView.draw_tool = ("polygon" if self.polyBtn.isChecked()
                            else "brush" if self.brushBtn.isChecked()
                            else "freehand")
        axView.canvas.setCursor(
            _contour_cursor("brush" if axView.draw_tool == "brush" else "pen"))

    # ----------------------------------------------------- structure setup --
    def _populate_structs(self, current=None):
        self.structCombo.blockSignals(True)
        self.structCombo.clear()
        self.structCombo.addItem(self.NEW_ITEM)
        for i, st in enumerate(self.viewer.planC.structure):
            self.structCombo.addItem(f"{i}: {st.structureName}")
        self.structCombo.setCurrentIndex(0 if current is None else current + 1)
        self.structCombo.blockSignals(False)

    def _on_struct_selected(self, idx):
        if self._dirty:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setWindowTitle("Contouring")
            box.setText("Discard unsaved edits?")
            box.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            box.setModal(False)
            box.setAttribute(Qt.WA_DeleteOnClose, True)

            yesBtn = box.button(QtWidgets.QMessageBox.Yes)

            def _on_finished(_):
                if box.clickedButton() is yesBtn:
                    self._apply_struct_selection(idx)
                else:
                    self._populate_structs(self.structNum)   # revert combo
            box.finished.connect(_on_finished)
            box.show()
            box.raise_()
            return
        self._apply_struct_selection(idx)

    def _apply_struct_selection(self, idx):
        planC = self.viewer.planC
        shape = self.viewer.scan3M.shape
        self._undo = []
        self._dirty = False
        if idx <= 0:                              # new structure
            self.structNum = None
            self.mask3M = np.zeros(shape, dtype=bool)
            self.nameEdit.setText(f"ROI_{len(planC.structure) + 1}")
            try:
                from cerr.dataclasses import structure as cerrStruct
                col = np.asarray(cerrStruct.getColorForStructNum(
                    len(planC.structure)), dtype=float).ravel()
                self.color = tuple(np.clip(
                    col / 255.0 if col.max() > 1 else col, 0, 1))
            except Exception:  # noqa: BLE001
                self.color = (0.91, 0.77, 0.26)
        else:
            strNum = idx - 1
            mask = self.viewer._struct_mask(strNum)
            if mask is None or mask.shape != shape:
                _show_warning(
                    self, "Contouring",
                    "This structure is not defined on the current scan grid; "
                    "starting from an empty mask instead.")
                mask = np.zeros(shape, dtype=bool)
            self.structNum = strNum
            self.mask3M = mask.astype(bool).copy()
            self.nameEdit.setText(
                planC.structure[strNum].structureName)
            self.color = self.viewer._struct_color(strNum)
        self.viewer.refresh_views()

    # ------------------------------------------------------------ editing ---
    def _push_undo(self, k):
        self._undo.append((k, self.mask3M[:, :, k].copy()))
        del self._undo[:-50]                       # cap the undo history

    def _brush_region(self, pts):
        """Pixels within brush_radius of the stroke polyline (disk brush)."""
        v = self.viewer
        xs, ys = v.xV, v.yV
        stepx = xs[1] - xs[0]
        stepy = ys[1] - ys[0]
        r = self.brushSpin.value()
        seed = np.zeros((len(ys), len(xs)), dtype=bool)
        pts = np.asarray(pts, dtype=float)
        if len(pts) > 1:    # sample segments densely so the trail is solid
            step = 0.5 * min(abs(stepx), abs(stepy))
            segs = []
            for a, b in zip(pts[:-1], pts[1:]):
                n = max(int(np.hypot(*(b - a)) / step), 1)
                segs.append(np.linspace(a, b, n + 1))
            pts = np.vstack(segs)
        cols = np.clip(np.round((pts[:, 0] - xs[0]) / stepx).astype(int),
                       0, len(xs) - 1)
        rows = np.clip(np.round((pts[:, 1] - ys[0]) / stepy).astype(int),
                       0, len(ys) - 1)
        seed[rows, cols] = True
        rx = max(int(np.ceil(r / abs(stepx))), 1)
        ry = max(int(np.ceil(r / abs(stepy))), 1)
        oy, ox = np.ogrid[-ry:ry + 1, -rx:rx + 1]
        disk = (ox * stepx) ** 2 + (oy * stepy) ** 2 <= r ** 2
        return binary_dilation(seed, structure=disk)

    def _cur_slice(self):
        return self.viewer.slices[self.axView.winId]

    def _on_brush_stroke(self, winId, pts, isStart):
        """Paint one brush step (press or drag segment) in real time."""
        if winId != self.axView.winId or self.mask3M is None:
            return
        k = self._cur_slice()
        if isStart:
            self._push_undo(k)      # one undo entry per drag
        region = self._brush_region(pts)
        if self.eraseBtn.isChecked():
            self.mask3M[:, :, k] &= ~region
        else:
            self.mask3M[:, :, k] |= region
        self._dirty = True
        self._live_update_axial()

    def _on_brush_done(self, winId):
        """Brush drag ended: full refresh so all views show the result."""
        if winId != self.axView.winId:
            return
        self._liveIm = None         # refresh_views clears the axes anyway
        self._liveContour = None
        self.viewer.refresh_views()

    def _live_update_axial(self):
        """Cheap live overlay update on the axial view only (no dose/contour
        recomputation), so brushing stays responsive."""
        v = self.viewer
        view = self.axView
        if self._liveIm is not None:
            try:
                self._liveIm.remove()
            except Exception:  # noqa: BLE001
                pass
            self._liveIm = None
        self._remove_live_contour()
        cslc = self.mask3M[:, :, self._cur_slice()]
        if np.any(cslc):
            # imshow() resets the axes limits to the image extent; preserve the
            # current pan/zoom so brushing doesn't shift the view.
            xlim, ylim = view.ax.get_xlim(), view.ax.get_ylim()
            extent = [v.xV[0], v.xV[-1], v.yV[-1], v.yV[0]]
            self._liveIm = view.ax.imshow(
                np.ma.masked_where(~cslc, cslc.astype(float)),
                cmap=ListedColormap([self.color]), extent=extent,
                alpha=0.35, vmin=0, vmax=1, interpolation="nearest",
                aspect="equal", zorder=9)
            # live dashed boundary, matching the committed-contour style
            self._liveContour = view.ax.contour(
                v.xV, v.yV, cslc.astype(float), levels=[0.5],
                colors=[self.color], linewidths=1.6, linestyles="--")
            view.ax.set_xlim(xlim)
            view.ax.set_ylim(ylim)
        view.canvas.draw_idle()

    def _remove_live_contour(self):
        if self._liveContour is not None:
            try:
                self._liveContour.remove()
            except Exception:  # noqa: BLE001
                try:                       # matplotlib < 3.8
                    for coll in self._liveContour.collections:
                        coll.remove()
                except Exception:  # noqa: BLE001
                    pass
            self._liveContour = None

    def _on_stroke(self, winId, pts):
        if winId != self.axView.winId or self.mask3M is None:
            return
        v = self.viewer
        if self.brushBtn.isChecked():
            inside = self._brush_region(pts)
        else:
            if len(pts) < 3:
                return
            poly = MplPath(list(pts) + [pts[0]])
            X, Y = np.meshgrid(v.xV, v.yV)
            inside = poly.contains_points(
                np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
        if not np.any(inside):
            return
        k = self._cur_slice()
        self._push_undo(k)
        if self.eraseBtn.isChecked():
            self.mask3M[:, :, k] &= ~inside
        else:
            self.mask3M[:, :, k] |= inside
        self._dirty = True
        v.refresh_views()

    def delete_slice(self):
        if self.mask3M is None:
            return
        k = self._cur_slice()
        self._push_undo(k)
        self.mask3M[:, :, k] = False
        self._dirty = True
        self.viewer.refresh_views()

    def copy_adjacent(self, direction):
        """Copy the superior(+1)/inferior(-1) neighbor slice's mask onto the
        current slice (CERR's copy sup/inf). Superior/inferior follow the scan's
        z direction (+z = inferior in pyCERR coords), not a fixed slice order."""
        if self.mask3M is None:
            return
        k = self._cur_slice()
        nSlc = self.mask3M.shape[2]
        zV = self.viewer.zV
        # Index step toward the more-inferior (larger z) neighbor; superior is
        # the opposite step.
        infStep = 1 if (len(zV) < 2 or zV[-1] >= zV[0]) else -1
        step = -infStep if direction > 0 else infStep    # +direction = superior
        src = k + step
        if not 0 <= src < nSlc:
            return
        self._push_undo(k)
        self.mask3M[:, :, k] = self.mask3M[:, :, src]
        self._dirty = True
        self.viewer.refresh_views()

    def undo(self):
        if not self._undo:
            return
        k, prev = self._undo.pop()
        self.mask3M[:, :, k] = prev
        self.viewer.refresh_views()

    # --------------------------------------------------------------- save ---
    def save(self):
        if self.mask3M is None or not np.any(self.mask3M):
            _show_info(
                self, "Contouring", "The mask is empty - draw something first.")
            return
        name = self.nameEdit.text().strip() or "ROI"
        v = self.viewer
        try:
            from cerr.dataclasses import structure as cerrStruct
            v._busy("Saving structure ...")
            v.planC = cerrStruct.importStructureMask(
                self.mask3M, self.scanNum, name, v.planC,
                structNum=self.structNum)
            if self.structNum is None:
                self.structNum = len(v.planC.structure) - 1
            v.maskCache.clear()
            v.after_load(keep_view=True)
            self._populate_structs(self.structNum)
            self._dirty = False
            v._done(f"Saved structure '{name}'.")
        except Exception as e:  # noqa: BLE001
            v._done()
            _show_error(self, "Contouring",
                                           f"Could not save structure:\n{e}")

    # -------------------------------------------------------------- close ---
    def closeEvent(self, event):
        if self._dirty and not getattr(self, "_force_close", False):
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setWindowTitle("Contouring")
            box.setText("Discard unsaved edits?")
            box.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            box.setModal(False)
            box.setAttribute(Qt.WA_DeleteOnClose, True)

            yesBtn = box.button(QtWidgets.QMessageBox.Yes)

            def _on_finished(_):
                # finished fires after the box has dismissed, so closing the
                # panel here is safe (no re-entrancy with the box's own close).
                if box.clickedButton() is yesBtn:
                    self._force_close = True
                    self.close()
            box.finished.connect(_on_finished)
            box.show()
            box.raise_()
            event.ignore()           # wait for the (non-modal) answer
            return
        self._detach(self.axView)
        self.viewer.contourCtl = None
        self.viewer.refresh_views()
        event.accept()


# ---------------------------------------------------------------------------#
#  Registration QA tool (cf. the napari QA modes in cerr.viewer:
#  Mirrorscope / Sidebyside / AlternateGrid), plus Toggle.
#  Composites the moving scan (resampled onto the base grid) with the base
#  scan in every 2D view; the split line is draggable with the left button.
# ---------------------------------------------------------------------------#
class RegQaDialog(QtWidgets.QDialog):
    def __init__(self, viewer):
        super().__init__(viewer)
        self.setWindowTitle("Registration QA")
        self.setModal(False)
        self.viewer = viewer
        self.split = {}              # orientation -> split coord (data units)
        self.lens = {}               # orientation -> (cx, cy) mirror box center
        self.baseFrac = 1.0          # Toggle mode base weight (moving = 1-x)

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.baseCombo = QtWidgets.QComboBox()
        self.movCombo = QtWidgets.QComboBox()
        for i, s in enumerate(viewer.planC.scan):
            name = f"{i}: {getattr(s.scanInfo[0], 'imageType', 'scan')}"
            self.baseCombo.addItem(name)
            self.movCombo.addItem(name)
        self.baseCombo.setCurrentIndex(viewer.scanNum)
        self.movCombo.setCurrentIndex(0 if viewer.scanNum != 0 else 1)
        self.baseCombo.currentIndexChanged.connect(self._on_base_changed)
        self.movCombo.currentIndexChanged.connect(
            lambda *_: self.viewer.refresh_views())
        form.addRow("Base scan:", self.baseCombo)
        form.addRow("Moving scan:", self.movCombo)
        lay.addLayout(form)

        modeBox = QtWidgets.QGroupBox("Display mode")
        ml = QtWidgets.QVBoxLayout(modeBox)
        self.mirrorBtn = QtWidgets.QRadioButton(
            "Mirrorscope (moving mirrored about the line; drag it)")
        self.sbsBtn = QtWidgets.QRadioButton(
            "Side-by-side (base | moving; drag the line)")
        self.gridBtn = QtWidgets.QRadioButton(
            "Alternate grid (checkerboard)")
        self.toggleBtn = QtWidgets.QRadioButton("Toggle / blend base-moving")
        self.mirrorBtn.setChecked(True)
        for b in (self.mirrorBtn, self.sbsBtn, self.gridBtn, self.toggleBtn):
            ml.addWidget(b)
            b.toggled.connect(self._on_mode_changed)
        sizeRow = QtWidgets.QHBoxLayout()
        sizeRow.addWidget(QtWidgets.QLabel("Mirror half-width / tile (cm):"))
        self.sizeSpin = QtWidgets.QDoubleSpinBox()
        self.sizeSpin.setRange(0.2, 10.0)
        self.sizeSpin.setSingleStep(0.5)
        self.sizeSpin.setValue(2.0)
        self.sizeSpin.valueChanged.connect(
            lambda *_: self.viewer.refresh_views())
        sizeRow.addWidget(self.sizeSpin)
        sizeRow.addStretch(1)
        ml.addLayout(sizeRow)

        # Toggle-mode cross-fade: base opacity (moving gets 1 - base)
        self.fadeRow = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(self.fadeRow)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(QtWidgets.QLabel("Base"))
        self.fadeSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.fadeSlider.setRange(0, 100)
        # slider runs Base (left, 0) -> Moving (right, 100)
        self.fadeSlider.setValue(int((1.0 - self.baseFrac) * 100))
        self.fadeSlider.setToolTip(
            "Blend base (left) <-> moving (right); T flips base/moving")
        self.fadeSlider.valueChanged.connect(self._on_fade)
        fl.addWidget(self.fadeSlider, 1)
        fl.addWidget(QtWidgets.QLabel("Moving"))
        ml.addWidget(self.fadeRow)
        lay.addWidget(modeBox)

        hint = QtWidgets.QLabel(
            "Left-drag in any 2D view moves the split line\n"
            "(Mirrorscope and Side-by-side modes).")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        bb.rejected.connect(self.close)
        bb.button(QtWidgets.QDialogButtonBox.Close).clicked.connect(self.close)
        lay.addWidget(bb)

        sc = QtWidgets.QShortcut(QtGui.QKeySequence("T"), self)
        sc.setContext(Qt.ApplicationShortcut)
        sc.activated.connect(self._flip_fade)
        self._on_mode_changed()

        for v in viewer.views.values():
            v.qa_split_cb = self._on_split_drag
        viewer.regCtl = self
        viewer.refresh_views()

    # ------------------------------------------------------------ state ----
    MODES = ("Mirrorscope", "Sidebyside", "AlternateGrid", "Toggle")

    def mode(self):
        if self.sbsBtn.isChecked():
            return "Sidebyside"
        if self.gridBtn.isChecked():
            return "AlternateGrid"
        if self.toggleBtn.isChecked():
            return "Toggle"
        return "Mirrorscope"

    def configure(self, base=None, moving=None, mode=None, size=None,
                  base_frac=None):
        """Programmatically set base/moving scans, display mode, mirror-box /
        tile size (cm) and toggle blend fraction (base weight, 0..1)."""
        if base is not None:
            self.baseCombo.setCurrentIndex(int(base))   # -> _on_base_changed
        if moving is not None:
            self.movCombo.setCurrentIndex(int(moving))
        if mode is not None:
            btn = {"Mirrorscope": self.mirrorBtn, "Sidebyside": self.sbsBtn,
                   "AlternateGrid": self.gridBtn,
                   "Toggle": self.toggleBtn}.get(mode)
            if btn is None:
                raise ValueError(f"mode must be one of {self.MODES}")
            btn.setChecked(True)
        if size is not None:
            self.sizeSpin.setValue(float(size))
        if base_frac is not None:
            self.fadeSlider.setValue(
                int(round((1.0 - float(base_frac)) * 100)))
        self.viewer.refresh_views()

    def _on_base_changed(self, idx):
        if idx != self.viewer.scanNum:   # base defines the reference grid
            self.viewer.scanCombo.setCurrentIndex(idx)
        else:
            self.viewer.refresh_views()

    def sync_base(self, idx):
        """Reflect the viewer's active scan as the QA base (no feedback loop)."""
        self.baseCombo.blockSignals(True)
        self.baseCombo.setCurrentIndex(idx)
        self.baseCombo.blockSignals(False)

    def _on_mode_changed(self, *_):
        self.fadeRow.setVisible(self.toggleBtn.isChecked())
        self.viewer.refresh_views()

    def _on_fade(self, val):
        self.baseFrac = 1.0 - val / 100.0     # left=base, right=moving
        self.viewer.refresh_views()

    def _flip_fade(self):
        self.fadeSlider.setValue(100 if self.fadeSlider.value() <= 50 else 0)

    def _on_split_drag(self, winId, xdata, ydata, _isDrag):
        mode = self.mode()
        if mode not in ("Mirrorscope", "Sidebyside"):
            return False
        orient = self.viewer.views[winId].orientation
        if mode == "Mirrorscope":            # 2D box center
            self.lens[orient] = (float(xdata), float(ydata))
        else:                                # vertical split line
            self.split[orient] = float(xdata)
        self.viewer.refresh_views(only=winId)
        return True

    # ------------------------------------------------------- compositing ----
    def compose_slice(self, winId, img, hV, vV):
        """Composite RGB image of base+moving for one view slice, each scan
        rendered with its own colormap. None falls back to normal display."""
        v = self.viewer
        movIdx = self.movCombo.currentIndex()
        res = v._overlay_interp(movIdx)
        if res is None:
            return None
        interp, mLo, mHi = res
        wl = v.wlByScan.get(movIdx)
        if wl is not None:
            mLo, mHi = wl[0] - wl[1] / 2.0, wl[0] + wl[1] / 2.0
        pts, shp = v._grid_points(winId, hV, vV)
        mSlc = np.nan_to_num(interp(pts).reshape(shp), nan=mLo)
        gM = np.clip((mSlc - mLo) / max(mHi - mLo, 1e-6), 0, 1)
        vmin = v.windowCenter - v.windowWidth / 2.0
        vmax = v.windowCenter + v.windowWidth / 2.0
        gB = np.clip((img - vmin) / max(vmax - vmin, 1e-6), 0, 1)

        # render each scan through its own (per-scan) colormap to RGB
        movCmap = v.dispByScan.get(movIdx, ("gray", 1.0))[0]
        rgbB = plt.get_cmap(v.scanCmap)(gB)[..., :3]
        rgbM = plt.get_cmap(movCmap)(gM)[..., :3]

        mode = self.mode()
        if mode == "Toggle":     # cross-fade: base weight, moving = 1 - base
            f = self.baseFrac
            return np.clip(f * rgbB + (1.0 - f) * rgbM, 0, 1)
        if mode == "AlternateGrid":
            t = self.sizeSpin.value()
            tr = max(int(round(t / abs(vV[1] - vV[0]))), 1)
            tc = max(int(round(t / abs(hV[1] - hV[0]))), 1)
            rIdx = (np.arange(gB.shape[0]) // tr)[:, None]
            cIdx = (np.arange(gB.shape[1]) // tc)[None, :]
            comp = rgbB.copy()
            tileMask = ((rIdx + cIdx) % 2).astype(bool)
            comp[tileMask] = rgbM[tileMask]
            return comp

        orient = v.views[winId].orientation
        dh = hV[1] - hV[0]
        comp = rgbB.copy()
        if mode == "Sidebyside":
            hMin, hMax = min(hV[0], hV[-1]), max(hV[0], hV[-1])
            s = self.split.get(orient)
            if s is None or not hMin <= s <= hMax:
                s = 0.5 * (hV[0] + hV[-1])
                self.split[orient] = s
            cs = (s - hV[0]) / dh
            idx = np.arange(gB.shape[1])
            comp[:, idx > cs] = rgbM[:, idx > cs]
            return comp

        # Mirrorscope: a square box centered at the (draggable) lens point.
        # Left of the center line shows the base scan; right of it shows the
        # moving scan flipped about that line - so a well-registered pair
        # forms a symmetric (mirror) image across the line.
        dv = vV[1] - vV[0]
        c = self.lens.get(orient)
        if c is None:
            c = (0.5 * (hV[0] + hV[-1]), 0.5 * (vV[0] + vV[-1]))
            self.lens[orient] = c
        cx, cy = c
        cc = (cx - hV[0]) / dh           # box-center column (mirror line)
        rc = (cy - vV[0]) / dv           # box-center row
        half = self.sizeSpin.value()
        cw = abs(half / dh)
        rw = abs(half / dv)
        rows = np.arange(gB.shape[0])[:, None]
        cols = np.arange(gB.shape[1])[None, :]
        inBox = (np.abs(cols - cc) <= cw) & (np.abs(rows - rc) <= rw)
        rightHalf = inBox & (cols >= cc)               # right of mirror line
        srcCol = np.round(2 * cc - cols).astype(int)   # reflect about cc
        srcFull = np.broadcast_to(srcCol, gB.shape)
        valid = rightHalf & (srcFull >= 0) & (srcFull < gB.shape[1])
        rr, cci = np.where(valid)
        comp[rr, cci] = rgbM[rr, srcFull[rr, cci]]      # left half stays base
        return comp

    def draw_guides(self, view):
        """Side-by-side: a vertical split line. Mirrorscope: the square lens
        box and its center mirror line."""
        mode = self.mode()
        if mode == "Sidebyside":
            s = self.split.get(view.orientation)
            if s is not None:
                view.ax.axvline(s, color="#e8c542", lw=1.2, zorder=14)
        elif mode == "Mirrorscope":
            c = self.lens.get(view.orientation)
            if c is None:
                return
            cx, cy = c
            half = self.sizeSpin.value()
            view.ax.add_patch(mpatches.Rectangle(
                (cx - half, cy - half), 2 * half, 2 * half, fill=False,
                edgecolor="#e8c542", lw=1.2, zorder=14))
            view.ax.plot([cx, cx], [cy - half, cy + half], color="#e8c542",
                         lw=0.7, ls="--", alpha=0.8, zorder=14)

    # -------------------------------------------------------------- close ---
    def closeEvent(self, event):
        for v in self.viewer.views.values():
            v.qa_split_cb = None
            v._qa_drag = False
        self.viewer.regCtl = None
        self.viewer.refresh_views()
        event.accept()


# ---------------------------------------------------------------------------#
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


_APP = None  # keep a reference so the QApplication is never garbage-collected


def _get_app():
    global _APP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        _apply_theme_palette(app)
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
