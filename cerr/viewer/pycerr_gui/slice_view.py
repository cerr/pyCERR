"""SliceView: one orthogonal slice view (matplotlib canvas in a Qt widget)."""
from cerr.viewer.pycerr_gui.common import *  # noqa: F401,F403

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
        self._scanIm = None          # persistent base-scan image (set_data reuse)
        self.draw_mode = False       # contouring active on this view
        self.draw_tool = "freehand"  # "freehand" | "polygon" | "brush"
        self.brush_radius = 0.5      # cm (data units)
        self.brush_erase = False     # tints the brush ball (draw vs erase)
        self._stroke = None          # [(x, y), ...] while drag-drawing
        self._stroke_line = None
        self._poly = None            # clicked vertices in polygon mode
        self._poly_line = None
        self._brush_circle = None    # brush-size cursor
        self.vtk_widget = None       # pyvista QtInteractor for the 3D view
        self.qa_split_cb = None      # registration-QA split drag hook
        self._qa_drag = False
        self._chBg = None            # canvas background for crosshair blitting

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
        self.canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        self.canvas.mpl_connect("draw_event", self._capture_crosshair_bg)

    @property
    def is3d(self):
        return self.orientation == VIEW_3D

    # --------------------------------------------------- crosshair blitting --
    # The crosshair lines are created 'animated' (excluded from normal draws).
    # After every full canvas draw we cache the rendered background and paint
    # the lines on top; moving the crosshair then only needs restore + blit
    # instead of a full ~100 ms redraw of the whole view.
    def _capture_crosshair_bg(self, _event=None):
        if self.is3d or self.xline is None:
            self._chBg = None
            return
        try:
            self._chBg = self.canvas.copy_from_bbox(self.ax.bbox)
            self._paint_crosshair()
        except Exception:  # noqa: BLE001  (non-Agg renderer, e.g. savefig)
            self._chBg = None

    def _paint_crosshair(self):
        for ln in (self.xline, self.yline):
            if ln is not None:
                self.ax.draw_artist(ln)
        self.canvas.blit(self.ax.bbox)

    def blit_crosshair(self):
        """Fast crosshair reposition: restore the cached background and repaint
        just the two lines. Falls back to a full redraw when no background has
        been captured yet (e.g. before the first paint)."""
        if self._chBg is None or self.xline is None:
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._chBg)
        self._paint_crosshair()

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
                self.vtk_widget.doubleClicked.connect(self.reset_view)
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
        self._chBg = None
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
        if self.brush_erase:          # reddish pink for erase
            edge, face = (1.0, 0.25, 0.45, 0.95), (1.0, 0.25, 0.45, 0.22)
        else:                         # aqua green for draw
            edge, face = (0.0, 0.85, 0.6, 0.95), (0.0, 0.85, 0.6, 0.22)
        if self._brush_circle is None or self._brush_circle.axes is not self.ax:
            self._brush_circle = mpatches.Circle(
                (x, y), self.brush_radius, facecolor=face, edgecolor=edge,
                lw=2.0, zorder=15)
            self.ax.add_patch(self._brush_circle)
        else:
            self._brush_circle.center = (x, y)
            self._brush_circle.set_radius(self.brush_radius)
            self._brush_circle.set_facecolor(face)
            self._brush_circle.set_edgecolor(edge)
        self.ax.set_xlim(xlim)        # keep the view fixed while brushing
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def _on_axes_leave(self, _event):
        # drop the brush-size preview when the cursor leaves the image, so it
        # doesn't linger at the edge (only while not mid-stroke)
        if self._stroke is None and self._brush_circle is not None:
            try:
                self._brush_circle.remove()
            except Exception:  # noqa: BLE001
                pass
            self._brush_circle = None
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
#  Standalone colorbar with draggable range markers (CERR-style)
#    * LEFT handles (yellow)  : colorbar/colormap mapping range
#    * RIGHT handles (cyan)   : data display range (values outside are hidden)
#  Double-click resets both ranges to the full axis [axisMin, axisMax].
#
#  ``RangeColorbarWidget`` is the shared base; ``DoseColorbarWidget`` (dose,
#  axis pinned at 0) and ``ScanColorbarWidget`` (base scan, arbitrary axis
#  including negative CT Hounsfield units) only differ in their colormap list,
#  default map and how the axis range is seeded.
# ---------------------------------------------------------------------------#
