"""Interactive colorbar widgets for dose and scan display."""
from cerr.viewer.pycerr_gui.common import *  # noqa: F401,F403

class RangeColorbarWidget(QtWidgets.QGraphicsView):
    """Retained-mode colorbar with two draggable range pairs.

    Built on a QGraphicsScene so dragging a handle only repositions that handle
    item (``setPos``) and, for the display range, resizes the two dimming
    overlays (``setRect``) - no full-widget repaint. The gradient pixmap is
    rebuilt only when the colormap or the colormap-mapping (yellow) range
    changes (a cbar-handle drag), never when just the display range or a handle
    moves.

    Orientation is set per subclass with ``_HORIZONTAL``. A horizontal bar can
    also show an intensity histogram above the gradient (``setHistogram``) so
    the user sees the value distribution while setting window/level."""
    rangesChanged = QtCore.pyqtSignal()

    GRAB_PX = 9          # grab tolerance for handles (px, across the value axis)
    # vertical layout
    TOP, BOT = 16, 16    # margins
    BAR_X, BAR_W = 44, 24
    # horizontal layout
    HMARG = 38           # left/right margin (room for end labels)
    HTOP = 4             # top margin (histogram starts here)
    HIST_H = 46          # histogram band height
    TOPGAP = 12          # yellow-handle band between histogram and bar
    HBAR_H = 22          # gradient bar height
    BOTGAP = 12          # cyan-handle band below the bar
    LABELH = 18          # tick-label band at the bottom

    _CMAP_NAMES = []                    # colormaps offered in the context menu
    _DEFAULT_CMAP = "gray"
    _KIND = "value"                     # noun used in the tooltip
    _HORIZONTAL = False                 # subclasses flip this for a wide bar

    def __init__(self, parent=None):
        super().__init__(parent)
        self._horizontal = bool(self._HORIZONTAL)
        if self._horizontal:
            self.setFixedHeight(self.HTOP + self.HIST_H + self.TOPGAP
                                + self.HBAR_H + self.BOTGAP + self.LABELH)
            self.setMinimumWidth(260)
        else:
            self.setFixedWidth(132)
            self.setMinimumHeight(240)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setStyleSheet("QGraphicsView{background:transparent;border:none;}")
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.viewport().setMouseTracking(True)

        self.axisMin = 0.0
        self.axisMax = 1.0
        self.cbarRange = [0.0, 1.0]      # colormap mapping range
        self.dispRange = [0.0, 1.0]      # display (mask) range
        self._dataMin = None             # true data range (for Reset)
        self._dataMax = None
        self._histSamples = None         # 1D intensities for the histogram
        self._drag = None                # ("cbar"|"disp", 0|1) while dragging
        self.cmapName = self._DEFAULT_CMAP
        self._set_cmap(self.cmapName)
        self._build_items()
        drag = ("Top yellow" if self._horizontal else "Left yellow")
        cyan = ("bottom cyan" if self._horizontal else "right cyan")
        self.setToolTip(f"{self._KIND.capitalize()} colorbar\n"
                        f"{drag} handles: colorbar/colormap range\n"
                        f"{cyan} handles: {self._KIND} display range\n"
                        "Right-click: colormap & exact ranges\n"
                        "Double-click: reset ranges")

    def _set_cmap(self, name):
        self.cmapName = name
        self.mplCmap = cerr_get_cmap(name)
        self._lut = cerr_get_lut(name, 256)
        if hasattr(self, "_barItem"):   # rebuild the gradient on cmap change
            self._rebuild_gradient()

    def _span(self):
        return max(self.axisMax - self.axisMin, 1e-9)

    # ------------------------------------------------------- scene items ----
    def _build_items(self):
        """Create the persistent scene items once; positions/pixmap are updated
        in place on drag/resize instead of being re-created."""
        s = self._scene
        if self._horizontal:            # intensity histogram above the bar
            self._histItem = QtWidgets.QGraphicsPathItem()
            self._histItem.setBrush(QtGui.QBrush(QtGui.QColor(90, 140, 200, 150)))
            self._histItem.setPen(QtGui.QPen(QtGui.QColor(60, 100, 160), 1))
            self._histItem.setZValue(0)
            s.addItem(self._histItem)

        self._barItem = QtWidgets.QGraphicsPixmapItem()
        self._barItem.setZValue(0)
        s.addItem(self._barItem)

        dimBrush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 200))
        noPen = QtGui.QPen(Qt.NoPen)
        self._dimTop = s.addRect(0, 0, 0, 0, noPen, dimBrush)   # above disp max
        self._dimBot = s.addRect(0, 0, 0, 0, noPen, dimBrush)   # below disp min
        for it in (self._dimTop, self._dimBot):
            it.setZValue(1)

        self._frameItem = s.addRect(0, 0, 0, 0,
                                    QtGui.QPen(QtGui.QColor("#aaaaaa"), 1))
        self._frameItem.setZValue(2)

        self._tickLines, self._tickLabels = [], []
        tickPen = QtGui.QPen(QtGui.QColor("#888888"), 1)
        for _ in range(6):
            ln = s.addLine(0, 0, 0, 0, tickPen)
            ln.setZValue(2)
            self._tickLines.append(ln)
            tx = s.addSimpleText("")
            tx.setBrush(QtGui.QColor("black"))
            f = tx.font()
            f.setPointSize(9)
            tx.setFont(f)
            tx.setZValue(2)
            self._tickLabels.append(tx)

        self._handleItems, self._handleLabels = {}, {}
        for which, color in (("cbar", QtGui.QColor("#e8c542")),
                             ("disp", QtGui.QColor("#3ad6e0"))):
            for j in (0, 1):
                poly = QtWidgets.QGraphicsPolygonItem(
                    self._handle_polygon(which))
                poly.setBrush(color)
                poly.setPen(QtGui.QPen(Qt.black, 1))
                poly.setZValue(3)
                s.addItem(poly)
                self._handleItems[(which, j)] = poly
                lab = s.addSimpleText("")
                lab.setBrush(QtGui.QColor("black"))
                f = lab.font()
                f.setPointSize(9)
                lab.setFont(f)
                lab.setZValue(3)
                self._handleLabels[(which, j)] = lab

    def _handle_polygon(self, which):
        """Triangle in the item's local coords, tip at (0, 0) pointing at the
        bar; the item is placed by moving that tip to (x, y) via setPos.

        Vertical: cbar points from the left, disp from the right. Horizontal:
        cbar (top) points down, disp (bottom) points up."""
        tri = QtGui.QPolygonF()
        if self._horizontal:
            dy = -(self.TOPGAP - 2) if which == "cbar" else (self.BOTGAP - 2)
            tri << QtCore.QPointF(0, 0) \
                << QtCore.QPointF(-6, dy) << QtCore.QPointF(6, dy)
        else:
            dx = -11 if which == "cbar" else 11   # cbar=left, disp=right
            tri << QtCore.QPointF(0, 0) \
                << QtCore.QPointF(dx, -6) << QtCore.QPointF(dx, 6)
        return tri

    # ------------------------------------------------------------ public ----
    def setHistogram(self, samples):
        """Set (or clear, with None) the intensity samples whose distribution is
        drawn above a horizontal bar. Large arrays are subsampled for speed."""
        if samples is None:
            self._histSamples = None
        else:
            a = np.asarray(samples).ravel()
            if a.size > 120000:          # subsample first (shape is what matters)
                a = a[::int(a.size // 120000) + 1]
            a = a.astype(np.float32, copy=False)
            self._histSamples = a[np.isfinite(a)]
        self._update_histogram()

    def setMarkers(self, vals):
        """Show tick marks across the bar at the given axis values (e.g. the
        selected isodose levels). Pass an empty list to clear."""
        self.markerVals = [float(v) for v in vals]
        self._update_markers()

    def _update_markers(self):
        for it in getattr(self, "_markerItems", []):
            self._scene.removeItem(it)
        self._markerItems = []
        vals = getattr(self, "markerVals", [])
        if not vals or self._vp_size() <= 1:
            return
        r = self._bar_rect()
        haloPen = QtGui.QPen(QtGui.QColor("white"), 3)
        inkPen = QtGui.QPen(QtGui.QColor("black"), 1)
        for v in vals:
            if not (self.axisMin <= v <= self.axisMax):
                continue
            p = self._valpos(v)
            for pen, z in ((haloPen, 2.5), (inkPen, 2.6)):
                if self._horizontal:
                    ln = self._scene.addLine(p, r.top() - 4, p, r.bottom() + 4,
                                             pen)
                else:
                    ln = self._scene.addLine(r.left() - 4, p, r.right() + 4, p,
                                             pen)
                ln.setZValue(z)     # above the dimming overlays
                self._markerItems.append(ln)

    def setRange(self, axisMin, axisMax, cbarRange=None, dispRange=None):
        """Set the axis extent and (optionally) the two ranges.

        When ``cbarRange``/``dispRange`` are omitted the corresponding range is
        reset to the full axis. Callers that only widen the axis and want to
        preserve the current handle positions should pass them explicitly."""
        self.axisMin = float(axisMin)
        self.axisMax = max(float(axisMax), self.axisMin + 1e-6)
        self.cbarRange = list(cbarRange) if cbarRange is not None \
            else [self.axisMin, self.axisMax]
        self.dispRange = list(dispRange) if dispRange is not None \
            else [self.axisMin, self.axisMax]
        self._rebuild()

    # ---------------------------------------------------------- geometry ----
    def _vp_size(self):
        """Extent of the viewport along the value axis (px)."""
        return self.viewport().width() if self._horizontal \
            else self.viewport().height()

    def _bar_rect(self):
        if self._horizontal:
            w = max(self.viewport().width() - 2 * self.HMARG, 1)
            top = self.HTOP + self.HIST_H + self.TOPGAP
            return QtCore.QRectF(self.HMARG, top, w, self.HBAR_H)
        h = max(self.viewport().height() - self.TOP - self.BOT, 1)
        return QtCore.QRectF(self.BAR_X, self.TOP, self.BAR_W, h)

    def _valpos(self, v):
        """Value -> pixel position along the value axis (x if horizontal, else
        y). Low values are left/bottom, high values right/top."""
        r = self._bar_rect()
        f = (v - self.axisMin) / self._span()
        if self._horizontal:
            return r.left() + f * r.width()
        return r.bottom() - f * r.height()

    def _pos2val(self, p):
        r = self._bar_rect()
        if self._horizontal:
            f = (p - r.left()) / max(r.width(), 1)
        else:
            f = (r.bottom() - p) / max(r.height(), 1)
        return self.axisMin + float(np.clip(f, 0, 1)) * self._span()

    # --------------------------------------------------------- item update --
    def _rebuild(self):
        """Full relayout after a size/axis/range change."""
        if self._vp_size() <= 1:
            return          # not yet shown/sized; showEvent will rebuild
        self._scene.setSceneRect(0, 0, self.viewport().width(),
                                 self.viewport().height())
        self._rebuild_gradient()
        self._update_histogram()
        self._update_dim()
        self._update_ticks()
        self._update_handles()
        self._update_markers()

    def _rebuild_gradient(self):
        """Rebuild the gradient pixmap (colormap over the cbar range). Dimming
        outside the display range is handled by the overlay rects, so this is
        only needed when the colormap or cbar range changes."""
        if not hasattr(self, "_barItem") or self._vp_size() <= 1:
            return
        r = self._bar_rect()
        H = max(int(round(r.height())), 1)
        W = max(int(round(r.width())), 1)
        cbLo, cbHi = self.cbarRange
        span = max(cbHi - cbLo, 1e-9)
        if self._horizontal:            # value increases left -> right
            xs = np.arange(W)
            vals = self.axisMin + (xs / max(W - 1, 1)) * self._span()
            idx = (np.clip((vals - cbLo) / span, 0, 1) * 255).astype(np.intp)
            cols = self._lut[idx]                       # (W, 3)
            rgba = np.empty((H, W, 4), np.uint8)
            rgba[..., :3] = cols[None, :, :]
        else:                           # value increases bottom -> top
            ys = np.arange(H)                       # 0 = top row = high value
            vals = self.axisMax - (ys / H) * self._span()
            idx = (np.clip((vals - cbLo) / span, 0, 1) * 255).astype(np.intp)
            cols = self._lut[idx]                       # (H, 3)
            rgba = np.empty((H, W, 4), np.uint8)
            rgba[..., :3] = cols[:, None, :]
        rgba[..., 3] = 255
        self._barImg = np.ascontiguousarray(rgba)
        img = QtGui.QImage(self._barImg.data, W, H, 4 * W,
                           QtGui.QImage.Format_RGBA8888)
        self._barItem.setPixmap(QtGui.QPixmap.fromImage(img))
        self._barItem.setPos(r.left(), r.top())
        self._frameItem.setRect(r)

    def _update_histogram(self):
        """Redraw the intensity histogram above a horizontal bar (aligned to the
        value axis). A no-op for vertical bars or when no samples are set."""
        if not self._horizontal or not hasattr(self, "_histItem"):
            return
        a = self._histSamples
        r = self._bar_rect()
        if a is None or a.size == 0 or self.viewport().width() <= 1:
            self._histItem.setPath(QtGui.QPainterPath())
            return
        nb = int(max(min(r.width() // 2, 160), 16))
        counts, _edges = np.histogram(
            a, bins=nb, range=(self.axisMin, self.axisMax))
        mx = counts.max()
        bottom = self.HTOP + self.HIST_H
        if mx <= 0:
            self._histItem.setPath(QtGui.QPainterPath())
            return
        # sqrt scaling so low-frequency intensities (e.g. lesions) stay visible
        norm = np.sqrt(counts.astype(float) / float(mx))
        path = QtGui.QPainterPath()
        path.moveTo(r.left(), bottom)
        for i in range(nb):
            x = r.left() + (i + 0.5) / nb * r.width()
            path.lineTo(x, bottom - norm[i] * self.HIST_H)
        path.lineTo(r.right(), bottom)
        path.closeSubpath()
        self._histItem.setPath(path)

    def _update_dim(self):
        """Resize the two dimming overlays to cover values outside dispRange."""
        r = self._bar_rect()
        dLo, dHi = self.dispRange
        if self._horizontal:
            xLo, xHi = self._valpos(dLo), self._valpos(dHi)
            self._dimBot.setRect(r.left(), r.top(),   # below disp min (left)
                                 max(xLo - r.left(), 0), r.height())
            self._dimTop.setRect(xHi, r.top(),        # above disp max (right)
                                 max(r.right() - xHi, 0), r.height())
        else:
            yHi, yLo = self._valpos(dHi), self._valpos(dLo)
            self._dimTop.setRect(r.left(), r.top(), r.width(),
                                 max(yHi - r.top(), 0))
            self._dimBot.setRect(r.left(), yLo, r.width(),
                                 max(r.bottom() - yLo, 0))

    def _update_ticks(self):
        r = self._bar_rect()
        for i, frac in enumerate(np.linspace(0, 1, 6)):
            v = self.axisMin + frac * self._span()
            p = self._valpos(v)
            tx = self._tickLabels[i]
            tx.setText(f"{v:.4g}")
            br = tx.boundingRect()
            if self._horizontal:
                yb = r.bottom() + self.BOTGAP
                self._tickLines[i].setLine(p, yb, p, yb + 4)
                tx.setPos(p - br.width() / 2, yb + 5)
            else:
                tickX = r.right() + 18
                self._tickLines[i].setLine(tickX, p, tickX + 4, p)
                tx.setPos(tickX + 7, p - br.height() / 2)

    def _update_handles(self):
        """Reposition each handle triangle + its value label (setPos only)."""
        r = self._bar_rect()
        vw = self.viewport().width()
        for which, rng in (("cbar", self.cbarRange), ("disp", self.dispRange)):
            for j in (0, 1):
                v = rng[j]
                p = self._valpos(v)
                poly = self._handleItems[(which, j)]
                lab = self._handleLabels[(which, j)]
                lab.setText(f"{v:.3g}")
                br = lab.boundingRect()
                if self._horizontal:
                    if which == "cbar":
                        poly.setPos(p, r.top() - 1)
                        lab.setPos(min(max(p - br.width() / 2, 0),
                                       vw - br.width()),
                                   r.top() - self.TOPGAP - br.height())
                    else:
                        poly.setPos(p, r.bottom() + 1)
                        lab.setPos(min(max(p - br.width() / 2, 0),
                                       vw - br.width()), r.bottom() + 1)
                else:
                    if which == "cbar":
                        poly.setPos(r.left() - 3, p)
                        lab.setPos(r.left() - 16 - br.width(),
                                   p - br.height() / 2)
                    else:
                        poly.setPos(r.right() + 4, p)
                        lab.setPos(vw - 2 - br.width(), p - br.height() / 2)

    # -------------------------------------------------------- Qt overrides --
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rebuild()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._rebuild()

    # ------------------------------------------------------------ mouse -----
    def _hit_test(self, pos):
        r = self._bar_rect()
        if self._horizontal:
            which = "cbar" if pos.y() < r.center().y() else "disp"
            coord = pos.x()
        else:
            which = "cbar" if pos.x() < r.center().x() else "disp"
            coord = pos.y()
        rng = self.cbarRange if which == "cbar" else self.dispRange
        best, bestDist = None, self.GRAB_PX + 1
        for j in (0, 1):
            d = abs(coord - self._valpos(rng[j]))
            if d < bestDist:
                best, bestDist = (which, j), d
        return best

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        pos = self.mapToScene(ev.pos())
        self._drag = self._hit_test(pos)
        if self._drag:
            self._move_to(pos)

    # ----------------------------------------------------- context menu -----
    def contextMenuEvent(self, ev):
        menu = QtWidgets.QMenu(self)

        cmapMenu = menu.addMenu("Colormap")
        grp = QtWidgets.QActionGroup(cmapMenu)
        for name in self._CMAP_NAMES:
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
        menu.addAction("Reset ranges", self._reset_ranges)
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
        self._set_cmap(name)          # rebuilds the gradient
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
            sp.setRange(self.axisMin, self.axisMax)
            sp.setSingleStep(self._span() / 100.0)
            sp.setValue(val)
            form.addRow(label, sp)
            spins.append(sp)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        def _apply():
            eps = self._span() * 1e-3
            cbLo, cbHi = spins[0].value(), spins[1].value()
            dLo, dHi = spins[2].value(), spins[3].value()
            self.cbarRange = [min(cbLo, cbHi - eps), max(cbHi, cbLo + eps)]
            self.dispRange = [min(dLo, dHi - eps), max(dHi, dLo + eps)]
            self._rebuild_gradient()
            self._update_dim()
            self._update_handles()
            self.rangesChanged.emit()
        # Non-modal (a modal exec_ hangs in an integrated event loop): apply on OK.
        dlg.accepted.connect(_apply)
        dlg.setModal(False)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.show()
        dlg.raise_()

    def _sync_disp_to_cbar(self):
        self.dispRange = list(self.cbarRange)
        self._update_dim()
        self._update_handles()
        self.rangesChanged.emit()

    def _reset_ranges(self):
        """Reset the colorbar and display ranges. The axis is restored to the
        true data range (when known) so a previously widened window doesn't
        leave the bar stuck zoomed-out - the whole widget (gradient, ticks,
        handles, histogram) then redraws."""
        if self._dataMin is not None and self._dataMax is not None:
            self.axisMin = float(self._dataMin)
            self.axisMax = max(float(self._dataMax), self.axisMin + 1e-6)
        self.cbarRange = [self.axisMin, self.axisMax]
        self.dispRange = [self.axisMin, self.axisMax]
        self._rebuild()
        self.rangesChanged.emit()

    def mouseMoveEvent(self, ev):
        pos = self.mapToScene(ev.pos())
        if self._drag:
            self._move_to(pos)
        else:
            self.setCursor(Qt.PointingHandCursor if self._hit_test(pos)
                           else Qt.ArrowCursor)

    def mouseReleaseEvent(self, _ev):
        if self._drag:
            self._drag = None
            self.rangesChanged.emit()

    def mouseDoubleClickEvent(self, _ev):
        self._reset_ranges()

    def _move_to(self, pos):
        which, j = self._drag
        rng = self.cbarRange if which == "cbar" else self.dispRange
        v = self._pos2val(pos.x() if self._horizontal else pos.y())
        eps = self._span() * 1e-3
        rng[j] = min(v, rng[1] - eps) if j == 0 else max(v, rng[0] + eps)
        rng[j] = float(np.clip(rng[j], self.axisMin, self.axisMax))
        # retained-mode update: move only what changed. A cbar (yellow) handle
        # remaps the colormap so the gradient is rebuilt; a disp (cyan) handle
        # only re-dims, so we just resize the overlay rects.
        self._update_handles()
        if which == "cbar":
            self._rebuild_gradient()
        else:
            self._update_dim()
        self.rangesChanged.emit()    # live update while dragging


class DoseColorbarWidget(RangeColorbarWidget):
    """Dose colorbar: axis pinned to [0, doseMax], CERR/matplotlib dose maps."""
    _CMAP_NAMES = DOSE_CMAP_NAMES
    _DEFAULT_CMAP = DEFAULT_DOSE_CMAP
    _KIND = "dose"

    def setDose(self, doseMax):
        self._dataMin, self._dataMax = 0.0, max(float(doseMax), 1e-6)
        self.setRange(0.0, max(float(doseMax), 1e-6))


class ScanColorbarWidget(RangeColorbarWidget):
    """Base-scan colorbar: arbitrary axis (e.g. CT Hounsfield units), gray by
    default. The colormap-mapping (yellow) range is the window (center/width);
    the display (cyan) range hides intensities outside it in the 2D views. Shown
    horizontally with an intensity histogram above the gradient."""
    _CMAP_NAMES = SCAN_CMAPS
    _DEFAULT_CMAP = "gray"
    _KIND = "scan"
    _HORIZONTAL = True

    def setScan(self, center, width, dataMin, dataMax):
        lo = center - width / 2.0
        hi = center + width / 2.0
        self._dataMin, self._dataMax = float(dataMin), float(dataMax)
        axisMin = min(float(dataMin), lo)
        axisMax = max(float(dataMax), hi)
        # window -> colormap range; display range starts at the full axis
        self.setRange(axisMin, axisMax, cbarRange=[lo, hi])

    def setWindow(self, center, width):
        """Update only the colormap (yellow) range from a window, widening the
        axis if needed and preserving the user's display (cyan) range."""
        lo = center - width / 2.0
        hi = center + width / 2.0
        eps = self._span() * 1e-3
        # was the display range covering the whole axis? keep it full if so
        dispFull = (self.dispRange[0] <= self.axisMin + eps
                    and self.dispRange[1] >= self.axisMax - eps)
        self.axisMin = min(self.axisMin, lo)
        self.axisMax = max(self.axisMax, hi)
        if dispFull:
            self.dispRange = [self.axisMin, self.axisMax]
        self.cbarRange = [lo, hi]
        self._rebuild()


# ---------------------------------------------------------------------------#
#  Main window
# ---------------------------------------------------------------------------#
