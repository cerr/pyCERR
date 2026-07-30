"""Viewer dialogs: DVH, contouring, export and registration-QA tools."""
from cerr.viewer.pycerr_gui.common import *  # noqa: F401,F403

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
        self._lastLiveContourT = 0.0     # throttle for the live boundary

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
        self.drawBtn.toggled.connect(self._on_mode_changed)  # recolor the ball
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
        bb.rejected.connect(self.close)   # Close has RejectRole -> rejected
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
        axView.brush_erase = self.eraseBtn.isChecked()
        axView.draw_tool = ("polygon" if self.polyBtn.isChecked()
                            else "brush" if self.brushBtn.isChecked()
                            else "freehand")
        axView.canvas.setCursor(
            _contour_cursor("brush" if axView.draw_tool == "brush" else "pen"))
        if axView.draw_tool == "brush" and axView._brush_circle is not None:
            cx, cy = axView._brush_circle.center   # recolor on-screen ball now
            axView._update_brush_cursor(cx, cy)

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
            self._confirm_discard(
                onYes=lambda: self._apply_struct_selection(idx),
                onNo=lambda: self._populate_structs(self.structNum))
            return
        self._apply_struct_selection(idx)

    def _confirm_discard(self, onYes, onNo=None):
        """"Discard unsaved edits?" confirmation (modal QDialog shown via
        show(), so it grabs focus yet stays non-blocking). A guard ensures only
        one prompt is ever open, so repeated triggers cannot stack dialogs.
        """
        if getattr(self, "_discardDlg", None) is not None:
            self._discardDlg.raise_()
            self._discardDlg.activateWindow()
            return
        dlg = QtWidgets.QDialog(self)
        self._discardDlg = dlg
        dlg.destroyed.connect(lambda *_: setattr(self, "_discardDlg", None))
        dlg.setWindowTitle("Contouring")
        # Modal (but shown via show(), not exec_): grabs input focus so the
        # FIRST click lands on a button - a non-modal dialog needs one click to
        # activate the window and a second to press the button. show() keeps it
        # non-blocking, so it is safe in the %gui qt notebook event loop too.
        dlg.setModal(True)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(QtWidgets.QLabel("Discard unsaved edits?"))
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        yesBtn = QtWidgets.QPushButton("Yes")
        noBtn = QtWidgets.QPushButton("No")
        noBtn.setDefault(True)
        row.addWidget(yesBtn)
        row.addWidget(noBtn)
        lay.addLayout(row)

        def _yes():
            dlg.close()
            if onYes is not None:
                QtCore.QTimer.singleShot(0, onYes)

        def _no():
            dlg.close()
            if onNo is not None:
                QtCore.QTimer.singleShot(0, onNo)

        yesBtn.clicked.connect(_yes)
        noBtn.clicked.connect(_no)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

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
            self._lastLiveContourT = 0.0   # draw the boundary on the first step
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
        """Cheap live overlay update on the axial view only, so brushing stays
        responsive. The filled overlay is reused via set_data every step, while
        the dashed boundary (an expensive marching-squares contour) is throttled
        - it is fully redrawn on release in _on_brush_done."""
        v = self.viewer
        view = self.axView
        cslc = self.mask3M[:, :, self._cur_slice()]
        extent = [v.xV[0], v.xV[-1], v.yV[-1], v.yV[0]]
        data = np.ma.masked_where(~cslc, cslc.astype(float))

        # --- filled overlay: reuse the artist (set_data) instead of recreating
        if self._liveIm is None or self._liveIm.axes is not view.ax:
            # imshow() resets the axes limits; preserve the current pan/zoom.
            xlim, ylim = view.ax.get_xlim(), view.ax.get_ylim()
            self._liveIm = view.ax.imshow(
                data, cmap=ListedColormap([self.color]), extent=extent,
                alpha=0.35, vmin=0, vmax=1, interpolation="nearest",
                aspect="equal", zorder=9)
            view.ax.set_xlim(xlim)
            view.ax.set_ylim(ylim)
        else:
            self._liveIm.set_data(data)

        # --- dashed boundary: throttled (recomputing ax.contour every motion
        # event is the dominant cost). At most ~20 redraws/s while dragging.
        now = time.monotonic()
        if now - self._lastLiveContourT > 0.05:
            self._lastLiveContourT = now
            xlim, ylim = view.ax.get_xlim(), view.ax.get_ylim()
            self._remove_live_contour()
            if np.any(cslc):
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
            def _yes():
                self._force_close = True
                self.close()
            self._confirm_discard(onYes=_yes)
            event.ignore()           # wait for the (non-modal) answer
            return
        self._detach(self.axView)
        self.viewer.contourCtl = None
        self.viewer.refresh_views()
        event.accept()


# ---------------------------------------------------------------------------#
#  Scan / dose export tool: pick one scan (or dose) and write it to NIfTI.
# ---------------------------------------------------------------------------#
class ScanDoseExportDialog(QtWidgets.QDialog):
    """Non-modal dialog to export one scan or dose to a NIfTI file."""

    def __init__(self, viewer, kind):
        super().__init__(viewer)
        self.viewer = viewer
        self.kind = kind                     # "scan" or "dose"
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(f"Export {kind} to NIfTI")

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.combo = QtWidgets.QComboBox()
        if kind == "scan":
            for i, s in enumerate(viewer.planC.scan):
                mod = getattr(s.scanInfo[0], "imageType", "scan")
                self.combo.addItem(f"{i}: {mod}")
            cur = viewer.scanNum
        else:
            for i, d in enumerate(viewer.planC.dose):
                self.combo.addItem(
                    f"{i}: {getattr(d, 'fractionGroupID', 'dose')}")
            cur = viewer.doseNum
        if 0 <= cur < self.combo.count():
            self.combo.setCurrentIndex(cur)
        form.addRow(f"{kind.capitalize()}:", self.combo)
        lay.addLayout(form)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addStretch(1)
        expBtn = QtWidgets.QPushButton("Export...")
        expBtn.setDefault(True)
        closeBtn = QtWidgets.QPushButton("Close")
        expBtn.clicked.connect(self._export)
        closeBtn.clicked.connect(self.close)
        btnRow.addWidget(expBtn)
        btnRow.addWidget(closeBtn)
        lay.addLayout(btnRow)

    def _export(self):
        idx = self.combo.currentIndex()
        if idx < 0:
            return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Export {self.kind} to NIfTI",
            filter="NIfTI (*.nii.gz *.nii)")
        if not f:
            return
        try:
            obj = (self.viewer.planC.scan if self.kind == "scan"
                   else self.viewer.planC.dose)[idx]
            obj.saveNii(f)
            self.viewer.statusBar().showMessage(
                f"Exported {self.kind} {idx} to {f}")
            self.close()
        except Exception as e:  # noqa: BLE001
            _show_error(self, "Export error", str(e))


# ---------------------------------------------------------------------------#
#  Structure export tool: pick structures and write them to a single NIfTI
#  (label map / 4D stack) or a single DICOM RTSTRUCT file.
# ---------------------------------------------------------------------------#
class StructureExportDialog(QtWidgets.QDialog):
    """Non-modal dialog to export selected structures to NIfTI or DICOM."""

    def __init__(self, viewer, fmt):
        super().__init__(viewer)
        self.viewer = viewer
        self.fmt = fmt                       # "nii" or "dicom"
        isNii = fmt == "nii"
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("Export structures to "
                            + ("NIfTI" if isNii else "DICOM RTSTRUCT"))

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Select structures to export:"))
        self.listW = QtWidgets.QListWidget()
        for i, st in enumerate(viewer.planC.structure):
            it = QtWidgets.QListWidgetItem(f"{i}: {st.structureName}")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            it.setData(Qt.UserRole, i)
            self.listW.addItem(it)
        lay.addWidget(self.listW, 1)

        selRow = QtWidgets.QHBoxLayout()
        bAll = QtWidgets.QPushButton("Select all")
        bNone = QtWidgets.QPushButton("Select none")
        bAll.clicked.connect(lambda: self._set_all(Qt.Checked))
        bNone.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        selRow.addWidget(bAll)
        selRow.addWidget(bNone)
        selRow.addStretch(1)
        lay.addLayout(selRow)

        if isNii:
            self.sepChk = QtWidgets.QCheckBox(
                "Separate binary mask per structure (4D); "
                "otherwise one label map")
            lay.addWidget(self.sepChk)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addStretch(1)
        expBtn = QtWidgets.QPushButton("Export...")
        expBtn.setDefault(True)
        closeBtn = QtWidgets.QPushButton("Close")
        expBtn.clicked.connect(self._export)
        closeBtn.clicked.connect(self.close)
        btnRow.addWidget(expBtn)
        btnRow.addWidget(closeBtn)
        lay.addLayout(btnRow)

    def _set_all(self, state):
        for i in range(self.listW.count()):
            self.listW.item(i).setCheckState(state)

    def _selected(self):
        return [self.listW.item(i).data(Qt.UserRole)
                for i in range(self.listW.count())
                if self.listW.item(i).checkState() == Qt.Checked]

    def _same_scan(self, strNumV):
        """All selected structures must share one associated scan."""
        scans = {self.viewer.planC.structure[s].getStructureAssociatedScan(
                     self.viewer.planC) for s in strNumV}
        return len(scans) == 1

    def _export(self):
        strNumV = self._selected()
        if not strNumV:
            _show_info(self, "Export", "Select at least one structure.")
            return
        if not self._same_scan(strNumV):
            _show_warning(
                self, "Export",
                "Selected structures are associated with different scans.\n"
                "Please export structures from a single scan at a time.")
            return
        planC = self.viewer.planC
        if self.fmt == "nii":
            f, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export structures to NIfTI",
                filter="NIfTI (*.nii.gz *.nii)")
            if not f:
                return
            try:
                # getLabelMap matches by structure name -> integer label.
                labelDict = {planC.structure[s].structureName: i + 1
                             for i, s in enumerate(strNumV)}
                dim = 4 if (hasattr(self, "sepChk")
                            and self.sepChk.isChecked()) else 3
                pc.saveNiiStructure(f, labelDict, planC, strNumV, dim=dim)
                self.viewer.statusBar().showMessage(
                    f"Exported {len(strNumV)} structure(s) to {f}")
                self.close()
            except Exception as e:  # noqa: BLE001
                _show_error(self, "Export error", str(e))
        else:
            f, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export structures to RTSTRUCT",
                filter="DICOM (*.dcm)")
            if not f:
                return
            try:
                from cerr.dcm_export import rtstruct_iod
                rtstruct_iod.create(strNumV, f, planC,
                                    {"seriesDescription": "Exported from pyCERR"})
                self.viewer.statusBar().showMessage(
                    f"Exported {len(strNumV)} structure(s) to {f}")
                self.close()
            except Exception as e:  # noqa: BLE001
                _show_error(self, "Export error", str(e))


# ---------------------------------------------------------------------------#
#  urOMT tool: run the cerr.uromt fluid-transport pipeline (Part 1 + Part 2)
#  on the longitudinal scans in planC, with a structure as the ROI. The
#  optimization runs in a background thread so the GUI stays responsive.
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
        bb.rejected.connect(self.close)   # Close has RejectRole -> rejected
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
        rendered with its own colormap. None falls back to normal display.

        Registration QA ignores the scans' Scan-Display opacities: base and
        moving are drawn fully opaque (opacity 1) in Mirrorscope, AlternateGrid
        and Side-by-side, and blended by the fade slider (base weight, moving =
        1 - base) in Toggle. The selected scan opacities are restored when QA is
        exited (this composite is only used while the QA tool is open)."""
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

        # render each scan through its own Scan Display colormap to RGB
        movCmap = v._scan_display(movIdx)[0]
        rgbB = cerr_get_cmap(v._scan_display(v.scanNum)[0])(gB)[..., :3]
        rgbM = cerr_get_cmap(movCmap)(gM)[..., :3]

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


# ---------------------------------------------------------------------------#
#  Structure consensus / comparison tool
#  pyCERR counterpart of CERR's structCompare.m: compares several observer
#  segmentations of the same target (STAPLE, Fleiss' kappa, agreement/volume
#  statistics) and can add a consensus structure back into planC.
# ---------------------------------------------------------------------------#
class StructureConsensusDialog(QtWidgets.QDialog):
    """Non-modal dialog to compare observer structures and build a consensus."""

    _METHOD_KEYS = {
        "STAPLE (probabilistic)": "staple",
        "Majority vote (> 50%)": "majority",
        "Agreement fraction >= threshold": "agreement",
        "Union (>= 1 observer)": "union",
        "Intersection (all observers)": "intersection",
    }

    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.planC = viewer.planC
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("Structure Consensus / Comparison")
        self.resize(820, 580)
        self._result = None

        lay = QtWidgets.QHBoxLayout(self)

        # ----- left: structure selection + options ------------------------ #
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel(
            "Observer structures (select 2+ on the same scan):"))
        self.strList = QtWidgets.QListWidget()
        self._populate_structs()
        left.addWidget(self.strList, 1)

        # Statistics depend only on the selected structures, not on the
        # consensus method, so this sits above the method options.
        self.btnCompare = QtWidgets.QPushButton("Compute statistics")
        self.btnCompare.clicked.connect(self.compute_stats)
        left.addWidget(self.btnCompare)

        methodBox = QtWidgets.QGroupBox("Consensus method")
        ml = QtWidgets.QVBoxLayout(methodBox)
        self.methodBtns = {}
        self.methodGrp = QtWidgets.QButtonGroup(self)
        for label in self._METHOD_KEYS:
            b = QtWidgets.QRadioButton(label)
            self.methodGrp.addButton(b)
            ml.addWidget(b)
            self.methodBtns[label] = b
            b.toggled.connect(self._update_threshold_state)
        next(iter(self.methodBtns.values())).setChecked(True)   # STAPLE

        thrRow = QtWidgets.QHBoxLayout()
        thrRow.addWidget(QtWidgets.QLabel("Threshold:"))
        self.thrSpin = QtWidgets.QDoubleSpinBox()
        self.thrSpin.setRange(0.0, 1.0)
        self.thrSpin.setSingleStep(0.05)
        self.thrSpin.setValue(0.5)
        thrRow.addWidget(self.thrSpin)
        thrRow.addStretch(1)
        ml.addLayout(thrRow)
        left.addWidget(methodBox)

        nameRow = QtWidgets.QHBoxLayout()
        nameRow.addWidget(QtWidgets.QLabel("New structure name:"))
        self.nameEdit = QtWidgets.QLineEdit()
        self.nameEdit.setPlaceholderText("(auto)")
        nameRow.addWidget(self.nameEdit, 1)
        left.addLayout(nameRow)

        self.btnCreate = QtWidgets.QPushButton("Create consensus structure")
        self.btnCreate.clicked.connect(self.create_structure)
        left.addWidget(self.btnCreate)

        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(340)
        lay.addWidget(lw)

        # ----- right: results text ---------------------------------------- #
        self.report = QtWidgets.QTextEdit()
        self.report.setReadOnly(True)
        self.report.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        mono = QtGui.QFont("Courier New")
        mono.setStyleHint(QtGui.QFont.Monospace)
        self.report.setFont(mono)
        lay.addWidget(self.report, 1)

        self._update_threshold_state()

    # ------------------------------------------------------------------ ui --
    def _populate_structs(self):
        self.strList.clear()
        for i, st in enumerate(self.planC.structure):
            try:
                scanNum = scn.getScanNumFromUID(st.assocScanUID, self.planC)
            except Exception:  # noqa: BLE001
                scanNum = "?"
            it = QtWidgets.QListWidgetItem(
                f"{i}: {st.structureName}  [scan {scanNum}]")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            it.setData(Qt.UserRole, i)
            self.strList.addItem(it)

    def _selected_structs(self):
        out = []
        for i in range(self.strList.count()):
            it = self.strList.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _current_method(self):
        for label, b in self.methodBtns.items():
            if b.isChecked():
                return self._METHOD_KEYS[label]
        return "staple"

    def _update_threshold_state(self, *_):
        # Selecting the default radio button can fire this before the spin box
        # exists (it is created after the method buttons).
        if not hasattr(self, "thrSpin"):
            return
        self.thrSpin.setEnabled(self._current_method() in ("staple",
                                                           "agreement"))

    # -------------------------------------------------------------- actions --
    def compute_stats(self):
        from cerr.contour import structure_consensus as sc
        structNumV = self._selected_structs()
        if len(structNumV) < 2:
            _show_info(self, "Structure Consensus",
                       "Select at least two structures on the same scan.")
            return
        try:
            self.viewer._busy("Computing structure consensus statistics ...")
            self._result = sc.compareStructures(structNumV, self.planC)
            self.report.setPlainText(sc.summaryText(self._result))
            self.viewer._done("Consensus statistics computed.")
        except Exception as e:  # noqa: BLE001
            self.viewer._done()
            self._result = None
            _show_error(self, "Structure Consensus",
                        f"Could not compute statistics:\n{e}")

    def create_structure(self):
        from cerr.contour import structure_consensus as sc
        structNumV = self._selected_structs()
        if len(structNumV) < 2:
            _show_info(self, "Structure Consensus",
                       "Select at least two structures on the same scan.")
            return
        method = self._current_method()
        threshold = float(self.thrSpin.value())
        name = self.nameEdit.text().strip() or None
        # Reuse cached statistics only when they cover the same structures and
        # already contain a STAPLE map if this method needs one.
        result = self._result
        if result is not None:
            sameStructs = (result.get("structNumV") == list(structNumV))
            hasStaple = result.get("staple3M") is not None
            if not sameStructs or (method == "staple" and not hasStaple):
                result = None
        try:
            self.viewer._busy(f"Building '{method}' consensus structure ...")
            self.planC, strNum = sc.createConsensusStructure(
                structNumV, self.planC, method=method, threshold=threshold,
                structName=name, result=result)
            self.viewer.planC = self.planC
            self.viewer.maskCache.clear()
            self.viewer.after_load(keep_view=True)
            self._populate_structs()
            newName = self.planC.structure[strNum].structureName
            self.viewer._done(f"Added consensus structure '{newName}'.")
            _show_info(self, "Structure Consensus",
                       f"Added structure {strNum}: '{newName}'.")
        except Exception as e:  # noqa: BLE001
            self.viewer._done()
            _show_error(self, "Structure Consensus",
                        f"Could not create consensus structure:\n{e}")
