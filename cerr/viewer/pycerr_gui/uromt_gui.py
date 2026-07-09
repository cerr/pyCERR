"""urOMT (fluid-transport) run and result-view dialogs."""
from cerr.viewer.pycerr_gui.common import *  # noqa: F401,F403

class _UROMTWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, planC, scanNumV, structNum, settingsFile, timeSel=None,
                 preview=False, parent=None):
        super().__init__(parent)
        self.planC = planC
        self.scanNumV = scanNumV
        self.structNum = structNum
        self.settingsFile = settingsFile
        self.timeSel = timeSel        # optional (first, jump, last) override
        self.preview = preview        # fast half-res / fewer-iteration run

    def run(self):
        try:
            from cerr.uromt import buildConfig, prepareData
            from cerr.uromt.solver import runUROMT
            from cerr.uromt.analyze import runEULA, runGLAD
            from cerr.dataclasses.uromt import buildFromConfig, saveUROMTToPlan
            cfg = buildConfig(self.scanNumV, self.structNum, self.settingsFile)
            if self.timeSel is not None:
                first, jump, last = self.timeSel
                cfg.settings["time"] = {"first_time": first, "time_jump": jump,
                                        "last_time": last}
            if self.preview:          # fast interactive run (see UROMTDialog)
                cfg.do_resize = 1
                cfg.size_factor = 0.5
                cfg.maxUiter = min(int(getattr(cfg, "maxUiter", 6)), 4)
            cfg = prepareData(cfg, self.planC)
            res = runUROMT(
                cfg, statusCallback=lambda f, m: self.progress.emit(f, m))
            self.progress.emit(0.98, "Eulerian / Lagrangian post-processing ...")
            res["Eul"] = runEULA(res)
            res["Lag"] = runGLAD(res)
            obj = buildFromConfig(cfg, res, res["Eul"], res["Lag"])
            idx = saveUROMTToPlan(self.planC, obj)   # store in planC.urOMT
            self.done.emit(idx)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class UROMTDialog(QtWidgets.QDialog):
    """Non-modal urOMT launcher / result control panel: pick the ROI structure
    and a model-settings JSON, run on all scans (as ordered time points), and
    overlay any stored run (``planC.urOMT``) on the main scan/segmentation
    views. Each run is stored on the plan container as ``planC.urOMT``."""

    _OVERLAY_VIEWS = [("Eulerian speed", "speed"), ("Eulerian rate", "rate"),
                      ("Eulerian Peclet", "peclet"), ("Eulerian flux", "fluxmag"),
                      ("Velocity vectors", "velocity"),
                      ("Flux vectors", "flux"), ("Pathlines", "pathlines")]

    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.worker = None
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("urOMT - fluid transport (beta)")

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(
            "Runs urOMT on all %d scans as longitudinal time points\n"
            "(they must be co-registered onto one grid)."
            % len(viewer.planC.scan)))

        form = QtWidgets.QFormLayout()
        self.structCombo = QtWidgets.QComboBox()
        for i, st in enumerate(viewer.planC.structure):
            self.structCombo.addItem("%d: %s" % (i, st.structureName), i)
        form.addRow("ROI structure:", self.structCombo)

        # time-point selection (1-based first : jump : last into the scan list)
        nScans = len(viewer.planC.scan)
        self.firstSpin = QtWidgets.QSpinBox()
        self.firstSpin.setRange(1, nScans)
        self.firstSpin.setValue(1)
        self.jumpSpin = QtWidgets.QSpinBox()
        self.jumpSpin.setRange(1, max(1, nScans - 1))
        self.jumpSpin.setValue(1)
        self.lastSpin = QtWidgets.QSpinBox()
        self.lastSpin.setRange(1, nScans)
        self.lastSpin.setValue(nScans)
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(QtWidgets.QLabel("first"))
        trow.addWidget(self.firstSpin)
        trow.addWidget(QtWidgets.QLabel("jump"))
        trow.addWidget(self.jumpSpin)
        trow.addWidget(QtWidgets.QLabel("last"))
        trow.addWidget(self.lastSpin)
        form.addRow("Time points:", trow)

        from cerr.uromt.config import _DEFAULT_SETTINGS
        self.settingsEdit = QtWidgets.QLineEdit(_DEFAULT_SETTINGS)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(self.settingsEdit, 1)
        srow.addWidget(browse)
        form.addRow("Model settings:", srow)

        # preview mode: half-resolution grid (8x fewer voxels) + fewer Gauss-
        # Newton steps for a fast interactive run. niter_pcg is left untouched on
        # purpose - under-solving the CG makes GN steps fail the line search and
        # triggers Levenberg retries that re-solve the CG and run *slower*.
        self.previewCheck = QtWidgets.QCheckBox(
            "Preview (fast: half-res, maxUiter=4)")
        self.previewCheck.setToolTip(
            "Quick, lower-fidelity run for interactive checking: resizes the ROI "
            "to half resolution (8x fewer voxels) and caps Gauss-Newton steps at "
            "4. Uncheck for a full-resolution final run.")
        form.addRow("", self.previewCheck)

        # existing stored runs on planC.urOMT (load a previous calculation)
        self.runsCombo = QtWidgets.QComboBox()
        self.runsCombo.setToolTip("Select a previously computed urOMT run "
                                  "stored on planC.urOMT to visualize.")
        self.runsCombo.currentIndexChanged.connect(self._onRunSelected)
        form.addRow("Existing runs:", self.runsCombo)

        # timepoint -> displayed scan (scans may not be stored in temporal
        # order; this maps the timepoint to the correct scan index by
        # acquisition time and drives the main viewer's scan display)
        self.tpSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.tpSlider.setToolTip("Display the scan acquired at this timepoint "
                                 "(mapped to the correct scan index by "
                                 "acquisition time).")
        self.tpSlider.valueChanged.connect(self._onTimepoint)
        self.tpLabel = QtWidgets.QLabel("")
        tprow = QtWidgets.QHBoxLayout()
        tprow.addWidget(self.tpSlider, 1)
        tprow.addWidget(self.tpLabel)
        form.addRow("Show timepoint:", tprow)

        # which result to overlay on the scan views, and its opacity
        self.overlayCombo = QtWidgets.QComboBox()
        for label, _ in self._OVERLAY_VIEWS:
            self.overlayCombo.addItem(label)
        self.overlayCombo.currentIndexChanged.connect(self._onOverlayChanged)
        self.alphaSpin = QtWidgets.QDoubleSpinBox()
        self.alphaSpin.setRange(0.05, 1.0)
        self.alphaSpin.setSingleStep(0.05)
        self.alphaSpin.setValue(0.6)
        self.alphaSpin.valueChanged.connect(self._onOverlayChanged)
        # vector density: draw one arrow every N voxels (1 = every voxel)
        self.densitySpin = QtWidgets.QSpinBox()
        self.densitySpin.setRange(1, 20)
        self.densitySpin.setValue(1)
        self.densitySpin.setToolTip("Vector overlay density: draw one arrow "
                                    "every N voxels (1 = one per voxel). "
                                    "Increase to declutter dense fields.")
        self.densitySpin.valueChanged.connect(self._onOverlayChanged)
        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(self.overlayCombo, 1)
        orow.addWidget(QtWidgets.QLabel("opacity"))
        orow.addWidget(self.alphaSpin)
        orow.addWidget(QtWidgets.QLabel("vec every"))
        orow.addWidget(self.densitySpin)
        form.addRow("Overlay:", orow)

        # opacity of the urOMT overlay in the 3-D views (moved here from the main
        # panel); drives PyCerrViewer.plane3dOpacity
        self.plane3dSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.plane3dSlider.setRange(0, 100)
        self.plane3dSlider.setValue(int(round(self.viewer.plane3dOpacity * 100)))
        self.plane3dSlider.setToolTip(
            "Transparency of the urOMT result overlay in the 3-D views.")
        self.plane3dSlider.valueChanged.connect(self.viewer.on_plane_opacity)
        form.addRow("3-D overlay opacity:", self.plane3dSlider)
        lay.addLayout(form)

        # colorbar legend for the active overlay (lives here, not on the main
        # viewer slices); updated by PyCerrViewer.set_uromt_overlay
        self.cbarFig = Figure(figsize=(3.2, 0.6))
        self.cbarFig.patch.set_alpha(0.0)
        self.cbarCanvas = FigureCanvas(self.cbarFig)
        self.cbarCanvas.setFixedHeight(58)
        self.cbarCanvas.setToolTip("Colour scale of the displayed urOMT metric.")
        lay.addWidget(self.cbarCanvas)

        self.progress = QtWidgets.QLabel("")
        lay.addWidget(self.progress)

        btnRow = QtWidgets.QHBoxLayout()
        btnRow.addStretch(1)
        self.runBtn = QtWidgets.QPushButton("Run")
        self.runBtn.setDefault(True)
        self.runBtn.clicked.connect(self._run)
        self.showBtn = QtWidgets.QPushButton("Show on scan")
        self.showBtn.setEnabled(False)
        self.showBtn.setToolTip("Overlay the selected run's result on the "
                                "scan / segmentation in the main pyCERR views.")
        self.showBtn.clicked.connect(self._showResults)
        self.clearBtn = QtWidgets.QPushButton("Clear")
        self.clearBtn.setEnabled(False)
        self.clearBtn.setToolTip("Remove the urOMT overlay from the views.")
        self.clearBtn.clicked.connect(self._clearOverlay)
        self.saveBtn = QtWidgets.QPushButton("Save maps (NIfTI)")
        self.saveBtn.setEnabled(False)
        self.saveBtn.setToolTip("Save the selected run's Eulerian maps (speed, "
                                "effSpeed, rate, Peclet, |flux|) as individual "
                                "3-D NIfTI files per metric per time interval, "
                                "aligned to the scan, into a chosen folder.")
        self.saveBtn.clicked.connect(self._saveMapsNii)
        closeBtn = QtWidgets.QPushButton("Close")
        closeBtn.clicked.connect(self.close)
        btnRow.addWidget(self.runBtn)
        btnRow.addWidget(self.showBtn)
        btnRow.addWidget(self.clearBtn)
        btnRow.addWidget(self.saveBtn)
        btnRow.addWidget(closeBtn)
        lay.addLayout(btnRow)

        self._overlayShown = False
        self._tpScanNums = []
        self._populateRuns()        # list any runs already on planC.urOMT
        self._populateTimepoints()  # timepoint -> scan map for the slider

    def _browse(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "urOMT model settings", filter="JSON (*.json)")
        if f:
            self.settingsEdit.setText(f)

    def _run(self):
        from cerr.mri_metrics.dce_mri import getScanOrder
        structNum = self.structCombo.currentData()
        # order scans by acquisition time (planC scan order may not be temporal)
        scanNumV = getScanOrder(self.viewer.planC)
        timeSel = (self.firstSpin.value(), self.jumpSpin.value(),
                   self.lastSpin.value())
        self.runBtn.setEnabled(False)
        self.progress.setText("Starting urOMT ...")
        self.worker = _UROMTWorker(self.viewer.planC, scanNumV, structNum,
                                   self.settingsEdit.text() or None,
                                   timeSel=timeSel,
                                   preview=self.previewCheck.isChecked(),
                                   parent=self)
        self.worker.progress.connect(
            lambda f, m: self.progress.setText("[%3.0f%%] %s" % (100 * f, m)))
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._error)
        self.worker.start()

    @staticmethod
    def _runLabel(i, obj):
        setup = getattr(obj, "UROMTSetup", {}) or {}
        res = getattr(obj, "UROMTResult", {}) or {}
        nIv = len(res.get("u", []))
        return "[%d] %s  (struct %s, %d interval%s)" % (
            i, getattr(obj, "UROMTUID", "?"), setup.get("structNum"),
            nIv, "" if nIv == 1 else "s")

    def _populateRuns(self, select=None):
        """Refresh the existing-runs dropdown from planC.urOMT."""
        self.runsCombo.blockSignals(True)
        self.runsCombo.clear()
        runs = getattr(self.viewer.planC, "urOMT", None) or []
        for i, obj in enumerate(runs):
            self.runsCombo.addItem(self._runLabel(i, obj), i)
        if not runs:
            self.runsCombo.addItem("(no runs yet - click Run)", -1)
        if select is not None:
            for i in range(self.runsCombo.count()):
                if self.runsCombo.itemData(i) == select:
                    self.runsCombo.setCurrentIndex(i)
                    break
        self.runsCombo.blockSignals(False)
        self._onRunSelected()

    def _onRunSelected(self):
        has = self.runsCombo.currentData() not in (None, -1)
        self.showBtn.setEnabled(has)
        self.saveBtn.setEnabled(has)
        self._populateTimepoints()          # timepoints follow the selected run
        if self._overlayShown and has:      # switch overlay to the new run
            self._showResults()

    def _saveMapsNii(self):
        """Save the selected run's Eulerian maps as NIfTI on the scan grid."""
        idx = self.runsCombo.currentData()
        runs = getattr(self.viewer.planC, "urOMT", None) or []
        if idx in (None, -1) or idx >= len(runs):
            return
        res = runs[idx].UROMTResult
        outDir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Save urOMT maps to NIfTI (choose a folder)")
        if not outDir:
            return
        try:
            from cerr.uromt.export import saveEulerianMapsNii
            eul = self.viewer._uromtEulIntervals(idx, res)
            fsn = res.get("frameScanNums") or [0]
            paths = saveEulerianMapsNii(eul, self.viewer.planC, int(fsn[0]),
                                        outDir, prefix="uromt%d" % idx)
            self.progress.setText("Saved %d NIfTI file(s) to %s"
                                  % (len(paths), outDir))
            _show_info(self, "urOMT", "Saved %d map file(s) to\n%s :\n  %s"
                       % (len(paths), outDir,
                          "\n  ".join(os.path.basename(p) for p in paths)))
        except Exception as e:  # noqa: BLE001
            _show_error(self, "urOMT save", str(e))

    def _onOverlayChanged(self):
        if self._overlayShown:              # live-update the active overlay
            self._showResults()

    def _updateColorbar(self, ov):
        """Draw the active overlay's colour scale in the dialog's colorbar
        canvas (called by the viewer after the overlay is (re)built)."""
        import matplotlib
        from matplotlib import colorbar as mcbar, colors as mcolors
        self.cbarFig.clear()
        vr = (ov or {}).get("vrange")
        if not vr or vr[1] is None or vr[1] <= vr[0]:
            self.cbarCanvas.draw_idle()
            return
        lo, hi = vr
        cmName = "bwr" if ov.get("view") == "rate" else "turbo"
        cmObj = (matplotlib.colormaps[cmName]
                 if hasattr(matplotlib, "colormaps")
                 else matplotlib.cm.get_cmap(cmName))
        ax = self.cbarFig.add_axes([0.04, 0.45, 0.92, 0.32])
        cb = mcbar.ColorbarBase(ax, cmap=cmObj,
                                norm=mcolors.Normalize(vmin=lo, vmax=hi),
                                orientation="horizontal")
        cb.set_label(ov.get("label", ov.get("view", "urOMT")), fontsize=8)
        cb.ax.tick_params(labelsize=7)
        self.cbarCanvas.draw_idle()

    def _selectedRun(self):
        idx = self.runsCombo.currentData()
        runs = getattr(self.viewer.planC, "urOMT", None) or []
        if idx in (None, -1) or idx >= len(runs):
            return None
        return runs[idx]

    def _populateTimepoints(self):
        """Build the timepoint -> scan-index map for the slider. Uses the
        selected run's frameScanNums when available (the frames actually used,
        already in temporal order), else all scans ordered by acquisition
        time."""
        run = self._selectedRun()
        if run is not None:
            fsn = ((run.UROMTSetup or {}).get("frameScanNums")
                   or (run.UROMTResult or {}).get("frameScanNums") or [])
            self._tpScanNums = list(fsn)
        if run is None or not self._tpScanNums:
            from cerr.mri_metrics.dce_mri import getScanOrder
            self._tpScanNums = getScanOrder(self.viewer.planC)
        n = len(self._tpScanNums)
        self.tpSlider.blockSignals(True)
        self.tpSlider.setRange(0, max(0, n - 1))
        self.tpSlider.setEnabled(n > 0)
        self.tpSlider.blockSignals(False)
        self._updateTpLabel(self.tpSlider.value())

    def _updateTpLabel(self, t):
        from cerr.uromt.data import scanTimeLabel
        if not self._tpScanNums:
            self.tpLabel.setText("-")
            return
        t = int(np.clip(t, 0, len(self._tpScanNums) - 1))
        s = self._tpScanNums[t]
        self.tpLabel.setText("t %d/%d  scan #%d  %s"
                             % (t + 1, len(self._tpScanNums), s,
                                scanTimeLabel(self.viewer.planC, s)))

    def _onTimepoint(self, t):
        """Slider moved -> display the scan acquired at this timepoint (keeping
        the locators on the structure) and refresh the overlay for this interval."""
        if not self._tpScanNums:
            return
        t = int(np.clip(t, 0, len(self._tpScanNums) - 1))
        scanNum = self._tpScanNums[t]
        self._updateTpLabel(t)
        try:
            self.viewer.set_scan(scanNum, keep_view=True)  # don't recentre
            if self._overlayShown:           # re-render the overlay for this t
                self._showResults(interval=t)
        except Exception as e:  # noqa: BLE001
            _show_error(self, "urOMT timepoint", str(e))

    def _finished(self, idx):
        res = self.viewer.planC.urOMT[idx].UROMTResult
        self.runBtn.setEnabled(True)
        nIv = len(res["u"])
        self.progress.setText("Done: %d interval(s); stored as planC.urOMT[%d]"
                              % (nIv, idx))
        self._populateRuns(select=idx)      # add the new run and select it
        _show_info(self.viewer, "urOMT",
                   "urOMT finished: %d time interval(s) solved on a %s ROI "
                   "grid.\nStored as planC.urOMT[%d]. Pick an 'Overlay' and "
                   "click 'Show on scan' to render it on the scan / "
                   "segmentation in the main views."
                   % (nIv, "x".join(map(str, res["n"])), idx))

    def _showResults(self, interval=None):
        idx = self.runsCombo.currentData()
        if idx in (None, -1):
            return
        view = self._OVERLAY_VIEWS[self.overlayCombo.currentIndex()][1]
        if interval is None:
            interval = self.tpSlider.value()
        try:
            self.viewer.set_uromt_overlay(idx, view=view,
                                          alpha=self.alphaSpin.value(),
                                          interval=int(interval),
                                          subsample=self.densitySpin.value())
            self._overlayShown = True
            self.clearBtn.setEnabled(True)
            self.progress.setText("Overlay: %s on planC.urOMT[%d] (t=%d)"
                                  % (view, idx, int(interval) + 1))
        except Exception as e:  # noqa: BLE001
            _show_error(self, "urOMT overlay", str(e))

    def _clearOverlay(self):
        try:
            self.viewer.clear_uromt_overlay()
        except Exception:  # noqa: BLE001
            pass
        self._overlayShown = False
        self.clearBtn.setEnabled(False)
        self._updateColorbar(None)          # blank the colorbar legend
        self.progress.setText("Overlay cleared.")

    def closeEvent(self, event):
        self._clearOverlay()                # don't leave a stale overlay behind
        if getattr(self.viewer, "_uromtDialog", None) is self:
            self.viewer._uromtDialog = None
        super().closeEvent(event)

    def _error(self, msg):
        self.runBtn.setEnabled(True)
        self.progress.setText("Failed.")
        _show_error(self, "urOMT error", msg)



class UROMTViewDialog(QtWidgets.QDialog):
    """Embedded urOMT viewer for a run stored on ``planC.urOMT``."""

    _VIEWS = [("Eulerian speed", "speed"), ("Eulerian rate", "rate"),
              ("Eulerian Peclet", "peclet"), ("Velocity vectors", "velocity"),
              ("Flux vectors", "flux"), ("Pathlines", "pathlines")]
    _AXES = [("Axis 2 (slc)", 2), ("Axis 0 (row)", 0), ("Axis 1 (col)", 1)]

    def __init__(self, viewer, index):
        super().__init__(viewer)
        self.viewer = viewer
        self.run = viewer.planC.urOMT[index]
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowTitle("urOMT view - planC.urOMT[%d]" % index)
        self.resize(620, 680)

        res = self.run.UROMTResult
        self.n = [int(v) for v in res["n"]]
        self.Eul = self.run.UROMTEulerian or None
        self.Lag = self.run.UROMTLagrangian or None
        if not self.Eul or not self.Lag:           # compute on demand
            from cerr.uromt.analyze import runEULA, runGLAD
            self.Eul = self.Eul or runEULA(res)
            self.Lag = self.Lag or runGLAD(res)
        vol = (self.run.UROMTSetup or {}).get("vol") or []
        self.bg = (np.mean([np.asarray(v, float) for v in vol], axis=0)
                   if vol else None)

        self.fig = Figure(facecolor="black", layout="tight")
        self.canvas = FigureCanvas(self.fig)

        self.viewCombo = QtWidgets.QComboBox()
        for label, _ in self._VIEWS:
            self.viewCombo.addItem(label)
        self.axisCombo = QtWidgets.QComboBox()
        for label, _ in self._AXES:
            self.axisCombo.addItem(label)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.sliceLabel = QtWidgets.QLabel("")
        self.threeDCheck = QtWidgets.QCheckBox("3D")
        self.threeDCheck.setToolTip("Render the whole ROI volume in 3D "
                                    "(pathlines / vectors / scalar cloud).")

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("View:"))
        ctrl.addWidget(self.viewCombo, 1)
        ctrl.addWidget(self.threeDCheck)
        ctrl.addWidget(QtWidgets.QLabel("Plane:"))
        ctrl.addWidget(self.axisCombo)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(QtWidgets.QLabel("Slice:"))
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.sliceLabel)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(ctrl)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(srow)

        self.viewCombo.currentIndexChanged.connect(self._redraw)
        self.axisCombo.currentIndexChanged.connect(self._onAxis)
        self.slider.valueChanged.connect(self._redraw)
        self.threeDCheck.toggled.connect(self._on3d)
        self._onAxis()                              # sets slider range + draws

    def _curAxis(self):
        return self._AXES[self.axisCombo.currentIndex()][1]

    def _on3d(self, is3d):
        # the plane/slice controls only apply to the 2-D slice view
        self.axisCombo.setEnabled(not is3d)
        self.slider.setEnabled(not is3d)
        self._redraw()

    def _onAxis(self):
        axis = self._curAxis()
        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n[axis] - 1)
        self.slider.setValue(self.n[axis] // 2)
        self.slider.blockSignals(False)
        self._redraw()

    def _redraw(self):
        from cerr.uromt.viz import drawUROMTSlice, drawUROMT3D
        view = self._VIEWS[self.viewCombo.currentIndex()][1]
        try:
            if self.threeDCheck.isChecked():
                self.sliceLabel.setText("3D")
                drawUROMT3D(self.fig, self.run.UROMTResult, self.Eul, self.Lag,
                            view=view)
            else:
                axis = self._curAxis()
                k = self.slider.value()
                self.sliceLabel.setText("%d/%d" % (k, self.n[axis] - 1))
                drawUROMTSlice(self.fig, self.run.UROMTResult, self.Eul,
                               self.Lag, view=view, axis=axis, sliceIdx=k,
                               bg=self.bg)
            self.canvas.draw_idle()
        except Exception as e:  # noqa: BLE001
            _show_error(self, "urOMT view", str(e))


# ---------------------------------------------------------------------------#
#  Registration QA tool (cf. the napari QA modes in cerr.viewer:
#  Mirrorscope / Sidebyside / AlternateGrid), plus Toggle.
#  Composites the moving scan (resampled onto the base grid) with the base
#  scan in every 2D view; the split line is draggable with the left button.
# ---------------------------------------------------------------------------#
