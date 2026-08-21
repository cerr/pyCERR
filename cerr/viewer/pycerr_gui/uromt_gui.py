"""urOMT (fluid-transport) run and result-view dialogs."""
import json
from cerr.viewer.pycerr_gui.common import *  # noqa: F401,F403

class _UROMTWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(float, str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, planC, scanNumV, structNum, settingsFile, timeSel=None,
                 settings=None, parent=None):
        super().__init__(parent)
        self.planC = planC
        self.scanNumV = scanNumV
        self.structNum = structNum
        self.settingsFile = settingsFile
        self.timeSel = timeSel        # optional (first, jump, last) override
        self.settings = settings      # edited settings dict, overrides the file

    def run(self):
        try:
            from cerr.uromt import buildConfig, prepareData
            from cerr.uromt.config import UROMTConfig
            from cerr.uromt.solver import solveUROMT
            from cerr.uromt.analyze import runEULA, runGLAD
            from cerr.dataclasses.uromt import buildFromConfig, saveUROMTToPlan
            # An edited settings dict (from the settings editor, or carried over
            # from a previously stored run) is used as-is; otherwise the JSON
            # file is loaded.
            if self.settings:
                cfg = UROMTConfig(dict(self.settings), self.scanNumV,
                                  self.structNum)
                cfg.settingsFile = self.settingsFile   # where they came from
            else:
                cfg = buildConfig(self.scanNumV, self.structNum,
                                  self.settingsFile)
            if self.timeSel is not None:
                first, jump, last = self.timeSel
                cfg.settings["time"] = {"first_time": first, "time_jump": jump,
                                        "last_time": last}
            cfg = prepareData(cfg, self.planC)
            res = solveUROMT(
                cfg, statusCallback=lambda f, m: self.progress.emit(f, m))
            self.progress.emit(0.98, "Eulerian / Lagrangian post-processing ...")
            res["Eul"] = runEULA(res)
            res["Lag"] = runGLAD(res)
            obj = buildFromConfig(cfg, res, res["Eul"], res["Lag"])
            idx = saveUROMTToPlan(self.planC, obj)   # store in planC.urOMT
            self.done.emit(idx)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class UROMTSettingsDialog(QtWidgets.QDialog):
    """Editor for the urOMT model settings (the JSON that drives the solver).

    Shows every setting as an editable row, so a run can be set up without
    hand-editing JSON. Values are read back with ``json.loads`` so numbers stay
    numbers and lists/dicts stay structured; anything that is not valid JSON is
    kept as the plain string that was typed (that is how string settings like
    ``smooth_method`` are entered).

    The time-point selection is deliberately NOT listed: it is driven by the
    ``Time points`` row of the main dialog, which overwrites ``settings['time']``
    for the run.
    """

    _HIDDEN = ("time",)

    def __init__(self, parent, settings, path=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("urOMT model settings")
        self.resize(460, 560)
        self._path = path
        self._other = {k: v for k, v in settings.items() if k in self._HIDDEN}

        lay = QtWidgets.QVBoxLayout(self)
        src = path or "(settings of the selected run)"
        self.srcLabel = QtWidgets.QLabel("Source: %s" % src)
        self.srcLabel.setWordWrap(True)
        lay.addWidget(self.srcLabel)
        lay.addWidget(QtWidgets.QLabel(
            "Time points are set in the main urOMT dialog, not here."))

        keys = [k for k in settings if k not in self._HIDDEN]
        self.table = QtWidgets.QTableWidget(len(keys), 2, self)
        self.table.setHorizontalHeaderLabels(["Setting", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, k in enumerate(keys):
            item = QtWidgets.QTableWidgetItem(str(k))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)   # keys are fixed
            self.table.setItem(row, 0, item)
            self.table.setItem(row, 1,
                               QtWidgets.QTableWidgetItem(self._fmt(
                                   settings[k])))
        self.table.resizeColumnsToContents()
        lay.addWidget(self.table, 1)

        btn = QtWidgets.QHBoxLayout()
        defBtn = QtWidgets.QPushButton("Restore defaults")
        defBtn.setToolTip("Reload the values from the bundled default "
                          "settings JSON.")
        defBtn.clicked.connect(self._restoreDefaults)
        saveBtn = QtWidgets.QPushButton("Save to JSON...")
        saveBtn.setToolTip("Write these settings to a JSON file (the current "
                           "file, or a new one).")
        saveBtn.clicked.connect(self._saveAs)
        okBtn = QtWidgets.QPushButton("OK")
        okBtn.setDefault(True)
        okBtn.setToolTip("Use these settings for the next Run (without "
                         "writing a file).")
        okBtn.clicked.connect(self._accept)
        cancelBtn = QtWidgets.QPushButton("Cancel")
        cancelBtn.clicked.connect(self.reject)
        btn.addWidget(defBtn)
        btn.addWidget(saveBtn)
        btn.addStretch(1)
        btn.addWidget(okBtn)
        btn.addWidget(cancelBtn)
        lay.addLayout(btn)

        self.settings = dict(settings)      # result, replaced on accept
        self.savedPath = None               # set when the user writes a file

    @staticmethod
    def _fmt(v):
        """Cell text for a value: plain for scalars, JSON for anything else."""
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float, bool)) or v is None:
            return json.dumps(v)
        return json.dumps(v)

    @staticmethod
    def _parse(text):
        """Cell text -> value. Valid JSON keeps its type; anything else is the
        string as typed, which is how word settings (``smooth_method``,
        ``solver``, ``yes``/``no`` flags) are entered."""
        t = text.strip()
        try:
            return json.loads(t)
        except (ValueError, TypeError):
            return text

    def _collect(self):
        out = dict(self._other)
        for row in range(self.table.rowCount()):
            k = self.table.item(row, 0).text()
            cell = self.table.item(row, 1)
            out[k] = self._parse(cell.text() if cell is not None else "")
        return out

    def _restoreDefaults(self):
        from cerr.uromt.config import _DEFAULT_SETTINGS, loadModelSettings
        try:
            d = loadModelSettings(None)
        except Exception as e:  # noqa: BLE001
            _show_error(self, "urOMT settings", str(e))
            return
        for row in range(self.table.rowCount()):
            k = self.table.item(row, 0).text()
            if k in d:
                self.table.item(row, 1).setText(self._fmt(d[k]))
        self.srcLabel.setText("Source: %s (defaults)" % _DEFAULT_SETTINGS)
        self._path = _DEFAULT_SETTINGS

    def _saveAs(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save urOMT model settings", self._path or "",
            filter="JSON (*.json)")
        if not f:
            return
        try:
            with open(f, "w") as fh:
                json.dump(self._collect(), fh, indent=2, sort_keys=True)
        except Exception as e:  # noqa: BLE001
            _show_error(self, "urOMT settings", str(e))
            return
        self._path = f
        self.savedPath = f
        self.srcLabel.setText("Source: %s" % f)

    def _accept(self):
        self.settings = self._collect()
        self.accept()


class UROMTDialog(QtWidgets.QDialog):
    """Non-modal urOMT launcher / result control panel: pick the ROI structure
    and a model-settings JSON, run on all scans (as ordered time points), and
    overlay any stored run (``planC.urOMT``) on the main scan/segmentation
    views. Each run is stored on the plan container as ``planC.urOMT``."""

    #: Colour-channel quantities, shared by the vector and pathline overlays.
    _COLOR_QUANTITIES = [("speed |v|", "speed"),
                         ("eff. speed |v_eff|", "effSpeed"),
                         ("Peclet", "peclet"), ("rate r", "rate"),
                         ("density rho", "rho"), ("|flux|", "flux")]

    _OVERLAY_VIEWS = [("Eulerian speed", "speed"), ("Eulerian rate", "rate"),
                      ("Eulerian Peclet", "peclet"), ("Eulerian flux", "fluxmag"),
                      ("Surface flux (in/out)", "surfflux"),
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

        # Existing stored runs FIRST: the common case is looking at a run that
        # already exists, and selecting one repopulates the setup below it (ROI,
        # time points, model settings) with what that run actually used.
        self.runsCombo = QtWidgets.QComboBox()
        self.runsCombo.setToolTip(
            "Select a previously computed urOMT run stored on planC.urOMT to "
            "visualize. The ROI, time points and model settings below are "
            "repopulated from the selected run.")
        self.runsCombo.currentIndexChanged.connect(self._onRunSelected)
        form.addRow("Existing runs:", self.runsCombo)

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
        self.settingsEdit.setToolTip(
            "Model-settings JSON used for the next Run. Selecting an existing "
            "run shows the settings that run was computed with.")
        self.settingsEdit.textChanged.connect(self._onSettingsPathTyped)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        editBtn = QtWidgets.QPushButton("Edit...")
        editBtn.setToolTip(
            "Open the model settings in an editor: change any option, apply it "
            "to the next Run, and optionally save it back to JSON (either over "
            "the current file or as a new one).")
        editBtn.clicked.connect(self._editSettings)
        srow = QtWidgets.QHBoxLayout()
        srow.addWidget(self.settingsEdit, 1)
        srow.addWidget(browse)
        srow.addWidget(editBtn)
        form.addRow("Model settings:", srow)
        # In-memory settings: None = load the JSON path above at Run time; a
        # dict = edited (or taken from a selected run) and used as-is.
        self._settings = None
        # Last real settings FILE seen, used to open the browse/save dialogs
        # where the settings live rather than in the working directory.
        self._settingsPath = _DEFAULT_SETTINGS

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
        self.overlayCombo.setToolTip(
            "What to draw on the scan. 'Eulerian flux' is the MAGNITUDE of the "
            "flux density (unsigned - it says how fast tracer moves, not "
            "where it goes); 'Surface flux (in/out)' integrates the outward "
            "normal flux over the ROI boundary, colouring where tracer LEAVES "
            "(warm) and ENTERS (cool) and reporting the in / out / net totals "
            "for the displayed time interval.")
        self.overlayCombo.currentIndexChanged.connect(self._onOverlayChanged)
        self.alphaSpin = QtWidgets.QDoubleSpinBox()
        self.alphaSpin.setRange(0.05, 1.0)
        self.alphaSpin.setSingleStep(0.05)
        self.alphaSpin.setValue(0.6)
        self.alphaSpin.valueChanged.connect(self._onOverlayChanged)
        # overlay density, shared: one arrow / one pathline seed every N voxels
        self.densitySpin = QtWidgets.QSpinBox()
        self.densitySpin.setRange(1, 20)
        self.densitySpin.setValue(2)
        self.densitySpin.setToolTip(
            "Overlay density for BOTH vectors and pathlines: keep one arrow / "
            "one pathline seed every N voxels (1 = one per voxel). In the 2-D "
            "slice views it thins IN-PLANE, in 3-D in all three directions "
            "(so N=2 is one eighth of the ROI). Display only - nothing is "
            "recomputed. Default 2 keeps dense ROIs legible.")
        self.densitySpin.valueChanged.connect(self._onOverlayChanged)
        # display-only length scale for arrows AND pathlines: urOMT velocity is
        # barely constrained where density is low, so a few excursions can span
        # the whole FOV and swamp the picture.
        self.lengthSpin = QtWidgets.QDoubleSpinBox()
        self.lengthSpin.setRange(0.05, 100.0)
        self.lengthSpin.setSingleStep(0.25)
        self.lengthSpin.setValue(1.0)
        self.lengthSpin.setDecimals(2)
        self.lengthSpin.setToolTip(
            "Length scale for the vector arrows AND the pathlines (display "
            "only; the stored result is unchanged). 1x is the true extent - "
            "real transport is often well under a voxel, so raise this to see "
            "it, and remember a scaled path ends where the particle did NOT "
            "go (the factor is shown in the colorbar label). Pathlines scale "
            "about their seed, keeping their shape.")
        self.lengthSpin.valueChanged.connect(self._onOverlayChanged)
        # stroke weight for BOTH the vector arrows and the pathlines
        self.lineWidthSpin = QtWidgets.QDoubleSpinBox()
        self.lineWidthSpin.setRange(0.2, 6.0)
        self.lineWidthSpin.setSingleStep(0.2)
        self.lineWidthSpin.setValue(2.0)          # readable default weight
        self.lineWidthSpin.setDecimals(1)
        self.lineWidthSpin.setToolTip(
            "Line thickness for the vector arrows and the pathlines, in 2-D "
            "and 3-D (2.0 = the default). Thicken when arrows are sparse or "
            "the display is zoomed out; thin them when dense fields overlap.")
        self.lineWidthSpin.valueChanged.connect(self._onOverlayChanged)
        # WHICH quantity drives the colour of the vectors AND the pathlines.
        # Every one of them is sampled once by Part 4 (pathlines) / held in the
        # per-interval Eulerian maps (vectors), so switching is a look-up.
        self.colorByCombo = QtWidgets.QComboBox()
        for label, key in self._COLOR_QUANTITIES:
            self.colorByCombo.addItem(label, key)
        self.colorByCombo.setToolTip(
            "Quantity used to COLOUR the vectors and the pathlines: speed |v|, "
            "effective (flux) speed |v_eff|, Peclet, rate r, density rho or "
            "|flux|. Vector LENGTH always stays the drawn field's magnitude - "
            "only the colour channel changes. A vector and the pathline "
            "segment through the same voxel show the same value.")
        self.colorByCombo.currentIndexChanged.connect(self._onOverlayChanged)
        # ---- reduction + grow animation, shared by vectors and pathlines -----
        # How the "colour by" quantity is reduced (and, for pathlines,
        # consequently how fast they draw).
        self.pathColorCombo = QtWidgets.QComboBox()
        for label in ("median", "mean", "max", "along path"):
            self.pathColorCombo.addItem(label)
        self.pathColorCombo.setCurrentText("median")
        self.pathColorCombo.setToolTip(
            "How the 'colour by' quantity is REDUCED, for pathlines and "
            "vectors alike: median / mean / max over a path's vertices (one "
            "colour per path) or over the run's time intervals at a vector's "
            "voxel; 'along path' is the un-reduced value - per vertex along "
            "the path, and the displayed interval for a vector. The statistics "
            "draw ~2x faster because a path is then a single polyline rather "
            "than one matplotlib object per segment; 'along path' is the one "
            "that shows a path accelerating.")
        self.pathColorCombo.currentIndexChanged.connect(self._onOverlayChanged)

        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(self.overlayCombo, 1)
        orow.addWidget(QtWidgets.QLabel("colour by"))
        orow.addWidget(self.colorByCombo, 1)
        orow.addWidget(QtWidgets.QLabel("reduce"))
        orow.addWidget(self.pathColorCombo, 1)
        form.addRow("Overlay:", orow)

        # One animation control with a view-dependent meaning, kept common by
        # animating whatever the drawn overlay actually varies over: pathlines
        # carry their own time axis (every path has the same vertex count, so a
        # vertex fraction IS a time fraction), while vectors and the Eulerian
        # maps are per-INTERVAL fields - there the slider scrubs the timepoint
        # and Play loops through them.
        self.growSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.growSlider.setRange(0, 100)
        self.growSlider.setValue(100)
        self.growSlider.valueChanged.connect(self._onGrowChanged)
        self.playBtn = QtWidgets.QPushButton("Play")
        self.playBtn.setCheckable(True)
        self.playBtn.toggled.connect(self._onPlayToggled)
        # SINGLE-SHOT and restarted after each redraw finishes. A repeating
        # timer keeps firing while a slow frame is still drawing, so the events
        # queue up and the dialog stops responding to Stop; self-pacing lets the
        # animation run at whatever rate the redraw can sustain.
        self._growTimer = QtCore.QTimer(self)
        self._growTimer.setSingleShot(True)
        self._growTimer.timeout.connect(self._onGrowTick)
        # Display controls, all of them COMMON to the vector and pathline
        # overlays: one "every N" governs arrow and seed density alike (in-plane
        # in 2-D, all three directions in 3-D), one "length x" scales arrow
        # length and pathline reach, one opacity and one stroke weight.
        drow = QtWidgets.QHBoxLayout()
        drow.addWidget(QtWidgets.QLabel("opacity"))
        drow.addWidget(self.alphaSpin)
        drow.addWidget(QtWidgets.QLabel("every N"))
        drow.addWidget(self.densitySpin)
        drow.addWidget(QtWidgets.QLabel("length x"))
        drow.addWidget(self.lengthSpin)
        drow.addWidget(QtWidgets.QLabel("line w"))
        drow.addWidget(self.lineWidthSpin)
        form.addRow("Display:", drow)

        prow = QtWidgets.QHBoxLayout()
        self.growLabel = QtWidgets.QLabel("grow")
        prow.addWidget(self.growLabel)
        prow.addWidget(self.growSlider, 1)
        prow.addWidget(self.playBtn)
        prow.addWidget(QtWidgets.QLabel("3-D opacity"))
        # opacity of the urOMT overlay in the 3-D views (moved here from the
        # main panel); drives PyCerrViewer.plane3dOpacity
        self.plane3dSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.plane3dSlider.setRange(0, 100)
        self.plane3dSlider.setValue(int(round(self.viewer.plane3dOpacity * 100)))
        self.plane3dSlider.setToolTip(
            "Transparency of the urOMT result overlay in the 3-D views.")
        self.plane3dSlider.valueChanged.connect(self.viewer.on_plane_opacity)
        prow.addWidget(self.plane3dSlider, 1)
        form.addRow("Animate:", prow)
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
        # Not the default button: the common action is displaying an existing
        # run, and Enter should not launch a long solve.
        self.runBtn.setAutoDefault(False)
        self.runBtn.clicked.connect(self._run)
        self.showBtn = QtWidgets.QPushButton("Show on scan")
        self.showBtn.setDefault(True)
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
        self.saveBtn.setToolTip(
            "Save the selected run's Eulerian maps (speed, effSpeed, rate, "
            "Peclet, |flux|, and the signed surface flux) as individual 3-D "
            "NIfTI files per metric per time interval, aligned to the scan, "
            "into a chosen folder - plus a CSV of the influx / outflux / net "
            "totals through the ROI surface per interval.")
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
        self._syncViewControls()    # colour-by / animation follow the view
        self._populateRuns()        # list any runs already on planC.urOMT
        self._populateTimepoints()  # timepoint -> scan map for the slider

    def _browse(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "urOMT model settings", self._settingsPathOrDefault(),
            "JSON (*.json)")
        if f:
            self._settingsPath = f
            self.settingsEdit.setText(f)      # -> _onSettingsPathTyped

    def _settingsPathOrDefault(self):
        """A real file for the settings dialogs to open at.

        Prefers the file this run's settings came from (a stored run carries it
        in ``UROMTSetup['settingsFile']``), then whatever is typed in the path
        field, and only then the bundled default - never the process working
        directory, which is where the dialogs used to land.
        """
        for cand in (getattr(self, "_settingsPath", None),
                     self.settingsEdit.text().strip()):
            if cand and os.path.isfile(cand):
                return cand
        from cerr.uromt.config import _DEFAULT_SETTINGS
        return _DEFAULT_SETTINGS

    def _onSettingsPathTyped(self):
        """A path typed or browsed to REPLACES any in-memory edits: the file is
        then the source of truth for the next Run."""
        self._settings = None
        txt = self.settingsEdit.text().strip()
        if txt and os.path.isfile(txt):
            self._settingsPath = txt

    def _currentSettings(self):
        """The settings the next Run would use: the edited/loaded dict when
        there is one, else the JSON file in the path field."""
        if self._settings:
            return dict(self._settings), None
        from cerr.uromt.config import loadModelSettings, _DEFAULT_SETTINGS
        path = self.settingsEdit.text().strip() or _DEFAULT_SETTINGS
        return loadModelSettings(path), path

    def _editSettings(self):
        """Open the model settings in an editor; OK applies them to the next
        Run, and the editor can also write them back to JSON."""
        try:
            settings, path = self._currentSettings()
        except Exception as e:  # noqa: BLE001
            _show_error(self, "urOMT settings", str(e))
            return
        dlg = UROMTSettingsDialog(self, settings,
                                  path or self._settingsPathOrDefault())
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        self._settings = dlg.settings
        if dlg.savedPath:                  # saved to a file -> point at it
            self._setSettingsPath(dlg.savedPath)
            self._settings = None          # the file now holds the edits
            self.progress.setText("Settings saved to %s" % dlg.savedPath)
        else:
            self._setSettingsPath("%s (edited)" % (path or "run settings"))
            self.progress.setText("Settings edited - they apply to the next "
                                  "Run (not saved to file).")

    def _setSettingsPath(self, text):
        """Set the settings path field without clearing the in-memory dict."""
        self.settingsEdit.blockSignals(True)
        self.settingsEdit.setText(text)
        self.settingsEdit.blockSignals(False)

    def _runSummary(self, timeSel):
        """One-line-per-item description of the run about to start."""
        first, jump, last = timeSel
        nScans = len(self.viewer.planC.scan)
        nFrames = len(range(max(1, first), min(nScans, last) + 1, max(1, jump)))
        src = (self.settingsEdit.text().strip() or "(default settings)"
               if self._settings is None else
               "%s (edited - not saved to file)" % self._settingsPathOrDefault())
        return ("ROI structure: %s\n"
                "Time points: %d of %d scans (%d:%d:%d) - %d interval(s)\n"
                "Model settings: %s"
                % (self.structCombo.currentText(), nFrames, nScans,
                   first, jump, last, max(0, nFrames - 1), src))

    def _run(self):
        from cerr.mri_metrics.dce_mri import getScanOrder
        structNum = self.structCombo.currentData()
        timeSel = (self.firstSpin.value(), self.jumpSpin.value(),
                   self.lastSpin.value())
        # urOMT is a long optimization (minutes to hours at full resolution) and
        # cannot be interrupted once the worker starts, so confirm the inputs
        # first - they are easy to leave on a previous run's values now that
        # selecting a run repopulates them.
        ok = QtWidgets.QMessageBox.question(
            self, "Run urOMT?",
            "Start the urOMT calculation with these inputs?\n\n%s\n\n"
            "This can take a long time and cannot be cancelled once started."
            % self._runSummary(timeSel),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if ok != QtWidgets.QMessageBox.Yes:
            self.progress.setText("Run cancelled.")
            return
        # order scans by acquisition time (planC scan order may not be temporal)
        scanNumV = getScanOrder(self.viewer.planC)
        self.runBtn.setEnabled(False)
        self.progress.setText("Starting urOMT ...")
        # With edited / run-derived settings the path field holds a label, not
        # a file, so pass the resolved file purely as provenance; otherwise
        # pass what was typed, so a bad path fails loudly instead of silently
        # falling back to the defaults.
        srcFile = (self._settingsPathOrDefault() if self._settings
                   else (self.settingsEdit.text().strip() or None))
        self.worker = _UROMTWorker(self.viewer.planC, scanNumV, structNum,
                                   srcFile,
                                   timeSel=timeSel,
                                   settings=self._settings,
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
        if has:                             # the dialog's default action
            self.showBtn.setDefault(True)
            self.showBtn.setFocus()
        self._loadSetupFromRun()            # ROI / time points / settings
        self._populateTimepoints()          # timepoints follow the selected run
        self._syncViewControls()            # ... and so does the time slider
        if self._overlayShown and has:      # switch overlay to the new run
            self._showResults()

    def _loadSetupFromRun(self):
        """Repopulate the run setup (ROI structure, time points, model
        settings) from the selected stored run, so what is on screen is what
        that run was computed with - and so a variation on it can be launched
        by changing one field and clicking Run."""
        run = self._selectedRun()
        if run is None:
            return
        setup = getattr(run, "UROMTSetup", {}) or {}
        structNum = setup.get("structNum")
        if structNum is not None:
            for i in range(self.structCombo.count()):
                if self.structCombo.itemData(i) == structNum:
                    self.structCombo.blockSignals(True)
                    self.structCombo.setCurrentIndex(i)
                    self.structCombo.blockSignals(False)
                    break
        # Point the settings dialogs at the JSON this run was computed with,
        # when it is still on disk.
        sf = setup.get("settingsFile")
        if sf and os.path.isfile(sf):
            self._settingsPath = sf
        settings = setup.get("settings") or {}
        if not settings:
            return
        t = settings.get("time") or {}
        nScans = len(self.viewer.planC.scan)
        for spin, key, dflt in ((self.firstSpin, "first_time", 1),
                                (self.jumpSpin, "time_jump", 1),
                                (self.lastSpin, "last_time", nScans)):
            try:
                v = int(t.get(key, dflt) if t.get(key) is not None else dflt)
            except (TypeError, ValueError):
                continue
            spin.blockSignals(True)
            spin.setValue(max(spin.minimum(), min(spin.maximum(), v)))
            spin.blockSignals(False)
        # The run's own settings become the ones a new Run would use; the path
        # field says so, since they came from the stored run and not a file.
        self._settings = dict(settings)
        sfTxt = sf if sf else "(settings of run %s)" % getattr(
            run, "UROMTUID", "?")
        self._setSettingsPath(sfTxt)

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
        self._syncViewControls()
        if self._overlayShown:              # live-update the active overlay
            self._showResults()

    def _currentView(self):
        return self._OVERLAY_VIEWS[self.overlayCombo.currentIndex()][1]

    def _isPathlineView(self):
        return self._currentView() == "pathlines"

    def _syncViewControls(self):
        """Point the view-dependent controls at what the drawn overlay varies.

        The colour-by quantity and its reduction apply to the geometry overlays
        (vectors, pathlines); an Eulerian colourwash IS the map it shows, so the
        pair is greyed out there rather than silently ignored.

        The animation slider means 'grow' for pathlines (a leading fraction of
        each path, which is a time fraction) and 'timepoint' for everything
        else, since vectors and maps are per-interval fields with no growth of
        their own - Play then loops through the timepoints.
        """
        view = self._currentView()
        on = view in ("velocity", "flux", "pathlines")
        self.colorByCombo.setEnabled(on)
        self.pathColorCombo.setEnabled(on)

        paths = view == "pathlines"
        nIvl = max(0, len(self._tpScanNums) - 1)
        self.growLabel.setText("time")
        # The slider is the RUN TIME AXIS in both modes; only its resolution
        # differs. Pathlines are integrated continuously, so they get
        # `_SUB_STEPS` positions inside each interval and grow smoothly;
        # vectors and maps are one field per interval, so a finer slider would
        # only redraw the same field.
        # Reprogram it only when its meaning or range changes: this method runs
        # on every redraw, and resetting the value here snapped the slider back
        # the instant it was scrubbed.
        mode = (("grow", nIvl * self._SUB_STEPS) if paths
                else ("time", nIvl))
        if mode != getattr(self, "_growMode", None):
            self._growMode = mode
            self.growSlider.blockSignals(True)
            self.growSlider.setRange(0, mode[1])
            # Pathlines default to whole paths (the end of the run), which is
            # what the old 100% grow default showed.
            self.growSlider.setValue(mode[1] if paths
                                     else self.tpSlider.value())
            self.growSlider.blockSignals(False)
            if self._growTimer.isActive():     # switched views mid-playback
                self._growTimer.setInterval(self._GROW_MS if paths
                                            else self._TIMEPOINT_MS)
        self.growSlider.setEnabled(nIvl > 0)
        self.growSlider.setToolTip(
            "Scrub the run's time axis. Every pathline is drawn from its seed "
            "up to this time - the CUMULATIVE trajectory so far, not the leg "
            "travelled during one interval - so paths lengthen as you scrub "
            "and fast ones outrun slow ones. The displayed scan is left on the "
            "timepoint you picked above; only the paths change."
            if paths else
            "Scrub the timepoint: vectors and Eulerian maps are drawn for the "
            "time interval selected here (the same interval as the 'Show "
            "timepoint' slider, which follows along).")
        self.playBtn.setToolTip(
            "Grow the pathlines from their seeds through the run, then repeat."
            if paths else
            "Loop through the timepoints, redrawing the field at each one.")
        self.playBtn.setEnabled(nIvl > 0)
        if not self.playBtn.isEnabled() and self.playBtn.isChecked():
            self.playBtn.setChecked(False)         # -> stops the timer

    #: Slider positions per time interval for the pathline growth animation.
    #: Pathlines are integrated in sub-steps, so growth can be shown between
    #: timepoints; the field overlays have nothing to show in between.
    _SUB_STEPS = 10

    def _growFraction(self):
        """Fraction of the run's time axis to draw pathlines up to.

        ``viz.growPathline`` keeps the leading fraction of every path's
        vertices, and every path carries one vertex per integration sub-step of
        the whole run - so this fraction is a TIME fraction and what is drawn
        is the cumulative trajectory from the seed up to that time, not the leg
        walked during one interval.
        """
        if not self._isPathlineView():
            return 1.0
        hi = self.growSlider.maximum()
        if hi <= 0:
            return 1.0
        return float(np.clip(self.growSlider.value() / float(hi), 0.0, 1.0))

    def _onGrowChanged(self, value):
        """Slider moved: grow the pathlines, or step the field overlay's
        timepoint.

        Growing pathlines does NOT touch the displayed scan - a pathline is a
        whole-run trajectory, so there is no one frame it belongs to, and
        swapping the backdrop mid-animation only makes the growth harder to
        follow. The scan stays on whatever the user picked with 'Show
        timepoint'. Vectors and maps are per-interval fields, so there the
        slider IS the timepoint and the scan follows it.
        """
        if self._isPathlineView():
            self._onOverlayChanged()           # redraw at the new growth
        elif self.tpSlider.value() != value:
            self.tpSlider.setValue(value)      # -> _onTimepoint redraws

    #: Timer intervals: a smooth 25 fps for the pathline growth sweep, and a
    #: slower step for the timepoint loop - each timepoint reloads the
    #: displayed scan and rebuilds the overlay, so stepping it at 25 fps would
    #: queue redraws faster than they complete.
    _GROW_MS = 40
    _TIMEPOINT_MS = 600

    def _onPlayToggled(self, on):
        """Sweep the grow slider (pathlines) or loop the timepoints (vectors
        and Eulerian maps)."""
        if on:
            paths = self._isPathlineView()
            if self.growSlider.value() >= self.growSlider.maximum():
                self.growSlider.setValue(0)     # restart a finished sweep
            self._growTimer.setInterval(self._GROW_MS if paths
                                        else self._TIMEPOINT_MS)
            self._growTimer.start()
            self.playBtn.setText("Stop")
        else:
            self._growTimer.stop()
            self.playBtn.setText("Play")

    def _onGrowTick(self):
        """Advance the time slider one step, wrapping at the end.

        Both modes LOOP: the pathlines regrow from their seeds and the field
        overlays cycle their timepoints, until the user stops it. The redraw
        happens synchronously inside the slider change, so the next tick is
        scheduled only once this frame is on screen.
        """
        hi = self.growSlider.maximum()
        if hi <= 0:                          # nothing to animate
            self.playBtn.setChecked(False)
            return
        v = self.growSlider.value()
        self.growSlider.setValue(0 if v >= hi else v + 1)   # -> redraw
        if self.playBtn.isChecked():
            self._growTimer.start()          # pace off the finished frame

    def _updateColorbar(self, ov):
        """Draw the active overlay's colour scale in the dialog's colorbar
        canvas (called by the viewer after the overlay is (re)built)."""
        import matplotlib
        from matplotlib import colorbar as mcbar, colors as mcolors
        self.cbarFig.clear()
        # The bar shows the COLOUR scale: for a vector overlay coloured by a
        # quantity that is `colorRange`, not `vrange` (which scales the arrow
        # lengths). For the maps and the pathlines the two are the same.
        vr = (ov or {}).get("colorRange") or (ov or {}).get("vrange")
        if not vr or vr[1] is None or vr[1] <= vr[0]:
            self.cbarCanvas.draw_idle()
            return
        lo, hi = vr
        cmName = ("bwr" if (ov.get("view") == "rate" or ov.get("diverging"))
                  else "turbo")
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
        # Mirror onto the animation slider in field mode only, where the two
        # sliders mean the same thing. In pathline mode the animation slider is
        # the growth axis and is independent of which scan is displayed.
        if not self._isPathlineView() and self.growSlider.value() != t:
            self.growSlider.blockSignals(True)      # mirror, don't re-trigger
            self.growSlider.setValue(t)
            self.growSlider.blockSignals(False)
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
            self.viewer.set_uromt_overlay(
                idx, view=view, alpha=self.alphaSpin.value(),
                interval=int(interval), subsample=self.densitySpin.value(),
                grow=self._growFraction(),
                lengthScale=self.lengthSpin.value(),
                lineWidth=self.lineWidthSpin.value(),
                pathColorBy=self.pathColorCombo.currentText()
                .replace(" path", ""),
                colorBy=self.colorByCombo.currentData())
            self._overlayShown = True
            self.clearBtn.setEnabled(True)
            msg = ("Overlay: %s on planC.urOMT[%d] (t=%d)"
                   % (view, idx, int(interval) + 1))
            sf = (getattr(self.viewer, "uromtOverlay", None)
                  or {}).get("surfaceFlux")
            if sf is not None:
                # outward-positive: net > 0 means the ROI is losing tracer
                msg += ("  |  influx %.4g, outflux %.4g, net %.4g (%s)"
                        % (sf["influx"], sf["outflux"], sf["net"],
                           "loss" if sf["net"] > 0 else "gain"))
            self.progress.setText(msg)
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
        self._growTimer.stop()              # never tick into a closed dialog
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
