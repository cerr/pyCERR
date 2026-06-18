"""imrtp_gui
~~~~~~~~~~~

pyCERR IMRTP GUI - a Qt port of Matlab CERR's ``IMRTP/IMRTPGui.m``.

Create a GUI to set up IM (intensity-modulated) beam/structure problems and
calculate beamlet dose.

Usage:
    >>> from cerr import plan_container as pc
    >>> from cerr.imrtp import imrtp_gui
    >>> planC = pc.loadDcmDir(r'path/to/dicom')
    >>> imrtp_gui.IMRTPGui(planC)          # blocks until window closed

    or, inside an existing Qt event loop (e.g. napari console):
    >>> gui = imrtp_gui.IMRTPGui(planC, block=False)

Panels (mirroring IMRTPGui.m):
    Beams               beam list, BEV check, New / Equispaced / Delete
    Geometry Preview    axial CT at isocenter with draggable gantry lines
    Select Scan         associated-scan picker
    Structures          goal list: isTarg / marg / sampRate / remove
    Beam Parameters     per-beam fields with auto-calculation checkboxes
    IM Parameters       dose-engine parameters (algorithm, DoseTerm, ...)
    VMC Parameters      Monte-Carlo parameters
    IM Dosimetry set    browse / delete / rename stored IM sets
    File                Recompute/Copy/Overwrite/Revert actions, Go/Show/Exit
    Status              status text + progress bar

Original Matlab author: JRA 4/30/04;  JJW 07/05/06;  APA 10/16/06.
This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

import copy
import math
import sys
from datetime import date

import numpy as np

try:
    from qtpy import QtWidgets, QtCore, QtGui            # napari installs qtpy
except ImportError:                                       # pragma: no cover
    from PyQt5 import QtWidgets, QtCore, QtGui

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from cerr.dataclasses import scan as scn
from cerr.dataclasses import structure as structr
from cerr.utils import uid as uid_utils

from . import imrtp_problem as imp
from . import imrtp as imrtp_run


FILE_ACTIONS = ['Recompute & add dosimetry',
                'Recompute & overwrite dosimetry',
                'Copy/Add dosimetry w/o calc.',
                'Overwrite dosimetry w/o calc.',
                'Revert to Original']

STALE_TITLE = 'IMRTP *(beamlets may be stale)'
FRESH_TITLE = 'IMRTP'


def _num(text, fallback=None):
    try:
        return float(text)
    except (TypeError, ValueError):
        return fallback


# ==========================================================================
# Geometry-preview canvas (the "bg" frame in IMRTPGui.m)
# ==========================================================================

class GeometryPreview(FigureCanvasQTAgg):
    """Axial CT thumbnail with a 100 cm gantry circle and draggable beam
    lines; dragging the mouse sets the current beam's gantry angle."""

    angleChanged = QtCore.Signal(float) if hasattr(QtCore, 'Signal') \
        else QtCore.pyqtSignal(float)
    dragFinished = QtCore.Signal() if hasattr(QtCore, 'Signal') \
        else QtCore.pyqtSignal()

    def __init__(self, parent=None):
        fig = Figure(figsize=(3.0, 3.0), facecolor='black')
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor('black')
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self._imHandle = None
        self._beamLines = []
        self._errText = None
        self._dragging = False
        self.mpl_connect('button_press_event', self._onPress)
        self.mpl_connect('motion_notify_event', self._onMotion)
        self.mpl_connect('button_release_event', self._onRelease)

    # -- drawing -----------------------------------------------------------
    def refresh(self, planC, im, currentBeam):
        ax = self.ax
        ax.clear()
        ax.set_facecolor('black')
        ax.set_xticks([]); ax.set_yticks([])
        R = 100.0   # gantry circle radius, cm (matches Matlab 100/gridUnits)

        scanNum = im.assocScanNum(planC)
        ctOK = False
        isoXY = (0.0, 0.0)
        if scanNum is not None and im.beams:
            beam = im.beams[max(0, currentBeam)] if 0 <= currentBeam < \
                len(im.beams) else im.beams[0]
            xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
            isoZ = beam.isocenter.z
            try:
                isoZres = imp.resolveIsocenter(beam, im, planC)
                isoZ = isoZres.z
                isoXY = (isoZres.x, isoZres.y)
            except Exception:
                pass
            if isinstance(isoZ, (int, float)) and \
                    (min(zV) <= isoZ <= max(zV)):
                sliceNum = int(np.argmin(np.abs(np.asarray(zV) - isoZ)))
                sliceM = planC.scan[scanNum].getScanArray()[:, :, sliceNum]
                ax.imshow(sliceM, cmap='gray',
                          extent=[xV[0], xV[-1], yV[-1], yV[0]],
                          origin='upper', zorder=1)
                ctOK = True
            else:
                ax.text(0, 0, 'An isocenterZ value is\noutside the CT '
                        'bounds.', color='white', ha='center', va='center')
        elif scanNum is not None:
            # No beams yet: show the middle slice for orientation.
            xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
            sliceM = planC.scan[scanNum].getScanArray()[:, :, len(zV) // 2]
            ax.imshow(sliceM, cmap='gray',
                      extent=[xV[0], xV[-1], yV[-1], yV[0]],
                      origin='upper', zorder=1)
            ctOK = True

        # Yellow circle of beam positions, centered on the isocenter.
        th = np.linspace(0, 2 * np.pi, 360)
        cx, cy = isoXY if ctOK else (0.0, 0.0)
        ax.plot(cx + np.sin(th) * R, cy + np.cos(th) * R,
                color='yellow', lw=1, zorder=2)
        self._center = (cx, cy)

        # Beam lines: source point plus two opposite rays (175 / 185 deg).
        for i, b in enumerate(im.beams):
            col = 'white' if i == currentBeam else 'blue'
            self._plotBeamLine(b.gantryAngle, col, R)

        pad = R * 1.12
        ax.set_xlim(cx - pad, cx + pad)
        ax.set_ylim(cy - pad, cy + pad)
        ax.set_aspect('equal')
        self.draw_idle()

    def _plotBeamLine(self, angle, color, R):
        cx, cy = self._center
        a = math.radians(angle)
        sx, sy = math.sin(a), math.cos(a)
        o1 = a + math.radians(175)
        o2 = a + math.radians(185)
        xs = [cx + math.sin(o1) * R, cx + sx * R, cx + math.sin(o2) * R]
        ys = [cy + math.cos(o1) * R, cy + sy * R, cy + math.cos(o2) * R]
        self.ax.plot(xs, ys, color=color, lw=1, zorder=3)

    # -- interaction -------------------------------------------------------
    def _angleFromEvent(self, event):
        cx, cy = getattr(self, '_center', (0.0, 0.0))
        dx, dy = event.xdata - cx, event.ydata - cy
        return math.floor(math.degrees(math.atan2(dx, dy)) % 360.0)

    def _onPress(self, event):
        if event.inaxes is self.ax and event.xdata is not None:
            self._dragging = True
            self.angleChanged.emit(self._angleFromEvent(event))

    def _onMotion(self, event):
        if self._dragging and event.inaxes is self.ax \
                and event.xdata is not None:
            self.angleChanged.emit(self._angleFromEvent(event))

    def _onRelease(self, event):
        if self._dragging:
            self._dragging = False
            self.dragFinished.emit()


def beamGeometry(beam, im, planC):
    """Compute a beam's display geometry in world coords (cm, pyCERR virtual):
    ``polylines`` - the divergent field pyramid edges, the iso/exit field
    rectangles and the central axis (used for the 3D view); ``apex`` and
    ``corners`` - the pyramid source point and 4 far field corners (used to
    compute the per-slice beam cross-section in the 2D views).

    Mirrors Matlab CERR's beam's-eye-view geometry: gantry in the axial
    plane (couch 0), source at SAD from the isocenter."""
    try:
        iso = imp.resolveIsocenter(beam, im, planC)
        if any(isinstance(v, str) for v in (iso.x, iso.y, iso.z)):
            return None
        I = np.array([float(iso.x), float(iso.y), float(iso.z)])
    except Exception:
        return None

    sad = float(beam.isodistance) or 100.0
    g = math.radians(beam.gantryAngle)
    # source is at +SAD along (sin g, cos g) from iso (gantry 0 = anterior)
    S = I + np.array([sad * math.sin(g), sad * math.cos(g), 0.0])
    d = I - S
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    d = d / n                                  # beam axis, source -> iso
    u = np.array([math.cos(g), -math.sin(g), 0.0])   # lateral (axial plane)
    v = np.array([0.0, 0.0, 1.0])              # superior-inferior

    # field half-size at iso: from the beamlet grid if available, else 5 cm
    def _half(posV, fallback=5.0):
        if posV is not None and len(np.atleast_1d(posV)):
            return float(np.max(np.abs(posV))) or fallback
        return fallback

    w = _half(getattr(beam, 'xPBPosV', None))
    h = _half(getattr(beam, 'yPBPosV', None))

    exitC = S + d * (2.0 * sad)                # exit plane at 2*SAD
    signs = [(+1, +1), (+1, -1), (-1, -1), (-1, +1)]
    cIso = [I + sx * w * u + sy * h * v for sx, sy in signs]
    cExit = [exitC + sx * 2 * w * u + sy * 2 * h * v for sx, sy in signs]

    polys = []
    for ce in cExit:                           # pyramid edges source -> corner
        polys.append(np.array([S, ce]))
    polys.append(np.array(cExit + [cExit[0]]))      # exit rectangle
    polys.append(np.array(cIso + [cIso[0]]))        # iso field rectangle
    polys.append(np.array([S, exitC]))              # central axis

    # pyramid geometry for the per-slice 2D cross-section
    return {'polylines': polys, 'apex': S, 'corners': np.array(cExit)}


def beamPolylines(beam, im, planC):
    """Backward-compatible: just the polylines of :func:`beamGeometry`."""
    geo = beamGeometry(beam, im, planC)
    return geo['polylines'] if geo else []


# ==========================================================================
# Main GUI
# ==========================================================================

class IMRTPGuiWindow(QtWidgets.QMainWindow):

    def __init__(self, planC, im=None, saveIndex=None, parent=None,
                 viewer=None):
        super().__init__(parent)
        if planC is None or len(planC.scan) == 0:
            raise ValueError('Load a plan into planC before opening IMRTPGui.')
        self.planC = planC
        self.viewer = viewer        # optional PyCerrViewer for dose/beam display
        self.currentBeam = -1
        self.saveIndex = -1 if saveIndex is None else saveIndex

        if im is not None:
            self.im = im
            self.currentBeam = 0 if im.beams else -1
        else:
            self.im = imp.initIMRTProblem(planC)
            # Make engine default match what is actually available;
            # prefer the QIB pencil beam, as in Matlab CERR.
            engines = imrtp_run.availableEngines()
            if 'QIB' in engines:
                self.im.params.algorithm = 'QIB'
            else:
                self.im.params.algorithm = engines[0] if engines else 'QIB'

        self.setWindowTitle(FRESH_TITLE)
        self.resize(1180, 720)
        self._build()
        self._refreshAll()
        if not imrtp_run.availableEngines():
            self.statusbarMsg('No dose engines registered. QIB/VMC '
                              'calculations disabled. See cerr.imrtp docs.')

    def closeEvent(self, event):
        # remove beam overlays from the viewer when IMRTP closes
        if self.viewer is not None:
            try:
                self.viewer.setBeams([])
            except Exception:  # noqa: BLE001
                pass
        super().closeEvent(event)

    # ------------------------------------------------------------------ UI
    def _build(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        grid = QtWidgets.QGridLayout(central)

        # ---- Beams frame ("bl") ----
        blBox = QtWidgets.QGroupBox('Beams')
        blLay = QtWidgets.QVBoxLayout(blBox)
        hdr = QtWidgets.QHBoxLayout()
        hdr.addStretch(1)
        hdr.addWidget(QtWidgets.QLabel("Beam's Eye View"))
        blLay.addLayout(hdr)
        self.beamList = QtWidgets.QTableWidget(0, 3)
        self.beamList.setHorizontalHeaderLabels(['#', 'Description', 'BEV'])
        self.beamList.horizontalHeader().setStretchLastSection(False)
        self.beamList.setColumnWidth(0, 28)
        self.beamList.setColumnWidth(1, 150)
        self.beamList.setColumnWidth(2, 44)
        self.beamList.verticalHeader().setVisible(False)
        self.beamList.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.beamList.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.beamList.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers)
        self.beamList.itemSelectionChanged.connect(self.selectBeam)
        blLay.addWidget(self.beamList)
        btnLay = QtWidgets.QHBoxLayout()
        for txt, fn in (('New', self.newBeam),
                        ('Equispaced', self.newEquispaced),
                        ('Delete', self.delBeam)):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            btnLay.addWidget(b)
        blLay.addLayout(btnLay)
        grid.addWidget(blBox, 0, 0)

        # ---- Geometry preview ("bg") ----
        bgBox = QtWidgets.QGroupBox('Geometry Preview')
        bgLay = QtWidgets.QVBoxLayout(bgBox)
        self.preview = GeometryPreview(bgBox)
        self.preview.angleChanged.connect(self.previewAngleChanged)
        self.preview.dragFinished.connect(self.previewDragFinished)
        bgLay.addWidget(self.preview)
        grid.addWidget(bgBox, 0, 1)

        # ---- Right column: scan + structures ----
        rightCol = QtWidgets.QVBoxLayout()
        ssScanBox = QtWidgets.QGroupBox('Select Scan')
        ssScanLay = QtWidgets.QHBoxLayout(ssScanBox)
        self.scanPop = QtWidgets.QComboBox()
        self.scanPop.activated.connect(self.selScan)
        ssScanLay.addWidget(self.scanPop)
        rightCol.addWidget(ssScanBox)

        ssBox = QtWidgets.QGroupBox('Structures')
        ssLay = QtWidgets.QVBoxLayout(ssBox)
        addLay = QtWidgets.QHBoxLayout()
        addLay.addWidget(QtWidgets.QLabel('Add:'))
        self.addStructPop = QtWidgets.QComboBox()
        self.addStructPop.activated.connect(self.addGoal)
        addLay.addWidget(self.addStructPop, 1)
        ssLay.addLayout(addLay)
        self.goalTable = QtWidgets.QTableWidget(0, 5)
        self.goalTable.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection)
        self.goalTable.setHorizontalHeaderLabels(
            ['Index / Name', 'isTarg', 'marg', 'sampRate', ''])
        self.goalTable.verticalHeader().setVisible(False)
        self.goalTable.setColumnWidth(0, 130)
        self.goalTable.setColumnWidth(1, 46)
        self.goalTable.setColumnWidth(2, 44)
        self.goalTable.setColumnWidth(3, 60)
        self.goalTable.setColumnWidth(4, 28)
        ssLay.addWidget(self.goalTable)
        rightCol.addWidget(ssBox, 1)
        grid.addLayout(rightCol, 0, 2)

        # ---- Beam parameters ("bp") ----
        bpBox = QtWidgets.QGroupBox('Beam Parameters    '
                                    '(checkboxes toggle auto field '
                                    'calculation)')
        bpGrid = QtWidgets.QGridLayout(bpBox)
        self.bpWidgets = {}    # fieldName tuple -> (autoChk|None, editor)
        nFields = len(imp.BEAM_FIELD_NAMES)
        half = (nFields + 1) // 2
        for i, fName in enumerate(imp.BEAM_FIELD_NAMES):
            col = 0 if i < half else 3
            row = i if i < half else i - half
            label = QtWidgets.QLabel(''.join(fName))
            bpGrid.addWidget(label, row, col)
            chk = None
            if not imp.BEAM_FIELD_EDITABLE[i]:
                chk = QtWidgets.QCheckBox()
                chk.setChecked(True)
                chk.setToolTip('Auto-calculate this field')
                chk.stateChanged.connect(self.autoCheckChanged)
                bpGrid.addWidget(chk, row, col + 1)
            choices = imp.BEAM_FIELD_CHOICES.get(fName)
            if choices:
                ed = QtWidgets.QComboBox()
                ed.addItems([str(c) for c in choices])
                ed.activated.connect(
                    lambda _=None, f=fName: self.beamParamChanged(f))
            else:
                ed = QtWidgets.QLineEdit()
                ed.editingFinished.connect(
                    lambda f=fName: self.beamParamChanged(f))
            bpGrid.addWidget(ed, row, col + 2)
            self.bpWidgets[fName] = (chk, ed)
        grid.addWidget(bpBox, 1, 0, 1, 2)

        # ---- IM parameters ("ip") ----
        ipBox = QtWidgets.QGroupBox('IM Parameters')
        ipGrid = QtWidgets.QGridLayout(ipBox)
        self.ipWidgets = {}
        for i, pName in enumerate(imp.PARAM_NAMES):
            ipGrid.addWidget(QtWidgets.QLabel(''.join(pName)), i, 0)
            if pName == ('algorithm',):
                ed = QtWidgets.QComboBox()
                ed.addItems(imrtp_run.availableEngines() or ['QIB'])
                ed.activated.connect(
                    lambda _=None, p=pName: self.imParamChanged(p))
            elif pName in imp.PARAM_CHOICES:
                ed = QtWidgets.QComboBox()
                ed.addItems(imp.PARAM_CHOICES[pName])
                ed.activated.connect(
                    lambda _=None, p=pName: self.imParamChanged(p))
            else:
                ed = QtWidgets.QLineEdit()
                ed.editingFinished.connect(
                    lambda p=pName: self.imParamChanged(p))
            ipGrid.addWidget(ed, i, 1)
            self.ipWidgets[pName] = ed
        grid.addWidget(ipBox, 1, 2)

        # ---- VMC parameters ("mc") column ----
        mcBox = QtWidgets.QGroupBox('VMC Parameters')
        mcGrid = QtWidgets.QGridLayout(mcBox)
        self.mcWidgets = {}
        for i, pName in enumerate(imp.MC_PARAM_NAMES):
            mcGrid.addWidget(QtWidgets.QLabel(''.join(pName)), i, 0)
            if pName in imp.MC_PARAM_CHOICES:
                ed = QtWidgets.QComboBox()
                ed.addItems(imp.MC_PARAM_CHOICES[pName])
                ed.activated.connect(
                    lambda _=None, p=pName: self.mcParamChanged(p))
            else:
                ed = QtWidgets.QLineEdit()
                ed.editingFinished.connect(
                    lambda p=pName: self.mcParamChanged(p))
            mcGrid.addWidget(ed, i, 1)
            self.mcWidgets[pName] = ed
        grid.addWidget(mcBox, 0, 3, 3, 1)
        # The VMC panel is only relevant for the VMC++ engine; visibility
        # is driven by the selected algorithm (see
        # _updateParamPanelVisibility).
        self.ipBox = ipBox
        self.mcBox = mcBox

        # ---- Bottom row: IM browser ("ib"), File ("us"), Status ("wb") ----
        bottom = QtWidgets.QHBoxLayout()

        wbBox = QtWidgets.QGroupBox('Status')
        wbLay = QtWidgets.QVBoxLayout(wbBox)
        self.statusText = QtWidgets.QLabel('')
        self.waitBar = QtWidgets.QProgressBar()
        self.waitBar.setRange(0, 100)
        self.waitBar.setValue(0)
        wbLay.addWidget(self.statusText)
        wbLay.addWidget(self.waitBar)
        bottom.addWidget(wbBox, 2)

        ibBox = QtWidgets.QGroupBox('IM Dosimetry set')
        ibLay = QtWidgets.QGridLayout(ibBox)
        self.browsePop = QtWidgets.QComboBox()
        self.browsePop.activated.connect(self.browseIM)
        ibLay.addWidget(self.browsePop, 0, 0, 1, 2)
        delBtn = QtWidgets.QPushButton('Delete')
        delBtn.clicked.connect(self.deleteIM)
        ibLay.addWidget(delBtn, 0, 2)
        ibLay.addWidget(QtWidgets.QLabel('Rename'), 1, 0)
        self.renameEdit = QtWidgets.QLineEdit()
        self.renameEdit.editingFinished.connect(self.renameIM)
        ibLay.addWidget(self.renameEdit, 1, 1, 1, 2)
        bottom.addWidget(ibBox, 2)

        usBox = QtWidgets.QGroupBox('File')
        usLay = QtWidgets.QGridLayout(usBox)
        self.filePop = QtWidgets.QComboBox()
        self.filePop.addItems(FILE_ACTIONS)
        usLay.addWidget(self.filePop, 0, 0, 1, 2)
        goBtn = QtWidgets.QPushButton('Go')
        goBtn.clicked.connect(self.save)
        usLay.addWidget(goBtn, 0, 2)
        showBtn = QtWidgets.QPushButton('Show')
        showBtn.clicked.connect(self.showDose)
        usLay.addWidget(showBtn, 1, 1)
        exitBtn = QtWidgets.QPushButton('Exit')
        exitBtn.clicked.connect(self.close)
        usLay.addWidget(exitBtn, 1, 2)
        bottom.addWidget(usBox, 2)

        grid.addLayout(bottom, 2, 0, 1, 3)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 3)
        grid.setColumnStretch(3, 2)
        grid.setRowStretch(0, 4)
        grid.setRowStretch(1, 4)
        grid.setRowStretch(2, 1)

    # ------------------------------------------------------------ helpers
    def statusbarMsg(self, msg):
        self.statusText.setText(str(msg))

    def waitbar(self, frac):
        self.waitBar.setValue(int(round(100 * max(0.0, min(1.0, frac)))))
        QtWidgets.QApplication.processEvents()

    def markStale(self):
        self.setWindowTitle(STALE_TITLE)
        self.im.isFresh = False

    def _structsInScan(self):
        """Structure objects (and absolute indices) on the associated scan."""
        scanNum = self.im.assocScanNum(self.planC)
        out = []
        for i, s in enumerate(self.planC.structure):
            try:
                if scn.getScanNumFromUID(s.assocScanUID, self.planC) == scanNum:
                    out.append((i, s))
            except Exception:
                pass
        return out

    # ----------------------------------------------------------- refreshes
    def _refreshAll(self):
        self.refreshScan()
        self.refreshStructs()
        self.refreshBeams()
        self.refreshPreview()
        self.refreshBeamParams()
        self.refreshIMParams()
        self.refreshMCParams()
        self.refreshBrowser()
        self._updateViewerBeams()   # keep BEV overlays in sync with geometry

    def refreshScan(self):
        self.scanPop.blockSignals(True)
        self.scanPop.clear()
        for i, s in enumerate(self.planC.scan):
            sType = getattr(s.scanInfo[0], 'imageType', '') or \
                getattr(s, 'scanType', '')
            self.scanPop.addItem('%d %s' % (i + 1, sType))
        scanNum = self.im.assocScanNum(self.planC)
        if scanNum is not None:
            self.scanPop.setCurrentIndex(scanNum)
        self.scanPop.blockSignals(False)
        # structure pulldown for this scan
        self.addStructPop.blockSignals(True)
        self.addStructPop.clear()
        for _, s in self._structsInScan():
            self.addStructPop.addItem(s.structureName)
        self.addStructPop.setCurrentIndex(-1)
        self.addStructPop.blockSignals(False)

    def refreshStructs(self):
        tbl = self.goalTable
        tbl.blockSignals(True)
        tbl.clearContents()
        tbl.setRowCount(0)
        tbl.setRowCount(len(self.im.goals))
        for i, g in enumerate(self.im.goals):
            try:
                absNum = structr.getStructNumFromUID(g.strUID, self.planC)
                rgb = structr.getColorForStructNum(absNum)
                bg = QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
                idxStr = str(absNum + 1)
            except Exception:
                bg = self.palette().color(QtGui.QPalette.Window)
                idxStr = 'N-A'
            nameItem = QtWidgets.QTableWidgetItem(
                ('%s. %s' % (idxStr, g.structName))[:20])
            nameItem.setBackground(bg)
            lum = 0.299 * bg.red() + 0.587 * bg.green() + 0.114 * bg.blue()
            nameItem.setForeground(QtGui.QColor(
                'black' if lum > 128 else 'white'))
            nameItem.setFlags(QtCore.Qt.ItemIsEnabled)
            tbl.setItem(i, 0, nameItem)
            tbl.setRowHeight(i, 26)

            chk = QtWidgets.QCheckBox()
            chk.setChecked(g.is_target())
            chk.stateChanged.connect(
                lambda _=None, n=i: self.targBoxClicked(n))
            w = QtWidgets.QWidget()
            l = QtWidgets.QHBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.setAlignment(QtCore.Qt.AlignCenter)
            l.addWidget(chk)
            tbl.setCellWidget(i, 1, w)

            margEd = QtWidgets.QLineEdit(str(g.PBMargin))
            margEd.editingFinished.connect(
                lambda n=i, e=margEd: self.pbMarginText(n, e))
            tbl.setCellWidget(i, 2, margEd)

            sampEd = QtWidgets.QLineEdit(str(g.xySampleRate))
            sampEd.editingFinished.connect(
                lambda n=i, e=sampEd: self.strSampleRate(n, e))
            tbl.setCellWidget(i, 3, sampEd)

            delBtn = QtWidgets.QPushButton('-')
            delBtn.setFixedWidth(24)
            delBtn.clicked.connect(lambda _=None, n=i: self.delGoal(n))
            tbl.setCellWidget(i, 4, delBtn)
        tbl.blockSignals(False)

    def refreshBeams(self):
        tbl = self.beamList
        tbl.blockSignals(True)
        tbl.setRowCount(len(self.im.beams))
        for i, b in enumerate(self.im.beams):
            numItem = QtWidgets.QTableWidgetItem('%d.' % (i + 1))
            numItem.setFlags(QtCore.Qt.ItemIsEnabled |
                             QtCore.Qt.ItemIsSelectable)
            tbl.setItem(i, 0, numItem)
            nameItem = QtWidgets.QTableWidgetItem(b.beamDescription)
            nameItem.setFlags(QtCore.Qt.ItemIsEnabled |
                              QtCore.Qt.ItemIsSelectable)
            if i == self.currentBeam:
                nameItem.setBackground(QtGui.QColor('white'))
                nameItem.setForeground(QtGui.QColor('black'))
            else:
                nameItem.setBackground(QtGui.QColor('black'))
                nameItem.setForeground(QtGui.QColor('white'))
            tbl.setItem(i, 1, nameItem)
            if tbl.cellWidget(i, 2) is None:
                chk = QtWidgets.QCheckBox()
                chk.setToolTip("Show beam's eye view in viewer")
                chk.stateChanged.connect(self._updateViewerBeams)
                w = QtWidgets.QWidget()
                l = QtWidgets.QHBoxLayout(w)
                l.setContentsMargins(0, 0, 0, 0)
                l.setAlignment(QtCore.Qt.AlignCenter)
                l.addWidget(chk)
                tbl.setCellWidget(i, 2, w)
        if 0 <= self.currentBeam < len(self.im.beams):
            tbl.selectRow(self.currentBeam)
        tbl.blockSignals(False)

    def _bevChecked(self, i):
        """Is the BEV checkbox ticked for beam row i?"""
        w = self.beamList.cellWidget(i, 2)
        if w is None:
            return False
        chk = w.findChild(QtWidgets.QCheckBox)
        return chk is not None and chk.isChecked()

    def _updateViewerBeams(self, *_):
        """Push the geometry of all BEV-checked beams to the pyCERR viewer."""
        if self.viewer is None:
            if any(self._bevChecked(i) for i in range(len(self.im.beams))):
                self.statusbarMsg("Beam's eye view needs the pyCERR viewer "
                                  "(open IMRTP from the viewer's Tools menu).")
            return
        palette = [(0.20, 0.85, 0.90), (0.95, 0.75, 0.20), (0.40, 0.90, 0.40),
                   (0.95, 0.45, 0.85), (0.55, 0.65, 1.00), (0.95, 0.55, 0.35)]
        beams = []
        for i, b in enumerate(self.im.beams):
            if not self._bevChecked(i):
                continue
            geo = beamGeometry(b, self.im, self.planC)
            if geo:
                geo['color'] = palette[i % len(palette)]
                beams.append(geo)
        try:
            self.viewer.setBeams(beams)
        except Exception as e:  # noqa: BLE001
            self.statusbarMsg('Could not draw beams: %s' % e)

    def refreshPreview(self):
        self.preview.refresh(self.planC, self.im, self.currentBeam)

    def refreshBeamParams(self):
        haveBeam = 0 <= self.currentBeam < len(self.im.beams)
        beam = self.im.beams[self.currentBeam] if haveBeam else None
        for i, fName in enumerate(imp.BEAM_FIELD_NAMES):
            chk, ed = self.bpWidgets[fName]
            ed.blockSignals(True)
            ed.setEnabled(haveBeam)
            if chk is not None:
                chk.setEnabled(haveBeam)
            if not haveBeam:
                if isinstance(ed, QtWidgets.QLineEdit):
                    ed.setText('')
                ed.blockSignals(False)
                continue
            val = self._getBeamField(beam, fName)
            if isinstance(ed, QtWidgets.QComboBox):
                ix = ed.findText(str(val))
                if ix < 0:
                    ed.addItem(str(val))
                    ix = ed.count() - 1
                ed.setCurrentIndex(ix)
            else:
                if isinstance(val, float):
                    val = round(val, 4)
                ed.setText('' if val is None else str(val))
                auto = chk is not None and chk.isChecked()
                ed.setReadOnly(auto)
                ed.setStyleSheet('color: gray;' if auto else '')
            ed.blockSignals(False)

    def refreshIMParams(self):
        p = self.im.params
        vals = {('algorithm',): p.algorithm, ('DoseTerm',): p.DoseTerm,
                ('ScatterMethod',): p.ScatterMethod,
                ('Scatter', 'Threshold'): p.Scatter.Threshold,
                ('Scatter', 'RandomStep'): p.Scatter.RandomStep,
                ('xyDownsampleIndex',): p.xyDownsampleIndex,
                ('numCTSamplePts',): p.numCTSamplePts,
                ('cutoffDistance',): p.cutoffDistance}
        for pName, ed in self.ipWidgets.items():
            ed.blockSignals(True)
            v = vals[pName]
            if isinstance(ed, QtWidgets.QComboBox):
                ix = ed.findText(str(v))
                if ix < 0 and v:
                    ed.addItem(str(v)); ix = ed.count() - 1
                ed.setCurrentIndex(max(ix, 0))
            else:
                ed.setText(str(v))
            ed.blockSignals(False)

    def refreshMCParams(self):
        vmc = self.im.params.VMC
        for pName, ed in self.mcWidgets.items():
            ed.blockSignals(True)
            v = getattr(vmc, pName[0])
            if isinstance(ed, QtWidgets.QComboBox):
                ix = ed.findText(str(v))
                ed.setCurrentIndex(max(ix, 0))
            else:
                ed.setText('' if v is None else str(v))
            ed.blockSignals(False)
        self._updateParamPanelVisibility()

    def _updateParamPanelVisibility(self):
        """Show the VMC Parameters panel only when the VMC++ engine is
        selected; the IM Parameters panel (QIB / pencil-beam settings)
        is always shown and QIB is the default algorithm."""
        alg = str(self.im.params.algorithm or '').upper()
        self.mcBox.setVisible('VMC' in alg)

    def refreshBrowser(self):
        imList = imp.getIMList(self.planC)
        self.browsePop.blockSignals(True)
        self.browsePop.clear()
        for s in imList:
            self.browsePop.addItem(s.name)
        self.browsePop.addItem('IM doseSet %d' % (len(imList) + 1))
        if 0 <= self.saveIndex < len(imList):
            self.browsePop.setCurrentIndex(self.saveIndex)
        else:
            self.browsePop.setCurrentIndex(self.browsePop.count() - 1)
        self.browsePop.blockSignals(False)
        self.renameEdit.setText(self.im.name)

    # --------------------------------------------------------- beam fields
    @staticmethod
    def _getBeamField(beam, fName):
        obj = beam
        for part in fName:
            obj = getattr(obj, part)
        return obj

    @staticmethod
    def _setBeamField(beam, fName, value):
        obj = beam
        for part in fName[:-1]:
            obj = getattr(obj, part)
        setattr(obj, fName[-1], value)

    def beamParamChanged(self, fName):
        if not (0 <= self.currentBeam < len(self.im.beams)):
            return
        beam = self.im.beams[self.currentBeam]
        chk, ed = self.bpWidgets[fName]
        if chk is not None and chk.isChecked():
            return                          # auto field, ignore edits
        idx = imp.BEAM_FIELD_NAMES.index(fName)
        if isinstance(ed, QtWidgets.QComboBox):
            text = ed.currentText()
        else:
            text = ed.text()
        if imp.BEAM_FIELD_IS_NUM[idx]:
            val = _num(text)
            if val is None and text.upper() != 'COM':
                self.statusbarMsg('"%s" requires a numeric value.'
                                  % ''.join(fName))
                self.refreshBeamParams()
                return
            val = text.upper() if val is None else val
        else:
            val = text
        self._setBeamField(beam, fName, val)
        beam.beamlets = []                  # geometry changed -> stale
        self.markStale()
        # keep dependent auto fields in sync
        try:
            auto = {'isocenter': self._autoChecked('isocenter'),
                    'dateOfCreation': self._autoChecked('dateOfCreation'),
                    'sourceRel': self._autoChecked('sourceRel')}
            imp.conditionBeam(beam, self.im, self.planC, auto)
        except Exception as e:
            self.statusbarMsg(str(e))
        self.refreshBeams()
        self.refreshPreview()
        self.refreshBeamParams()

    def _autoChecked(self, what):
        if what == 'isocenter':
            keys = [('isocenter', 'x'), ('isocenter', 'y'), ('isocenter', 'z')]
        elif what == 'dateOfCreation':
            keys = [('dateOfCreation',)]
        else:
            keys = [('zRel',), ('xRel',), ('yRel',)]
        return all(self.bpWidgets[k][0].isChecked() for k in keys
                   if self.bpWidgets[k][0] is not None)

    def autoCheckChanged(self):
        self.refreshBeamParams()

    # -------------------------------------------------------------- beams
    def newBeam(self):
        nB = len(self.im.beams)
        if 0 <= self.currentBeam < nB:
            beam = copy.deepcopy(self.im.beams[self.currentBeam])
        else:
            beam = imp.createDefaultBeam(nB + 1, self.planC)
        beam.beamNum = nB + 1
        beam.gantryAngle = 0.0
        beam.beamDescription = 'Beam %d' % (nB + 1)
        beam.beamUID = uid_utils.createUID('BEAM')
        beam.beamlets = []
        self.im.beams.append(beam)
        self.currentBeam = nB
        self.markStale()
        self.refreshBeams()
        self.refreshPreview()
        self.refreshBeamParams()

    def newEquispaced(self):
        n, ok = QtWidgets.QInputDialog.getInt(
            self, 'Beam Creation', 'Add how many equispaced beams?', 5, 1, 50)
        if not ok:
            return
        start, ok = QtWidgets.QInputDialog.getInt(
            self, 'Beam Creation', 'Starting point? (0-359)', 0, 0, 359)
        if not ok:
            return
        template = self.im.beams[self.currentBeam] \
            if 0 <= self.currentBeam < len(self.im.beams) else None
        firstNew = len(self.im.beams)
        imp.addEquispacedBeams(self.im, n, start, self.planC, template)
        self.currentBeam = firstNew
        self.markStale()
        self.refreshBeams()
        self.refreshPreview()
        self.refreshBeamParams()

    def delBeam(self):
        if not (0 <= self.currentBeam < len(self.im.beams)):
            return
        del self.im.beams[self.currentBeam]
        for i, b in enumerate(self.im.beams):
            b.beamNum = i + 1
        self.currentBeam = min(self.currentBeam, len(self.im.beams) - 1)
        self.markStale()
        self.refreshBeams()
        self.refreshPreview()
        self.refreshBeamParams()

    def selectBeam(self):
        rows = self.beamList.selectionModel().selectedRows()
        if rows:
            self.currentBeam = rows[0].row()
            self.refreshBeams()
            self.refreshPreview()
            self.refreshBeamParams()

    def previewAngleChanged(self, angle):
        if not (0 <= self.currentBeam < len(self.im.beams)):
            return
        beam = self.im.beams[self.currentBeam]
        beam.gantryAngle = float(angle)
        beam.beamlets = []
        self.markStale()
        try:
            imp.conditionBeam(beam, self.im, self.planC,
                              {'isocenter': self._autoChecked('isocenter'),
                               'dateOfCreation': False,
                               'sourceRel': self._autoChecked('sourceRel')})
        except Exception:
            pass
        self.refreshPreview()
        self.refreshBeamParams()

    def previewDragFinished(self):
        self.refreshBeams()
        self._updateViewerBeams()   # gantry angle changed -> redraw beams

    # --------------------------------------------------------------- scan
    def selScan(self, scanIdx):
        if scanIdx == self.im.assocScanNum(self.planC):
            return
        self.im.assocScanUID = self.planC.scan[scanIdx].scanUID
        self.im.goals = []
        for b in self.im.beams:
            b.beamlets = []
            b.beamUID = uid_utils.createUID('BEAM')
        self.markStale()
        self.refreshScan()
        self.refreshStructs()
        self.refreshPreview()

    # --------------------------------------------------------------- goals
    def addGoal(self, relStrNum):
        if relStrNum < 0:
            return
        imp.addGoal(self.im, relStrNum, self.planC)
        self.markStale()
        self.refreshStructs()
        self.addStructPop.setCurrentIndex(-1)

    def delGoal(self, goalNum):
        if not (0 <= goalNum < len(self.im.goals)):
            return
        del self.im.goals[goalNum]
        for b in self.im.beams:        # beamlets indexed by goal -> stale
            b.beamlets = []
        self.markStale()
        self.refreshStructs()

    def pbMarginText(self, goalNum, editor):
        val = _num(editor.text())
        if val is None:
            return
        self.im.goals[goalNum].PBMargin = val
        for b in self.im.beams:
            b.beamlets = []
        self.markStale()

    def strSampleRate(self, goalNum, editor):
        val = _num(editor.text())
        if val is None:
            return
        self.im.goals[goalNum].xySampleRate = int(val)
        for b in self.im.beams:
            b.beamlets = []
        self.markStale()

    def targBoxClicked(self, goalNum):
        w = self.goalTable.cellWidget(goalNum, 1)
        chk = w.findChild(QtWidgets.QCheckBox)
        self.im.goals[goalNum].isTarget = 'Yes' if chk.isChecked() else 'No'
        # Beams whose isocenter depends on the target COM are now stale.
        for b in self.im.beams:
            if any(isinstance(v, str) for v in
                   (b.isocenter.x, b.isocenter.y, b.isocenter.z)):
                b.beamlets = []
        self.markStale()
        self.refreshPreview()
        self.refreshBeamParams()

    # ------------------------------------------------------------- params
    def imParamChanged(self, pName):
        ed = self.ipWidgets[pName]
        text = ed.currentText() if isinstance(ed, QtWidgets.QComboBox) \
            else ed.text()
        p = self.im.params
        idx = imp.PARAM_NAMES.index(pName)
        val = _num(text, text) if imp.PARAM_IS_NUM[idx] else text
        if pName == ('Scatter', 'Threshold'):
            p.Scatter.Threshold = val
        elif pName == ('Scatter', 'RandomStep'):
            p.Scatter.RandomStep = val
        else:
            setattr(p, pName[0], val)
        if pName == ('algorithm',):
            self._updateParamPanelVisibility()
        self.markStale()

    def mcParamChanged(self, pName):
        ed = self.mcWidgets[pName]
        text = ed.currentText() if isinstance(ed, QtWidgets.QComboBox) \
            else ed.text()
        idx = imp.MC_PARAM_NAMES.index(pName)
        val = _num(text, text) if imp.MC_PARAM_IS_NUM[idx] else text
        setattr(self.im.params.VMC, pName[0], val)
        self.markStale()

    # ------------------------------------------------------------ browser
    def browseIM(self, idx):
        imList = imp.getIMList(self.planC)
        if idx < len(imList):
            self.im = imList[idx]
            self.saveIndex = idx
            self.currentBeam = 0 if self.im.beams else -1
        else:
            self.im = imp.initIMRTProblem(self.planC)
            self.saveIndex = -1
            self.currentBeam = -1
        self.setWindowTitle(FRESH_TITLE if self.im.isFresh else STALE_TITLE)
        self._refreshAll()

    def deleteIM(self):
        imList = imp.getIMList(self.planC)
        idx = self.browsePop.currentIndex()
        if idx >= len(imList):
            return
        if QtWidgets.QMessageBox.question(
                self, 'Delete IM', 'Delete "%s" from the plan?'
                % imList[idx].name) != QtWidgets.QMessageBox.Yes:
            return
        del imList[idx]
        if self.saveIndex == idx:
            self.saveIndex = -1
        elif self.saveIndex > idx:
            self.saveIndex -= 1
        self.refreshBrowser()
        self.statusbarMsg('IM set deleted.')

    def renameIM(self):
        newName = self.renameEdit.text().strip()
        if newName:
            self.im.name = newName
            self.refreshBrowser()

    # -------------------------------------------------------- save / run
    def save(self):
        """'Go' button: dispatch on the File action popup."""
        action = self.filePop.currentText()
        try:
            if action == 'Recompute & add dosimetry':
                self._recompute()
                self.saveIndex = imp.saveIMToPlan(self.im, self.planC, None)
            elif action == 'Recompute & overwrite dosimetry':
                self._recompute()
                self.saveIndex = imp.saveIMToPlan(
                    self.im, self.planC,
                    self.saveIndex if self.saveIndex >= 0 else None)
            elif action == 'Copy/Add dosimetry w/o calc.':
                self.saveIndex = imp.saveIMToPlan(
                    copy.deepcopy(self.im), self.planC, None)
            elif action == 'Overwrite dosimetry w/o calc.':
                self.saveIndex = imp.saveIMToPlan(
                    self.im, self.planC,
                    self.saveIndex if self.saveIndex >= 0 else None)
            elif action == 'Revert to Original':
                imList = imp.getIMList(self.planC)
                if 0 <= self.saveIndex < len(imList):
                    self.im = imList[self.saveIndex]
                else:
                    self.im = imp.initIMRTProblem(self.planC)
                self.currentBeam = 0 if self.im.beams else -1
                self._refreshAll()
                self.statusbarMsg('Reverted.')
                return
            self.setWindowTitle(FRESH_TITLE if self.im.isFresh
                                else STALE_TITLE)
            self.refreshBrowser()
            self.statusbarMsg('Done: %s' % action)
        except Exception as e:
            self.statusbarMsg('Error: %s' % e)
            QtWidgets.QMessageBox.warning(self, 'IMRTP', str(e))
        finally:
            self.waitbar(0)

    def _recompute(self):
        def cb(msg, frac):
            self.statusbarMsg(msg)
            if frac is not None:
                self.waitbar(frac)
        dose3M = imrtp_run.runIMRTP(self.im, self.planC, cb)
        self._lastDoseNum = imrtp_run.doseToPlanC(dose3M, self.im, self.planC)
        self.setWindowTitle(FRESH_TITLE)
        self.statusbarMsg('Dose added to planC.dose[%d].' % self._lastDoseNum)

    def showDose(self):
        """'Show' button: display the computed dose. Uses the pyCERR Qt
        viewer (pycerr_gui) when one is attached; otherwise falls back to
        the napari viewer."""
        doseNum = getattr(self, '_lastDoseNum', None)
        if doseNum is None and len(self.planC.dose) > 0:
            doseNum = len(self.planC.dose) - 1
        if doseNum is None:
            self.statusbarMsg('No dose computed yet; press Go first.')
            return
        if self.viewer is not None:
            try:
                self.viewer.display_dose(doseNum)
                self.statusbarMsg('Dose %d shown in the pyCERR viewer.'
                                  % (doseNum + 1))
                return
            except Exception as e:           # fall through to napari
                self.statusbarMsg('pyCERR viewer unavailable: %s' % e)
        try:
            from cerr import viewer as vwr
            scanNum = self.im.assocScanNum(self.planC) or 0
            strNums = [structr.getStructNumFromUID(g.strUID, self.planC)
                       for g in self.im.goals]
            vwr.showNapari(self.planC, scan_nums=[scanNum],
                           struct_nums=strNums, dose_nums=[doseNum],
                           displayMode='2d')
        except Exception as e:
            self.statusbarMsg('Viewer unavailable: %s' % e)


# ==========================================================================
# Entry point
# ==========================================================================

def IMRTPGui(planC, im=None, saveIndex=None, block=True, viewer=None):
    """Open the IMRTP GUI for ``planC``.

    Args:
        planC: pyCERR plan container with at least one scan loaded.
        im: optional existing :class:`~cerr.imrtp.imrtp_problem.IMRTProblem`
            to edit (analog of ``IMRTPGui('init', IM, index)`` in Matlab).
        saveIndex: index of ``im`` within ``planC.im`` when editing a stored
            set.
        block: when True (script usage) run the Qt event loop until the
            window is closed; when False (napari / IPython with an active
            event loop) just show the window and return it.
        viewer: optional ``PyCerrViewer`` to show the computed dose and the
            beam's-eye-view overlays in (instead of opening napari).

    Returns:
        IMRTPGuiWindow: the window instance.
    """
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv)
    win = IMRTPGuiWindow(planC, im=im, saveIndex=saveIndex, viewer=viewer)
    win.show()
    if block and owns_app:
        app.exec_() if hasattr(app, 'exec_') else app.exec()
    return win
