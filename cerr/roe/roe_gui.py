"""
roe_gui.py — Radiotherapy Outcomes Explorer (ROE) GUI for pyCERR
================================================================
A PyQt5 + matplotlib interactive tool for exploring NTCP/TCP/BED
dosimetric model outcomes as a function of dose scale or fractionation.

Usage
-----
    from cerr.roe.roe_gui import launch
    win = launch(planC)          # pass an existing planC
    win = launch()               # open with empty planC

Layout
------
    QMainWindow
    └── QSplitter (horizontal)
        ├── Left panel (360 px fixed)  — QTabWidget
        │   ├── Tab 0 "Settings"
        │   │   ├── Load Models button
        │   │   ├── Model list (QListWidget)
        │   │   ├── Structure mapping table
        │   │   ├── Dose plan combo
        │   │   ├── Prescribed dose + fractions
        │   │   ├── Model parameters table
        │   │   ├── Plot mode combo
        │   │   └── Plot button
        │   └── Tab 1 "Constraints"  (placeholder)
        └── Right panel
            ├── Matplotlib FigureCanvas (dual-axis)
            ├── Slider + label
            └── Manual value entry
"""

from __future__ import annotations

import copy
import json
import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Qt imports
# ---------------------------------------------------------------------------
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Matplotlib / backend
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# pyCERR backend imports
# ---------------------------------------------------------------------------
from cerr.dvh import getDVH, doseHist
from cerr.dataclasses.dose import fractionSizeCorrect, fractionNumCorrect
from cerr.dataclasses.structure import getMatchingIndex
from cerr.roe.dosimetric_models import (
    logitFn,
    LKBFn,
    linearFn,
    appeltLogit,
    coxFn,
    biexpFn,
    lungBED,
    lungTCP,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_BIN_WIDTH = 0.05  # Gy — histogram bin width
_SCALE_MIN, _SCALE_MAX = 0.5, 1.5
_SCALE_POINTS = 99
_DELTA_FRX_MIN, _DELTA_FRX_MAX = -10, 10

# 8-colour cycle (colour-blind friendly)
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

# Map model type strings to which axis they use
_TYPE_NTCP = "NTCP"
_TYPE_TCP  = "TCP"
_TYPE_BED  = "BED"

# Plot mode indices
_MODE_SCALE   = 0   # NTCP/TCP vs dose scale
_MODE_DELTA   = 1   # NTCP/TCP vs ΔFractions
_MODE_BED     = 2   # NTCP vs BED
_MODE_TCP     = 3   # NTCP vs TCP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_type(model: dict) -> str:
    """Return 'NTCP', 'TCP', or 'BED' from model dict."""
    return model.get("type", _TYPE_NTCP).upper()


def _get_struct_list(model: dict) -> List[str]:
    """Return list of required structure names for the given model."""
    structs = model.get("parameters", {}).get("structures", [])
    if isinstance(structs, dict):
        return list(structs.keys())
    elif isinstance(structs, str):
        return [structs]
    elif isinstance(structs, list):
        return structs
    return []


def _find_struct_index(name: str, available: List[str]) -> Optional[int]:
    """
    Try exact then contains matching; return first matching index or None.
    """
    idxV = getMatchingIndex(name, available, matchCriteria="exact")
    if idxV is not None and len(idxV) > 0:
        return int(idxV[0])
    idxV = getMatchingIndex(name, available, matchCriteria="contains")
    if idxV is not None and len(idxV) > 0:
        return int(idxV[0])
    return None


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ROEGui(QMainWindow):
    """Radiotherapy Outcomes Explorer — interactive NTCP/TCP/BED explorer."""

    def __init__(self, planC=None):
        super().__init__()
        self.planC = planC

        # ---- application state ----
        self.models: List[dict] = []          # loaded model dicts
        self._dvh_cache: Dict[Tuple, Tuple] = {}  # key: (model_name, struct_names) → (bins_list, vols_list)
        self._struct_assignments: Dict[int, List[str]] = {}  # model index → selected struct names
        self._crosshair_line = None            # vertical dashed line on plot
        self._selected_model_row = -1

        self.setWindowTitle("Radiotherapy Outcomes Explorer (ROE)")
        self.resize(1200, 700)

        self._setup_ui()
        self._populate_dose_combo()

    # ======================================================================
    # UI Construction
    # ======================================================================

    def _setup_ui(self):
        """Build all widgets and layouts."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)

        # --- Main splitter ---
        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        # ---- LEFT panel ----
        left_widget = self._build_left_panel()
        left_widget.setFixedWidth(370)
        splitter.addWidget(left_widget)

        # ---- RIGHT panel ----
        right_widget = self._build_right_panel()
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

        # ---- Status bar ----
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load model files to begin.")

    # ------------------------------------------------------------------
    # Left panel
    # ------------------------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        """Create the left tab widget wrapped in a scroll area."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.tab_widget.addTab(self._build_settings_tab(), "Settings")
        self.tab_widget.addTab(self._build_constraints_tab(), "Constraints")

        return container

    def _build_settings_tab(self) -> QWidget:
        """Build the Settings tab with all controls."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(6)

        bold = QFont()
        bold.setBold(True)

        # ---- Load Models ----
        self.btn_load = QPushButton("Load Models")
        self.btn_load.clicked.connect(self.load_models)
        layout.addWidget(self.btn_load)

        layout.addWidget(self._section_label("Loaded Models", bold))
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(110)
        self.model_list.currentRowChanged.connect(self.on_model_selected)
        self.model_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_list.customContextMenuRequested.connect(self._model_list_context_menu)
        layout.addWidget(self.model_list)

        # ---- Structure mapping ----
        layout.addWidget(self._section_label("Structure Mapping", bold))
        self.struct_table = QTableWidget(0, 2)
        self.struct_table.setHorizontalHeaderLabels(["Required", "Available"])
        self.struct_table.horizontalHeader().setStretchLastSection(True)
        self.struct_table.setMaximumHeight(110)
        layout.addWidget(self.struct_table)

        # ---- Dose plan ----
        layout.addWidget(self._section_label("Dose Plan", bold))
        dose_row = QHBoxLayout()
        dose_row.addWidget(QLabel("Plan:"))
        self.dose_combo = QComboBox()
        self.dose_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dose_combo.currentIndexChanged.connect(self._on_plan_changed)
        dose_row.addWidget(self.dose_combo)
        layout.addLayout(dose_row)

        # ---- Prescribed dose + fractions ----
        rx_row = QHBoxLayout()
        rx_row.addWidget(QLabel("Rx dose (Gy):"))
        self.rx_dose_edit = QLineEdit("60")
        self.rx_dose_edit.setMaximumWidth(60)
        self.rx_dose_edit.editingFinished.connect(self._on_rx_changed)
        rx_row.addWidget(self.rx_dose_edit)
        rx_row.addWidget(QLabel("Fractions:"))
        self.rx_frx_edit = QLineEdit("30")
        self.rx_frx_edit.setMaximumWidth(50)
        self.rx_frx_edit.editingFinished.connect(self._on_rx_changed)
        rx_row.addWidget(self.rx_frx_edit)
        layout.addLayout(rx_row)

        # ---- Model parameters ----
        layout.addWidget(self._section_label("Model Parameters", bold))
        self.param_table = QTableWidget(0, 2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.param_table.horizontalHeader().setStretchLastSection(True)
        self.param_table.setMaximumHeight(160)
        self.param_table.itemChanged.connect(self._on_param_changed)
        layout.addWidget(self.param_table)

        # ---- Plot mode ----
        layout.addWidget(self._section_label("Plot Mode", bold))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "NTCP/TCP vs Dose Scale",
            "NTCP/TCP vs ΔFractions",
            "NTCP vs BED",
            "NTCP vs TCP",
        ])
        self.mode_combo.currentIndexChanged.connect(self.switch_plot_mode)
        layout.addWidget(self.mode_combo)

        # ---- Plot button ----
        self.btn_plot = QPushButton("Plot")
        self.btn_plot.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; "
            "font-weight: bold; padding: 6px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #218838; }"
        )
        self.btn_plot.clicked.connect(self.plot_models)
        layout.addWidget(self.btn_plot)

        layout.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _build_constraints_tab(self) -> QWidget:
        """Placeholder Constraints tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        table = QTableWidget(0, 1)
        table.setHorizontalHeaderLabels(["No constraints loaded"])
        layout.addWidget(table)
        return widget

    # ------------------------------------------------------------------
    # Right panel
    # ------------------------------------------------------------------

    def _build_right_panel(self) -> QWidget:
        """Build matplotlib canvas and slider controls."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # ---- Matplotlib figure ----
        self.fig = Figure(figsize=(8, 5), tight_layout=True)
        self.ax_main = self.fig.add_subplot(111)
        self.ax_twin = self.ax_main.twinx()

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        # ---- Slider row ----
        slider_row = QHBoxLayout()

        self.slider_label_prefix = QLabel("Scale:")
        slider_row.addWidget(self.slider_label_prefix)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(50)
        self.slider.setMaximum(150)
        self.slider.setValue(100)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.slider)

        self.slider_val_label = QLabel("1.00")
        self.slider_val_label.setMinimumWidth(40)
        slider_row.addWidget(self.slider_val_label)

        slider_row.addWidget(QLabel("Manual:"))
        self.manual_entry = QLineEdit()
        self.manual_entry.setMaximumWidth(60)
        self.manual_entry.setPlaceholderText("1.00")
        self.manual_entry.returnPressed.connect(self._on_manual_entry)
        slider_row.addWidget(self.manual_entry)

        layout.addLayout(slider_row)
        return container

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _section_label(text: str, font: QFont) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(font)
        return lbl

    # ======================================================================
    # planC helpers
    # ======================================================================

    def _populate_dose_combo(self):
        """Fill the dose plan combo from planC.dose."""
        self.dose_combo.blockSignals(True)
        self.dose_combo.clear()
        if self.planC is not None:
            for i, d in enumerate(self.planC.dose):
                label = getattr(d, "fractionGroupID", None) or f"Dose {i}"
                self.dose_combo.addItem(str(label), userData=i)
        self.dose_combo.blockSignals(False)

    # Keep a public alias used in docstring
    populate_dose_combo = _populate_dose_combo

    def _avail_struct_names(self) -> List[str]:
        """Return list of structure names from planC (empty list if no planC)."""
        if self.planC is None:
            return []
        return [s.structureName for s in self.planC.structure]

    def _dose_index(self) -> int:
        """Return currently selected dose index (0 if nothing selected)."""
        idx = self.dose_combo.currentData()
        return int(idx) if idx is not None else 0

    def _rx_dose(self) -> float:
        try:
            return float(self.rx_dose_edit.text())
        except ValueError:
            return 60.0

    def _rx_frx(self) -> int:
        try:
            return int(self.rx_frx_edit.text())
        except ValueError:
            return 30

    # ======================================================================
    # Event handlers — data changes
    # ======================================================================

    def _on_plan_changed(self):
        """Clear DVH cache when dose plan changes."""
        self._dvh_cache.clear()

    def _on_rx_changed(self):
        """Clear DVH cache when Rx changes (fractionation correction depends on it)."""
        self._dvh_cache.clear()

    # ======================================================================
    # Model loading
    # ======================================================================

    def load_models(self):
        """
        Open a file dialog to select one or more model JSON files.
        Parsed models are appended to self.models and shown in the list widget.
        """
        default_dir = os.path.join(
            os.path.dirname(__file__), "model_parameters"
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select model JSON files",
            default_dir,
            "JSON files (*.json);;All files (*)",
        )
        if not paths:
            return

        for path in paths:
            try:
                with open(path, "r") as fh:
                    model = json.load(fh)
                # Store file path for reference
                model["_file_path"] = path
                self.models.append(model)
                name = model.get("name", os.path.basename(path))
                mtype = _model_type(model)
                item = QListWidgetItem(f"[{mtype}] {name}")
                self.model_list.addItem(item)
            except Exception as exc:
                self.status_bar.showMessage(f"Error loading {path}: {exc}")

        self.status_bar.showMessage(f"Loaded {len(paths)} model(s). Total: {len(self.models)}.")

    def _model_list_context_menu(self, pos):
        """Show a right-click context menu on the model list with a Remove option."""
        from PyQt5.QtWidgets import QMenu
        item = self.model_list.itemAt(pos)
        if item is None:
            return
        row = self.model_list.row(item)
        menu = QMenu(self)
        action_remove = menu.addAction("Remove model")
        if menu.exec_(self.model_list.viewport().mapToGlobal(pos)) == action_remove:
            self._remove_model(row)

    def _remove_model(self, row: int):
        """Remove the model at *row* from self.models and the list widget."""
        if row < 0 or row >= len(self.models):
            return
        name = self.models[row].get("name", f"model {row}")
        self.models.pop(row)
        self.model_list.takeItem(row)
        # Evict all cache entries for this model name
        for key in list(self._dvh_cache.keys()):
            if isinstance(key, tuple) and key[0] == name:
                del self._dvh_cache[key]
        # Rebuild _struct_assignments with keys shifted down past the removed row
        self._struct_assignments = {
            (k if k < row else k - 1): v
            for k, v in self._struct_assignments.items()
            if k != row
        }
        # Clear the parameter / structure tables if the removed row was selected
        self.struct_table.setRowCount(0)
        self.param_table.setRowCount(0)
        self._selected_model_row = -1
        self.status_bar.showMessage(f"Removed '{name}'. Total: {len(self.models)}.")

    # ======================================================================
    # Model selection → populate tables
    # ======================================================================

    def on_model_selected(self, row: int):
        """
        Called when user clicks a model in the list.
        Saves the current structure mapping, then repopulates for the new model.
        """
        # Persist whatever is currently displayed before switching away
        self._save_struct_assignments(self._selected_model_row)

        self._selected_model_row = row
        if row < 0 or row >= len(self.models):
            return
        model = self.models[row]
        self.populate_structure_table(model, row)
        self.populate_param_table(model)

    def _save_struct_assignments(self, model_row: int):
        """Read the structure table combos and persist them for *model_row*."""
        if model_row < 0 or model_row >= len(self.models):
            return
        selections: List[str] = []
        for r in range(self.struct_table.rowCount()):
            combo = self.struct_table.cellWidget(r, 1)
            if combo is not None:
                text = combo.currentText()
                selections.append("" if text == "-- select --" else text)
            else:
                selections.append("")
        self._struct_assignments[model_row] = selections

    def populate_structure_table(self, model: dict, model_row: int = -1):
        """
        Fill the structure mapping table for the selected model.
        Column 0: required structure name (read-only).
        Column 1: QComboBox of available structures from planC.

        Previously saved selections for *model_row* are restored; auto-match
        is only used for rows that have no saved choice yet.
        """
        required = _get_struct_list(model)
        available = self._avail_struct_names()
        saved = self._struct_assignments.get(model_row, [])

        self.struct_table.blockSignals(True)
        self.struct_table.setRowCount(len(required))

        for row_idx, struct_name in enumerate(required):
            # Col 0 — required name (read-only)
            item = QTableWidgetItem(struct_name)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.struct_table.setItem(row_idx, 0, item)

            # Col 1 — combo of available structures
            combo = QComboBox()
            combo.addItem("-- select --")
            combo.addItems(available)

            # Restore saved selection if present, otherwise auto-match
            saved_name = saved[row_idx] if row_idx < len(saved) else ""
            if saved_name and saved_name in available:
                combo.setCurrentIndex(available.index(saved_name) + 1)
            else:
                best_idx = _find_struct_index(struct_name, available)
                if best_idx is not None:
                    combo.setCurrentIndex(best_idx + 1)  # offset for "-- select --"

            self.struct_table.setCellWidget(row_idx, 1, combo)

        self.struct_table.blockSignals(False)

    def populate_param_table(self, model: dict):
        """
        Populate the model parameters table.
        Only show leaf parameters (not 'structures') that have a 'val' key.
        Params with a 'desc' dict get a QComboBox; otherwise QLineEdit.
        """
        params = model.get("parameters", {})

        # Flatten to (param_name, entry_dict) pairs, skipping 'structures'
        flat: List[Tuple[str, dict]] = []
        for key, val in params.items():
            if key.lower() == "structures":
                continue
            if isinstance(val, dict) and "val" in val:
                flat.append((key, val))

        self.param_table.blockSignals(True)
        self.param_table.setRowCount(len(flat))

        for row_idx, (pname, entry) in enumerate(flat):
            # Col 0 — param name
            item = QTableWidgetItem(pname)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            # Store reference so _on_param_changed can update the model
            item.setData(Qt.UserRole, (pname,))
            self.param_table.setItem(row_idx, 0, item)

            # Col 1 — editable value
            current_val = entry.get("val", "")

            if "desc" in entry:
                # Show a combo with the description keys
                combo = QComboBox()
                desc_dict = entry["desc"]
                combo.addItems(list(desc_dict.keys()))
                # Pre-select current value
                for k, v in desc_dict.items():
                    if v == current_val:
                        combo.setCurrentText(k)
                        break
                combo.currentTextChanged.connect(
                    lambda text, pn=pname, dd=desc_dict, m=model:
                        self._update_model_param(m, pn, dd[text])
                )
                self.param_table.setCellWidget(row_idx, 1, combo)
            else:
                edit_item = QTableWidgetItem(str(current_val))
                edit_item.setData(Qt.UserRole, (pname,))
                self.param_table.setItem(row_idx, 1, edit_item)

        self.param_table.blockSignals(False)

    # ======================================================================
    # Parameter editing
    # ======================================================================

    def _update_model_param(self, model: dict, param_name: str, value):
        """Update a scalar parameter value in the model dict."""
        if param_name in model.get("parameters", {}):
            model["parameters"][param_name]["val"] = value
        self._dvh_cache.pop(model.get("name", ""), None)

    def _on_param_changed(self, item: QTableWidgetItem):
        """
        Called when the user edits a cell in the param table.
        Updates the backing model dict.
        """
        row = item.row()
        col = item.column()
        if col != 1:
            return

        # Retrieve param name from column 0
        name_item = self.param_table.item(row, 0)
        if name_item is None:
            return
        pname = name_item.text()

        model_row = self._selected_model_row
        if model_row < 0 or model_row >= len(self.models):
            return
        model = self.models[model_row]
        params = model.get("parameters", {})
        if pname not in params:
            return

        text = item.text().strip()
        # Try to convert to numeric
        try:
            val = float(text)
        except ValueError:
            val = text

        params[pname]["val"] = val
        # Invalidate all cache entries for this model (cache keys are tuples)
        mname = model.get("name", "")
        for key in list(self._dvh_cache.keys()):
            if isinstance(key, tuple) and key[0] == mname:
                del self._dvh_cache[key]

    # ======================================================================
    # Structure assignments
    # ======================================================================

    def get_structure_assignments(self) -> Dict[int, List[int]]:
        """
        Read the structure mapping table and return a dict:
            model_index (in self.models) → list of planC structure indices.

        For models not currently selected (and therefore not shown in the
        struct_table), we fall back to automatic name matching.
        """
        assignments: Dict[int, List[int]] = {}
        available = self._avail_struct_names()

        # If the currently selected model is shown in struct_table, use that
        sel_row = self._selected_model_row
        if 0 <= sel_row < len(self.models):
            model = self.models[sel_row]
            required = _get_struct_list(model)
            indices = []
            for r in range(self.struct_table.rowCount()):
                combo = self.struct_table.cellWidget(r, 1)
                if combo is None:
                    continue
                ci = combo.currentIndex()
                if ci <= 0:
                    indices.append(None)
                else:
                    indices.append(ci - 1)  # undo the "-- select --" offset
            assignments[sel_row] = indices

        # For all other models, auto-match
        for mi, model in enumerate(self.models):
            if mi in assignments:
                continue
            required = _get_struct_list(model)
            indices = []
            for sname in required:
                idx = _find_struct_index(sname, available)
                indices.append(idx)
            assignments[mi] = indices

        return assignments

    # ======================================================================
    # DVH computation
    # ======================================================================

    def get_dvh_for_model(
        self, model: dict, dose_num: int,
        assigned_names: Optional[List[str]] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute raw (un-corrected) DVH bins and volumes for each structure
        required by *model*.  Results are cached in self._dvh_cache keyed by
        (model name, tuple of resolved structure names).

        Parameters
        ----------
        assigned_names : optional list of structure names chosen by the user in
            the structure-assignment table.  When provided, used in preference
            over the model JSON's own structure names.

        Returns
        -------
        bins_list : list of np.ndarray  — one array per required structure
        vols_list : list of np.ndarray  — one array per required structure
        """
        required = _get_struct_list(model)
        available = self._avail_struct_names()

        # Resolve each required structure to the user's assignment (if any)
        resolved: List[str] = []
        for i, sname in enumerate(required):
            user_name = (
                assigned_names[i]
                if assigned_names and i < len(assigned_names) and assigned_names[i]
                else None
            )
            lookup = user_name if user_name else sname
            sidx = _find_struct_index(lookup, available)
            if sidx is None and user_name:
                # Fall back to the model's own name
                sidx = _find_struct_index(sname, available)
                lookup = sname
            resolved.append(lookup if sidx is not None else sname)

        cache_key = (model.get("name", id(model)), tuple(resolved))
        if cache_key in self._dvh_cache:
            return self._dvh_cache[cache_key]

        bins_list: List[np.ndarray] = []
        vols_list: List[np.ndarray] = []

        for lookup in resolved:
            sidx = _find_struct_index(lookup, available)
            if sidx is None:
                raise ValueError(
                    f"Structure '{lookup}' not found in planC "
                    f"(model '{model.get('name', '')}')."
                )
            dosesV, volsV, is_err = getDVH(sidx, dose_num, self.planC)
            if is_err:
                raise RuntimeError(
                    f"DVH calculation failed for structure '{lookup}'."
                )
            bins_v, vols_v = doseHist(dosesV, volsV, _DEFAULT_BIN_WIDTH)
            bins_list.append(np.asarray(bins_v, dtype=float))
            vols_list.append(np.asarray(vols_v, dtype=float))

        self._dvh_cache[cache_key] = (bins_list, vols_list)
        return bins_list, vols_list

    # ======================================================================
    # Model curve computation
    # ======================================================================

    def _apply_frx_correction(
        self,
        raw_bins: np.ndarray,
        model: dict,
        frx_size: Optional[float],
        frx_num: Optional[int],
    ) -> np.ndarray:
        """
        Apply fractionation correction to a single dose-bin array.

        Parameters
        ----------
        raw_bins  : uncorrected dose bin array
        model     : model dict (contains correctionType, abRatio, etc.)
        frx_size  : fraction size to use (Gy); None if not correcting by size
        frx_num   : number of fractions to use; None if not correcting by number

        Returns
        -------
        corrected dose bin array
        """
        if model.get("fractionCorrect", "no").lower() != "yes":
            return raw_bins

        ab = float(model.get("abRatio", 3))
        ctype = model.get("correctionType", "").lower()

        if ctype == "frxsize":
            std_size = float(model.get("stdFractionSize", 2.0))
            corrected = fractionSizeCorrect(
                raw_bins, std_size, ab, self.planC, frx_size
            )
        elif ctype == "frxnum":
            std_num = int(model.get("stdNumFractions", 35))
            corrected = fractionNumCorrect(
                raw_bins, std_num, ab, self.planC, frx_num
            )
        else:
            corrected = raw_bins

        return np.asarray(corrected, dtype=float)

    def _evaluate_model(
        self,
        model: dict,
        corr_bins_list: List[np.ndarray],
        vols_list: List[np.ndarray],
    ) -> float:
        """
        Evaluate the model function for the given corrected bins/volumes.
        Returns the scalar NTCP/TCP/BED value.
        """
        # Make a deep copy so we don't permanently mutate the model
        param_dict = copy.deepcopy(model["parameters"])

        fn_name = model["function"]
        fn = {
            "logitFn":    logitFn,
            "LKBFn":      LKBFn,
            "linearFn":   linearFn,
            "appeltLogit": appeltLogit,
            "coxFn":      coxFn,
            "biexpFn":    biexpFn,
            "lungBED":    lungBED,
            "lungTCP":    lungTCP,
        }.get(fn_name)

        if fn is None:
            raise ValueError(f"Unknown model function: {fn_name}")

        # Pass single array when only one structure
        if len(corr_bins_list) == 1:
            result = fn(param_dict, corr_bins_list[0], vols_list[0])
        else:
            result = fn(param_dict, corr_bins_list, vols_list)

        return float(result)

    def _compute_model_curve(
        self,
        model: dict,
        raw_bins_list: List[np.ndarray],
        raw_vols_list: List[np.ndarray],
        x_values: np.ndarray,
        mode: int,
    ) -> np.ndarray:
        """
        Compute the outcome curve for *model* across *x_values*.

        Parameters
        ----------
        x_values : for mode 0 — scale factors (float);
                   for mode 1 — delta fractions (int);
                   for mode 2 — same as mode 1 (delta frx);
                   for mode 3 — same as mode 1 (delta frx)

        Returns
        -------
        np.ndarray of outcome values (same length as x_values)
        """
        rx_dose = self._rx_dose()
        rx_frx  = self._rx_frx()
        results = np.full(len(x_values), np.nan)
        first_error: Optional[str] = None

        for i, xv in enumerate(x_values):
            try:
                if mode == _MODE_SCALE:
                    scale = float(xv)
                    frx_size = rx_dose / rx_frx if rx_frx > 0 else None
                    frx_num  = rx_frx
                    # Scale each raw bin array
                    corr_bins_list = [
                        self._apply_frx_correction(
                            raw_bins * scale, model, frx_size, frx_num
                        )
                        for raw_bins in raw_bins_list
                    ]
                else:
                    # Modes 1, 2, 3 — vary number of fractions
                    delta = int(xv)
                    new_frx = max(1, rx_frx + delta)
                    new_frx_size = rx_dose / new_frx if new_frx > 0 else None
                    corr_bins_list = [
                        self._apply_frx_correction(
                            raw_bins, model, new_frx_size, new_frx
                        )
                        for raw_bins in raw_bins_list
                    ]

                results[i] = self._evaluate_model(model, corr_bins_list, raw_vols_list)
            except Exception as exc:
                if first_error is None:
                    first_error = str(exc)
                # Leave as NaN for this point

        if first_error is not None and not np.any(np.isfinite(results)):
            # All points failed — surface the first error to the status bar
            model_name = model.get("name", "")
            self.status_bar.showMessage(f"[{model_name}] Curve error: {first_error}")

        return results

    # ======================================================================
    # Main plot entry point
    # ======================================================================

    def plot_models(self):
        """
        Compute and plot curves for all loaded models.
        Called when the user clicks the Plot button.
        """
        if not self.models:
            self.status_bar.showMessage("No models loaded. Use 'Load Models' first.")
            return
        if self.planC is None:
            self.status_bar.showMessage("No planC loaded.")
            return

        mode = self.mode_combo.currentIndex()
        dose_num = self._dose_index()

        # Clear axes
        self.ax_main.cla()
        self.ax_twin.cla()
        self.ax_twin.yaxis.set_label_position("right")
        self.ax_twin.yaxis.tick_right()
        self._crosshair_line = None

        # Choose X axis values
        if mode == _MODE_SCALE:
            x_values = np.linspace(_SCALE_MIN, _SCALE_MAX, _SCALE_POINTS)
        else:
            x_values = np.arange(_DELTA_FRX_MIN, _DELTA_FRX_MAX + 1, dtype=float)

        # Gather per-type curves — tuples are (y, label, color, model_dict)
        ntcp_curves: List[Tuple[np.ndarray, str, str, dict]] = []
        tcp_curves:  List[Tuple[np.ndarray, str, str, dict]] = []
        bed_curves:  List[Tuple[np.ndarray, str, str, dict]] = []

        # Save the current model's structure assignments before iterating
        if self._selected_model_row >= 0:
            self._save_struct_assignments(self._selected_model_row)

        for mi, model in enumerate(self.models):
            color = _COLORS[mi % len(_COLORS)]
            name  = model.get("name", f"Model {mi}")
            mtype = _model_type(model)

            # Use user-assigned structure names when available
            assigned_names = self._struct_assignments.get(mi, [])

            try:
                raw_bins_list, raw_vols_list = self.get_dvh_for_model(
                    model, dose_num, assigned_names
                )
            except Exception as exc:
                self.status_bar.showMessage(f"[{name}] DVH error: {exc}")
                continue

            try:
                y = self._compute_model_curve(
                    model, raw_bins_list, raw_vols_list, x_values, mode
                )
            except Exception as exc:
                self.status_bar.showMessage(f"[{name}] Compute error: {exc}")
                traceback.print_exc()
                continue

            if mtype == _TYPE_NTCP:
                ntcp_curves.append((y, name, color, model))
            elif mtype == _TYPE_TCP:
                tcp_curves.append((y, name, color, model))
            elif mtype == _TYPE_BED:
                bed_curves.append((y, name, color, model))

        # ---- Dispatch to mode-specific renderer ----
        try:
            if mode == _MODE_SCALE:
                self._render_mode_scale(x_values, ntcp_curves, tcp_curves, bed_curves)
            elif mode == _MODE_DELTA:
                self._render_mode_delta(x_values, ntcp_curves, tcp_curves, bed_curves)
            elif mode == _MODE_BED:
                self._render_mode_bed(
                    x_values, ntcp_curves, tcp_curves, bed_curves,
                    self._rx_dose(), self._rx_frx()
                )
            elif mode == _MODE_TCP:
                self._render_mode_tcp(x_values, ntcp_curves, tcp_curves, bed_curves)
        except Exception as exc:
            self.status_bar.showMessage(f"Plot error: {exc}")
            traceback.print_exc()
            return

        self.canvas.draw()
        self.status_bar.showMessage("Plot updated.")

    # ======================================================================
    # Mode renderers
    # ======================================================================

    def _apply_dual_axis_labels(self, x_label: str):
        """Set axis labels and style for dual-Y plots."""
        self.ax_main.set_xlabel(x_label)
        self.ax_main.set_ylabel("NTCP", color="black")

        self.ax_twin.set_visible(True)
        self.ax_twin.yaxis.set_label_position("right")
        self.ax_twin.yaxis.tick_right()
        self.ax_twin.set_ylabel("TCP / BED (Gy)", color="#1f77b4", labelpad=8)
        self.ax_twin.tick_params(axis="y", labelcolor="#1f77b4")

        self.ax_main.set_ylim(0, 1.05)
        self.ax_main.grid(True, alpha=0.3)

    def _render_mode_scale(
        self,
        x: np.ndarray,
        ntcp: list,
        tcp: list,
        bed: list,
    ):
        """Mode 0: NTCP/TCP vs dose scale factor."""
        for (y, label, color, _model) in ntcp:
            self.ax_main.plot(x, y, color=color, linestyle="-", label=label)
        for (y, label, color, _model) in tcp:
            self.ax_twin.plot(x, y, color=color, linestyle="--", label=label)
        for (y, label, color, _model) in bed:
            self.ax_twin.plot(x, y, color=color, linestyle=":", label=label)

        self._apply_dual_axis_labels("Dose scale factor")
        self._build_combined_legend(ntcp, tcp, bed)

        # Draw initial crosshair at slider position
        self.update_slider_line(self.slider.value())

    def _render_mode_delta(
        self,
        x: np.ndarray,
        ntcp: list,
        tcp: list,
        bed: list,
    ):
        """Mode 1: NTCP/TCP vs delta fractions."""
        for (y, label, color, _model) in ntcp:
            self.ax_main.plot(x, y, color=color, linestyle="-", label=label)
        for (y, label, color, _model) in tcp:
            self.ax_twin.plot(x, y, color=color, linestyle="--", label=label)
        for (y, label, color, _model) in bed:
            self.ax_twin.plot(x, y, color=color, linestyle=":", label=label)

        self._apply_dual_axis_labels("ΔFractions")
        self.ax_main.set_xticks(np.arange(_DELTA_FRX_MIN, _DELTA_FRX_MAX + 1, 2))
        self._build_combined_legend(ntcp, tcp, bed)
        self.update_slider_line(self.slider.value())

    def _render_mode_bed(
        self,
        x: np.ndarray,
        ntcp: list,
        tcp: list,
        bed: list,
        rx_dose: float,
        rx_frx: int,
    ):
        """
        Mode 2: NTCP vs BED (parametric — delta frx is the parameter).

        BED is computed from the LQ model for each NTCP model using its own
        abRatio:  BED = D_total * (1 + (D_total / n) / ab)

        No separate BED-type model is required; all loaded NTCP models are plotted.
        If dedicated BED-type models are also loaded they are plotted in addition.
        """
        # Compute BED x-axis per NTCP model from the LQ formula
        for (ntcp_y, ntcp_label, ntcp_color, model) in ntcp:
            ab = float(model.get("abRatio", 3))
            bed_x = np.array([
                rx_dose * (1.0 + (rx_dose / max(1, rx_frx + int(dv))) / ab)
                for dv in x
            ])
            self.ax_main.plot(
                bed_x, ntcp_y, color=ntcp_color, linestyle="-", label=ntcp_label
            )

        # Also plot any explicitly-typed BED models
        for (bed_y, bed_label, bed_color, _model) in bed:
            self.ax_twin.plot(x, bed_y, color=bed_color, linestyle=":", label=bed_label)

        if not ntcp and not bed:
            self.status_bar.showMessage("No models computed for Mode 2.")
            return

        self.ax_main.set_xlabel("BED (Gy)")
        self.ax_main.set_ylabel("NTCP")
        self.ax_main.set_ylim(0, 1.05)
        self.ax_main.grid(True, alpha=0.3)
        handles, labels = self.ax_main.get_legend_handles_labels()
        if handles:
            self.ax_main.legend(handles, labels, fontsize=8, loc="best")
        self.ax_twin.set_visible(bool(bed))

    def _render_mode_tcp(
        self,
        x: np.ndarray,
        ntcp: list,
        tcp: list,
        bed: list,
    ):
        """
        Mode 3: NTCP vs TCP (parametric — delta frx is the parameter).
        Requires at least one TCP-type model to be loaded alongside NTCP models.
        """
        if not tcp:
            self.status_bar.showMessage(
                "Mode 3 (NTCP vs TCP) requires at least one TCP-type model. "
                "Load a JSON with \"type\": \"TCP\"."
            )
            return

        for (tcp_y, tcp_label, _tcp_color, _tcp_model) in tcp:
            for (ntcp_y, ntcp_label, ntcp_color, _ntcp_model) in ntcp:
                lbl = f"{ntcp_label} vs {tcp_label}"
                self.ax_main.plot(
                    tcp_y, ntcp_y, color=ntcp_color, linestyle="-", label=lbl
                )

        self.ax_main.set_xlabel("TCP")
        self.ax_main.set_ylabel("NTCP")
        self.ax_main.set_ylim(0, 1.05)
        self.ax_main.grid(True, alpha=0.3)
        handles, labels = self.ax_main.get_legend_handles_labels()
        if handles:
            self.ax_main.legend(handles, labels, fontsize=8, loc="best")
        self.ax_twin.set_visible(False)

    # ======================================================================
    # Legend helper
    # ======================================================================

    def _build_combined_legend(self, ntcp: list, tcp: list, bed: list):
        """Build a combined legend covering both axes."""
        handles = []
        for (_, label, color, _model) in ntcp:
            handles.append(
                mlines.Line2D([], [], color=color, linestyle="-", label=label)
            )
        for (_, label, color, _model) in tcp:
            handles.append(
                mlines.Line2D([], [], color=color, linestyle="--", label=f"{label} (TCP)")
            )
        for (_, label, color, _model) in bed:
            handles.append(
                mlines.Line2D([], [], color=color, linestyle=":", label=f"{label} (BED)")
            )
        if handles:
            self.ax_main.legend(handles=handles, fontsize=8, loc="best")

    # ======================================================================
    # Slider / crosshair
    # ======================================================================

    def _on_slider_changed(self, value: int):
        """Convert slider integer to float label and update crosshair."""
        mode = self.mode_combo.currentIndex()
        if mode == _MODE_SCALE:
            fval = value / 100.0
            self.slider_val_label.setText(f"{fval:.2f}")
            self.manual_entry.setText(f"{fval:.2f}")
        else:
            # delta frx mode: slider range 50–150 mapped to -10..+10
            delta = value - 100  # centre at 0
            self.slider_val_label.setText(str(delta))
            self.manual_entry.setText(str(delta))
        self.update_slider_line(value)

    def _on_manual_entry(self):
        """Parse manual entry and move slider + crosshair."""
        text = self.manual_entry.text().strip()
        mode = self.mode_combo.currentIndex()
        try:
            if mode == _MODE_SCALE:
                fval = float(text)
                fval = max(_SCALE_MIN, min(_SCALE_MAX, fval))
                self.slider.setValue(int(round(fval * 100)))
            else:
                delta = int(round(float(text)))
                delta = max(_DELTA_FRX_MIN, min(_DELTA_FRX_MAX, delta))
                self.slider.setValue(delta + 100)
        except ValueError:
            pass

    def update_slider_line(self, slider_value: int):
        """
        Draw (or move) a vertical dashed crosshair at the slider position.
        Also annotates the value.
        """
        mode = self.mode_combo.currentIndex()
        if mode in (_MODE_BED, _MODE_TCP):
            return  # no crosshair for parametric modes

        # Map slider integer to plot X coordinate
        if mode == _MODE_SCALE:
            x_pos = slider_value / 100.0
        else:
            x_pos = float(slider_value - 100)

        # Remove old line(s)
        if self._crosshair_line is not None:
            try:
                self._crosshair_line.remove()
            except Exception:
                pass
            self._crosshair_line = None

        # Remove old annotations
        for ann in self.ax_main.texts:
            try:
                ann.remove()
            except Exception:
                pass

        # Draw new crosshair
        self._crosshair_line = self.ax_main.axvline(
            x=x_pos, color="black", linestyle="--", linewidth=1.2, alpha=0.7
        )

        # Annotate value
        ylim = self.ax_main.get_ylim()
        y_pos = ylim[1] * 0.95 if ylim[1] > 0 else 0.95
        if mode == _MODE_SCALE:
            label = f"Scale={x_pos:.2f}"
        else:
            label = f"ΔFrx={int(x_pos):+d}"
        self.ax_main.text(
            x_pos, y_pos, label,
            fontsize=8, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
        )

        self.canvas.draw_idle()

    # ======================================================================
    # Plot mode switching
    # ======================================================================

    def switch_plot_mode(self, index: int):
        """
        Reconfigure slider range and visibility for the selected plot mode.
        """
        if index == _MODE_SCALE:
            self.slider.setMinimum(50)
            self.slider.setMaximum(150)
            self.slider.setValue(100)
            self.slider_label_prefix.setText("Scale:")
            self.slider.setVisible(True)
        elif index == _MODE_DELTA:
            self.slider.setMinimum(90)   # 90 → delta=-10
            self.slider.setMaximum(110)  # 110 → delta=+10
            self.slider.setValue(100)
            self.slider_label_prefix.setText("ΔFrx:")
            self.slider.setVisible(True)
        else:
            # Modes 2 & 3 — hide slider
            self.slider.setVisible(False)
            self.slider_label_prefix.setText("")

        # Clear plot when mode changes
        self.ax_main.cla()
        self.ax_twin.cla()
        self.ax_twin.set_visible(True)
        self.ax_twin.yaxis.set_label_position("right")
        self.ax_twin.yaxis.tick_right()
        self.canvas.draw()


# ============================================================================
# Module-level launch function
# ============================================================================

def launch(planC=None):
    """
    Launch the ROE GUI.

    Parameters
    ----------
    planC : pyCERR PlanC object, optional
        If None the GUI opens without patient data (DVH calculations will be
        skipped until a planC is attached).

    Returns
    -------
    ROEGui
        The main window instance (keep a reference to prevent garbage collection).

    Example
    -------
    >>> from cerr.roe.roe_gui import launch
    >>> win = launch(planC)
    """
    app = QApplication.instance()
    _owns_app = app is None
    if _owns_app:
        app = QApplication(sys.argv)

    win = ROEGui(planC)
    win.show()

    # Only run the event loop when we created the QApplication ourselves.
    # If an existing loop is already running (napari, IPython %gui qt, etc.)
    # the caller's loop will dispatch events and exec_() must NOT be called.
    if _owns_app:
        app.exec_()

    return win


# ============================================================================
# Script entry point
# ============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ROEGui(planC=None)
    win.show()
    sys.exit(app.exec_())
