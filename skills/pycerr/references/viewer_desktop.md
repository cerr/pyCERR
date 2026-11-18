# pyCERR desktop viewer (`cerr.viewer.pycerr_gui`)

A Qt viewer with linked axial, sagittal and coronal views and a 3D view, plus
tools for contouring, registration QA, DVH, gamma, structure consensus and 3D
volume rendering. It needs the `viewer` extra: `pip install "pycerr[viewer]"`.

## Opening it

```python
from cerr.viewer.pycerr_gui import launch, show, capture

planC = launch(planC)            # blocking; returns planC with anything edited or imported in the GUI
v = show(planC)                  # non-blocking; IPython/Jupyter: run `%gui qt` first
planC = v.getPlanC()             # current state at any time
v.setPlanC(otherPlanC)           # swap the plan shown
v.refresh_views()                # redraw after changing planC from the script
capture(planC, 'view.png', target='window', setup=lambda v: v.set_window_preset('Lung'))
                                 # render off-screen and save a screenshot, no interaction
```

A plain script that calls `show()` must keep the Qt event loop running at the
end:

```python
from qtpy.QtWidgets import QApplication
app = QApplication.instance()
(app.exec_ if hasattr(app, 'exec_') else app.exec)()
```

## Scripting API (methods of the viewer object)

| Area | Methods |
|---|---|
| Scan | `set_scan(scanNum)`, `set_window_preset(name)`, `set_window_level(center, width)`, `set_scan_colormap(name)`, `set_scan_opacity(alpha)` |
| Dose | `set_dose(doseNum)` (None or < 0 hides), `set_dose_alpha(alpha)`, `display_dose(doseNum)` |
| Structures | `set_structures_visible('all' / 'none' / [indices])`, `set_structure_dots(on)`, `set_contour_linewidth(w)` |
| Navigation | `goto_structure(structNum)` (centre on its centre of mass), `set_slice(orientation, k)`, `set_crosshairs(on)`, `set_orientation_labels(on)`, `reset_all_views()` |
| Layout | `set_layout('single' / 'two' / 'default' / 'columns' / 'grid')`, `set_orientation(winId, 'Axial' / 'Sagittal' / 'Coronal' / '3D')`, `set_lock_views(on)` |
| Registration QA | `start_reg_qa(base, moving, mode, size, base_frac)`, `stop_reg_qa()` |
| Deformation field | `set_dvf_overlay(deformNum, scanNum=None, resolution=None, structNum=None, surfFlag=False, subtractMedian=False, colorBy='length', ...)`, `clear_dvf_overlay()` |
| Tools | `show_contour_dialog()`, `show_3d_volume()`, `show_dvh_dialog()`, `show_gamma_dialog()`, `show_consensus_dialog()`, `show_reg_dialog()`, `show_roe_gui()` |
| DVH | `compute_dvh(doseNum, structNums, num_bins)`, `export_dvh(path, doseNum, structNums, num_bins)` |
| Data | `import_dicom(path)`, `import_dicom_files(files)`, `load_path(path)` |
| Screenshot | `save_screenshot(path, target='window' / 'views' / 'A'..'D' / 'Axial'... / '3d', dpi=150)` |

- **Windows** are named A–D:
  - `default`: A large on the left; B and C stacked on the right.
  - `two`: A and B side by side.
  - `columns`: A, B, C.
  - `grid`: 2 × 2, with D showing 3D.
- **Window presets** are (center, width):

  | Preset | Center, width |
  |---|---|
  | Soft Tissue | 40, 400 |
  | Lung | −500, 1500 |
  | Vrt/Bone | 400, 1500 |
  | Brain | 40, 80 |
  | Liver | 80, 305 |
  | Head | 45, 125 |
  | Abd/Med | −10, 330 |
  | Spine | 30, 300 |
  | PET SUV | 2.5, 5 |

- **Colormaps:** gray, bone, hot, jet, viridis, magma, plasma, copper, cool,
  Greens, Reds, Blues.

## Two scans together (e.g. a CT and its texture map, or a registered pair)

All windows show the same base scan, set with `set_scan`. There are two ways
to see a second scan:

1. **Registration QA:** both scans in every window, on the same slice.
   - Call `v.start_reg_qa(base=0, moving=1, mode=...)`. The mode is one of:
     - `'Sidebyside'`: a split line, base on one side and moving on the other;
     - `'AlternateGrid'`: a checkerboard;
     - `'Toggle'`: a blend, weighted with `base_frac`;
     - `'Mirrorscope'`: a mirror box.
   - `v.stop_reg_qa()` ends it.
2. **A second window showing the other scan, scrolling with the first.**
   - In the GUI: right-click a window > ScanSet > pick the scan, then
     right-click > "Lock slices across matching views".
   - Programmatically: `set_layout('two')`, `set_orientation('B', 'Axial')`,
     `set_lock_views(True)`. The per-window ScanSet choice has no public
     method yet, so set it from the right-click menu.

A texture map covers only the box around its structure, so outside that box
the map side is empty.

## 3D

- **3D cut planes:** a window set to `'3D'`, e.g. window D in the `grid` layout,
  shows the three orthogonal planes in 3D.
- **True volume rendering:** `show_3d_volume()` opens a separate window.

When the user says "3D visualization" or "volume rendering", they usually mean
`show_3d_volume()`.
