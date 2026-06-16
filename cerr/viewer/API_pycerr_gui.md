# pyCERR Viewer — Programmatic Control API

This document describes how to drive the pyCERR Qt viewer (`pycerr_gui.py`) from
Python: opening a viewer, changing what it displays, navigating, and saving
screenshots to file — without manual interaction.

All entry points live in `cerr.viewer.pycerr_gui`:

```python
from cerr.viewer.pycerr_gui import show, launch, capture, PyCerrViewer
```

---

## 1. Getting a viewer handle

| Function | Blocks? | Use for |
|----------|---------|---------|
| `show(planC=None)` | No | Interactive sessions (IPython/Jupyter with `%gui qt5`); returns the live `PyCerrViewer` so you can keep scripting it. |
| `launch(planC=None, dicomDir=None)` | Yes | Plain scripts; opens the viewer, runs the Qt event loop, and returns the (possibly edited) `planC` when the window closes. |
| `capture(planC, out_path, target=..., setup=..., size=...)` | No | One-shot, non-interactive screenshot generation (builds a viewer, applies a setup callback, writes a file, closes). |

```python
import cerr.plan_container as pc
from cerr.viewer.pycerr_gui import show

planC = pc.loadDcmDir(r"C:/data/patient1")
v = show(planC)          # v is a PyCerrViewer; drive it with the methods below
```

> In a plain `.py` script (no IPython event loop), prefer `launch()` or `capture()`.
> `show()` needs an active Qt event loop (`%gui qt5` in IPython, or napari).

---

## 2. Control API (methods on the returned `PyCerrViewer`)

Every setter also updates the on-screen control and triggers a redraw.

### Data

| Method | Description |
|--------|-------------|
| `setPlanC(planC)` | Replace the displayed plan container. |
| `getPlanC()` | Return the current `planC` (includes anything edited in the GUI). |
| `import_dicom(path)` | Import a DICOM directory into the current plan (no dialog when `path` is given). |
| `refresh_views(only=None)` | Redraw after editing `planC` externally. `only` may be a winId. |

### Scan / window-level

| Method | Description |
|--------|-------------|
| `set_scan(scanNum)` | Select the base scan (index into `planC.scan`). |
| `set_window_level(center, width)` | Manual window center/width (per-scan; remembered). |
| `set_window_preset(name)` | Named CT window preset (see §4). |
| `set_scan_colormap(name)` | Base-scan colormap (see §4 `SCAN_CMAPS`). |
| `set_scan_opacity(alpha)` | Base-scan opacity, `0..1`. |

### Dose

| Method | Description |
|--------|-------------|
| `set_dose(doseNum)` | Show dose `doseNum`; `doseNum < 0` or `None` hides dose. |
| `set_dose_alpha(alpha)` | Dose colorwash opacity, `0..1`. |
| `display_dose(doseNum)` | Refresh from `planC` (pick up a newly added dose) and show it; raises the window. |

Advanced dose colormap / range control is on the colorbar widget:

```python
v.colorbar._set_cmap("jet")           # any name in cerr_colormaps.CERR_COLORMAP_NAMES
v.colorbar.cbarRange = [0.0, 60.0]    # colormap mapping range (Gy)
v.colorbar.dispRange = [10.0, 60.0]   # display (mask) range (Gy)
v.colorbar.update(); v.refresh_views()
```

### Structures

| Method | Description |
|--------|-------------|
| `set_structures_visible(which)` | `"all"`, `"none"`, or a list of structure indices. |
| `set_structure_dots(on)` | Toggle contour vertex dots ("Alaly dots"). |
| `set_contour_linewidth(width)` | Contour line width (points). |
| `goto_structure(strNum)` | Center all three orthogonal views on a structure's center of mass. |

### Views, layout & navigation

| Method | Description |
|--------|-------------|
| `set_layout(name)` | `"single"`, `"two"`, `"default"`, `"columns"`, `"grid"`. |
| `set_orientation(winId, orientation)` | Set a window's view (`"Axial"/"Sagittal"/"Coronal"/"3D"`). |
| `set_slice(orientation, k)` | Move the window showing `orientation` to slice index `k`. |
| `set_crosshairs(on)` | Toggle linked crosshairs. |
| `set_orientation_labels(on)` | Toggle L/R/A/P/S/I labels. |
| `set_lock_views(on)` | Lock slices across windows of the same orientation. |
| `reset_all_views()` | Reset pan/zoom on all views. |

### Beams (IMRTP overlays)

| Method | Description |
|--------|-------------|
| `setBeams(beams)` | Draw beam overlays (2D cross-sections + 3D pyramids). `beams` is a list of `{"polylines":[Nx3...], "apex":(3,), "corners":(4,3), "color":(r,g,b)}`; `[]` clears. |

### Registration QA

| Method | Description |
|--------|-------------|
| `start_reg_qa(base=None, moving=None, mode="Mirrorscope", size=None, base_frac=None)` | Open and configure the registration QA tool. `base`/`moving` are scan indices; `mode` is `"Mirrorscope"`/`"Sidebyside"`/`"AlternateGrid"`/`"Toggle"`; `size` is the mirror-box/tile size (cm); `base_frac` is the Toggle-mode base weight (`0..1`). Returns the `RegQaDialog`. |
| `stop_reg_qa()` | Close the registration QA tool. |

### DVH

| Method | Description |
|--------|-------------|
| `compute_dvh(doseNum=None, structNums=None, num_bins=400)` | Compute cumulative DVHs. Returns `(doseAxis_Gy, {structName: volumePct})`, each structure interpolated onto a shared `0..max` dose axis. `doseNum` defaults to the displayed dose; `structNums` defaults to all structures (or pass an int / list). |
| `export_dvh(path, doseNum=None, structNums=None, num_bins=400)` | Same as `compute_dvh` but also writes a wide-format CSV (`Dose(Gy), <struct>, ...`). Returns the same `(axis, table)`. |

### Reading state

Useful attributes you can read back: `v.planC`, `v.scanNum`, `v.doseNum`,
`v.windowCenter`, `v.windowWidth`, `v.slices` (winId → index),
`v.lastSlice` (orientation → index), `v.activeWins`, `v.views` (winId →
`SliceView`).

---

## 3. Screenshots

```python
v.save_screenshot(path, target="window", dpi=150)
```

| `target` | Captures |
|----------|----------|
| `"window"` | The whole GUI window. |
| `"views"` | Just the panel of view windows. |
| `"A"`/`"B"`/`"C"`/`"D"` | A single view window. |
| an orientation (`"Axial"`, `"Sagittal"`, `"Coronal"`, `"3D"`) | The window currently showing it. |

2D views are saved from the matplotlib figure (crisp, honors `dpi`); the 3D view
is saved with a pyvista/VTK render. The file format follows the path extension
(`.png`, `.jpg`, ...). Returns the path written; raises `ValueError` if no window
matches the target.

---

## 4. Valid values

- **Window ids:** `"A"`, `"B"`, `"C"`, `"D"` (D is the 2×2 layout's 4th window,
  3D by default).
- **Orientations:** `VIEW_AXIAL` (`"Axial"`), `VIEW_SAGITTAL` (`"Sagittal"`),
  `VIEW_CORONAL` (`"Coronal"`), `VIEW_3D` (`"3D"`).
- **Layouts:** `"single"`, `"two"`, `"default"`, `"columns"`, `"grid"`.
- **CT window presets** (`CT_WINDOW_PRESETS`): `"Abd/Med"`, `"Head"`, `"Liver"`,
  `"Lung"`, `"Spine"`, `"Vrt/Bone"`, `"Soft Tissue"`, `"Brain"`, `"PET SUV"`.
- **Scan colormaps** (`SCAN_CMAPS`): `gray, bone, hot, jet, viridis, magma,
  plasma, copper, cool, Greens, Reds, Blues`.

```python
from cerr.viewer.pycerr_gui import (VIEW_AXIAL, VIEW_SAGITTAL, VIEW_CORONAL,
                                    VIEW_3D, CT_WINDOW_PRESETS, SCAN_CMAPS)
```

---

## 5. Examples by use case

### 5.1 Open a dataset and set it up

```python
import cerr.plan_container as pc
from cerr.viewer.pycerr_gui import launch

planC = pc.loadDcmDir(r"C:/data/patient1")

def setup_on_open(v):
    v.set_layout("default")
    v.set_window_preset("Lung")
    v.set_dose(0); v.set_dose_alpha(0.4)
    v.set_structures_visible("all")

# launch() blocks; do setup via a one-shot timer is optional — simplest is show():
# here we just open and interact manually:
planC = launch(planC)        # returns updated planC when you close the window
```

### 5.2 Non-interactive figure of one slice

```python
from cerr.viewer.pycerr_gui import capture

def prep(v):
    v.set_layout("single")              # one big axial view
    v.set_window_preset("Soft Tissue")
    v.set_structures_visible("all")
    v.set_structure_dots(False)
    v.set_slice("Axial", 60)

capture(planC, "patient1_axial60.png", target="A", setup=prep, size=(900, 900))
```

### 5.3 Four-up overview (axial/sagittal/coronal + 3D) with dose

```python
def prep(v):
    v.set_layout("grid")
    v.set_window_preset("Lung")
    v.set_dose(0); v.set_dose_alpha(0.45)
    v.set_structures_visible("all")
    v.goto_structure(0)                 # center on the first structure

capture(planC, "overview.png", target="window", setup=prep, size=(1400, 950))
```

### 5.4 Batch: one screenshot per patient

```python
import glob, os
import cerr.plan_container as pc
from cerr.viewer.pycerr_gui import capture

def prep(v):
    v.set_layout("default")
    v.set_window_preset("Soft Tissue")
    v.set_structures_visible("all")
    v.goto_structure(0)

for d in glob.glob(r"C:/data/cohort/*/"):
    planC = pc.loadDcmDir(d)
    out = os.path.join("figs", os.path.basename(d.rstrip("/\\")) + ".png")
    capture(planC, out, target="window", setup=prep)
```

### 5.5 Structure QC montage (one image per structure)

```python
from cerr.viewer.pycerr_gui import show

v = show(planC)                          # interactive session (or build manually)
v.set_layout("default")
v.set_window_preset("Soft Tissue")
for i, st in enumerate(planC.structure):
    v.set_structures_visible([i])        # show only this structure
    v.goto_structure(i)                  # center all views on it
    v.save_screenshot(f"qc_{i}_{st.structureName}.png", target="views")
```

### 5.6 Multi-scan fusion screenshot

```python
def prep(v):
    v.set_scan(0)                        # base scan
    v.set_scan_colormap("gray")
    # enable an overlay scan via the fusion panel state:
    v.overlayState[1] = {"on": True, "alpha": 0.5, "cmap": "hot"}
    v._populate_overlay_rows(); v.refresh_views()
    v.set_layout("default")

capture(planC, "fusion.png", target="views", setup=prep)
```

### 5.7 3D view screenshot

```python
def prep(v):
    v.set_layout("grid")                 # window D is the 3D view
    v.set_dose(0); v.set_dose_alpha(0.4) # isodose surfaces in 3D
    v.set_structures_visible("all")      # structure surfaces in 3D

# needs a real GL context (not offscreen); run on a machine with a display:
capture(planC, "render3d.png", target="3D", setup=prep, size=(1100, 900))
```

### 5.8 Per-axis content (different scan/dose per window)

```python
v = show(planC)
v.set_layout("columns")                  # A, B, C
v.set_orientation("A", "Axial")
v.set_orientation("B", "Axial")          # two axial windows
v.set_orientation("C", "Coronal")
# right-click menus set per-axis scan/dose/structs; the underlying state is:
v.axisSel["B"]["dose"] = -1              # hide dose only in window B
v.refresh_views()
```

### 5.9 Registration QA, scripted (+ screenshot)

```python
v = show(planC)                          # planC with >= 2 scans
# mirror-box comparison of scan 0 (base) and scan 1 (moving), 2 cm box:
v.start_reg_qa(base=0, moving=1, mode="Mirrorscope", size=2.0)
v.save_screenshot("qa_mirror.png", target="views")

# side-by-side, then a toggle blend at 30% base / 70% moving:
v.start_reg_qa(mode="Sidebyside")
v.start_reg_qa(mode="Toggle", base_frac=0.30)
v.stop_reg_qa()
```

### 5.10 Export DVHs to CSV (no GUI interaction)

```python
# all structures vs the displayed dose -> wide CSV (Dose(Gy), <struct>, ...)
axis, table = v.export_dvh("dvh_all.csv")

# a specific dose and subset of structures:
v.export_dvh("dvh_targets.csv", doseNum=0, structNums=[0, 2, 3])

# compute without writing a file, then post-process (e.g. D95, V20):
axis, table = v.compute_dvh(doseNum=0)
import numpy as np
ptv = table["PTV"]
d95 = np.interp(95.0, ptv[::-1], axis[::-1])      # dose covering 95% volume
v20 = float(np.interp(20.0, axis, ptv))           # % volume receiving >= 20 Gy
print("PTV D95 =", round(d95, 2), "Gy;  V20 =", round(v20, 1), "%")
```

### 5.11 Headless / batch on a server (2D only)

```python
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"   # set BEFORE importing the viewer

import cerr.plan_container as pc
from cerr.viewer.pycerr_gui import capture

planC = pc.loadNiiScan("scan.nii.gz", imageType="CT SCAN")
capture(planC, "headless_axial.png", target="A",
        setup=lambda v: v.set_layout("single"))
```

> Offscreen rendering is reliable for 2D targets (`"A"`, `"views"`, `"window"`
> without the 3D view). The pyvista 3D view needs a real OpenGL context, so use a
> display for `target="3D"` or 3D-containing layouts.

---

## 6. Notes

- Each setter mirrors a GUI control, so programmatic and manual changes stay in
  sync; you can mix them freely in an interactive session.
- Window/level, colormap and opacity persist per scan; switching scans restores
  each scan's settings.
- Registration QA and DVH are scriptable (see `start_reg_qa()`, `compute_dvh()`,
  `export_dvh()` above). Contouring and IMRTP remain interactive — open them with
  `show_contour_dialog()` / `show_imrtp_gui()`.
