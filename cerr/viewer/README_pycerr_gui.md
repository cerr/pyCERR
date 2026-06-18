# pyCERR Viewer GUI

A CERR-style (MATLAB CERR) slice viewer built on **pyCERR**, in a single file: `pycerr_gui.py`.

> **Scripting the viewer?** See [API_pycerr_gui.md](API_pycerr_gui.md) for the full
> programmatic control + screenshot API with worked examples.

## Setup (Windows, your paths)

```bat
pip install PyQt5 matplotlib scipy
pip install pyvista pyvistaqt   :: optional, GPU-accelerated 3D view
```

pyCERR itself: either `pip install pyCERR`, **or** use your local checkout — the script
already points at it via:

```python
PYCERR_PATH = r"C:\software\pyCERR_master\pyCERR"
```

(adjust if your `cerr` package folder lives elsewhere — `PYCERR_PATH` must be the folder
that *contains* the `cerr` directory).

## Run

```bat
python pycerr_gui.py
python pycerr_gui.py C:\path\to\dicom_dir   :: auto-import on launch
```

You can also **drag and drop** onto the open window to load data:
- a **DICOM directory** (or a single `.dcm` file → its folder is imported),
- a **NIfTI** file (`.nii` / `.nii.gz`) — loaded as a scan, or as a dose/structure
  when a scan is already loaded and the file name contains `dose` /
  `mask`/`label`/`seg`/`struct`/`roi`,
- a **`.pkl`** plan container.

Dropping several files at once loads scans/plans before dose/structure NIfTIs that
depend on them.

## Features (mapped to classic CERR)

| CERR GUI                          | pyCERR Viewer                                              |
|-----------------------------------|------------------------------------------------------------|
| File > Import DICOM               | File > Import DICOM directory (`pc.loadDcmDir`)            |
| Drag & drop                       | Drop a DICOM folder, NIfTI (`.nii/.nii.gz`) or `.pkl` onto the window to load it |
| Axial/Sagittal/Coronal viewports  | 3 linked views, sliders + mouse-wheel slice navigation     |
| Pan / zoom                        | Drag to pan, right-drag up/down to zoom, double-click or `R` to reset |
| Per-axis ScanSet/DoseSet/StructSet menu | Right-click a view: choose the scan, dose & structures shown in that view |
| Draw Ruler                        | Right-click > Draw Ruler: drag to measure in-plane distance (cm) |
| Linked crosshairs                 | Track the other views' slices; toggle via View menu or `X` |
| Scan fusion                       | "Scan overlays" panel: overlay any scan with per-scan opacity & colormap |
| CT window presets                 | Abd/Med, Head, Liver, Lung, Spine, Bone, Soft Tissue, ...  |
| Scan colormap & opacity           | Scan panel: choose a colormap and opacity for the base scan |
| Structure fill/contour toggles    | Structure list with checkboxes, DICOM colors (`rasterseg.getStrMask`) |
| Contour points / line width / locate | Structures panel: "Points" toggles contour vertex dots, "Line" sets contour width, double-click a structure to center all views on it |
| Dose colorwash + colorbar         | Dose combo, alpha slider, jet colorwash, colorbar          |
| Crosshairs + position readout     | Status bar: x/y/z (cm), scan value (HU/SUV), dose (Gy)     |
| Patient orientation markers       | L/R/A/P/S/I at the edges of each 2D view + 3D triad (toggle with `O`) |
| Contouring (contourControl)       | Tools > Contouring — draw/erase structures on the axial view, save via `importStructureMask` |
| DVH menu                          | Tools > DVH — cumulative DVHs via `cerr.dvh.getDVH/doseHist` |
| IMRTP GUI                         | Tools > IMRTP — beamlet dose calculation (`cerr.imrtp.imrtp_gui`) |
| ROE                               | Tools > ROE — Radiotherapy Outcomes Explorer (`cerr.roe.roe_gui`) |
| Save plan                         | File > Save planC (.pkl) / Open planC (.pkl)               |

Also supports NIfTI import: scan (`loadNiiScan`), dose (`loadNiiDose`),
and label masks (`loadNiiStructure`).

## Patient orientation markers
Each 2D view shows the patient orientation at its edge midpoints — **L**eft,
**R**ight, **A**nterior, **P**osterior, **S**uperior, **I**nferior — derived from
pyCERR's virtual coordinate axes (+x = L, +y = A, +z = I), so e.g. the axial view
reads R | L horizontally and A / P vertically. The 3D view shows a labeled L/A/I
orientation triad in its corner. Toggle with View > Show orientation labels (`O`).

## Pan, zoom & crosshairs
- **Drag** (left or middle button) pans a view; **right-drag up/down** zooms in/out
  about the point where the drag started.
- **Double-click** a view (or `R` / View > Reset pan/zoom) to restore the full extent.
- Crosshairs in each view track the slice positions of the other two views and update
  live as you scroll; toggle them with `X` or View > Show crosshairs.
- **Drag a crosshair line** (grab the vertical/horizontal line, or their intersection)
  to scroll the perpendicular views to that point — a quick way to navigate to a
  feature you can see in one plane.

## Structures panel
Below the structure checklist:
- **Points** — overlay vertex dots ("Alaly dots") along each visible structure's
  contour, colored like the structure (subsampled so they read as polygon points).
- **Line** — the contour line width (cm-independent, in points) for all structures.
- **Double-click** a structure in the list to jump all three orthogonal views to that
  structure's center of mass (its isocenter slice in each plane), updating the sliders
  and crosshairs.

## Per-axis right-click menu (CERR's View/ScanSet/DoseSet/StructSet)
Right-click (without dragging) inside any view to choose what *that view* displays,
mirroring MATLAB CERR's per-axis context menu:
- **View** — switch this window between Axial, Sagittal and Coronal (so e.g. two
  windows can both show axial slices). The window keeps its own slice slider,
  scaled to the new orientation.
- **View > Layout** (main menu) — choose the window arrangement, similar to CERR's
  layout options: *One view*, *Two views side-by-side*, *One large + two stacked*
  (default), *Three columns* (equal sizes), or *Four views (2x2)* (four equal
  quadrants). Each window keeps its slice position, pan/zoom and per-axis
  overrides when you switch layouts; newly shown windows join at the last-visited
  slice of their orientation.
- **3D view** — the fourth window of the 2x2 layout shows a 3D rendering of the
  three orthogonal planes currently selected in the 2D views (any window can be
  switched to it via right-click > View > 3D). Each plane is textured with the
  scan at the current window/level (full resolution) and outlined in its locator
  color (axial yellow, sagittal cyan, coronal magenta); scrolling any 2D view
  moves the corresponding plane in 3D.

  The 3D display is GPU-accelerated via **pyvista/VTK** (`pip install pyvista
  pyvistaqt`) — artifact-free plane intersections, smooth rotation. If pyvista is
  not installed it falls back to a (slower, downsampled) matplotlib rendering.
  Left-drag rotates, right-drag or wheel zooms, middle-drag pans; a plain
  right-click opens the View menu to switch the window back to a 2D orientation.

  The 3D view also displays (pyvista mode): **structure surfaces** — smoothed,
  semi-transparent meshes in each structure's color, following the Structures
  checklist — and **isodose surfaces** at 30/50/70/90% of the dose maximum,
  colored by the dose colormap/colorbar range and following the Dose combo and
  alpha slider. A labeled L/A/I orientation triad sits in the corner, with each
  letter colored to match its axis.

  **The planes are draggable** (pyvista mode): press the left button on a plane
  (or its colored outline) and drag along its normal to scroll through slices —
  the plane moves live in 3D and the linked 2D views, sliders and crosshairs
  follow, exactly like dragging CERR's plane locators. Dragging starts only when
  the press actually hits a plane; pressing empty space rotates the camera as
  usual.
- **Lock slices across matching views** (also in the View menu, shortcut `L`) —
  when enabled, scrolling any window also scrolls every other window showing the
  *same* orientation, keeping them at the same location. Windows showing different
  orientations are unaffected. Locked state is indicated in each view's title.
- **ScanSet** — "Auto" follows the Scan combo in the left panel; picking a scan shows
  it only in this view (resampled onto the reference scan's grid, with its remembered
  window/level or an automatic 2-98 percentile window).
- **DoseSet** — "Auto" follows the Dose combo, "None" hides dose in this view, or pick
  any dose for this view only. The colorbar ranges (in Gy) apply to all views.
- **StructSet** — "Auto" follows the Structures checklist; toggling a structure
  switches this view to a manual set (starting from what is currently shown).
Overridden views show their selection in the view title, e.g. `[scan 1, dose 0]`.

The menu also offers **Draw Ruler** (CERR's ruler tool): once checked, left-drag in
that view to measure — the line shows `+` end markers and the distance in cm next to
the end point, with live readout in the status bar. The ruler persists across slice
changes and pan/zoom (drawing again replaces it); pick Draw Ruler again to clear it
and return to normal pan behavior. Distances are in-plane, in pyCERR's virtual
coordinate units (cm).

## Registration QA (Tools > Registration QA)
Compare two scans the way the napari-based `cerr.viewer` QA does, in every 2D view
(the moving scan is resampled onto the base scan's grid, each displayed with its own
window/level):
- **Mirrorscope** — a square lens box (side = 2 x the selected half-width) in which
  the moving scan is mirrored left-right about the box's center line: matching
  anatomy should meet symmetrically there. Left-drag positions the box anywhere in
  the view (it does not span the full height).
- **Side-by-side** — base left of the (draggable) line, moving right of it.
- **Alternate grid** — checkerboard of base/moving tiles (tile size in cm).
- **Toggle / blend** — a **Base** ↔ **Moving** slider cross-fades the two scans:
  at the left (Base) you see the base scan, at the right (Moving) the moving scan,
  in between a weighted blend; `T` flips between full base and full moving.
Each scan keeps its own colormap — both the base and the moving scan are rendered
with the colormap assigned to that scan in the Scan panel (which persists per scan),
so e.g. a PET moving scan set to `hot` reads against a `gray` base. Choosing a
different base scan re-references the viewer to that scan's grid. Closing the dialog
restores the normal display and left-drag panning.

## Base scan colormap & opacity
The Scan panel has a **Colormap** picker and an **Opacity** slider that apply to the
base scan in every 2D view, independent of the registration QA tool — useful for
PET/functional images (e.g. a `hot` or `jet` scan) or for dialling the scan back so
structures and dose stand out.

## Scan overlays (fusion)
When more than one scan is loaded, a "Scan overlays (fusion)" panel appears under the
Scan group. Every non-base scan gets a row with a visibility checkbox, a colormap
picker and an opacity slider. Overlays are resampled onto the base-scan grid with
trilinear interpolation (regions outside the overlay's FOV stay transparent) and are
auto-windowed to their 2-98 percentile range.

## Contouring tools
Tools > Contouring opens a CERR-style contouring panel (cf. MATLAB CERR's
`contourControl`, which also operates on the transverse axis):
- Pick **`<New structure>`** or an existing structure to edit (its current mask is
  loaded as the working copy); set/rename via the Name field.
- **Draw (add)** / **Erase**, in one of three drawing modes (cf. CERR's
  `drawContour` draw/drawBall modes):
  - **Freehand** — left-drag a region; on release it is filled into the slice mask.
  - **Polygon (line segments)** — left-click to place vertices connected by line
    segments (rubber-band preview follows the cursor); right-click or double-click
    closes the polygon. Changing slice cancels an open polygon.
  - **Brush** — paints disks of adjustable radius (cm) **in real time** as you drag
    (CERR's drawBall): the contour overlay updates live under the cursor, with a
    dotted circle showing the brush size. A single click stamps one disk. With
    **Erase**, the brush works as a live circular eraser.
  The working mask is shown filled + dashed on all three views in the structure's
  color.
- **Delete slice contour**, **Copy from superior/inferior slice** (CERR's copy
  sup/inf), and **Undo** (last 50 slice edits).
- **Save to planC** converts the mask to contours via
  `cerr.dataclasses.structure.importStructureMask` — appending a new structure or
  replacing the edited one — and refreshes the viewer's structure list.
Closing the panel discards unsaved edits (after confirmation) and restores normal
pan behavior. Contouring is locked to the scan that was active when it opened.

## IMRTP & ROE launchers
Tools > IMRTP opens the beamlet dose calculation GUI and Tools > ROE opens the
Radiotherapy Outcomes Explorer. Both windows operate on the *same* planC instance
as the viewer (changes are shared in memory). After IMRTP adds a dose (or any
external code modifies planC), use **Tools > Refresh from planC** to update the
viewer's scan/dose/structure lists without reopening it.

When IMRTP is launched from this viewer's Tools menu:
- Its **Show** button displays the computed dose directly in this viewer (selecting
  the new dose and bringing the window forward) instead of opening napari.
- Ticking a beam's **BEV** checkbox draws that beam in every view, similar to MATLAB
  CERR's beam's-eye-view. In each **2D view** the beam's *cross-section at the current
  slice* is drawn (the intersection of the divergent field pyramid with the slice
  plane), so the outline changes as you scroll and disappears on slices the beam does
  not reach. In the **3D view** the full field pyramid (source through the aperture to
  the exit plane) is rendered. Each checked beam gets its own color; unticking removes
  it, and closing IMRTP clears them all.

## Notes
- Dose is resampled onto the scan grid with trilinear interpolation, so scan and dose
  grids may differ (as in CERR).
- Structure masks are cached after first render for fast slice scrolling.
- Window center/width are remembered per scan: switching scans (or importing a
  dose/structure) restores each scan's last setting instead of resetting it.
- Non-CT scans (PET/MR) get an automatic 2–98 percentile window the first time
  they are displayed.

## Dose colorbar (update)
The colorbar is a standalone widget right of the views:
- **Yellow handles (left)**: colorbar/colormap mapping range
- **Cyan handles (right)**: dose display range (doses outside are hidden)
- **Right-click menu**: colormap picker (all 21 CERR colormaps, with previews),
  "Set ranges..." for exact numeric values, sync display-to-colorbar, reset
- **Double-click**: reset ranges

Ship `cerr_colormaps.py` next to `pycerr_gui.py` — it contains all colormaps from
MATLAB CERR's `CERRColorMap.m` (jetmod, full, full2, star, starinterp, gray, gray256,
grayud64, doublecolorinvert, thedrewspecial, graycenter0width300, ppt, coolwarm,
hotcold, copper, red, green, blue, yellow, fireice, weather). Default: `starinterp`,
matching CERR's `optS.doseColormap`.

## Programmatic use: pass planC in, get the updated planC back

```python
import cerr.plan_container as pc
from pycerr_gui import launch, show

planC = pc.loadDcmDir(r"C:\data\pat1")

# --- Blocking (plain scripts) -------------------------------------
planC = launch(planC)          # opens viewer; returns when window closes
print(len(planC.structure))    # includes anything imported in the GUI
# launch() mutates planC in place, so your original variable is also current.

# --- Non-blocking (IPython / Jupyter: run `%gui qt` first) --------
viewer = show(planC)
# ...interact with the GUI while your session stays live...
planC = viewer.planC           # current state at any time (same as getPlanC())
viewer.setPlanC(otherPlanC)    # swap in a different plan programmatically
viewer.after_load(keep_view=True)  # redraw after editing planC externally
```

`launch(planC=None, dicomDir=None)` also accepts a DICOM directory to import on
startup. The command line still works: `python pycerr_gui.py [dicom_dir]`.

## Programmatic control & screenshots
`show()` / `launch()` return the live `PyCerrViewer`, which exposes a scripting API
that drives the same controls as the GUI (each call also updates the on-screen
widgets):

```python
from cerr.viewer.pycerr_gui import show
v = show(planC)
v.set_layout("grid")                 # one/two/default/columns/grid
v.set_scan(0); v.set_window_preset("Lung")
v.set_dose(0); v.set_dose_alpha(0.4)
v.set_structures_visible([0, 2]); v.set_structure_dots(True)
v.set_orientation("B", "Axial"); v.set_slice("Axial", 42)
v.goto_structure(0)                  # center all views on a structure
v.set_crosshairs(True); v.set_lock_views(True)
```

Capture screenshots to file with `save_screenshot(path, target=...)`:
- `"window"` – the whole GUI, `"views"` – just the view panel
- `"A"/"B"/"C"/"D"` or an orientation (`"Axial"`, `"3D"`, ...) – a single view
- 2D views save the matplotlib figure (crisp, `dpi=`), the 3D view saves a
  pyvista/VTK render.

```python
v.save_screenshot("shot.png", target="window")
v.save_screenshot("axial.png", target="Axial", dpi=200)
```

For non-interactive figure generation in scripts, the module-level `capture()`
builds a viewer, applies a `setup(viewer)` callback, writes one screenshot, and
closes:

```python
from cerr.viewer.pycerr_gui import capture
def prep(v):
    v.set_layout("grid"); v.set_window_preset("Lung")
    v.set_dose(0); v.goto_structure(0)
capture(planC, "fig.png", target="window", setup=prep)
```

Headless/batch: set `os.environ['QT_QPA_PLATFORM'] = 'offscreen'` before creating
the viewer. 2D targets render reliably offscreen; the pyvista 3D view needs a real
GL context.

## Jupyter / Google Colab (headless servers)
The PyQt5 GUI needs a display, so it cannot run on headless servers. Use
`pycerr_nbviewer.py` instead — the notebook viewer (ipywidgets + matplotlib):

```python
!pip install pyCERR ipywidgets          # Colab
from pycerr_nbviewer import showNB
viewer = showNB(planC)                  # interactive viewer in the cell
planC = viewer.planC                     # live handle, same as desktop API
viewer.refresh()                         # redraw after external planC edits
```
Upload `pycerr_nbviewer.py` and `cerr_colormaps.py` next to the notebook.
See `pycerr_viewer_colab_demo.ipynb` for a ready-to-run example.

The notebook viewer mirrors most desktop features that make sense statically:
axial/sagittal/coronal views, CT window presets + manual W/L, **base-scan
colormap & opacity**, dose colorwash (all CERR colormaps) with colorbar/display
range sliders, structures (All/None, **contour line width**, **vertex dots**,
**Center** button to jump to a structure's center of mass), **patient-orientation
labels**, crosshairs, and the DVH tool with **CSV export**. It also exposes a
scripting API mirroring the desktop one:

```python
viewer.set_scan(0); viewer.set_window_preset("Lung")
viewer.set_dose(0); viewer.set_dose_alpha(0.4)
viewer.set_structures_visible([0, 2]); viewer.set_structure_dots(True)
viewer.goto_structure(0)
viewer.export_dvh("dvh.csv")             # cumulative DVHs -> CSV
viewer.save_screenshot("views.png")      # current three-view figure
```

Desktop-only features (3D VTK view, interactive contouring, IMRTP beams, drag
interactions, registration QA) are not in the notebook viewer — use the Qt GUI.

### Crosshairs (notebook viewer)
Each view shows dashed crosshairs at the slice positions of the other two views,
driven by the three slice sliders, with a readout of the 3D intersection point
(x/y/z in cm, scan value, dose). Toggle with the **Crosshairs** checkbox.
