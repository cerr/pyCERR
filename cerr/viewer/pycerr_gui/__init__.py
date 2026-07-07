"""
pyCERR Viewer GUI
=================
A QT-based desktop viewer built on pyCERR.

Features (mirroring the classic CERR slice viewer):
  * Import DICOM directory / NIfTI scan, dose & structure files / planC pickle
  * Axial, sagittal and coronal linked views with slice sliders & mouse wheel
  * Pan (drag) and zoom (right-drag up/down) per view; double-click / R to reset
  * Right-click menu per view: choose the orientation (Axial/Sagittal/Coronal)
    plus the scan, dose & structures it displays (CERR-style per-axis
    View/ScanSet/DoseSet/StructSet selection)
  * Lock slices across matching views (View menu or L): windows showing the
    same orientation scroll together
  * View > Layout: one view, two side-by-side, one large + two stacked,
    three equal columns, or four equal views in a 2x2 grid
  * 3D view (default 4th window of the 2x2 layout, or right-click > View >
    3D): shows the three orthogonal planes selected in the 2D views,
    textured with the scan and outlined in the plane-locator colors.
    GPU-accelerated via pyvista/VTK when installed (pip install pyvista
    pyvistaqt; falls back to matplotlib otherwise); rotate with left-drag,
    zoom with right-drag or wheel, pan with middle-drag. The planes are
    DRAGGABLE: grab one with the left button to scroll through slices -
    the linked 2D views follow live. Also shows structure surfaces (from
    the Structures checklist) and 30/50/70/90% isodose surfaces (from the
    Dose combo, colored by the dose colorbar)
  * Ruler tool per view (right-click > Draw Ruler): drag to measure distances
    in cm, CERR-style; toggle again to clear
  * Contouring tools (Tools > Contouring): CERR-style draw/erase of structure
    contours on the axial view - freehand, point-by-point polygon (line
    segments, right/double-click to close) and disk brush modes - with
    per-slice delete/copy, undo, and save back to planC via
    importStructureMask (new or edited structures)
  * Crosshairs linked across views (View menu or X to toggle)
  * Patient-orientation labels (L/R/A/P/S/I) at the edges of each 2D view
    and an orientation triad in the 3D view (View menu or O to toggle)
  * Multi-scan fusion: overlay any loaded scan with per-scan opacity & colormap
  * Scan Display dialog (panel button or View menu): per-scan window/level
    presets (Soft Tissue, Lung, Bone, Brain, ...) + manual W/L, colormap,
    opacity, and a scan colorbar with draggable colormap-mapping (window) and
    data display ranges (right-click for colormap/exact ranges). The main
    figure shows just the base scan and fused overlays.
  * Structure contour overlays with per-structure colors & visibility toggles,
    optional contour vertex dots ("Points"), adjustable contour line width, and
    double-click a structure to center all three views on it
  * Dose colorwash overlay with adjustable transparency, colorbar & dose units
  * Crosshair readout: position (cm), scan value (HU/SUV), dose (Gy)
  * DVH tool (cumulative dose-volume histograms, CERR-style)
  * Registration QA tool (Tools menu): compare a base and a moving scan via
    Mirrorscope (moving mirrored about a draggable line), Side-by-side,
    Alternate grid (checkerboard) and Toggle/blend (base-moving cross-fade
    slider) displays in every 2D view; each scan uses its own colormap
  * Tools menu launchers for the IMRTP GUI (beamlet dose calculation) and the
    ROE GUI (Radiotherapy Outcomes Explorer), sharing this viewer's planC.
    IMRTP shows its computed dose in this viewer and draws beam's-eye-view
    overlays (checked beams): the per-slice field cross-section in each 2D
    view (updates as you scroll) and the full field pyramid in the 3D view

Requirements:
  pip install pyCERR PyQt5 matplotlib scipy
  -- or add your local pyCERR checkout to sys.path (see PYCERR_PATH below).

Run:
  python -m cerr.viewer.pycerr_gui [optional: path to DICOM directory]
"""

from cerr.viewer.pycerr_gui.common import *  # noqa: F401,F403
from cerr.viewer.pycerr_gui.slice_view import SliceView  # noqa: F401
from cerr.viewer.pycerr_gui.colorbars import (RangeColorbarWidget,  # noqa: F401
    DoseColorbarWidget, ScanColorbarWidget)
from cerr.viewer.pycerr_gui.main_window import PyCerrViewer  # noqa: F401
from cerr.viewer.pycerr_gui.dialogs import (DvhDialog, ContourDialog,  # noqa: F401
    ScanDoseExportDialog, StructureExportDialog, RegQaDialog)
from cerr.viewer.pycerr_gui.uromt_gui import (_UROMTWorker, UROMTDialog,  # noqa: F401
    UROMTViewDialog)
from cerr.viewer.pycerr_gui.volume3d import Volume3DDialog  # noqa: F401
from cerr.viewer.pycerr_gui.app import (launch, show, capture, main,  # noqa: F401
    _get_app)
