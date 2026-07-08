cerr.viewer package
===================

Visualization sub-package. Three interchangeable viewers share the same planC
data model:

* ``pycerr_napari`` - napari-based 2D/3D viewer and magicgui widgets.
* ``pycerr_gui`` - PyQt5 desktop slice viewer (CERR-style), with contouring,
  registration QA, IMRTP/ROE launchers and a programmatic control API.
* ``pycerr_nbviewer`` - ipywidgets + matplotlib viewer for Jupyter/Colab.

For backward compatibility the public napari API is also reachable directly on
the package, e.g. ``cerr.viewer.showNapari(...)``; this is resolved lazily so
that importing ``pycerr_gui`` / ``pycerr_nbviewer`` does not pull in napari.

Submodules
----------

cerr.viewer.pycerr\_napari module
---------------------------------

.. automodule:: cerr.viewer.pycerr_napari
   :members:
   :undoc-members:
   :show-inheritance:

cerr.viewer.pycerr\_gui package
-------------------------------

PyQt5 CERR-style desktop viewer. Organized as a package; the public API
(``launch``, ``show``, ``capture``, ``PyCerrViewer`` and the dialogs) is
re-exported from ``cerr.viewer.pycerr_gui`` for backward compatibility, and
``python -m cerr.viewer.pycerr_gui [dicomDir]`` launches it.

Highlights:

* Linked axial / sagittal / coronal slice views with per-axis scan, dose and
  structure selection, crosshairs, orientation labels and configurable layouts.
* Per-scan display settings (window/level, colormap, opacity) managed in the
  Scan Display dialog and shared by tools such as Registration QA; multi-scan
  fusion overlays.
* Dose display as a colorwash or as CERR-style isodose lines with selectable
  levels (% of max, % of prescription, or Gy) and matching colorbar tick marks.
* 3D visualization dialog: GPU volume render with independent scan / dose /
  structure opacity, a resolution control (1/4 – full), a draggable clip box
  confined to a dotted scan-bounds outline, and structure / isodose surfaces.
* Contouring tools, DVH, Registration QA, and IMRTP / ROE launchers.
* NIfTI import that prompts for scan / dose / segmentation and auto-associates
  dose and label masks with the scan whose grid matches the file.

.. automodule:: cerr.viewer.pycerr_gui
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cerr.viewer.pycerr_gui.main_window
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cerr.viewer.pycerr_gui.slice_view
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cerr.viewer.pycerr_gui.colorbars
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cerr.viewer.pycerr_gui.dialogs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cerr.viewer.pycerr_gui.volume3d
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cerr.viewer.pycerr_gui.uromt_gui
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cerr.viewer.pycerr_gui.app
   :members:
   :undoc-members:
   :show-inheritance:

cerr.viewer.pycerr\_nbviewer module
-----------------------------------

.. automodule:: cerr.viewer.pycerr_nbviewer
   :members:
   :undoc-members:
   :show-inheritance:

cerr.viewer.cerr\_colormaps module
----------------------------------

.. automodule:: cerr.viewer.cerr_colormaps
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: cerr.viewer
   :members:
   :undoc-members:
   :show-inheritance:
