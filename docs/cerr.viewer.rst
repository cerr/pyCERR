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

cerr.viewer.pycerr\_gui module
------------------------------

.. automodule:: cerr.viewer.pycerr_gui
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
