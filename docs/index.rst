.. pyCERR documentation master file, created by
   sphinx-quickstart on Fri May 31 16:31:02 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyCERR documentation
====================

pyCERR (Python-based Computational Environment for Radiological Research) stores
all data for a patient in a single ``PlanC`` container — scans, structures
(segmentations), dose, treatment plans (beams) and deformations — defined in
``cerr.plan_container``.

Features
--------

**Data import / export**

* DICOM import of CT/MR/PT/US/NM scans plus RTSTRUCT, RTDOSE and RTPLAN (``cerr.plan_container.loadDcmDir``)
* NIfTI import/export of scans, segmentations and dose
* Full-``PlanC`` HDF5 serialization (``saveToH5`` / ``loadFromH5``)
* DICOM export of RTSTRUCT, image series (CT/MR/PT/US/NM, or Secondary Capture
  for derived scans), RTDOSE and RTPLAN (``cerr.dcm_export``)

**Segmentation & contours**

* Lazy polygon → binary-mask rasterization (``cerr.contour.rasterseg.getStrMask``)
* Import label maps / binary masks as structures (``cerr.dataclasses.structure.importStructureMask``)

**Radiomics (IBSI-compliant)**

* Scalar features: morphology, first-order, GLCM / GLRLM / GLSZM / GLDZM / NGTDM / NGLDM (IBSI-1)
* Convolutional texture / filter-response maps: mean, LoG, Laws, Gabor, wavelet (IBSI-2)
* ``cerr.radiomics.ibsi1.computeScalarFeatures``

**Dosimetry & outcomes**

* Dose–volume histograms and metrics — Dx, Vx, MOHx, MOCx, mean dose (``cerr.dvh``)
* Radiotherapy outcome models — NTCP/TCP: LKB, logistic, Cox, Appelt (``cerr.roe``)
* IMRT planning / beamlet dose calculation (``cerr.imrtp``)

**Image processing**

* Deformable image registration via plastimatch / ANTs (``cerr.registration``)
* Resampling, intensity preprocessing and masking (``cerr.utils``)
* Semi-quantitative DCE-MRI features (``cerr.mri_metrics``)
* Helpers for AI model training / inference (``cerr.utils.ai_pipeline``)

**Visualization** — three interchangeable viewers driven by the same ``planC``
(``cerr.viewer``):

* ``pycerr_napari`` — napari 2D/3D viewer
* ``pycerr_gui`` — PyQt5 CERR-style desktop viewer: linked ortho views, per-scan
  window/colormap/opacity, colorwash or isodose-line dose display, a 3D volume
  render with a draggable clip box, contouring, DVH and Registration QA
* ``pycerr_nbviewer`` — Jupyter / Colab notebook viewer

.. toctree::
   :maxdepth: 2
   :caption: cerr and its sub-packages

   cerr
   cerr.dataclasses
   cerr.contour
   cerr.radiomics
   cerr.dcm_export
   cerr.registration
   cerr.utils
   cerr.viewer
   cerr.imrtp
   cerr.roe
   cerr.mri_metrics
   cerr.datasets
   tests

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


