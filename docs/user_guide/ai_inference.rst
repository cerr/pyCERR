AI model inference
==================

pyCERR runs pretrained deep-learning segmentation models on scans in a
``planC`` and brings the results back as structures. It reads
DICOM or NIfTI images, pre-processes them to the input the model was trained on, maps the
output back onto the original scan, and exports it.

There are two ways to use it:

* **Models from the CERR model library**, through ``cerr.ai_models``.
  Each model repository (such as `SMIT <https://github.com/cerr/smit>`_) is
  cloned with its own Python environment, and its weights are installed
  separately.
* **Your own model.** The same pre- and post-processing tools work around any
  network you can call from Python or the command line.

This page covers both, then the pre- and post-processing options in detail.

The CERR model library
----------------------------------

Requirements
~~~~~~~~~~~~

Installing a model needs ``git`` and `uv <https://docs.astral.sh/uv/>`_ on
the ``PATH``. Model weights are not in the repositories; request a download
link for each model you need from the model maintainers.

List, install and inspect
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cerr import ai_models
   from cerr.ai_models import install_utils, run_utils

   models = install_utils.listModels()        # prints and returns {number: name}

   model = 'CT_ThoraxOARs_SMITplus'           # or its number: 14
   installDir = '/path/to/models'
   ai_models.install(model, installDir,
                     weightsUrl='<link you received>')  # or weightsFile='/path/to/download'

   run_utils.listInputs(model, installDir)    # inputs this model expects

Model names are maintained in ``cerr/ai_models/model_list.py``.

:func:`ai_models.install <cerr.ai_models.install_utils.main>` performs three
steps:

1. **Clone.** The model's repository is cloned to ``<installDir>/<modelName>``, or
   updated with ``git pull`` if it is already there.
2. **Environment.** A uv environment is built in ``<installDir>/<modelName>/.venv``
   with the Python version in the repository's ``uv_config.txt``, then every
   ``requirements/*.txt`` is installed in name order with the ``uv_flags``
   from ``uv_config.txt``. Its dependencies (PyTorch, model code) never touch
   the environment pyCERR runs in. On macOS, PyTorch is installed from PyPI,
   since it has no CUDA builds there.
3. **Weights.** The repository's ``get_weights.py`` downloads (or copies)
   the weights to ``<installDir>/<modelName>/weights/<weightsFile>.safetensors`` and
   checks them against the checksum in ``models.yaml``.

Models of one repository share the clone and the environment, so installing
a second SMIT model only adds its weights. Pass ``forceWeights=True`` to
replace installed weights, ``rebuildEnv=True`` to rebuild the environment,
and ``credentials='user:token'`` for a private repository.

Run
~~~

.. code-block:: python

   userInputs = {'input_path': '/data/pt1/dicom',      # DICOM dir, folder of NIfTI files, or a NIfTI file
                 'output_path': '/data/pt1/ai_output',
                 'session_path': '/data/pt1/session'}  # optional scratch space
   ai_models.run(model, installDir, userInputs, verbose=True)

:func:`ai_models.run <cerr.ai_models.run_utils.main>` runs the model in its
environment as

.. code-block:: bash

   run_segmentation.py --model <modelName> --data_dir <input_path> --results_dir <output_path>

and the model writes, for each case, ``<case>_seg.nii.gz`` (a label map on the
input scan's grid) and ``labels.json`` (label value to structure name) to
``output_path``; for DICOM input also ``<case>_<modelName>_RTSTRUCT.dcm``.
A single NIfTI file is first copied to ``<session_path>/input``. If the
model's entrypoint defines a ``--session_dir`` option, ``session_path`` is
passed to it`; otherwise it is used only by pyCERR.

Additional keys in ``userInputs`` are passed to the model as
``--<key> <value>`` (``True`` passes ``--<key>`` alone), for example
``'use_tta': True`` or ``'device': 'cpu'``. ``'gpu': 1`` selects a GPU
through ``CUDA_VISIBLE_DEVICES``. ``listInputs`` shows the options a model
accepts.

To import a model's outputs into a ``planC`` you already hold:

.. code-block:: python

   planC = run_utils.importSeg(model, installDir, '/data/pt1/ai_output', scanNum, planC)

Pass ``caseName`` when ``output_path`` holds the results of several cases.

A model that writes several label maps per case lists them in the ``labels``
of its ``models.yaml`` entry (or ``labels.json``);``importSeg`` loads all of them, ``.dcm`` files as RTSTRUCTs and ``.nii.gz``
files with the pattern's labels.

Pre- and post-processing for a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``cerr/ai_models/prep/<modelName>.py`` or
``cerr/ai_models/post/<modelName>.py`` exists, ``run`` calls it before or after
the model (``input_path`` is then one case: a DICOM directory or a NIfTI file).
The post-processed segmentation is written to ``output_path`` in place of the
model's output, under the same file names and in the same format:
``<case>_seg.nii.gz`` and ``labels.json``, or ``<case>_<modelName>_RTSTRUCT.dcm``
for DICOM input. With pre-processing only, the model's output is copied from the
session directory to ``output_path``. Either way ``importSeg`` reads
``output_path`` as for any other model.

A pre-processing module defines ``REQUIRED_INPUTS`` and
``processInputData``:

.. code-block:: python

   REQUIRED_INPUTS = {'input_path': {'required': True},
                      'output_path': {'required': True},
                      'session_path': {'required': True}}

   def processInputData(userInputs):
       """Load the input, prepare the scan, write the model input to
       <session_path>/input. userInputs also contains 'model_dir'."""
       ...
       return planC, procScanNum, scanNum, sessionUserInputs

``scanNum`` is the original scan, ``procScanNum`` the prepared one, and
``sessionUserInputs`` the inputs passed on to the model (typically with
``input_path`` and ``output_path`` pointing into the session directory).

A post-processing module defines ``postProcAndImportSeg``:

.. code-block:: python

   def postProcAndImportSeg(planC, procScanNum, scanNum, userInputs, outDir):
       """Import the model output from outDir, map it to scanNum and clean it up."""
       ...
       return planC, structNumV

``structNumV`` lists the output structures; ``run`` writes those whose names are
in the model's labels.

``cerr/ai_models/prep/MR_Rectum_GTV_SMIT.py`` is a complete, short example.

Running your own model
----------------------

Any model that reads NIfTI and writes a label map fits this pattern:

.. code-block:: python

   import subprocess
   from cerr import plan_container as pc
   from cerr.utils import image_proc
   from cerr.dataclasses import structure as cerrStr

   planC = pc.loadDcmDir('/data/pt1/dicom')
   scanNum = 0
   scan3M = planC.scan[scanNum].getScanArray()
   gridS = planC.scan[scanNum].getScanXYZVals()

   # 1. Prepare the input and keep it in planC with its grid
   net3M, _, netGridS = image_proc.resizeScanAndMask(scan3M, None, gridS,
                                                     [512, 512, scan3M.shape[2]],
                                                     'padorcrop3d')
   planC = pc.importScanArray(net3M, *netGridS, 'CT', scanNum, planC)
   procScanNum = len(planC.scan) - 1
   planC.scan[procScanNum].saveNii('/tmp/session/input/scan.nii.gz')

   # 2. Run the model
   subprocess.run(['python', 'predict.py', '/tmp/session/input', '/tmp/session/output'],
                  check=True)

   # 3. Import the output on the prepared scan, then map it to the original
   labels = {'Heart': 1, 'Esophagus': 2}
   n0 = len(planC.structure)
   planC = pc.loadNiiStructure('/tmp/session/output/scan.nii.gz', procScanNum, planC, labels)
   for structNum in range(n0, len(planC.structure)):
       planC = cerrStr.copyToScan(structNum, scanNum, planC)

:func:`cerr.utils.ai_pipeline.createSessionDir` creates uniquely named
session input and output folders for batch runs, and
:func:`cerr.utils.ai_pipeline.getScanNumFromIdentifier` picks the right scan
from a study by modality, series description or date:

.. code-block:: python

   from cerr.utils.ai_pipeline import getScanNumFromIdentifier
   scanNum = getScanNumFromIdentifier({'imageType': 'CT SCAN'}, planC)[0]

Pre-processing options
----------------------

A network expects a fixed matrix size, often a particular field of view,
orientation and intensity range. The tools below produce that input **and a
coordinate grid for it**, so the prepared scan can be imported into ``planC``
and lines up with the original. That is what lets the model output be mapped
back.

.. figure:: ../_static/img/ai_preprocessing.png
   :alt: The head and neck CT as acquired, padded to 512 by 512, cropped to the body and resized to 256 by 256, and as a sagittal stack.
   :width: 100%

   The head and neck CT prepared three ways, each drawn in its own grid with
   the body outline of the original scan (green). The outline lines up in
   every panel. Right: the prepared scan rearranged into sagittal slices.

Resize
~~~~~~

:func:`~cerr.utils.image_proc.resizeScanAndMask` changes the matrix size:

.. code-block:: python

   from cerr.utils import image_proc

   out3M, outMask4M, outGridS = image_proc.resizeScanAndMask(
       scan3M, mask4M, gridS, [nRows, nCols, nSlices], method)

``mask4M`` is optional (pass ``None``) and holds one binary mask per
structure along the fourth axis. ``gridS`` is ``getScanXYZVals()`` of the
scan.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - ``method``
     - Result
   * - ``padorcrop3d``
     - Pads or crops rows and columns symmetrically to the output size, each
       axis independently; voxel size is unchanged. Padding uses the scan
       minimum. The usual choice when the network takes, say, 512 × 512.
   * - ``pad3d``, ``unpad3d``
     - The same, but only padding or only cropping. ``unpad3d`` undoes
       ``pad3d``.
   * - ``bilinear``, ``bicubic``, ``nearest``
     - Interpolates each slice to the output rows and columns, changing the
       voxel size. Masks are always resized with ``nearest``.
   * - ``pad2d``, ``unpad2d``
     - Cuts a fixed-size window centred on a per-slice bounding box
       (``limitsM``), and the inverse.
   * - ``padslices``, ``unpadslices``
     - Adds or removes slices at the end.

With ``preserveAspectFlag=True``, ``padorcrop3d`` and the interpolating
methods first pad the image to a square with the background intensity, so
anatomy is not stretched.

.. code-block:: python

   # crop to the body, then resize to the network input without distortion
   from cerr.utils import mask as maskUtils

   body3M = maskUtils.getPatientOutline(scan3M)
   rMin, rMax, cMin, cMax, sMin, sMax, _ = maskUtils.computeBoundingBox(body3M, maskFlag=5)
   crop3M = scan3M[rMin:rMax + 1, cMin:cMax + 1, sMin:sMax + 1]
   cropGridS = (gridS[0][cMin:cMax + 1], gridS[1][rMin:rMax + 1], gridS[2][sMin:sMax + 1])
   net3M, _, netGridS = image_proc.resizeScanAndMask(crop3M, None, cropGridS,
                                                     [256, 256, crop3M.shape[2]],
                                                     'bilinear', preserveAspectFlag=True)

:func:`~cerr.utils.mask.computeBoundingBox` returns inclusive limits, hence
the ``+ 1``; ``maskFlag`` adds a margin in voxels. Note the grid slicing:
``xV`` runs along columns and ``yV`` along rows.

Other preparation
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Need
     - Tool
   * - Field of view: crop to the body or to an organ
     - :func:`cerr.utils.mask.getPatientOutline` (:doc:`body_outline`),
       :func:`cerr.utils.mask.computeBoundingBox`
   * - Orientation: feed sagittal or coronal slices to a 2D network
     - :func:`cerr.utils.image_proc.transformScan` with ``'sagittal'`` or
       ``'coronal'``; it transposes the scan, masks and grid together.
   * - Voxel size: resample to a fixed spacing
     - :func:`cerr.radiomics.preprocess.imgResample3D` with
       :func:`cerr.radiomics.preprocess.getResampledGrid`
   * - Intensity: clip and normalise
     - NumPy on the scan array, for example ``np.clip(scan3M, -1000, 1000)``;
       :func:`cerr.radiomics.preprocess.calcRobustZscore` for MR
   * - Several inputs: fixed and moving scans, or multi-parametric MR
     - :doc:`registration` to bring them onto one grid first
   * - Choose the scan from a study
     - :func:`cerr.utils.ai_pipeline.getScanNumFromIdentifier`

Post-processing options
-----------------------

A raw network output usually needs three things: mapping back onto the
original scan, removing stray voxels, and closing small gaps.

.. figure:: ../_static/img/ai_postprocessing.png
   :alt: A segmentation with small false-positive islands and a hole, then after keeping the largest component, then after closing.
   :width: 100%

   A stand-in model output for the phantom GTV, produced on a 128 × 128
   version of the scan with eight false-positive islands and a hole, then
   copied to the original 201 × 204 grid (left). Keeping the largest connected
   component removes the islands (centre); closing fills the hole (right).
   The reference GTV is outlined in cyan.

.. code-block:: python

   from cerr.dataclasses import structure as cerrStr

   # Model output on the prepared scan -> structure on the original scan
   planC = pc.importStructureMask(pred3M, procScanNum, 'AI_GTV', planC)
   planC = cerrStr.copyToScan(len(planC.structure) - 1, scanNum, planC)
   aiNum = len(planC.structure) - 1

   # Keep the largest connected component, then close gaps up to 1 cm
   _, planC = cerrStr.getLargestConnComps(aiNum, 1, planC=planC, saveFlag=True,
                                          procSructName='AI_GTV_largest')
   _, planC = cerrStr.getClosedMask(len(planC.structure) - 1, 1.0, planC, saveFlag=True,
                                    procSructName='AI_GTV_final')

.. code-block:: text

   AI_GTV           Dice vs GTV 0.982  components 7
   AI_GTV_largest   Dice vs GTV 0.987  components 1
   AI_GTV_final     Dice vs GTV 0.982  components 1

Closing fills the hole but also the concave notches along the edge, so it can
lower Dice against a contour that follows those notches. Choose the element
size to match the gaps you expect.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Step
     - Tool
   * - Map to the original scan
     - :func:`cerr.dataclasses.structure.copyToScan`, or
       :func:`~cerr.utils.image_proc.resizeScanAndMask` with the inverse method
       on the mask array
   * - Remove islands
     - :func:`cerr.dataclasses.structure.getLargestConnComps`
       (``numConnComponents=2`` for paired organs)
   * - Fill holes and gaps
     - :func:`cerr.dataclasses.structure.getClosedMask`
   * - Smooth jagged edges
     - :func:`~cerr.dataclasses.structure.getGaussianBlurredMask`,
       :func:`~cerr.dataclasses.structure.getBsplineSmoothing`; see
       :doc:`smoothing`
   * - Clip to the body or another structure
     - :func:`~cerr.dataclasses.structure.structIntersect`
   * - Combine several models
     - :doc:`structure_consensus`
   * - Score against a reference
     - :doc:`segmentation_metrics`
   * - Export
     - :func:`cerr.dcm_export.rtstruct_iod.create`,
       :func:`cerr.plan_container.saveNiiStructure`

See also
--------

* SMIT model repository: https://github.com/cerr/smit
* Notebooks: ``03_autosegmentation`` in https://github.com/cerr/pyCERR-Notebooks.
* API reference: :doc:`../cerr.ai_models`, :doc:`../cerr.utils`.
* Figure script: ``docs/_figure_scripts/ai_pipeline.py``.
