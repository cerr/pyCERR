# pyCERR - A Python-based Computational Environment for Radiological Research

pyCERR provides a convenient data structure for radiological imaging metadata and their associations: scans, segmentations, radiotherapy dose, treatment plans and deformable registrations for a patient are held together in a single container. Utilities are provided to extract, transform and organize this metadata, and to visualize the results of image processing.

On this foundation, pyCERR supports a wide range of applications in radiology and radiation oncology research:

- **Image analysis**: IBSI-compliant radiomics and texture maps, PET SUV computation, DCE-MRI uptake features and optimal mass transport (urOMT) analysis
- **Segmentation**: contouring, structure operations, consensus contours and auto-segmentation with pre-trained AI models
- **Registration**: deformable image registration (plastimatch, ANTs) with tools for registration quality assurance
- **Dosimetry and outcomes**: dose-volume histograms and metrics, gamma dose comparison, and normal tissue complication / tumor control probability models
- **Treatment planning**: IMRT dose calculation and fluence optimization, including lattice radiotherapy
- **AI workflows**: data preparation for model training and inference, on a workstation or in the cloud

## Quick Start
Create a virtual environment with [uv](https://docs.astral.sh/uv/), install pyCERR
with the desktop viewer, and launch Python:
```bash
uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install "pycerr[viewer]"
python
```

In Python, load an example dataset and show it in the desktop viewer:
```python
from cerr import plan_container as pc
from cerr.viewer.pycerr_gui import show
from cerr import datasets

dcmDir = datasets.fetch_sample_data('head_and_neck')   # local if present, else downloaded
planC = pc.loadDcmDir(dcmDir)
v = show(planC)
```

The desktop viewer needs the `viewer` extra. A plain `pycerr` install covers
everything else, including the Jupyter viewer; see [Install pyCERR](#install-pycerr).

## Features

All data for a patient lives in a single `PlanC` container — scans, structures
(segmentations), dose, treatment plans (beams) and deformations — defined in
`cerr.plan_container`. Around it, pyCERR provides:

**Data import / export**
- DICOM import of CT/MR/PT/US/NM scans plus RTSTRUCT, RTDOSE and RTPLAN — `pc.loadDcmDir`
- NIfTI import/export of scans, segmentations and dose — `pc.loadNiiScan` / `loadNiiStructure` / `loadNiiDose`, `.saveNii`
- Full-`PlanC` HDF5 serialization — `pc.saveToH5` / `pc.loadFromH5`
- DICOM RTSTRUCT export — `cerr.dcm_export.rtstruct_iod`

**Segmentation & contours**
- Lazy polygon → binary-mask rasterization — `cerr.contour.rasterseg.getStrMask`
- Import label maps / binary masks as structures — `cerr.dataclasses.structure.importStructureMask`

**Radiomics (IBSI-compliant)**
- Scalar features: morphology, first-order, GLCM / GLRLM / GLSZM / GLDZM / NGTDM / NGLDM (IBSI-1)
- Convolutional texture / filter-response maps: mean, LoG, Laws, Gabor, wavelet (IBSI-2)
- `cerr.radiomics.ibsi1.computeScalarFeatures`, configured via JSON settings

**Dosimetry & outcomes**
- Dose–volume histograms and metrics (Dx, Vx, MOHx, MOCx, mean dose) — `cerr.dvh`
- Radiotherapy outcome models (NTCP/TCP: LKB, logistic, Cox, Appelt) — `cerr.roe`
- IMRT planning / beamlet (influence-matrix) dose calculation — `cerr.imrtp`

**Image processing**
- Deformable image registration via plastimatch / ANTs — `cerr.registration`
- Resampling, intensity preprocessing and masking — `cerr.utils`
- Semi-quantitative DCE-MRI features — `cerr.mri_metrics`
- Helpers for AI model training / inference pipelines — `cerr.utils.ai_pipeline`

**Visualization** — three interchangeable viewers driven by the same `planC`
(a PyQt5 CERR-style desktop GUI, a Jupyter/Colab notebook viewer, and
napari 2D/3D); see [visualize scan, dose and segmentation](#visualize-scan-dose-and-segmentation) below.

## Documentation

- **Example notebooks** (grouped by topic): https://github.com/cerr/pyCERR-Notebooks
- **API reference** — https://pycerr.readthedocs.io (built from `docs/`; build locally with `cd docs && make html`)
- **Desktop GUI scripting API** — [`cerr/viewer/API_pycerr_gui.md`](https://github.com/cerr/pyCERR/blob/main/cerr/viewer/API_pycerr_gui.md)
- **Test suite & coverage** — [`tests/README.md`](https://github.com/cerr/pyCERR/blob/main/tests/README.md)

## Install pyCERR

pyCERR is published on PyPI as [`pycerr`](https://pypi.org/project/pycerr/) and
supports Python 3.9–3.13 (3.12 recommended). Install it into an isolated
virtual environment. The steps below use [uv](https://docs.astral.sh/uv/getting-started/installation/);
install it first if you don't have it (`curl -LsSf https://astral.sh/uv/install.sh | sh`
on macOS/Linux, `winget install --id=astral-sh.uv` on Windows, or `brew install uv`).

### 1. Create and activate a virtual environment
```bash
uv venv --python 3.12              # creates .venv/, downloading Python 3.12 if needed
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install pyCERR with the extras you need
```bash
uv pip install pycerr                      # core: I/O, radiomics, DVH, registration, Jupyter viewer
uv pip install "pycerr[viewer]"            # + PyQt5 desktop viewer (pycerr_gui), ROE and IMRTP GUIs
uv pip install "pycerr[viewer,napari]"     # + napari 2D/3D viewer
```

| Extra | Adds | Needed for |
|-------|------|------------|
| *(none)* | core dependencies, `ipywidgets` | Python API, headless/batch processing, containers, `cerr.viewer.pycerr_nbviewer` (Jupyter / Colab) |
| `viewer` | `PyQt5`, `pyvista`, `pyvistaqt` | `cerr.viewer.pycerr_gui` desktop viewer (incl. 3D view), ROE and IMRTP GUIs |
| `napari` | `napari[all]`, `napari-animation` | `cerr.viewer.pycerr_napari` |
| `ants` | `antspyx` | ANTs-based registration, `cerr.registration.ants_reg` |
| `gpu` | `cupy-cuda13x` and NVIDIA CUDA 13 runtime wheels | optional GPU acceleration of the urOMT solver |

Extras can be combined, e.g. `uv pip install "pycerr[viewer,napari,ants]"`.
The `viewer` extra is not in the base install because PyQt5 has no aarch64
Linux wheel; on arm64 Linux (ARM servers, Apple Silicon Docker images) install
plain `pycerr`. Launching `pycerr_gui` without the extra raises an ImportError
naming the install command.

AI segmentation models (`cerr.ai_models`) additionally need `model_installer`,
which is distributed from GitHub only:
```bash
uv pip install "model_installer @ git+https://github.com/cerr/model_installer.git"
```

### Install the latest development version
```bash
uv pip install "pycerr[viewer] @ git+https://github.com/cerr/pyCERR.git@main"
```
or, to work on pyCERR itself, clone the repository and make an editable install:
```bash
git clone https://github.com/cerr/pyCERR.git
cd pyCERR
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[viewer,napari]" pytest
```

### Jupyter
Install JupyterLab in the same environment to run the example notebooks:
```bash
uv pip install jupyterlab
jupyter lab
```
In Google Colab, install with `!pip install pycerr` (the notebook viewer needs no extra).

## Example Notebooks
Example notebooks are hosted at https://github.com/cerr/pyCERR-Notebooks/. Clone this repository to use notebooks as a starting point.
```bash
git clone https://github.com/cerr/pyCERR-Notebooks.git
```

## Example snippets

Run Python from the virtual environment created above and try out the following code samples.

### Import modules for planC and viewer
    import numpy as np
    from cerr import plan_container as pc
    from cerr.viewer import pycerr_gui           # desktop viewer (pycerr_gui.show, ...); needs pycerr[viewer]

### Read DICOM directory contents to planC
    dcmDir = r"\\path\to\Data\dicom\directory"
    planC = pc.loadDcmDir(dcmDir)
    
### Read NifTi scan to planC
    scanNiiFileName = r"\\path\to\Data\scan.nii.gz"
    planC = pc.loadNiiScan(scanNiiFileName, imageType = "CT SCAN")

### Read NifTi scan in a specified orientation to planC
    planC = pc.loadNiiScan(scanNiiFileName, imageType = "CT SCAN", direction='LPS')   # 3-letter orientation code, e.g. 'LPS', 'RAS'

### Read NifTi scan and append to an existing planC
    planC = pc.loadNiiScan(scanNiiFileName, imageType = "CT SCAN", direction='LPS', initplanC=planC)
    
### Read NifTi segmentation to planC
    structNiiFileName = r"\\path\to\Data\structure.nii.gz"
    assocScanNum = 0
    labelDict = {'GTV_P': 1, 'GTV_N': 2}    # structure name -> label value
    planC = pc.loadNiiStructure(structNiiFileName, assocScanNum, planC, labelDict)

### Export Structures to DICOM
    from cerr.dcm_export import rtstruct_iod

    structDcmFileName = r"\\path\to\Data\structure.dcm"
    structNums = [0,2,3]
    exportOpts = {'seriesDescription': "Exported by pyCERR"}
    rtstruct_iod.create(structNums,structDcmFileName,planC,exportOpts)

### Export Scan, Structure and Dose to NifTi
    scanNiiFileName = r"\\path\to\Data\scan.nii.gz"
    scanNum = 0
    planC.scan[scanNum].saveNii(scanNiiFileName)
    
    structNiiFileName = r"\\path\to\Data\structure.nii.gz"
    structNum = 0
    planC.structure[structNum].saveNii(structNiiFileName, planC)    
    
    doseNiiFileName = r"\\path\to\Data\dose.nii.gz"
    doseNum = 0
    planC.dose[doseNum].saveNii(doseNiiFileName)
    

### visualize scan, dose and segmentation
pyCERR ships three interchangeable viewers under the `cerr.viewer` sub-package,
all driven by the same `planC`:

| Viewer | Module | Install | Best for |
|--------|--------|---------|----------|
| PyQt5 desktop | `cerr.viewer.pycerr_gui` (`show`) | `pycerr[viewer]` | CERR-style slice viewer: contouring, registration QA, IMRTP/ROE, scripting API |
| notebook | `cerr.viewer.pycerr_nbviewer` (`showNB`) | `pycerr` | Jupyter / JupyterLab / VS Code / Google Colab |
| napari 2D/3D | `cerr.viewer.pycerr_napari` (`showNapari`) | `pycerr[napari]` | quick interactive review, 3D rendering |


#### PyQt5 desktop viewer
    from cerr.viewer import pycerr_gui
    viewer = pycerr_gui.show(planC)            # opens the CERR-style desktop GUI
    # ... or launch empty and drag-and-drop a DICOM directory / NIfTI file in.
    # The viewer exposes a scripting API (set_scan/set_dose/goto_structure,
    # registration-QA setup, DVH export, save_screenshot, ...); see
    # cerr/viewer/API_pycerr_gui.md for the full reference.

#### Notebook viewer (Jupyter / Colab)
    from cerr.viewer import pycerr_nbviewer
    viewer = pycerr_nbviewer.showNB(planC, scan_nums=[0], struct_nums=strNumList,
                                     dose_nums=[0])

#### napari viewer
    from cerr.viewer import pycerr_napari
    scanNumList = [0]
    doseNumList = [0]
    numStructs = len(planC.structure)
    strNumList = np.arange(numStructs)
    displayMode = '2d' # '2d' or '3d'
    vectDict = {}
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                   pycerr_napari.showNapari(planC, scan_nums=scanNumList, struct_nums=strNumList,\
    	       dose_nums=doseNumList, vectors_dict=vectDict, displayMode = '2d')

### Compute DVH-based metrics
    from cerr import dvh
    structNum = 0
    doseNum = 0
    dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)
    binWidth = 0.025
    doseBinsV,volHistV = dvh.doseHist(dosesV, volsV, binWidth)
    percent = 70
    dvh.MOHx(doseBinsV,volHistV,percent)

### Compute radiomics
    import os
    from cerr import datasets
    from cerr.radiomics import ibsi1

    scanNum = 0
    structNum = 0
    # JSON settings file: resampling, intensity discretization and feature classes.
    # A sample is shipped with pyCERR; copy and edit it for your own analysis.
    settingsFile = os.path.join(os.path.dirname(datasets.__file__),
                                'radiomics_settings', 'original_settings.json')
    featDict, diagDict = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)
    # featDict maps feature names to values (IBSI-1: shape, first-order, GLCM, GLRLM, GLSZM, ...)

### Compute texture
    import os
    from cerr import datasets
    from cerr.radiomics import texture_utils

    scanNum = 0
    structNum = 0
    # JSON settings file for a convolutional filter (IBSI-2). Samples shipped with pyCERR:
    # mean_filter.json, LoG_filter.json, Rot_inv_laws_energy_filter.json, gabor_filter.json
    settingsFile = os.path.join(os.path.dirname(datasets.__file__),
                                'convolutional_filter_settings', 'LoG_filter.json')
    planC = texture_utils.generateTextureMapFromPlanC(planC, scanNum, structNum, settingsFile)
    # The texture map is added as a new scan, cropped around the structure
    textureScanNum = len(planC.scan) - 1
    texture3M = planC.scan[textureScanNum].getScanArray()
