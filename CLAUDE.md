# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyCERR (Python-based Computational Environment for Radiological Research) is a medical imaging library for processing, visualizing, and analyzing radiological data. It supports DICOM, NIfTI, and HDF5 I/O, and provides tools for radiomics, dose-volume histogram analysis, and deformable image registration.

## Development Commands

### Installation
```bash
conda create -y --name pycerr python=3.11
conda activate pycerr
pip install -e ".[napari]"
```

### Running Tests
```bash
# All tests
pytest tests/

# Single test file
pytest tests/test_ibsi1_features.py

# Single test function
pytest tests/test_ibsi1_features.py::test_function_name
```

### Linting
```bash
# Syntax errors and undefined names (enforced in CI)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Style check (max line length 127, max complexity 10)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### Building Docs
```bash
cd docs && make html
```

## Architecture

### Central Data Container: `PlanC`

All imaging data lives in a `PlanC` instance (defined in `cerr/plan_container.py`). It holds lists of domain objects:

- `planC.scan` — volumetric images (CT, MR, PT, US, NM)
- `planC.structure` — segmentations (RTSTRUCT, SEG)
- `planC.dose` — dose distributions (RTDOSE)
- `planC.beams` — treatment plans (RTPLAN)
- `planC.deform` — deformable registration results
- `planC.header` — metadata (version, creation date)

### Key Modules

| Module | Purpose |
|--------|---------|
| `cerr/plan_container.py` | PlanC container + all I/O (DICOM, NIfTI, H5, pickle) |
| `cerr/dataclasses/` | Domain dataclasses: `scan.py`, `structure.py`, `dose.py`, `beams.py`, `deform.py` |
| `cerr/radiomics/` | IBSI-compliant texture/radiomic features and filters |
| `cerr/contour/` | Contour processing and rasterization (polygon → mask) |
| `cerr/viewer.py` | Interactive napari-based 2D/3D visualization |
| `cerr/dvh.py` | Dose-Volume Histogram calculation |
| `cerr/registration/` | Deformable image registration via plastimatch |
| `cerr/dcm_export/` | DICOM export (RTSTRUCT, RTDOSE IODs) |
| `cerr/utils/` | AI pipeline setup, image processing, masking, UID generation |
| `cerr/roe/` | Radiomics Outcome Explorer (dosimetric models) |
| `cerr/datasets/` | IBSI radiomics phantoms and reference values for validation |

### Typical Data Flow

```
loadDcmDir(dcmDir) → PlanC
  ├─ Scan: DICOM pixels → coordinate transforms → optionally SUV (PET)
  ├─ Structure: DICOM contour points → coordinate transform → lazy rasterization
  ├─ Dose: DICOM dose grid → coordinate transform
  └─ Beams: RTPLAN fields

PlanC → process:
  ├─ radiomics.ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)
  ├─ dvh.getDVH(structNum, doseNum, planC)
  └─ registration.register.register_scans(planC, ...)

PlanC → export:
  ├─ saveToH5(planC, h5File)   # HDF5 serialization
  ├─ scan.saveNii(fileName)    # NIfTI export
  └─ dcm_export/               # DICOM RT export
```

### Coordinate Systems

Each `Scan` object carries three transformation matrices:
- `Image2PhysicalTransM` — voxel indices → DICOM physical space (mm)
- `Image2VirtualPhysicalTransM` — voxel indices → pyCERR virtual space
- `cerrToDcmTransM` — pyCERR virtual → DICOM physical

pyCERR's virtual coordinate system differs from DICOM's: it uses a right-handed system with `(x, y, z)` = `(col, row, slice)` ordering. Conversion happens in `convertDcmToCerrVirtualCoords()` inside each dataclass loader.

### Contour Rasterization

Structure contour points are stored as polygon vertices (DICOM format). Rasterization to binary masks is performed lazily via `cerr/contour/rasterseg.py` — masks are computed on demand, not pre-stored. Use `structure.getStructureMask3M(structNum, scanNum, planC)` to get a 3D mask array.

### Radiomics

Radiomics feature extraction is IBSI-compliant (International Biomarker Standardization Initiative). The `cerr/datasets/` directory contains the IBSI phantoms and reference values used in `tests/test_ibsi1_features.py` and `test_ibsi2_features.py` to validate correctness. Settings are passed via JSON files in `cerr/radiomics/settings/`.

## CI

GitHub Actions (`.github/workflows/python-package.yml`) runs linting and pytest on Python 3.9–3.12 (Ubuntu). Tests download reference data from external URLs; network access is required for the test suite.
