# pyCERR Test Suite

This document describes the automated tests for pyCERR: what they cover, where
their data comes from, and how to run them. It is intended to help users and
contributors understand the scope and limits of the suite's coverage.

## At a glance

- **85 tests** across **11 files** (`pytest` collection count).
- **Fully offline.** Every test uses data **bundled in the repository** under
  `cerr/datasets/`. No network access or external download is required.
- **Two kinds of tests:**
  1. **Validation / reference-standard tests** — radiomics (IBSI-1, IBSI-2) and
     dosimetric outcome models (ROE), checked against published reference values
     or recorded benchmarks. These guard *scientific correctness*.
  2. **I/O round-trip & structural tests** — DICOM/NIfTI/HDF5 load-save,
     contour rasterization, DVH computation, and module-import smoke tests.
     These guard the *data pipeline* against regressions.
- **CI:** GitHub Actions runs `flake8` + `pytest` on **Python 3.9–3.12** plus
  **3.13**, on Ubuntu, under `xvfb` (so the napari/PyQt5 import tests run
  headless). See `.github/workflows/python-package.yml`.

## Running the tests

```bash
# All tests
pytest tests/

# A single file
pytest tests/test_ibsi1_features.py

# A single test
pytest tests/test_dvh.py::test_getDVH_uniform_dose

# Verbose, as CI runs it
pytest -v
```

The radiomics tests (especially the IBSI-2 filter maps) are compute-heavy, so a
full run takes on the order of 30 minutes. The I/O / structural tests run in a
few seconds each.

### Linting (also enforced in CI)

```bash
# Hard failure on syntax errors / undefined names
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
# Style report (non-blocking)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

## Bundled test data

All inputs live under `cerr/datasets/`:

| Path | Contents | Used by |
|------|----------|---------|
| `radiomics_phantom_dicom/pat_1` | CT DICOM series + one RTSTRUCT (`GTV-1`) | IBSI-1, and all I/O / structural tests |
| `ibsi_radiomics_dicom/ibsi2_phase_1` | IBSI-2 synthetic digital phantoms | IBSI-2 filter tests |
| `ibsi_radiomics_dicom/ibsi2_phase_2` | IBSI-2 CT phantom (`phantom.nii` + `mask.nii`) | IBSI-2 feature tests |
| `radiomics_settings/ibsi_settings/` | JSON extraction settings per IBSI config | IBSI-1, IBSI-2 |
| `reference_values_for_tests/` | Benchmark CSVs + per-feature tolerances | IBSI-1, IBSI-2 |

## Validation / reference-standard tests

### `test_ibsi1_features.py` — IBSI-1 scalar radiomics (6 tests)

Validates pyCERR's scalar radiomic features against the **IBSI-1 reference
standard** on the bundled CT phantom. Each test runs one IBSI configuration and
asserts every computed feature matches the benchmark within its published
per-feature tolerance (`np.testing.assert_allclose(..., atol=tolerance)`).

| Test | Config | Interpolation | Texture aggregation |
|------|--------|---------------|---------------------|
| `test_config_a_original_feats_merged_texture` | A (2.5D) | none | merged across directions |
| `test_config_a_original_feats_all_dirs` | A (2.5D) | none | per direction |
| `test_config_b_bilinear_interp_feats_merged_texture` | B (2.5D) | bilinear | merged |
| `test_config_b_bilinear_interp_feats_all_dirs` | B (2.5D) | bilinear | per direction |
| `test_config_c_trilinear_interp_feats_merged_texture` | C (3D) | trilinear | merged |
| `test_config_c_trilinear_interp_feats_all_dirs` | C (3D) | trilinear | per direction |

Exercises: `cerr.radiomics.ibsi1.computeScalarFeatures` (morphology,
first-order statistics, and all texture matrices — GLCM, GLRLM, GLSZM, GLDZM,
NGTDM, NGLDM).

### `test_ibsi2_features.py` — IBSI-2 filtered-image features (8 tests)

Computes scalar features on **filter-response images** (IBSI-2 phase 2) and
compares them against recorded pyCERR values (a **regression / consistency**
reference; the IBSI/MATLAB-CERR comparison is present but commented out).
Tolerance is 1% percentage difference. Covers statistics on the original and
resampled image plus mean, LoG, and rotation-invariant Laws-energy filter
responses in 2D and 3D.

### `test_ibsi2_filters.py` — IBSI-2 convolutional filters (14 tests)

Compares pyCERR's **filter response maps** against the **IBSI-2 phase-1
consensus** maps on synthetic digital phantoms. Exercises
`cerr.radiomics.texture_utils` / `texture_filters`:

mean, Laplacian-of-Gaussian (LoG), Laws kernels, rotation-invariant Laws, Laws
energy, rotation-invariant Laws energy, Gabor (2D and 2.5D), wavelet (Haar/Coif
etc.), and rotation-invariant wavelet — each in 2D and/or 3D.

### `test_roe_models.py` — Radiotherapy Outcomes Estimator models (39 tests)

Validates the **dosimetric outcome model mathematics** in
`cerr.roe.dosimetric_models` and the fraction-size / fraction-number
corrections in `cerr.dataclasses.dose`. No data files — these are pure-function
tests. Models covered: LKB (rectal bleeding), logistic/Cox bronchial stenosis &
toxicity, logistic/Cox esophagitis (Huang, Jackson, Wijsman), and the Appelt
pneumonitis model with risk factors.

Each model is checked for: output in the valid probability range `[0, 1]`,
monotonicity in dose, agreement with reference NTCP values at known doses
(e.g. 40 Gy), correct near-zero-dose behavior, and the expected direction of
covariate / risk-factor effects.

## I/O round-trip & structural tests

These use the bundled DICOM phantom and assert that pyCERR's data pipeline is
lossless or behaves analytically.

### `test_dicom_load.py` — DICOM import sanity (2 tests)

`loadDcmDir` produces a scan whose array shape matches `getScanSize`, whose
coordinate vectors (`getScanXYZVals`) line up with the array axes and have
monotonic z; and the bundled RTSTRUCT is correctly associated with the scan.

### `test_nii_export_import.py` — NIfTI round-trip (2 tests)

- `test_scan_export_import`: CT scan → `saveNii` → `loadNiiScan` → array matches
  exactly.
- `test_dose_export_import`: a synthetic non-uniform dose → `saveNii` →
  `loadNiiDose` → array matches (`allclose`). *(pyCERR has no DICOM RTDOSE
  export; NIfTI is the supported dose export, so the round-trip is tested
  there.)*

### `test_rasterization.py` — contour ↔ mask round-trip (3 tests)

A synthetic axis-aligned box mask survives `importStructureMask` (mask →
polygon contours) → `getStrMask` (polygon → mask) **exactly** (Dice = 1, exact
voxel count), and the bundled RTSTRUCT rasterizes to a non-empty, in-grid mask.
Exercises `cerr.dataclasses.structure.importStructureMask` and
`cerr.contour.rasterseg`.

### `test_dvh.py` — DVH against an analytic case (3 tests)

With a spatially-uniform dose `D` over a box structure: every structure voxel
reads `D`; the integrated histogram volume equals voxel-count × voxel-volume;
and `Dx`/`Vx`/`MOHx`/`meanDose` all collapse to `D`. Exercises `cerr.dvh`
(`getDVH`, `doseHist`, and the summary metrics) plus `importDoseArray` /
`getDoseAt`.

### `test_dcm_export_rtstruct.py` — RTSTRUCT export round-trip (1 test)

Exports structure 0 with `cerr.dcm_export.rtstruct_iod.create`, reloads it
alongside the CT slices, and verifies the reimport is **bit-identical**: same
structure name, scan association, slice coverage, and rasterized mask.

### `test_h5_roundtrip.py` — HDF5 serialization round-trip (1 test)

`saveToH5` → `loadFromH5` preserves scan pixels, the structure mask, and the
dose array. (Implementing this test drove the completion of H5 dose
serialization, previously a no-op.)

### `test_module_imports.py` — import smoke tests (6 tests)

Plain-import checks for the large GUI / planning modules that have no functional
unit tests: `cerr.viewer.pycerr_napari` / `pycerr_gui` / `pycerr_nbviewer`,
`cerr.imrtp.imrtp`, and the `cerr.viewer` lazy-proxy behavior. Heavy optional
GUI dependencies are skipped (`pytest.importorskip`) where absent so the suite
stays portable; where present, the module must import cleanly. These catch
syntax errors, broken relative imports, and package-shadowing regressions.

## Coverage by module

| Area | Module(s) | Coverage |
|------|-----------|----------|
| Radiomics — scalar features | `cerr.radiomics.ibsi1` | **Strong** — IBSI-1 reference-validated |
| Radiomics — filters | `cerr.radiomics.texture_filters`, `texture_utils` | **Strong** — IBSI-2 consensus-validated |
| Outcome models | `cerr.roe.dosimetric_models` | **Strong** — reference + property tests |
| DICOM import | `cerr.plan_container.loadDcmDir` | **Good** — geometry + association |
| NIfTI I/O | scan/dose `saveNii`, `loadNiiScan/Dose` | **Good** — round-trip |
| HDF5 I/O | `saveToH5` / `loadFromH5` | **Good** — scan+structure+dose round-trip |
| RTSTRUCT export | `cerr.dcm_export.rtstruct_iod` | **Good** — round-trip |
| Rasterization | `cerr.contour.rasterseg`, `structure.importStructureMask` | **Good** — exact round-trip |
| DVH | `cerr.dvh` | **Good** — analytic |
| Viewers / IMRTP | `cerr.viewer.*`, `cerr.imrtp.*` | **Import-only** smoke tests |

### Not yet covered (known gaps)

- **DICOM RTDOSE / REG export** (`cerr.dcm_export.reg_iod`; RTDOSE export is not
  implemented).
- **Deformable registration** (`cerr.registration`) — depends on external
  binaries (plastimatch / ANTs).
- **Functional imaging** (`cerr.mri_metrics.dce_mri`).
- **IMRTP dose-calculation numerics** (`cerr.imrtp.dosecalc.*`) — only imports
  are tested, not computed dose values.
- **Viewer rendering / interaction** (`cerr.viewer.*`) — only imports are
  tested, not GUI behavior.
- **Coordinate-transform internals** beyond what the I/O round-trips exercise.

## Conventions for adding tests

- Prefer the **bundled phantom** (`cerr/datasets/radiomics_phantom_dicom/pat_1`)
  so tests stay offline and deterministic.
- Build synthetic structures with `importStructureMask` and synthetic dose with
  `importDoseArray` when you need a controlled, analytic case.
- Write outputs to pytest's `tmp_path` rather than the repo root.
- Guard optional heavy dependencies with `pytest.importorskip` so the test
  skips (rather than errors) where the dependency is absent.
- Round-trip and analytic tests are valuable: two real bugs (unimplemented H5
  dose serialization, and a `np.string_` removed in NumPy 2.0) were found this
  way.
