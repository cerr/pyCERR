# Changelog

All notable changes to pyCERR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.4.0] - 2026-10-16

### Added

- **`MOHx` and `MOCx` accept the cutoff as an absolute volume**, through a
  `volumeType` argument matching `Dx`: `volumeType=1` (default) reads the
  cutoff as a percentage of the structure, `volumeType=0` as cc. For example,
  `dvh.MOHx(doseBinsV, volsHistV, 2, 0)` is the mean dose of the hottest 2 cc.
- **Desktop viewer: File > Clear scene** (Ctrl+Shift+N) unloads all scans,
  structures and doses and closes the tool windows, so the next import starts a
  new planC instead of adding to the current one. Also callable as
  `viewer.clear_scene()`.

### Changed

- **Desktop viewer: Refresh from planC moved from the Tools menu to the File
  menu**, next to Clear scene.
- **Desktop viewer: with more than 30 structures in planC, structures start
  unchecked** (hidden) instead of all shown, so opening a large plan no longer
  rasterizes every structure first. On a 98-structure head-and-neck plan the
  viewer opens in 1.5 s instead of 53 s. Structures added after loading (e.g.
  saved from contouring) still start checked. The limit is
  `PyCerrViewer.MAX_DEFAULT_ON_STRUCTS`.
- Desktop viewer: the scan's default window and colorbar range are computed in
  one pass over a sample of at most 2 million voxels instead of two exact
  percentile passes over the whole volume.
- **The Qt GUIs run on Qt6 as well as Qt5.** `cerr.viewer.pycerr_gui`, the
  ROE GUI (`cerr.roe.launch`) and the IMRTP GUI now import Qt through
  [qtpy](https://github.com/spyder-ide/qtpy) instead of PyQt5 directly, and use
  matplotlib's binding-neutral `QtAgg` backend instead of `Qt5Agg`. They run
  with PyQt5, PyQt6, PySide2 or PySide6. qtpy takes the first binding it finds
  (PyQt5 before PyQt6); set `QT_API=pyqt6` to choose. The Qt5-only idioms (unscoped enums,
  `exec_()`, `pyqtSignal`, `QMouseEvent.pos()`) are replaced by forms that
  work on both. The `viewer` extra now installs `qtpy` plus PyQt5, or PyQt6 on
  aarch64 Linux, where PyQt5 has no wheel, so `pip install "pycerr[viewer]"`
  now works on ARM Linux too.
- **The `MOHx` / `MOCx` cutoff argument is renamed from `percent` to
  `volCutoff`, and their ROE model parameters to `volCutoff` and `volType`**,
  the names `Dx` uses. Calls that pass the cutoff positionally are unaffected.
  A model file still using the `percent` key now raises a `ValueError` naming
  the new keys, instead of a `KeyError`.
- `MOHx` and `MOCx` raise `ValueError` for a cutoff outside 0-100 percent,
  a cutoff larger than the structure volume, or an empty histogram, instead of
  returning 0.
- **`rasterseg.getStrMask` is about 18x faster** (98 structures of a
  512x512x386 head-and-neck CT: 46.5 s down to 2.5 s; median per structure
  128 ms down to 8 ms). It now writes the raster segments straight into the
  3-D mask instead of placing a stack of slices with fancy indexing, which
  alone took 40 s. `raster_to_mask` no longer loops over segments in Python.
  Both return exactly the same masks as before.
- **Behaviour change: `loadNiiScan` now reorients NIfTI scans to LPS by
  default** (`direction='LPS'`; it used to be `''`), so they come in with the
  same `imageOrientationPatient` as axial DICOM, `[1,0,0,0,1,0]`, and display
  the right way up in `pycerr_gui`. Before, a file stored RAS-style showed
  rotated 180 degrees. Physical coordinates are unchanged, but **scan arrays
  loaded from such a NIfTI may now be flipped** relative to earlier releases
  (rows and columns reversed for RAS files). Pass `direction=''` (or `None`) to
  keep the file's own voxel order. The registration helpers, `loadScanFromDB`
  and the AI-model pipelines pass `direction=''` explicitly, so their scans and
  exported masks keep the input file's grid. 4-D NIfTI scans are reoriented one
  volume at a time, since SimpleITK's `DICOMOrient` handles 3-D images only.

### Fixed

- `loadNiiDose` decided the slice order of the dose from `planC.scan[0]`
  instead of the associated scan, so a dose attached to any scan other than
  the first could come in with its slices reversed when scan 0 had a different
  slice order (for example a single-slice scan).
- `loadNiiStructure` did not resample 4-D label maps onto the associated scan,
  so their masks lined up only when the file shared the scan's exact grid and
  orientation. They are now resampled like 3-D label maps.
- **`MOCx` returned 0 whenever the coldest dose bin held more than the
  requested fraction of the volume**, e.g. MOC5 of a homogeneously irradiated
  structure, or of an organ with a low-dose plateau outside the field. The value
  looked like a plausible dose rather than an error.
- **`MOHx` and `MOCx` averaged the wrong volume.** The bin straddling the
  cutoff was included whole by `MOHx` and dropped whole by `MOCx`, so the
  averaged volume depended on how much volume shared that bin: MOH20 of a
  structure with 10% at 60 Gy and 90% at 10 Gy averaged the whole structure
  and returned 15 Gy instead of 35 Gy. Both now take exactly the requested
  volume, counting only the needed fraction of the boundary bin, which leaves
  only the half-bin resolution shared by all histogram metrics. Results on
  smooth dose distributions with fine bins change little (the user guide
  examples move by up to 0.03 Gy); results where much volume shares one bin
  can change substantially. MATLAB CERR's `calc_MOHx` / `calc_MOCx` use the
  previous logic, so the two now differ for these metrics.

## [2.3.3] - 2026-10-15

### Added

- **User guide in the Sphinx documentation.** New pages under `docs/`: a
  Quickstart, an end-to-end tutorial, the `planC` data model and coordinate
  systems, and guides for DVH metrics, outcome models (ROE), radiomics,
  registration and segmentation, each with worked examples and figures. The
  figures are generated from the bundled lung phantom by the scripts in
  `docs/_figure_scripts` (`python docs/_figure_scripts/make_all.py`); the
  dosimetry pages use a synthetic dose because no RTDOSE is bundled.
- **`Example:` sections in the docstrings of 32 commonly used functions**
  (import/export, DVH, outcome models, radiomics, structure operations,
  registration), and corrected argument descriptions for `dvh.getDVH`,
  `dvh.Dx`, `dvh.Vx`, `importScanArray`, `importStructureMask` and
  `registerScans`. Documentation only.
- **API reference pages for `cerr.gamma`, `cerr.contour.structure_consensus`,
  `cerr.segmentation` and `cerr.ai_models`**, which were missing. The docs
  build now also needs `sphinx-copybutton` and `pyyaml`
  (`docs/requirements.txt`).
- **`Scan.isSUV(suvType='BW')`**, which reports whether a scan actually holds
  SUV of the given normalization. When `convertToSUV` cannot convert an image
  the scanner already normalized (Units `GML` or `CM2ML`), the values it leaves
  behind are still an SUV, just of the scanner's normalization, so checking the
  units cannot tell a failed conversion from a successful one. `isSUV` checks
  the units and the SUV type recorded on every slice.
- **`SUVConversionWarning`** in `cerr.dataclasses.scan`, a `UserWarning`
  subclass issued on every path where `convertToSUV` cannot produce the
  requested SUV. Pipelines that must not continue without SUV can escalate it
  with `warnings.simplefilter('error', SUVConversionWarning)`. Warnings that
  only report an assumption made during a successful conversion, such as an
  administration date inferred from the acquisition date, remain plain
  `UserWarning`s.

### Changed

- **Promotion from `testing` to `main` is now a fast-forward, and release
  tags go on the commit both branches share.** Promotion used to create a
  merge commit on `main`, and releases were tagged there, so the tags were
  not reachable from `testing`: builds and editable installs from `testing`
  reported a stale version (`2.1.1.devN` after 2.3.2 had shipped). A one-time
  merge of `main` into `testing` joined the histories;
  `tools/sync-branches.sh` now fast-forwards `main` and refuses to promote if
  `main` has commits `testing` lacks. The new `tools/tag-release.sh vX.Y.Z`
  tags the shared commit after checking that both branches are on it and
  that the changelog has the section. Published packages are not affected;
  existing tags are unchanged.
- **`convertToSUV` refusal messages now say what was left behind.** Each ends
  by stating that the scan is left unconverted. When stored `GML` values cannot
  be converted, the message names the normalization that remains (for example
  "SUVbw not computed ... so the values remain SUVibw"). The message for
  unsupported units now names the units instead of listing only `BQML` and
  `CNTS`. Code that matched on the previous message text should match on
  `SUVConversionWarning` instead.

### Fixed

- **PET SUV refused for series whose Series and Acquisition Time differ only
  by rounding.** With Decay Correction `START` and no vendor private scan
  start datetime, the Acquisition DateTime is used as the scan start when it
  equals the Series DateTime, but the comparison was exact. A Series Time
  stored with fractional seconds (`154441.359`) against a truncated
  Acquisition Time (`154441`) therefore fell through to back-computing the
  scan start from frame timing, which needs Actual Frame Duration; when that
  was empty, `convertToSUV` warned and left the scan unconverted. The two are
  now treated as identical within 1.0 s, the tolerance the IBSI-SUV manual
  uses for rounding mismatches (`ACQ_SERIES_TOLERANCE_SECS`). Such series
  that did have Actual Frame Duration were previously converted using the
  back-computed scan start; they now use the Acquisition DateTime directly,
  so their SUVs may shift slightly.

## [2.3.2] - 2026-10-09

### Fixed

- **Shape features were wrong with in-plane resampling and a non-zero
  `resolutionZCm`.** With `inPlane: "yes"` only x and y are resampled, but the
  output grid was still built from `resolutionZCm`. When that differed from
  the slice spacing, the structure was squashed (or stretched) along z:
  `shape_volume`, `shape_filledVolume` and `firstOrder_totalEnergy` were off
  by the ratio of the two spacings, and the axis lengths, surface area,
  3D / sagittal / coronal diameters, sphericity, compactness, flatness and
  elongation were wrong as well. Other first-order features and the texture
  features (GLCM, GLRLM, GLSZM, GLDM, NGTDM) were not affected. In-plane
  resampling now always keeps the scan's slice spacing. `inPlane: "yes"` with
  `resolutionZCm: 0`, as in the IBSI settings files, was already correct.
- **The sample `original_settings.json` triggered the above.** It combined
  `inPlane: "yes"` with `resolutionZCm: 0.1`; for 3 mm slices the reported
  shape volume was a third of the true value. `resolutionZCm` is now `0`.
  **Re-extract shape features and total energy computed with this file.**
- **`dosimetric_models.runFromPredictors` failed for LKB models.** `LKBFn`
  did not use a supplied gEUD, so `runFromPredictors('Rectal bleeding (grade
  2+)', {'Rectum gEUD': 70})` raised `UnboundLocalError`.
- **`dvh.medianDose` was not a volume-weighted median.** It returned the
  middle of the non-empty dose bins regardless of how much volume each held.
  It now returns the dose at which half of the structure volume is reached,
  in agreement with `Dx(doseBinsV, volsHistV, 50, 1)`. Values returned by
  `medianDose` change for any non-symmetric DVH.

- **The `importDoseArray` docstring named a `doseInfo` key that is ignored.**
  It said the dictionary takes `'units'`; keys are `Dose` attribute names, so
  the one that sets the dose units is `'doseUnits'`. The docstring also said
  the function returns a planC with a scan imported; it imports a dose.
  Documentation only.

### Changed

- **`saveToH5` writes the whole `planC` by default.** Called without index
  lists it used to write a file with no scans, structures or doses. The
  defaults are now "all"; passing lists still selects a subset, and an empty
  list writes none of that type.

## [2.3.1] - 2026-10-09

### Fixed

- **`saveToH5` failed for structures with more than one contour on a slice.**
  Rings (an outer and an inner contour) and multi-part structures raised
  `ValueError: Unable to ... create group (name already exists)`, because the
  `segments` group was created once per contour instead of once per slice.
  Such structures now save and reload with identical masks. Files written
  for single-contour structures are unchanged, and existing HDF5 files load
  as before.

## [2.3.0] - 2026-10-09

### Added

- **Outcome models can be evaluated from pre-computed predictors.**
  `cerr.roe.dosimetric_models.runFromPredictors(modelFile, predictors)` takes
  a model name, JSON path or parameter dict plus a `{name: value}` dict and
  returns the model output without a `planC`. Dose metrics are named
  `"<structure> <metric>"` (e.g. `"Esophagus meanDose"`) and must already be
  fractionation-corrected if the model requires it; patient predictors use
  their parameter name (e.g. `"concurrentChemo"`). In a model file, a numeric
  `val` on a dose metric is likewise used as given instead of being computed
  from the DVH.

### Fixed

- **`parseDcmHeader` read PatientName into the PatientID column.** The tag list
  had Patient's Name (0010,0010) twice, so `loadDcmDir` never used Patient ID
  (0010,0020) when grouping files into scans. Files that share a patient name
  but have different patient IDs are now kept apart by ID as well as by their
  study and series UIDs.
- **A dose imported from NIfTI could not be exported.** `Dose.saveNii` (and
  `getSitkImage`) raised `IndexError` for a dose loaded with `loadNiiDose`,
  because the import did not store the image orientation and position. They
  are now set on import, and the dose round-trips with the same values and
  grid.
- **"Bronchial toxicity (Grade 5)" skipped its fraction-size correction.** The
  model file spelled `correctionType` as `"fsize"`, which was not recognised,
  so the model was evaluated on the uncorrected dose. It is now `"frxSize"`
  and the correction to 2 Gy fractions (alpha/beta = 3) is applied. Results
  for this model change whenever the input fraction size is not 2 Gy.
- **`cerr.roe` could not be imported without the `viewer` extra.** The package
  loaded the ROE GUI on import, so `from cerr.roe import dosimetric_models`
  failed with `No module named 'PyQt5'` on a base install (since 2.2.1, when
  PyQt5 became optional). The GUI is now loaded only by `cerr.roe.launch()`.
- **`cerr.ai_models` could not be imported on a base install.** It reads the
  models' `run_spec.yaml` with PyYAML, which was not a declared dependency and
  was present only when another package pulled it in. `pyyaml` is now a core
  dependency.
- **The `loadNiiScan` docstring gave `'HFS'` as an example `direction`.** The
  value is a 3-letter orientation code such as `'LPS'` or `'RAS'`; a patient
  position like `'HFS'` raises an error. Documentation only.

### Changed

- **Outcome models no longer assume a value for clinical predictors.** In the
  built-in model files, patient predictors that defaulted to `0` ("No",
  "Male", ...) are now unset, and `dosimetric_models.run` raises
  `ValueError: No value supplied for predictor(s): ...` until they are
  provided. Set `model['parameters'][name]['val']` on the parameter dict
  before calling `run`, or pass the values to `runFromPredictors`. Affected
  models: Esophagitis (Huang), Esophagitis (Wijsman), Pneumonitis (Appelt)
  and Xerostomia (grade 2+).
- **`dosimetric_models.get_corrected_dvbins` is renamed `getCorrectedDVbins`.**
  The old name is removed; update any code that calls it.
- **Model parameter files list dose metrics per structure.**
  `parameters.structures` is now a dict of
  `{structure: {metric: {"val": ..., "weight": ..., "params": ...}}}` in every
  built-in model, including the two that used a bare structure name
  (Pneumonitis (Appelt) and Rectal bleeding (grade 2+)). For the LKB model
  (Rectal bleeding), the volume parameter `n` moves from the top level into
  the `gEUD` metric's `params`. Custom model files in the old layout need the
  same change.
- **Dose metrics in model files are looked up by name, not evaluated as
  code.** A metric must be one of `meanDose`, `minDose`, `maxDose`, `Dx`,
  `Vx`, `gEUD`, `MOHx` or `MOCx`. `dosimetric_models.run` also works on a copy
  of a parameter dict passed to it, so the caller's dict is not modified.
- **Outcome-model tests are reorganised.** `tests/test_roe_models.py` is
  replaced by `tests/test_dosimetric_models.py`.
- **`loadDcmDir` raises `FileNotFoundError` for a path that does not exist.**
  It used to return an empty `PlanC`, so a mistyped path surfaced later as an
  unrelated `IndexError`. An existing directory with no DICOM files still
  returns an empty `PlanC`. A single file path containing spaces is now
  accepted as one file instead of being split on the spaces.

## [2.2.3] - 2026-09-28

### Changed

- **Faster GLRLM and GLSZM.** `calcRLM` walks all runs of every grey level
  together along flat-index strides, visiting each voxel once per direction
  instead of once per level. `calcSZM` labels the zones of all levels in a
  single `skimage.measure.label` pass, and `szmToScalarFeatures` drops empty
  zone-size columns before building its temporaries. Both matrices are
  identical to 2.2.2; about 10x faster on the IBSI 1 phantom.

## [2.2.2] - 2026-09-28

### Fixed

- **GLCM sum entropy dropped the p(x+y) = 2 term.** The emptiness check on the
  indices of grey-level sum 2 used `np.any`, and the only cell with that sum
  sits at index 0, so the term p(1,1)·log₂p(1,1) was left out whenever the
  lowest grey level neighboured itself. Sum entropy was underestimated for
  such structures. The same check in the difference-entropy loop is fixed too.
- **NGTDM contrast skipped grey-level shifts.** The contrast double sum
  skipped shift *k* whenever grey level *k* was empty, which dropped valid
  pairs for any structure with empty grey levels (typically homogeneous
  organs with fixed bin number).
- **NGTDM level probabilities now follow IBSI.** `p` counted every voxel in
  the structure but was normalised by the number of voxels with a valid
  neighbourhood (N<sub>v,c</sub>), so it could sum to more than 1 in 2D/2.5D.
  It now counts only voxels with at least one neighbour. All NGTDM features
  can change slightly for 2D/2.5D calculations. With both NGTDM fixes, contrast
  matches the IBSI 1 consensus values for configurations A, B and C.

### Changed

- **`polyFill` is vectorized**, 3-23x faster on typical contours, with
  pixel-identical output (checked against the previous implementation in
  `tests/test_poly_fill.py`). Contours extending past the image edge are now
  clipped; previously they raised `IndexError` past the far edge and wrapped
  around at negative indices.
- **`importStructureMask` pads a uint8 mask** instead of an int64 copy,
  cutting its memory use 8x. Output is unchanged.

## [2.2.1] - 2026-09-26

### Changed

- **The Qt desktop viewer's dependencies are now an optional extra.** `PyQt5`,
  `pyvista` and `pyvistaqt` moved out of the base install into
  `pip install "pycerr[viewer]"`. PyQt5 publishes no aarch64 Linux wheel, so
  `pip install pycerr` on arm64 Linux (Apple Silicon Docker images, ARM
  servers) tried to build it from source and failed for want of `qmake`.
  Everything that does not draw a Qt window is unaffected: the Python API,
  batch/headless processing in containers, and the Jupyter viewer
  (`cerr.viewer.pycerr_nbviewer`), which uses `ipywidgets` + matplotlib and
  keeps `ipywidgets` as a base dependency. Launching the desktop viewer without
  the extra now raises an ImportError naming the install command instead of
  `No module named 'PySide2'`.

## [2.2.0] - 2026-09-17

### Added

- **Deformation vector field display in pyCERR-GUI**, in the Registration QA
  tool (Tools > Registration QA). A stored `planC.deform` is drawn as a quiver
  on all three 2D views and in the 3D scene - the counterpart of the napari
  `Vectors` layer that `showNapari(..., vectors_dict=...)` builds. Arrows can be
  drawn at true 1:1 scale (a 3 mm displacement is a 3 mm arrow) or normalized to
  the field of view; sampled at a chosen resolution, restricted to a structure
  or to its surface, with the median (bulk) displacement optionally subtracted
  to leave the local deformation; and coloured by `length`, `|dx|`, `|dy|`,
  `|dz|` or their signed forms, in mm. Scriptable via
  `PyCerrViewer.set_dvf_overlay()` / `clear_dvf_overlay()` and the new
  `deform`/`dvf_*` keywords of `start_reg_qa()`. Only deformations that carry a
  vector field (`deformOutFileType` `'dvf'` or `'vf'`) are offered - a rigid,
  ANTs or B-spline-coefficient transform has no per-voxel displacements.
- `cerr.dataclasses.deform.hasDvfMatrix()` and `DVF_FILE_TYPES`, the shared
  check for whether a `Deform` carries a sampled vector field.
- `cerr.registration.register.getDvfGrid()`, which samples a `Deform` on a
  regular grid as three dense component arrays in pyCERR virtual cm - the grid
  twin of `getDvfVectors()`, which returns the same vectors scattered, and
  taking the same `outputResV` / `structNum` / `surfFlag` controls. The shared
  DICOM-LPS-mm to virtual-cm conversion is now `dvfToCerrVirtual()`, used by
  both.
- **Enhanced PET Image import** (SOP Class `1.2.840.10008.5.1.4.1.1.130`).
  Functional-group macros are now resolved through `getFunctionalGroupItem()`,
  which reads the Per-Frame Functional Groups Sequence first and falls back to
  the Shared Functional Groups Sequence, per the DICOM rule that a macro appears
  in exactly one of the two. This covers Plane Position, Plane Orientation,
  Pixel Measures, Pixel Value Transformation, Real World Value Mapping and
  Frame VOI LUT. Previously the parser read these only from the per-frame group
  and raised `AttributeError` on `PlaneOrientationSequence` for any encoder that
  placed orientation in the shared group.
- **`suvType='AS_STORED'`** in `Scan.convertToSUV()` and the `loadDcmDir`
  options, which keeps the normalization the scanner already applied instead of
  converting to body weight. Because the result then depends on the acquisition
  protocol rather than on the request, it should not be used to pool
  measurements across a cohort.
- Datetime helpers `parseDcmDateTime()`, `parseDcmTimeOfDay()` and
  `combineDcmDateAndTime()` for DICOM DA/TM/DT values, including fractional
  seconds and UTC offsets.
- `getDecayReferenceDateTime()`, `getAdministrationDateTime()` and
  `getAverageCountRateTime()`, implementing the scan-start and average
  count-rate-time strategies from the IBSI-SUV manual.
- `getSuvNormalizationFactor()`, providing the body weight, lean body mass
  (Morgan, James/Morgan, Janmahasatian), ideal body weight and body surface
  area factors in one place.
- `ScanInfo` fields required by the above: `frameReferenceTime`,
  `actualFrameDuration`, `decayFactor`, `injectionDateTime`,
  `petDecayCorrected` and `frameAcquisitionDateTime`.

### Changed

- **SUV computation now follows the
  [IBSI-SUV manual](https://oncoray.github.io/suv_computation/suv.html).**
  Stored values are brought to a body-weight SUV through an explicit
  decay-correction reference datetime rather than time-of-day arithmetic.
- **Images the scanner already normalized are re-scaled rather than passed
  through.** When Units are `GML` or `CM2ML`, the normalization recorded in SUV
  Type (0054,1006) is divided out and body weight re-applied. Previously such
  images were returned as stored, so an image normalized by lean body mass,
  ideal body weight or body surface area was reported as though it were
  SUVbw.
- **`suvType` is now always a concrete type.** It defaults to `'BW'`; `None`
  and `''` are rejected with `ValueError` as ambiguous, since they cannot be
  distinguished from an unset option and guessing between body weight and the
  stored normalization would silently change the values. Unrecognized types
  raise instead of failing quietly.
- **Missing or invalid attributes now warn and leave the scan unconverted**
  instead of returning a value derived from incomplete metadata. This applies
  to absent patient weight, radionuclide total dose, half life, frame duration
  and unusable units.
- Patient's Weight of 1000 or greater is interpreted as grams, and Patient's
  Sex of `'O'` uses the mean of the sex-specific factors for the normalizations
  that require one.
- `Scan.convertToSUV()` and `loadDcmDir()` documentation now describes
  `suvType` in terms of the requested output normalization rather than the
  value stored in the file, and lists the full set of supported types.

### Fixed

- `cerr.dataclasses.structure.getJsonList()` / `saveJson()` raised
  `TypeError: Unexpected type int64` on any structure given a default color -
  those imported from NIfTI or a mask, smoothed with B-splines, or read from an
  RTSTRUCT without `ROIDisplayColor`. `getColorForStructNum()` now returns
  native ints, and the JSON encoders convert numpy scalars, arrays and pydicom
  sequences, so structures restored from HDF5 or pickles serialize too.
- `cerr.registration.register.getDvfVectors()` raised `AttributeError` when
  `structNum` was omitted, although the argument is documented as optional: the
  structure mask was built before the check that needs it.
- DICOM CT import raised `TypeError` when a series had no `SliceThickness`
  (0018,0050) value. `parseScanInfoFields()` now falls back to
  `SpacingBetweenSlices`, matching the guard in `loadSortedScanInfo()`.
- **Scan start datetime resolution when `DecayCorrection` is `START`.** The
  reference time is now taken, in order of decreasing reliability, from the
  vendor private scan start datetime (Siemens `0071,1022`, GE `0009,100D`) or
  Decay Correction DateTime (`0018,9701`); from the Acquisition DateTime when
  it equals the Series DateTime; and otherwise by back-computation from the
  frame timing attributes, using `t_acq - dt` for GE and `t_acq + T_ave - dt`
  for other manufacturers. Previously Series Time was used, falling back to the
  earliest Acquisition Time whenever that was earlier, which produced incorrect
  SUVs for series whose Series Time had been shifted during processing.
- **Uptake periods spanning midnight, and administration dates inconsistent
  with the acquisition date.** All time arithmetic now carries a date
  component; previously it used seconds since midnight, so an administration on
  the preceding day produced a decay factor several orders of magnitude too
  small.
- **`DecayCorrection` of `NONE`.** The administered dose is now corrected to the
  frame measurement time `t_acq + T_ave`. Previously no correction was applied
  at all, leaving the result wrong by the whole uptake decay factor.
- **Contour matching for multi-frame images.** Every frame of a multi-frame
  image shares one SOP Instance UID, so a referenced SOP Instance UID cannot
  identify a slice. Contours now fall back to matching by z-coordinate when the
  UID does not uniquely identify one, which previously yielded empty masks.
- Rescale Type is now read from the Pixel Value Transformation macro, and image
  units are derived from it for PET images that omit the top-level Units
  (0054,1001) attribute, as Enhanced PET does.

### Validation

Verified against the
[IBSI-SUV digital reference objects](https://github.com/oncoray/suv_computation),
a public collection of synthetic phantoms that encode one object many different
ways (units `BQML`, `CNTS`, `GML` and `CM2ML`; decay correction `START`,
`ADMIN` and `NONE`; vendor private tags; dose in Bq and MBq; and the Enhanced
PET Image IOD).

- All 43 reference DROs return SUVbw of 0.2, 1.0 and 4.0 for the minimum,
  median and maximum within the supplied mask.
- Every supported `suvType` reproduces the DRO that stores that normalization
  natively, matched on Patient's Sex.
- All 15 `error` DROs, which omit attributes SUV computation requires, warn and
  decline to produce a value.

A notebook reproducing these results is available in
[pyCERR-Notebooks](https://github.com/cerr/pyCERR-Notebooks) at
`09_functional_imaging/compute_suv_from_dro.ipynb`.

## [2.1.0]

See the [release notes](https://github.com/cerr/pyCERR/releases) for versions
up to and including 2.1.0.
