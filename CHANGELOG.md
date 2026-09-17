# Changelog

All notable changes to pyCERR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
