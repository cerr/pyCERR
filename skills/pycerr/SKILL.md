---
name: pycerr
description: Analyse radiotherapy and medical-imaging data in Python with pyCERR. Load DICOM, NIfTI or HDF5 into a planC; get scans, structure masks and doses as arrays; compute DVH metrics, IBSI radiomics features and texture (filter) maps; register scans; export DICOM RTSTRUCT, NIfTI or HDF5; and view data in the pyCERR desktop viewer. Use for any task that mentions pyCERR, CERR, planC, or radiotherapy DICOM (RTSTRUCT, RTDOSE, RTPLAN) analysis in Python.
---

# pyCERR

pyCERR (`pip install pycerr`, imported as `cerr`) keeps everything known about
one patient in a single plan container, `planC`. Almost every function takes a
`planC`, and the ones that add something return an updated one.

This skill was checked against **pyCERR 2.5.4**. Check the installed version
with `importlib.metadata.version("pycerr")`.

## How to work

<!-- examples:step -->
1. **Read the guide for the area first** (see "References"). Their code blocks
   show the real calls with real arguments. Adapt them; don't write pyCERR
   calls from memory.
2. **Check the names** before handing over or running a script:
   `python scripts/check_pycerr_names.py my_script.py`. It reports any `cerr.*`
   module, function or class that does not exist in the installed pyCERR. If it
   flags something, find the real name in a reference. Never keep a guessed
   name.
3. Say what the script does, and what the user must fill in (paths, structure
   names).

## The plan container

| Attribute | Holds |
|---|---|
| `planC.scan` | CT, MR, PET, US, NM, and derived images (texture maps, warped scans) |
| `planC.structure` | segmentations (RTSTRUCT, SEG, NIfTI masks, arrays) |
| `planC.dose` | dose grids (RTDOSE, arrays) |
| `planC.beams` | RTPLAN beam and fraction-group sequences |
| `planC.deform` | rigid and deformable registrations |

- Objects are addressed by list position, from 0: `scanNum`, `structNum`, `doseNum`.
- Links between objects are UIDs, not positions. A structure's scan is
  `planC.structure[i].getStructureAssociatedScan(planC)`. A dose's scan is
  `scn.getScanNumFromUID(planC.dose[i].assocScanUID, planC)`.
- **Reassign:** `planC = pc.importStructureMask(...)`. Functions that add
  objects return the container.

**Units and axes. Get these right; they are the commonest silent error.**

| Quantity | Convention |
|---|---|
| Lengths, spacing, coordinates | **centimetres** (DICOM files are mm; ×10 when a parameter says `_mm`) |
| Arrays | indexed `[row, column, slice]` |
| `xV, yV, zV = scan.getScanXYZVals()` | `xV` per column; `yV` per row and **decreasing** with the row index; `zV` per slice |
| CT intensities | HU via `scan.getScanArray()` (not the raw `scanArray`, which lacks the CT offset) |
| PET | converted to SUV at import (`loadDcmDir(dir, opts={'suvType': ...})`) |
| Dose | Gy. A dose has its own grid; `dose.getDoseAt(x, y, z)` interpolates |
| Volume | cc: `mask3M.sum() * np.prod(scan.getScanSpacing())` |

## Task → function (checked signatures)

```python
from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.dataclasses import structure as cerrStr
from cerr.dataclasses import scan as scn
from cerr import dvh
from cerr.radiomics import ibsi1, texture_utils
```

| Task | Call |
|---|---|
| Load DICOM folder (scans, RTSTRUCT, RTDOSE, RTPLAN) | `planC = pc.loadDcmDir(dcmDir)`; append a study: `pc.loadDcmDir(dir2, initplanC=planC)` |
| Load NIfTI | `pc.loadNiiScan(file, imageType='CT SCAN')`, `pc.loadNiiStructure(file, assocScanNum, planC, labels_dict)`, `pc.loadNiiDose(file, assocScanNum, planC)` |
| Load / save the container | `pc.loadFromH5(h5)`, `pc.saveToH5(planC, h5, scanNumV=..., structNumV=...)` |
| Arrays into planC | `pc.importScanArray(scan3M, xV, yV, zV, modality, assocScanNum, planC)`, `pc.importStructureMask(mask3M, assocScanNum, name, planC)`, `pc.importDoseArray(dose3M, xV, yV, zV, planC, assocScanNum)` |
| Structure by name | `names = [s.structureName for s in planC.structure]`; `cerrStr.getMatchingIndex(name, names, 'exact')` |
| Structure mask | `mask3M = rs.getStrMask(structNum, planC)` (bool, on the grid of its associated scan) |
| DVH | `dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)`, then `doseBinsV, volsHistV = dvh.doseHist(dosesV, volsV, 0.05)` |
| DVH metrics | `dvh.meanDose`, `dvh.minDose`, `dvh.maxDose`, `dvh.Dx`, `dvh.Vx`, `dvh.MOHx`, `dvh.MOCx`, `dvh.eud` (all take `doseBinsV, volsHistV, ...`) |
| Radiomics features | `featDict, diagDict = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)`; `ibsi1.writeFeaturesToFile(featDict, csv)` |
| Texture / filter map | `planC = texture_utils.generateTextureMapFromPlanC(planC, scanNum, structNum, settingsFile)`; the map is the new last scan |
| Copy a structure to another scan | `planC = cerrStr.copyToScan(structNum, scanNum, planC)` |
| Registration (ANTs) | `cerr.registration.register.registerScans(...)`, then `warpScan`, `warpStructures`, `warpDose` (see `references/user_guide/registration.rst`) |
| DICOM RTSTRUCT export | `cerr.dcm_export.rtstruct_iod.create(structNumV, filePath, planC)` |
| NIfTI export | `planC.scan[i].saveNii(f)`, `planC.structure[i].saveNii(f, planC)`, `planC.dose[i].saveNii(f)`, label map: `pc.saveNiiStructure(f, labelDict, planC)` |
| SimpleITK image | `.getSitkImage()` on a scan or dose; `.getSitkImage(planC)` on a structure |

## Core recipes

```python
planC = pc.loadDcmDir(r'/path/to/patient')
names = [s.structureName for s in planC.structure]
structNum = cerrStr.getMatchingIndex('PTV', names, 'exact')[0]
scanNum = planC.structure[structNum].getStructureAssociatedScan(planC)

# DVH metrics for dose 0
dosesV, volsV, isErr = dvh.getDVH(structNum, 0, planC)
doseBinsV, volsHistV = dvh.doseHist(dosesV, volsV, 0.05)
meanGy = dvh.meanDose(doseBinsV, volsHistV)
d95Gy  = dvh.Dx(doseBinsV, volsHistV, 95, 1)    # D95%: volume given in percent (1, default)
d2ccGy = dvh.Dx(doseBinsV, volsHistV, 2, 0)     # D2cc: volume given in cc (0)
v20cc  = dvh.Vx(doseBinsV, volsHistV, 20, 0)    # V20Gy in cc (0, default)
v20pct = dvh.Vx(doseBinsV, volsHistV, 20, 1) * 100   # V20Gy as % (1 returns a fraction 0-1)
```

## Conventions and pitfalls

- **`Dx` and `Vx` default to different units.** `Dx(…, volumeType=1)` takes the
  volume in % by default; `Vx(…, volumeType=0)` returns cc by default.
  - "How many cc get 50 Gy" is `Vx(…, 50, 0)`; "V20 in %" is `Vx(…, 20, 1) * 100`.
  - Say which unit you report.
- **Structure names vary between patients:** `Parotid_L`, `L Parotid`,
  `PAROTID_LT`. List `names` first, and match ignoring case, spaces and `_`.
  When several match or none does, show the user the list rather than guessing.
- **Masks live on the structure's own scan.** Use
  `getStructureAssociatedScan`; index that scan's array with the mask, not
  scan 0 by habit.
- **Texture maps and features are different things.**
  - `generateTextureMapFromPlanC` adds a filtered **image** as a new scan,
    cropped to the structure plus a margin. The original scan is unchanged.
  - `computeScalarFeatures` returns **numbers**. Adding a filter under
    `imageType` in its settings file computes features on the filtered image.
- **LoG parameters:** `Sigma_mm` is the scale and `CutOff_mm` the kernel
  half-width. Use `CutOff_mm` ≈ 4 × `Sigma_mm` (e.g. 3 and 12); a cutoff equal
  to sigma truncates the kernel. Settings fields and filters are in
  `references/user_guide/radiomics.rst` and `texture.rst`.
- **Window/level is (center, width).** "W 1600 / L −550" is
  `set_window_level(-550, 1600)`; the Lung preset is (−500, 1500).
- **Dose grids differ from scan grids.** `getDVH` and `getDoseAt` handle the
  interpolation; don't index a dose array with a scan mask.
- **HDF5 is the way to save and reload a whole planC.** NIfTI and DICOM
  exports carry one object type each.
- **Optional extras:** the desktop viewer needs `pycerr[viewer]` and
  registration needs `pycerr[ants]`. If an import fails, say which extra to
  install.

## Viewing: the pyCERR desktop viewer

Use the Qt desktop viewer, `cerr.viewer.pycerr_gui`:

```python
from cerr.viewer.pycerr_gui import launch, show
planC = launch(planC)          # blocks until the window closes; returns the edited planC
v = show(planC)                # non-blocking (IPython/Jupyter: run %gui qt first)
v.set_window_preset('Lung'); v.goto_structure(structNum); v.set_dose(0)
```

`references/viewer_desktop.md` lists its scripting API, including:
- layouts and window presets;
- comparing two scans (registration QA modes; second window with locked slices);
- the DVF overlay, contouring and 3D volume rendering;
- screenshots.

## References (read the one you need)

| File | Covers |
|---|---|
| `references/quickstart.rst` | a first session end to end |
| `references/user_guide/planc.rst` | data model, loaders, UIDs, coordinates, saving |
| `references/user_guide/dvh.rst` | DVH computation, metrics, plan sums, fraction correction |
| `references/user_guide/radiomics.rst` | settings file, preprocessing, feature classes, cohorts |
| `references/user_guide/texture.rst` | filters (LoG, Gabor, Laws, wavelets, mean, Sobel), maps |
| `references/user_guide/registration.rst`, `registration_qa.rst` | ANTs registration, warping, DVF QA |
| `references/user_guide/segmentation.rst`, `ai_inference.rst` | AI segmentation pipelines |
| `references/user_guide/roe.rst` | NTCP/TCP outcome models |
| `references/user_guide/smoothing.rst`, `structure_consensus.rst`, `segmentation_metrics.rst`, `body_outline.rst` | structure processing and comparison |
| `references/end_to_end.rst` | one workflow from import to outcome model |
| `references/viewer_desktop.md` | desktop viewer scripting |
<!-- examples:table -->
