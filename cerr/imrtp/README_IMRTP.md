# IMRTP GUI for pyCERR

Python port of Matlab CERR's `IMRTP/IMRTPGui.m`
(https://github.com/cerr/CERR/blob/master/IMRTP/IMRTPGui.m) for pyCERR.

## Files

```
cerr/imrtp/__init__.py        package entry; exposes IMRTPGui(planC)
cerr/imrtp/imrtp_problem.py   IM data model (port of initIMRTProblem.m):
                              IMRTProblem, IMBeam, IMGoal, IMParams,
                              VMCParams + conditionBeam / addEquispacedBeams
cerr/imrtp/imrtp.py           runIMRTP driver (port of IMRTP.m flow),
                              doseToPlanC (port of dose2CERR.m),
                              registerEngine / availableEngines,
                              built-in 'PB-Demo' engine
cerr/imrtp/imrtp_gui.py       the Qt GUI (port of IMRTPGui.m)
cerr/imrtp/dosecalc/          dose-calculation engines:
  __init__.py                 engine registration, qibEngine, recalcDose
  qib.py                      getQIBDose / GetPBConsts / compression ports
  qib_data.py                 QIB kernel data loader (loadPBData.m)
  raytrace.py                 getPBRays.m / getPBRayData.m ports
  influence.py                generateQIBInfluence.m / getIMDose.m ports
  recalc.py                   recalcDose (call_doseRecal.m port)
  montecarlo.py               DPM / VMC++ placeholders (recompDose)
  QIBData/                    Ahnesjo kernel tables (6 / 18 MV)
demo_imrtp_gui.py             runnable demo (synthetic plan or DICOM dir)
```

## Install

Copy the `cerr/imrtp` folder into your pyCERR source tree so it becomes:

    C:\software\pyCERR_master\pyCERR\cerr\imrtp\

If pyCERR is installed in editable mode (`pip install -e .`) nothing else is
needed. Otherwise re-run `pip install .` from `C:\software\pyCERR_master\pyCERR`,
or copy `imrtp` into the installed package
(`...\Lib\site-packages\cerr\imrtp`).

Requirements beyond core pyCERR: a Qt binding (already present if you
installed `pyCERR[napari]`; else `pip install PyQt5 qtpy`) and matplotlib
(a core pyCERR dependency).

## Usage

```python
from cerr import plan_container as pc
from cerr.imrtp import IMRTPGui

planC = pc.loadDcmDir(r'C:/data/myPlan')
IMRTPGui(planC)                 # blocks until the window closes
# inside napari / an existing Qt loop:
gui = IMRTPGui(planC, block=False)
```

Panel-by-panel mapping to IMRTPGui.m:

| Matlab frame            | pyCERR panel        | Behaviour |
|-------------------------|---------------------|-----------|
| Beams ("bl")            | Beams               | beam list, BEV checkboxes, New / Equispaced / Delete |
| Geometry Preview ("bg") | Geometry Preview    | axial CT at isocenter, 100 cm gantry circle; click-drag rotates the selected beam's gantry angle |
| Select Scan             | Select Scan         | changing scan clears goals and beamlets |
| Structures ("ss")       | Structures          | colored rows; isTarg / marg (PBMargin) / sampRate; "-" removes the goal |
| Beam Parameters ("bp")  | Beam Parameters     | all 20 fields incl. sigma_100; checkboxes toggle auto-calculation (isocenter from target COM; xRel = SAD*sin(g), yRel = SAD*cos(g)) |
| IM Parameters ("ip")    | IM Parameters       | algorithm, DoseTerm, ScatterMethod, thresholds, sampling |
| VMC Parameters ("mc")   | VMC Parameters      | the 12 Monte-Carlo fields |
| IM Dosimetry set ("ib") | IM Dosimetry set    | browse/delete/rename IM sets stored on planC.im |
| File ("us")             | File                | Recompute & add / Recompute & overwrite / Copy-Add / Overwrite / Revert; Go, Show, Exit |
| Status ("wb")           | Status              | message + progress bar |

Edits that invalidate computed beamlets (gantry drag, margins, goals, scan
change, ...) clear them and retitle the window
`IMRTP *(beamlets may be stale)`, matching the Matlab behaviour.

## Where results go

* `planC.im`     - list of saved `IMRTProblem` sets (this attribute is added
                   to the plan container on first use)
* `planC.dose`   - computed dose appended via `imrtp.doseToPlanC`
                   (fractionGroupID = the IM set name); 'Show' opens it in
                   the napari viewer with the goal structures.

## Dose engines

The GUI's *algorithm* list is driven by a registry. Engines shipped with
this package:

| Engine    | Status        | Notes                                          |
|-----------|---------------|------------------------------------------------|
| `QIB`     | implemented   | photon pencil beam, port of CERR's QIB chain    |
| `DPM`     | placeholder   | Monte Carlo; see `cerr/imrtp/dosecalc/montecarlo.py` |
| `VMC++`   | placeholder   | Monte Carlo; see `cerr/imrtp/dosecalc/montecarlo.py` |
| `PB-Demo` | demo only     | geometric exerciser, not dosimetric             |

Custom engines can be registered:

```python
from cerr.imrtp import registerEngine

def myEngine(im, planC, status):
    # im: IMRTProblem (beams conditioned, goals resolved)
    # return dose3M on the scan grid (rows x cols x slices)
    ...
registerEngine('MyEngine', myEngine)
```

## QIB dose calculation (`cerr.imrtp.dosecalc`)

Python port of Matlab CERR's QIB (Quadrant Infinite Beam) photon
pencil-beam algorithm - the chain `IMRTP.m -> getPBRayData.m ->
generateQIBInfluence.m -> getQIBDose.m -> getIMDose.m` - plus a
`recalcDose()` driver mirroring `IMRTP/recompDose/call_doseRecal.m`.

The kernel is the Ahnesjo analytical pencil-beam fit
(Med. Phys. 19, 263-273 (1992)): `A(z)exp(-a r)/r + B(z)exp(-b r)/r` with
fast lookup of precomputed quadrant infinite integrals, and an optional
Gaussian source smear (`beam.sigma_100`). Kernel data ships in
`cerr/imrtp/dosecalc/QIBData/` (`aahn_6b.dat`, `aahn_18b.dat`,
`qib_tables.npz`, converted from the CERR `.mat` tables; auto-downloaded
from the CERR GitHub if missing).

Usage from the GUI: select algorithm `QIB`, mark a target, add beams, Go.

Usage from code (the `call_doseRecal.m` workflow):

```python
from cerr.imrtp import initIMRTProblem, addGoal, addEquispacedBeams
from cerr.imrtp.dosecalc import recalcDose

im = initIMRTProblem(planC)
g = addGoal(im, 0, planC); g.isTarget = 'yes'     # PTV
addEquispacedBeams(im, 5, 0.0, planC)
for b in im.beams:
    b.beamEnergy = 6                              # 6 or 18 MV only

doseNum, dose3D, im = recalcDose(
    planC, im,
    structNumsV=[0, 1, 2],   # planC.structure indices to compute dose in
    sampleRateV=[2, 2, 8],   # e.g. targets/critical 2, skin 8 (powers of 2)
    imrtpName='QIB reCalc',
    ptvStructNum=0,          # optional: scale to a clinical dose...
    clinicDoseNum=0,         # ...by matching this dose's PTV metric
    scaleMetric='meanDose')  # or 'Dx' (dxPercent=98)
```

Key behaviors carried over from Matlab:

* Dose is computed only at the requested structures' sample points
  (influence-matrix style), with the structure surface always included.
* `IM Parameters > DoseTerm` selects the kernel components: `primary`,
  `scatter`, `nogauss+scatter`, `GaussPrimary`,
  `GaussPrimary+scatter` (default).
* `ScatterMethod` (`exponential` / `random` / `threshold`) +
  `Scatter.Threshold` compress low-dose scatter in the influence
  (`applyIMRTCompression.m`); `call_doseRecal.m` used threshold 0.1.
* `beamletWeightsV` (one weight per pencil beam, beam-by-beam) applies
  leaf-sequence/IMRT weights; `None` gives open (unit) fields.
* Only 6 and 18 MV kernel tables exist; other energies raise an error.
* Radiological depth is from a 500 cm ray trace at
  `params.numCTSamplePts` samples, water = CT number 1000.

Deviations from Matlab noted in docstrings: sub-sampled structures are
filled from the nearest sampled point (Matlab: 3-D linear interpolation),
and the optional CT downsample step of `call_doseRecal.m` is not ported.

**The QIB port is research code and has not been clinically validated.
Do not use for clinical decision making.**

## Monte Carlo placeholders

`cerr/imrtp/dosecalc/montecarlo.py` defines the intended interfaces for
the DPM / VMC++ engines of CERR's `IMRTP/recompDose` toolbox
(`beam2MCdose`, `calcDoseByBeamMeterset`, `MCRecompParams` with the
`call_doseRecal.m` defaults: leak 0.032, `6MV10x10MDA.spectrum`,
nhist 1e5, ...). They raise `NotImplementedError` with pointers to the
Matlab references: https://github.com/cerr/CERR/tree/master/IMRTP/recompDose

A self-contained `'PB-Demo'` divergent-beam engine also ships so the full
workflow (Go -> dose -> Show) is testable without the QIB data. It is a
geometric demonstration only - NOT dosimetrically validated.
