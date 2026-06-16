"""
run_qib_from_rtplan.py
======================

Set up and run pyCERR's QIB dose calculation on a DICOM dataset, using the
beams from its RTPLAN.

Configured for:
    C:\\Users\\aptea\\Downloads\\0617-381906_09-09-2000-87022

What it does
------------
1. Loads the DICOM directory (CT + RTSTRUCT + RTPLAN) into a planC.
2. Lists the structures so you can confirm the target / OAR indices.
3. Builds an IMRTProblem from the RTPLAN beams (gantry / couch /
   collimator angles, isocenter, SAD and nominal energy per beam) via
   ``cerr.imrtp.dosecalc.imFromRTPlan`` -- the geometry import of CERR's
   ``recompDose/beam2MCdose.m``.
4. Runs the QIB pencil-beam dose (``recalcDose``, the call_doseRecal.m
   workflow) over the chosen structures / sample rates.
5. Adds the dose to planC and (optionally) scales it to an existing
   clinical RTDOSE by matching the PTV mean dose.

EDIT the CONFIG block below (at least PTV_NAME / structure choices) to
match this dataset, then run:

    python run_qib_from_rtplan.py

Notes
-----
* QIB kernel tables exist for 6 and 18 MV only; other nominal energies are
  mapped to the nearest of these (a warning is printed).
* This is a research pencil-beam port and is NOT clinically validated.
"""

import os
import numpy as np

from cerr import plan_container as pc
from cerr import viewer as vwr
from cerr.contour import rasterseg as rs
from cerr.imrtp.dosecalc import imFromRTPlan, recalcDose

# ----------------------------------------------------------------------
# CONFIG  -- edit these for your dataset
# ----------------------------------------------------------------------
DCM_DIR = r"L:\Data\RTOG0617\DICOM\NSCLC-Cetuximab_flat_dirs\0617-381906_09-09-2000-87022"

# Target (PTV) the beams must cover.  Give its name (case-insensitive,
# substring match) OR set PTV_STRUCT_NUM to a planC.structure index.
PTV_NAME = "PTV"
PTV_STRUCT_NUM = None

# Structures to compute dose in, and their in-plane sample rates (powers
# of 2; e.g. skin/body 8, targets & critical structures 2).  Names are
# matched case-insensitively; unknown names are skipped with a warning.
# Leave DOSE_STRUCTS = None to compute the PTV only.
DOSE_STRUCTS = ["SKIN"]     # e.g. ["PTV", "Cord", "Skin"]
SAMPLE_RATES = [4]     # e.g. [2, 2, 8]  (same length as DOSE_STRUCTS)

BEAMS_NUM = 0           # which RTPLAN in planC.beams
SCAN_NUM = 0            # associated CT scan
BEAMLET_SIZE_CM = 1.0   # beamlet width at isocenter
PB_MARGIN_CM = 0.5      # beamlet margin around the target

# Scale the recomputed dose to a clinical RTDOSE by PTV mean dose?
# Set CLINIC_DOSE_NUM to the planC.dose index of the clinical dose, or
# None to leave the QIB result unscaled.
CLINIC_DOSE_NUM = 0
SCALE_METRIC = "meanDose"   # or "Dx" (with DX_PERCENT)
DX_PERCENT = 98.0

SAVE_PLANC = ""         # path to a .pkl to save the result, or "" to skip
# ----------------------------------------------------------------------


def _findStruct(planC, name):
    name = name.lower()
    hits = [i for i, s in enumerate(planC.structure)
            if name in s.structureName.lower()]
    return hits[0] if hits else None


def main():
    if not os.path.isdir(DCM_DIR):
        raise SystemExit("DICOM directory not found:\n  %s" % DCM_DIR)

    print("Loading DICOM from:\n  %s" % DCM_DIR)
    planC = pc.loadDcmDir(DCM_DIR)
    print("  scans: %d | structures: %d | doses: %d | RTPLANs: %d"
          % (len(planC.scan), len(planC.structure), len(planC.dose),
             len(planC.beams)))
    if not len(planC.beams):
        raise SystemExit("No RTPLAN found in this directory; cannot use "
                         "the plan's beams.")

    print("\nStructures:")
    for i, s in enumerate(planC.structure):
        print("  [%2d] %s" % (i, s.structureName))

    # Report the RTPLAN beams.
    plan = planC.beams[BEAMS_NUM]
    print("\nRTPLAN '%s' beams:" % getattr(plan, "RTPlanLabel", BEAMS_NUM))
    for bs in np.atleast_1d(plan.BeamSequence):
        cp = np.atleast_1d(bs.ControlPointSequence)[0]
        print("  #%s %-14s gantry %6.1f  couch %5.1f  energy %s  "
              "SAD %.0fmm  (%s)"
              % (getattr(bs, "BeamNumber", "?"),
                 getattr(bs, "BeamName", ""),
                 float(getattr(cp, "GantryAngle", 0.0)),
                 float(getattr(cp, "PatientSupportAngle", 0.0) or 0.0),
                 getattr(cp, "NominalBeamEnergy", "?"),
                 float(getattr(bs, "SourceAxisDistance", 0.0) or 0.0),
                 getattr(bs, "TreatmentDeliveryType", "TREATMENT")))

    # Resolve the target.
    ptvNum = PTV_STRUCT_NUM
    if ptvNum is None:
        ptvNum = _findStruct(planC, PTV_NAME)
    if ptvNum is None:
        raise SystemExit("Could not find target structure '%s'. Set "
                         "PTV_NAME or PTV_STRUCT_NUM in the CONFIG block."
                         % PTV_NAME)
    print("\nTarget: [%d] %s"
          % (ptvNum, planC.structure[ptvNum].structureName))

    # Resolve the dose structures / sample rates.
    if DOSE_STRUCTS is None:
        structNumsV = [ptvNum]
        sampleRateV = [2]
    else:
        structNumsV, sampleRateV = [], []
        rates = SAMPLE_RATES or [2] * len(DOSE_STRUCTS)
        for nm, r in zip(DOSE_STRUCTS, rates):
            n = _findStruct(planC, nm)
            if n is None:
                print("  WARNING: structure '%s' not found, skipping." % nm)
                continue
            structNumsV.append(n)
            sampleRateV.append(int(r))
        if ptvNum not in structNumsV:
            structNumsV.insert(0, ptvNum)
            sampleRateV.insert(0, 2)

    print("Computing dose in:",
          [(n, planC.structure[n].structureName, "rate=%d" % r)
           for n, r in zip(structNumsV, sampleRateV)])

    # Build the IM problem from the RTPLAN beams.
    im = imFromRTPlan(planC, beamsNum=BEAMS_NUM, targetStructNum=ptvNum,
                      structNumsV=structNumsV, sampleRateV=sampleRateV,
                      scanNum=SCAN_NUM, PBMargin=PB_MARGIN_CM,
                      beamletDelta=BEAMLET_SIZE_CM)
    print("\nIM problem '%s': %d beams, %d goals (algorithm=%s)"
          % (im.name, len(im.beams), len(im.goals), im.params.algorithm))

    # Run QIB.
    def status(msg, frac=None):
        if frac is None:
            print("  " + msg)
        else:
            print("  [%3.0f%%] %s" % (100 * frac, msg))

    doseNum, dose3D, im = recalcDose(
        planC, im,
        structNumsV=structNumsV, sampleRateV=sampleRateV,
        algorithm="QIB", imrtpName="QIB from RTPLAN",
        ptvStructNum=(ptvNum if CLINIC_DOSE_NUM is not None else None),
        clinicDoseNum=CLINIC_DOSE_NUM,
        scaleMetric=SCALE_METRIC, dxPercent=DX_PERCENT,
        statusCallback=status)

    print("\nDONE. QIB dose added as planC.dose[%d] ('%s')."
          % (doseNum, planC.dose[doseNum].fractionGroupID))
    print("  dose max %.3f Gy, mean-in-PTV %.3f Gy"
          % (dose3D.max(), dose3D[rs.getStrMask(ptvNum, planC)
                                  .astype(bool)].mean()))

    if SAVE_PLANC:
        pc.savePlanC(planC, SAVE_PLANC) if hasattr(pc, "savePlanC") \
            else planC.save_planC(SAVE_PLANC) if hasattr(planC, "save_planC")\
            else None
        print("  saved planC to %s" % SAVE_PLANC)

    print("\nNOTE: QIB is a research pencil-beam port and is NOT "
          "clinically validated.")
    return planC


if __name__ == "__main__":
    planC = main()
    len(planC.dose)
    structNumsV = [_findStruct(planC, strName) for strName in DOSE_STRUCTS]
    structNumsV.append(_findStruct(planC, PTV_NAME))
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
            vwr.showNapari(planC, [0], structNumsV, [0,1], {}, '2d')


