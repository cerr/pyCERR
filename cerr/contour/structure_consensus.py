"""structure_consensus module.

Tools for comparing multiple observer segmentations of the same object on the
same scan and for generating a consensus segmentation.  This is the pyCERR
counterpart of MATLAB CERR's ``structCompare.m``
(``CERR_core/Contouring/StructureConsensus``).

Given ``N`` structures that all delineate the same target on the same scan the
routines here compute:

  * a per-voxel observer-agreement fraction (fraction of observers that
    included each voxel),
  * a STAPLE probabilistic consensus (Warfield et al., IEEE TMI 2004) together
    with each observer's estimated sensitivity and specificity,
  * Fleiss' kappa inter-observer agreement,
  * a per-voxel reliability map, and
  * agreement / volume statistics at a range of agreement thresholds,

and can add consensus segmentations back into the plan container so they can be
visualized, exported or used like any other structure.

Typical use::

    from cerr.contour import structure_consensus as sc

    # Inspect agreement between observer contours 0, 1, 2
    result = sc.compareStructures([0, 1, 2], planC)
    print(sc.summaryText(result))

    # Add a majority-vote and a STAPLE consensus structure to planC
    planC, _ = sc.createConsensusStructure([0, 1, 2], planC,
                                           method='majority')
    planC, _ = sc.createConsensusStructure([0, 1, 2], planC,
                                           method='staple', threshold=0.5)
"""

import numpy as np
import SimpleITK as sitk

import cerr.contour.rasterseg as rs
import cerr.dataclasses.scan as scn


# ---------------------------------------------------------------------------#
#  Helpers
# ---------------------------------------------------------------------------#
def _getObserverMasks(structNumV, planC):
    """Collect binary masks for a set of structures and validate them.

    Args:
        structNumV (list[int]): indices into ``planC.structure``.
        planC (cerr.plan_container.PlanC): pyCERR plan container.

    Returns:
        tuple: ``(maskList, assocScanNum, structNames)`` where ``maskList`` is a
        list of boolean ``np.ndarray`` masks (one per structure, all the same
        shape), ``assocScanNum`` is the index of the shared associated scan, and
        ``structNames`` is the list of structure names.

    Raises:
        ValueError: if fewer than two structures are supplied or if the
            structures are not all associated with the same scan.
    """
    structNumV = [int(s) for s in structNumV]
    if len(structNumV) < 2:
        raise ValueError("At least two structures are required to build a "
                         "consensus.")

    assocScans = [scn.getScanNumFromUID(planC.structure[s].assocScanUID, planC)
                  for s in structNumV]
    if len(set(assocScans)) != 1:
        raise ValueError("All structures must be associated with the same scan. "
                         "Got scan indices: %s" % assocScans)
    assocScanNum = assocScans[0]

    maskList = []
    structNames = []
    refShape = None
    for s in structNumV:
        mask3M = rs.getStrMask(s, planC).astype(bool)
        if refShape is None:
            refShape = mask3M.shape
        elif mask3M.shape != refShape:
            raise ValueError("Structure masks have inconsistent shapes: "
                             "%s vs %s" % (refShape, mask3M.shape))
        maskList.append(mask3M)
        structNames.append(planC.structure[s].structureName)

    return maskList, assocScanNum, structNames


def _boundingBox(maskList):
    """Return slice objects for the union bounding box of a list of masks.

    Restricting the STAPLE / kappa computation to the union bounding box (rather
    than the whole scan) matches CERR's ``structCompare`` behavior and avoids
    background voxels dominating the statistics.

    Args:
        maskList (list[np.ndarray]): list of boolean masks with the same shape.

    Returns:
        tuple[slice, slice, slice] or None: bounding-box slices, or ``None`` if
        every mask is empty.
    """
    union = np.zeros(maskList[0].shape, dtype=bool)
    for m in maskList:
        union |= m
    if not union.any():
        return None
    coords = np.where(union)
    slices = tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)
    return slices


def computeStaple(maskList, foregroundValue=1.0):
    """Run the STAPLE algorithm on a list of binary observer masks.

    Args:
        maskList (list[np.ndarray]): list of boolean masks (same shape).
        foregroundValue (float): label value treated as foreground.

    Returns:
        tuple: ``(staple3M, sensitivity, specificity)`` where ``staple3M`` is a
        float probability map with the same shape as the input masks, and
        ``sensitivity`` / ``specificity`` are per-observer arrays estimated by
        STAPLE.
    """
    sitkImgs = [sitk.GetImageFromArray(m.astype(np.uint8)) for m in maskList]
    stapler = sitk.STAPLEImageFilter()
    stapler.SetForegroundValue(float(foregroundValue))
    out = stapler.Execute(sitkImgs)
    staple3M = sitk.GetArrayFromImage(out).astype(np.float32)
    sensitivity = np.asarray(stapler.GetSensitivity(), dtype=float)
    specificity = np.asarray(stapler.GetSpecificity(), dtype=float)
    return staple3M, sensitivity, specificity


def computeFleissKappa(rateMat):
    """Compute Fleiss' kappa for a binary observer rating matrix.

    Args:
        rateMat (np.ndarray): array of shape ``(nVoxels, nObservers)`` of 0/1
            ratings.

    Returns:
        float: Fleiss' kappa (``nan`` if it is undefined, e.g. perfect chance
        agreement).
    """
    rateMat = np.asarray(rateMat)
    N, n = rateMat.shape
    if N == 0 or n < 2:
        return float("nan")
    # Category counts per subject (voxel): column 0 = "out", column 1 = "in".
    nIn = rateMat.sum(axis=1)
    nOut = n - nIn
    # Proportion of all ratings assigned to each category.
    p_in = nIn.sum() / (N * n)
    p_out = nOut.sum() / (N * n)
    P_e = p_in ** 2 + p_out ** 2
    # Per-subject agreement.
    P_i = (nIn ** 2 + nOut ** 2 - n) / (n * (n - 1))
    P_bar = P_i.mean()
    denom = 1.0 - P_e
    if denom <= 0:
        return float("nan")
    return float((P_bar - P_e) / denom)


# ---------------------------------------------------------------------------#
#  Main comparison
# ---------------------------------------------------------------------------#
def compareStructures(structNumV, planC, stapleForeground=1.0,
                      runStaple=True):
    """Compare multiple observer segmentations of the same target.

    Args:
        structNumV (list[int]): indices into ``planC.structure`` of the observer
            segmentations to compare (all must be on the same scan).
        planC (cerr.plan_container.PlanC): pyCERR plan container.
        stapleForeground (float): foreground label for the STAPLE algorithm.
        runStaple (bool): if ``False`` skip the (relatively expensive) STAPLE
            estimation; the STAPLE fields in the result are then ``None``.

    Returns:
        dict: results with the following keys

            * ``structNumV`` / ``structNames`` / ``assocScanNum``
            * ``numObservers`` (int)
            * ``voxelVolume_cc`` (float)
            * ``agreementFraction3M`` (np.ndarray, float, full scan size) -
              fraction of observers including each voxel
            * ``staple3M`` (np.ndarray or None) - STAPLE probability map
            * ``reliability3M`` (np.ndarray, float) - per-voxel reliability
            * ``sensitivity`` / ``specificity`` (np.ndarray or None) - per
              observer STAPLE estimates
            * ``kappa`` (float) - Fleiss' kappa over the union bounding box
            * ``observerVolumes_cc`` (np.ndarray) - volume of each observer
            * ``volumeStats`` (dict) - min/max/mean/std of observer volumes (cc)
            * ``agreementThresholds`` (np.ndarray) - fractions ``k/N``
            * ``volumeByAgreement_cc`` (np.ndarray) - agreement-mask volume at
              each threshold
            * ``stapleVolumeByThreshold_cc`` (np.ndarray or None) - STAPLE volume
              at each threshold
            * ``intersection_cc`` / ``union_cc`` (float)
    """
    maskList, assocScanNum, structNames = _getObserverMasks(structNumV, planC)
    nObs = len(maskList)
    fullShape = maskList[0].shape

    spacing = planC.scan[assocScanNum].getScanSpacing()  # cm
    voxelVol = float(np.prod(spacing))  # cc

    # Per-voxel agreement fraction over the whole scan.
    agreeCount = np.zeros(fullShape, dtype=np.float32)
    for m in maskList:
        agreeCount += m
    agreementFraction3M = agreeCount / nObs

    # Per-observer volumes.
    observerVolumes = np.asarray([float(m.sum()) * voxelVol for m in maskList])

    # Restrict STAPLE / kappa / reliability to the union bounding box.
    bbox = _boundingBox(maskList)
    staple3M = None
    sensitivity = None
    specificity = None
    reliability3M = np.zeros(fullShape, dtype=np.float32)
    kappa = float("nan")

    if bbox is not None:
        cropMasks = [m[bbox] for m in maskList]
        rateMat = np.stack([m.ravel() for m in cropMasks],
                           axis=1).astype(np.float32)  # (nVox, nObs)

        # Fleiss' kappa over the cropped region.
        kappa = computeFleissKappa(rateMat)

        # Reliability map (cf. structCompare): chance-corrected mean rating.
        raterProb = rateMat.mean(axis=0)                    # per observer
        chanceProb = np.sqrt(raterProb * (1.0 - raterProb))  # per observer
        with np.errstate(divide="ignore", invalid="ignore"):
            corrected = (rateMat - chanceProb) / (1.0 - chanceProb)
        corrected[~np.isfinite(corrected)] = 0.0
        reliabilityCrop = corrected.mean(axis=1).reshape(cropMasks[0].shape)
        reliability3M[bbox] = reliabilityCrop.astype(np.float32)

        if runStaple:
            stapleCrop, sensitivity, specificity = computeStaple(
                cropMasks, foregroundValue=stapleForeground)
            staple3M = np.zeros(fullShape, dtype=np.float32)
            staple3M[bbox] = stapleCrop

    # Agreement thresholds k/N for k = 1 .. N.
    kV = np.arange(1, nObs + 1)
    agreementThresholds = kV / nObs
    volumeByAgreement = np.asarray(
        [float((agreeCount >= k).sum()) * voxelVol for k in kV])
    intersection_cc = volumeByAgreement[-1]   # all observers agree
    union_cc = volumeByAgreement[0]           # at least one observer

    stapleVolumeByThreshold = None
    if staple3M is not None:
        stapleVolumeByThreshold = np.asarray(
            [float((staple3M >= t).sum()) * voxelVol
             for t in agreementThresholds])

    volumeStats = {
        "min": float(observerVolumes.min()),
        "max": float(observerVolumes.max()),
        "mean": float(observerVolumes.mean()),
        "std": float(observerVolumes.std()),
    }

    return {
        "structNumV": [int(s) for s in structNumV],
        "structNames": structNames,
        "assocScanNum": int(assocScanNum),
        "numObservers": nObs,
        "voxelVolume_cc": voxelVol,
        "agreementFraction3M": agreementFraction3M,
        "staple3M": staple3M,
        "reliability3M": reliability3M,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "kappa": kappa,
        "observerVolumes_cc": observerVolumes,
        "volumeStats": volumeStats,
        "agreementThresholds": agreementThresholds,
        "volumeByAgreement_cc": volumeByAgreement,
        "stapleVolumeByThreshold_cc": stapleVolumeByThreshold,
        "intersection_cc": intersection_cc,
        "union_cc": union_cc,
    }


# ---------------------------------------------------------------------------#
#  Consensus mask / structure generation
# ---------------------------------------------------------------------------#
_METHODS = ("staple", "majority", "union", "intersection", "agreement")


def getConsensusMask(result, method="staple", threshold=0.5):
    """Derive a binary consensus mask from a :func:`compareStructures` result.

    Args:
        result (dict): output of :func:`compareStructures`.
        method (str): one of

            * ``'staple'`` - threshold the STAPLE probability map at ``threshold``
            * ``'majority'`` - voxels included by more than half of observers
            * ``'agreement'`` - voxels where the observer-agreement fraction is
              ``>= threshold``
            * ``'union'`` - voxels included by at least one observer
            * ``'intersection'`` - voxels included by all observers

        threshold (float): probability / agreement-fraction threshold used by the
            ``'staple'`` and ``'agreement'`` methods (0-1).

    Returns:
        np.ndarray: boolean consensus mask with full scan shape.

    Raises:
        ValueError: for an unknown ``method`` or if STAPLE was not computed.
    """
    method = method.lower()
    if method not in _METHODS:
        raise ValueError("Unknown method '%s'. Choose from %s."
                         % (method, ", ".join(_METHODS)))
    frac = result["agreementFraction3M"]
    if method == "staple":
        if result["staple3M"] is None:
            raise ValueError("STAPLE map is not available in this result "
                             "(was compareStructures run with runStaple=False?).")
        return result["staple3M"] >= threshold
    if method == "majority":
        return frac > 0.5
    if method == "agreement":
        return frac >= threshold
    if method == "union":
        return frac > 0.0
    # intersection
    return frac >= 1.0


def createConsensusStructure(structNumV, planC, method="staple", threshold=0.5,
                             structName=None, result=None, importResult=True):
    """Compute a consensus segmentation and (optionally) add it to ``planC``.

    Args:
        structNumV (list[int]): observer structure indices to combine.
        planC (cerr.plan_container.PlanC): pyCERR plan container.
        method (str): consensus method, see :func:`getConsensusMask`.
        threshold (float): threshold for ``'staple'`` / ``'agreement'`` methods.
        structName (str or None): name for the new structure. A descriptive
            default is generated when ``None``.
        result (dict or None): a previously computed :func:`compareStructures`
            result to reuse (avoids recomputation); computed when ``None``.
        importResult (bool): if ``True`` import the consensus mask as a new
            structure in ``planC``; if ``False`` only return the mask.

    Returns:
        tuple: ``(planC, structNum)`` when ``importResult`` is ``True`` (with
        ``structNum`` the index of the new structure), otherwise
        ``(mask3M, result)``.
    """
    if result is None:
        runStaple = (method.lower() == "staple")
        result = compareStructures(structNumV, planC, runStaple=runStaple)
    mask3M = getConsensusMask(result, method=method, threshold=threshold)

    if not importResult:
        return mask3M, result

    if structName is None:
        if method.lower() in ("staple", "agreement"):
            structName = "Consensus_%s_%g" % (method.upper(), threshold)
        else:
            structName = "Consensus_%s" % method.upper()

    from cerr.dataclasses import structure as structr
    planC = structr.importStructureMask(
        mask3M, result["assocScanNum"], structName, planC, None)
    return planC, len(planC.structure) - 1


# ---------------------------------------------------------------------------#
#  Reporting
# ---------------------------------------------------------------------------#
def summaryText(result):
    """Return a human-readable multi-line summary of a comparison result.

    Args:
        result (dict): output of :func:`compareStructures`.

    Returns:
        str: formatted summary suitable for printing or a GUI text box.
    """
    lines = []
    lines.append("Structure consensus comparison")
    lines.append("=" * 34)
    lines.append("Observers (%d):" % result["numObservers"])
    for name, vol in zip(result["structNames"], result["observerVolumes_cc"]):
        lines.append("    %-28s %9.3f cc" % (name, vol))
    vs = result["volumeStats"]
    lines.append("")
    lines.append("Observer volume  min/max/mean/std (cc): "
                 "%.3f / %.3f / %.3f / %.3f"
                 % (vs["min"], vs["max"], vs["mean"], vs["std"]))
    lines.append("Union volume (>=1 observer):        %9.3f cc"
                 % result["union_cc"])
    lines.append("Intersection volume (all observers):%9.3f cc"
                 % result["intersection_cc"])
    lines.append("Fleiss' kappa:                      %9.4f"
                 % result["kappa"])
    if result["sensitivity"] is not None:
        sens = result["sensitivity"]
        spec = result["specificity"]
        lines.append("STAPLE sensitivity  mean/std: %.4f / %.4f"
                     % (sens.mean(), sens.std()))
        lines.append("STAPLE specificity  mean/std: %.4f / %.4f"
                     % (spec.mean(), spec.std()))
    lines.append("")
    lines.append("Volume vs. agreement threshold:")
    lines.append("  %-14s %-14s %-14s" % ("threshold", "agreement(cc)",
                                          "STAPLE(cc)"))
    stapleVol = result["stapleVolumeByThreshold_cc"]
    for i, t in enumerate(result["agreementThresholds"]):
        sv = ("%.3f" % stapleVol[i]) if stapleVol is not None else "-"
        lines.append("  %-14.3f %-14.3f %-14s"
                     % (t, result["volumeByAgreement_cc"][i], sv))
    return "\n".join(lines)
