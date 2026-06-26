"""Export urOMT Eulerian maps to NIfTI on the scan grid.

Each per-interval scalar map (speed, effSpeed, rate, Peclet, |flux|) is placed
back into the full scan grid (so it aligns with the scan in any NIfTI viewer)
and written as an **individual 3-D NIfTI file per metric per time interval**,
using the scan's SimpleITK geometry - the same path pyCERR's ``scan.saveNii``
uses, so origin / spacing / direction match the scan exactly.
"""
import os

import numpy as np

# runEULAIntervals keys saved (flux is the vector field -> magnitude)
EULER_METRICS = ["speed", "effSpeed", "rate", "peclet", "flux"]


def _roiMapToScan(roiMap, bbox, scanShape):
    """Place an ROI-grid map into a full scan-grid array (zoom resized runs up
    to the bbox extent first). Mirrors :func:`cerr.uromt.viz.eulerianMapToScan`
    but for any metric array."""
    from scipy.ndimage import zoom
    rs_, re_, cs_, ce_, ss_, se_ = bbox
    target = (re_ - rs_, ce_ - cs_, se_ - ss_)
    m = np.asarray(roiMap, dtype=float)
    if m.shape != tuple(target):
        m = zoom(m, [t / s for t, s in zip(target, m.shape)], order=1)
    full = np.zeros(scanShape, dtype=float)
    full[rs_:re_, cs_:ce_, ss_:se_] = m
    return full


def _scangridToSitk(arr3, scan):
    """Scan-grid 3-D array -> SimpleITK image with the scan's geometry (matches
    ``Scan.getSitkImage``: z,y,x axis order + slice-order flip + CopyInfo)."""
    import SimpleITK as sitk
    from cerr.dataclasses.scan import flipSliceOrderFlag
    sa = np.transpose(np.asarray(arr3, dtype=np.float32), (2, 0, 1))   # z,y,x
    if flipSliceOrderFlag(scan):
        sa = np.flip(sa, axis=0)
    img = sitk.GetImageFromArray(np.ascontiguousarray(sa))
    img.CopyInformation(scan.getSitkImage())
    return img


def saveEulerianMapsNii(eul, planC, scanNum, outDir, prefix="uromt"):
    """Write the per-interval Eulerian scalar maps as NIfTI on the scan grid.

    Args:
        eul (dict): output of :func:`cerr.uromt.analyze.runEULAIntervals`.
        planC: plan container.
        scanNum (int): scan whose geometry the maps are written with (any of the
            co-registered run frames; e.g. ``frameScanNums[0]``).
        outDir (str): output directory (created if needed).
        prefix (str): file-name prefix.

    Returns:
        list[str]: the written file paths - one 3-D NIfTI per metric per time
        interval, named ``<prefix>_<metric>_t<NN>.nii.gz`` (NN = 1-based interval
        index). When the run has a single interval the ``_tNN`` suffix is still
        added (``_t01``).
    """
    import SimpleITK as sitk
    scan = planC.scan[scanNum]
    scanShape = tuple(int(v) for v in scan.getScanArray().shape)
    bbox = eul["bbox"]
    nIv = len(eul["speed"])
    os.makedirs(outDir, exist_ok=True)
    written = []
    for name in EULER_METRICS:
        if name not in eul or not eul[name]:
            continue
        outName = "fluxmag" if name == "flux" else name
        for t in range(nIv):
            arr = eul[name][t]
            if name == "flux":                       # (3,*n) vector -> magnitude
                arr = np.sqrt(np.sum(np.asarray(arr) ** 2, axis=0))
            full = _roiMapToScan(arr, bbox, scanShape)
            img = _scangridToSitk(full, scan)
            path = os.path.join(outDir, "%s_%s_t%02d.nii.gz"
                                % (prefix, outName, t + 1))
            sitk.WriteImage(img, path)
            written.append(path)
    return written


def saveVectorFieldNii(field, n, bbox, planC, scanNum, outDir, prefix):
    """Save a urOMT 3-component vector field (velocity or flux) as NIfTI on the
    scan grid - one scalar file per component.

    Args:
        field: ``(3, prod(n))`` array on the ROI grid; component 0 is along the
            row (y) axis, 1 the column (x) axis, 2 the slice (z) axis (the urOMT
            convention, as returned by ``result["u"][t].mean(axis=2)`` or
            ``Eul["flux"][t]``).
        n (list): ROI grid dims; ``bbox`` the ROI bounding box; ``planC`` /
            ``scanNum`` give the scan geometry; ``outDir`` / ``prefix`` the
            output location.

    Returns:
        list[str]: the three written paths (``..._vrow``, ``_vcol``, ``_vslice``)
        - components along row, column and slice, in the scan grid.
    """
    import SimpleITK as sitk
    from cerr.uromt import viz
    scan = planC.scan[scanNum]
    scanShape = tuple(int(v) for v in scan.getScanArray().shape)
    comps = viz.fieldToScan(field, n, bbox, scanShape)   # [row, col, slice]
    os.makedirs(outDir, exist_ok=True)
    written = []
    for arr, lab in zip(comps, ("vrow", "vcol", "vslice")):
        img = _scangridToSitk(arr, scan)
        path = os.path.join(outDir, "%s_%s.nii.gz" % (prefix, lab))
        sitk.WriteImage(img, path)
        written.append(path)
    return written
