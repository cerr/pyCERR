"""Shared helpers for the scripts that generate the documentation figures.

Every figure in the user guide is produced by a script in this directory from
data bundled with pyCERR, so the figures can be regenerated at any time::

    python docs/_figure_scripts/make_all.py

The scripts need only a core ``pycerr`` install, plus the ``ants`` extra for
``registration.py`` and ``registration_qa.py`` and the ``viewer`` extra for
``registration_qa.py``.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from cerr import datasets                    # noqa: E402
from cerr import plan_container as pc        # noqa: E402
from cerr.contour import rasterseg as rs     # noqa: E402

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "_static", "img")
DATA_DIR = os.path.dirname(datasets.__file__)
PHANTOM_DIR = os.path.join(DATA_DIR, "radiomics_phantom_dicom", "pat_1")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                     "savefig.dpi": 110, "figure.dpi": 110})


def save(fig, name):
    """Write ``fig`` to docs/_static/img/<name>.png and close it."""
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, name + ".png")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", os.path.relpath(path))
    return path


def loadPhantom():
    """Load the bundled lung CT phantom (one CT scan, one structure 'GTV-1')."""
    return pc.loadDcmDir(PHANTOM_DIR)


def addSyntheticDose(planC, structNum=0, scanNum=0, peakGy=60.0):
    """Add a smooth, synthetic dose centred on a structure.

    The bundled datasets carry no clinical RTDOSE, so the dosimetry pages use
    this illustrative distribution: a flat-topped blob that covers the structure
    and falls off into the surrounding tissue. It is NOT a treatment plan.
    """
    xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
    mask3M = rs.getStrMask(structNum, planC)
    rowV, colV, slcV = np.where(mask3M)
    x0, y0, z0 = xV[colV].mean(), yV[rowV].mean(), zV[slcV].mean()
    xM, yM, zM = np.meshgrid(xV, yV, zV)       # (rows, cols, slices)
    r2 = ((xM - x0) / 6.5) ** 2 + ((yM - y0) / 6.5) ** 2 + ((zM - z0) / 5.5) ** 2
    dose3M = peakGy * np.exp(-r2 ** 4)
    planC = pc.importDoseArray(dose3M, xV, yV, zV, planC, scanNum,
                               {"fractionGroupID": "Synthetic", "doseUnits": "GY"})
    return planC


def sliceExtent(xV, yV):
    """``imshow`` extent for an axial slice in pyCERR virtual coordinates (cm)."""
    dx, dy = abs(xV[1] - xV[0]), abs(yV[1] - yV[0])
    return [xV[0] - dx / 2, xV[-1] + dx / 2, yV[-1] - dy / 2, yV[0] + dy / 2]


def showAxial(ax, planC, scanNum, slc, window=(-160, 240), cmap="gray"):
    """Draw one axial slice of a scan on ``ax`` in virtual x/y coordinates."""
    scan3M = planC.scan[scanNum].getScanArray()
    xV, yV, _ = planC.scan[scanNum].getScanXYZVals()
    im = ax.imshow(scan3M[:, :, slc], cmap=cmap, vmin=window[0], vmax=window[1],
                   extent=sliceExtent(xV, yV))
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    return im


def contourMask(ax, planC, scanNum, mask3M, slc, color, label=None, lw=1.5):
    """Outline a binary mask on an axial slice drawn with :func:`showAxial`."""
    xV, yV, _ = planC.scan[scanNum].getScanXYZVals()
    if mask3M[:, :, slc].any():
        ax.contour(xV, yV, mask3M[:, :, slc].astype(float), levels=[0.5],
                   colors=[color], linewidths=lw)
    if label:
        ax.plot([], [], color=color, lw=lw, label=label)


def knownWarpIndices(shape):
    """Moving-scan (row, col, slice) sampled by each fixed voxel under the
    synthetic deformation used by the registration pages: a shift plus a
    smooth sinusoidal warp."""
    nR, nC, nS = shape
    rM, cM, sM = np.meshgrid(np.arange(nR), np.arange(nC), np.arange(nS), indexing="ij")
    rQ = rM + 7 + 5 * np.sin(2 * np.pi * cM / nC)
    cQ = cM - 9 + 5 * np.sin(2 * np.pi * rM / nR)
    sQ = sM + 1.0
    return rQ, cQ, sQ


def knownInverseIndices(shape, nIter=20):
    """Inverse of :func:`knownWarpIndices`: the moving-scan (row, col, slice)
    whose content lands on each fixed voxel. This is what a registration of the
    moving scan to the fixed scan should recover."""
    nR, nC, nS = shape
    rF, cF, sF = np.meshgrid(np.arange(nR), np.arange(nC), np.arange(nS), indexing="ij")
    rM, cM, sM = rF - 7.0, cF + 9.0, sF - 1.0
    for _ in range(nIter):                     # fixed-point iteration converges fast
        rM = rF - 7 - 5 * np.sin(2 * np.pi * cM / nC)
        cM = cF + 9 - 5 * np.sin(2 * np.pi * rM / nR)
    return rM, cM, sM


def warpKnown(vol3M, order):
    """Resample ``vol3M`` through the synthetic deformation."""
    from scipy.ndimage import map_coordinates
    return map_coordinates(vol3M, list(knownWarpIndices(vol3M.shape)), order=order,
                           mode="nearest")


def addMovingScan(planC, baseScanNum=0, structNum=0):
    """Append a synthetically deformed copy of a scan, and of one structure,
    to planC. Returns (planC, movScanNum, movStructNum)."""
    xV, yV, zV = planC.scan[baseScanNum].getScanXYZVals()
    fixed3M = planC.scan[baseScanNum].getScanArray().astype(float)
    planC = pc.importScanArray(warpKnown(fixed3M, 1), xV, yV, zV, "CT", baseScanNum, planC)
    movScanNum = len(planC.scan) - 1
    mask3M = rs.getStrMask(structNum, planC).astype(float)
    planC = pc.importStructureMask(warpKnown(mask3M, 0) > 0.5, movScanNum,
                                   planC.structure[structNum].structureName + " (moving)",
                                   planC)
    return planC, movScanNum, len(planC.structure) - 1
