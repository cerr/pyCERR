"""Shared helpers for the scripts that generate the documentation figures.

Every figure in the user guide is produced by a script in this directory from
data bundled with pyCERR, so the figures can be regenerated at any time::

    python docs/_figure_scripts/make_all.py

The scripts need only a core ``pycerr`` install (plus ``antspyx`` for
``registration.py``).
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
