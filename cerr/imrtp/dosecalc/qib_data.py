"""qib_data
~~~~~~~~~

Loader for the QIB (Quadrant Infinite Beam) pencil-beam kernel data.
Python port of Matlab CERR ``IMRTP/loadPBData.m``.

The QIB algorithm is based on an analytical fit of Monte-Carlo generated
pencil-beam data due to Ahnesjo et al. [Med. Phys. 19, 263-273 (1992)]:

    D(r, z) = A(z) exp(-a(z) r)/r + B(z) exp(-b(z) r)/r

with fast table look-up of *quadrant infinite integrals* (integrals of the
exponential / Gaussian kernel over an infinite 2-D quadrant).

Data files (in ``dosecalc/QIBData/``):
    aahn_6b.dat / aahn_18b.dat  depth tables of A, a, B, b for 6 / 18 MV
    qib_tables.npz              ``QIB`` and ``QIBGauss`` quadrant-integral
                                lookup tables (1593 x 1593, originally
                                ``QIB_lin_0pt125.mat`` / ``QIBGauss_lin_0pt125.mat``)

If ``qib_tables.npz`` is missing, the original ``.mat`` files are downloaded
from the CERR GitHub repository and converted (requires ``scipy``).

This file is part of pyCERR and is distributed under the terms of the
Lesser GNU Public License (same terms as CERR).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'QIBData')

_CERR_RAW = ('https://raw.githubusercontent.com/cerr/CERR/master/IMRTP/'
             'QIBData/')


@dataclass
class QIBDataS:
    """Container mirroring the Matlab ``QIBDataS`` structure."""
    aahn6b: np.ndarray = None        # depth, A, a, B, b   (6 MV)
    aahn18b: np.ndarray = None       # depth, A, a, B, b   (18 MV)
    QIB: np.ndarray = None           # quadrant integrals, exponential kernel
    QIBGauss: np.ndarray = None      # quadrant integrals, Gaussian kernel
    deltaQBM: float = 0.0125         # table grid spacing (scaled units)
    QBMidIndexX: int = 792           # 0-based index where intensity = 0.5
    QBMidIndexY: int = 792           # (Matlab value 793, 1-based)
    # Constants of the Gaussian-smear model (Ahnesjo):
    k1: float = 1.1284
    k2: float = 0.476
    k3: float = 0.0354
    k4: float = 0.715


_CACHE: QIBDataS = None


def _fetchTables(npzPath: str):
    """Download the original CERR .mat lookup tables and convert to .npz."""
    import urllib.request
    try:
        from scipy.io import loadmat
    except ImportError as e:
        raise ImportError(
            'qib_tables.npz is missing and scipy is required to convert the '
            'original CERR .mat tables. Either install scipy or place '
            'qib_tables.npz in %s' % _DATA_DIR) from e
    tabs = {}
    for key, fname in (('QIB', 'QIB_lin_0pt125.mat'),
                       ('QIBGauss', 'QIBGauss_lin_0pt125.mat')):
        dest = os.path.join(_DATA_DIR, fname)
        if not os.path.exists(dest):
            print('Downloading %s from CERR GitHub...' % fname)
            urllib.request.urlretrieve(_CERR_RAW + fname, dest)
        m = loadmat(dest)
        tabs[key] = m[fname[:-4]].astype(np.float32)
    np.savez_compressed(npzPath, **tabs)
    return tabs


def loadPBData() -> QIBDataS:
    """Load (and cache) the QIB pencil-beam kernel data.

    Port of ``loadPBData.m``.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    d = QIBDataS()
    d.aahn6b = np.loadtxt(os.path.join(_DATA_DIR, 'aahn_6b.dat'))
    d.aahn18b = np.loadtxt(os.path.join(_DATA_DIR, 'aahn_18b.dat'))

    npzPath = os.path.join(_DATA_DIR, 'qib_tables.npz')
    if os.path.exists(npzPath):
        z = np.load(npzPath)
        d.QIB = np.asarray(z['QIB'], dtype=np.float64)
        d.QIBGauss = np.asarray(z['QIBGauss'], dtype=np.float64)
    else:
        tabs = _fetchTables(npzPath)
        d.QIB = tabs['QIB'].astype(np.float64)
        d.QIBGauss = tabs['QIBGauss'].astype(np.float64)

    _CACHE = d
    return d
