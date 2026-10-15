"""Tests for DVH computation (``cerr.dvh``) against an analytic uniform-dose case.

Builds a synthetic box structure and a spatially-uniform dose on the bundled
DICOM phantom grid (no network). With a constant dose ``D`` over the whole grid
the DVH is exactly predictable: every structure voxel receives ``D``, the
integrated histogram volume equals the structure volume, and the
Dx/Vx/MOHx/MOCx/meanDose summary metrics all collapse to ``D``. This pins down
``getDVH`` (mask integration + dose interpolation) and the histogram metrics,
which are otherwise only exercised indirectly. MOHx/MOCx are further checked
against an exact voxel-level reference on synthetic dose distributions.
"""
import os
import numpy as np
import pytest

from cerr import datasets
from cerr import plan_container as pc
from cerr.dataclasses import structure as structr
from cerr.contour.rasterseg import getStrMask
from cerr import dvh

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')

UNIFORM_DOSE = 40.0  # Gy


def _setup():
    """Phantom planC + a box structure + a spatially-uniform dose grid."""
    planC = pc.loadDcmDir(phantom_dir)
    nRows, nCols, nSlc = planC.scan[0].getScanSize()

    mask = np.zeros((nRows, nCols, nSlc), dtype=bool)
    r0, r1 = nRows // 4, nRows - nRows // 4
    c0, c1 = nCols // 4, nCols - nCols // 4
    s0 = max(1, nSlc // 2 - 1)
    s1 = min(nSlc - 1, nSlc // 2 + 2)
    mask[r0:r1, c0:c1, s0:s1] = True
    planC = structr.importStructureMask(mask, 0, 'dvh_box', planC)
    structNum = len(planC.structure) - 1

    # Spatially-uniform dose on the scan grid.
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    dose3M = np.full((nRows, nCols, nSlc), UNIFORM_DOSE, dtype=float)
    planC = pc.importDoseArray(dose3M, xV, yV, zV, planC, 0)
    doseNum = len(planC.dose) - 1
    return planC, structNum, doseNum


def test_getDVH_uniform_dose():
    """Every structure voxel sees the uniform dose."""
    planC, structNum, doseNum = _setup()
    dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)
    assert isErr == 0
    assert len(dosesV) > 0
    np.testing.assert_allclose(dosesV, UNIFORM_DOSE, rtol=1e-3, atol=1e-3)


def test_dvh_volume_matches_mask():
    """Integrated DVH volume equals (voxel count) x (voxel volume)."""
    planC, structNum, doseNum = _setup()
    dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)

    xV, yV, zV = planC.scan[0].getScanXYZVals()
    dx = abs(xV[1] - xV[0])
    dy = abs(yV[1] - yV[0])
    dz = abs(zV[1] - zV[0])
    voxVol = dx * dy * dz  # cc

    nVox = int(getStrMask(structNum, planC).sum())
    assert nVox > 0
    np.testing.assert_allclose(volsV.sum(), nVox * voxVol, rtol=0.02)


def test_dvh_metrics_uniform_dose():
    """Dx / Vx / MOHx / MOCx / meanDose collapse to the uniform dose."""
    planC, structNum, doseNum = _setup()
    dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)
    binWidth = 0.2
    doseBinsV, volHistV = dvh.doseHist(dosesV, volsV, binWidth)

    # Fraction of volume below D is the whole structure; above D is none.
    np.testing.assert_allclose(
        dvh.Vx(doseBinsV, volHistV, UNIFORM_DOSE - 1.0, volumeType=1),
        1.0, atol=1e-6)
    assert dvh.Vx(doseBinsV, volHistV, UNIFORM_DOSE + 1.0, volumeType=1) == 0

    # Dose to the hottest 50% of volume, mean of hottest 90%, and mean dose
    # all equal D to within one bin.
    assert abs(dvh.Dx(doseBinsV, volHistV, 50.0, volumeType=1) - UNIFORM_DOSE) <= binWidth
    assert abs(dvh.MOHx(doseBinsV, volHistV, 90) - UNIFORM_DOSE) <= binWidth
    # MOCx used to return 0 here: the single dose bin holds more than x% of the
    # volume, and the coldest whole bin was dropped instead of partly counted.
    assert abs(dvh.MOCx(doseBinsV, volHistV, 10) - UNIFORM_DOSE) <= binWidth
    assert abs(dvh.meanDose(doseBinsV, volHistV) - UNIFORM_DOSE) <= binWidth


def test_median_dose_is_volume_weighted():
    # Three dose levels; most of the volume sits at the highest one. The median
    # is the dose at which half the volume is reached, not the middle bin.
    dosesV = np.array([1.0, 2.0, 10.0])
    volsV = np.array([1.0, 1.0, 10.0])
    doseBinsV, volsHistV = dvh.doseHist(dosesV, volsV, 0.05)
    np.testing.assert_allclose(dvh.medianDose(doseBinsV, volsHistV), 10.0, atol=0.05)

    # Equal-volume voxels with a skewed dose: agrees with the plain median of
    # the voxel doses (to within a bin) and with D50%.
    rng = np.random.default_rng(0)
    dosesV = 60.0 * rng.random(20000) ** 3
    volsV = np.full(dosesV.shape, 0.01)
    binWidth = 0.05
    doseBinsV, volsHistV = dvh.doseHist(dosesV, volsV, binWidth)
    med = dvh.medianDose(doseBinsV, volsHistV)
    np.testing.assert_allclose(med, np.median(dosesV), atol=binWidth)
    np.testing.assert_allclose(med, dvh.Dx(doseBinsV, volsHistV, 50, 1), atol=binWidth)


def _exactTailMean(dosesV, volsV, targetVol, hottest):
    """Mean dose of exactly targetVol cc taken from the hot or cold end, per voxel."""
    order = np.argsort(dosesV)[::-1] if hottest else np.argsort(dosesV)
    d, v = dosesV[order], volsV[order]
    before = np.concatenate(([0.0], np.cumsum(v)[:-1]))
    taken = np.clip(targetVol - before, 0, v)
    return np.sum(d * taken) / targetVol


_RNG = np.random.default_rng(0)
_N = 50000
_TAIL_CASES = {
    'uniform': np.full(_N, 50.0),
    'two-level': np.r_[np.full(int(0.9 * _N), 10.0), np.full(int(0.1 * _N), 60.0)],
    'plateau': np.r_[np.full(int(0.3 * _N), 2.0), _RNG.uniform(2, 60, _N - int(0.3 * _N))],
    'continuous': _RNG.uniform(0, 70, _N),
    'target-like': _RNG.normal(60, 1.5, _N).clip(0),
}


@pytest.mark.parametrize('case', sorted(_TAIL_CASES))
@pytest.mark.parametrize('binWidth', [0.05, 0.5, 2.0])
def test_moh_moc_match_exact_voxel_reference(case, binWidth):
    """MOHx/MOCx average exactly x% of the volume, so the only error left is
    the half-bin resolution of the histogram. Before the fix MOCx returned 0
    for the uniform, two-level and plateau cases, and MOH20 of the two-level
    case averaged the whole structure (15 Gy instead of 35 Gy)."""
    dosesV = _TAIL_CASES[case]
    volsV = np.full(dosesV.shape, 0.001)
    doseBinsV, volsHistV = dvh.doseHist(dosesV, volsV, binWidth)
    for pct in (1, 5, 20, 50, 95, 100):
        target = pct / 100 * volsV.sum()
        assert abs(dvh.MOHx(doseBinsV, volsHistV, pct)
                   - _exactTailMean(dosesV, volsV, target, True)) <= binWidth / 2 + 1e-9
        assert abs(dvh.MOCx(doseBinsV, volsHistV, pct)
                   - _exactTailMean(dosesV, volsV, target, False)) <= binWidth / 2 + 1e-9


def test_moh_moc_cc_matches_equivalent_percent():
    dosesV = _TAIL_CASES['continuous']
    volsV = np.full(dosesV.shape, 0.002)
    doseBinsV, volsHistV = dvh.doseHist(dosesV, volsV, 0.1)
    totalCc = volsV.sum()                                   # 100 cc
    for fn in (dvh.MOHx, dvh.MOCx):
        np.testing.assert_allclose(fn(doseBinsV, volsHistV, 2, 0),
                                   fn(doseBinsV, volsHistV, 2 / totalCc * 100, 1))
        np.testing.assert_allclose(fn(doseBinsV, volsHistV, totalCc, 0),
                                   dvh.meanDose(doseBinsV, volsHistV))


def test_moh_moc_limits():
    """100% is the mean dose; 0% is the hottest / coldest bin that holds volume."""
    doseBinsV, volsHistV = dvh.doseHist(np.array([5.0, 10.0, 30.0]),
                                        np.array([1.0, 2.0, 1.0]), 1.0)
    np.testing.assert_allclose(dvh.MOHx(doseBinsV, volsHistV, 100),
                               dvh.meanDose(doseBinsV, volsHistV))
    np.testing.assert_allclose(dvh.MOCx(doseBinsV, volsHistV, 100),
                               dvh.meanDose(doseBinsV, volsHistV))
    np.testing.assert_allclose(dvh.MOHx(doseBinsV, volsHistV, 0),
                               dvh.maxDose(doseBinsV, volsHistV))
    np.testing.assert_allclose(dvh.MOCx(doseBinsV, volsHistV, 0),
                               dvh.minDose(doseBinsV, volsHistV))


def test_moh_moc_roe_parameters():
    doseBinsV, volsHistV = dvh.doseHist(_TAIL_CASES['continuous'],
                                        np.full(_N, 0.001), 0.1)
    expected = dvh.MOHx(doseBinsV, volsHistV, 5, 1)
    assert dvh.MOHx(doseBinsV, volsHistV,
                    {'volCutoff': {'val': 5}, 'volType': {'val': 1}}) == expected
    # volType defaults to percent, as for the plain call.
    assert dvh.MOHx(doseBinsV, volsHistV, {'volCutoff': {'val': 5}}) == expected
    assert dvh.MOCx(doseBinsV, volsHistV, {'volCutoff': {'val': 2}, 'volType': {'val': 0}}) \
        == dvh.MOCx(doseBinsV, volsHistV, 2, 0)
    # The former 'percent' key fails loudly instead of being silently ignored.
    with pytest.raises(ValueError, match="'percent' was renamed"):
        dvh.MOCx(doseBinsV, volsHistV, {'percent': {'val': 5}})


@pytest.mark.parametrize('args', [(-1,), (101,), (5, 2), (1e6, 0)])
def test_moh_moc_reject_invalid_cutoffs(args):
    doseBinsV, volsHistV = dvh.doseHist(np.array([5.0, 10.0]), np.array([1.0, 1.0]), 1.0)
    for fn in (dvh.MOHx, dvh.MOCx):
        with pytest.raises(ValueError):
            fn(doseBinsV, volsHistV, *args)
