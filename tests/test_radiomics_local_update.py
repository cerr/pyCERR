"""Local feature updates must match a full recompute.

``LocalFeatureUpdater.featuresWithoutBox`` updates the whole-ROI texture
matrices for an ROI with a box of voxels removed. Every feature must equal
``ibsi1.calcRadiomicsForImgType`` on the reduced mask (to float32 rounding),
for merged and per-direction matrices and for 3D and 2D directionality. When
removing the box would change the gray-level bins it must return None.
Uses the bundled phantom and synthetic images; no network.
"""
import copy
import json
import os

import numpy as np
import pytest

from cerr import datasets
from cerr import plan_container as pc
from cerr.radiomics import ibsi1, preprocess
from cerr.radiomics.local_update import LocalFeatureUpdater, glcmCounts, glcmNormalize
from cerr.radiomics.gray_level_cooccurence import calcCooccur
from cerr.utils.mask import computeBoundingBox

datasetsDir = os.path.dirname(datasets.__file__)
phantom_dir = os.path.join(datasetsDir, 'radiomics_phantom_dicom', 'pat_1')
settingsFile = os.path.join(datasetsDir, 'radiomics_settings', 'original_settings.json')

RTOL = 1e-5  # float32 rounding in the GLCM normalization


def _settings(avgType='feature', directionality='3D', binNum=None):
    with open(settingsFile) as f:
        paramS = copy.deepcopy(json.load(f))
    paramS['featureClass'].pop('shape')
    tex = paramS['settings']['texture']
    tex['avgType'] = avgType
    tex['directionality'] = directionality
    if binNum is not None:
        tex.pop('binwidth')
        tex['binNum'] = binNum
        for key in ['minClipIntensity', 'maxClipIntensity', 'minSegThreshold', 'maxSegThreshold']:
            tex.pop(key)
    return paramS


def _flat(featDict, paramS):
    tex = paramS['settings']['texture']
    flatS = ibsi1.createFlatFeatureDict(featDict, 'original', tex['avgType'], tex['directionality'])
    return {k: np.asarray(v, dtype=float) for k, v in flatS.items()}


def _assertMatchesFull(img3M, mask3M, gridS, paramS, boxes):
    updater = LocalFeatureUpdater(img3M, mask3M, gridS, paramS)
    for box in boxes:
        localS = updater.featuresWithoutBox(box)
        assert localS is not None
        reduced3M = mask3M.copy()
        reduced3M[box] = False
        fullS = ibsi1.calcRadiomicsForImgType(img3M.copy(), reduced3M, reduced3M, gridS, paramS)
        localFlat, fullFlat = _flat(localS, paramS), _flat(fullS, paramS)
        assert localFlat.keys() == fullFlat.keys()
        for key in fullFlat:
            np.testing.assert_allclose(localFlat[key], fullFlat[key], rtol=RTOL, atol=1e-12, err_msg=key)


def _syntheticImage(seed=0, shape=(24, 22, 10)):
    """Smooth random CT-like texture inside an ellipsoid ROI."""
    rng = np.random.default_rng(seed)
    img3M = rng.normal(0, 60, shape)
    for _ in range(2):  # a little spatial smoothing, so runs and zones are non-trivial
        img3M = (img3M + np.roll(img3M, 1, 0) + np.roll(img3M, 1, 1) + np.roll(img3M, 1, 2)) / 2
    r, c, s = np.indices(shape)
    mask3M = ((r - 11.5) / 10) ** 2 + ((c - 10.5) / 9) ** 2 + ((s - 4.5) / 4.5) ** 2 <= 1
    gridS = {'PixelSpacingV': np.array([0.1, 0.1, 0.3])}
    return img3M, mask3M, gridS


@pytest.mark.parametrize('avgType,directionality', [('feature', '3D'), ('texture', '3D'), ('feature', '2D')])
def test_local_update_matches_full_recompute(avgType, directionality):
    img3M, mask3M, gridS = _syntheticImage()
    boxes = [(slice(10, 13), slice(9, 12), slice(3, 6)),   # interior
             (slice(1, 4), slice(9, 12), slice(3, 6)),     # on the ROI edge
             (slice(5, 8), slice(4, 7), slice(0, 3))]      # first slices
    _assertMatchesFull(img3M, mask3M, gridS, _settings(avgType, directionality), boxes)


def test_local_update_matches_full_recompute_on_phantom():
    planC = pc.loadDcmDir(phantom_dir)
    paramS = _settings()
    scan3M, mask3M, _, gridS, paramS, _ = preprocess.preProcessForRadiomics(0, 0, paramS, planC)
    r0, r1, c0, c1, s0, s1, _ = computeBoundingBox(mask3M)
    crop = (slice(r0, r1 + 1), slice(c0, c1 + 1), slice(s0, s1 + 1))
    img3M, mask3M = scan3M[crop].astype(float), np.asarray(mask3M[crop], dtype=bool)
    nR, nC, nS = mask3M.shape
    boxes = [(slice(nR // 2, nR // 2 + 3), slice(nC // 2, nC // 2 + 3), slice(nS // 2, nS // 2 + 3)),
             (slice(nR // 3, nR // 3 + 3), slice(2 * nC // 3, 2 * nC // 3 + 3), slice(nS // 3, nS // 3 + 3))]
    _assertMatchesFull(img3M, mask3M, gridS, paramS, boxes)


def test_glcm_counts_match_calcCooccur():
    img3M, mask3M, gridS = _syntheticImage(seed=1)
    paramS = _settings()
    vol3M = img3M.copy()
    _, _, _, _, _, _, _, _, offsetsM, minClip, maxClip, binNum, binWidth = ibsi1.getRadiomicsSettings(paramS)
    q3M = ibsi1.getQuantizedVolume(vol3M, paramS, mask3M, binNum, minClip, maxClip, binWidth)
    nL = int(q3M.max())
    for cooccurType in (1, 2):
        ref = calcCooccur(q3M.copy(), offsetsM, nL, cooccurType)
        new = glcmNormalize(glcmCounts(q3M, offsetsM, nL, cooccurType == 2))
        np.testing.assert_allclose(new, ref, rtol=1e-6, atol=1e-9)


def test_returns_none_when_bins_would_change():
    img3M, mask3M, gridS = _syntheticImage(seed=2)
    paramS = _settings(binNum=16)  # bins from the ROI's own min and max
    updater = LocalFeatureUpdater(img3M, mask3M, gridS, paramS)
    roiV = np.where(mask3M, img3M, -np.inf)
    r, c, s = np.unravel_index(np.argmax(roiV), img3M.shape)  # the single brightest voxel
    box = (slice(r, r + 1), slice(c, c + 1), slice(s, s + 1))
    assert updater.featuresWithoutBox(box) is None
