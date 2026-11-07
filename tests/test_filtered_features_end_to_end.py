"""Filters that ran in processImage but failed when computing features on them.

1. ``Laws`` / ``RotationInvariantLaws``: ``processImage`` accepts these names, but
   ``ibsi1.createFieldNameFromParameters`` only knew ``lawsconvolution``, so every
   settings file with a Laws (non-energy) filter raised ``Invalid image type``.
2. ``Gabor3d`` (2.5D Gabor): ``gaborFilter3d`` transposed the shared scan once per
   accumulated response, so the coronal pass filtered the wrong plane and any scan
   whose dimensions differ failed to stack the plane responses. Its features also had
   no field name.

Fully offline using the bundled phantom.
"""
import json
import os

import numpy as np
import pytest

from cerr import datasets
from cerr import plan_container as pc
from cerr.radiomics import ibsi1, texture_filters, texture_utils

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')


def _settings(imageType):
    return {
        "structures": ["GTV-1"],
        "imageType": imageType,
        "settings": {
            "resample": {"resolutionXCm": 0.2, "resolutionYCm": 0.2, "resolutionZCm": 0.2,
                         "interpMethod": "sitkLinear", "inPlane": "no"},
            "cropToMask": {"method": "expand", "size": [6, 6, 6]},
            "padding": {"method": "mirror", "size": [6, 6, 6], "cropToMask": "yes"},
            "texture": {"directionality": "3D", "avgType": "feature", "voxelOffset": 1,
                        "patchRadiusVox": [1, 1, 1], "imgDiffThresh": 0, "binwidth": 25,
                        "minSegThreshold": -1000, "maxSegThreshold": 400,
                        "minClipIntensity": -1000, "maxClipIntensity": 400},
            "firstOrder": {"offsetForEnergy": 0, "binWidthEntropy": 25}},
        "featureClass": {"firstOrder": {"featureList": ["all"]}}}


@pytest.fixture(scope="module")
def planC():
    return pc.loadDcmDir(phantom_dir)


def _features(planC, tmp_path, imageType):
    f = tmp_path / "settings.json"
    f.write_text(json.dumps(_settings(imageType)))
    feats, _ = ibsi1.computeScalarFeatures(0, 0, str(f), planC)
    return feats


@pytest.mark.parametrize("imageType, prefix", [
    ({"Laws": {"Direction": "3d", "Type": "E5L5S5", "Normalize": "yes"}}, "laws_3d_typeE5L5S5"),
    ({"RotationInvariantLaws": {"Direction": "3d", "Type": "E5L5S5", "Normalize": "yes",
                                "RotationInvariance": {"Dim": "3D", "AggregationMethod": "max"}}},
     "rotationinvariantlaws_3d_typeE5L5S5"),
    ({"Gabor3d": {"Sigma_mm": 5, "Wavlength_mm": 4, "SpatialAspectRatio": 0.5, "Orientation": [0, 90],
                  "OrientationAggregation": "average", "ImagePlane": ["Axial", "Sagittal", "Coronal"],
                  "PlaneAggregation": "average"}}, "gabor3d_"),
])
def test_features_on_filtered_image(planC, tmp_path, imageType, prefix):
    feats = _features(planC, tmp_path, imageType)
    names = [k for k in feats if k.startswith(prefix) and "firstOrder" in k]
    assert names, f"no {prefix}* features in {sorted(feats)[:5]}"
    assert all(np.isfinite(feats[k]) for k in names)


def _gabor3d_reference(scan3M, sigma, wavelength, gamma, thetaV, aggS, radius, paddingV):
    """Definition: 2D Gabor in each orthogonal plane, rotated back, averaged over planes."""
    toPlane = {'axial': None, 'sagittal': (2, 0, 1), 'coronal': (2, 1, 0)}
    fromPlane = {'axial': None, 'sagittal': (1, 2, 0), 'coronal': (2, 1, 0)}
    planes = []
    for p in ('axial', 'sagittal', 'coronal'):
        x = scan3M if toPlane[p] is None else np.transpose(scan3M, toPlane[p])
        out, _ = texture_filters.gaborFilter(x, sigma, wavelength, gamma, thetaV, aggS, radius, paddingV)
        r = next(iter(out.values()))
        planes.append(r if fromPlane[p] is None else np.transpose(r, fromPlane[p]))
    return np.mean(planes, axis=0)


@pytest.mark.parametrize("shape", [(24, 20, 14), (16, 16, 16)])
def test_gabor3d_matches_plane_definition(shape):
    rng = np.random.default_rng(0)
    scan3M = rng.normal(0, 1, shape)
    thetaV = np.array([0, 45, 90, 135])
    aggS = {'OrientationAggregation': 'average', 'PlaneAggregation': 'average'}
    radius, paddingV = np.array([6, 6]), np.array([0, 0, 0])
    out, _ = texture_filters.gaborFilter3d(scan3M, 2.0, 4.0, 0.5, thetaV, aggS, radius, paddingV)
    got = next(iter(out.values()))
    assert got.shape == shape
    want = _gabor3d_reference(scan3M, 2.0, 4.0, 0.5, thetaV, {'OrientationAggregation': 'average'},
                              radius, paddingV)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_gabor3d_processimage_non_cubic():
    rng = np.random.default_rng(1)
    scan3M = rng.normal(40, 20, (30, 26, 18))
    mask3M = np.ones(scan3M.shape, bool)
    p = {"Sigma_mm": 3.0, "Wavlength_mm": 4.0, "SpatialAspectRatio": 0.5, "Orientation": [0, 90],
         "OrientationAggregation": "average", "ImagePlane": ["Axial", "Sagittal", "Coronal"],
         "PlaneAggregation": "average", "VoxelSize_mm": np.array([1.0, 1.0, 1.0]),
         "Padding": {"Size": [0, 0, 0], "Method": "none", "Flag": False}}
    out = texture_utils.processImage("Gabor3d", scan3M, mask3M, p)
    assert next(iter(out.values())).shape == scan3M.shape
