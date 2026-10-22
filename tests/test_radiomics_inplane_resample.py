"""In-plane resampling must keep the scan's slice spacing.

With ``inPlane: yes`` only x and y are resampled. A non-zero ``resolutionZCm``
in the settings used to be applied to the output grid anyway, so the structure
was squashed along z and every size-dependent shape feature was wrong (volume
by the ratio of the two spacings). Uses the bundled phantom; no network.
"""
import copy
import json
import os

import numpy as np

from cerr import datasets
from cerr import plan_container as pc
from cerr.contour.rasterseg import getStrMask
from cerr.radiomics import ibsi1

datasetsDir = os.path.dirname(datasets.__file__)
phantom_dir = os.path.join(datasetsDir, 'radiomics_phantom_dicom', 'pat_1')
settingsFile = os.path.join(datasetsDir, 'radiomics_settings', 'original_settings.json')


def _shapeFeatures(planC, tmp_path, tag, resolutionZCm=None):
    with open(settingsFile) as f:
        settings = json.load(f)
    settings = copy.deepcopy(settings)
    settings['featureClass'] = {'shape': {'featureList': ['all']}}
    if resolutionZCm is not None:
        settings['settings']['resample']['inPlane'] = 'yes'
        settings['settings']['resample']['resolutionZCm'] = resolutionZCm
    outFile = str(tmp_path / f'settings_{tag}.json')
    with open(outFile, 'w') as f:
        json.dump(settings, f)
    featDict, _ = ibsi1.computeScalarFeatures(0, 0, outFile, planC)
    return {key: float(val) for key, val in featDict.items() if 'shape' in key}


def test_inplane_resampling_ignores_z_resolution(tmp_path):
    planC = pc.loadDcmDir(phantom_dir)
    dxCm, dyCm, dzCm = planC.scan[0].getScanSpacing()
    maskVolMm3 = getStrMask(0, planC).sum() * dxCm * dyCm * dzCm * 1000

    zeroZ = _shapeFeatures(planC, tmp_path, 'z0', resolutionZCm=0)
    # a z resolution that differs from the 3 mm slice spacing must not matter
    otherZ = _shapeFeatures(planC, tmp_path, 'z01', resolutionZCm=0.1)

    np.testing.assert_allclose(zeroZ['original_shape_volume'], maskVolMm3, rtol=0.01)
    assert zeroZ.keys() == otherZ.keys()
    for key in zeroZ:
        np.testing.assert_allclose(otherZ[key], zeroZ[key], rtol=1e-9, err_msg=key)


def test_bundled_settings_give_physical_shape_volume(tmp_path):
    planC = pc.loadDcmDir(phantom_dir)
    dxCm, dyCm, dzCm = planC.scan[0].getScanSpacing()
    maskVolMm3 = getStrMask(0, planC).sum() * dxCm * dyCm * dzCm * 1000
    shipped = _shapeFeatures(planC, tmp_path, 'shipped')
    np.testing.assert_allclose(shipped['original_shape_volume'], maskVolMm3, rtol=0.01)
