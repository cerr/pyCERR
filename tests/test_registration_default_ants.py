"""The generic functions in cerr.registration.register use ANTs by default and
dispatch warps to the tool that produced a transform; plastimatch is deprecated."""
import os
import warnings

import numpy as np
import pytest

from cerr import datasets
from cerr import plan_container as pc
from cerr.contour.rasterseg import getStrMask

ants = pytest.importorskip("ants")

from cerr.registration import register  # noqa: E402

phantom_dir = os.path.join(os.path.dirname(datasets.__file__), 'radiomics_phantom_dicom', 'pat_1')
SHIFT = (6, 4, 2)


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    """Phantom (scan 0) and a copy rolled by SHIFT voxels (scan 1) with the GTV
    rolled the same way, registered with the default registerScans."""
    planC = pc.loadDcmDir(phantom_dir)
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    planC = pc.importScanArray(np.roll(planC.scan[0].getScanArray(), SHIFT, axis=(0, 1, 2)),
                               xV, yV, zV, 'CT', 0, planC)
    planC = pc.importStructureMask(np.roll(getStrMask(0, planC), SHIFT, axis=(0, 1, 2)), 1,
                                   'GTV moving', planC)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)          # no deprecation on the default path
        planC = register.registerScans(planC, 0, planC, 1, str(tmp_path_factory.mktemp('tx')),
                                       deforAlgorithm='rigid')
    return planC


def _dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def test_register_scans_uses_ants(registered):
    planC = registered
    deformS = planC.deform[-1]
    assert deformS.registrationTool == 'ants'
    assert deformS.algorithm == 'Rigid'
    assert len(planC.scan) == 3                                 # warped scan appended
    fixed3M = planC.scan[0].getScanArray().astype(float)
    warped3M = planC.scan[2].getScanArray().astype(float)
    inner = (slice(15, -15), slice(15, -15), slice(4, -4))
    assert np.corrcoef(fixed3M[inner].ravel(), warped3M[inner].ravel())[0, 1] > 0.95


def test_generic_warps_dispatch_to_ants(registered):
    planC = registered
    deformS = planC.deform[-1]
    n0 = len(planC.structure)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        planC = register.warpStructures(planC, 0, planC, [1], deformS)
    assert len(planC.structure) == n0 + 1
    assert _dice(getStrMask(len(planC.structure) - 1, planC), getStrMask(0, planC)) > 0.95
    del planC.structure[-1]


def test_vector_field_jacobian_and_warp_with_field(registered, tmp_path):
    planC = registered
    deformS = planC.deform[-1]
    nDeform = len(planC.deform)
    planC = register.calcVectorField(deformS, planC, 0, str(tmp_path))
    vfDeform = planC.deform[-1]
    assert len(planC.deform) == nDeform + 1 and vfDeform.deformOutFileType == 'vf'

    # Jacobian of a rigid transform is 1 everywhere it is defined
    for d in (deformS, vfDeform):
        planC = register.calcJacobian(d, planC)
        jac3M = planC.scan[-1].getScanArray()
        assert planC.scan[-1].scanInfo[0].imageType == 'Jacobian'
        assert abs(np.median(jac3M) - 1) < 0.01
        del planC.scan[-1]

    # The sampled field warps a structure like the transform itself
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        planC = register.warpStructures(planC, 0, planC, [1], vfDeform)
    assert _dice(getStrMask(len(planC.structure) - 1, planC), getStrMask(0, planC)) > 0.95
    del planC.structure[-1]
    del planC.deform[-1]


def test_plastimatch_is_deprecated_and_unknown_tool_rejected(registered, tmp_path):
    planC = registered
    with pytest.raises(ValueError, match="registrationTool"):
        register.registerScans(planC, 0, planC, 1, str(tmp_path), registrationTool='elastix')
    with pytest.warns(FutureWarning, match="deprecated"):
        try:
            register.registerScans(planC, 0, planC, 1, str(tmp_path),
                                   registrationTool='plastimatch')
        except Exception:
            pass                                   # plastimatch itself may not be installed
