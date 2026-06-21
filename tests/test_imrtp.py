"""Smoke test for the IMRTP pencil-beam (QIB) influence/dose engine.

Builds a tiny synthetic plan (water-cylinder CT + a spherical PTV) and checks
that the QIB engine fills beamlet influences and that getIMDose produces a sane
dose distribution. No network; small enough to run in a few seconds.
"""
import os
import tempfile

import nibabel as nib
import numpy as np

import cerr.plan_container as pc
from cerr.dataclasses import structure as structr


def _tiny_plan():
    N, NS = 24, 8
    ct = (np.zeros((N, N, NS), dtype=np.int16) - 1000)        # air
    rr, cc = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    body = ((rr - N // 2) ** 2 + (cc - N // 2) ** 2) < (N * 0.42) ** 2
    for s in range(NS):
        ct[:, :, s][body] = 0                                # water
    tmp = tempfile.mkdtemp()
    f = os.path.join(tmp, 'ct.nii.gz')
    nib.save(nib.Nifti1Image(ct, np.diag([3., 3., 3., 1.])), f)
    planC = pc.loadNiiScan(f, imageType='CT SCAN')

    rr, cc, ss = np.meshgrid(np.arange(N), np.arange(N), np.arange(NS),
                             indexing='ij')
    ptv = (rr - N // 2) ** 2 + (cc - N // 2) ** 2 + ((ss - NS // 2) * 2) ** 2 \
        < 5 ** 2
    planC = pc.importStructureMask(ptv.astype(np.uint8), 0, 'PTV', planC)
    return planC


def test_qib_influence_and_dose():
    from cerr.imrtp import imrtp_problem as imp
    from cerr.imrtp.dosecalc import generateQIBInfluence, getIMDose

    planC = _tiny_plan()
    scanSize = tuple(int(v) for v in planC.scan[0].getScanSize())

    im = imp.initIMRTProblem(planC)
    gPTV = imp.addGoal(im, 0, planC)
    gPTV.isTarget = 'yes'
    gPTV.xySampleRate = 1
    imp.addEquispacedBeams(im, 2, 0.0, planC)        # 2 beams
    im.params.algorithm = 'QIB'
    for b in im.beams:
        imp.conditionBeam(b, im, planC, getattr(b, 'autoFieldsOverride', None))

    generateQIBInfluence(im, planC)
    pbCounts = [b.RTOGPBVectorsM.shape[0] if b.beamlets else 0
                for b in im.beams]
    nPB = int(sum(pbCounts))
    assert nPB > 0, 'QIB engine produced no beamlets'

    ptvNum = structr.getStructNumFromUID(gPTV.strUID, planC)
    dose3M = getIMDose(im, np.ones(nPB), [ptvNum], planC)

    assert dose3M.shape == scanSize
    assert np.all(np.isfinite(dose3M))
    assert np.all(dose3M >= 0)
    # the target should receive dose from at least one beam
    from cerr.contour import rasterseg as rs
    ptvMask = rs.getStrMask(ptvNum, planC).astype(bool)
    assert dose3M[ptvMask].max() > 0
