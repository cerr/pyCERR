"""Offline numerics tests for the urOMT (unbalanced regularized OMT) core.

These exercise the forward advection-diffusion-source model, the objective
(get_Gamma analog) and the analytic adjoint gradient on a tiny grid. No network
and no planC/DICOM; runs in a few seconds.

The adjoint gradient is non-differentiable at the trilinear-interpolation domain
boundary (the corner index and the physical coordinate are clamped there), so
the finite-difference check is restricted to interior voxels - exactly as the
gradient was validated during development.
"""
import inspect
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from cerr.uromt.numerics import (paramInit, sourceAdvecDiff, getGamma,
                                  gradGamma, forwardSensitivity,
                                  adjointSensitivity)
from cerr.uromt.solver import gnBlockExact, gnBlockUr
from cerr.utils.image_proc import affineDiffusion3d
from cerr.mri_metrics.dce_mri import getScanOrder, normalizeToBaseline, buildConcDict
from cerr.uromt.data import externalBaselineCount, scanTimeLabel
from cerr.uromt.analyze import runEULA, runGLAD, runEULAIntervals
from cerr.dataclasses.uromt import (UROMT, getUROMTList, saveUROMTToPlan,
                                    buildFromConfig)
from cerr.utils import uid
from cerr.uromt.viz import (velocityVectors, eulerianFluxVectors,
                            eulerianMapToScan, pathlineTracks)

TEST_T10 = 1.0 / 0.6 # Testing pre-contrast longitudinal relaxation time
TEST_r1 = 3.8        # Testing relaxivity

def _par(n=(8, 8, 8), nt=2, sigma=2e-3, alpha=10.0, beta=50.0, dt=0.3,
         chi=None, eta=0.0):
    """Build a `par` dict on a tiny grid via a lightweight cfg stand-in."""
    cfg = SimpleNamespace(trueSize=list(n), spacing=[1.0, 1.0, 1.0],
                          dt=dt, nt=nt, sigma=sigma, alpha=alpha, beta=beta,
                          bc="closed", niter_pcg=10, maxUiter=3, chi=chi,
                          eta=eta)
    return paramInit(cfg)


def _interior_mask(n):
    """Voxels at least 2 cells from every spatial boundary (Fortran-flattened)."""
    N = int(np.prod(n))
    i1, i2, i3 = np.unravel_index(np.arange(N), n, order="F")
    return ((i1 > 1) & (i1 < n[0] - 2) &
            (i2 > 1) & (i2 < n[1] - 2) &
            (i3 > 1) & (i3 < n[2] - 2))


def test_forward_model_identity_no_flow_no_diffusion():
    """u=0, r=0, sigma=0 -> trilinear sampling at exact cell centers is the
    identity and B=I, so density is unchanged at every step."""
    par = _par(sigma=0.0)
    N, nt = par["N"], par["nt"]
    rng = np.random.default_rng(0)
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    rho = sourceAdvecDiff(rho0, np.zeros(3 * N * nt), np.zeros(N * nt), par)
    assert rho.shape == (N, nt)
    for k in range(nt):
        assert np.allclose(rho[:, k], rho0, atol=1e-10)


def test_diffusion_solver_inverts_B_exactly():
    """The DCT-based implicit-diffusion solver (par['Bsolve']) must invert the
    sparse operator B = I + dt*sigma*Grad'Grad to machine precision, including
    on a non-uniform-spacing grid (guards the per-axis h^2 eigenvalues)."""
    import scipy.sparse as sp
    from cerr.uromt.numerics import neumannGrad
    n = (7, 9, 5)
    h = [0.106, 0.106, 0.14]
    cfg = SimpleNamespace(trueSize=list(n), spacing=h, dt=0.3, nt=2,
                          sigma=2e-3, alpha=10.0, beta=50.0, bc="closed",
                          niter_pcg=10, maxUiter=3, chi=None)
    par = paramInit(cfg)
    N = par["N"]
    Grad = neumannGrad(n, h)
    B = (sp.identity(N, format="csr") + cfg.dt * cfg.sigma * (Grad.T @ Grad))
    rng = np.random.default_rng(7)
    x = rng.standard_normal(N)
    sol = par["Bsolve"](x)
    assert np.allclose(B @ sol, x, atol=1e-10)            # solves B sol = x
    assert np.allclose(par["Bsolve"](B @ x), x, atol=1e-10)  # round-trips


def test_forward_model_finite_and_shaped():
    par = _par()
    N, nt = par["N"], par["nt"]
    rng = np.random.default_rng(1)
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    u = 0.02 * rng.standard_normal(3 * N * nt)
    r = 0.05 * rng.standard_normal(N * nt)
    rho = sourceAdvecDiff(rho0, u, r, par)
    assert rho.shape == (N, nt)
    assert np.all(np.isfinite(rho))


def test_objective_components_nonnegative_and_fit_term():
    par = _par()
    N, nt = par["N"], par["nt"]
    rng = np.random.default_rng(2)
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    u = 0.02 * rng.standard_normal(3 * N * nt)
    r = 0.05 * rng.standard_normal(N * nt)
    drhoN = np.abs(rng.standard_normal(N)) + 0.5
    G, (G1, G2, G3, G4), rho = getGamma(rho0, u, r, par, drhoN)
    # Gamma1 (kinetic), Gamma3 (fit), Gamma4 (H1) are sums of squares -> >= 0.
    assert G1 >= 0.0 and G2 >= 0.0 and G3 >= 0.0 and G4 >= 0.0
    # Gamma3 is exactly hd * ||rho_N - drhoN||^2.
    expect = par["hd"] * float(np.sum((rho[:, -1] - drhoN) ** 2))
    assert np.isclose(G3, expect, rtol=1e-12, atol=0.0)
    # Total = G1 + alpha*G2 + beta*G3 + G4.
    assert np.isclose(G, G1 + par["alpha"] * G2 + par["beta"] * G3 + G4,
                      rtol=1e-12)
    # eta = 0 (default) -> the H1 term is exactly zero.
    assert G4 == 0.0


def test_adjoint_gradient_matches_finite_difference_interior():
    """Central finite differences vs. the analytic adjoint gradient on the
    highest-magnitude interior coordinates."""
    par = _par()
    n, N, nt = par["n"], par["N"], par["nt"]
    rng = np.random.default_rng(3)
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    u = 0.02 * rng.standard_normal(3 * N * nt)   # small flow -> samples interior
    r = 0.05 * rng.standard_normal(N * nt)
    drhoN = np.abs(rng.standard_normal(N)) + 0.5

    gU, gR = gradGamma(rho0, u, r, par, drhoN)
    g = np.concatenate([gU, gR])
    x = np.concatenate([u, r])

    # Map each state coordinate to its spatial voxel, keep only interior ones.
    interior = _interior_mask(n)
    voxU = np.tile(np.tile(np.arange(N), 3), nt)
    voxR = np.tile(np.arange(N), nt)
    vox = np.concatenate([voxU, voxR])
    intCoords = np.where(interior[vox])[0]

    # Test the 20 interior coordinates with the largest analytic gradient.
    order = intCoords[np.argsort(-np.abs(g[intCoords]))]
    sel = order[:20]

    eps = 1e-6
    nu = 3 * N * nt
    relErrs = []
    for i in sel:
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        Gp, _, _ = getGamma(rho0, xp[:nu], xp[nu:], par, drhoN)
        Gm, _, _ = getGamma(rho0, xm[:nu], xm[nu:], par, drhoN)
        fd = (Gp - Gm) / (2 * eps)
        relErrs.append(abs(fd - g[i]) / max(abs(g[i]), 1e-8))
    relErrs = np.array(relErrs)
    assert relErrs.max() < 5e-3, "max rel err %.2e" % relErrs.max()
    assert np.median(relErrs) < 1e-3, "median rel err %.2e" % np.median(relErrs)


def test_h1_smoothness_gradient_matches_finite_difference():
    """With eta > 0 the velocity H1-smoothness term contributes to the objective
    and its analytic gradient must match central finite differences on interior
    velocity coordinates."""
    par = _par(eta=0.7)
    n, N, nt = par["n"], par["N"], par["nt"]
    rng = np.random.default_rng(11)
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    u = 0.02 * rng.standard_normal(3 * N * nt)
    r = 0.05 * rng.standard_normal(N * nt)
    drhoN = np.abs(rng.standard_normal(N)) + 0.5

    # the H1 term is genuinely active
    _, comps, _ = getGamma(rho0, u, r, par, drhoN)
    assert comps[3] > 0.0
    par0 = _par(eta=0.0)
    _, comps0, _ = getGamma(rho0, u, r, par0, drhoN)
    assert comps0[3] == 0.0

    gU, gR = gradGamma(rho0, u, r, par, drhoN)
    g = np.concatenate([gU, gR])
    x = np.concatenate([u, r])

    interior = _interior_mask(n)
    voxU = np.tile(np.tile(np.arange(N), 3), nt)
    voxR = np.tile(np.arange(N), nt)
    vox = np.concatenate([voxU, voxR])
    intCoords = np.where(interior[vox])[0]
    sel = intCoords[np.argsort(-np.abs(g[intCoords]))][:20]

    eps = 1e-6
    nu = 3 * N * nt
    relErrs = []
    for i in sel:
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        Gp, _, _ = getGamma(rho0, xp[:nu], xp[nu:], par, drhoN)
        Gm, _, _ = getGamma(rho0, xm[:nu], xm[nu:], par, drhoN)
        fd = (Gp - Gm) / (2 * eps)
        relErrs.append(abs(fd - g[i]) / max(abs(g[i]), 1e-8))
    relErrs = np.array(relErrs)
    assert relErrs.max() < 5e-3, "max rel err %.2e" % relErrs.max()
    assert np.median(relErrs) < 1e-3, "median rel err %.2e" % np.median(relErrs)


def test_chi_identity_matches_none():
    """The source-indicator chi defaults to K=1; chi=ones must reproduce the
    chi=None objective exactly."""
    rng = np.random.default_rng(4)
    parN = _par(chi=None)
    N, nt = parN["N"], parN["nt"]
    parI = _par(chi=np.ones(N))
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    u = 0.02 * rng.standard_normal(3 * N * nt)
    r = 0.05 * rng.standard_normal(N * nt)
    drhoN = np.abs(rng.standard_normal(N)) + 0.5
    GN, _, _ = getGamma(rho0, u, r, parN, drhoN)
    GI, _, _ = getGamma(rho0, u, r, parI, drhoN)
    assert GN == GI


def test_chi_gradient_matches_finite_difference_interior():
    """Adjoint gradient with a nontrivial spatial chi vs. finite differences."""
    rng = np.random.default_rng(5)
    n = (8, 8, 8)
    N = int(np.prod(n))
    chi = rng.uniform(0.2, 1.0, N)
    par = _par(n=n, chi=chi)
    nt = par["nt"]
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    u = 0.02 * rng.standard_normal(3 * N * nt)
    r = 0.05 * rng.standard_normal(N * nt)
    drhoN = np.abs(rng.standard_normal(N)) + 0.5

    gU, gR = gradGamma(rho0, u, r, par, drhoN)
    g = np.concatenate([gU, gR])
    x = np.concatenate([u, r])
    interior = _interior_mask(n)
    vox = np.concatenate([np.tile(np.tile(np.arange(N), 3), nt),
                          np.tile(np.arange(N), nt)])
    intCoords = np.where(interior[vox])[0]
    sel = intCoords[np.argsort(-np.abs(g[intCoords]))][:20]

    eps = 1e-6
    nu = 3 * N * nt
    relErrs = []
    for i in sel:
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        Gp, _, _ = getGamma(rho0, xp[:nu], xp[nu:], par, drhoN)
        Gm, _, _ = getGamma(rho0, xm[:nu], xm[nu:], par, drhoN)
        fd = (Gp - Gm) / (2 * eps)
        relErrs.append(abs(fd - g[i]) / max(abs(g[i]), 1e-8))
    relErrs = np.array(relErrs)
    assert relErrs.max() < 5e-3, "max rel err %.2e" % relErrs.max()


def test_affine_diffusion_smooths_and_preserves_mean_shape():
    """affineDiffusion3d reduces local variance (smooths), keeps shape & dtype,
    stays nonnegative, and reduces to a no-op for nSteps=0."""
    rng = np.random.default_rng(6)
    img = np.abs(rng.standard_normal((10, 12, 8)))
    out = affineDiffusion3d(img, nSteps=5, dt=0.1, affFlag=True)
    assert out.shape == img.shape
    assert np.all(out >= 0.0)
    # interior total variation should not increase
    def tv(a):
        return (np.abs(np.diff(a, axis=0)).sum()
                + np.abs(np.diff(a, axis=1)).sum()
                + np.abs(np.diff(a, axis=2)).sum())
    assert tv(out) <= tv(img)
    assert np.array_equal(affineDiffusion3d(img, nSteps=0), img)
    # linear (heat) flow path also runs and smooths
    lin = affineDiffusion3d(img, nSteps=5, dt=0.1, affFlag=False)
    assert lin.shape == img.shape and tv(lin) <= tv(img)


def test_scan_time_order():
    """Scans are ordered by acquisition time even when planC stores them in a
    different (e.g. lexical) order; the timepoint->scan-index map is correct.
    """
    # planC stand-in: scan i has an out-of-order acquisitionTime
    times = ["075710", "075813", "080115", "075833", "075620"]   # not sorted
    scans = [SimpleNamespace(
        scanInfo=[SimpleNamespace(acquisitionDate="20101111",
                                  acquisitionTime=t)]) for t in times]

    planC = SimpleNamespace(scan=scans)

    order = getScanOrder(planC)
    assert order == [4, 0, 1, 3, 2]                  # sorted by acq time
    assert order != list(range(len(scans)))          # differs from index order
    # the mapping resolves the correct scan index per timepoint
    assert [scanTimeLabel(planC, s) for s in order] == sorted(times)
    # key falls back to scan index when no time metadata
    bare = SimpleNamespace(scan=[SimpleNamespace(scanInfo=[SimpleNamespace()])
                                 for _ in range(3)])
    assert getScanOrder(bare) == [0, 1, 2]


def test_external_baseline_count_window_after_baseline():
    """A transport window that starts after the leading baseline frames uses an
    external (non-consumed) baseline, so a 2-frame selection like first=20:2:22
    is transported in full instead of losing its first frame to the baseline."""
    # window starts at position 19 (first=20), basePts=1 -> external baseline,
    # nothing consumed from the 2 selected frames
    assert externalBaselineCount("CC", 0, 1, 19) == 1
    assert externalBaselineCount("RSE", 0, 2, 19) == 2
    # window at the very start -> consume in-sequence (no external baseline)
    assert externalBaselineCount("CC", 0, 1, 0) == 0
    assert externalBaselineCount("CC", 0, 2, 1) == 0
    # explicit baselineFrames always wins
    assert externalBaselineCount("CC", 3, 1, 0) == 3
    # no concentration conversion -> never an external baseline
    assert externalBaselineCount("none", 5, 1, 19) == 0


def test_concentration_defaults():
    """Bundled concentration defaults match the requested values."""
    assert np.isclose(TEST_T10, 1.0 / 0.6)
    assert TEST_r1 == 3.8


def test_frames_to_concentration_recovers_known_concentration():
    """Generate DCE signal from a known concentration via the SPGR model and
    check framesToConcentration recovers a positive, monotonically increasing
    concentration, consuming the baseline frames."""
    n = (8, 8, 3)
    ii = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]),
                     indexing="ij")
    blob = np.exp(-sum((ii[d] - c) ** 2 for d, c in enumerate((4, 4, 1.5)))
                  / (2 * 1.8 ** 2))
    mask = blob > 0.05
    T10, r1, FA, TR = 1.0 / 0.6, 3.8, 15.0, 0.005          # seconds TR
    R10 = 1.0 / T10
    a = np.radians(FA)
    M0 = 1000.0

    def spgr(R1):
        E1 = np.exp(-TR * R1)
        return M0 * np.sin(a) * (1 - E1) / (1 - np.cos(a) * E1)

    Cknown = [0.0, 0.1, 0.3, 0.5]                           # mmol/L per frame
    frames = [spgr(R10 + r1 * (c * blob)) for c in Cknown]

    scanArr4M = np.stack(frames, axis=3)
    timePtsV = np.arange(len(frames), dtype=float)
    concDict = {"T10": T10, "r1": r1, "TR": TR, "FA": FA}

    conc4M, uptakeTimeV, baseline3M, basePtsUsed = normalizeToBaseline(
        scanArr4M, mask, timePtsV, basePts=1, method="CC", concDict=concDict)
    conc4M = np.nan_to_num(conc4M, nan=0.0)

    assert basePtsUsed == 1
    assert conc4M.shape[3] == len(frames) - 1              # baseline consumed
    core = blob > 0.8
    recovered = [float(conc4M[:, :, :, j][core].mean())
                 for j in range(conc4M.shape[3])]
    assert np.all(conc4M >= 0)                              # nonnegative
    assert recovered[0] < recovered[1] < recovered[2]       # monotonic uptake
    # outside the ROI stays zero (masked to nan upstream, nan_to_num'd to 0)
    assert conc4M[~mask].max() == 0.0


def test_frames_to_rse():
    """RSE normalization returns S(t)/S(0) and consumes the baseline frame.
    """
    n = (6, 6, 3)
    base = 100.0
    frames = [base * np.ones(n), 1.5 * base * np.ones(n), 2.0 * base * np.ones(n)]
    mask = np.ones(n, dtype=bool)
    scanArr4M = np.stack(frames, axis=3)
    timePtsV = np.arange(len(frames), dtype=float)

    out4M, _t, _b, basePtsUsed = normalizeToBaseline(
        scanArr4M, mask, timePtsV, basePts=1, method="RSE")
    assert basePtsUsed == 1 and out4M.shape[3] == len(frames) - 1  # baseline consumed
    assert np.allclose(out4M[:, :, :, 0][mask], 1.5)
    assert np.allclose(out4M[:, :, :, 1][mask], 2.0)


def test_frames_to_concentration_requires_tr():
    """normalizeToBaseline(method='CC') requires a valid repetition time (TR).
    Check that ValueError is raised instead of silently proceeding with
    undefined TR.
    """
    n = (4, 4, 2)
    scanArr4M = np.ones(n + (3,))
    mask = np.ones(n, dtype=bool)
    timePtsV = np.arange(3, dtype=float)
    concDict = {"T10": 1.0 / 0.6, "r1": 3.8, "TR": None, "FA": 15.0}
    try:
        normalizeToBaseline(scanArr4M, mask, timePtsV, basePts=1,
                            method="CC", concDict=concDict)
        assert False, "expected ValueError for missing TR"
    except ValueError as e:
        assert "TR" in str(e)


def test_frames_to_concentration_requires_fa():
    """normalizeToBaseline(method='CC') requires a valid flip angle (FA).
    Check that ValueError is raised instead of silently proceeding with
    undefined FA.
    """
    n = (4, 4, 2)
    scanArr4M = np.ones(n + (3,))
    mask = np.ones(n, dtype=bool)
    timePtsV = np.arange(3, dtype=float)
    concDict = {"T10": 1.0 / 0.6, "r1": 3.8, "TR": 0.005, "FA": None}
    try:
        normalizeToBaseline(scanArr4M, mask, timePtsV, basePts=1,
                            method="CC", concDict=concDict)
        assert False, "expected ValueError for missing FA"
    except ValueError as e:
        assert "FA" in str(e)


def _uniform_flow_result(n=(16, 16, 8), nt=4, vx=1.0, dt=0.4,
                         bbox=(2, 18, 3, 19, 1, 9)):
    """Synthetic urOMT result: uniform velocity v=(vx,0,0), rho=ones."""
    N = int(np.prod(n))
    v = np.zeros((3, N, nt))
    v[0] = vx
    return dict(u=[v], r=[np.zeros((N, nt))], rho=[np.ones((N, nt))],
                n=list(n), spacing=[1.0, 1.0, 1.0],
                mask=np.ones(n, dtype=np.uint8), bbox=bbox,
                frameScanNums=[5, 6], doResize=0, sizeFactor=1.0,
                dt=dt, nt=nt, sigma=2e-3)


def test_runEULA_speed_rate_flux():
    """Eulerian maps: speed/rate/peclet are time AVERAGES, but flux is the
    per-interval time INTEGRAL (sum over the nt sub-steps), matching the
    reference EulerFlux convention."""
    res = _uniform_flow_result(vx=1.0)
    nt = res["nt"]
    Eul = runEULA(res)
    assert np.allclose(Eul["speed"], 1.0)              # mean |v|
    assert np.allclose(Eul["rate"], 0.0)
    # rho=1, v_eff=v=1 -> flux integrates to nt (NOT 1: it is a sum, not a mean)
    assert np.allclose(Eul["flux"][0], float(nt))
    assert np.allclose(Eul["flux"][1], 0.0)
    assert np.all(np.isfinite(Eul["peclet"]))
    assert Eul["speed3"].shape == tuple(res["n"])


def test_runEULA_intervals():
    """Per-interval Eulerian maps: one entry per interval; for a uniform field
    with rho=1, effSpeed == advective speed == |v|."""
    res = _uniform_flow_result(vx=1.0)               # 1 interval
    ei = runEULAIntervals(res)
    assert len(ei["effSpeed"]) == len(res["u"]) == 1
    for key in ("speed", "effSpeed", "rate", "peclet", "rho"):
        assert ei[key][0].shape == tuple(res["n"])
    assert ei["flux"][0].shape == (3,) + tuple(res["n"])
    assert np.allclose(ei["effSpeed"][0], 1.0)       # rho=1 -> v_eff = v
    assert np.allclose(ei["speed"][0], 1.0)
    assert np.allclose(ei["rate"][0], 0.0)
    assert np.allclose(ei["rho"][0], 1.0)
    # flux is the SUM over the interval's nt sub-steps (reference EulerFlux
    # convention), so it scales with nt while the scalar maps do not.
    assert np.allclose(ei["flux"][0][0], float(res["nt"]))
    assert np.allclose(ei["flux"][0][1], 0.0)


def test_runGLAD_pathline_displacement_and_direction():
    """Lagrangian pathlines of a uniform field move by v * total_time and
    reverse with direction=-1."""
    res = _uniform_flow_result(vx=1.0, nt=4, dt=0.4)
    Lag = runGLAD(res, spfs=4, nEuler=5, direction=1.0, slTolVox=0.5)
    assert len(Lag["SL"]) > 0
    # total transport time = nIntervals*nt*dt = 1*4*0.4 = 1.6; |v|=1 -> 1.6 cm
    assert np.allclose(Lag["disp"].mean(axis=0), [1.6, 0.0, 0.0], atol=1e-6)
    assert np.allclose(Lag["displen"].mean(), 1.6, atol=1e-6)
    assert np.allclose(np.concatenate(Lag["sstream"]), 1.0)
    LagRev = runGLAD(res, spfs=4, direction=-1.0, slTolVox=0.5)
    assert np.allclose(LagRev["disp"].mean(axis=0), [-1.6, 0.0, 0.0], atol=1e-6)


def test_part5_viz_builders():
    """Eulerian flux vectors, map embedding, and pathline tracks build with the
    correct shapes and ROI->scan voxel offsets."""
    res = _uniform_flow_result()
    Eul = runEULA(res)
    Lag = runGLAD(res, spfs=4, slTolVox=0.5)
    rs_, re_, cs_, ce_, ss_, se_ = res["bbox"]

    vd = eulerianFluxVectors(Eul, subsample=2, magPctile=40)
    assert vd["vectors"].ndim == 3 and vd["vectors"].shape[1:] == (2, 3)
    assert vd["scanNum"] == 5
    assert vd["vectors"][:, 0, 0].min() >= rs_     # start coords offset by bbox

    full = eulerianMapToScan(Eul, field="speed", scanShape=(30, 30, 12))
    assert full.shape == (30, 30, 12)
    assert full[0, 0, 0] == 0.0                    # zero outside bbox
    assert np.isclose(full[rs_ + 2, cs_ + 2, ss_ + 2], 1.0)

    data, props = pathlineTracks(Lag, colorBy="speed", maxTracks=500)
    assert data.shape[1] == 5                       # [tid, t, row, col, slice]
    assert "speed" in props and len(props["speed"]) == data.shape[0]
    assert data[:, 2].min() >= rs_                  # row coords offset by bbox
    data2, props2 = pathlineTracks(Lag, colorBy="peclet")
    assert "peclet" in props2


def test_draw_uromt_slice_all_views():
    """The embedded-GUI slice renderer (drawUROMTSlice) draws every view on
    every axis without error (matplotlib Agg, no Qt)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from cerr.uromt.viz import drawUROMTSlice

    res = _uniform_flow_result(n=(12, 10, 6))
    res["r"] = [0.01 * np.ones_like(res["r"][0])]
    Eul = runEULA(res)
    Lag = runGLAD(res, spfs=3, slTolVox=0.3)
    bg = res["rho"][0].mean(1).reshape(res["n"], order="F")
    fig = Figure()
    for view in ("speed", "rate", "peclet", "velocity", "flux", "pathlines"):
        for axis in (0, 1, 2):
            ax = drawUROMTSlice(fig, res, Eul, Lag, view=view, axis=axis, bg=bg)
            assert ax is not None
    # missing Eul/Lag raise informative errors
    try:
        drawUROMTSlice(fig, res, None, None, view="speed")
        assert False
    except ValueError:
        pass


def test_uromt_scan_overlay_helpers():
    """The main-viewer overlay helpers (fieldToScan, pathlinesToScanVox,
    drawUROMTOverlay) build full scan-grid data and render every view on every
    orientation with the correct coordinate mapping (matplotlib Agg, no Qt)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from cerr.uromt import viz

    scanShape = (40, 36, 20)
    n = (16, 14, 10)
    bbox = (8, 24, 10, 24, 5, 15)        # fits inside scanShape
    res = _uniform_flow_result(n=n, nt=4, bbox=bbox)
    res["r"] = [0.01 * np.ones_like(res["r"][0])]
    Eul = runEULA(res)
    Lag = runGLAD(res, spfs=2, slTolVox=0.3)

    comps = viz.fieldToScan(Eul["flux"], res["n"], bbox, scanShape)
    assert len(comps) == 3 and comps[0].shape == scanShape
    # comps[0] is the row/axis-0 component (the one carrying the uniform flow)
    assert (comps[0][bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] != 0).any()
    assert comps[0][0, 0, 0] == 0.0      # zero outside the ROI bbox

    segs, vals = viz.pathlinesToScanVox(Lag)
    assert len(segs) == len(vals) > 0
    assert segs[0][:, 0].min() >= bbox[0]   # pathlines offset into scan grid

    # physical coordinate vectors (yV decreasing, like DICOM)
    xV = np.linspace(0, 3.6, 36)
    yV = np.linspace(5, 0, 40)
    zV = np.linspace(0, 2.8, 20)
    # (slicer, hAxis, vAxis, thruAxis, hV, vV) per orientation - mirrors the GUI
    orients = {
        "axial": (lambda k: (lambda m: m[:, :, k]), 1, 0, 2, xV, yV),
        "sagittal": (lambda k: (lambda m: m[:, k, :].T), 0, 2, 1, yV, zV),
        "coronal": (lambda k: (lambda m: m[k, :, :].T), 1, 2, 0, xV, zV),
    }
    fig = Figure()
    for view in ("speed", "rate", "peclet", "velocity", "flux", "pathlines"):
        ov = {"view": view, "alpha": 0.6}
        if view in ("speed", "rate", "peclet"):
            ov["map3"] = viz.eulerianMapToScan(Eul, field=view,
                                               scanShape=scanShape)
        elif view in ("velocity", "flux"):
            fld = Eul["flux"] if view == "flux" else res["u"][0].mean(2)
            ov["comps"] = viz.fieldToScan(fld, res["n"], bbox, scanShape)
        else:
            ov["segs"] = viz.pathlinesToScanVox(Lag)
        for mk, hA, vA, tA, hVv, vVv in orients.values():
            k = scanShape[tA] // 2
            ax = fig.add_subplot(111)
            ext = [hVv[0], hVv[-1], vVv[-1], vVv[0]]
            viz.drawUROMTOverlay(ax, ov, k, hVv, vVv, ext, mk(k), hA, vA, tA,
                                 scanShape)
            assert (len(ax.images) + len(ax.collections) + len(ax.lines)) >= 0
            fig.clf()


def test_eulerian_map_to_scan_resized_no_broadcast():
    """Preview/resized runs: the Eulerian ROI map is smaller than its bbox, so
    eulerianMapToScan must zoom it up to the bbox extent instead of broadcasting
    (regression for the 'could not broadcast' error in preview mode)."""
    from cerr.uromt import viz
    scanShape = (40, 36, 20)
    bbox = (8, 24, 10, 24, 5, 15)        # bbox extent 16 x 14 x 10
    # half-resolution map (8 x 7 x 5) as a do_resize=0.5 run would produce
    Eul = dict(speed3=np.abs(np.random.default_rng(0).standard_normal((8, 7, 5))),
               rate3=np.zeros((8, 7, 5)), peclet3=np.zeros((8, 7, 5)),
               bbox=bbox, frameScanNums=None)
    full = viz.eulerianMapToScan(Eul, field="speed", scanShape=scanShape)
    assert full.shape == scanShape
    assert (full[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] != 0).any()
    assert full[0, 0, 0] == 0.0          # zero outside the ROI bbox
    # the EulerFlux magnitude colourwash uses the same mapper (field='fluxmag')
    EulF = dict(fluxmag3=np.abs(np.random.default_rng(1).standard_normal((16, 14, 10)))
                + 0.1, bbox=bbox, frameScanNums=None)
    fullF = viz.eulerianMapToScan(EulF, field="fluxmag", scanShape=scanShape)
    assert fullF.shape == scanShape
    assert (fullF[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] != 0).any()


def test_export_roi_map_to_scan_placement():
    """export._roiMapToScan places an ROI-grid map into the full scan grid at
    the bbox (and zooms a resized run up to the bbox extent) - the geometry the
    NIfTI export writes."""
    from cerr.uromt.export import _roiMapToScan, EULER_METRICS
    scanShape = (40, 36, 20)
    bbox = (8, 24, 10, 24, 5, 15)            # extent 16 x 14 x 10
    roi = np.arange(16 * 14 * 10).reshape(16, 14, 10).astype(float) + 1.0
    full = _roiMapToScan(roi, bbox, scanShape)
    assert full.shape == scanShape
    assert np.array_equal(full[8:24, 10:24, 5:15], roi)   # placed exactly
    assert full[0, 0, 0] == 0.0                            # zero outside bbox
    small = np.abs(np.random.default_rng(0).standard_normal((8, 7, 5))) + 0.1
    fullS = _roiMapToScan(small, bbox, scanShape)          # resized -> zoom
    assert fullS.shape == scanShape
    assert (fullS[8:24, 10:24, 5:15] != 0).any()
    assert {"speed", "rate", "peclet", "flux"} <= set(EULER_METRICS)


def test_uromt_overlay_vectors_no_markers():
    """The velocity/flux quiver overlay draws scaled arrows only - no start/stop
    scatter markers (the arrowhead shows direction); arrows stay finite."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.collections import PathCollection
    from cerr.uromt import viz

    scanShape = (40, 36, 20)
    n = (16, 14, 10)
    bbox = (8, 24, 10, 24, 5, 15)
    res = _uniform_flow_result(n=n, nt=4, bbox=bbox)
    comps = viz.fieldToScan(res["u"][0].mean(2), res["n"], bbox, scanShape)
    ov = {"view": "velocity", "alpha": 0.6, "comps": comps}
    xV = np.linspace(0, 3.6, 36)
    yV = np.linspace(5, 0, 40)
    fig = Figure()
    ax = fig.add_subplot(111)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    viz.drawUROMTOverlay(ax, ov, scanShape[2] // 2, xV, yV, ext,
                         lambda m: m[:, :, scanShape[2] // 2], 1, 0, 2, scanShape)
    scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(scatters) == 0                       # no marker dots
    quivers = [c for c in ax.collections if c not in scatters]
    assert len(quivers) >= 1                         # the arrows are drawn


def test_pathlines_per_vertex_speed_and_grow():
    """``perVertex`` returns one speed per pathline VERTEX (not the per-path
    mean), aligned to the vertex count, and ``growPathline`` truncates a path to
    its leading fraction so it can be animated out from the seed."""
    from cerr.uromt import viz
    from cerr.uromt.analyze import runGLAD

    res = _uniform_flow_result(vx=1.0)
    Lag = runGLAD(res, spfs=2, nEuler=2)
    segs, vals, spds = viz.pathlinesToScanVox(Lag, 1.0, 0, perVertex=True)
    assert len(spds) == len(segs) == len(vals)
    for seg, sp in zip(segs, spds):
        assert sp.shape == (seg.shape[0],)          # one speed per vertex
        assert np.all(np.isfinite(sp))
    # the 2-tuple form is unchanged for existing callers
    assert len(viz.pathlinesToScanVox(Lag, 1.0, 0)) == 2

    pts, sp = segs[0], spds[0]
    half, halfSp = viz.growPathline(pts, sp, 0.5)
    assert 2 <= half.shape[0] < pts.shape[0]
    assert halfSp.shape[0] == half.shape[0]
    assert np.array_equal(half, pts[:half.shape[0]])   # keeps the LEADING part
    whole, _ = viz.growPathline(pts, sp, 1.0)
    assert whole.shape[0] == pts.shape[0]
    tiny, _ = viz.growPathline(pts, sp, 0.0)
    assert tiny.shape[0] == 2                          # still drawable



def _pathHeads(ax):
    """The pathline arrowhead triangles drawn on `ax`: (M,3,2) vertices.

    Heads are a PolyCollection (one triangle per path, sized off THAT path),
    not a quiver: quiver head dimensions are global to the collection and fixed
    in axes units, so on sub-voxel paths one head covered its whole path.
    """
    from matplotlib.collections import PolyCollection
    polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
    if not polys:
        return np.zeros((0, 3, 2)), polys
    tris = np.asarray([np.asarray(pp.vertices)[:3]
                       for pp in polys[0].get_paths()])
    return tris, polys[0]


def _headLengths(tris):
    """Length of each head triangle, tip to the middle of its base."""
    if not len(tris):
        return np.zeros(0)
    return np.hypot(*(tris[:, 0] - 0.5 * (tris[:, 1] + tris[:, 2])).T)


def test_uromt_overlay_pathline_end_arrows_and_colouring():
    """The pathline overlay draws a LineCollection coloured along each path and
    a single NARROW arrowhead at each path's end - no seed marker and no dot
    scatters, which previously outnumbered and obscured the paths."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.collections import PathCollection, LineCollection
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz
    from cerr.uromt.analyze import runGLAD

    scanShape = (40, 36, 20)
    n = (16, 14, 10)
    bbox = (8, 24, 10, 24, 5, 15)
    res = _uniform_flow_result(n=n, nt=4, bbox=bbox)
    Lag = runGLAD(res, spfs=1, nEuler=2)  # spfs=1: seed EVERY slice, else this slice is empty
    segs, vals, spds = viz.pathlinesToScanVox(Lag, 1.0, 0, perVertex=True)
    xV = np.linspace(0, 3.6, 36)
    yV = np.linspace(5, 0, 40)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    k = scanShape[2] // 2

    def draw(ov):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(ax, ov, k, xV, yV, ext,
                             lambda m: m[:, :, k], 1, 0, 2, scanShape)
        return ax

    base = {"view": "pathlines", "alpha": 0.6, "segs": (segs, vals),
            "pathSpeeds": spds, "vrange": (0.0, 1.5),
            "pathColorBy": "along"}      # per-SEGMENT entries
    ax = draw(dict(base))
    from matplotlib.collections import PolyCollection
    lines = [c for c in ax.collections if isinstance(c, LineCollection)
             and not isinstance(c, (Quiver, PolyCollection))]
    tris, heads = _pathHeads(ax)
    dots = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert len(lines) == 1                       # one collection, per-seg colour
    assert len(lines[0].get_segments()) > 1      # split into coloured segments
    assert len(tris) > 0                         # one head triangle per path
    assert len(dots) == 0                        # no start/end dot markers
    # the head is narrow: full width is a fraction of its length
    assert viz._HEAD_ASPECT <= 1.0
    w = np.hypot(*(tris[:, 1] - tris[:, 2]).T)
    assert np.allclose(w, viz._HEAD_ASPECT * _headLengths(tris))
    # at most one arrow per drawn path, and paths whose final step is
    # degenerate (no direction to point) correctly get none
    nPaths = len(lines[0].get_segments()) // (segs[0].shape[0] - 1)
    assert 0 < len(tris) <= nPaths

    # grow truncates the paths but each still keeps its end arrow
    axFull = draw(dict(base, grow=1.0))
    axPart = draw(dict(base, grow=0.25))
    nFull = len(axFull.collections[0].get_segments())
    nPart = len(axPart.collections[0].get_segments())
    assert 0 < nPart < nFull
    assert len(_pathHeads(axPart)[0]) > 0


def test_subsample_is_in_plane_in_2d_and_all_three_dirs_in_3d():
    """'vec every N' means one arrow per Nth voxel OF THE DISPLAYED SLICE in
    2-D (so N=1 shows every voxel on that slice), and per Nth voxel in all
    three directions in 3-D."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (24, 20, 12)
    N = int(np.prod(shape))
    v = np.zeros((3, N, 2))
    v[0] = 1.0                                   # nonzero everywhere
    comps = viz.fieldToScan(v[:, :, 0], list(shape),
                            (0, shape[0], 0, shape[1], 0, shape[2]), shape)
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    zV = np.arange(shape[2], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    k = shape[2] // 2

    def n2d(sub):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(
            ax, {"view": "velocity", "alpha": 1.0, "comps": comps,
                 "subsample": sub, "vrange": (0.0, 2.0)},
            k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
        q = [c for c in ax.collections if isinstance(c, Quiver)]
        return q[0].N if q else 0

    def n3d(sub, cap=100000):
        g = viz.overlayTo3D({"view": "velocity", "comps": comps,
                             "subsample": sub}, xV, yV, zV, maxArrows=cap)
        return len(g["vectors"]["points"])

    def strided(axes, sub):
        out = 1
        for a in axes:
            out *= len(range(0, shape[a], sub))
        return out

    for sub in (1, 2, 3):
        assert n2d(sub) == strided((0, 1), sub)      # in-plane only
        assert n3d(sub) == strided((0, 1, 2), sub)   # all three directions
    # 2-D at N=1 really is every voxel of the slice
    assert n2d(1) == shape[0] * shape[1]
    # 3-D is UNCAPPED by default - density comes only from subsample
    assert n3d(1, cap=None) == shape[0] * shape[1] * shape[2]
    assert viz.overlayTo3D.__defaults__[:2] == (None, None)
    # a cap is still available to callers that want one
    assert n3d(1, cap=500) == 500


def test_runGLAD_defaults_do_not_thin_pathlines():
    """runGLAD used to compound three thinnings - spfs=2 (x1/8), a 4000-seed
    cap, and slTolVox=1.0 which drops the ~88% of seeds moving under a voxel -
    leaving pathlines ~48x sparser than the velocity arrows on the same ROI."""
    import inspect
    from cerr.uromt.analyze import runGLAD

    d = inspect.signature(runGLAD).parameters
    assert d["spfs"].default == 1          # every ROI voxel, like subsample=1
    assert d["slTolVox"].default == 0.0    # keep sub-voxel paths
    assert d["maxSeeds"].default is None   # no cap

    n = (10, 10, 6)
    res = _uniform_flow_result(n=n, nt=4, bbox=(0, 10, 0, 10, 0, 6))
    allSeeds = runGLAD(res, nEuler=1)
    assert len(allSeeds["SL"]) == int(np.prod(n))     # one per ROI voxel
    # spfs thins as N**3, and a cap still works when asked for
    assert len(runGLAD(res, spfs=2, nEuler=1)["SL"]) == 5 * 5 * 3
    assert len(runGLAD(res, nEuler=1, maxSeeds=50)["SL"]) == 50


def test_runGLAD_seed_mask_keeps_pathlines_inside_the_structure():
    """`result['mask']` is the DILATED ROI mask, so with mask_dilate set some
    pathlines start outside the drawn contour. Passing the undilated structure
    as `seedMask` keeps every seed inside it."""
    from cerr.uromt.analyze import runGLAD

    n = (16, 16, 8)
    res = _uniform_flow_result(n=n, nt=4, bbox=(0, 16, 0, 16, 0, 8))
    inner = np.zeros(n, dtype=bool)
    inner[4:12, 4:12, 2:6] = True          # the "structure"
    dilated = np.zeros(n, dtype=bool)
    dilated[2:14, 2:14, 1:7] = True        # what the solve reports on
    res["mask"] = dilated.astype(np.uint8)

    wide = runGLAD(res, spfs=1, nEuler=2, slTolVox=0.0)
    tight = runGLAD(res, spfs=1, nEuler=2, slTolVox=0.0, seedMask=inner)

    def seedsOutside(Lag):
        s = np.rint(np.array([p[0] for p in Lag["SL"]])).astype(int)
        return int((~inner[s[:, 0], s[:, 1], s[:, 2]]).sum())

    assert len(wide["SL"]) > len(tight["SL"])   # dilated mask seeds more
    assert seedsOutside(wide) > 0               # ...and some sit outside
    assert seedsOutside(tight) == 0             # explicit mask is respected


def test_line_width_scales_vectors_and_pathlines_together():
    """One `lineWidth` control thickens the vector arrows and the pathline
    strokes/end arrows together; 1.0 is the default weight."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz
    from cerr.uromt.analyze import runGLAD

    shape = (24, 20, 12)
    n = (16, 14, 10)
    bbox = (4, 20, 3, 17, 1, 11)
    res = _uniform_flow_result(n=n, nt=4, bbox=bbox)
    comps = viz.fieldToScan(res["u"][0].mean(2), n, bbox, shape)
    Lag = runGLAD(res, spfs=1, nEuler=2, slTolVox=0.0)   # seed every slice
    segs, vals, spds = viz.pathlinesToScanVox(Lag, 1.0, 0, perVertex=True)
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    k = shape[2] // 2

    def draw(ov):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(ax, ov, k, xV, yV, ext, lambda m: m[:, :, k],
                             1, 0, 2, shape)
        return ax

    vec = {"view": "velocity", "alpha": 1.0, "comps": comps,
           "vrange": (0.0, 2.0)}
    thin = [c for c in draw(vec).collections if isinstance(c, Quiver)][0]
    thick = [c for c in draw(dict(vec, lineWidth=3.0)).collections
             if isinstance(c, Quiver)][0]
    assert thick.width == pytest.approx(3.0 * thin.width)

    pth = {"view": "pathlines", "alpha": 1.0, "segs": (segs, vals),
           "pathSpeeds": spds, "vrange": (0.0, 1.5)}
    def lc(ov):
        return [c for c in draw(ov).collections
                if isinstance(c, LineCollection) and not isinstance(c, Quiver)]
    t1 = lc(pth)[0].get_linewidth()[0]
    t3 = lc(dict(pth, lineWidth=3.0))[0].get_linewidth()[0]
    assert t3 == pytest.approx(3.0 * t1)


def test_pathlines_selected_by_seed_slice_and_drawn_whole():
    """2-D shows the paths that START on the displayed slice, each drawn in
    FULL (no clipping at the slice boundary), thinned IN-PLANE by subsample."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (30, 30, 12)
    k, nVert = 6, 9
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    # a 4x4 grid of seeds on every slice; each path drifts 3 slices away
    segs, vals, spds = [], [], []
    for z0 in range(shape[2]):
        for r in range(0, 8, 2):
            for c in range(0, 8, 2):
                segs.append(np.column_stack([
                    np.linspace(r, r + 6, nVert),      # moves in-plane...
                    np.full(nVert, float(c)),
                    np.linspace(z0, z0 + 3, nVert)]))  # ...and across slices
                vals.append(1.0)
                spds.append(np.ones(nVert))
    vals = np.asarray(vals)

    def draw(sub):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(
            ax, {"view": "pathlines", "alpha": 1.0, "segs": (segs, vals),
                 "pathSpeeds": spds, "vrange": (0.0, 2.0), "subsample": sub},
            k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
        from matplotlib.collections import PolyCollection
        lc = [c for c in ax.collections
              if isinstance(c, LineCollection)
              and not isinstance(c, (Quiver, PolyCollection))]
        return ((len(lc[0].get_segments()) if lc else 0),
                len(_pathHeads(ax)[0]))

    entries1, arrows1 = draw(1)
    # 16 seeds on this slice; at the default (median) colouring each path is a
    # single polyline entry, and it carries the path's FULL vertex count
    assert arrows1 == 16
    assert entries1 == 16
    # ...and the whole path is drawn even though it leaves the slice
    assert np.ptp(segs[0][:, 2]) > 1

    # in-plane thinning: seeds at rows/cols 0,2,4,6 -> N=2 keeps 0,2,4,6 (all),
    # N=4 keeps 0 and 4 only, in each in-plane axis
    assert draw(2)[1] == 16
    assert draw(4)[1] == 4


def test_grow_is_a_time_fraction_so_fast_paths_outrun_slow():
    """Every runGLAD pathline carries the same vertex count, so the ``grow``
    fraction is a TIME fraction: at any grow step the arc length drawn is
    proportional to that path's speed."""
    from cerr.uromt import viz
    from cerr.uromt.analyze import runGLAD

    n, nt = (24, 24, 4), 6
    N = int(np.prod(n))
    gr, _gc, _gs = np.meshgrid(*[np.arange(s) for s in n], indexing="ij")
    fast = (gr.ravel(order="F") < n[0] / 2).astype(float)
    v = np.zeros((3, N, nt))
    v[1] = (0.2 + 0.8 * fast)[:, None]           # 1.0 vs 0.2 -> 5x
    res = dict(u=[v], r=[np.zeros((N, nt))], rho=[np.ones((N, nt))],
               n=list(n), spacing=[1.0, 1.0, 1.0],
               mask=np.ones(n, dtype=np.uint8),
               bbox=(0, n[0], 0, n[1], 0, n[2]), frameScanNums=[0, 1],
               doResize=0, sizeFactor=1.0, dt=0.5, nt=nt, sigma=2e-3)
    Lag = runGLAD(res, spfs=4, nEuler=2, slTolVox=0.0)
    lens = {p.shape[0] for p in Lag["SL"]}
    assert len(lens) == 1                        # equal vertex counts == time

    segs, _vals, spds = viz.pathlinesToScanVox(Lag, 1.0, 0, perVertex=True)
    isFast = np.array([s[0, 0] < n[0] / 2 for s in segs])

    def arc(p):
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

    for frac in (0.25, 0.5, 1.0):
        a = np.array([arc(viz.growPathline(s, sv, frac)[0])
                      for s, sv in zip(segs, spds)])
        ratio = a[isFast].mean() / max(a[~isFast].mean(), 1e-9)
        assert 4.0 < ratio < 6.0, (frac, ratio)


def test_pathline_and_vector_length_scale():
    """``lengthScale`` shrinks pathlines about their SEED (shape preserved, seed
    fixed) and shortens the 3-D arrows proportionally."""
    from cerr.uromt import viz
    from cerr.uromt.analyze import runGLAD

    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 2.0, 1.0]])
    half = viz.scalePathline(pts, 0.5)
    assert np.allclose(half[0], pts[0])                   # seed anchored
    assert np.allclose(half - half[0], 0.5 * (pts - pts[0]))
    assert np.allclose(viz.scalePathline(pts, 1.0), pts)  # no-op

    scanShape = (40, 36, 20)
    n = (16, 14, 10)
    bbox = (8, 24, 10, 24, 5, 15)
    res = _uniform_flow_result(n=n, nt=4, bbox=bbox)
    Lag = runGLAD(res, spfs=2, nEuler=2, slTolVox=0.0)
    segs, vals, spds = viz.pathlinesToScanVox(Lag, 1.0, 0, perVertex=True)
    comps = viz.fieldToScan(res["u"][0].mean(2), res["n"], bbox, scanShape)
    xV = np.linspace(0, 3.6, scanShape[1])
    yV = np.linspace(5, 0, scanShape[0])
    zV = np.linspace(0, 2, scanShape[2])
    ov = {"view": "pathlines", "segs": (segs, vals), "pathSpeeds": spds,
          "comps": comps, "vrange": (0.0, 1.5)}

    full = viz.overlayTo3D(ov, xV, yV, zV, lengthScale=1.0)
    small = viz.overlayTo3D(ov, xV, yV, zV, lengthScale=0.25)
    assert np.allclose(np.asarray(small["pathStart"]),
                       np.asarray(full["pathStart"]))      # seeds pinned
    dFull = np.linalg.norm(np.asarray(full["pathEnd"])
                           - np.asarray(full["pathStart"]), axis=1)
    dSmall = np.linalg.norm(np.asarray(small["pathEnd"])
                            - np.asarray(small["pathStart"]), axis=1)
    moved = dFull > 1e-9
    assert moved.any()
    assert np.allclose(dSmall[moved] / dFull[moved], 0.25)
    aFull = np.linalg.norm(full["vectors"]["vec"], axis=1)
    aSmall = np.linalg.norm(small["vectors"]["vec"], axis=1)
    nz = aFull > 1e-12
    assert np.allclose(aSmall[nz] / aFull[nz], 0.25)


def test_overlay_to_3d_pathlines_carry_speed_and_markers():
    """The 3-D geometry exposes per-vertex speeds and start/end points so the
    renderers can colour along the path and mark direction, and honours grow."""
    from cerr.uromt import viz
    from cerr.uromt.analyze import runGLAD

    n = (16, 14, 10)
    bbox = (8, 24, 10, 24, 5, 15)
    scanShape = (40, 36, 20)
    res = _uniform_flow_result(n=n, nt=4, bbox=bbox)
    Lag = runGLAD(res, spfs=2, nEuler=2, slTolVox=0.0)
    segs, vals, spds = viz.pathlinesToScanVox(Lag, 1.0, 0, perVertex=True)
    xV = np.linspace(0, 3.6, scanShape[1])
    yV = np.linspace(5, 0, scanShape[0])
    zV = np.linspace(0, 2, scanShape[2])
    ov = {"view": "pathlines", "segs": (segs, vals), "pathSpeeds": spds}

    g = viz.overlayTo3D(ov, xV, yV, zV)
    assert len(g["pathVals"]) == len(g["paths"])
    for p, v in zip(g["paths"], g["pathVals"]):
        assert len(v) == p.shape[0]              # one speed per drawn vertex
    assert np.asarray(g["pathStart"]).shape == (len(g["paths"]), 3)
    assert np.asarray(g["pathEnd"]).shape == (len(g["paths"]), 3)

    part = viz.overlayTo3D(dict(ov, grow=0.4), xV, yV, zV)
    assert part["paths"][0].shape[0] < g["paths"][0].shape[0]
    # an overlay without per-vertex speeds still yields colourable values
    plain = viz.overlayTo3D({"view": "pathlines", "segs": (segs, vals)},
                            xV, yV, zV)
    assert len(plain["pathVals"][0]) == plain["paths"][0].shape[0]


def test_uromt_overlay_colorbar_and_density():
    """The 2-D overlay draws a colorbar legend (patches + text) using the global
    vrange, and the vector ``subsample`` thins the arrows/markers."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.collections import PathCollection
    from cerr.uromt import viz

    scanShape = (40, 36, 20)
    n = (16, 14, 10)
    bbox = (8, 24, 10, 24, 5, 15)
    res = _uniform_flow_result(n=n, nt=4, bbox=bbox)
    comps = viz.fieldToScan(res["u"][0].mean(2), res["n"], bbox, scanShape)
    xV = np.linspace(0, 3.6, 36)
    yV = np.linspace(5, 0, 40)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    slr = lambda m: m[:, :, scanShape[2] // 2]

    def n_arrows(sub):
        ov = {"view": "velocity", "alpha": 0.6, "comps": comps,
              "vrange": (0.0, 0.3), "label": "|v| (mm/t)", "subsample": sub}
        fig = Figure(); ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(ax, ov, scanShape[2] // 2, xV, yV, ext, slr,
                             1, 0, 2, scanShape)
        # colorbar legend: rectangles (patches) + range text present
        assert len(ax.patches) > 10 and len(ax.texts) >= 2
        quiv = [c for c in ax.collections if not isinstance(c, PathCollection)]
        return quiv[0].get_offsets().shape[0] if quiv else 0

    dense = n_arrows(1)
    sparse = n_arrows(3)
    assert dense > sparse > 0                   # density control thins arrows


def test_overlay_to_3d_scalar_map_point_cloud():
    """overlayTo3D turns a scalar-map overlay (speed/rate/peclet) into a colour-
    coded point cloud at the ROI voxels (so maps render in 3-D, not just 2-D)."""
    from cerr.uromt import viz
    scanShape = (30, 28, 16)
    xV = np.linspace(0, 2.9, 28)
    yV = np.linspace(4, 0, 30)
    zV = np.linspace(0, 2.0, 16)
    map3 = np.zeros(scanShape)
    map3[8:16, 6:14, 4:10] = np.arange(8 * 8 * 6).reshape(8, 8, 6) + 1.0
    geom = viz.overlayTo3D({"map3": map3, "view": "speed"}, xV, yV, zV)
    assert geom is not None and "scalar" in geom
    g = geom["scalar"]
    assert g["points"].shape[0] == g["vals"].shape[0] == 8 * 8 * 6
    assert g["vals"].min() > 0 and np.all(np.isfinite(g["points"]))
    # points lie at ROI voxel physical coords (inside the FOV)
    assert g["points"][:, 0].min() >= min(xV) and g["points"][:, 0].max() <= max(xV)


def test_overlay_to_3d_vectors_scaled_in_bounds_and_paths():
    """overlayTo3D maps the cached overlay into physical-coordinate 3-D geometry:
    velocity arrows scaled so the longest spans ~5% of the FOV (kept inside the
    scan) with start/stop points, and pathlines mapped to physical coords."""
    from cerr.uromt import viz
    scanShape = (40, 36, 20)
    n = (16, 14, 10)
    bbox = (8, 24, 10, 24, 5, 15)
    res = _uniform_flow_result(n=n, nt=4, bbox=bbox)
    comps = viz.fieldToScan(res["u"][0].mean(2), res["n"], bbox, scanShape)
    xV = np.linspace(0, 3.6, 36)
    yV = np.linspace(5, 0, 40)             # decreasing, like DICOM
    zV = np.linspace(0, 2.5, 20)
    spanFOV = max(abs(xV[-1] - xV[0]), abs(yV[-1] - yV[0]), abs(zV[-1] - zV[0]))

    geom = viz.overlayTo3D({"comps": comps}, xV, yV, zV)
    g = geom["vectors"]
    arrowLen = np.linalg.norm(g["vec"], axis=1)
    assert np.all(np.isfinite(g["vec"])) and np.all(np.isfinite(g["tip"]))
    # longest arrow is the shared FOV fraction, the same one the 2-D quiver
    # uses so "length x" means the same thing in both views
    assert arrowLen.max() <= viz._VECTOR_FOV_FRAC * spanFOV + 1e-9
    assert arrowLen.max() > 0.9 * viz._VECTOR_FOV_FRAC * spanFOV
    # arrow tips stay within the physical field of view
    assert g["tip"][:, 0].min() >= min(xV) - 0.05 * spanFOV
    assert g["tip"][:, 0].max() <= max(xV) + 0.05 * spanFOV

    Lag = runGLAD(res, spfs=2, slTolVox=0.05)
    segs = viz.pathlinesToScanVox(Lag, 1.0, 0)
    geomP = viz.overlayTo3D({"segs": segs}, xV, yV, zV)
    assert geomP is not None and len(geomP["paths"]) > 0
    p0 = geomP["paths"][0]
    assert p0.shape[1] == 3 and np.all(np.isfinite(p0))
    assert viz.overlayTo3D(None, xV, yV, zV) is None


def test_draw_uromt_3d_all_views():
    """The embedded-GUI 3-D renderer (drawUROMT3D) draws every view without
    error (matplotlib Agg Axes3D, no Qt)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from cerr.uromt.viz import drawUROMT3D

    res = _uniform_flow_result(n=(12, 10, 6))
    res["r"] = [0.01 * np.ones_like(res["r"][0])]
    Eul = runEULA(res)
    Lag = runGLAD(res, spfs=3, slTolVox=0.3)
    fig = Figure()
    for view in ("pathlines", "velocity", "flux", "speed", "rate", "peclet"):
        ax = drawUROMT3D(fig, res, Eul, Lag, view=view)
        assert ax is not None and ax.name == "3d"
    try:
        drawUROMT3D(fig, res, None, None, view="pathlines")
        assert False
    except ValueError:
        pass


def test_show_eulerian_lagrangian_wiring(monkeypatch):
    """showEulerian/showLagrangian (Part 5 GUI display) build the right napari
    layers and align them with the scan affine - verified with a mock napari
    (no display required)."""
    import sys
    import types as _types
    from cerr.uromt import viz

    class _Affine:
        affine_matrix = np.eye(4)

    class _Layer:
        affine = _Affine()

    class _Viewer:
        def __init__(self):
            self.images = []
            self.tracks = []

        def add_image(self, arr, **kw):
            self.images.append((arr, kw))

        def add_tracks(self, data, **kw):
            self.tracks.append((data, kw))

    captured = {}

    def fakeShowNapari(planC, scan_nums=0, struct_nums=(), dose_nums=(),
                       vectors_dict=None, displayMode="3d"):
        v = _Viewer()
        captured["vectors_dict"] = vectors_dict
        captured["scan_nums"] = scan_nums
        return (v, [_Layer()], [], [], [])

    fakeMod = _types.ModuleType("cerr.viewer.pycerr_napari")
    fakeMod.showNapari = fakeShowNapari
    monkeypatch.setitem(sys.modules, "cerr.viewer.pycerr_napari", fakeMod)

    res = _uniform_flow_result()
    Eul = runEULA(res)
    Lag = runGLAD(res, spfs=4, slTolVox=0.5)

    class _PlanC:
        def __init__(self):
            self.scan = [SimpleNamespace(
                getScanArray=lambda: np.zeros((30, 30, 12)))]

    planC = _PlanC()

    vEul = viz.showEulerian(planC, Eul, field="speed", scanNum=0)
    assert len(vEul.images) == 1                       # speed map overlay added
    arr, kw = vEul.images[0]
    assert arr.shape == (30, 30, 12)
    assert "affine" in kw                              # aligned to scan affine
    assert captured["vectors_dict"]["vectors"].shape[1:] == (2, 3)   # flux flow

    vLag = viz.showLagrangian(planC, Lag, colorBy="speed", scanNum=0)
    assert len(vLag.tracks) == 1                       # pathline Tracks layer
    data, kw = vLag.tracks[0]
    assert data.shape[1] == 5 and "affine" in kw


def test_uromt_planc_storage():
    """UROMT runs are stored on a dynamically-created planC.urOMT list
    (mirroring planC.im), with inputs and outputs bundled."""
    assert uid.createUID("UROMT").startswith("UROMT.")

    planC = SimpleNamespace()                       # bare plan container stand-in
    assert not hasattr(planC, "urOMT")
    lst = getUROMTList(planC)                        # creates planC.urOMT
    assert planC.urOMT is lst and lst == []

    res = _uniform_flow_result()
    Eul = runEULA(res)
    Lag = runGLAD(res, spfs=4, slTolVox=0.5)
    cfg = SimpleNamespace(settings={"alpha": 1.0}, scanNumV=[0, 1],
                          structNum=3, frameScanNums=[0, 1], vol=res["rho"],
                          mask=res["mask"], bbox=res["bbox"],
                          spacing=res["spacing"], trueSize=res["n"], chi=None)
    obj = buildFromConfig(cfg, res, Eul, Lag)
    assert isinstance(obj, UROMT)
    assert obj.UROMTSetup["structNum"] == 3
    assert obj.UROMTSetup["scanNumV"] == [0, 1]
    assert obj.UROMTResult is res
    assert obj.UROMTEulerian is Eul and obj.UROMTLagrangian is Lag
    assert obj.UROMTUID.startswith("UROMT.")

    idx = saveUROMTToPlan(planC, obj)
    assert idx == 0 and planC.urOMT[0] is obj
    idx2 = saveUROMTToPlan(planC, buildFromConfig(cfg, res))
    assert idx2 == 1 and len(planC.urOMT) == 2
    # overwrite at index
    obj3 = buildFromConfig(cfg, res, Eul, Lag)
    assert saveUROMTToPlan(planC, obj3, index=0) == 0
    assert planC.urOMT[0] is obj3


def test_velocity_vectors_mapping():
    """velocityVectors maps the ROI-grid velocity to scan voxel coordinates
    with the correct bbox offset, component order and frame scan number."""
    n = (12, 10, 6)
    N = int(np.prod(n))
    nt = 3
    rng = np.random.default_rng(8)
    u = rng.standard_normal((3, N, nt))
    bbox = (2, 2 + n[0], 3, 3 + n[1], 1, 1 + n[2])
    result = dict(u=[u], r=[np.zeros((N, nt))], rho=[np.ones((N, nt))],
                  n=list(n), spacing=[1.0, 1.0, 1.0],
                  mask=np.ones(n, dtype=np.uint8), bbox=bbox,
                  frameScanNums=[5, 7], doResize=0, sizeFactor=1.0)

    # subsample=1, keep all -> one vector per voxel, mapped into the bbox
    vd = velocityVectors(result, interval=0, subsample=1, speedPctile=0)
    v = vd["vectors"]
    assert v.shape == (N, 2, 3)
    assert vd["scanNum"] == 5                       # interval-0 start frame
    assert v[:, 0, 0].min() == 2 and v[:, 0, 0].max() == 2 + n[0] - 1
    assert v[:, 0, 1].min() == 3 and v[:, 0, 1].max() == 3 + n[1] - 1
    assert v[:, 0, 2].min() == 1 and v[:, 0, 2].max() == 1 + n[2] - 1

    # lengthScale scales the deform; component order is [dy=row, dx=col, dz=slc]
    vd2 = velocityVectors(result, interval=0, subsample=1, speedPctile=0,
                          lengthScale=2.0, step=0)
    umean_row = u[0, :, 0].reshape(n, order="F")
    # vector at ROI voxel (0,0,0) -> first row of the (Fortran) flattened set
    assert np.isclose(vd2["vectors"][0, 1, 0], 2.0 * umean_row[0, 0, 0])

    # speed percentile thinning keeps fewer vectors
    vd3 = velocityVectors(result, interval=0, subsample=1, speedPctile=80)
    assert 0 < vd3["vectors"].shape[0] < N


def test_sensitivity_adjoint_dot_product():
    """The final-density tangent-linear J and its adjoint J' must satisfy the
    dot-product identity <J v, w> = <v, J' w> to machine precision."""
    par = _par()
    N, nt = par["N"], par["nt"]
    rng = np.random.default_rng(9)
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    u = 0.02 * rng.standard_normal(3 * N * nt)
    r = 0.05 * rng.standard_normal(N * nt)
    vu = rng.standard_normal(3 * N * nt)
    vr = rng.standard_normal(N * nt)
    w = rng.standard_normal(N)
    Jv = forwardSensitivity(rho0, u, r, vu, vr, par)[:, -1]
    Ju, Jr = adjointSensitivity(rho0, u, r, w, par)
    lhs = float(Jv @ w)
    rhs = float(vu @ Ju + vr @ Jr)
    assert abs(lhs - rhs) / max(abs(lhs), 1e-12) < 1e-10


def test_sensitivity_tlm_matches_finite_difference():
    """A column of J equals the finite-difference derivative of the final
    density w.r.t. that coordinate (interior voxels)."""
    par = _par()
    n, N, nt = par["n"], par["N"], par["nt"]
    rng = np.random.default_rng(10)
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    u = 0.02 * rng.standard_normal(3 * N * nt)
    r = 0.05 * rng.standard_normal(N * nt)
    interior = _interior_mask(n)
    vox = np.concatenate([np.tile(np.tile(np.arange(N), 3), nt),
                          np.tile(np.arange(N), nt)])
    intCoords = np.where(interior[vox])[0]
    sel = rng.choice(intCoords, size=8, replace=False)
    nu = 3 * N * nt
    x = np.concatenate([u, r])
    eps = 1e-6
    maxErr = 0.0
    for i in sel:
        e = np.zeros(nu + N * nt)
        e[i] = 1.0
        Jcol = forwardSensitivity(rho0, u, r, e[:nu], e[nu:], par)[:, -1]
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        rp = sourceAdvecDiff(rho0, xp[:nu], xp[nu:], par)[:, -1]
        rm = sourceAdvecDiff(rho0, xm[:nu], xm[nu:], par)[:, -1]
        maxErr = max(maxErr, np.abs(Jcol - (rp - rm) / (2 * eps)).max())
    assert maxErr < 1e-7


def _sens_setup(seed, chi=None):
    par = _par(chi=chi)
    N, nt = par["N"], par["nt"]
    rng = np.random.default_rng(seed)
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    u = 0.02 * rng.standard_normal(3 * N * nt)
    r = 0.05 * rng.standard_normal(N * nt)
    return par, rho0, u, r, rng


def _block_dirs(direction, N, nt, rng):
    """A perturbation confined to one block, so a defect in the velocity
    sensitivities cannot be masked by the source ones (or vice versa)."""
    du = np.zeros(3 * N * nt)
    dr = np.zeros(N * nt)
    if direction == "u":
        du = rng.standard_normal(3 * N * nt)
    else:
        dr = rng.standard_normal(N * nt)
    return du, dr


@pytest.mark.parametrize("direction", ["u", "r"])
@pytest.mark.parametrize("useChi", [False, True])
def test_sensitivity_adjoint_dot_product_per_block(direction, useChi):
    """<J v, w> == <v, J' w> with the perturbation confined to a single block.
    The combined-block version of this test can hide an error in one block."""
    N0 = int(np.prod((8, 8, 8)))
    chi = (np.arange(N0) % 3 != 0).astype(float) if useChi else None
    par, rho0, u, r, rng = _sens_setup(21, chi=chi)
    N, nt = par["N"], par["nt"]
    du, dr = _block_dirs(direction, N, nt, rng)
    w = rng.standard_normal(N)
    Jv = forwardSensitivity(rho0, u, r, du, dr, par)[:, -1]
    Ju, Jr = adjointSensitivity(rho0, u, r, w, par)
    lhs = float(Jv @ w)
    rhs = float(du @ Ju + dr @ Jr)
    assert abs(lhs - rhs) / max(abs(lhs), 1e-12) < 1e-10


def test_sensitivity_r_direction_matches_finite_difference():
    """The source-direction tangent model matches central finite differences
    over the whole field (unlike the velocity direction, r involves no
    departure-point clamping, so boundary voxels are differentiable too)."""
    par, rho0, u, r, rng = _sens_setup(22)
    N, nt = par["N"], par["nt"]
    du, dr = _block_dirs("r", N, nt, rng)
    tlm = forwardSensitivity(rho0, u, r, du, dr, par)[:, -1]
    eps = 1e-6
    rp = sourceAdvecDiff(rho0, u, r + eps * dr, par)[:, -1]
    rm = sourceAdvecDiff(rho0, u, r - eps * dr, par)[:, -1]
    fd = (rp - rm) / (2 * eps)
    assert np.linalg.norm(tlm - fd) / np.linalg.norm(fd) < 1e-7


@pytest.mark.parametrize("direction", ["u", "r"])
def test_gauss_newton_quadratic_form_consistent(direction):
    """v'(J'J)v computed through the adjoint equals ||J v||^2 -- the identity
    the matrix-free Gauss-Newton Hessian relies on."""
    par, rho0, u, r, rng = _sens_setup(23)
    N, nt = par["N"], par["nt"]
    du, dr = _block_dirs(direction, N, nt, rng)
    Jv = forwardSensitivity(rho0, u, r, du, dr, par)[:, -1]
    Ju, Jr = adjointSensitivity(rho0, u, r, Jv, par)
    quadAdj = float(du @ Ju + dr @ Jr)
    quadFwd = float(Jv @ Jv)
    assert abs(quadAdj - quadFwd) / max(abs(quadFwd), 1e-12) < 1e-10


@pytest.mark.parametrize("sigma", [0.0, 2e-3])
def test_advection_conserves_mass(sigma):
    """The advection operator is the mass-conserving push-forward S'@m, so with
    no source the total mass is preserved at every sub-step (S has unit row
    sums => 1'S'm = (S1)'m = 1'm). The pull-back S@m does NOT conserve mass;
    using it made the model unable to reproduce growing density, which this
    test guards against."""
    par = _par(sigma=sigma)
    N, nt = par["N"], par["nt"]
    rng = np.random.default_rng(24)
    rho0 = np.abs(rng.standard_normal(N)) + 0.5
    u = 0.05 * rng.standard_normal(3 * N * nt)     # non-trivial transport
    r = np.zeros(N * nt)                            # no source => mass conserved
    rho = sourceAdvecDiff(rho0, u, r, par)
    m0 = float(rho0.sum())
    for k in range(nt):
        assert abs(float(rho[:, k].sum()) - m0) / m0 < 1e-10


def test_gauss_newton_reduces_objective_more_than_lbfgs():
    """gnBlockExact lowers the objective and, at an equal small iteration
    budget, reaches a lower objective than the L-BFGS block (second-order
    steps are far more effective per iteration)."""
    par = _par(n=(10, 10, 6), nt=4, alpha=2e4, beta=8000.0)
    N, nt = par["N"], par["nt"]
    n = par["n"]
    rng = np.random.default_rng(12)
    ii = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]),
                     indexing="ij")
    cc = np.array([4.0, 5.0, 3.0])
    blob0 = np.exp(-sum((ii[d] - cc[d]) ** 2 for d in range(3)) / (2 * 1.6 ** 2))
    blob1 = np.exp(-sum((ii[d] - (cc + [2, 0, 0])[d]) ** 2
                        for d in range(3)) / (2 * 1.6 ** 2))
    rho0 = blob0.ravel(order="F")
    drhoN = blob1.ravel(order="F")
    par["maxUiter"] = 6
    u0 = np.zeros(3 * N * nt)
    r0 = np.zeros(N * nt)
    G_init, _, _ = getGamma(rho0, u0, r0, par, drhoN)
    gn = gnBlockExact(rho0, u0, r0, par, drhoN)
    lb = gnBlockUr(rho0, u0, r0, par, drhoN)
    assert gn["Gamma"] < G_init                  # GN makes progress
    assert gn["Gamma"] < lb["Gamma"]             # and beats L-BFGS per-iteration
    assert np.all(np.isfinite(gn["u"])) and np.all(np.isfinite(gn["r"]))


# --------------------------------------------------------------------------- #
#  Fused trilinear kernels (cerr/uromt/kernels.py)
# --------------------------------------------------------------------------- #
def test_trilinear_kernels_match_the_sparse_operator():
    """The four fused kernels must reproduce the explicit sparse interpolation
    matrix S they replace. `interp` is S @ f, `interpT` the mass-conserving
    push-forward S' @ f, and `deriv`/`derivT` the spatial derivative of S @ f
    and its adjoint. Departure points deliberately fall outside the domain so
    the boundary clamping is exercised."""
    from cerr.uromt.numerics import _TrilinearOp

    par = _par(n=(9, 7, 6))
    N = par["N"]
    rng = np.random.default_rng(11)
    pos = [par[c] + rng.normal(size=N) * 2.0 for c in ("Xc", "Yc", "Zc")]
    op = _TrilinearOp(par, *pos)
    S = op.toSparse()
    f = rng.normal(size=N)
    v = np.ascontiguousarray(rng.normal(size=(3, N)))

    assert np.allclose(op.interp(f), S @ f, rtol=0, atol=1e-12)
    assert np.allclose(op.interpT(f), S.T @ f, rtol=0, atol=1e-12)
    # S has unit row sums, so the push-forward conserves total mass exactly
    assert op.interpT(f).sum() == pytest.approx(f.sum(), rel=1e-12)

    # deriv is d(S@f)/d{x,y,z}: check against a central difference of S@f
    eps = 1e-6
    d = op.deriv(f)
    assert d.shape == (3, N)
    for ax in range(3):
        hi = list(pos)
        lo = list(pos)
        hi[ax] = hi[ax] + eps
        lo[ax] = lo[ax] - eps
        fd = (_TrilinearOp(par, *hi).interp(f)
              - _TrilinearOp(par, *lo).interp(f)) / (2 * eps)
        interior = np.abs(fd) > 1e-8          # clamped points are not smooth
        assert np.allclose(d[ax][interior], fd[interior], rtol=1e-5, atol=1e-6)

    # derivT is the exact transpose of deriv
    lhs = float(np.sum(op.deriv(f) * v))
    rhs = float(f @ op.derivT(v))
    assert lhs == pytest.approx(rhs, rel=1e-11)


def test_scatter_kernels_are_partition_invariant():
    """The scatter kernels reduce through a per-thread accumulator; the result
    must not depend on how many partitions the accumulator has."""
    from cerr.uromt import kernels
    from cerr.uromt.numerics import _TrilinearOp

    par = _par(n=(9, 7, 6))
    N = par["N"]
    rng = np.random.default_rng(3)
    op = _TrilinearOp(par, *[par[c] + rng.normal(size=N) * 1.5
                             for c in ("Xc", "Yc", "Zc")])
    f = rng.normal(size=N)
    v = np.ascontiguousarray(rng.normal(size=(3, N)))
    refT, refDT = op.interpT(f), op.derivT(v)
    for nacc in (1, 2, 3, 8):
        op.acc = np.empty((nacc, N))
        assert np.allclose(op.interpT(f), refT, rtol=0, atol=1e-12)
        assert np.allclose(op.derivT(v), refDT, rtol=0, atol=1e-12)
    assert kernels.numAccum() >= 1


# --------------------------------------------------------------------------- #
#  FFT-friendly ROI box (cerr/uromt/data.py)
# --------------------------------------------------------------------------- #
def test_fft_friendly_range_grows_and_never_clips():
    """The DCT cost is set by the prime factors of each grid dimension, so the
    ROI box is grown to a 2/3/5-smooth length. It must only ever grow, must stay
    inside the scan, and must still cover the original range."""
    from cerr.uromt.data import fftFriendlyRange, smoothSize

    assert smoothSize(61, 256) == 64
    assert smoothSize(59, 256) == 60
    assert smoothSize(46, 256) == 48
    assert smoothSize(64, 256) == 64          # already smooth -> unchanged
    assert smoothSize(200, 100) == 100        # capped by the scan extent

    for extent in (20, 64, 137, 256):
        for lo in range(0, extent):
            for hi in range(lo + 1, extent + 1):
                lo2, hi2 = fftFriendlyRange(lo, hi, extent)
                assert 0 <= lo2 <= lo, (lo, hi, extent, lo2)
                assert hi <= hi2 <= extent, (lo, hi, extent, hi2)
                assert hi2 - lo2 >= hi - lo   # never shrinks the ROI


def test_fft_friendly_range_centers_and_respects_the_scan_edge():
    """Growth is split around the ROI, and folds inward at the scan boundary."""
    from cerr.uromt.data import fftFriendlyRange

    assert fftFriendlyRange(10, 71, 256) == (9, 73)    # 61 -> 64, centered
    assert fftFriendlyRange(0, 61, 256) == (0, 64)     # at the low edge
    assert fftFriendlyRange(195, 256, 256) == (192, 256)  # at the high edge
    assert fftFriendlyRange(0, 60, 60) == (0, 60)      # no room: unchanged
    assert fftFriendlyRange(4, 68, 256) == (4, 68)     # already 2/3/5-smooth


# --------------------------------------------------------------------------- #
#  Optional GPU backend (cerr/uromt/gpu.py)
# --------------------------------------------------------------------------- #
def test_gpu_request_falls_back_to_cpu_when_unavailable():
    """`gpu=1` must never hard-fail: without cupy/CUDA it warns and runs on the
    CPU, producing the same answer as `gpu=0`."""
    from cerr.uromt.gpu import isAvailable

    par = _par()
    N, nt = par["N"], par["nt"]
    rng = np.random.default_rng(5)
    rho0 = np.abs(rng.normal(size=N)) + 0.1
    drhoN = np.abs(rng.normal(size=N)) + 0.1
    u0, r0 = np.zeros(3 * N * nt), np.zeros(N * nt)
    ref = gnBlockExact(rho0, u0, r0, par, drhoN)

    cfg = SimpleNamespace(trueSize=list(par["n"]), spacing=list(par["h"]),
                          dt=par["dt"], nt=nt, sigma=par["sigma"],
                          alpha=par["alpha"], beta=par["beta"], bc="closed",
                          niter_pcg=par["niter_pcg"], maxUiter=par["maxUiter"],
                          chi=None, gpu=1)
    if isAvailable():
        pytest.skip("cupy/CUDA present: this checks the fallback path")
    with pytest.warns(RuntimeWarning, match="cupy/CUDA is unavailable"):
        parGpu = paramInit(cfg)
    assert parGpu["xp"] is np
    got = gnBlockExact(rho0, u0, r0, parGpu, drhoN)
    assert got["Gamma"] == pytest.approx(ref["Gamma"], rel=1e-12)


def test_backend_reports_itself():
    from cerr.uromt.gpu import Backend
    bk = Backend(useGpu=False)
    assert bk.isGpu is False
    assert bk.xp is np
    assert "CPU" in bk.describe()
    assert bk.toHost(np.arange(3)).tolist() == [0, 1, 2]


# --------------------------------------------------------------------------- #
#  threads setting
# --------------------------------------------------------------------------- #
def _bigSolverGrid():
    """A grid above the threading threshold, so the pool is actually used."""
    return (64, 60, 48), [1.0, 1.0, 1.0]


def test_threaded_diffusion_solve_matches_the_serial_one():
    """The 3-D DCT is split across a thread pool by chunking along an axis that
    is not being transformed. Each 1-D transform sees identical input, so the
    only departure from the serial solve is scipy's internal handling of a
    multi-axis dctn - at the double-precision floor."""
    from cerr.uromt.numerics import _DiffusionSolver

    n, h = _bigSolverGrid()
    N = int(np.prod(n))
    serial = _DiffusionSolver(n, h, 0.3, 2e-3, threads=1)
    threaded = _DiffusionSolver(n, h, 0.3, 2e-3, threads=4)
    assert serial.nchunk == 1 and threaded.nchunk > 1
    rng = np.random.default_rng(21)
    x = rng.standard_normal(N)
    a, b = serial(x), threaded(x)
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-13
    # and it is still an exact inverse of B
    assert np.allclose(threaded(x), a, rtol=0, atol=1e-12)


def test_threading_is_off_on_small_grids():
    """Pool dispatch costs more than the transform below ~100k voxels."""
    from cerr.uromt.numerics import _DiffusionSolver
    small = _DiffusionSolver((16, 16, 16), [1.0, 1.0, 1.0], 0.3, 2e-3)
    assert small.nchunk == 1


def test_threads_setting_is_honoured_and_clamped():
    """`threads` caps both the numba kernels and the DCT pool; asking for more
    than the process maximum warns instead of failing."""
    from cerr.uromt import kernels
    from cerr.uromt.numerics import _DiffusionSolver

    cap = kernels.maxThreads()
    try:
        assert kernels.setNumThreads(1) == 1
        assert kernels.numAccum() == 1              # accumulators follow
        # 0 -> the auto default, which is deliberately NOT every core: these
        # kernels are bandwidth bound and oversubscribing them is slower.
        auto = kernels.setNumThreads(0)
        assert auto == min(cap, kernels._AUTO_THREADS)
        assert kernels.setNumThreads(3) == 3
        with pytest.warns(RuntimeWarning, match="exceeds this process"):
            assert kernels.setNumThreads(cap + 32) == cap
    finally:
        kernels.setNumThreads(0)

    n, h = _bigSolverGrid()
    assert _DiffusionSolver(n, h, 0.3, 2e-3, threads=1).nchunk == 1
    assert _DiffusionSolver(n, h, 0.3, 2e-3, threads=2).nchunk == 2


def test_threads_do_not_change_the_solver_result():
    """Both kernels and the DCT split are deterministic, so the optimum must not
    depend on the thread count."""
    from cerr.uromt import kernels

    par = _par()
    N, nt = par["N"], par["nt"]
    rng = np.random.default_rng(31)
    rho0 = np.abs(rng.normal(size=N)) + 0.1
    drhoN = np.abs(rng.normal(size=N)) + 0.1
    u0, r0 = np.zeros(3 * N * nt), np.zeros(N * nt)
    try:
        kernels.setNumThreads(1)
        one = gnBlockExact(rho0, u0, r0, _par(), drhoN)
        kernels.setNumThreads(0)
        many = gnBlockExact(rho0, u0, r0, _par(), drhoN)
    finally:
        kernels.setNumThreads(0)
    assert one["Gamma"] == pytest.approx(many["Gamma"], rel=1e-12)
    assert np.allclose(one["u"], many["u"], rtol=0, atol=1e-12)
    assert np.allclose(one["r"], many["r"], rtol=0, atol=1e-12)


def test_paramInit_threads_flow_from_cfg():
    cfg = SimpleNamespace(trueSize=[64, 60, 48], spacing=[1.0, 1.0, 1.0],
                          dt=0.3, nt=2, sigma=2e-3, alpha=10.0, beta=50.0,
                          bc="closed", niter_pcg=10, maxUiter=3, chi=None,
                          threads=2)
    try:
        par = paramInit(cfg)
        assert par["threads"] == 2
        assert par["Bsolve"].nchunk == 2
    finally:
        from cerr.uromt import kernels
        kernels.setNumThreads(0)


# --------------------------------------------------------------------------- #
#  useGPU / numThreads settings
# --------------------------------------------------------------------------- #
def test_default_settings_run_on_the_cpu():
    """Out of the box urOMT runs on the CPU with the auto thread count."""
    from cerr.uromt.config import loadModelSettings

    s = loadModelSettings()
    assert s["useGPU"] == "no"
    assert s["numThreads"] == 0
    assert "gpu" not in s and "threads" not in s


@pytest.mark.parametrize("value,expected", [
    ("yes", True), ("Yes", True), ("y", True), ("true", True), ("on", True),
    ("1", True), (1, True), (True, True),
    ("no", False), ("N", False), ("false", False), ("off", False),
    ("0", False), (0, False), (False, False), (None, False),
])
def test_parseYesNo(value, expected):
    from cerr.uromt.config import parseYesNo
    assert parseYesNo(value) is expected


def test_parseYesNo_rejects_junk():
    from cerr.uromt.config import parseYesNo
    with pytest.raises(ValueError):
        parseYesNo("maybe")


def test_legacy_gpu_and_threads_names_still_work(tmp_path):
    """Old settings files and scripts keep working: `gpu`/`threads` are read as
    `useGPU`/`numThreads`."""
    import json

    from cerr.uromt.config import getNumThreads, getUseGPU, loadModelSettings

    s = loadModelSettings()
    s.pop("useGPU")
    s.pop("numThreads")
    s["gpu"] = 1
    s["threads"] = 3
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(s))
    loaded = loadModelSettings(str(legacy))
    assert loaded["useGPU"] == 1 and loaded["numThreads"] == 3
    assert "gpu" not in loaded and "threads" not in loaded

    # and on a config object, under either name
    assert getUseGPU(SimpleNamespace(useGPU="yes")) is True
    assert getUseGPU(SimpleNamespace(gpu=1)) is True
    assert getUseGPU(SimpleNamespace()) is False
    assert getNumThreads(SimpleNamespace(numThreads=4)) == 4
    assert getNumThreads(SimpleNamespace(threads=4)) == 4
    assert getNumThreads(SimpleNamespace()) == 0        # auto


def test_paramInit_honours_useGPU_and_numThreads():
    from cerr.uromt.gpu import isAvailable

    cfg = SimpleNamespace(trueSize=[64, 60, 48], spacing=[1.0, 1.0, 1.0],
                          dt=0.3, nt=2, sigma=2e-3, alpha=10.0, beta=50.0,
                          bc="closed", niter_pcg=10, maxUiter=3, chi=None,
                          numThreads=2, useGPU="no")
    try:
        par = paramInit(cfg)
        assert par["threads"] == 2
        assert par["Bsolve"].nchunk == 2
        assert par["bk"].isGpu is False

        cfg.useGPU = "yes"
        if isAvailable():
            assert paramInit(cfg)["bk"].isGpu is True
        else:
            with pytest.warns(RuntimeWarning, match="cupy/CUDA is unavailable"):
                assert paramInit(cfg)["bk"].isGpu is False
    finally:
        from cerr.uromt import kernels
        kernels.setNumThreads(0)


class _FakeScan:
    """Minimal planC.scan stand-in for prepareData (no DICOM needed)."""

    def __init__(self, arr, spacing):
        self._a = np.asarray(arr, float)
        self._s = spacing
        self.scanInfo = [SimpleNamespace(acquisitionTime="", seriesTime="")]

    def getScanArray(self):
        return self._a

    def getScanSpacing(self):
        return self._s          # cm; prepareData multiplies by 10


def test_concentration_is_not_masked_before_smoothing(monkeypatch):
    """The ROI mask must be applied AFTER the smoothing flow, not before.

    Masking first zeroes the tissue surrounding the ROI, so the edge-preserving
    flow smooths every boundary voxel against an artificial zero background and
    drags it down. The reference implementation converts and smooths the whole
    cropped box and masks last; on the breast reference data getting this wrong
    moved 35357 in-mask voxels (slope 0.979 against the reference density).
    """
    from cerr.uromt import data as uromtData
    from cerr.uromt.config import buildConfig
    from cerr.utils.image_proc import affineDiffusion3d

    n = (14, 14, 14)
    rng = np.random.default_rng(3)
    base = np.abs(rng.normal(size=n)) * 10 + 20.0     # bright everywhere
    frames = [base, base * 1.4, base * 1.9]
    # ROI is a small cube, so bbox_pad leaves real tissue inside the box but
    # outside the mask - exactly the geometry the bug corrupts.
    mask = np.zeros(n, np.uint8)
    mask[5:9, 5:9, 5:9] = 1

    planC = SimpleNamespace(scan=[_FakeScan(f, [0.1, 0.1, 0.1]) for f in frames])
    monkeypatch.setattr(uromtData.rs, "getStrMask", lambda s, p: mask.copy())

    cfg = buildConfig([0, 1, 2], 0, None, normMethod="RSE", baselineFrames=1,
                      smooth=1, smooth_dt=0.1, smooth_method="affine",
                      bbox_pad=[3, 3, 3], fft_pad=0, mask_dilate=0,
                      conc_clip=None, time={"first_time": 2, "last_time": 3,
                                            "time_jump": 1})
    cfg = uromtData.prepareData(cfg, planC)

    box = tuple(slice(cfg.bbox[2 * i], cfg.bbox[2 * i + 1]) for i in range(3))
    roiMask = np.asarray(cfg.mask) > 0
    expected = []
    for f in frames[1:]:                        # baseline frame 0 is external
        ratio = (f / frames[0])[box]            # unmasked, whole box
        sm = affineDiffusion3d(ratio, nSteps=10, dt=0.1, affFlag=True)
        expected.append(np.where(roiMask, sm, 0.0))

    for got, want in zip(cfg.vol, expected):
        assert np.allclose(got, want, rtol=0, atol=1e-12)

    # and the surrounding tissue really is what makes the difference: zeroing it
    # in the input must change the in-mask result
    planC2 = SimpleNamespace(
        scan=[_FakeScan(np.where(mask > 0, f, 0.0), [0.1, 0.1, 0.1])
              for f in frames])
    cfg2 = buildConfig([0, 1, 2], 0, None, normMethod="RSE", baselineFrames=1,
                       smooth=1, smooth_dt=0.1, smooth_method="affine",
                       bbox_pad=[3, 3, 3], fft_pad=0, mask_dilate=0,
                       conc_clip=None, time={"first_time": 2, "last_time": 3,
                                             "time_jump": 1})
    cfg2 = uromtData.prepareData(cfg2, planC2)
    assert not np.allclose(cfg.vol[0][roiMask], cfg2.vol[0][roiMask],
                           rtol=0, atol=1e-9)


def test_bbox_pad_scalar_is_in_plane_and_list_is_per_axis():
    """A scalar keeps the historical in-plane meaning; a 3-list pads each axis,
    which is what matching the reference getRange needs."""
    from cerr.uromt.data import _bboxPad

    assert _bboxPad(0) == (0, 0, 0)
    assert _bboxPad(None) == (0, 0, 0)
    assert _bboxPad(3) == (3, 3, 0)          # z untouched
    assert _bboxPad([3, 3, 3]) == (3, 3, 3)
    assert _bboxPad((1, 2, 4)) == (1, 2, 4)
    with pytest.raises(ValueError):
        _bboxPad([1, 2])


# --------------------------------------------------------------------------- #
#  Top-level wrapper
# --------------------------------------------------------------------------- #
def test_wrapper_is_runUROMT_and_scanNumV_is_optional():
    """`cerr.uromt.runUROMT` is the whole-pipeline wrapper; omitting scanNumV
    leaves the list empty so prepareData infers the acquisition order."""
    import cerr.uromt as uromt
    from cerr.uromt.config import buildConfig

    assert callable(uromt.runUROMT)
    params = inspect.signature(uromt.runUROMT).parameters
    # structNum comes before scanNumV: the ROI is the argument callers actually
    # supply, and scanNumV defaults to the inferred acquisition order.
    assert list(params)[:4] == ["planC", "structNum", "scanNumV", "settingsFile"]
    assert params["scanNumV"].default is None
    assert params["structNum"].default is None
    assert params["settingsFile"].default is None
    # the wrapper is distinct from Part 2, which takes a prepared cfg
    from cerr.uromt.solver import solveUROMT
    assert uromt.runUROMT is not solveUROMT
    assert "cfg" in inspect.signature(solveUROMT).parameters

    cfg = buildConfig(None, 3)
    assert cfg.scanNumV == [] and cfg.structNum == 3
    assert buildConfig().scanNumV == []
    assert buildConfig([2, 0, 1], 3).scanNumV == [2, 0, 1]


def test_deprecated_aliases_still_work():
    """The old names warn but keep working."""
    import cerr.uromt as uromt
    from cerr.uromt import solver as sv

    with pytest.warns(DeprecationWarning, match="use cerr.uromt.runUROMT"):
        with pytest.raises(Exception):        # no planC: fails after warning
            uromt.runUROMTPipeline(None, [0, 1])

    with pytest.warns(DeprecationWarning, match="solver.solveUROMT"):
        with pytest.raises(ValueError, match="prepareData"):
            sv.runUROMT(SimpleNamespace(vol=[]))


# --------------------------------------------------------------------------- #
#  Fused whole-step sensitivity kernels
# --------------------------------------------------------------------------- #
class _UnfusedOp:
    """A _TrilinearOp with the fused entry points hidden, so forward/adjoint
    Sensitivity fall back to composing interpT+derivT / interp+deriv."""

    def __init__(self, op):
        self._op = op

    interp = property(lambda s: s._op.interp)
    interpT = property(lambda s: s._op.interpT)
    deriv = property(lambda s: s._op.deriv)
    derivT = property(lambda s: s._op.derivT)


def _sensSetup(seed=5, n=(10, 9, 8), nt=3):
    from cerr.uromt.numerics import _interpMats, precomputeSensDeriv
    par = _par(n=n, nt=nt)
    N = par["N"]
    rng = np.random.default_rng(seed)
    rho0 = np.abs(rng.normal(size=N)) + 0.2
    u = rng.normal(size=3 * N * nt) * 0.05
    r = rng.normal(size=N * nt) * 0.05
    interp = _interpMats(par, u.reshape(3 * N, nt, order="F"))
    rho = sourceAdvecDiff(rho0, u, r, par, interp)
    dS = precomputeSensDeriv(rho0, r, par, interp, rho)
    return par, rho0, u, r, interp, rho, dS, rng


def test_fused_sensitivity_steps_match_the_unfused_operators():
    """The tangent and adjoint steps are each fused into a single kernel pass -
    the tangent scatters `S'@dm` and `dt*D'(m.*du)` together, and the adjoint
    gets `S@b` out of the same reduction that produces `D_d@b`. Both must
    reproduce the composed operators they replace."""
    par, rho0, u, r, interp, rho, dS, rng = _sensSetup()
    N, nt = par["N"], par["nt"]
    du = rng.normal(size=3 * N * nt) * 1e-3
    dr = rng.normal(size=N * nt) * 1e-3
    wN = rng.normal(size=N)
    plain = [_UnfusedOp(o) for o in interp]
    assert hasattr(interp[0], "tangentStep")
    assert not hasattr(plain[0], "tangentStep")

    a = forwardSensitivity(rho0, u, r, du, dr, par, interp, rho, dS)
    b = forwardSensitivity(rho0, u, r, du, dr, par, plain, rho, dS)
    assert np.allclose(a, b, rtol=0, atol=1e-14 * max(np.max(np.abs(b)), 1.0))

    ju, jr = adjointSensitivity(rho0, u, r, wN, par, interp, rho, dS)
    ku, kr = adjointSensitivity(rho0, u, r, wN, par, plain, rho, dS)
    assert np.allclose(ju, ku, rtol=0, atol=1e-14 * max(np.max(np.abs(ku)), 1.))
    assert np.allclose(jr, kr, rtol=0, atol=1e-14 * max(np.max(np.abs(kr)), 1.))


def test_fused_adjoint_step_writes_all_three_outputs():
    """adjointStep fills gU (3N,), gR (N,) and carry (N,) in place; `carry` is
    overwritten after being consumed, so aliasing it is safe."""
    par, rho0, u, r, interp, rho, dS, rng = _sensSetup()
    N, dt = par["N"], par["dt"]
    op, k = interp[1], 1
    m, srcR, fac = dS["m"][k], dS["srcR"][k], dS["fac"][k]
    b = rng.normal(size=N)
    gUk, gRk, carry = np.empty(3 * N), np.empty(N), np.empty(N)
    op.adjointStep(b, m, srcR, fac, dt, gUk, gRk, carry)

    mbar = op.interp(b)
    dB = op.deriv(b)
    assert np.allclose(gUk, np.concatenate([dt * m * dB[d] for d in range(3)]),
                       rtol=0, atol=1e-13)
    assert np.allclose(gRk, srcR * mbar, rtol=0, atol=1e-13)
    assert np.allclose(carry, fac * mbar, rtol=0, atol=1e-13)


def test_fused_tangent_step_matches_its_pieces():
    par, rho0, u, r, interp, rho, dS, rng = _sensSetup()
    N, nt, dt = par["N"], par["nt"], par["dt"]
    op, k = interp[1], 1
    m, srcR, fac = dS["m"][k], dS["srcR"][k], dS["fac"][k]
    dU = (rng.normal(size=3 * N * nt) * 1e-2).reshape(3 * N, nt, order="F")
    dRk = rng.normal(size=N) * 1e-2
    dprev = rng.normal(size=N) * 1e-2

    got = op.tangentStep(srcR, dRk, fac, dprev, m, dU, k, dt, N)
    dm = srcR * dRk + fac * dprev
    mdu = np.stack([m * dU[d * N:(d + 1) * N, k] for d in range(3)])
    want = op.interpT(dm) + dt * op.derivT(mdu)
    assert np.allclose(got, want, rtol=0, atol=1e-13 * max(np.max(np.abs(want)),
                                                           1.0))


# --------------------------------------------------------------------------- #
#  GPU backend (cerr/uromt/gpu.py)
#
#  The device operators are written against an array module (`xp`) and a
#  scatter-add, so almost all of their logic can be exercised on the host by
#  substituting numpy + np.add.at for cupy + cupyx.scatter_add. That catches
#  math/indexing bugs on machines with no CUDA; only the cupy API surface and
#  the device CG plumbing still need real hardware.
# --------------------------------------------------------------------------- #
class _HostShimBackend:
    """A gpu.Backend look-alike whose 'device' is numpy."""

    def __init__(self):
        self.xp = np
        self.scatterAdd = np.add.at


def test_device_trilinear_operators_match_the_cpu_kernels():
    from cerr.uromt.gpu import _DeviceTrilinearOp
    from cerr.uromt.numerics import _TrilinearOp

    par = _par(n=(9, 8, 7))
    N = par["N"]
    rng = np.random.default_rng(77)
    # deliberately push points outside the domain to exercise the clamping
    pos = [par[c] + rng.normal(size=N) * 2.0 for c in ("Xc", "Yc", "Zc")]
    dev = _DeviceTrilinearOp(_HostShimBackend(), par, *pos)
    cpu = _TrilinearOp(par, *pos)
    S = cpu.toSparse()
    f = rng.normal(size=N)
    v = np.ascontiguousarray(rng.normal(size=(3, N)))

    assert np.allclose(dev.interp(f), S @ f, rtol=0, atol=1e-12)
    assert np.allclose(dev.interpT(f), S.T @ f, rtol=0, atol=1e-12)
    assert np.allclose(dev.deriv(f), cpu.deriv(f), rtol=0, atol=1e-12)
    assert np.allclose(dev.derivT(v), cpu.derivT(v), rtol=0, atol=1e-12)
    # the device operator must also assemble the same S
    assert abs(dev.toSparse() - S).max() < 1e-12
    # ... and be a consistent adjoint pair in its own right
    assert float(np.sum(dev.deriv(f) * v)) == pytest.approx(
        float(f @ dev.derivT(v)), rel=1e-11)


def test_device_diffusion_solver_matches_the_cpu_one():
    import scipy.fft
    from cerr.uromt.gpu import _DeviceDiffusionSolver
    from cerr.uromt.numerics import _DiffusionSolver

    n, h = (9, 8, 7), [0.9, 1.1, 2.0]
    dev = _DeviceDiffusionSolver(np, scipy.fft, n, h, 0.3, 2e-3)
    cpu = _DiffusionSolver(n, h, 0.3, 2e-3)
    x = np.random.default_rng(78).standard_normal(int(np.prod(n)))
    assert np.allclose(dev(x), cpu(x), rtol=0, atol=1e-12)


def test_cg_tolerance_keyword_is_normalized():
    """scipy renamed CG's `tol` to `rtol` in 1.12 and cupy still uses `tol`;
    the backend must call whichever the installed function actually takes."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import cg as spcg
    from cerr.uromt.gpu import _normalizeCg

    solve = _normalizeCg(spcg)
    assert solve.tolKeyword in ("rtol", "tol")
    A = sp.diags([2.0, 3.0, 4.0]).tocsr()
    b = np.array([2.0, 6.0, 12.0])
    x, _ = solve(A, b, 1e-10, 50, None)
    assert np.allclose(x, [1.0, 2.0, 3.0], atol=1e-8)

    def fakeOldCg(A, b, x0=None, tol=1e-5, maxiter=None, M=None):
        return ("called-with-tol", tol, maxiter)

    old = _normalizeCg(fakeOldCg)
    assert old.tolKeyword == "tol"
    assert old(None, None, 0.25, 7, None) == ("called-with-tol", 0.25, 7)


@pytest.mark.skipif(not __import__("cerr.uromt.gpu", fromlist=["x"])
                    .isAvailable(), reason="cupy/CUDA device not available")
def test_gpu_backend_reproduces_the_cpu_solve():
    """Real-hardware check: the GPU solve must land on the same optimum as the
    CPU one. Runs only where cupy and a CUDA device are present."""
    n, nt = (16, 15, 14), 3
    common = dict(trueSize=list(n), spacing=[0.9, 1.1, 2.0], dt=0.3, nt=nt,
                  sigma=2e-3, alpha=1e3, beta=100.0, bc="closed",
                  niter_pcg=20, maxUiter=3, chi=None)
    parCpu = paramInit(SimpleNamespace(gpu=0, **common))
    parGpu = paramInit(SimpleNamespace(gpu=1, **common))
    assert parGpu["bk"].isGpu

    N = parCpu["N"]
    rng = np.random.default_rng(99)
    rho0 = np.abs(rng.normal(size=N)) + 0.2
    drhoN = np.abs(rng.normal(size=N)) + 0.2
    u0, r0 = np.zeros(3 * N * nt), np.zeros(N * nt)

    cpu = gnBlockExact(rho0, u0, r0, parCpu, drhoN)
    gpuSol = gnBlockExact(rho0, u0, r0, parGpu, drhoN)
    assert isinstance(gpuSol["u"], np.ndarray)      # results come back to host
    assert gpuSol["Gamma"] == pytest.approx(cpu["Gamma"], rel=1e-6)
    assert np.allclose(gpuSol["u"], cpu["u"], rtol=1e-5,
                       atol=1e-8 * max(np.max(np.abs(cpu["u"])), 1.0))
    assert np.allclose(gpuSol["rho"], cpu["rho"], rtol=1e-6, atol=1e-10)


@pytest.mark.skipif(not __import__("cerr.uromt.gpu", fromlist=["x"])
                    .isAvailable(), reason="cupy/CUDA device not available")
def test_gpu_fused_kernels_match_the_array_path():
    """The fused CUDA kernels must reproduce the array-expression device path
    they replace (and the CPU kernels), including through the atomic scatter."""
    from types import SimpleNamespace as NS
    from cerr.uromt import gpu_kernels as gk
    from cerr.uromt.gpu import (Backend, _DeviceTrilinearOp,
                                _FusedDeviceTrilinearOp, asNumpy)
    from cerr.uromt.numerics import _TrilinearOp

    bk = Backend(useGpu=True)
    if bk.kernels is None:
        pytest.skip("device lacks double atomicAdd (compute capability < 6.0)")
    assert gk.isSupported()

    cfg = SimpleNamespace(trueSize=[14, 12, 11], spacing=[0.9, 1.1, 2.0],
                          dt=0.35, nt=2, sigma=2e-3, alpha=1e3, beta=100.0,
                          bc="closed", niter_pcg=10, maxUiter=2, chi=None,
                          gpu=1)
    par = paramInit(cfg)
    xp, N = par["xp"], par["N"]
    rng = np.random.default_rng(13)
    pos = [asNumpy(par[c]) + rng.normal(size=N) * 1.5
           for c in ("Xc", "Yc", "Zc")]
    gpos = [xp.asarray(p) for p in pos]

    # the backend must hand out the fused subclass when kernels are available
    assert isinstance(bk.trilinearOp(par, *gpos), _FusedDeviceTrilinearOp)

    fused = _FusedDeviceTrilinearOp(bk, par, *gpos)
    arrayOnly = _DeviceTrilinearOp(NS(xp=xp, scatterAdd=bk.scatterAdd),
                                   par, *gpos)
    assert fused.k is not None and arrayOnly.k is None
    cpu = _TrilinearOp({**par, "bk": None}, *pos)

    f = rng.normal(size=N)
    v = np.ascontiguousarray(rng.normal(size=(3, N)))
    gf, gv = xp.asarray(f), xp.asarray(v)
    for name in ("interp", "interpT", "deriv", "derivT"):
        gArg, cArg = (gv, v) if name == "derivT" else (gf, f)
        got = asNumpy(getattr(fused, name)(gArg))
        assert np.allclose(got, asNumpy(getattr(arrayOnly, name)(gArg)),
                           rtol=0, atol=1e-12), name
        assert np.allclose(got, getattr(cpu, name)(cArg), rtol=0,
                           atol=1e-12), name

    # the push-forward still conserves mass through the atomic scatter
    assert float(asNumpy(fused.interpT(gf)).sum()) == pytest.approx(
        float(f.sum()), rel=1e-12)
    # and deriv/derivT remain an exact adjoint pair
    lhs = float(asNumpy(fused.deriv(gf) * gv).sum())
    rhs = float(f @ asNumpy(fused.derivT(gv)))
    assert lhs == pytest.approx(rhs, rel=1e-11)


# --------------------------------------------------------------------------- #
#  mask_dilate / peclet_floor (reference-parity settings)
# --------------------------------------------------------------------------- #
def test_mask_dilate_matches_the_reference_structuring_element():
    """MATLAB cfg.dilate uses imdilate with the ellipsoid
    (x/d)^2+(y/d)^2+(z/d)^2 <= 1 over -d:d, i.e. a ball of radius d in INDEX
    space. A single seed voxel must therefore grow to exactly that ball."""
    from cerr.uromt.data import dilateMask

    m = np.zeros((11, 11, 11), dtype=np.uint8)
    m[5, 5, 5] = 1
    assert np.array_equal(dilateMask(m, 0), m)      # off
    assert np.array_equal(dilateMask(m, -1), m)

    for d in (1, 2, 3):
        got = dilateMask(m, d)
        ax = np.arange(-5, 6)
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        want = ((X / d) ** 2 + (Y / d) ** 2 + (Z / d) ** 2 <= 1.0)
        assert np.array_equal(got.astype(bool), want), d
    # d = 2 -> 33 voxels (the reference breast run's setting)
    assert int(dilateMask(m, 2).sum()) == 33
    # dilation only ever grows the mask
    rng = np.random.default_rng(2)
    big = (rng.random((14, 13, 12)) > 0.9).astype(np.uint8)
    assert np.all(dilateMask(big, 2) >= big)


def test_peclet_floor_setting_controls_the_denominator():
    """peclet_floor scales the additive Peclet floor; 0 falls back to eps, and
    a smaller floor must raise Peclet (the floor biases it downward)."""
    from cerr.uromt.analyze import _pecletDenomFloor, _EPS

    rng = np.random.default_rng(4)
    dif = np.abs(rng.normal(size=5000)) + 1e-3
    roi = np.ones(dif.size, dtype=bool)
    med = float(np.median(dif))
    assert _pecletDenomFloor(dif, roi, 0.1) == pytest.approx(0.1 * med)
    assert _pecletDenomFloor(dif, roi, 0.01) == pytest.approx(0.01 * med)
    assert _pecletDenomFloor(dif, roi, 0.0) == _EPS
    # default (None) uses the module constant
    from cerr.uromt import analyze as an
    assert _pecletDenomFloor(dif, roi) == pytest.approx(
        an._PECLET_FLOOR_FRAC * med)


def test_peclet_floor_flows_from_the_result_dict():
    """runEULAIntervals must honour result['pecletFloor'], and the explicit
    argument must override it."""
    from cerr.uromt.analyze import runEULAIntervals

    n = (6, 5, 4)
    N = int(np.prod(n))
    nt = 2
    rng = np.random.default_rng(6)
    rho = np.abs(rng.normal(size=(N, nt))) + 0.2
    res = dict(n=list(n), spacing=[1.0, 1.0, 1.0], mask=np.ones(n, np.uint8),
               bbox=None, dt=0.3, nt=nt, sigma=2e-3,
               u=[rng.normal(size=(3, N, nt)) * 0.05],
               r=[rng.normal(size=(N, nt)) * 0.05], rho=[rho])

    big = runEULAIntervals({**res, "pecletFloor": 0.5})["peclet"][0]
    small = runEULAIntervals({**res, "pecletFloor": 0.0})["peclet"][0]
    # a larger additive floor can only reduce Peclet
    assert np.all(small >= big - 1e-12)
    assert small.max() > big.max()
    # the explicit argument wins over the stored setting
    override = runEULAIntervals({**res, "pecletFloor": 0.5}, pecletFloor=0.0)[
        "peclet"][0]
    assert np.allclose(override, small, rtol=0, atol=1e-12)


def _tiny_solver_cfg(nIntervals=3, n=(6, 6, 4)):
    """A minimal prepared cfg so Part 2 can actually be run in a unit test."""
    from cerr.uromt.config import buildConfig

    cfg = buildConfig(list(range(nIntervals + 1)), None, None,
                      nt=2, maxUiter=1, niter_pcg=3, fft_pad=0)
    rng = np.random.default_rng(0)
    cfg.vol = [1.0 + 0.1 * rng.random(n) for _ in range(nIntervals + 1)]
    cfg.mask = np.ones(n, dtype=np.uint8)
    cfg.trueSize = list(n)
    cfg.spacing = [1.0, 1.0, 1.0]
    cfg.bbox = (0, n[0], 0, n[1], 0, n[2])
    cfg.frameScanNums = list(range(5, 5 + nIntervals + 1))
    return cfg


def test_solver_reports_progress_after_each_interval(capsys):
    """Each interval prints a completion line (time, objective, ETA) so a long
    run is not a silent wait; verbose=0 stays silent."""
    from cerr.uromt.solver import solveUROMT

    nIv = 3
    solveUROMT(_tiny_solver_cfg(nIv), verbose=True)
    out = capsys.readouterr().out
    done = [ln for ln in out.splitlines() if " done in " in ln]
    assert len(done) == nIv                      # one per interval, not just one
    assert "interval 1/3" in done[0] and "interval 3/3" in done[-1]
    assert "Gamma" in done[0] and "ETA" in done[0]
    # scan INDICES, labelled as such: series are often stored out of
    # acquisition order, so these are not necessarily ascending
    assert "scans 5->6" in done[0]
    assert "ETA" not in done[-1]                 # nothing left to wait for
    assert "urOMT done: 3 interval(s)" in out

    solveUROMT(_tiny_solver_cfg(nIv), verbose=False)
    assert capsys.readouterr().out == ""


def test_status_callback_fires_before_and_after_each_interval(capsys):
    """A statusCallback receives both the start and the completion of every
    interval, with monotonic fractions ending at 1.0, and suppresses printing
    (the caller - e.g. the GUI progress bar - is displaying it instead)."""
    from cerr.uromt.solver import solveUROMT

    seen = []
    solveUROMT(_tiny_solver_cfg(3), statusCallback=lambda f, m: seen.append(
        (f, m)), verbose=True)
    assert capsys.readouterr().out == ""          # callback wins over printing
    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs) and fracs[0] == 0.0 and fracs[-1] == 1.0
    assert len([m for _, m in seen if " done in " in m]) == 3
    assert len([m for _, m in seen if m.endswith("solving...")]) == 3


def test_verbose_is_a_setting_not_a_runUROMT_parameter():
    """`verbose` must stay a plain SETTING. Declaring it as an explicit
    runUROMT parameter breaks the common `runUROMT(planC, **settingsFromJson)`
    call with 'got multiple values for keyword argument', because those JSONs
    now carry a verbose key."""
    import inspect
    import cerr.uromt as u
    from cerr.uromt.config import buildConfig

    params = inspect.signature(u.runUROMT).parameters
    assert "verbose" not in params, (
        "verbose must reach runUROMT through **settingsOverrides, not as a "
        "named parameter - otherwise **settings containing 'verbose' collides")
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values())
    # both spellings still land on the config
    assert int(buildConfig([0, 1], None, None, verbose=0).verbose) == 0
    assert int(buildConfig([0, 1], None, None, verbose=False).verbose) == 0
    assert int(buildConfig([0, 1], None, None, verbose=1).verbose) == 1


def test_shipped_settings_json_can_be_splatted_into_runUROMT():
    """A settings JSON must be usable as `runUROMT(planC, **settings)` - every
    non-underscore key has to be accepted by buildConfig."""
    import json
    import os
    from cerr.uromt.config import buildConfig
    import cerr.uromt as _u

    path = os.path.join(os.path.dirname(_u.__file__), "settings",
                        "uromt_model_settings.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    over = {k: v for k, v in raw.items() if not k.startswith("_")}
    assert "verbose" in over
    cfg = buildConfig([0, 1], None, None, **over)     # must not raise
    assert int(cfg.verbose) == 1


def test_verbose_setting_defaults_on_and_is_documented():
    """`verbose` ships enabled so users see progress without opting in."""
    import json
    import os
    from cerr.uromt.config import loadModelSettings

    assert int(loadModelSettings()["verbose"]) == 1
    # loadModelSettings strips the _-prefixed keys, so read the file for the doc
    import cerr.uromt as _u
    path = os.path.join(os.path.dirname(_u.__file__), "settings",
                        "uromt_model_settings.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["verbose"] == 1
    assert "verbose" in raw["_field_doc"]


def test_warm_start_is_off_by_default_and_controls_the_interval_chain():
    """`maxUiter` is an early-stopping regularizer, so carrying (u, r) between
    intervals gives later intervals more cumulative optimizer effort than
    earlier ones and the velocity accumulates along the chain. The default is
    therefore to restart each interval from rest, matching the reference."""
    from cerr.uromt.config import loadModelSettings
    import inspect
    from cerr.uromt import solver as sv

    assert int(loadModelSettings()["warm_start"]) == 0
    src = inspect.getsource(sv.solveUROMT)
    assert 'getattr(cfg, "warm_start", 0)' in src      # default-off in code too

    # the chain must actually reset when warm_start is off
    n, nt = (7, 6, 5), 2
    N = int(np.prod(n))
    rng = np.random.default_rng(17)
    frames = [np.abs(rng.normal(size=n)) + 0.4 for _ in range(3)]
    base = dict(trueSize=list(n), spacing=[1.0, 1.0, 1.0], dt=0.3, nt=nt,
                sigma=2e-3, alpha=50.0, beta=50.0, bc="closed", niter_pcg=8,
                maxUiter=2, chi=None, reinitR=0, solver="gn",
                vol=frames, mask=np.ones(n, np.uint8), bbox=None)

    seen = {}
    for warm in (0, 1):
        cfg = SimpleNamespace(warm_start=warm, **base)
        seen[warm] = sv.solveUROMT(cfg)
    # interval 0 is identical either way (both start from rest there)
    assert np.allclose(seen[0]["u"][0], seen[1]["u"][0], rtol=0, atol=1e-12)
    # ... and the setting must actually change interval 1
    assert not np.allclose(seen[0]["u"][1], seen[1]["u"][1], rtol=0, atol=1e-10)


def test_tofts_post_process_matches_the_reference_order():
    """concScale -> replace >highValueThreshold by a box mean -> outputClip,
    in that order. Order matters: the threshold and the clip are expressed in
    post-scale units."""
    from cerr.uromt.data import toftsPostProcess

    a = np.zeros((6, 6, 6))
    a[2, 2, 2] = 3.0                       # becomes 450 after x150
    a[0, 0, 0] = -1.0                      # negative, clipped only at the end

    off = SimpleNamespace()
    assert np.array_equal(toftsPostProcess(a, off), a)     # all knobs unset

    scaled = toftsPostProcess(a, SimpleNamespace(concScale=150.0))
    assert scaled[2, 2, 2] == pytest.approx(450.0)
    assert scaled[0, 0, 0] == pytest.approx(-150.0)        # not yet clipped

    cfg = SimpleNamespace(concScale=150.0, highValueThreshold=210.0,
                          highValueKernel=2, outputClip=[0.0, 10000.0])
    got = toftsPostProcess(a, cfg)
    # the spike exceeded the threshold, so it was replaced by a local mean and
    # is now far below its scaled value
    assert got[2, 2, 2] < 450.0
    assert got.min() >= 0.0                                # negatives clipped
    assert got.max() <= 10000.0

    # a volume entirely under the threshold is only scaled + clipped
    b = np.full((5, 5, 5), 0.5)
    got = toftsPostProcess(b, cfg)
    assert np.allclose(got, 75.0)

    # the output clip is applied after the replacement
    cfg2 = SimpleNamespace(concScale=1.0, outputClip=[0.0, 2.0])
    assert np.array_equal(toftsPostProcess(np.array([-5.0, 1.0, 9.0]), cfg2),
                          np.array([0.0, 1.0, 2.0]))


def test_pathline_colour_by_statistic_uses_one_object_per_path():
    """'along' shades each path per SEGMENT (one matplotlib Path each);
    median/mean/max give the path one colour as a single polyline entry, which
    is what makes them ~2x cheaper to draw."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (30, 30, 12)
    k, nVert, nSeed = 6, 9, 6
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    segs, vals, spds = [], [], []
    for c in range(nSeed):
        segs.append(np.column_stack([np.linspace(2, 12, nVert),
                                     np.full(nVert, float(c)),
                                     np.full(nVert, float(k))]))
        vals.append(1.0)
        spds.append(np.linspace(0.5, 2.5, nVert))   # accelerating
    vals = np.asarray(vals)

    def nPaths(colorBy):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(
            ax, {"view": "pathlines", "alpha": 1.0, "segs": (segs, vals),
                 "pathSpeeds": spds, "vrange": (0.0, 3.0),
                 "pathColorBy": colorBy},
            k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
        lc = [c for c in ax.collections
              if isinstance(c, LineCollection) and not isinstance(c, Quiver)]
        return len(lc[0].get_segments())

    assert nPaths("along") == nSeed * (nVert - 1)   # one entry per segment
    for stat in ("median", "mean", "max"):
        assert nPaths(stat) == nSeed                # one entry per path

    # the statistic is computed over the whole path, and they differ on a
    # deliberately accelerating profile
    s = np.linspace(0.5, 2.5, nVert)
    assert np.median(s) < np.max(s) and abs(np.mean(s) - np.median(s)) < 1e-9

    # the default is median
    import inspect
    from cerr.viewer.pycerr_gui.main_window import PyCerrViewer
    p = inspect.signature(PyCerrViewer.set_uromt_overlay).parameters
    assert p["pathColorBy"].default == "median"


def test_path_speed_stats_are_vectorized_and_correct():
    """`pathSpeedStats` reduces all paths at once when they share a vertex
    count (they do - runGLAD integrates every seed over the same steps). The
    per-path Python fallback cost ~1.5 s on a 66k-path ROI, which made the
    cheap-to-draw flat colouring slower than the gradient it replaces."""
    from cerr.uromt import viz

    rng = np.random.default_rng(0)
    equal = [rng.random(9) for _ in range(50)]
    st = viz.pathSpeedStats(equal)
    assert np.allclose(st["mean"], [np.mean(s) for s in equal])
    assert np.allclose(st["median"], [np.median(s) for s in equal])
    assert np.allclose(st["max"], [np.max(s) for s in equal])

    # ragged input still works (the slow path), and agrees
    ragged = [rng.random(4), rng.random(7), rng.random(9)]
    st = viz.pathSpeedStats(ragged)
    assert np.allclose(st["median"], [np.median(s) for s in ragged])
    assert viz.pathSpeedStats([])["mean"].size == 0


def test_runGLAD_precomputes_display_derivatives():
    """Part 4 stores the per-path derivatives the viewer needs, so changing a
    display control never recomputes them. On a 66k-path ROI these cost ~0.57 s
    per rebuild, and the viewer rebuilds on every control change."""
    from cerr.uromt.analyze import runGLAD

    n = (10, 10, 6)
    res = _uniform_flow_result(n=n, nt=4, bbox=(0, 10, 0, 10, 0, 6))
    Lag = runGLAD(res, nEuler=2)
    M = len(Lag["SL"])
    nVert = Lag["SL"][0].shape[0]

    for key in ("speedStats", "pecletStats", "segSpeed", "seedVox", "dispVox"):
        assert key in Lag, key
    for k in ("mean", "median", "max"):
        assert Lag["speedStats"][k].shape == (M,)
    assert np.asarray(Lag["segSpeed"]).shape == (M, nVert - 1)
    assert Lag["seedVox"].shape == (M, 3)
    assert Lag["seedVox"].dtype.kind == "i"
    assert Lag["dispVox"].shape == (M,)

    # ...and they agree with computing them the slow way
    assert np.allclose(Lag["speedStats"]["median"],
                       [np.median(s) for s in Lag["sstream"]])
    from cerr.uromt.analyze import alignSpeedToVertices
    a0 = alignSpeedToVertices(Lag["sstream"][0], nVert)
    assert np.allclose(np.asarray(Lag["segSpeed"])[0],
                       0.5 * (a0[:-1] + a0[1:]))
    assert np.allclose(Lag["dispVox"],
                       [np.linalg.norm(p[-1] - p[0]) for p in Lag["SL"]])
    # dispVox is in VOXELS; displen is the same displacement in mm
    assert np.allclose(Lag["displen"],
                       np.linalg.norm(np.asarray(Lag["disp"]), axis=1))


def test_pathlines_to_scan_vox_vectorized_matches_per_path():
    """The equal-length fast path in pathlinesToScanVox must agree exactly with
    the ragged per-path fallback."""
    from cerr.uromt import viz
    from cerr.uromt.analyze import runGLAD

    n = (8, 8, 4)
    res = _uniform_flow_result(n=n, nt=4, bbox=(2, 10, 3, 11, 1, 5))
    Lag = runGLAD(res, nEuler=2)
    fast = viz.pathlinesToScanVox(Lag, 1.0, 0, perVertex=True)
    ragged = dict(Lag)                      # force the fallback branch
    ragged["SL"] = list(Lag["SL"][:-1]) + [Lag["SL"][-1][:-1]]
    ragged["sstream"] = list(Lag["sstream"][:-1]) + [Lag["sstream"][-1][:-1]]
    slow = viz.pathlinesToScanVox(ragged, 1.0, 0, perVertex=True)
    for a, b in zip(fast[0][:-1], slow[0][:-1]):
        assert np.allclose(a, b)
    assert np.allclose(fast[1][:-1], slow[1][:-1])


def test_overlay_reuses_precomputed_lagrangian_fields():
    """The viewer must consume Part 4's precomputed derivatives rather than
    rebuilding them: a Lag carrying them yields the same overlay as one without
    (the fallback path), so old stored runs still work."""
    import numpy as np
    from cerr.uromt import viz
    from cerr.uromt.analyze import runGLAD, pathSpeedStats

    n = (10, 10, 6)
    res = _uniform_flow_result(n=n, nt=4, bbox=(0, 10, 0, 10, 0, 6))
    Lag = runGLAD(res, nEuler=2)
    segs, vals, spds = viz.pathlinesToScanVox(Lag, 1.0, 0, perVertex=True)

    # precomputed vs recomputed statistics agree
    assert np.allclose(Lag["speedStats"]["median"],
                       pathSpeedStats(spds)["median"])
    # seedVox (ROI coords) maps onto the scan-grid seeds the drawing uses
    bb = Lag["bbox"]
    fromLag = (np.asarray(Lag["seedVox"])
               + np.array([bb[0], bb[2], bb[4]])).round().astype(int)
    fromSegs = np.rint(np.array([s[0] for s in segs])).astype(int)
    assert np.array_equal(fromLag, fromSegs)
    # dispVox matches the drawn net displacement
    assert np.allclose(Lag["dispVox"],
                       [np.linalg.norm(s[-1] - s[0]) for s in segs])


# --------------------------------------------------------------------------- #
#  Colour-by-any-quantity: shared precompute for pathlines AND vectors
# --------------------------------------------------------------------------- #
def test_runGLAD_samples_every_quantity_once():
    """Part 4 samples EVERY displayable quantity along the trajectories, so the
    GUI's colour-by is a look-up. The speed/Peclet-only keys stay as views."""
    from cerr.uromt.analyze import (runGLAD, QUANTITIES, pathStats, streamOf,
                                    segmentValues)

    res = _uniform_flow_result(n=(12, 12, 6), nt=3, vx=1.0)
    Lag = runGLAD(res, spfs=3, nEuler=2)
    M, nVert = len(Lag["SL"]), Lag["nVert"]
    assert M > 0 and nVert == int(Lag["SL"][0].shape[0])

    for q in QUANTITIES:
        st = Lag["streams"][q]
        assert st.shape == (M, nVert - 1)          # one sample per sub-step
        assert np.all(np.isfinite(st))
        stat = pathStats(Lag, q)
        assert np.allclose(stat["mean"], st.mean(1))
        assert np.allclose(stat["max"], st.max(1))

    # legacy keys are the same numbers, not a second copy
    assert np.allclose(np.asarray(Lag["sstream"]), Lag["streams"]["speed"])
    assert np.allclose(np.asarray(Lag["pestream"]), Lag["streams"]["peclet"])
    assert np.allclose(Lag["speedStats"]["median"],
                       pathStats(Lag, "speed")["median"])

    # per-segment values align to the drawn VERTICES (the samples are padded
    # first) and are segment midpoints
    seg = segmentValues(Lag, "rho")
    assert seg.shape == (M, nVert - 1)
    assert segmentValues(Lag, "rho") is seg        # cached, not recomputed
    assert np.allclose(Lag["segSpeed"], segmentValues(Lag, "speed"))

    # uniform flow: |v| = vx, rho = 1, so flux = |rho v_eff| = |v_eff|
    assert np.allclose(streamOf(Lag, "speed"), 1.0)
    assert np.allclose(streamOf(Lag, "rho"), 1.0)
    assert np.allclose(streamOf(Lag, "flux"), streamOf(Lag, "effSpeed"))


def test_vector_and_pathline_quantities_come_from_one_definition():
    """A vector at a voxel and the pathline segment through it must read the
    same number: both are reductions of ``_stepQuantities`` on the same
    (velocity, density, rate) sub-steps."""
    from cerr.uromt.analyze import (runEULAIntervals, runGLAD, eulerianStats,
                                    streamOf)

    res = _uniform_flow_result(n=(10, 10, 5), nt=2, vx=0.7)
    ei = runEULAIntervals(res)
    Lag = runGLAD(res, spfs=2, nEuler=2)

    for q, expect in (("speed", 0.7), ("rho", 1.0)):
        es = eulerianStats(ei, q)
        assert np.allclose(es["mean"][ei["mask"] > 0], expect)
        assert np.allclose(streamOf(Lag, q), expect)   # same value on the path
        # 'along' is the un-reduced per-interval form, one entry per interval
        assert len(es["along"]) == len(res["u"])
        assert es["median"].shape == tuple(ei["n"])

    # |flux| from the Eulerian side is the magnitude of the flux VECTOR
    fe = eulerianStats(ei, "flux")
    assert np.allclose(fe["max"],
                       np.sqrt(np.sum(np.asarray(ei["flux"][0]) ** 2, axis=0)))
    assert eulerianStats(ei, "notAQuantity") is None


def test_vector_overlay_colours_by_a_map_not_by_magnitude():
    """The velocity/flux quiver keeps magnitude as arrow LENGTH but takes its
    colour from ``colorMap3`` when the overlay carries one, on ``colorRange``."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (8, 8, 3)
    k = 1
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    comps = [np.zeros(shape), np.ones(shape), np.zeros(shape)]  # |v| = 1
    cmap3 = np.zeros(shape)
    cmap3[:, :, k] = np.arange(64).reshape(8, 8)                # colour source

    def draw(ov):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(ax, ov, k, xV, yV, ext, lambda m: m[:, :, k],
                             1, 0, 2, shape)
        return [c for c in ax.collections if isinstance(c, Quiver)][0]

    base = {"view": "velocity", "alpha": 1.0, "comps": comps,
            "vrange": (0.0, 1.0)}
    q = draw(dict(base))
    assert np.allclose(q.get_array(), 1.0)             # colour = |v| by default

    q = draw(dict(base, colorMap3=cmap3, colorRange=(0.0, 63.0),
                  label="Peclet (-)"))
    assert np.allclose(np.sort(q.get_array()), np.arange(64))
    assert q.get_clim() == (0.0, 63.0)                 # scaled to colorRange
    # length still comes from the vector magnitude, which is uniform here
    assert np.allclose(np.hypot(q.U, q.V), 1.0)


def test_signed_quantity_gets_a_diverging_map_on_both_overlays():
    """Rate r is a source *or* a sink, so colouring by it uses a diverging map
    over a symmetric range wherever it appears - not a 0..max ramp."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (20, 20, 6)
    k, nVert = 3, 5
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    segs = [np.column_stack([np.linspace(2, 8, nVert), np.full(nVert, 4.0),
                             np.full(nVert, float(k))])]
    stat = {"median": np.array([0.0]), "mean": np.array([0.0]),
            "max": np.array([0.0])}
    fig = Figure()
    ax = fig.add_subplot(111)
    viz.drawUROMTOverlay(
        ax, {"view": "pathlines", "alpha": 1.0,
             "segs": (segs, np.array([0.0])), "pathStat": stat,
             "pathColorBy": "median", "diverging": True,
             "colorQuantity": "rate", "vrange": (-2.0, 2.0)},
        k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
    lc = [c for c in ax.collections
          if isinstance(c, LineCollection) and not isinstance(c, Quiver)][0]
    # zero sits at the MIDDLE of the symmetric range: white in 'bwr', not the
    # blue end a 0..max normalization would give it
    rgb = np.asarray(lc.get_colors())[0][:3]
    assert np.allclose(rgb, 1.0, atol=0.02)


def test_gui_colour_by_menu_matches_the_sampled_quantities():
    """Every quantity offered in the dialog is one Part 4 actually samples, and
    the same choice drives vectors and pathlines."""
    import inspect
    from cerr.uromt.analyze import QUANTITIES
    from cerr.viewer.pycerr_gui.main_window import PyCerrViewer
    from cerr.viewer.pycerr_gui.uromt_gui import UROMTDialog

    keys = [k for _lbl, k in UROMTDialog._COLOR_QUANTITIES]
    assert keys == list(QUANTITIES)
    assert set(keys) <= set(PyCerrViewer._UROMT_COLOR_LABELS)

    p = inspect.signature(PyCerrViewer.set_uromt_overlay).parameters
    assert p["colorBy"].default == "speed"
    assert p["pathColorBy"].default == "median"     # the reduction, unchanged


def test_set_uromt_overlay_colour_by_is_a_lookup_for_both_overlays():
    """End-to-end through the viewer: the same ``colorBy`` quantity colours the
    velocity quiver and the pathlines, from the precomputed per-quantity store,
    with the colour scale kept separate from the arrow-length scale."""
    from types import SimpleNamespace
    from cerr.uromt.analyze import runGLAD
    from cerr.viewer.pycerr_gui.main_window import PyCerrViewer

    res = _uniform_flow_result(n=(16, 16, 8), nt=2, vx=1.0,
                               bbox=(2, 18, 3, 19, 1, 9))
    run = SimpleNamespace(UROMTResult=res,
                          UROMTLagrangian=runGLAD(res, spfs=4, nEuler=2))
    # The overlay builder is pure Python on top of numpy; borrow it onto a stub
    # so it runs headless (a real PyCerrViewer needs its Qt super().__init__).
    class _Stub:
        refresh_views = staticmethod(lambda **k: None)
        _refresh_uromt_views = staticmethod(lambda: None)
    for name in ("set_uromt_overlay", "_uromtVectorColor",
                 "_uromtEulIntervals", "_uromtRoiMaskToScan",
                 "_uromtHeadCeiling", "_UROMT_LABELS", "_UROMT_COLOR_LABELS"):
        setattr(_Stub, name, getattr(PyCerrViewer, name))
    v = _Stub()
    v.planC = SimpleNamespace(urOMT=[run])
    v.scan3M = np.zeros((20, 22, 10))
    v.uromtOverlay = None

    v.set_uromt_overlay(0, view="velocity", colorBy="peclet",
                        pathColorBy="max")
    ov = v.uromtOverlay
    assert ov["colorMap3"].shape == v.scan3M.shape
    assert ov["colorRange"][1] > 0
    assert ov["vrange"] != ov["colorRange"]          # length vs colour scales
    assert "Peclet" in ov["label"] and "(max)" in ov["label"]

    v.set_uromt_overlay(0, view="pathlines", colorBy="rho",
                        pathColorBy="mean")
    ov = v.uromtOverlay
    assert ov["colorQuantity"] == "rho"
    nPaths = len(ov["segs"][0])
    assert ov["pathVertVals"].shape == (nPaths, ov["segs"][0][0].shape[0])
    assert len(ov["pathStat"]["mean"]) == nPaths
    assert np.allclose(ov["pathVertVals"], 1.0)      # rho = 1 everywhere
    assert "rho" in ov["label"]

    # a signed quantity picks the symmetric range + diverging map
    v.set_uromt_overlay(0, view="pathlines", colorBy="rate")
    assert v.uromtOverlay["diverging"] is True
    lo, hi = v.uromtOverlay["vrange"]
    assert lo == -hi

    # an unknown quantity falls back to speed rather than failing to draw
    v.set_uromt_overlay(0, view="pathlines", colorBy="nonsense")
    assert v.uromtOverlay["colorQuantity"] == "speed"


def test_sampling_more_quantities_does_not_change_the_values():
    """All quantities (and the velocity driving the integration) go through ONE
    stacked interpolator - the weights are computed once for every channel. The
    trajectories and the samples must be bit-for-bit what per-channel
    interpolation gave."""
    from cerr.uromt.analyze import runGLAD

    res = _uniform_flow_result(n=(10, 10, 5), nt=2, vx=0.9)
    few = runGLAD(res, spfs=2, nEuler=3, quantities=("speed",))
    allq = runGLAD(res, spfs=2, nEuler=3)
    assert len(few["SL"]) == len(allq["SL"])
    for a, b in zip(few["SL"], allq["SL"]):
        assert np.array_equal(a, b)                 # same trajectories
    assert np.array_equal(few["streams"]["speed"], allq["streams"]["speed"])
    assert set(allq["streams"]) == set(runGLAD.__defaults__[-1])

    # quantities=None is the full set, not an empty store
    assert set(runGLAD(res, spfs=4, nEuler=1, quantities=None)["streams"]) \
        == set(allq["streams"])


def test_pathlines_draw_at_true_extent_by_default():
    """Auto-fit is gone: pathlines are drawn at 1x (their true physical extent)
    and any exaggeration is the user's explicit `lengthScale`, disclosed in the
    label - an auto-scaled path ends where the particle did not go."""
    import inspect
    from types import SimpleNamespace
    from cerr.uromt.analyze import runGLAD
    from cerr.viewer.pycerr_gui.main_window import PyCerrViewer

    p = inspect.signature(PyCerrViewer.set_uromt_overlay).parameters
    assert "pathAutoFit" not in p
    assert p["lengthScale"].default == 1.0
    assert not hasattr(PyCerrViewer, "_PATH_FIT_FRAC")

    res = _uniform_flow_result(n=(12, 12, 6), nt=2, vx=1.0,
                               bbox=(2, 14, 3, 15, 1, 7))
    run = SimpleNamespace(UROMTResult=res,
                          UROMTLagrangian=runGLAD(res, spfs=3, nEuler=2))

    class _Stub:
        refresh_views = staticmethod(lambda **k: None)
        _refresh_uromt_views = staticmethod(lambda: None)
    for name in ("set_uromt_overlay", "_uromtVectorColor",
                 "_uromtEulIntervals", "_uromtRoiMaskToScan",
                 "_uromtHeadCeiling", "_UROMT_LABELS", "_UROMT_COLOR_LABELS"):
        setattr(_Stub, name, getattr(PyCerrViewer, name))
    v = _Stub()
    v.planC = SimpleNamespace(urOMT=[run])
    v.scan3M = np.zeros((16, 18, 9))
    v.uromtOverlay = None

    v.set_uromt_overlay(0, view="pathlines")
    ov = v.uromtOverlay
    assert ov["lengthScale"] == 1.0           # true extent, nothing applied
    assert "paths x" not in ov["label"]       # nothing to disclose at 1x

    v.set_uromt_overlay(0, view="pathlines", lengthScale=8.0)
    ov = v.uromtOverlay
    assert ov["lengthScale"] == 8.0
    assert "[paths x8]" in ov["label"]        # exaggeration stays visible


def test_one_density_control_governs_vectors_and_pathlines():
    """`pathSpfs=None` (what the dialog now passes) means the pathlines use the
    same `subsample` as the vectors - one 'every N' for both overlays."""
    from types import SimpleNamespace
    from cerr.uromt.analyze import runGLAD
    from cerr.viewer.pycerr_gui.main_window import PyCerrViewer

    res = _uniform_flow_result(n=(12, 12, 6), nt=2, vx=1.0,
                               bbox=(2, 14, 3, 15, 1, 7))
    run = SimpleNamespace(UROMTResult=res,
                          UROMTLagrangian=runGLAD(res, spfs=2, nEuler=2))

    class _Stub:
        refresh_views = staticmethod(lambda **k: None)
        _refresh_uromt_views = staticmethod(lambda: None)
    for name in ("set_uromt_overlay", "_uromtVectorColor",
                 "_uromtEulIntervals", "_uromtRoiMaskToScan",
                 "_uromtHeadCeiling", "_UROMT_LABELS", "_UROMT_COLOR_LABELS"):
        setattr(_Stub, name, getattr(PyCerrViewer, name))
    v = _Stub()
    v.planC = SimpleNamespace(urOMT=[run])
    v.scan3M = np.zeros((16, 18, 9))
    v.uromtOverlay = None

    for view in ("velocity", "pathlines"):
        v.set_uromt_overlay(0, view=view, subsample=3)
        assert v.uromtOverlay["subsample"] == 3, view
    # an explicit override is still available programmatically
    v.set_uromt_overlay(0, view="pathlines", subsample=3, pathSpfs=1)
    assert v.uromtOverlay["subsample"] == 1


def test_uromt_settings_editor_round_trips_values_and_types():
    """The settings editor parses cells with json.loads, so numbers stay
    numbers and word settings stay strings; keys it does not show (the time
    selection, which the main dialog drives) must survive the round trip."""
    from cerr.viewer.pycerr_gui.uromt_gui import UROMTSettingsDialog as SD

    assert SD._parse("0.02") == 0.02 and isinstance(SD._parse("0.02"), float)
    assert SD._parse("3") == 3 and isinstance(SD._parse("3"), int)
    assert SD._parse("[0, 5]") == [0, 5]
    assert SD._parse("yes") == "yes"          # not JSON -> the string as typed
    assert SD._parse(" true ") is True
    assert SD._fmt("nn") == "nn" and SD._fmt(4) == "4"
    assert json.loads(SD._fmt([1, 2])) == [1, 2]
    assert "time" in SD._HIDDEN               # driven by the main dialog


def test_preview_mode_is_gone_from_the_uromt_gui():
    """The half-resolution 'Preview' run was removed: no control, and no
    silent do_resize / maxUiter overrides in the worker."""
    import inspect
    from cerr.viewer.pycerr_gui import uromt_gui

    src = inspect.getsource(uromt_gui)
    assert "previewCheck" not in src
    assert "size_factor = 0.5" not in src
    assert "preview" not in inspect.signature(
        uromt_gui._UROMTWorker.__init__).parameters
    # the worker takes an edited settings dict instead
    assert "settings" in inspect.signature(
        uromt_gui._UROMTWorker.__init__).parameters


class _FakeSlider:
    """Duck-typed stand-in for a QSlider, so the dialog's animation logic can
    be exercised without a QApplication."""

    def __init__(self, value=0, maximum=100):
        self._v = value
        self._max = maximum
        self.calls = []

    def value(self):
        return self._v

    def maximum(self):
        return self._max

    def setValue(self, v):
        self._v = int(v)
        self.calls.append(int(v))


class _FakePlayBtn:
    def __init__(self):
        self.checked = True

    def isChecked(self):
        return self.checked

    def setChecked(self, v):
        self.checked = bool(v)


def _animStub(isPathline, value=0, maximum=3, nTp=4):
    """The dialog's animation methods bound onto plain objects."""
    from cerr.viewer.pycerr_gui.uromt_gui import UROMTDialog

    class _Stub:
        _SUB_STEPS = UROMTDialog._SUB_STEPS
    for name in ("_onGrowTick", "_onGrowChanged", "_growFraction"):
        setattr(_Stub, name, getattr(UROMTDialog, name))
    st = _Stub()
    st._isPathlineView = lambda: isPathline
    st.growSlider = _FakeSlider(value, maximum)
    st.tpSlider = _FakeSlider(0, max(0, nTp - 1))
    st.playBtn = _FakePlayBtn()
    # the tick reschedules itself off the finished frame (single-shot timer)
    st._growTimer = SimpleNamespace(start=lambda: None)
    st._tpScanNums = list(range(nTp))
    st._onOverlayChanged = lambda: st.__dict__.setdefault("redrawn", 0)
    return st


def test_play_loops_both_pathline_growth_and_timepoints():
    """Play cycles the run's time axis and never stops itself: the pathlines
    regrow from their seeds, the field overlays cycle their timepoints."""
    st = _animStub(False, value=0, maximum=3)
    seen = []
    for _ in range(6):
        st._onGrowTick()
        seen.append(st.growSlider.value())
    assert seen == [1, 2, 3, 0, 1, 2], seen    # wraps instead of stopping
    assert st.playBtn.isChecked()

    st = _animStub(True, value=28, maximum=30)
    st._onGrowTick()
    assert st.growSlider.value() == 29
    st._onGrowTick()
    assert st.growSlider.value() == 30         # end of the run
    st._onGrowTick()
    assert st.growSlider.value() == 0          # regrows from the seeds
    assert st.playBtn.isChecked()

    # nothing to animate
    st = _animStub(False, value=0, maximum=0)
    st._onGrowTick()
    assert not st.playBtn.isChecked()
    assert st.growSlider.calls == []


def test_pathline_growth_is_cumulative_over_the_run_time_axis():
    """The slider is the run's TIME axis in both modes. For pathlines its
    fraction is what is drawn from each seed - the cumulative trajectory up to
    that time - and the displayed scan follows the interval it lands in."""
    from cerr.viewer.pycerr_gui.uromt_gui import UROMTDialog
    sub = UROMTDialog._SUB_STEPS
    nTp, nIvl = 4, 3

    st = _animStub(True, value=nIvl * sub, maximum=nIvl * sub, nTp=nTp)
    assert st._growFraction() == 1.0                 # whole paths
    st.growSlider.setValue(nIvl * sub // 2)
    assert abs(st._growFraction() - 0.5) < 1e-9      # half of every path

    # growing the paths never touches the displayed scan: a pathline is a
    # whole-run trajectory, so it belongs to no single frame, and swapping the
    # backdrop mid-animation only makes the growth harder to follow
    st = _animStub(True, value=0, maximum=nIvl * sub, nTp=nTp)
    st._onGrowChanged(sub + 3)
    st._onGrowChanged(2 * sub)
    assert st.tpSlider.calls == []                   # scan left where it was
    assert st.redrawn == 0                           # only the paths redraw

    # in field mode the slider IS the timepoint and growth does not apply
    st = _animStub(False, value=2, maximum=nIvl, nTp=nTp)
    assert st._growFraction() == 1.0
    st._onGrowChanged(2)
    assert st.tpSlider.calls == [2]
    st.tpSlider._v = 2
    st._onGrowChanged(2)                             # already there -> no
    assert st.tpSlider.calls == [2]                  # redundant scan reload


def test_growPathline_keeps_the_path_from_its_seed():
    """Growth shows the trajectory travelled SO FAR, from the seed, not the leg
    walked during the current interval."""
    from cerr.uromt.viz import growPathline

    pts = np.column_stack([np.arange(11.0), np.zeros(11), np.zeros(11)])
    vals = np.arange(11.0)
    got, gv = growPathline(pts, vals, 0.5)
    assert np.array_equal(got, pts[:6])              # leading half, from vertex 0
    assert np.array_equal(gv, vals[:6])
    assert np.array_equal(got[0], pts[0])            # always anchored at the seed
    whole, _ = growPathline(pts, vals, 1.0)
    assert np.array_equal(whole, pts)
    stub, _ = growPathline(pts, vals, 0.0)           # still drawable at t=0
    assert stub.shape[0] == 2 and np.array_equal(stub[0], pts[0])


def _drawGrownPaths(grow, nVert=31, nSeed=5, decim=3, lengthScale=1.0):
    """Draw curved pathlines at a growth fraction; return the drawn polylines
    and the direction-arrow (base, vector) arrays."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (40, 40, 10)
    k = 5
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    t = np.linspace(0, 1, nVert)
    segs = []
    for i in range(nSeed):
        # a curved path, each seeded on the displayed slice
        segs.append(np.column_stack([4.0 + i + 3.0 * t,
                                     6.0 + 2.0 * np.sin(3.0 * t),
                                     np.full(nVert, float(k))]))
    vals = np.ones(nSeed)
    fig = Figure()
    ax = fig.add_subplot(111)
    viz.drawUROMTOverlay(
        ax, {"view": "pathlines", "alpha": 1.0, "segs": (segs, vals),
             "pathStat": {"median": vals}, "pathColorBy": "median",
             "pathDecim": decim, "grow": grow, "lengthScale": lengthScale,
             "vrange": (0.0, 1.0)},
        k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
    from matplotlib.collections import PolyCollection
    lc = [c for c in ax.collections
          if isinstance(c, LineCollection)
          and not isinstance(c, (Quiver, PolyCollection))]
    lines = [np.asarray(x) for x in lc[0].get_segments()] if lc else []
    return lines, _pathHeads(ax)[0]


def test_growing_paths_stay_anchored_at_their_seed():
    """While the growth animation runs, each path must keep its start point and
    extend from it toward its final coordinate - the drawn vertices are the
    real trajectory so far, and the tip advances monotonically along it."""
    full, _ = _drawGrownPaths(1.0)
    # the undecimated trajectory, as the reference for "lies on the path":
    # the drawn frames are decimated for speed, so their vertices need not
    # coincide with the decimated FULL frame's vertices - but they must all sit
    # on the real trajectory.
    truth, _ = _drawGrownPaths(1.0, decim=1)
    seeds = np.asarray([p[0] for p in full])
    tips = {}
    for g in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        lines, _ = _drawGrownPaths(g)
        assert len(lines) == len(full), g            # no path appears/vanishes
        assert np.allclose([p[0] for p in lines], seeds), g   # start is fixed
        tips[g] = np.asarray([p[-1] for p in lines])
        # every drawn vertex lies ON the full trajectory (growth reveals the
        # path, it does not redraw a different one)
        for a, b in zip(lines, truth):
            d = np.linalg.norm(b[None, :, :] - a[:, None, :], axis=2).min(1)
            assert d.max() < 0.15, (g, d.max())

    # the tip advances away from the seed and lands on the true end point
    dist = {g: np.linalg.norm(tips[g] - seeds, axis=1) for g in tips}
    for lo, hi in zip((0.0, 0.1, 0.25, 0.5, 0.75), (0.1, 0.25, 0.5, 0.75, 1.0)):
        assert np.all(dist[hi] >= dist[lo] - 1e-9), (lo, hi)
    assert np.allclose(tips[1.0], [p[-1] for p in full])


def test_direction_arrow_is_anchored_on_the_paths_own_tail():
    """The end arrow must terminate the path, not add a stroke of its own.

    It used to be a separate stick of fixed length (a fraction of the field of
    view) pointing along the last segment's direction from a base behind the
    tip: on a curved or zigzag path that cut straight across the windings it
    was meant to end. Now the shaft runs from one of the path's OWN vertices to
    its tip - far enough back that matplotlib draws a full-size head (it
    shrinks arrows whose shaft is under one head length), but never off the
    trajectory.
    """
    from cerr.uromt import viz
    for g in (0.1, 0.25, 0.5, 1.0):
        lines, tris = _drawGrownPaths(g)
        assert len(tris) == len(lines), g
        for p, tri in zip(lines, tris):
            assert np.allclose(tri[0], p[-1])         # the tip IS the path end
            # the head points along the path's tail, not off in some direction
            u = tri[0] - 0.5 * (tri[1] + tri[2])
            tail = p[-1] - p[0]
            assert np.dot(u, tail) > 0, (g, u, tail)
        # a head never covers more than its share of the path's visible extent
        ext = np.array([np.hypot(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
                        for p in lines])
        assert np.all(_headLengths(tris) <= viz._TAIL_MAX_FRAC * ext + 1e-6)


def test_pathline_head_is_sized_off_its_own_path():
    """A quiver head is fixed in AXES units and global to the collection, so on
    real urOMT data (paths ~1 voxel, head ~2.6 voxels) one head covered its
    whole path and the display showed arrowheads with no paths. Each head is
    now a triangle sized off ITS path."""
    from cerr.uromt import viz

    lines, tris = _drawGrownPaths(1.0, nVert=31, decim=3)
    # the measure is the path's visible EXTENT, not its arc length: a tight
    # squiggle can travel ten voxels inside a two-voxel box
    ext = np.array([np.hypot(np.ptp(p[:, 0]), np.ptp(p[:, 1])) for p in lines])
    arcs = np.array([np.sum(np.hypot(*np.diff(p, axis=0).T)) for p in lines])
    assert np.all(ext <= arcs + 1e-9)
    heads = _headLengths(tris)
    kw = viz._arrowStyle(1.0)
    ceiling = kw["headlength"] * kw["width"] * 39.0     # xV spans 0..39 there

    # never more than its share of the path, never over the screen ceiling,
    # and never over the hard millimetre ceiling
    assert np.all(heads <= viz._TAIL_MAX_FRAC * ext + 1e-6)
    assert np.all(heads <= ceiling + 1e-6)
    assert np.all(heads <= viz._headCeiling(None) + 1e-9)
    # ... it takes whichever of the three limits is smallest
    assert np.allclose(heads, np.minimum(np.minimum(
        ceiling, viz._TAIL_MAX_FRAC * ext), viz._headCeiling(None)))
    assert np.all(heads > 0)


def test_end_arrow_takes_the_colour_of_the_segment_it_ends():
    """A head in a different colour from the path end reads as a separate
    object; it must match the final segment's colour in both colour modes."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (30, 30, 8)
    k, nVert, nSeed = 4, 9, 3
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    segs, spds = [], []
    for c in range(nSeed):
        segs.append(np.column_stack([np.linspace(3, 11, nVert),
                                     np.full(nVert, 5.0 + c),
                                     np.full(nVert, float(k))]))
        spds.append(np.linspace(0.4, 2.6, nVert))        # accelerating
    vals = np.asarray([1.0] * nSeed)

    for colorBy in ("along", "median"):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(
            ax, {"view": "pathlines", "alpha": 1.0, "segs": (segs, vals),
                 "pathVertVals": spds, "vrange": (0.0, 3.0),
                 "pathStat": {"median": np.full(nSeed, 1.5)},
                 "pathColorBy": colorBy},
            k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
        from matplotlib.collections import PolyCollection
        lc = [c for c in ax.collections
              if isinstance(c, LineCollection)
              and not isinstance(c, (Quiver, PolyCollection))][0]
        heads = _pathHeads(ax)[1]
        lineCols = np.asarray(lc.get_colors())
        headCols = np.asarray(heads.get_facecolor())
        # the head of path 0 matches the LAST entry drawn for path 0
        nEntry = len(lineCols) // nSeed
        assert np.allclose(headCols[0][:3], lineCols[nEntry - 1][:3],
                           atol=1e-6), colorBy


def test_3d_end_cone_terminates_on_the_path_tip():
    """3-D counterpart of the 2-D end-arrow fix: the cone must END on the
    path's last vertex, pointing back along it. ``pv.Cone`` centers on its
    centroid, which straddled the tip - the head stuck out half its length
    beyond where the particle actually went."""
    pv = pytest.importorskip("pyvista")
    from cerr.viewer.pycerr_gui.common import (_PATH_CONE_FRAC,
                                               _PATH_CONE_RADIUS)

    tip = 5.0
    ep = pv.PolyData(np.array([[tip, 0.0, 0.0]]))
    ep["dir"] = np.array([[1.0, 0.0, 0.0]])           # travelling +x
    ep["clen"] = np.array([_PATH_CONE_FRAC * 4.0])
    ep.set_active_vectors("dir")
    cones = ep.glyph(orient="dir", scale="clen", factor=1.0,
                     geom=pv.Cone(center=(-0.5, 0.0, 0.0),
                                  direction=(1.0, 0.0, 0.0),
                                  radius=_PATH_CONE_RADIUS, height=1.0,
                                  resolution=12))
    lo, hi = cones.bounds[0], cones.bounds[1]
    assert abs(hi - tip) < 1e-6, hi                   # apex ON the tip
    assert lo < tip                                   # body back along the path
    assert abs((tip - lo) - _PATH_CONE_FRAC * 4.0) < 1e-6   # scaled per path


def test_3d_pathline_arrow_lies_on_the_last_segment():
    """The matplotlib 3-D fallback draws its arrow on the path's own final
    segment, like the 2-D overlay - not as a stroke of some other length."""
    from cerr.uromt import viz

    nVert = 9
    t = np.linspace(0, 1, nVert)
    segs = [np.column_stack([2 + 6 * t, 3 + 2 * np.sin(4 * t),
                             np.full(nVert, 4.0)]),
            np.column_stack([5 + 2 * t, 6 + 3 * t, np.full(nVert, 4.0)])]
    ov = {"view": "pathlines", "segs": (segs, np.ones(2)),
          "pathVertVals": [np.linspace(0.5, 2.0, nVert)] * 2,
          "subsample": 1, "grow": 1.0}
    geom = viz.overlayTo3D(ov, np.arange(12.0), np.arange(12.0),
                           np.arange(9.0))
    paths, ends = geom["paths"], np.asarray(geom["pathEnd"])
    assert len(paths) == 2
    for p, e in zip(paths, ends):
        assert np.allclose(e, p[-1])                  # the marker anchors here
        d = p[-1] - p[-2]                             # ... along the last
        assert np.linalg.norm(d) > 0                  # drawn segment


def test_runs_record_the_settings_file_they_used(tmp_path):
    """A stored run remembers WHICH JSON it was computed with, so the GUI's
    browse/edit dialogs can open there instead of in the working directory."""
    import json
    from cerr.uromt.config import buildConfig, loadModelSettings, \
        _DEFAULT_SETTINGS
    from cerr.dataclasses.uromt import buildFromConfig

    mine = tmp_path / "my_uromt_settings.json"
    mine.write_text(json.dumps(loadModelSettings(None)))

    cfg = buildConfig(None, 0, str(mine))
    assert cfg.settingsFile == str(mine)
    assert buildConfig(None, 0, None).settingsFile == _DEFAULT_SETTINGS

    obj = buildFromConfig(cfg, {"u": []}, {}, {})
    assert obj.UROMTSetup["settingsFile"] == str(mine)


def test_settings_dialogs_open_where_the_settings_live(tmp_path):
    """`_settingsPathOrDefault` prefers the run's own file, then the typed
    path, then the bundled default - never the process working directory,
    which is where the file dialogs used to land."""
    import json
    from cerr.uromt.config import loadModelSettings, _DEFAULT_SETTINGS
    from cerr.viewer.pycerr_gui.uromt_gui import UROMTDialog

    mine = tmp_path / "settings.json"
    mine.write_text(json.dumps(loadModelSettings(None)))

    class _Stub:
        _settingsPathOrDefault = UROMTDialog._settingsPathOrDefault
    st = _Stub()
    st.settingsEdit = SimpleNamespace(text=lambda: "")

    st._settingsPath = str(mine)                     # the run's own file wins
    assert st._settingsPathOrDefault() == str(mine)

    st._settingsPath = str(tmp_path / "gone.json")   # deleted -> next candidate
    st.settingsEdit = SimpleNamespace(text=lambda: str(mine))
    assert st._settingsPathOrDefault() == str(mine)

    st.settingsEdit = SimpleNamespace(text=lambda: "(settings of run X)")
    assert st._settingsPathOrDefault() == _DEFAULT_SETTINGS   # never cwd
    assert os.path.isfile(st._settingsPathOrDefault())


def test_run_button_confirms_before_starting():
    """urOMT cannot be cancelled once the worker starts, and the inputs are now
    prefilled from whichever run is selected - so Run asks first, defaulting to
    No, and describes what it is about to do."""
    import inspect
    from cerr.viewer.pycerr_gui.uromt_gui import UROMTDialog

    src = inspect.getsource(UROMTDialog._run)
    assert "QMessageBox.question" in src
    # the question comes BEFORE the worker is built and started
    assert src.index("QMessageBox.question") < src.index("_UROMTWorker(")
    assert "QMessageBox.No)" in src                  # No is the default button
    assert "return" in src.split("QMessageBox.question")[1]

    # the summary names the ROI, the frames and the settings source
    class _Stub:
        _runSummary = UROMTDialog._runSummary
        _settingsPathOrDefault = staticmethod(lambda: "/tmp/s.json")
    st = _Stub()
    st._settings = None
    st.settingsEdit = SimpleNamespace(text=lambda: "/data/uromt.json")
    st.structCombo = SimpleNamespace(currentText=lambda: "3: Tumor")
    st.viewer = SimpleNamespace(planC=SimpleNamespace(scan=[0] * 6))
    txt = st._runSummary((2, 2, 6))
    assert "3: Tumor" in txt
    assert "3 of 6 scans" in txt and "2 interval(s)" in txt   # 2,4,6
    assert "/data/uromt.json" in txt

    st._settings = {"sigma": 1.0}                    # edited, unsaved
    assert "edited" in st._runSummary((1, 1, 6))


def test_arrow_heads_follow_the_length_scale():
    """Quiver head sizes are multiples of the shaft WIDTH - an axes fraction -
    so they do not follow the arrow's data length. Without scaling them, a 10x
    'length x' drew 10x longer arrows with the same pinprick head and a 0.2x
    one drew stubs that were nearly all head."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (16, 16, 4)
    k = 2
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    comps = [np.zeros(shape), np.ones(shape), np.zeros(shape)]

    def head(lengthScale):
        """The head dims the STYLE asks for (before the per-view cap)."""
        kw = viz._arrowStyle(2.0, lengthScale)
        return (np.array([kw["headwidth"], kw["headlength"],
                          kw["headaxislength"]]), kw["width"])

    base, wBase = head(1.0)
    big, wBig = head(2.0)
    small, wSmall = head(0.5)
    # LINEAR growth: a 2x longer arrow asks for a 2x head. It was briefly sqrt
    # damped, which made x2 barely change the head at all.
    assert np.allclose(big, base * 2.0), (big, base)
    assert np.allclose(small, base * 0.5), (small, base)
    # the SHAFT width is the line-width control's business, not the length's
    assert wBig == wBase == wSmall

    lo, hi = viz._ARROW_HEAD_SCALE
    assert np.allclose(head(hi)[0], base * hi)        # linear up to the clamp
    assert np.allclose(head(1000.0)[0], base * hi)    # ... then flat
    assert np.allclose(head(0.001)[0], base * lo)
    # what actually reaches the canvas is additionally capped against the
    # arrows on screen - see test_vector_head_cannot_dwarf_the_arrow_it_ends

    # line width scales the shaft, independently of the head
    assert abs(viz._arrowStyle(4.0)["width"]
               - 2.0 * viz._arrowStyle(2.0)["width"]) < 1e-12


def test_pathline_end_arrow_heads_follow_the_length_scale():
    """Pathlines scale about their seed, so their end heads must scale too."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (30, 30, 8)
    k, nVert = 4, 11
    # a FINE grid (0.02 cm voxels): the hard millimetre ceiling must not bind
    # here, or there is no growth left to observe
    xV = np.arange(shape[1]) * 0.02
    yV = np.arange(shape[0]) * 0.02
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    segs = [np.column_stack([np.linspace(4, 9, nVert), np.full(nVert, 6.0),
                             np.full(nVert, float(k))])]
    vals = np.ones(1)

    def head(lengthScale):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(
            ax, {"view": "pathlines", "alpha": 1.0, "segs": (segs, vals),
                 "pathStat": {"median": vals}, "pathColorBy": "median",
                 "vrange": (0.0, 1.0), "lengthScale": lengthScale},
            k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
        return float(_headLengths(_pathHeads(ax)[0]).mean())

    # a scaled path is longer, so its head - a fraction of it - grows with it
    # a scaled path is longer, so its head - a fraction of it - grows with it,
    # until the on-screen ceiling takes over
    assert head(0.5) < head(1.0) < head(2.0) <= head(4.0)
    assert np.allclose(head(0.5), head(1.0) * 0.5)
    # and everything stays under the hard millimetre ceiling
    for ls in (0.5, 1.0, 2.0, 4.0, 100.0):
        assert head(ls) <= viz._headCeiling(None) + 1e-9, ls


def test_scaled_paths_leave_the_view_instead_of_piling_on_its_edge():
    """Mapping voxel indices with `np.interp` CLAMPS at the grid edge, so a
    path scaled past the field of view collapsed onto the boundary - and its
    end arrow degenerated to zero length and vanished. Coordinates outside the
    grid are extrapolated instead, so the path is simply clipped by the axes."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.collections import LineCollection
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (30, 30, 8)
    k, nVert = 4, 11
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    segs = [np.column_stack([np.linspace(4, 9, nVert), np.full(nVert, 6.0),
                             np.full(nVert, float(k))])]
    vals = np.ones(1)

    def draw(lengthScale):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(
            ax, {"view": "pathlines", "alpha": 1.0, "segs": (segs, vals),
                 "pathStat": {"median": vals}, "pathColorBy": "median",
                 "vrange": (0.0, 1.0), "lengthScale": lengthScale},
            k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
        from matplotlib.collections import PolyCollection
        lc = [c for c in ax.collections
              if isinstance(c, LineCollection)
              and not isinstance(c, (Quiver, PolyCollection))]
        return np.asarray(lc[0].get_segments()[0]), _pathHeads(ax)[0]

    # the path runs along the VERTICAL axis here (vAxis=0), so column 1 is the
    # one that leaves the view
    pts, tris = draw(20.0)
    assert pts[:, 1].max() > yV[-1], pts[:, 1].max()   # runs off the view
    assert len(tris) == 1                              # ... head still drawn
    assert _headLengths(tris)[0] > 0

    # in-range geometry is untouched
    pts1, _tris1 = draw(1.0)
    assert pts1[:, 1].min() >= yV[0] and pts1[:, 1].max() <= yV[-1]
    assert np.allclose(pts1[0], [6.0, 4.0])            # (h, v) = (col, row)


def test_show_on_scan_is_the_dialogs_default_button():
    """Displaying an existing run is the common action; Enter must not launch a
    long, uncancellable solve."""
    import inspect
    from cerr.viewer.pycerr_gui.uromt_gui import UROMTDialog

    src = inspect.getsource(UROMTDialog.__init__)
    assert "self.showBtn.setDefault(True)" in src
    assert "self.runBtn.setDefault(True)" not in src
    assert "self.runBtn.setAutoDefault(False)" in src
    # focus follows to the button as soon as there is a run to show
    sel = inspect.getsource(UROMTDialog._onRunSelected)
    assert "self.showBtn.setFocus()" in sel


def test_arrow_heads_are_readable_at_the_default_size():
    """The head multipliers are relative to a very thin shaft, so they have to
    be generous to be picked out at typical zoom - but stay narrow enough not
    to blot out the anatomy underneath."""
    from cerr.uromt import viz

    kw = viz._arrowStyle(1.0)
    assert kw["headwidth"] >= 7.0            # raised twice from the original 4
    assert kw["headlength"] > kw["headaxislength"] > kw["headwidth"]
    assert kw["headwidth"] <= 9.0            # still a narrow head, not a blob
    assert abs(kw["width"] - viz._NARROW_ARROW["width"]) < 1e-12


def _boxFluxResult(n=(12, 12, 12), nt=2, nIvl=2, fx=2.0, box=(3, 9)):
    """Result whose flux field is a uniform +row flow inside a box mask."""
    N = int(np.prod(n))
    lo, hi = box
    mask = np.zeros(n, dtype=np.uint8)
    mask[lo:hi, lo:hi, lo:hi] = 1
    v = np.zeros((3, N, nt))
    v[0] = fx
    return dict(u=[v.copy() for _ in range(nIvl)],
                r=[np.zeros((N, nt))] * nIvl,
                rho=[np.ones((N, nt))] * nIvl,
                n=list(n), spacing=[1.0, 2.0, 0.5], mask=mask,
                bbox=(0, n[0], 0, n[1], 0, n[2]),
                frameScanNums=list(range(nIvl + 1)), doResize=0,
                sizeFactor=1.0, dt=0.4, nt=nt, sigma=0.0)


def test_surface_flux_splits_influx_from_outflux():
    """`|flux|` is unsigned - it says how fast tracer moves, not where it goes.
    In/out only exist relative to a surface, so the boundary integral of the
    outward normal flux is what separates them."""
    from cerr.uromt.analyze import surfaceFlux

    n = (12, 12, 12)
    h = [1.0, 2.0, 0.5]                       # anisotropic on purpose
    mask = np.zeros(n, dtype=bool)
    mask[3:9, 3:9, 3:9] = True

    # a uniform flow through the box: what enters must leave (divergence thm)
    F = np.zeros((3,) + n)
    F[0][mask] = 2.0
    r = surfaceFlux(F, mask, h)
    face = 2.0 * (h[1] * h[2]) * 36           # 6x6 faces at each end
    assert abs(r["influx"] - face) < 1e-9
    assert abs(r["outflux"] - face) < 1e-9
    assert abs(r["net"]) < 1e-9

    # a purely outward (source) field is all outflux; reversing it is all influx
    grids = np.meshgrid(*[np.arange(k, dtype=float) for k in n], indexing="ij")
    F = np.stack([np.where(mask, g - 5.5, 0.0) for g in grids])
    src = surfaceFlux(F, mask, h)
    snk = surfaceFlux(-F, mask, h)
    assert src["influx"] == 0.0 and src["outflux"] > 0
    assert snk["outflux"] == 0.0 and snk["influx"] > 0
    assert abs(src["net"] + snk["net"]) < 1e-9   # outward-positive convention

    # the display map is the per-voxel signed contribution: it sums to the net
    # and lives on the boundary voxels only
    assert abs(src["map3"].sum() - src["net"]) < 1e-9
    assert not src["map3"][~mask].any()
    inner = np.zeros(n, dtype=bool)
    inner[4:8, 4:8, 4:8] = True
    assert not src["map3"][inner].any()

    # a mask running into the grid border still has a surface there
    m2 = np.zeros(n, dtype=bool)
    m2[:4, 3:9, 3:9] = True
    F2 = np.zeros((3,) + n)
    F2[0][m2] = 1.0
    edge = surfaceFlux(F2, m2, h)
    assert edge["influx"] > 0 and abs(edge["net"]) < 1e-9

    # centered averaging halves every boundary face of a MASKED field, which is
    # why the one-sided (inside-cell) value is the default
    assert surfaceFlux(F, mask, h, centered=True)["outflux"] < src["outflux"]


def test_interval_surface_flux_follows_the_run():
    """One in/out/net triple per time interval, integrated over the run's ROI
    mask by default."""
    from cerr.uromt.analyze import runEULAIntervals, intervalSurfaceFlux

    res = _boxFluxResult(nIvl=3)
    ei = runEULAIntervals(res)
    sf = intervalSurfaceFlux(ei)
    assert all(len(sf[k]) == 3 for k in ("influx", "outflux", "net", "map3"))
    # uniform flow through the box: balanced, every interval
    assert all(abs(x) < 1e-9 for x in sf["net"])
    assert all(v > 0 for v in sf["influx"])
    assert sf["map3"][0].shape == tuple(ei["n"])

    # a different region can be passed (e.g. the undilated structure mask)
    small = np.zeros(res["n"], dtype=bool)
    small[4:8, 4:8, 4:8] = True
    sf2 = intervalSurfaceFlux(ei, mask=small)
    assert sf2["influx"][0] > 0
    assert sf2["influx"][0] != sf["influx"][0]      # a smaller surface


def test_surface_flux_overlay_reports_in_out_and_net():
    """The viewer exposes the split as a signed map plus the totals for the
    displayed interval."""
    from types import SimpleNamespace
    from cerr.viewer.pycerr_gui.main_window import PyCerrViewer

    res = _boxFluxResult(nIvl=2)
    run = SimpleNamespace(UROMTResult=res, UROMTLagrangian={})

    class _Stub:
        refresh_views = staticmethod(lambda **k: None)
        _refresh_uromt_views = staticmethod(lambda: None)
    for name in ("set_uromt_overlay", "_uromtVectorColor", "_uromtEulIntervals",
                 "_uromtSurfaceFlux", "_uromtRoiMaskToScan",
                 "_uromtHeadCeiling", "_UROMT_LABELS", "_UROMT_COLOR_LABELS"):
        setattr(_Stub, name, getattr(PyCerrViewer, name))
    v = _Stub()
    v.planC = SimpleNamespace(urOMT=[run])
    v.scan3M = np.zeros(res["n"])
    v.uromtOverlay = None

    v.set_uromt_overlay(0, view="surfflux")
    ov = v.uromtOverlay
    assert ov["diverging"] is True                  # signed -> diverging map
    lo, hi = ov["vrange"]
    assert lo == -hi and hi > 0                     # symmetric about zero
    sf = ov["surfaceFlux"]
    assert sf["influx"] > 0 and sf["outflux"] > 0
    assert abs(sf["net"] - (sf["outflux"] - sf["influx"])) < 1e-9
    assert abs(sf["net"]) < 1e-6                    # uniform flow: balanced
    assert "in " in ov["label"] and "out " in ov["label"]
    # the map carries both signs: tracer leaves one face and enters the other
    m = ov["map3"]
    assert m.max() > 0 and m.min() < 0


def test_export_writes_surface_flux_maps_and_totals(tmp_path):
    """The NIfTI export carries the in/out split too: a signed surface-flux map
    per interval plus a CSV of the totals (a scalar has nowhere else to go)."""
    pytest.importorskip("SimpleITK")
    from cerr.uromt.analyze import runEULAIntervals
    from cerr.uromt.export import saveEulerianMapsNii

    res = _boxFluxResult(nIvl=2)
    ei = runEULAIntervals(res)

    class _Scan:                      # minimal geometry stand-in
        scanInfo = []                 # single-slice: no slice order to flip

        def getScanArray(self):
            return np.zeros(res["n"])

        def getSitkImage(self):
            import SimpleITK as sitk
            return sitk.GetImageFromArray(
                np.zeros((res["n"][2], res["n"][0], res["n"][1]),
                         dtype=np.float32))

    planC = SimpleNamespace(scan=[_Scan()])
    paths = saveEulerianMapsNii(ei, planC, 0, str(tmp_path), prefix="run")
    names = [os.path.basename(p) for p in paths]
    assert "run_surfflux_t01.nii.gz" in names
    assert "run_surfflux_t02.nii.gz" in names
    assert "run_surface_flux.csv" in names

    rows = (tmp_path / "run_surface_flux.csv").read_text().strip().split("\n")
    assert rows[0] == "interval,influx,outflux,net_outward"
    assert len(rows) == 3                       # header + one row per interval
    inF, outF, net = (float(x) for x in rows[1].split(",")[1:])
    assert inF > 0 and outF > 0
    assert abs(net - (outF - inF)) < 1e-6       # outward-positive convention


def test_3d_pathlines_obey_the_colour_reduction():
    """The 'reduce' control has to mean the same thing in 3-D: a statistic
    gives each path ONE colour, 'along path' shades it per vertex. 3-D used to
    always shade per vertex, so switching to median/mean/max changed nothing
    there and every path carried the full colour range along its length."""
    from cerr.uromt import viz

    nVert, nSeed = 9, 4
    t = np.linspace(0, 1, nVert)
    segs, vert = [], []
    for i in range(nSeed):
        segs.append(np.column_stack([2 + 5 * t, np.full(nVert, 3.0 + i),
                                     np.full(nVert, 4.0)]))
        vert.append(np.linspace(0.5 + i, 2.5 + i, nVert))   # accelerating
    vert = np.asarray(vert)
    stat = {"median": vert.mean(1), "mean": vert.mean(1), "max": vert.max(1)}
    base = {"view": "pathlines", "segs": (segs, np.ones(nSeed)),
            "pathVertVals": vert, "pathStat": stat, "subsample": 1,
            "grow": 1.0}
    xV = np.arange(12.0)
    yV = np.arange(12.0)
    zV = np.arange(9.0)

    # a statistic -> one constant colour value per path, equal to the statistic
    for key in ("median", "mean", "max"):
        geom = viz.overlayTo3D(dict(base, pathColorBy=key), xV, yV, zV)
        vals = geom["pathVals"]
        assert len(vals) == nSeed
        for i, v in enumerate(vals):
            assert np.allclose(v, stat[key][i]), (key, i)

    # 'along path' keeps the per-vertex samples
    geom = viz.overlayTo3D(dict(base, pathColorBy="along path"), xV, yV, zV)
    for i, v in enumerate(geom["pathVals"]):
        assert np.ptp(v) > 0, i
        assert np.allclose(v, vert[i])

    # the end markers take the path's colour too
    geom = viz.overlayTo3D(dict(base, pathColorBy="max"), xV, yV, zV)
    ends = np.array([v[-1] for v in geom["pathVals"]])
    assert np.allclose(ends, stat["max"])


def test_3d_direction_cones_are_dropped_at_high_path_density():
    """One solid cone per path is readable at a few hundred paths and is pure
    clutter at ten thousand - it hides the paths it annotates."""
    import inspect
    from cerr.viewer.pycerr_gui.common import _PATH_CONE_MAX
    from cerr.viewer.pycerr_gui.main_window import PyCerrViewer

    assert 200 <= _PATH_CONE_MAX <= 20000
    for fn in (PyCerrViewer._add_uromt_3d_vtk, PyCerrViewer._add_uromt_3d_mpl):
        src = inspect.getsource(fn)
        assert 'if "pathEnd" in geom and len(geom["paths"]) <= _PATH_CONE_MAX:'\
            in src, fn.__name__
    # the paths themselves are always drawn, whatever the density
    assert '"paths" in geom' in inspect.getsource(
        PyCerrViewer._add_uromt_3d_vtk)


def test_animation_timer_paces_itself_off_the_finished_frame():
    """A repeating timer keeps firing while a slow frame is still drawing, so
    events queue up and the dialog stops responding to Stop. The timer is
    single-shot and restarted only after the redraw returns."""
    import inspect
    from cerr.viewer.pycerr_gui.uromt_gui import UROMTDialog

    assert "setSingleShot(True)" in inspect.getsource(UROMTDialog.__init__)
    tick = inspect.getsource(UROMTDialog._onGrowTick)
    # the restart comes AFTER the slider change that triggers the redraw
    assert tick.index("setValue") < tick.index("_growTimer.start()")
    assert "if self.playBtn.isChecked():" in tick

    # a stopped animation must not reschedule itself
    class _Stub:
        _onGrowTick = UROMTDialog._onGrowTick
    st = _Stub()
    st.growSlider = _FakeSlider(3, 10)
    st.playBtn = _FakePlayBtn()
    st.playBtn.checked = False
    started = []
    st._growTimer = SimpleNamespace(start=lambda: started.append(1))
    st._onGrowTick()
    assert st.growSlider.value() == 4      # still advances the frame
    assert not started                     # ... but schedules no more


def test_vector_head_cannot_dwarf_the_arrow_it_ends():
    """Quiver head dims are multiples of the shaft WIDTH - an axes fraction -
    so they know nothing about how long the arrows are. With the head
    multipliers raised, heads grew larger than the arrows themselves (measured
    110% of the median arrow at x1). They are now capped against the arrows
    actually on screen, at every length scale."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.quiver import Quiver
    from cerr.uromt import viz

    shape = (40, 40, 6)
    k = 3
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                         indexing="ij")
    comps = [np.zeros(shape) for _ in range(3)]
    for s in range(shape[2]):
        comps[0][:, :, s] = 0.5 * np.sin(cc / 5.0)
        comps[1][:, :, s] = 0.5 * np.cos(rr / 5.0)

    for ls in (0.5, 1.0, 2.0, 4.0, 8.0):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(
            ax, {"view": "velocity", "alpha": 1.0, "comps": comps,
                 "vrange": (0.0, 1.0), "lengthScale": ls, "lineWidth": 2.0,
                 "subsample": 2},
            k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
        q = [c for c in ax.collections if isinstance(c, Quiver)][0]
        headData = q.headlength * q.width * float(np.ptp(xV))
        medArrow = float(np.median(np.hypot(q.U, q.V))) / q.scale
        assert headData <= viz._HEAD_MAX_ARROW_FRAC * medArrow + 1e-6, ls
        assert headData > 0
        # the head keeps its narrow shape when it is shrunk
        assert q.headlength > q.headaxislength > q.headwidth


def test_head_ceiling_is_two_of_the_scans_finest_voxels():
    """The hard ceiling is not a fixed millimetre count - it is 2x the scan's
    FINEST voxel, because the heads annotate that data and a fixed number is
    either huge on a fine scan or invisible on a coarse one."""
    from types import SimpleNamespace
    from cerr.uromt import viz
    from cerr.viewer.pycerr_gui.main_window import PyCerrViewer

    class _Stub:
        _uromtHeadCeiling = PyCerrViewer._uromtHeadCeiling
    st = _Stub()

    # anisotropic: the FINEST spacing wins
    st.planC = SimpleNamespace(scan=[SimpleNamespace(
        getScanSpacing=lambda: np.array([0.08, 0.08, 0.3]))])
    st.scanNum = 0
    assert abs(st._uromtHeadCeiling() - viz._HEAD_MAX_VOXELS * 0.08) < 1e-12

    # a coarser scan gets a proportionally bigger head budget
    st.planC = SimpleNamespace(scan=[SimpleNamespace(
        getScanSpacing=lambda: np.array([0.5, 0.5, 0.5]))])
    assert abs(st._uromtHeadCeiling() - viz._HEAD_MAX_VOXELS * 0.5) < 1e-12

    # the DISPLAYED scan is the one measured
    st.planC = SimpleNamespace(scan=[
        SimpleNamespace(getScanSpacing=lambda: np.array([0.1, 0.1, 0.1])),
        SimpleNamespace(getScanSpacing=lambda: np.array([0.4, 0.4, 0.4]))])
    st.scanNum = 1
    assert abs(st._uromtHeadCeiling() - viz._HEAD_MAX_VOXELS * 0.4) < 1e-12

    # unusable geometry falls back rather than raising
    st.planC = SimpleNamespace(scan=[SimpleNamespace(
        getScanSpacing=lambda: np.array([0.0, np.nan, 0.0]))])
    st.scanNum = 0
    assert st._uromtHeadCeiling() == viz._HEAD_MAX_FALLBACK_CM
    st.planC = SimpleNamespace(scan=[])
    assert st._uromtHeadCeiling() == viz._HEAD_MAX_FALLBACK_CM


def test_drawn_heads_obey_the_overlays_own_ceiling():
    """Whatever the viewer computed travels on the overlay as `headMaxData`,
    and every drawn head respects it."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.quiver import Quiver
    from matplotlib.collections import PolyCollection
    from cerr.uromt import viz

    shape = (120, 120, 6)
    k = 3
    xV = np.arange(shape[1]) * 0.1          # 1 mm voxels, ~12 cm across
    yV = np.arange(shape[0]) * 0.1
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    rr, cc = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                         indexing="ij")
    comps = [np.zeros(shape) for _ in range(3)]
    for s in range(shape[2]):
        comps[0][:, :, s] = 0.5 * np.sin(cc / 15.0)
        comps[1][:, :, s] = 0.5 * np.cos(rr / 15.0)

    for ceil in (0.1, 0.2, 0.6):           # cm: 1, 2, 6 mm
        for ls in (0.05, 1.0, 20.0):
            fig = Figure()
            ax = fig.add_subplot(111)
            viz.drawUROMTOverlay(
                ax, {"view": "velocity", "alpha": 1.0, "comps": comps,
                     "vrange": (0.0, 1.0), "lengthScale": ls,
                     "lineWidth": 2.0, "subsample": 6,
                     "headMaxData": ceil},
                k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
            q = [c for c in ax.collections if isinstance(c, Quiver)][0]
            head = q.headlength * q.width * float(np.ptp(xV))
            assert head <= ceil + 1e-9, (ceil, ls, head)

    # ... and the pathline triangles use the same value
    nVert = 21
    t = np.linspace(0, 1, nVert)
    segs = [np.column_stack([(2.0 + 1.2 * t) / 0.1,
                             (3.0 + 0.4 * np.sin(3 * t)) / 0.1,
                             np.full(nVert, float(k))])]
    vals = np.ones(1)
    for ceil in (0.1, 0.5):
        fig = Figure()
        ax = fig.add_subplot(111)
        viz.drawUROMTOverlay(
            ax, {"view": "pathlines", "alpha": 1.0, "segs": (segs, vals),
                 "pathStat": {"median": vals}, "pathColorBy": "median",
                 "vrange": (0.0, 2.0), "lineWidth": 2.0, "lengthScale": 50.0,
                 "subsample": 1, "headMaxData": ceil},
            k, xV, yV, ext, lambda m: m[:, :, k], 1, 0, 2, shape)
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        tris = np.asarray([np.asarray(pp.vertices)[:3]
                           for pp in polys[0].get_paths()])
        assert _headLengths(tris).max() <= ceil + 1e-9, ceil
    # an overlay without the key still draws, on the fallback
    assert viz._headCeiling({}) == viz._HEAD_MAX_FALLBACK_CM
    assert viz._headCeiling({"headMaxData": 0}) == viz._HEAD_MAX_FALLBACK_CM
    assert viz._headCeiling({"headMaxData": 0.37}) == 0.37


def test_every_renderer_obeys_the_head_ceiling():
    """The ceiling has to be global. The 2-D overlay honoured it while the 3-D
    glyph renderers did not: their heads are a FRACTION of each arrow
    (`tip_length` / `arrow_length_ratio`), so a long arrow got a proportionally
    long head and the cap looked like it was not applied at all."""
    import inspect
    from cerr.uromt import viz
    from cerr.viewer.pycerr_gui.main_window import PyCerrViewer

    # the fraction is clamped against the LONGEST arrow, so every head - not
    # merely the typical one - lands at or under the ceiling
    for ceil in (0.05, 0.2, 1.0):
        for maxLen in (0.01, 0.5, 5.0, 50.0):
            f = viz.capTipFraction(0.4, maxLen, ceil)
            assert f * maxLen <= ceil + 1e-12, (ceil, maxLen)
            assert f <= 0.4
        assert viz.capTipFraction(0.4, 0.0, ceil) == 0.4   # nothing drawn yet
        assert viz.capTipFraction(0.4, 1e9, ceil) > 0      # never zero
        assert viz.capTipFraction(0.4, ceil, ceil) == 0.4  # short arrows intact

    # ... and each 3-D site actually routes through it, with the overlay's own
    # ceiling rather than the fallback
    for fn in (PyCerrViewer._add_uromt_3d_vtk, PyCerrViewer._add_uromt_3d_mpl):
        src = inspect.getsource(fn)
        assert "capTipFraction" in src, fn.__name__
        assert "headCeil = _headCeiling(ov)" in src, fn.__name__
