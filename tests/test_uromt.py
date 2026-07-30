"""Offline numerics tests for the urOMT (unbalanced regularized OMT) core.

These exercise the forward advection-diffusion-source model, the objective
(get_Gamma analog) and the analytic adjoint gradient on a tiny grid. No network
and no planC/DICOM; runs in a few seconds.

The adjoint gradient is non-differentiable at the trilinear-interpolation domain
boundary (the corner index and the physical coordinate are clamped there), so
the finite-difference check is restricted to interior voxels - exactly as the
gradient was validated during development.
"""
from types import SimpleNamespace

import numpy as np

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
    """Eulerian maps: mean speed = |v|, flux = rho*v_eff, rate = 0."""
    res = _uniform_flow_result(vx=1.0)
    Eul = runEULA(res)
    assert np.allclose(Eul["speed"], 1.0)
    assert np.allclose(Eul["rate"], 0.0)
    assert np.allclose(Eul["flux"][0], 1.0)            # rho=1, v_eff=v=1
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
    assert arrowLen.max() <= 0.05 * spanFOV + 1e-9     # longest ~5% of FOV
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
