"""urOMT post-processing - Eulerian (runEULA) and Lagrangian (runGLAD).

Ports of Parts 3 & 4 of the MATLAB urOMT ``driver_RatBrain.m`` (``runEULA.m`` /
``paramInitEULApar.m`` and ``runGLAD.m`` / ``paramInitGLADpar.m``), adapted from
the MATLAB file-based pipeline to operate in-memory on the dict returned by
:func:`cerr.uromt.solver.runUROMT`.

The urOMT advection-diffusion model
``rho_t + div(rho v) = sigma Lap(rho) + rho r`` has an equivalent advective form
with the **effective (flux) velocity**

    v_eff = v - sigma * grad(log rho)

i.e. the OMT velocity ``v`` minus the diffusive drift. Eulerian maps summarize
the fields over time; Lagrangian pathlines integrate ``v_eff`` to trace
transport, with the Peclet number ``Pe = |v| / |sigma grad(log rho)|``
contrasting advective vs. diffusive speed.
"""

import numpy as np

_EPS = 1e-8


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _cellCenters(n, h):
    return [(np.arange(ni) + 0.5) * hi for ni, hi in zip(n, h)]


def _gradLog(rho3, h):
    """Cell-centered grad(log rho) via central differences; returns a (3, N)
    array (components along axes 0,1,2 = row,col,slice), Fortran-flattened."""
    lr = np.log(np.maximum(rho3, 0.0) + _EPS)
    g = np.gradient(lr, h[0], h[1], h[2], edge_order=1)
    return np.stack([gi.ravel(order="F") for gi in g], axis=0)


def _stepFields(v3N, rhoN, n, h, sigma):
    """For one (velocity, density) step return effective velocity v_eff (3,N),
    advective speed |v| (N,), diffusive speed |sigma grad log rho| (N,)."""
    rho3 = rhoN.reshape(n, order="F")
    vdiff = sigma * _gradLog(rho3, h)                  # (3,N)
    vEff = v3N - vdiff
    advSpeed = np.sqrt(np.sum(v3N ** 2, axis=0))
    difSpeed = np.sqrt(np.sum(vdiff ** 2, axis=0))
    return vEff, advSpeed, difSpeed


def _globalSteps(result):
    """Yield (g, v (3,N), rho (N,), r (N,)) for every global sub-step g over all
    intervals (g = interval*nt + k)."""
    nt = int(result["nt"])
    for t, (u, rr, rho) in enumerate(zip(result["u"], result["r"],
                                         result["rho"])):
        for k in range(nt):
            yield t * nt + k, u[:, :, k], rho[:, k], rr[:, k]


# --------------------------------------------------------------------------- #
#  Part 3: Eulerian post-processing (runEULA.m)
# --------------------------------------------------------------------------- #
def runEULA(result, maskOnly=True):
    """Eulerian post-processing: time-averaged speed, rate, Peclet and flux maps.

    Args:
        result (dict): output of :func:`cerr.uromt.solver.runUROMT`.
        maskOnly (bool): zero the maps outside the ROI mask.

    Returns:
        dict ``Eul`` with flattened (N,) maps ``speed`` (mean |v|), ``rate``
        (mean r), ``peclet`` (mean |v|/|diffusion|), the mean flux vector
        ``flux`` (3,N) = mean(rho * v_eff), their 3-D ROI-grid reshapes
        (``speed3``/``rate3``/``peclet3``), and grid metadata (``n``,
        ``spacing``, ``mask``, ``bbox``, ``frameScanNums``).
    """
    n = [int(v) for v in result["n"]]
    h = [float(v) for v in result["spacing"]]
    sigma = float(result.get("sigma", 0.0))
    N = int(np.prod(n))

    speed = np.zeros(N)
    rate = np.zeros(N)
    peclet = np.zeros(N)
    flux = np.zeros((3, N))
    nSteps = 0
    for _g, v, rho, r in _globalSteps(result):
        vEff, adv, dif = _stepFields(v, rho, n, h, sigma)
        speed += adv
        rate += r
        peclet += adv / (dif + _EPS)
        flux += rho * vEff
        nSteps += 1
    if nSteps:
        speed /= nSteps
        rate /= nSteps
        peclet /= nSteps
        flux /= nSteps

    if maskOnly:
        m = (np.asarray(result["mask"]) > 0).ravel(order="F")
        speed[~m] = 0.0
        rate[~m] = 0.0
        peclet[~m] = 0.0
        flux[:, ~m] = 0.0

    Eul = dict(speed=speed, rate=rate, peclet=peclet, flux=flux,
               speed3=speed.reshape(n, order="F"),
               rate3=rate.reshape(n, order="F"),
               peclet3=peclet.reshape(n, order="F"),
               n=n, spacing=h, mask=result["mask"], bbox=result["bbox"],
               frameScanNums=result.get("frameScanNums"))
    return Eul


def runEULAIntervals(result, maskOnly=True):
    """Per-interval Eulerian maps (time-averaged over each interval's nt
    sub-steps), as the MATLAB ``runEULA`` writes one set of maps per time
    interval. Returns lists (one entry per interval) of 3-D ROI-grid arrays.

    Keys: ``speed`` (|v|, advective), ``effSpeed`` (|v_eff|, flux velocity),
    ``rate`` (r), ``peclet`` (|v|/|diffusion|), ``flux`` (rho*v_eff, list of
    (3,*n)), ``rho`` (density); plus grid metadata. (MATLAB ``EulerS`` is the
    flux-velocity magnitude ``effSpeed``; ``EulerR``=rate, ``EulerPe``=peclet,
    ``EulerRho``=rho, ``EulerFlux``=flux.)
    """
    n = [int(v) for v in result["n"]]
    h = [float(v) for v in result["spacing"]]
    sigma = float(result.get("sigma", 0.0))
    nt = int(result["nt"])
    m = (np.asarray(result["mask"]) > 0).ravel(order="F")
    out = {k: [] for k in ("speed", "effSpeed", "rate", "peclet", "flux",
                           "rho")}
    for u, rr, rho in zip(result["u"], result["r"], result["rho"]):
        N = u.shape[1]
        acc = {k: np.zeros(N) for k in ("speed", "effSpeed", "rate", "peclet",
                                        "rho")}
        flux = np.zeros((3, N))
        for k in range(nt):
            v = u[:, :, k]
            rho_k = rho[:, k]
            vEff, adv, dif = _stepFields(v, rho_k, n, h, sigma)
            acc["speed"] += adv
            acc["effSpeed"] += np.sqrt(np.sum(vEff ** 2, axis=0))
            acc["rate"] += rr[:, k]
            acc["peclet"] += adv / (dif + _EPS)
            acc["rho"] += rho_k
            flux += rho_k * vEff
        for k in acc:
            acc[k] /= nt
        flux /= nt
        if maskOnly:
            for k in acc:
                acc[k][~m] = 0.0
            flux[:, ~m] = 0.0
        for k in ("speed", "effSpeed", "rate", "peclet", "rho"):
            out[k].append(acc[k].reshape(n, order="F"))
        out["flux"].append(flux.reshape((3,) + tuple(n), order="F"))
    out.update(n=n, spacing=h, mask=result["mask"], bbox=result["bbox"],
               frameScanNums=result.get("frameScanNums"))
    return out


# --------------------------------------------------------------------------- #
#  Part 4: Lagrangian post-processing (runGLAD.m)
# --------------------------------------------------------------------------- #
def _interpolators(field3, gc):
    """RegularGridInterpolator for each component of a (k, *n) field."""
    from scipy.interpolate import RegularGridInterpolator
    return [RegularGridInterpolator(gc, field3[c], bounds_error=False,
                                    fill_value=0.0) for c in range(len(field3))]


def runGLAD(result, spfs=2, nEuler=5, direction=1.0, minSpeed=0.0,
            slTolVox=1.0, maxSeeds=4000):
    """Lagrangian post-processing: integrate transport pathlines of the
    effective velocity ``v_eff`` seeded in the ROI.

    Args:
        result (dict): output of :func:`cerr.uromt.solver.runUROMT`.
        spfs (int): seed every ``spfs``-th masked voxel per axis.
        nEuler (int): Euler sub-steps per urOMT time sub-step.
        direction (float): +1 follows the urOMT velocity, -1 reverses it.
        minSpeed (float): stop advancing a particle where ``|v_eff| < minSpeed``.
        slTolVox (float): drop pathlines whose net displacement (voxels) is below
            this (removes near-stationary seeds).
        maxSeeds (int): cap on the number of seed particles (keeps it tractable).

    Returns:
        dict ``Lag`` with ``SL`` (list of (steps,3) pathlines in ROI voxel
        coords row,col,slice), ``sstream`` / ``pestream`` (per-pathline speed and
        Peclet lines), ``startp`` (M,3), ``disp`` (M,3) and ``displen`` (M,) net
        displacements (mm), ``ind_msk``, and grid metadata.
    """
    n = [int(v) for v in result["n"]]
    h = [float(v) for v in result["spacing"]]
    sigma = float(result.get("sigma", 0.0))
    dt = float(result["dt"])
    gc = _cellCenters(n, h)

    # ---- seed points: every spfs-th masked voxel (cell-centered coords) ----
    mask3 = np.asarray(result["mask"]) > 0
    seedMask = np.zeros_like(mask3)
    s = max(1, int(spfs))
    seedMask[::s, ::s, ::s] = True
    seedMask &= mask3
    ri, ci, si = np.where(seedMask)
    if ri.size > maxSeeds:                              # thin uniformly
        sel = np.linspace(0, ri.size - 1, maxSeeds).astype(int)
        ri, ci, si = ri[sel], ci[sel], si[sel]
    startVox = np.stack([ri, ci, si], axis=1).astype(float)
    pos = np.stack([(ri + 0.5) * h[0], (ci + 0.5) * h[1],
                    (si + 0.5) * h[2]], axis=1)         # (M,3) physical
    M = pos.shape[0]

    # Pre-allocated history arrays (avoid a per-seed Python append loop, which
    # was O(nIntervals*nt*nEuler*M) and dominated the full-run cost): one
    # recorded position per Euler sub-step plus the seed, speed/Peclet per
    # sub-step. Vectorized assignment over all M seeds at once.
    nEulerI = int(max(1, nEuler))
    nGlob = sum(1 for _ in _globalSteps(result))
    nRec = 1 + nGlob * nEulerI
    voxHist = np.empty((nRec, M, 3))
    voxHist[0] = startVox
    spdHist = np.empty((nGlob * nEulerI, M))
    peHist = np.empty((nGlob * nEulerI, M))
    mdt = dt / float(nEulerI)
    hInv = 1.0 / np.asarray(h)

    # ---- integrate through the time-varying effective velocity ------------
    rec = 0
    for _g, v, rho, _r in _globalSteps(result):
        vEff, adv, dif = _stepFields(v, rho, n, h, sigma)
        vc = [vEff[c].reshape(n, order="F") for c in range(3)]
        sp3 = adv.reshape(n, order="F")
        pe3 = (adv / (dif + _EPS)).reshape(n, order="F")
        vinterp = _interpolators(vc, gc)
        sinterp = _interpolators([sp3], gc)[0]
        pinterp = _interpolators([pe3], gc)[0]
        for _sub in range(nEulerI):
            vv = np.stack([vinterp[c](pos) for c in range(3)], axis=1)  # (M,3)
            spdHist[rec] = sinterp(pos)
            peHist[rec] = pinterp(pos)
            moving = np.sqrt(np.sum(vv ** 2, axis=1)) >= minSpeed
            pos = pos + direction * mdt * vv * moving[:, None]
            rec += 1
            voxHist[rec] = (pos - 0.5 * np.asarray(h)) * hInv   # physical->vox

    # ---- assemble, filter near-stationary pathlines -----------------------
    disp = voxHist[-1] - voxHist[0]                      # (M,3) voxels
    keep = np.where(np.linalg.norm(disp, axis=1) >= slTolVox)[0]
    SL, sstream, pestream, keepStart, dispV = [], [], [], [], []
    for m in keep:
        SL.append(voxHist[:, m, :].copy())
        sstream.append(spdHist[:, m].copy())
        pestream.append(peHist[:, m].copy())
        keepStart.append(startVox[m])
        dispV.append(disp[m] * np.asarray(h))            # cm displacement
    startp = np.asarray(keepStart) if keepStart else np.zeros((0, 3))
    dispCm = np.asarray(dispV) if dispV else np.zeros((0, 3))
    displen = (np.linalg.norm(dispCm, axis=1) if dispCm.size
               else np.zeros(0))

    Lag = dict(SL=SL, sstream=sstream, pestream=pestream, startp=startp,
               disp=dispCm, displen=displen, ind_msk=np.arange(len(SL)),
               n=n, spacing=h, bbox=result["bbox"], mask=result["mask"],
               frameScanNums=result.get("frameScanNums"))
    return Lag
