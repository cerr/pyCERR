"""urOMT Part 2 - run the optimization.

Port of ``runUROMT.m`` (the time-interval loop) and ``Inverse/GNblock_ur.m``
(the per-interval velocity/source solver). Two per-interval solvers are
available, selected by ``cfg.solver``:

* :func:`gnBlockExact` (``solver='gn'``, default) - the exact Gauss-Newton block
  of ``GNblock_ur.m``: the GN Hessian (analytic regularization diagonal plus the
  ``2*beta*hd*J'J`` data-misfit term) is applied matrix-free through the
  validated final-density sensitivities, solved by CG and a backtracking line
  search, iterated ``maxUiter`` times (few iterations = early-stopping
  regularization, as in the MATLAB).
* :func:`gnBlockUr` (``solver='lbfgs'``) - minimizes the same objective with the
  validated first-order adjoint gradient and scipy L-BFGS-B (first-pass path).
"""

import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg

from cerr.uromt.numerics import (paramInit, getGamma, gradGamma, _interpMats,
                                 sourceAdvecDiff, forwardSensitivity,
                                 adjointSensitivity, precomputeSensDeriv)


def gnBlockUr(rho0, u0, r0, par, drhoN, tag=""):
    """Minimize Gamma(u, r) for one time interval (GNblock_ur analog).

    Returns dict with optimized ``u`` (3N*nt,), ``r`` (N*nt,), evolved density
    ``rho`` (N x nt), the objective components and the optimizer result.
    """
    N, nt = par["N"], par["nt"]
    nu = 3 * N * nt

    def funjac(x):
        u = x[:nu]
        r = x[nu:]
        interp = _interpMats(par, u.reshape(3 * N, nt, order="F"))
        G, _, _ = getGamma(rho0, u, r, par, drhoN, interp)
        gU, gR = gradGamma(rho0, u, r, par, drhoN, interp)
        return float(G), np.concatenate([gU, gR])

    x0 = np.concatenate([np.asarray(u0, float).ravel(),
                         np.asarray(r0, float).ravel()])
    t0 = time.time()
    res = minimize(funjac, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": int(par["maxUiter"]),
                            "maxfun": 5 * int(par["maxUiter"]) + 5})
    u = res.x[:nu]
    r = res.x[nu:]
    G, comps, rho = getGamma(rho0, u, r, par, drhoN)
    return dict(u=u, r=r, rho=rho, Gamma=G,
                Gamma1=comps[0], Gamma2=comps[1], Gamma3=comps[2],
                nfev=res.nfev, time=time.time() - t0, tag=tag)


def gnBlockExact(rho0, u0, r0, par, drhoN, tag="", lmbda0=None, maxLM=6,
                 maxLs=8):
    """Exact Gauss-Newton block step for one interval (port of GNblock_ur).

    Applies the **same Gauss-Newton Hessian as MATLAB's GNblock_ur** -
    ``2*beta*hd * J' J`` (the data-misfit GN term, ``J = d rho_N / d(u,r)``)
    plus the kinetic/source regularization - matrix-free via the validated
    final-density sensitivities (:func:`forwardSensitivity` /
    :func:`adjointSensitivity`, the ``Sensitivities/`` operators). A Levenberg-
    Marquardt damping ``lam*I`` is added purely for conditioning (set
    ``lmbda0=0`` for pure Gauss-Newton)::

        H = H_reg + 2*beta*hd * J' J + lam*I

    where ``H_reg`` is the analytic (rho-fixed) Hessian of the kinetic and
    source penalties and ``J = d rho_N / d(u, r)``. Because the misfit term is
    hugely rank-deficient in ``u`` (rank <= N, while u has 3*N*nt DOFs) and the
    kinetic regularization ``diagU = 2*hd*dt*rho`` scales with the physical
    voxel volume ``hd``, the damping ``lam`` is scaled to the misfit diagonal
    ``2*beta*hd``
    and adapted Levenberg-Marquardt style: it is reduced toward Gauss-Newton
    while steps succeed and raised toward gradient-descent when they fail. Each
    trial solves ``H dx = -grad`` by **preconditioned** CG (Jacobi
    preconditioner from the regularization+damping diagonal, as in MATLAB's
    ``GNblock_ur`` PCG, so the near-undamped GN system converges) to tol 1e-4,
    then accepts an Armijo backtracking (halving) step; iterated ``maxUiter``
    times.

    Args:
        lmbda0 (float): initial damping relative to the misfit scale ``2*beta*hd``.
        maxLM (int): max Levenberg retries (damping increases) per outer step.
        maxLs (int): max backtracking line-search trials per CG solve.
    """
    N, nt, dt, hd = par["N"], par["nt"], par["dt"], par["hd"]
    alpha, beta = par["alpha"], par["beta"]
    chi = par.get("chi")
    nu = 3 * N * nt
    u = np.asarray(u0, float).ravel().copy()
    r = np.asarray(r0, float).ravel().copy()
    t0 = time.time()
    nfev = 0

    if lmbda0 is None:                      # initial damping (settings: gnLambda0)
        lmbda0 = float(par.get("gnLambda0", 0.1))
    chiCol = (np.ones((N, nt)) if chi is None else chi)
    scale = 2.0 * beta * hd                 # misfit Hessian diagonal scale
    lmRel = float(lmbda0)                    # damping relative to `scale`
    for _ in range(int(par["maxUiter"])):
        interp = _interpMats(par, u.reshape(3 * N, nt, order="F"))
        rho = sourceAdvecDiff(rho0, u, r, par, interp)
        G0, _, _ = getGamma(rho0, u, r, par, drhoN, interp)
        gU, gR = gradGamma(rho0, u, r, par, drhoN, interp)
        grad = np.concatenate([gU, gR])
        if np.linalg.norm(grad) < 1e-10:
            break

        # diagonal regularization Hessian (rho held fixed, GN style)
        diagU = (2.0 * hd * dt * np.tile(rho, (3, 1))).ravel(order="F")
        diagR = (2.0 * alpha * hd * dt * rho * chiCol).ravel(order="F")
        diagReg = np.concatenate([diagU, diagR])
        # per-step trilinear derivatives are constant across the CG matvecs of
        # this outer step (depend only on the fixed rho/r) - precompute once.
        dSlist = precomputeSensDeriv(rho0, r, par, interp, rho)
        ntot = nu + N * nt

        accepted = False
        for _lm in range(int(maxLM)):
            lam = lmRel * scale

            def matvec(x, lam=lam):
                vu = x[:nu]
                vr = x[nu:]
                hu = (diagU + lam) * vu          # regularization + damping
                hr = (diagR + lam) * vr
                # misfit Gauss-Newton term 2*beta*hd * J' (J v)
                Jv = forwardSensitivity(rho0, u, r, vu, vr, par, interp,
                                        rho, dSlist)[:, -1]
                Ju, Jr = adjointSensitivity(rho0, u, r, 2.0 * beta * hd * Jv,
                                            par, interp, rho, dSlist)
                return np.concatenate([hu + Ju, hr + Jr])

            H = LinearOperator((ntot, ntot), matvec=matvec)
            # Jacobi preconditioner from the (exactly known) regularization +
            # damping diagonal - the MATLAB GNblock_ur PCG preconditioner. It
            # conditions the velocity null-space the rank-deficient misfit term
            # can't see, so the near-undamped GN system converges; floored at a
            # fraction of its median to stay bounded where rho ~ 0.
            md = diagReg + lam
            pos = md[md > 0]
            floor = 1e-2 * (float(np.median(pos)) if pos.size else 1.0)
            Minv = 1.0 / np.maximum(md, floor)
            M = LinearOperator((ntot, ntot),
                               matvec=lambda z, mv=Minv: mv * z)
            dx, _ = cg(H, -grad, rtol=1e-4, maxiter=int(par["niter_pcg"]), M=M)
            gd = float(grad @ dx)               # directional derivative

            step = 1.0
            stepAccepted = False
            for _ls in range(int(maxLs)):       # Armijo backtracking (halving)
                uTry = u + step * dx[:nu]
                rTry = r + step * dx[nu:]
                GTry, _, _ = getGamma(rho0, uTry, rTry, par, drhoN)
                nfev += 1
                if GTry <= G0 + 1e-4 * step * gd:
                    u, r = uTry, rTry
                    stepAccepted = True
                    break
                step *= 0.5
            if stepAccepted:
                lmRel = max(lmRel * 0.5, 1e-10)  # success -> toward Gauss-Newton
                accepted = True
                break
            lmRel *= 4.0                         # failure -> more damping, retry
        if not accepted:
            break

    G, comps, rho = getGamma(rho0, u, r, par, drhoN)
    return dict(u=u, r=r, rho=rho, Gamma=G,
                Gamma1=comps[0], Gamma2=comps[1], Gamma3=comps[2],
                nfev=nfev, time=time.time() - t0, tag=tag)


_SOLVERS = {"lbfgs": gnBlockUr, "gn": gnBlockExact}


def runUROMT(cfg, statusCallback=None):
    """Run urOMT over the consecutive frame intervals (runUROMT.m analog).

    Requires ``cfg`` already populated by
    :func:`cerr.uromt.data.prepareData` (``cfg.vol``, ``cfg.mask``,
    ``cfg.trueSize``, ``cfg.spacing``).

    Args:
        cfg (UROMTConfig): configuration with prepared data.
        statusCallback (callable): optional ``f(fraction, message)`` progress
            hook.

    Returns:
        dict: results with per-interval lists ``u`` (3 x N x nt), ``r``
        (N x nt), ``rho`` (N x nt), objective components ``gamma``, plus the
        grid (``n``, ``spacing``), the ROI ``mask`` and ``bbox`` for mapping
        the velocity/source fields back into planC.
    """
    if not cfg.vol:
        raise ValueError("cfg has no data; run prepareData(cfg, planC) first.")
    par = paramInit(cfg)
    N, nt = par["N"], par["nt"]
    solverName = str(getattr(cfg, "solver", "lbfgs")).lower()
    blockSolver = _SOLVERS.get(solverName, gnBlockUr)
    frames = [np.asarray(v, float).ravel(order="F") for v in cfg.vol]
    nIntervals = len(frames) - 1

    out = dict(u=[], r=[], rho=[], gamma=[], n=par["n"],
               spacing=par["h"], mask=cfg.mask, bbox=cfg.bbox,
               frameScanNums=getattr(cfg, "frameScanNums", None),
               doResize=int(getattr(cfg, "do_resize", 0)),
               sizeFactor=float(getattr(cfg, "size_factor", 1.0)),
               dt=par["dt"], nt=par["nt"], sigma=par["sigma"])

    u = np.zeros(3 * N * nt)
    r = np.zeros(N * nt)
    rhoEnd = frames[0]
    reinit = bool(int(cfg.reinitR))
    for t in range(nIntervals):
        if statusCallback:
            statusCallback(t / nIntervals,
                           "urOMT interval %d/%d" % (t + 1, nIntervals))
        rho0 = frames[t] if (reinit or t == 0) else rhoEnd
        drhoN = frames[t + 1]
        sol = blockSolver(rho0, u, r, par, drhoN, tag="interval %d" % (t + 1))
        out["u"].append(sol["u"].reshape(3, N, nt, order="F"))
        out["r"].append(sol["r"].reshape(N, nt, order="F"))
        out["rho"].append(sol["rho"])
        out["gamma"].append({k: sol[k] for k in
                             ("Gamma", "Gamma1", "Gamma2", "Gamma3",
                              "nfev", "time")})
        rhoEnd = sol["rho"][:, -1]
        u, r = sol["u"], sol["r"]      # warm-start next interval
    if statusCallback:
        statusCallback(1.0, "urOMT done")
    return out
