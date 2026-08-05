"""urOMT Part 2 - run the optimization.

:func:`solveUROMT` is a port of ``runUROMT.m`` (the time-interval loop);
``Inverse/GNblock_ur.m`` is the per-interval velocity/source solver. Two
per-interval solvers are available, selected by ``cfg.solver``:

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
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator as _spLinOp, cg as _spcg

from cerr.uromt.gpu import _normalizeCg

from cerr.uromt.numerics import (paramInit, getGamma, gradGamma, _interpMats,
                                 sourceAdvecDiff, forwardSensitivity,
                                 adjointSensitivity, precomputeSensDeriv)


def gnBlockUr(rho0, u0, r0, par, drhoN, tag=""):
    """Minimize Gamma(u, r) for one time interval (GNblock_ur analog).

    Returns dict with optimized ``u`` (3N*nt,), ``r`` (N*nt,), evolved density
    ``rho`` (N x nt), the objective components and the optimizer result.

    CPU only - scipy's L-BFGS-B works on host arrays. The ``useGPU`` setting
    applies to :func:`gnBlockExact` (``solver='gn'``, the default).
    """
    if par.get("bk") is not None and par["bk"].isGpu:
        raise ValueError("solver='lbfgs' does not support useGPU='yes'; use "
                         "solver='gn' (the default) or set useGPU='no'.")
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
                Gamma4=comps[3],
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
    source penalties and ``J = d rho_N / d(u, r)``.

    **Defaults reproduce MATLAB's ``GNblock_ur``**, which calls plain
    ``pcg(H, -g, 0.01, niter_pcg)``: no preconditioner, no damping, CG tolerance
    1e-2, and a line search starting at 0.7 and halving with Armijo constant
    1e-8. Validated against the reference implementation: with these settings
    pyCERR reaches the same optimum (total objective within 0.3%, source-energy
    term within 1.3%), whereas enabling the Jacobi preconditioner lands on a
    ~3.7x worse minimum with the source term ~500x too small.

    The optional Jacobi preconditioner (``gnPrecond=1``) is built from the
    regularization+damping diagonal. It is **off by default** because
    ``diagR`` carries the factor ``alpha`` while ``diagU`` does not, so
    ``Minv = 1/(diagReg+lam)`` is ~``alpha`` larger on the velocity block: the
    preconditioned CG over-steps in velocity and starves the source ``r``.
    Likewise the Levenberg-Marquardt damping (``gnLambda0>0``, relative to the
    misfit scale ``2*beta*hd``, adapted down on accepted steps and up on
    rejected ones) is off by default; enable either only for conditioning if a
    particular problem will not converge.

    Args:
        lmbda0 (float): initial damping relative to the misfit scale
            ``2*beta*hd``; ``None`` -> ``par['gnLambda0']`` (default 0 = none).
        maxLM (int): max Levenberg retries per outer step (only meaningful when
            damping is enabled).
        maxLs (int): max backtracking line-search trials per CG solve.
    """
    N, nt, dt, hd = par["N"], par["nt"], par["dt"], par["hd"]
    alpha, beta = par["alpha"], par["beta"]
    chi = par.get("chi")
    xp = par.get("xp", np)
    bk = par.get("bk")
    cg = bk.cg if bk is not None else _normalizeCg(_spcg)
    LinearOperator = bk.LinearOperator if bk is not None else _spLinOp
    eta = float(par.get("eta", 0.0))         # velocity H1-smoothness weight
    Grad = par["Grad"]
    cSmooth = 2.0 * eta * hd * dt            # smoothness Hessian scale (const)
    nu = 3 * N * nt
    u = xp.asarray(u0, dtype=xp.float64).ravel().copy()
    r = xp.asarray(r0, dtype=xp.float64).ravel().copy()
    rho0 = xp.asarray(rho0, dtype=xp.float64)
    drhoN = xp.asarray(drhoN, dtype=xp.float64)
    t0 = time.time()
    nfev = 0

    if lmbda0 is None:                      # initial damping (settings: gnLambda0)
        lmbda0 = float(par.get("gnLambda0", 0.0))
    usePrecond = bool(int(par.get("gnPrecond", 0)))   # MATLAB uses plain pcg
    cgTol = float(par.get("gnCgTol", 1e-2))           # MATLAB pcg tol
    chiCol = (xp.ones((N, nt)) if chi is None else chi)
    scale = 2.0 * beta * hd                 # misfit Hessian diagonal scale
    lmRel = float(lmbda0)                    # damping relative to `scale`
    for _ in range(int(par["maxUiter"])):
        interp = _interpMats(par, u.reshape(3 * N, nt, order="F"))
        # one forward solve serves the objective and the gradient
        rho = sourceAdvecDiff(rho0, u, r, par, interp)
        G0, _, _ = getGamma(rho0, u, r, par, drhoN, interp, rho)
        gU, gR = gradGamma(rho0, u, r, par, drhoN, interp, rho)
        grad = xp.concatenate([gU, gR])
        if float(xp.linalg.norm(grad)) < 1e-10:
            break

        # diagonal regularization Hessian (rho held fixed, GN style)
        diagU = (2.0 * hd * dt * xp.tile(rho, (3, 1))).ravel(order="F")
        diagR = (2.0 * alpha * hd * dt * rho * chiCol).ravel(order="F")
        diagReg = xp.concatenate([diagU, diagR])
        if eta:                              # add H1 Laplacian diagonal (precond)
            lapDiag = par["lapDiag"]
            smoothDiag = cSmooth * xp.repeat(
                xp.tile(lapDiag, 3)[:, None], nt, axis=1).ravel(order="F")
            diagReg = diagReg.copy()
            diagReg[:nu] += smoothDiag
        # per-step trilinear derivatives are constant across the CG matvecs of
        # this outer step (depend only on the fixed rho/r) - precompute once.
        dSlist = precomputeSensDeriv(rho0, r, par, interp, rho)
        ntot = nu + N * nt

        accepted = False
        for _lm in range(int(maxLM)):
            lam = lmRel * scale
            # damped regularization diagonals are fixed for this CG solve
            diagUlam = diagU + lam
            diagRlam = diagR + lam

            def matvec(x, dU=diagUlam, dR=diagRlam):
                vu = x[:nu]
                vr = x[nu:]
                hu = dU * vu                     # kinetic regularization + damping
                hr = dR * vr
                if eta:                          # H1 smoothness Hessian (const)
                    VU = vu.reshape(3 * N, nt, order="F")
                    sm = xp.empty_like(VU)
                    for d in range(3):
                        sm[d * N:(d + 1) * N, :] = cSmooth * (
                            Grad.T @ (Grad @ VU[d * N:(d + 1) * N, :]))
                    hu = hu + sm.ravel(order="F")
                # misfit Gauss-Newton term 2*beta*hd * J' (J v)
                Jv = forwardSensitivity(rho0, u, r, vu, vr, par, interp,
                                        rho, dSlist)[:, -1]
                Ju, Jr = adjointSensitivity(rho0, u, r, 2.0 * beta * hd * Jv,
                                            par, interp, rho, dSlist)
                return xp.concatenate([hu + Ju, hr + Jr])

            H = LinearOperator((ntot, ntot), matvec=matvec)
            # Optional Jacobi preconditioner from the (exactly known)
            # regularization + damping diagonal, floored at a fraction of its
            # median to stay bounded where rho ~ 0. OFF by default: MATLAB's
            # GNblock_ur calls plain `pcg(H,-g,0.01,niter_pcg)`, and because
            # diagR carries the factor `alpha` while diagU does not, Minv is
            # ~alpha larger on the velocity block - the preconditioned CG then
            # over-steps velocity and starves the source r.
            M = None
            if usePrecond:
                md = diagReg + lam
                pos = md[md > 0]
                floor = 1e-2 * (float(xp.median(pos)) if pos.size else 1.0)
                Minv = 1.0 / xp.maximum(md, floor)
                M = LinearOperator((ntot, ntot),
                                   matvec=lambda z, mv=Minv: mv * z)
            dx, _ = cg(H, -grad, cgTol, int(par["niter_pcg"]), M)
            gd = float(grad @ dx)               # directional derivative

            step = 0.7                          # MATLAB muls = 0.7, then halved
            stepAccepted = False
            for _ls in range(int(maxLs)):       # Armijo backtracking (halving)
                uTry = u + step * dx[:nu]
                rTry = r + step * dx[nu:]
                GTry, _, _ = getGamma(rho0, uTry, rTry, par, drhoN)
                nfev += 1
                if GTry <= G0 + 1e-8 * step * gd:
                    u, r = uTry, rTry
                    stepAccepted = True
                    break
                step *= 0.5
            if stepAccepted:
                if lmRel > 0.0:
                    lmRel = max(lmRel * 0.5, 1e-10)  # success -> toward pure GN
                accepted = True
                break
            if lmRel <= 0.0:
                break                            # undamped (MATLAB): no retry
            lmRel *= 4.0                         # failure -> more damping, retry
        if not accepted:
            break

    G, comps, rho = getGamma(rho0, u, r, par, drhoN)
    toHost = bk.toHost if bk is not None else (lambda a: a)
    return dict(u=toHost(u), r=toHost(r), rho=toHost(rho), Gamma=G,
                Gamma1=comps[0], Gamma2=comps[1], Gamma3=comps[2],
                Gamma4=comps[3],
                nfev=nfev, time=time.time() - t0, tag=tag)


_SOLVERS = {"lbfgs": gnBlockUr, "gn": gnBlockExact}


def solveUROMT(cfg, statusCallback=None):
    """Run urOMT over the consecutive frame intervals (runUROMT.m analog).

    This is Part 2 alone. Requires ``cfg`` already populated by
    :func:`cerr.uromt.data.prepareData` (``cfg.vol``, ``cfg.mask``,
    ``cfg.trueSize``, ``cfg.spacing``); for the whole pipeline starting from a
    planC use :func:`cerr.uromt.runUROMT`.

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
               dt=par["dt"], nt=par["nt"], sigma=par["sigma"],
               pecletFloor=float(getattr(cfg, "peclet_floor", 0.1)))

    u = np.zeros(3 * N * nt)
    r = np.zeros(N * nt)
    rhoEnd = frames[0]
    reinit = bool(int(cfg.reinitR))
    warmStart = bool(int(getattr(cfg, "warm_start", 0)))
    for t in range(nIntervals):
        if statusCallback:
            statusCallback(t / nIntervals,
                           "urOMT interval %d/%d" % (t + 1, nIntervals))
        rho0 = frames[t] if (reinit or t == 0) else rhoEnd
        drhoN = frames[t + 1]
        sol = blockSolver(rho0, u, r, par, drhoN, tag="interval %d" % (t + 1))
        # velocity flat layout is [comp*N + voxel + 3N*k] (the solver stacks the
        # 3 components as blocks of N: U[0:N], U[N:2N], U[2N:3N]). Reshaping
        # straight to (3, N, nt) order="F" would read it as comp + 3*voxel,
        # interleaving the components and scrambling voxels (a tiled "montage" in
        # the speed/Peclet/flux maps). Reshape to (N, 3, nt) then move the
        # component axis to front.
        out["u"].append(sol["u"].reshape(N, 3, nt, order="F").transpose(1, 0, 2))
        out["r"].append(sol["r"].reshape(N, nt, order="F"))
        out["rho"].append(sol["rho"])
        out["gamma"].append({k: sol[k] for k in
                             ("Gamma", "Gamma1", "Gamma2", "Gamma3", "Gamma4",
                              "nfev", "time")})
        rhoEnd = sol["rho"][:, -1]
        if warmStart:
            # Carry (u, r) into the next interval. This converges further per
            # interval, but because `maxUiter` is small and acts as early
            # stopping, the velocity ACCUMULATES along the chain: on the
            # reference breast data the kinetic term grew monotonically
            # (4.5e4 -> 3.2e5 over 13 intervals) while the reference's stayed
            # flat, inflating the recovered speed severalfold. Set
            # `warm_start: 0` to restart each interval from rest, which is what
            # the reference implementation does.
            u, r = sol["u"], sol["r"]
        else:
            u = np.zeros(3 * N * nt)
            r = np.zeros(N * nt)
    if statusCallback:
        statusCallback(1.0, "urOMT done")
    return out


def runUROMT(cfg, statusCallback=None):
    """Deprecated alias for :func:`solveUROMT`.

    Renamed so it is not confused with :func:`cerr.uromt.runUROMT`, the
    whole-pipeline wrapper that starts from a planC.
    """
    warnings.warn("cerr.uromt.solver.runUROMT is deprecated; use "
                  "cerr.uromt.solver.solveUROMT (Part 2) or "
                  "cerr.uromt.runUROMT (the whole pipeline).",
                  DeprecationWarning, stacklevel=2)
    return solveUROMT(cfg, statusCallback)
