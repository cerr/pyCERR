"""urOMT Part 2 - run the optimization.

Port of ``runUROMT.m`` (the time-interval loop) and ``Inverse/GNblock_ur.m``
(the per-interval velocity/source solver). The MATLAB ``GNblock_ur`` uses an
exact Gauss-Newton block step (a tree of second-order adjoint sensitivities);
here :func:`gnBlockUr` minimizes the same objective with the validated
first-order adjoint gradient and an L-BFGS-B step - a faithful, runnable first
pass. Swap in the exact GN Hessian later if needed.
"""

import time

import numpy as np
from scipy.optimize import minimize

from cerr.uromt.numerics import (paramInit, getGamma, gradGamma, _interpMats)


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
    frames = [np.asarray(v, float).ravel(order="F") for v in cfg.vol]
    nIntervals = len(frames) - 1

    out = dict(u=[], r=[], rho=[], gamma=[], n=par["n"],
               spacing=par["h"], mask=cfg.mask, bbox=cfg.bbox,
               frameScanNums=getattr(cfg, "frameScanNums", None))

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
        sol = gnBlockUr(rho0, u, r, par, drhoN, tag="interval %d" % (t + 1))
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
