"""Profile the urOMT exact Gauss-Newton solver on a realistic-sized interval."""
import cProfile
import pstats
import time
from types import SimpleNamespace

import numpy as np

from cerr.uromt.numerics import paramInit, getGamma
from cerr.uromt.solver import gnBlockExact


def build(n=(40, 33, 25), nt=4):
    cfg = SimpleNamespace(trueSize=list(n), spacing=[0.106, 0.106, 0.14],
                          dt=0.3, nt=nt, sigma=2e-3, alpha=2e4, beta=8000.0,
                          bc="closed", niter_pcg=20, maxUiter=6, chi=None,
                          gnLambda0=0.1)
    par = paramInit(cfg)
    N = par["N"]
    ng = par["n"]
    ii = np.meshgrid(np.arange(ng[0]), np.arange(ng[1]), np.arange(ng[2]),
                     indexing="ij")
    cc = np.array([20.0, 16.0, 12.0])
    s = 4.0
    blob0 = np.exp(-sum((ii[d] - cc[d]) ** 2 for d in range(3)) / (2 * s ** 2))
    blob1 = np.exp(-sum((ii[d] - (cc + [3, 1, 0])[d]) ** 2
                        for d in range(3)) / (2 * s ** 2))
    rho0 = blob0.ravel(order="F")
    drhoN = blob1.ravel(order="F")
    u0 = np.zeros(3 * N * nt)
    r0 = np.zeros(N * nt)
    return par, rho0, drhoN, u0, r0


def main():
    par, rho0, drhoN, u0, r0 = build()
    print("grid", par["n"], "N", par["N"], "nt", par["nt"])
    # warm
    t0 = time.time()
    sol = gnBlockExact(rho0, u0, r0, par, drhoN)
    dt = time.time() - t0
    print("solve time %.2fs  Gamma %.4g  nfev %d" % (dt, sol["Gamma"],
                                                     sol["nfev"]))
    pr = cProfile.Profile()
    pr.enable()
    gnBlockExact(rho0, u0, r0, par, drhoN)
    pr.disable()
    st = pstats.Stats(pr).sort_stats("cumulative")
    st.print_stats(25)
    st.sort_stats("tottime").print_stats(20)


if __name__ == "__main__":
    main()
