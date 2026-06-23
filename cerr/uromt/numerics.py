"""urOMT numerics - operators, forward advection-diffusion model, objective and
adjoint gradient.

Port of the MATLAB urOMT core (``paramInitFunc.m``, ``Inverse/SourceAdvecDiff.m``,
``Inverse/get_Gamma.m``). The advection-diffusion-source PDE solved on a
cell-centered uniform grid is::

    rho_t + div(rho * u) = sigma * Laplacian(rho) + rho * r        (chi omitted)

State vectors use Fortran (column-major) ordering to match the MATLAB
reshape(.,n') convention: index = i1 + n1*i2 + n1*n2*i3.

The exact Gauss-Newton block solver of ``GNblock_ur.m`` uses a tree of
second-order adjoint-sensitivity operators (the ``Sensitivities/`` folder). Here
we instead supply the objective and its **first-order adjoint gradient** and let
a quasi-Newton optimizer (L-BFGS-B) descend - a faithful, validated first pass
(the gradient is finite-difference checked). The exact GN Hessian can be dropped
in later.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized


# --------------------------------------------------------------------------- #
#  Grid & operators (paramInitFunc.m)
# --------------------------------------------------------------------------- #
def cellCenteredGrid(n, h):
    """Cell-centered physical coordinates, flattened Fortran-order."""
    g = [(np.arange(ni) + 0.5) * hi for ni, hi in zip(n, h)]
    Xc, Yc, Zc = np.meshgrid(g[0], g[1], g[2], indexing="ij")
    return (Xc.ravel(order="F"), Yc.ravel(order="F"), Zc.ravel(order="F"))


def _ddx1d(ni, hi):
    """1-D cell-centered derivative to interior faces, Neumann (zero-flux) BC,
    as an (ni-1) x ni sparse matrix (so D'D is the Neumann Laplacian)."""
    if ni < 2:
        return sp.csr_matrix((0, ni))
    e = np.ones(ni)
    D = sp.spdiags([-e, e], [0, 1], ni - 1, ni) / hi
    return D.tocsr()


def neumannGrad(n, h):
    """Cell-centered gradient with Neumann BC (getCellCenteredGradMatrix 'ccn').
    Stacks the three directional derivatives; ``Grad.T @ Grad`` is the
    Neumann Laplacian (positive semi-definite)."""
    I1, I2, I3 = (sp.identity(ni, format="csr") for ni in n)
    D1, D2, D3 = _ddx1d(n[0], h[0]), _ddx1d(n[1], h[1]), _ddx1d(n[2], h[2])
    G1 = sp.kron(I3, sp.kron(I2, D1))
    G2 = sp.kron(I3, sp.kron(D2, I1))
    G3 = sp.kron(D3, sp.kron(I2, I1))
    return sp.vstack([G1, G2, G3]).tocsr()


def paramInit(cfg):
    """Build the ``par`` dictionary (paramInitFunc.m): grid, operators, the
    implicit-diffusion matrix B = I + dt*sigma*Grad'Grad, and parameters."""
    n = [int(v) for v in cfg.trueSize]
    h = [float(v) for v in cfg.spacing]
    N = int(np.prod(n))
    Grad = neumannGrad(n, h)
    Lap = (Grad.T @ Grad).tocsr()                 # positive (Grad'Grad)
    B = (sp.identity(N, format="csr") + cfg.dt * cfg.sigma * Lap).tocsc()
    Xc, Yc, Zc = cellCenteredGrid(n, h)
    par = dict(dim=3, n=n, h=h, N=N, hd=float(np.prod(h)),
               dt=float(cfg.dt), nt=int(cfg.nt), sigma=float(cfg.sigma),
               alpha=float(cfg.alpha), beta=float(cfg.beta),
               bc=cfg.bc, Xc=Xc, Yc=Yc, Zc=Zc, Grad=Grad, B=B,
               Bsolve=factorized(B), niter_pcg=int(cfg.niter_pcg),
               maxUiter=int(cfg.maxUiter))
    return par


# --------------------------------------------------------------------------- #
#  Trilinear interpolation matrix S and its spatial derivative (dTrilinears3d)
# --------------------------------------------------------------------------- #
def _trilinear(par, posX, posY, posZ):
    """Return (S, deriv) for points (posX,posY,posZ) in physical coords.

    S is a sparse (N x N) trilinear-interpolation matrix (S @ field samples
    `field` at the given points, closed/clamped boundaries). `deriv(field)`
    returns d(S@field)/d{x,y,z} (physical) per point as a (3, N) array, used
    for the velocity gradient.
    """
    n, h = par["n"], par["h"]
    n1, n2, n3 = n
    gx = np.clip(posX / h[0] - 0.5, 0, n1 - 1)
    gy = np.clip(posY / h[1] - 0.5, 0, n2 - 1)
    gz = np.clip(posZ / h[2] - 0.5, 0, n3 - 1)
    i0 = np.clip(np.floor(gx).astype(int), 0, n1 - 2)
    j0 = np.clip(np.floor(gy).astype(int), 0, n2 - 2)
    k0 = np.clip(np.floor(gz).astype(int), 0, n3 - 2)
    fx, fy, fz = gx - i0, gy - j0, gz - k0
    N = par["N"]
    rows = np.arange(N)

    def lin(i, j, k):
        return i + n1 * j + n1 * n2 * k

    corners = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                wx = fx if a else (1 - fx)
                wy = fy if b else (1 - fy)
                wz = fz if c else (1 - fz)
                w = wx * wy * wz
                col = lin(i0 + a, j0 + b, k0 + c)
                corners.append((a, b, c, col, w))
    data = np.concatenate([w for *_, w in corners])
    rr = np.tile(rows, 8)
    cc = np.concatenate([col for *_, col, _ in corners])
    S = sp.csr_matrix((data, (rr, cc)), shape=(N, N))

    def deriv(field):
        dX = np.zeros(N)
        dY = np.zeros(N)
        dZ = np.zeros(N)
        for a, b, c, col, _ in corners:
            fv = field[col]
            sx = (1.0 if a else -1.0)
            sy = (1.0 if b else -1.0)
            sz = (1.0 if c else -1.0)
            wx = fx if a else (1 - fx)
            wy = fy if b else (1 - fy)
            wz = fz if c else (1 - fz)
            dX += (sx * wy * wz) * fv
            dY += (wx * sy * wz) * fv
            dZ += (wx * wy * sz) * fv
        return np.array([dX / h[0], dY / h[1], dZ / h[2]])

    return S, deriv


def _interpMats(par, u):
    """Per-step interpolation matrices S_k and deriv closures for velocity u
    reshaped (3N, nt)."""
    N, nt, dt = par["N"], par["nt"], par["dt"]
    Xc, Yc, Zc = par["Xc"], par["Yc"], par["Zc"]
    out = []
    for k in range(nt):
        U1 = u[0:N, k]
        U2 = u[N:2 * N, k]
        U3 = u[2 * N:3 * N, k]
        out.append(_trilinear(par, Xc + dt * U1, Yc + dt * U2, Zc + dt * U3))
    return out


# --------------------------------------------------------------------------- #
#  Forward model (SourceAdvecDiff.m)
# --------------------------------------------------------------------------- #
def sourceAdvecDiff(rho0, u, r, par, interp=None):
    """Evolve rho through nt source -> advection -> diffusion steps.
    Returns rho (N x nt) for steps 1..nt."""
    N, nt, dt = par["N"], par["nt"], par["dt"]
    u = u.reshape(3 * N, nt, order="F")
    r = r.reshape(N, nt, order="F")
    if interp is None:
        interp = _interpMats(par, u)
    Bsolve = par["Bsolve"]
    rho = np.zeros((N, nt))
    prev = rho0
    for k in range(nt):
        m = (1.0 + dt * r[:, k]) * prev           # source
        S, _ = interp[k]
        adv = S @ m                               # advection
        cur = Bsolve(adv)                         # diffusion (B \ adv)
        rho[:, k] = cur
        prev = cur
    return rho


# --------------------------------------------------------------------------- #
#  Objective (get_Gamma.m) and adjoint gradient
# --------------------------------------------------------------------------- #
def getGamma(rho0, u, r, par, drhoN, interp=None):
    """Cost Gamma = Gamma1(kinetic) + alpha*Gamma2(source) + beta*Gamma3(fit).
    Returns (Gamma, (Gamma1, Gamma2, Gamma3), rho)."""
    N, nt, dt, hd = par["N"], par["nt"], par["dt"], par["hd"]
    U = u.reshape(3 * N, nt, order="F")
    R = r.reshape(N, nt, order="F")
    rho = sourceAdvecDiff(rho0, u, r, par, interp)
    uSq = U[0:N, :] ** 2 + U[N:2 * N, :] ** 2 + U[2 * N:3 * N, :] ** 2
    Gamma1 = hd * dt * float(np.sum(rho * uSq))
    Gamma2 = hd * dt * float(np.sum(rho * (R ** 2)))
    Gamma3 = hd * float(np.sum((rho[:, -1] - drhoN) ** 2))
    Gamma = Gamma1 + par["alpha"] * Gamma2 + par["beta"] * Gamma3
    return Gamma, (Gamma1, Gamma2, Gamma3), rho


def gradGamma(rho0, u, r, par, drhoN, interp=None):
    """Analytic adjoint gradient of getGamma w.r.t. (u, r).
    Returns (g_u (3N*nt,), g_r (N*nt,)) in Fortran-flattened layout."""
    N, nt, dt, hd = par["N"], par["nt"], par["dt"], par["hd"]
    alpha, beta = par["alpha"], par["beta"]
    U = u.reshape(3 * N, nt, order="F")
    R = r.reshape(N, nt, order="F")
    if interp is None:
        interp = _interpMats(par, U)
    rho = sourceAdvecDiff(rho0, u, r, par, interp)
    rhoPrev = np.concatenate([rho0[:, None], rho[:, :-1]], axis=1)  # rho_{j-1}
    Bsolve = par["Bsolve"]

    gU = np.zeros((3 * N, nt))
    gR = np.zeros((N, nt))
    carry = np.zeros(N)
    for k in range(nt - 1, -1, -1):           # step k produces rho[:,k]
        uSq_k = U[0:N, k] ** 2 + U[N:2 * N, k] ** 2 + U[2 * N:3 * N, k] ** 2
        explicit = hd * dt * (uSq_k + alpha * R[:, k] ** 2)
        if k == nt - 1:
            explicit = explicit + 2.0 * beta * hd * (rho[:, -1] - drhoN)
        lam = explicit + carry
        b = Bsolve(lam)                       # adjoint through diffusion (B sym)
        S, deriv = interp[k]
        prev = rhoPrev[:, k]
        m = (1.0 + dt * R[:, k]) * prev       # source field at step k
        mbar = S.T @ b                        # adjoint to m
        # velocity gradient: direct kinetic + implicit (advection derivative)
        dS = deriv(m)                         # (3, N) = d(S@m)/d{x,y,z}
        for d in range(3):
            gU[d * N:(d + 1) * N, k] = (2.0 * hd * dt * rho[:, k] * U[d * N:(d + 1) * N, k]
                                        + dt * b * dS[d])
        # source gradient: direct source penalty + implicit (m wrt r)
        gR[:, k] = 2.0 * hd * dt * alpha * rho[:, k] * R[:, k] + dt * prev * mbar
        # propagate adjoint to rho_{k-1}
        carry = (1.0 + dt * R[:, k]) * mbar
    return gU.ravel(order="F"), gR.ravel(order="F")
