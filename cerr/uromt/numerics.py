"""urOMT numerics - operators, forward advection-diffusion model, objective and
adjoint gradient.

Port of the MATLAB urOMT core (``paramInitFunc.m``, ``Inverse/SourceAdvecDiff.m``,
``Inverse/get_Gamma.m``). The advection-diffusion-source PDE solved on a
cell-centered uniform grid is::

    rho_t + div(rho * u) = sigma * Laplacian(rho) + rho * r        (chi omitted)

State vectors use Fortran (column-major) ordering to match the MATLAB
reshape(.,n') convention: index = i1 + n1*i2 + n1*n2*i3.

Two solver paths are provided (in ``solver.py``):

* the **first-order adjoint gradient** (:func:`gradGamma`) driving an L-BFGS-B
  optimizer, and
* the **exact Gauss-Newton block** (``GNblock_ur.m`` analog), which applies the
  GN Hessian matrix-free using the final-density sensitivities
  (:func:`forwardSensitivity` / :func:`adjointSensitivity`, the
  ``Sensitivities/`` folder analog).

The gradient, the tangent-linear model and its adjoint are all
finite-difference / dot-product validated.
"""

import numpy as np
import scipy.sparse as sp
from scipy.fft import dctn, idctn


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


class _DiffusionSolver:
    """Exact, FFT-fast solver for the implicit-diffusion step
    ``B = I + dt*sigma*Grad'Grad`` (Neumann cell-centered Laplacian).

    The Neumann Laplacian is separable (a Kronecker sum of 1-D operators) and is
    diagonalized exactly by the 3-D type-II DCT, so ``B \\ y`` is just
    ``idctn(dctn(y) / eig)`` with ``eig`` the analytic eigenvalues of B. This is
    O(N log N) and machine-exact (it agrees with the sparse LU solve to ~1e-15),
    and replaces the SuperLU factorization that dominated >90% of the solve time
    (its triangular solves fill in heavily for a 3-D Laplacian). B is symmetric,
    so the same operator serves the forward and adjoint diffusion solves.

    The 1-D Neumann-Laplacian eigenvalues are ``(2 - 2 cos(pi k / n)) / h^2``
    (k = 0..n-1); a singleton axis (n = 1) contributes 0, matching the empty
    derivative that :func:`neumannGrad` builds there.
    """

    def __init__(self, n, h, dt, sigma):
        self.n = tuple(int(v) for v in n)
        eig = []
        for ni, hi in zip(self.n, (float(v) for v in h)):
            k = np.arange(ni)
            eig.append((2.0 - 2.0 * np.cos(np.pi * k / ni)) / (hi * hi))
        self.eig = (1.0 + float(dt) * float(sigma) *
                    (eig[0][:, None, None] + eig[1][None, :, None]
                     + eig[2][None, None, :]))

    def __call__(self, x):
        X = np.asarray(x, dtype=np.float64).reshape(self.n, order="F")
        Xh = dctn(X, type=2, norm="ortho")
        Y = idctn(Xh / self.eig, type=2, norm="ortho")
        return Y.ravel(order="F")


def paramInit(cfg):
    """Build the ``par`` dictionary (paramInitFunc.m): grid, operators, the
    implicit-diffusion solver for B = I + dt*sigma*Grad'Grad, and parameters.

    Optional ``cfg.chi`` (the MATLAB source-indicator ``K``) is stored as an
    (N, nt) array; it elementwise-scales the relative source ``r`` in the
    forward model and the source penalty. ``None`` -> identity (K = 1)."""
    n = [int(v) for v in cfg.trueSize]
    h = [float(v) for v in cfg.spacing]
    N = int(np.prod(n))
    Grad = neumannGrad(n, h)
    Xc, Yc, Zc = cellCenteredGrid(n, h)
    par = dict(dim=3, n=n, h=h, N=N, hd=float(np.prod(h)),
               dt=float(cfg.dt), nt=int(cfg.nt), sigma=float(cfg.sigma),
               alpha=float(cfg.alpha), beta=float(cfg.beta),
               bc=cfg.bc, Xc=Xc, Yc=Yc, Zc=Zc, Grad=Grad,
               Bsolve=_DiffusionSolver(n, h, cfg.dt, cfg.sigma),
               niter_pcg=int(cfg.niter_pcg),
               maxUiter=int(cfg.maxUiter),
               gnLambda0=float(getattr(cfg, "gnLambda0", 0.1)),
               chi=_chiToArray(getattr(cfg, "chi", None), N, int(cfg.nt)))
    return par


def _chiToArray(chi, N, nt):
    """Normalize an optional source-indicator (K) into an (N, nt) array or
    ``None``. Accepts a scalar, an (N,) spatial map (broadcast over time), a
    flat (N*nt,) vector, or an (N, nt) array (Fortran-flattened convention)."""
    if chi is None:
        return None
    chi = np.asarray(chi, dtype=np.float64)
    if chi.ndim == 0:
        return np.full((N, nt), float(chi))
    chi = chi.ravel(order="F")
    if chi.size == N:
        return np.repeat(chi[:, None], nt, axis=1)
    if chi.size == N * nt:
        return chi.reshape(N, nt, order="F")
    raise ValueError("chi must be scalar, (N,), (N*nt,) or (N, nt); got size %d"
                     % chi.size)


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


def _trilinearApply(par, posX, posY, posZ, field):
    """Matrix-free forward trilinear interpolation: returns ``S(pos) @ field``
    without assembling the sparse matrix S (it gathers ``field`` at the eight
    surrounding cell centers and weight-sums). Equivalent to
    ``_trilinear(par, posX, posY, posZ)[0] @ field`` to machine precision but far
    cheaper, because the dominant cost of :func:`_trilinear` is the COO->CSR
    assembly - which (post-DCT) dominated the forward-only line-search
    evaluations. Used only where the explicit S / its transpose / its spatial
    derivative are *not* needed (the gradient and sensitivity paths still build
    the full matrices)."""
    n, h = par["n"], par["h"]
    n1, n2, n3 = n
    gx = np.clip(posX / h[0] - 0.5, 0, n1 - 1)
    gy = np.clip(posY / h[1] - 0.5, 0, n2 - 1)
    gz = np.clip(posZ / h[2] - 0.5, 0, n3 - 1)
    i0 = np.clip(np.floor(gx).astype(int), 0, n1 - 2)
    j0 = np.clip(np.floor(gy).astype(int), 0, n2 - 2)
    k0 = np.clip(np.floor(gz).astype(int), 0, n3 - 2)
    fx, fy, fz = gx - i0, gy - j0, gz - k0
    base = i0 + n1 * j0 + n1 * n2 * k0
    out = np.zeros(par["N"])
    for a in (0, 1):
        wx = fx if a else (1 - fx)
        for b in (0, 1):
            wy = fy if b else (1 - fy)
            for c in (0, 1):
                wz = fz if c else (1 - fz)
                col = base + a + n1 * b + n1 * n2 * c
                out += (wx * wy * wz) * field[col]
    return out


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
    Returns rho (N x nt) for steps 1..nt.

    When ``interp`` is not supplied (the forward-only line-search path) the
    advection uses the matrix-free :func:`_trilinearApply` instead of assembling
    the per-step sparse interpolation matrices, which is several times faster now
    that the diffusion solve is cheap. When ``interp`` is supplied (gradient /
    sensitivity paths, which already need the matrices) the explicit ``S @ m`` is
    used - results are identical to machine precision either way."""
    N, nt, dt = par["N"], par["nt"], par["dt"]
    U = u.reshape(3 * N, nt, order="F")
    r = r.reshape(N, nt, order="F")
    chi = par.get("chi")
    Bsolve = par["Bsolve"]
    rho = np.zeros((N, nt))
    prev = rho0
    if interp is None:                            # matrix-free forward advection
        Xc, Yc, Zc = par["Xc"], par["Yc"], par["Zc"]
        for k in range(nt):
            ck = 1.0 if chi is None else chi[:, k]
            m = (1.0 + dt * r[:, k] * ck) * prev
            adv = _trilinearApply(par, Xc + dt * U[0:N, k], Yc + dt * U[N:2 * N, k],
                                  Zc + dt * U[2 * N:3 * N, k], m)
            prev = Bsolve(adv)
            rho[:, k] = prev
        return rho
    for k in range(nt):
        ck = 1.0 if chi is None else chi[:, k]
        m = (1.0 + dt * r[:, k] * ck) * prev      # source (r scaled by chi)
        S, _ = interp[k]
        adv = S @ m                               # advection
        cur = Bsolve(adv)                         # diffusion (B \ adv)
        rho[:, k] = cur
        prev = cur
    return rho


# --------------------------------------------------------------------------- #
#  Sensitivities of the final density (Sensitivities/ folder analog)
#
#  The forward map rho_N(u, r) is the composition over nt steps of
#      m_k   = (1 + dt*r_k*chi_k) .* rho_{k-1}        (source)
#      a_k   = S_k(u_k) @ m_k                         (advection)
#      rho_k = B \ a_k                                (diffusion)
#  forwardSensitivity is the tangent-linear model J=d rho_traj/d(u,r); adjoint
#  Sensitivity is J' applied to a terminal cotangent on rho_N. Together they are
#  the matrix-free pieces the Gauss-Newton Hessian needs (get_drNduT, get_drNdrT,
#  ...). Validated by a dot-product (adjoint) test and finite differences.
# --------------------------------------------------------------------------- #
def precomputeSensDeriv(rho0, r, par, interp, rho):
    """Per-step trilinear spatial derivatives ``dS = d(S@m)/dpos`` of the source
    field ``m = (1+dt*r*chi)*rho_{k-1}``. These are constant across all CG
    matvecs within one Gauss-Newton outer step (they depend only on the fixed
    ``rho``/``r``), so precomputing them once avoids recomputing the trilinear
    derivative in every :func:`forwardSensitivity` / :func:`adjointSensitivity`
    call. Returns a list of (3, N) arrays."""
    N, nt, dt = par["N"], par["nt"], par["dt"]
    R = r.reshape(N, nt, order="F")
    chi = par.get("chi")
    rhoPrev = np.concatenate([rho0[:, None], rho[:, :-1]], axis=1)
    out = []
    for k in range(nt):
        ck = 1.0 if chi is None else chi[:, k]
        m = (1.0 + dt * R[:, k] * ck) * rhoPrev[:, k]
        out.append(interp[k][1](m))                   # deriv(m) -> (3,N)
    return out


def forwardSensitivity(rho0, u, r, du, dr, par, interp=None, rho=None,
                       dSlist=None):
    """Tangent-linear model: directional derivative of the density trajectory
    w.r.t. (u, r) in the direction (du, dr). Returns drho (N x nt); the last
    column is ``J @ (du, dr)`` (perturbation of the final density). ``dSlist``
    (from :func:`precomputeSensDeriv`) reuses the per-step spatial derivatives."""
    N, nt, dt = par["N"], par["nt"], par["dt"]
    U = u.reshape(3 * N, nt, order="F")
    R = r.reshape(N, nt, order="F")
    dU = du.reshape(3 * N, nt, order="F")
    dR = dr.reshape(N, nt, order="F")
    chi = par.get("chi")
    if interp is None:
        interp = _interpMats(par, U)
    if rho is None:
        rho = sourceAdvecDiff(rho0, u, r, par, interp)
    rhoPrev = np.concatenate([rho0[:, None], rho[:, :-1]], axis=1)
    Bsolve = par["Bsolve"]
    drho = np.zeros((N, nt))
    dprev = np.zeros(N)
    for k in range(nt):
        ck = 1.0 if chi is None else chi[:, k]
        fac = 1.0 + dt * R[:, k] * ck
        prev = rhoPrev[:, k]
        dm = dt * ck * prev * dR[:, k] + fac * dprev      # d(source)
        S, deriv = interp[k]
        dS = dSlist[k] if dSlist is not None else deriv(fac * prev)
        da = S @ dm
        for d in range(3):
            da = da + dt * dS[d] * dU[d * N:(d + 1) * N, k]
        dcur = Bsolve(da)                                 # d(diffusion)
        drho[:, k] = dcur
        dprev = dcur
    return drho


def adjointSensitivity(rho0, u, r, wN, par, interp=None, rho=None, dSlist=None):
    """Adjoint of :func:`forwardSensitivity` for a terminal cotangent ``wN`` on
    the final density: returns (Ju, Jr) = J' @ wN in Fortran-flattened layout.
    (No regularization terms - this is purely the forward-map Jacobian.)
    ``dSlist`` (from :func:`precomputeSensDeriv`) reuses per-step derivatives."""
    N, nt, dt = par["N"], par["nt"], par["dt"]
    U = u.reshape(3 * N, nt, order="F")
    R = r.reshape(N, nt, order="F")
    chi = par.get("chi")
    if interp is None:
        interp = _interpMats(par, U)
    if rho is None:
        rho = sourceAdvecDiff(rho0, u, r, par, interp)
    rhoPrev = np.concatenate([rho0[:, None], rho[:, :-1]], axis=1)
    Bsolve = par["Bsolve"]
    gU = np.zeros((3 * N, nt))
    gR = np.zeros((N, nt))
    carry = np.zeros(N)
    for k in range(nt - 1, -1, -1):
        ck = 1.0 if chi is None else chi[:, k]
        fac = 1.0 + dt * R[:, k] * ck
        lam = (wN if k == nt - 1 else 0.0) + carry
        b = Bsolve(lam)                                   # adjoint of diffusion
        S, deriv = interp[k]
        prev = rhoPrev[:, k]
        mbar = S.T @ b
        dS = dSlist[k] if dSlist is not None else deriv(fac * prev)
        for d in range(3):
            gU[d * N:(d + 1) * N, k] = dt * b * dS[d]      # adjoint to u
        gR[:, k] = ck * dt * prev * mbar                   # adjoint to r
        carry = fac * mbar                                 # adjoint to rho_{k-1}
    return gU.ravel(order="F"), gR.ravel(order="F")


# --------------------------------------------------------------------------- #
#  Objective (get_Gamma.m) and adjoint gradient
# --------------------------------------------------------------------------- #
def getGamma(rho0, u, r, par, drhoN, interp=None):
    """Cost Gamma = Gamma1(kinetic) + alpha*Gamma2(source) + beta*Gamma3(fit).
    Returns (Gamma, (Gamma1, Gamma2, Gamma3), rho)."""
    N, nt, dt, hd = par["N"], par["nt"], par["dt"], par["hd"]
    U = u.reshape(3 * N, nt, order="F")
    R = r.reshape(N, nt, order="F")
    chi = par.get("chi")
    rho = sourceAdvecDiff(rho0, u, r, par, interp)
    uSq = U[0:N, :] ** 2 + U[N:2 * N, :] ** 2 + U[2 * N:3 * N, :] ** 2
    rSq = R ** 2 if chi is None else (R ** 2) * chi
    Gamma1 = hd * dt * float(np.sum(rho * uSq))
    Gamma2 = hd * dt * float(np.sum(rho * rSq))
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
    chi = par.get("chi")
    rho = sourceAdvecDiff(rho0, u, r, par, interp)
    rhoPrev = np.concatenate([rho0[:, None], rho[:, :-1]], axis=1)  # rho_{j-1}
    Bsolve = par["Bsolve"]

    gU = np.zeros((3 * N, nt))
    gR = np.zeros((N, nt))
    carry = np.zeros(N)
    for k in range(nt - 1, -1, -1):           # step k produces rho[:,k]
        ck = 1.0 if chi is None else chi[:, k]
        uSq_k = U[0:N, k] ** 2 + U[N:2 * N, k] ** 2 + U[2 * N:3 * N, k] ** 2
        explicit = hd * dt * (uSq_k + alpha * R[:, k] ** 2 * ck)
        if k == nt - 1:
            explicit = explicit + 2.0 * beta * hd * (rho[:, -1] - drhoN)
        lam = explicit + carry
        b = Bsolve(lam)                       # adjoint through diffusion (B sym)
        S, deriv = interp[k]
        prev = rhoPrev[:, k]
        m = (1.0 + dt * R[:, k] * ck) * prev  # source field at step k (chi-scaled)
        mbar = S.T @ b                        # adjoint to m
        # velocity gradient: direct kinetic + implicit (advection derivative)
        dS = deriv(m)                         # (3, N) = d(S@m)/d{x,y,z}
        for d in range(3):
            gU[d * N:(d + 1) * N, k] = (2.0 * hd * dt * rho[:, k] * U[d * N:(d + 1) * N, k]
                                        + dt * b * dS[d])
        # source gradient: direct source penalty + implicit (m wrt r), chi-scaled
        gR[:, k] = ck * (2.0 * hd * dt * alpha * rho[:, k] * R[:, k]
                         + dt * prev * mbar)
        # propagate adjoint to rho_{k-1}
        carry = (1.0 + dt * R[:, k] * ck) * mbar
    return gU.ravel(order="F"), gR.ravel(order="F")
