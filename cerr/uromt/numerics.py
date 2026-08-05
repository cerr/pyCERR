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
from scipy.fft import dct, dctn, idct, idctn

from cerr.uromt import kernels


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

    **Threading.** A 3-D DCT is three sets of *independent* 1-D transforms, so
    the work can be split along an axis that is not currently being transformed
    and run on a thread pool (scipy.fft releases the GIL). Chunking is bit-exact
    - each 1-D transform sees byte-identical input - and only splitting dctn's
    axis list perturbs the result, at ~5e-16. scipy's own ``workers`` argument
    measurably does nothing here, hence the explicit pool. Set the pool size
    with the ``numThreads`` setting; ~4 chunks is the optimum (the transform is
    memory-bandwidth bound well before it runs out of cores).

    """

    _MAX_CHUNK = 4          # measured optimum; more chunks lose to overhead
    # Dispatching the pool costs a fixed ~1.9 ms, so threading only pays on a
    # large enough grid. Measured gain vs the serial solve: 0.25x at 14k voxels,
    # 0.89x at 74k, 1.6x at 184k, 2.6x at 166k on a prime-sized grid.
    _MIN_THREADED_N = 100000

    def __init__(self, n, h, dt, sigma, threads=0):
        self.n = tuple(int(v) for v in n)
        eig = []
        for ni, hi in zip(self.n, (float(v) for v in h)):
            k = np.arange(ni)
            eig.append((2.0 - 2.0 * np.cos(np.pi * k / ni)) / (hi * hi))
        self.eig = (1.0 + float(dt) * float(sigma) *
                    (eig[0][:, None, None] + eig[1][None, :, None]
                     + eig[2][None, None, :]))
        # multiply by the reciprocal in place instead of dividing
        self.invEig = 1.0 / self.eig
        self.nchunk = self._chunks(threads)
        self._pool = None

    def _chunks(self, threads):
        """Chunk count: capped by the thread budget, the grid size, and the
        axes we split. 1 means run serially."""
        if int(np.prod(self.n)) < self._MIN_THREADED_N:
            return 1
        want = int(threads or 0)
        if want <= 0:
            want = kernels.setNumThreads(0)
        # each chunk must own at least 2 planes of both split axes
        room = min(self.n[0], self.n[2]) // 2
        return max(1, min(want, self._MAX_CHUNK, room))

    @property
    def pool(self):
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(self.nchunk)
        return self._pool

    def __call__(self, x):
        X = np.asarray(x, dtype=np.float64).reshape(self.n, order="F")
        if self.nchunk > 1:
            Y = self._threaded(X)
        else:
            # `X` may be a view of the caller's array, so the forward transform
            # must not overwrite it; the spectral buffer is ours, so scale it in
            # place and let the inverse transform consume it.
            Xh = dctn(X, type=2, norm="ortho")
            Xh *= self.invEig
            Y = idctn(Xh, type=2, norm="ortho", overwrite_x=True)
        return Y.ravel(order="F")

    def _threaded(self, X):
        nc = self.nchunk
        h = np.empty(self.n, dtype=np.float64, order="F")
        zb = np.linspace(0, self.n[2], nc + 1).astype(int)
        xb = np.linspace(0, self.n[0], nc + 1).astype(int)

        def fwd01(i):                       # axes (0,1), split along axis 2
            s = slice(zb[i], zb[i + 1])
            h[:, :, s] = dctn(X[:, :, s], type=2, axes=(0, 1), norm="ortho")

        def mid2(i):                        # axis 2 + the spectral scaling
            s = slice(xb[i], xb[i + 1])
            blk = dct(h[s], type=2, axis=2, norm="ortho", overwrite_x=True)
            blk *= self.invEig[s]
            h[s] = idct(blk, type=2, axis=2, norm="ortho", overwrite_x=True)

        def inv01(i):                       # inverse axes (0,1)
            s = slice(zb[i], zb[i + 1])
            h[:, :, s] = idctn(h[:, :, s], type=2, axes=(0, 1), norm="ortho",
                               overwrite_x=True)

        for stage in (fwd01, mid2, inv01):
            list(self.pool.map(stage, range(nc)))
        return h


def paramInit(cfg):
    """Build the ``par`` dictionary (paramInitFunc.m): grid, operators, the
    implicit-diffusion solver for B = I + dt*sigma*Grad'Grad, and parameters.

    Optional ``cfg.chi`` (the MATLAB source-indicator ``K``) is stored as an
    (N, nt) array; it elementwise-scales the relative source ``r`` in the
    forward model and the source penalty. ``None`` -> identity (K = 1)."""
    from cerr.uromt.gpu import Backend           # local: optional cupy import
    from cerr.uromt.config import getNumThreads, getUseGPU
    threads = kernels.setNumThreads(getNumThreads(cfg))
    bk = Backend(getUseGPU(cfg))
    xp = bk.xp
    n = [int(v) for v in cfg.trueSize]
    h = [float(v) for v in cfg.spacing]
    N = int(np.prod(n))
    Grad = neumannGrad(n, h)
    # diag(Grad'Grad) = per-voxel column sum of Grad**2 (the Neumann-Laplacian
    # diagonal). Precomputed for the GN Hessian's Jacobi preconditioner when the
    # velocity H1-smoothness regularizer (eta > 0) is active.
    lapDiag = np.asarray(Grad.multiply(Grad).sum(axis=0)).ravel()
    Xc, Yc, Zc = cellCenteredGrid(n, h)
    chi = _chiToArray(getattr(cfg, "chi", None), N, int(cfg.nt))
    if bk.isGpu:                                 # keep the solve device-resident
        Xc, Yc, Zc = (xp.asarray(v) for v in (Xc, Yc, Zc))
        lapDiag = xp.asarray(lapDiag)
        if chi is not None:
            chi = xp.asarray(chi)
        if float(getattr(cfg, "eta", 0.0)):
            import cupyx.scipy.sparse as cusp
            Grad = cusp.csr_matrix(Grad)
    par = dict(dim=3, n=n, h=h, N=N, hd=float(np.prod(h)), bk=bk, xp=xp,
               threads=threads,
               dt=float(cfg.dt), nt=int(cfg.nt), sigma=float(cfg.sigma),
               alpha=float(cfg.alpha), beta=float(cfg.beta),
               eta=float(getattr(cfg, "eta", 0.0)),
               bc=cfg.bc, Xc=Xc, Yc=Yc, Zc=Zc, Grad=Grad, lapDiag=lapDiag,
               Bsolve=bk.diffusionSolver(n, h, cfg.dt, cfg.sigma,
                                         threads=threads),
               niter_pcg=int(cfg.niter_pcg),
               maxUiter=int(cfg.maxUiter),
               gnLambda0=float(getattr(cfg, "gnLambda0", 0.0)),
               gnPrecond=int(getattr(cfg, "gnPrecond", 0) or 0),
               gnCgTol=float(getattr(cfg, "gnCgTol", 1e-2)),
               chi=chi)
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
def _cornerIndex(par, posX, posY, posZ):
    """Shared setup for the trilinear operators.

    Returns ``(cols, wx, wy, wz)`` where ``cols`` is the list of eight corner
    column indices ordered ``(a, b, c) -> a*4 + b*2 + c`` (c = the slowest grid
    axis varies fastest in the list), and ``wx``/``wy``/``wz`` are the
    ``(1 - frac, frac)`` interpolation factors along each axis.
    """
    n, h = par["n"], par["h"]
    n1, n2, n3 = n
    npt = int(np.size(posX))
    cols = np.empty((8, npt), dtype=np.int64)
    ex, fx, ey, fy, ez, fz = (np.empty(npt) for _ in range(6))
    kernels.cornerIndexKernel(np.ascontiguousarray(posX, dtype=np.float64),
                              np.ascontiguousarray(posY, dtype=np.float64),
                              np.ascontiguousarray(posZ, dtype=np.float64),
                              float(h[0]), float(h[1]), float(h[2]),
                              int(n1), int(n2), int(n3),
                              cols, ex, fx, ey, fy, ez, fz)
    return cols, (ex, fx), (ey, fy), (ez, fz)


class _TrilinearOp:
    """The four trilinear operators at one set of departure points.

    Wraps the fused numba kernels in :mod:`cerr.uromt.kernels`. ``S`` is never
    assembled: :meth:`interp` is the gather ``S @ f`` and :meth:`interpT` the
    scatter ``S' @ f``, which is what the advection actually needs. That removes
    the per-outer-step COO->CSR assembly of ``nt`` sparse matrices entirely.

    The scatter accumulator is shared (``par['acc']``) rather than held per
    step - only one operator is ever mid-scatter at a time. That makes a single
    ``par`` unsafe to share between *threads*; parallelism across intervals must
    use processes.
    """

    __slots__ = ("N", "cols", "ex", "fx", "ey", "fy", "ez", "fz",
                 "ihx", "ihy", "ihz", "acc")

    def __init__(self, par, posX, posY, posZ):
        h = par["h"]
        self.N = int(par["N"])
        self.cols, wx, wy, wz = _cornerIndex(par, posX, posY, posZ)
        self.ex, self.fx = wx
        self.ey, self.fy = wy
        self.ez, self.fz = wz
        self.ihx, self.ihy, self.ihz = 1.0 / h[0], 1.0 / h[1], 1.0 / h[2]
        self.acc = accumBuffer(par)

    def _w(self):
        return (self.cols, self.ex, self.fx, self.ey, self.fy, self.ez, self.fz)

    def interp(self, field):
        """``S @ field`` - trilinear sample at the departure points."""
        out = np.empty(self.N)
        kernels.interpKernel(np.ascontiguousarray(field), *self._w(), out)
        return out

    def interpT(self, field):
        """``S' @ field`` - the mass-conserving push-forward scatter."""
        out = np.empty(self.N)
        kernels.interpTKernel(np.ascontiguousarray(field), *self._w(),
                              self.acc, out)
        return out

    def deriv(self, field):
        """``D_d @ field`` for d = x,y,z, i.e. d(S@field)/d{x,y,z}. (3, N)."""
        out = np.empty((3, self.N))
        kernels.derivKernel(np.ascontiguousarray(field), *self._w(),
                            self.ihx, self.ihy, self.ihz, out)
        return out

    def derivT(self, vecs):
        """``sum_d D_d' @ vecs[d]`` - the adjoint of :meth:`deriv`.

        Needed by the tangent-linear model of the push-forward advection
        ``a = S' @ m``: perturbing the departure points gives
        ``delta(S'm) = sum_d D_d' (m .* delta pos_d)``."""
        v = np.ascontiguousarray(vecs, dtype=np.float64)
        out = np.empty(self.N)
        kernels.derivTKernel(v, *self._w(), self.ihx, self.ihy, self.ihz,
                             self.acc, out)
        return out

    def tangentStep(self, srcR, dRk, fac, dprev, m, dU, k, dt, N):
        """One fused tangent-linear advection step: ``S'@dm + dt*D'(m .* du)``
        with ``dm = srcR*dRk + fac*dprev``. See
        :func:`cerr.uromt.kernels.tangentStepKernel`."""
        out = np.empty(self.N)
        kernels.tangentStepKernel(
            srcR, dRk, fac, dprev, m,
            dU[0:N, k], dU[N:2 * N, k], dU[2 * N:3 * N, k], dt,
            *self._w(), self.ihx, self.ihy, self.ihz, self.acc, out)
        return out

    def adjointStep(self, b, m, srcR, fac, dt, gUk, gRk, carry):
        """One fused adjoint sensitivity step, writing the velocity/source
        adjoints and the carry in place. See
        :func:`cerr.uromt.kernels.adjointStepKernel`."""
        kernels.adjointStepKernel(np.ascontiguousarray(b), m, srcR, fac, dt,
                                  *self._w(), self.ihx, self.ihy, self.ihz,
                                  gUk, gRk, carry)

    def toSparse(self):
        """Assemble S as an explicit sparse matrix (reference / tests only)."""
        N = self.N
        wx, wy, wz = (self.ex, self.fx), (self.ey, self.fy), (self.ez, self.fz)
        data = np.concatenate([wx[a] * wy[b] * wz[c]
                               for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        rr = np.tile(np.arange(N), 8)
        return sp.csr_matrix((data, (rr, self.cols.ravel())), shape=(N, N))


def mkOp(par, posX, posY, posZ):
    """Build the trilinear operators for the backend ``par`` was created with
    (the numba CPU kernels, or the cupy device implementation)."""
    bk = par.get("bk")
    if bk is None:
        return _TrilinearOp(par, posX, posY, posZ)
    return bk.trilinearOp(par, posX, posY, posZ)


def accumBuffer(par):
    """The shared scatter accumulator for this ``par`` (created on demand)."""
    acc = par.get("acc")
    if acc is None or acc.shape[1] != int(par["N"]):
        acc = kernels.accumBuffer(int(par["N"]))
        par["acc"] = acc
    return acc


def _trilinear(par, posX, posY, posZ):
    """The trilinear operators at (posX, posY, posZ), as a :class:`_TrilinearOp`.

    ``op.toSparse()`` gives the explicit (N x N) interpolation matrix S, whose
    row sums are 1 (closed/clamped boundaries); the solver never needs it.
    """
    return mkOp(par, posX, posY, posZ)


def _trilinearApplyT(par, posX, posY, posZ, field):
    """Matrix-free **push-forward** advection: returns ``S(pos)' @ field``
    without assembling S. Each point scatters its mass onto the eight
    surrounding cell centers with the trilinear weights (the transpose of the
    gather that computes ``S @ field``).

    This is the mass-conserving form used by the MATLAB ``SourceAdvecDiff.m``:
    S has unit row sums, so ``1'S'm = (S1)'m = 1'm`` - total mass is preserved
    exactly, which ``S @ m`` does *not* do. Equivalent to
    ``_trilinear(par, posX, posY, posZ).toSparse().T @ field`` to machine
    precision."""
    return mkOp(par, posX, posY, posZ).interpT(field)


def _interpMats(par, u):
    """Per-step :class:`_TrilinearOp` at the departure points of velocity ``u``
    (reshaped (3N, nt))."""
    N, nt, dt = par["N"], par["nt"], par["dt"]
    Xc, Yc, Zc = par["Xc"], par["Yc"], par["Zc"]
    out = []
    for k in range(nt):
        U1 = u[0:N, k]
        U2 = u[N:2 * N, k]
        U3 = u[2 * N:3 * N, k]
        out.append(mkOp(par, Xc + dt * U1, Yc + dt * U2,
                        Zc + dt * U3))
    return out


# --------------------------------------------------------------------------- #
#  Forward model (SourceAdvecDiff.m)
# --------------------------------------------------------------------------- #
def sourceAdvecDiff(rho0, u, r, par, interp=None):
    """Evolve rho through nt source -> advection -> diffusion steps.
    Returns rho (N x nt) for steps 1..nt.

    The advection is the **push-forward** ``a = S' @ m`` (mass-conserving), as in
    the MATLAB ``SourceAdvecDiff.m`` - *not* the semi-Lagrangian pull-back
    ``S @ m``. S has unit row sums, so ``S'`` conserves total mass exactly while
    ``S`` does not; using ``S`` makes the model unable to reproduce growing
    density (e.g. contrast uptake), which the optimizer then compensates for with
    spuriously large velocities and a near-zero source.

    ``interp`` (from :func:`_interpMats`) is reused when the caller already has
    the per-step operators; otherwise they are built on the fly, which is cheap
    now that nothing is assembled as a sparse matrix."""
    N, nt, dt = par["N"], par["nt"], par["dt"]
    U = u.reshape(3 * N, nt, order="F")
    r = r.reshape(N, nt, order="F")
    chi = par.get("chi")
    xp = par.get("xp", np)
    Bsolve = par["Bsolve"]
    Xc, Yc, Zc = par["Xc"], par["Yc"], par["Zc"]
    rho = xp.zeros((N, nt))
    prev = rho0
    for k in range(nt):
        ck = 1.0 if chi is None else chi[:, k]
        m = (1.0 + dt * r[:, k] * ck) * prev      # source (r scaled by chi)
        op = interp[k] if interp is not None else mkOp(
            par, Xc + dt * U[0:N, k], Yc + dt * U[N:2 * N, k],
            Zc + dt * U[2 * N:3 * N, k])
        cur = Bsolve(op.interpT(m))               # advection then diffusion
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
    """Per-step quantities that every CG matvec of one Gauss-Newton step reuses.

    With the push-forward advection ``a = S'@m`` both the tangent and the adjoint
    velocity terms are weighted by ``m = (1+dt*r*chi)*rho_{k-1}`` (the tangent
    needs ``D_d'(m .* du_d)``, the adjoint needs ``m .* (D_d @ b)``), and ``m``
    is constant across all CG matvecs of one Gauss-Newton step (it depends only
    on the fixed ``rho``/``r``). The source factor ``fac = 1+dt*r*chi`` and the
    ``r``-sensitivity weight ``srcR = dt*chi*rho_{k-1}`` are likewise fixed, so
    they are cached here rather than rebuilt per matvec.

    Returns a dict of per-step lists: ``{"m", "fac", "srcR"}``.
    """
    N, nt, dt = par["N"], par["nt"], par["dt"]
    R = r.reshape(N, nt, order="F")
    chi = par.get("chi")
    xp = par.get("xp", np)
    rhoPrev = xp.concatenate([rho0[:, None], rho[:, :-1]], axis=1)
    m, fac, srcR = [], [], []
    for k in range(nt):
        ck = 1.0 if chi is None else chi[:, k]
        prev = rhoPrev[:, k]
        fk = 1.0 + dt * R[:, k] * ck
        fac.append(fk)
        m.append(fk * prev)
        srcR.append(dt * ck * prev)
    return dict(m=m, fac=fac, srcR=srcR)


def forwardSensitivity(rho0, u, r, du, dr, par, interp=None, rho=None,
                       dSlist=None):
    """Tangent-linear model: directional derivative of the density trajectory
    w.r.t. (u, r) in the direction (du, dr). Returns drho (N x nt); the last
    column is ``J @ (du, dr)`` (perturbation of the final density). ``dSlist``
    (from :func:`precomputeSensDeriv`) reuses the per-step spatial derivatives."""
    N, nt, dt = par["N"], par["nt"], par["dt"]
    U = u.reshape(3 * N, nt, order="F")
    dU = du.reshape(3 * N, nt, order="F")
    dR = dr.reshape(N, nt, order="F")
    if interp is None:
        interp = _interpMats(par, U)
    if rho is None:
        rho = sourceAdvecDiff(rho0, u, r, par, interp)
    if dSlist is None:
        dSlist = precomputeSensDeriv(rho0, r, par, interp, rho)
    mList, facList, srcRList = dSlist["m"], dSlist["fac"], dSlist["srcR"]
    xp = par.get("xp", np)
    Bsolve = par["Bsolve"]
    drho = xp.zeros((N, nt))
    dprev = xp.zeros(N)
    fused = hasattr(interp[0], "tangentStep")
    for k in range(nt):
        op = interp[k]
        m = mList[k]
        # push-forward: a = S'@m, so
        #   da = S'@dm + sum_d D_d' (m .* dt*du_d),  dm = srcR*dr + fac*dprev
        if fused:
            da = op.tangentStep(srcRList[k], dR[:, k], facList[k], dprev, m,
                                dU, k, dt, N)
        else:
            dm = srcRList[k] * dR[:, k] + facList[k] * dprev
            mdu = xp.empty((3, N))
            for d in range(3):
                xp.multiply(m, dU[d * N:(d + 1) * N, k], out=mdu[d])
            da = op.interpT(dm) + dt * op.derivT(mdu)
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
    if interp is None:
        interp = _interpMats(par, U)
    if rho is None:
        rho = sourceAdvecDiff(rho0, u, r, par, interp)
    if dSlist is None:
        dSlist = precomputeSensDeriv(rho0, r, par, interp, rho)
    mList, facList, srcRList = dSlist["m"], dSlist["fac"], dSlist["srcR"]
    xp = par.get("xp", np)
    Bsolve = par["Bsolve"]
    # Fortran-ordered so that the `ravel(order="F")` below is a view, not a
    # 3*N*nt copy (~40 MB per CG matvec at a realistic ROI size).
    gU = xp.zeros((3 * N, nt), order="F")
    gR = xp.zeros((N, nt), order="F")
    carry = xp.zeros(N)
    fused = hasattr(interp[0], "adjointStep")
    for k in range(nt - 1, -1, -1):
        lam = (wN if k == nt - 1 else 0.0) + carry
        b = Bsolve(lam)                                   # adjoint of diffusion
        op = interp[k]
        m = mList[k]
        # push-forward a = S'@m: adjoint to m is S@b (not S'@b), and the
        # velocity term is weighted by m, with the derivative taken on b.
        if fused:
            # `carry` is read into `lam` above before being overwritten here,
            # so writing it in place is safe.
            op.adjointStep(b, m, srcRList[k], facList[k], dt,
                           gU[:, k], gR[:, k], carry)
        else:
            mbar = op.interp(b)
            dB = op.deriv(b)                               # (3, N) = D_d @ b
            for d in range(3):
                gU[d * N:(d + 1) * N, k] = dt * m * dB[d]  # adjoint to u
            gR[:, k] = srcRList[k] * mbar                  # adjoint to r
            carry = facList[k] * mbar                      # adjoint to rho_{k-1}
    return gU.ravel(order="F"), gR.ravel(order="F")


# --------------------------------------------------------------------------- #
#  Objective (get_Gamma.m) and adjoint gradient
# --------------------------------------------------------------------------- #
def getGamma(rho0, u, r, par, drhoN, interp=None, rho=None):
    """Cost Gamma = Gamma1(kinetic) + alpha*Gamma2(source) + beta*Gamma3(fit)
    + Gamma4(velocity H1-smoothness).
    Returns (Gamma, (Gamma1, Gamma2, Gamma3, Gamma4), rho).

    ``Gamma4 = eta*hd*dt * sum_k sum_d |Grad u_d(:,k)|^2`` penalizes the spatial
    gradient of each velocity component (the H1 seminorm). It is an optional
    pyCERR extension with no counterpart in the reference MATLAB implementation,
    and is **off by default** (``eta = 0``), which recovers the reference
    objective exactly. The urOMT misfit constrains velocity only where density
    flows, so it is formally under-determined in low/flat-density voxels, but in
    practice the recovered field is coherent once the advection is
    mass-conserving; enable this only to deliberately bias the velocity toward
    smoothness (e.g. for visualization).

    ``rho`` may be supplied when the caller has already evolved the density at
    this (u, r) - the forward solve is then skipped.
    """
    N, nt, dt, hd = par["N"], par["nt"], par["dt"], par["hd"]
    U = u.reshape(3 * N, nt, order="F")
    R = r.reshape(N, nt, order="F")
    chi = par.get("chi")
    xp = par.get("xp", np)
    if rho is None:
        rho = sourceAdvecDiff(rho0, u, r, par, interp)
    uSq = U[0:N, :] ** 2 + U[N:2 * N, :] ** 2 + U[2 * N:3 * N, :] ** 2
    rSq = R ** 2 if chi is None else (R ** 2) * chi
    Gamma1 = hd * dt * float(xp.sum(rho * uSq))
    Gamma2 = hd * dt * float(xp.sum(rho * rSq))
    Gamma3 = hd * float(xp.sum((rho[:, -1] - drhoN) ** 2))
    eta = float(par.get("eta", 0.0))
    Gamma4 = 0.0
    if eta:
        Grad = par["Grad"]
        s = 0.0
        for d in range(3):
            GU = Grad @ U[d * N:(d + 1) * N, :]   # (3N, nt) directional derivs
            s += float(xp.sum(GU * GU))
        Gamma4 = eta * hd * dt * s
    Gamma = Gamma1 + par["alpha"] * Gamma2 + par["beta"] * Gamma3 + Gamma4
    return Gamma, (Gamma1, Gamma2, Gamma3, Gamma4), rho


def gradGamma(rho0, u, r, par, drhoN, interp=None, rho=None):
    """Analytic adjoint gradient of getGamma w.r.t. (u, r).
    Returns (g_u (3N*nt,), g_r (N*nt,)) in Fortran-flattened layout.

    ``rho`` may be supplied when the caller has already evolved the density at
    this (u, r) - the forward solve is then skipped."""
    N, nt, dt, hd = par["N"], par["nt"], par["dt"], par["hd"]
    alpha, beta = par["alpha"], par["beta"]
    U = u.reshape(3 * N, nt, order="F")
    R = r.reshape(N, nt, order="F")
    if interp is None:
        interp = _interpMats(par, U)
    chi = par.get("chi")
    xp = par.get("xp", np)
    if rho is None:
        rho = sourceAdvecDiff(rho0, u, r, par, interp)
    rhoPrev = xp.concatenate([rho0[:, None], rho[:, :-1]], axis=1)  # rho_{j-1}
    Bsolve = par["Bsolve"]

    # Fortran-ordered so the final ravel(order="F") is a view, not a copy.
    gU = xp.zeros((3 * N, nt), order="F")
    gR = xp.zeros((N, nt), order="F")
    carry = xp.zeros(N)
    for k in range(nt - 1, -1, -1):           # step k produces rho[:,k]
        ck = 1.0 if chi is None else chi[:, k]
        uSq_k = U[0:N, k] ** 2 + U[N:2 * N, k] ** 2 + U[2 * N:3 * N, k] ** 2
        explicit = hd * dt * (uSq_k + alpha * R[:, k] ** 2 * ck)
        if k == nt - 1:
            explicit = explicit + 2.0 * beta * hd * (rho[:, -1] - drhoN)
        lam = explicit + carry
        b = Bsolve(lam)                       # adjoint through diffusion (B sym)
        op = interp[k]
        prev = rhoPrev[:, k]
        m = (1.0 + dt * R[:, k] * ck) * prev  # source field at step k (chi-scaled)
        mbar = op.interp(b)                   # adjoint to m (a = S'@m)
        # velocity gradient: direct kinetic + implicit (advection derivative).
        # For the push-forward a = S'@m the adjoint w.r.t. the departure point is
        # m .* (D_d @ b) -- the derivative acts on b and the weight is m (the
        # pull-back form had these swapped).
        dB = op.deriv(b)                      # (3, N) = D_d @ b
        for d in range(3):
            gU[d * N:(d + 1) * N, k] = (2.0 * hd * dt * rho[:, k] * U[d * N:(d + 1) * N, k]
                                        + dt * m * dB[d])
        # source gradient: direct source penalty + implicit (m wrt r), chi-scaled
        gR[:, k] = ck * (2.0 * hd * dt * alpha * rho[:, k] * R[:, k]
                         + dt * prev * mbar)
        # propagate adjoint to rho_{k-1}
        carry = (1.0 + dt * R[:, k] * ck) * mbar
    # velocity H1-smoothness gradient: d/du_d [eta*hd*dt*|Grad u_d|^2]
    #                                 = 2*eta*hd*dt * Grad' Grad u_d  (per comp/step)
    eta = float(par.get("eta", 0.0))
    if eta:
        Grad = par["Grad"]
        c = 2.0 * eta * hd * dt
        for d in range(3):
            gU[d * N:(d + 1) * N, :] += c * (Grad.T @ (Grad @ U[d * N:(d + 1) * N, :]))
    return gU.ravel(order="F"), gR.ravel(order="F")
