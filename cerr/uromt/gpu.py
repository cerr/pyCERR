"""Optional GPU (cupy) backend for the urOMT solver.

The urOMT Gauss-Newton solve is a good GPU fit: every inner operation is an
elementwise pass, a gather/scatter, or a 3-D DCT over a few hundred thousand
voxels, and the whole CG loop can stay resident on the device - there is one
host transfer per interval in and one out.

Requirements (all optional; pyCERR runs on the CPU without them)::

    pip install "pycerr[gpu]"

That pulls ``cupy-cuda13x`` plus the matching NVIDIA CUDA library wheels. Check
what your driver supports first - the wheel's CUDA major version must not exceed
it::

    python -c "import ctypes; v=ctypes.c_int(); ctypes.WinDLL('nvcuda.dll').cuDriverGetVersion(ctypes.byref(v)); print(v.value)"

10010 means CUDA 10.1, 13020 means 13.2, and so on. For a CUDA 12.x driver use
``cupy-cuda12x`` with the ``nvidia-*-cu12`` wheels; for 11.x, ``cupy-cuda11x``.

**The NVIDIA library wheels matter.** cupy locates its CUDA libraries through
``cuda-pathfinder``, which searches site-packages before ``CUDA_PATH``/``PATH``.
On a machine that also has an old CUDA toolkit installed, without those wheels
cupy happily loads the *stale* NVRTC and every kernel compile dies with
``nvrtc: error: invalid value for --std``. Installing them inside the
environment fixes it without touching system PATH. Note NVIDIA dropped the
version suffix at CUDA 13: the packages are ``nvidia-cuda-nvrtc``,
``nvidia-cufft``, ... (the ``-cu13`` names on PyPI are deprecated stubs that
fail to build).

Enable with the ``useGPU`` setting in the urOMT settings JSON, or
``buildConfig(..., useGPU='yes')``. If cupy or a usable device is missing, the solver
warns and runs on the CPU.

Implementation. The device operators are fused CUDA kernels
(:mod:`cerr.uromt.gpu_kernels`) mirroring the numba CPU kernels: the same
separable collapse (z, then y, then x), one launch per operator, and one launch
per *whole* sensitivity sub-step. Scatters use ``atomicAdd`` on doubles, native
from compute capability 6.0; on older devices the backend falls back to an
equivalent array-expression path built on ``cupy.add.at``. Because atomics do not
fix a summation order, repeated scatter results can differ at the ~1e-16 level.

**When the GPU is worth it.** Measured on a Quadro RTX 5000 (Turing, 16 GB)
against a 24-core CPU, nt=10, one Gauss-Newton block solve:

===============  ========  ===========  ===========  ==========
ROI              voxels    CPU matvec   GPU matvec   block solve
===============  ========  ===========  ===========  ==========
48 x 44 x 40      84,480    84.2 ms      43.4 ms      3.5x
64 x 60 x 48     184,320   161.3 ms      43.2 ms      6.8x
96 x 90 x 72     622,080   483.0 ms      60.5 ms     13.2x
===============  ========  ===========  ===========  ==========

The win grows with ROI size: GPU time is still far flatter than CPU time, so
small ROIs remain partly kernel-launch-latency bound. Fusing the operators
roughly doubled these figures - before it, the same three cases were 1.8x, 3.8x
and 7.6x.

**Status.** The device operators' *math* is tested on every machine: because
they are written against an array module plus a scatter-add, the test suite runs
them on a numpy shim (``np.add.at``) and checks them against the explicit sparse
matrix and the CPU kernels. ``test_gpu_backend_reproduces_the_cpu_solve``
additionally runs a full solve on real hardware (it skips without a device) and
has been verified to match the CPU optimum to ~1e-9, and
``test_gpu_fused_kernels_match_the_array_path`` checks the CUDA kernels against
the array path and the CPU kernels. ``cerr/scripts/bench_uromt_gpu.py``
produced the table above.
"""

import inspect
import warnings

from cerr.uromt import gpu_kernels as _gpuk


def _normalizeCg(cgFunc):
    """Adapt a CG implementation to one call signature.

    Both scipy and cupy renamed the relative-tolerance argument ``tol`` ->
    ``rtol`` (scipy 1.12, cupy 14), and older releases of either are still
    common. Dispatching on the actual signature rather than on the library keeps
    the solver working across versions of both, in whichever combination is
    installed.
    """
    params = inspect.signature(cgFunc).parameters
    key = "rtol" if "rtol" in params else "tol"

    def solve(A, b, rtol, maxiter, M=None):
        return cgFunc(A, b, maxiter=maxiter, M=M, **{key: rtol})

    solve.tolKeyword = key
    return solve


def isAvailable():
    """True when cupy is importable and a CUDA device is actually usable."""
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def asNumpy(a):
    """Bring an array back to the host (no-op for numpy arrays)."""
    try:
        import cupy
        if isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)
    except Exception:
        pass
    return a


class _DeviceDiffusionSolver:
    """``B \\ x`` for B = I + dt*sigma*Grad'Grad, via the 3-D DCT on the device.

    Same exact diagonalization as the CPU
    :class:`cerr.uromt.numerics._DiffusionSolver`; only the transform library
    differs.
    """

    def __init__(self, xp, fft, n, h, dt, sigma):
        self.xp = xp
        self.fft = fft
        self.n = tuple(int(v) for v in n)
        eig = []
        for ni, hi in zip(self.n, (float(v) for v in h)):
            k = xp.arange(ni)
            eig.append((2.0 - 2.0 * xp.cos(xp.pi * k / ni)) / (hi * hi))
        self.invEig = 1.0 / (1.0 + float(dt) * float(sigma) *
                             (eig[0][:, None, None] + eig[1][None, :, None]
                              + eig[2][None, None, :]))

    def __call__(self, x):
        X = self.xp.asarray(x, dtype=self.xp.float64).reshape(self.n,
                                                              order="F")
        Xh = self.fft.dctn(X, type=2, norm="ortho")
        Xh *= self.invEig
        Y = self.fft.idctn(Xh, type=2, norm="ortho", overwrite_x=True)
        return Y.ravel(order="F")


class _DeviceTrilinearOp:
    """Device counterpart of :class:`cerr.uromt.numerics._TrilinearOp`.

    Exposes the same four operators - ``interp`` (S @ f), ``interpT`` (S' @ f),
    ``deriv`` (d(S@f)/d{x,y,z}) and ``derivT`` (its adjoint) - evaluated
    separably so the eight corner weights never have to be materialized.
    """

    def __init__(self, backend, par, posX, posY, posZ):
        xp = backend.xp
        self.xp = xp
        self.scatterAdd = backend.scatterAdd
        n, h = par["n"], par["h"]
        n1, n2, n3 = (int(v) for v in n)
        self.N = int(par["N"])
        gx = xp.clip(xp.asarray(posX) / h[0] - 0.5, 0, n1 - 1)
        gy = xp.clip(xp.asarray(posY) / h[1] - 0.5, 0, n2 - 1)
        gz = xp.clip(xp.asarray(posZ) / h[2] - 0.5, 0, n3 - 1)
        i0 = xp.clip(xp.floor(gx).astype(xp.int64), 0, n1 - 2)
        j0 = xp.clip(xp.floor(gy).astype(xp.int64), 0, n2 - 2)
        k0 = xp.clip(xp.floor(gz).astype(xp.int64), 0, n3 - 2)
        fx, fy, fz = gx - i0, gy - j0, gz - k0
        base = i0 + n1 * j0 + n1 * n2 * k0
        # corner order (a, b, c) -> a*4 + b*2 + c, matching the CPU kernels.
        # Kept as one (8, N) C-contiguous block so the fused kernels can index
        # it as cols[j*N + i]; `self.cols` are row views onto it for the
        # array-expression path.
        self.colsArr = xp.ascontiguousarray(
            xp.stack([base + (a + n1 * b + n1 * n2 * c)
                      for a in (0, 1) for b in (0, 1) for c in (0, 1)]))
        self.cols = list(self.colsArr)
        self.ex, self.fx = 1.0 - fx, fx
        self.ey, self.fy = 1.0 - fy, fy
        self.ez, self.fz = 1.0 - fz, fz
        self.ihx, self.ihy, self.ihz = 1.0 / h[0], 1.0 / h[1], 1.0 / h[2]
        # None -> array-expression path (also the numpy host shim used in tests)
        self.k = getattr(backend, "kernels", None)
        if self.k is not None:
            self._w = (self.colsArr, self.ex, self.fx, self.ey, self.fy,
                       self.ez, self.fz)
            self._ih = (self.ihx, self.ihy, self.ihz)
            self._grid, self._block = _gpuk.grid(self.N)

    def _run(self, name, args):
        self.k.get_function(name)(self._grid, self._block, args)

    def interp(self, field):
        """``S @ field``."""
        if self.k is not None:
            out = self.xp.empty(self.N)
            self._run("interpK", (self.xp.ascontiguousarray(field),)
                      + self._w + (out, self.N))
            return out
        f = [field[c] for c in self.cols]
        g = [f[2 * t] * self.ez + f[2 * t + 1] * self.fz for t in range(4)]
        a = [g[2 * i] * self.ey + g[2 * i + 1] * self.fy for i in (0, 1)]
        return a[0] * self.ex + a[1] * self.fx

    def interpT(self, field):
        """``S' @ field`` - the mass-conserving push-forward scatter."""
        if self.k is not None:
            out = self.xp.zeros(self.N)
            self._run("interpTK", (self.xp.ascontiguousarray(field),)
                      + self._w + (out, self.N))
            return out
        out = self.xp.zeros(self.N)
        for i, wxi in enumerate((self.ex, self.fx)):
            t = field * wxi
            for j, wyi in enumerate((self.ey, self.fy)):
                s = t * wyi
                for k, wzi in enumerate((self.ez, self.fz)):
                    self.scatterAdd(out, self.cols[i * 4 + j * 2 + k], s * wzi)
        return out

    def deriv(self, field):
        """``D_d @ field`` for d = x, y, z. Returns (3, N)."""
        if self.k is not None:
            out = self.xp.empty((3, self.N))
            self._run("derivK", (self.xp.ascontiguousarray(field),)
                      + self._w + self._ih + (out, self.N))
            return out
        f = [field[c] for c in self.cols]
        g = [f[2 * t] * self.ez + f[2 * t + 1] * self.fz for t in range(4)]
        dz = [f[2 * t + 1] - f[2 * t] for t in range(4)]
        a = [g[2 * i] * self.ey + g[2 * i + 1] * self.fy for i in (0, 1)]
        dy = [g[2 * i + 1] - g[2 * i] for i in (0, 1)]
        z = [dz[2 * i] * self.ey + dz[2 * i + 1] * self.fy for i in (0, 1)]
        return self.xp.stack([
            (a[1] - a[0]) * self.ihx,
            (dy[0] * self.ex + dy[1] * self.fx) * self.ihy,
            (z[0] * self.ex + z[1] * self.fx) * self.ihz])

    def derivT(self, vecs):
        """``sum_d D_d' @ vecs[d]`` - the adjoint of :meth:`deriv`."""
        if self.k is not None:
            out = self.xp.zeros(self.N)
            v = self.xp.ascontiguousarray(vecs)
            self._run("derivTK", (v,) + self._w + self._ih + (out, self.N))
            return out
        v0, v1, v2 = vecs[0], vecs[1], vecs[2]
        t1, t2 = v1 * self.ihy, v2 * self.ihz
        ab = (-(v0 * self.ihx), v0 * self.ihx)
        dyb = (t1 * self.ex, t1 * self.fx)
        zb = (t2 * self.ex, t2 * self.fx)
        out = self.xp.zeros(self.N)
        for i in (0, 1):
            gb = (ab[i] * self.ey - dyb[i], ab[i] * self.fy + dyb[i])
            dzb = (zb[i] * self.ey, zb[i] * self.fy)
            for j in (0, 1):
                t = 2 * i + j
                self.scatterAdd(out, self.cols[2 * t],
                                gb[j] * self.ez - dzb[j])
                self.scatterAdd(out, self.cols[2 * t + 1],
                                gb[j] * self.fz + dzb[j])
        return out

    def toSparse(self):
        """Assemble S explicitly (host-side, reference / tests only)."""
        import numpy as np
        import scipy.sparse as sp
        wx = (asNumpy(self.ex), asNumpy(self.fx))
        wy = (asNumpy(self.ey), asNumpy(self.fy))
        wz = (asNumpy(self.ez), asNumpy(self.fz))
        data = np.concatenate([wx[a] * wy[b] * wz[c]
                               for a in (0, 1) for b in (0, 1)
                               for c in (0, 1)])
        cc = np.concatenate([asNumpy(c) for c in self.cols])
        rr = np.tile(np.arange(self.N), 8)
        return sp.csr_matrix((data, (rr, cc)), shape=(self.N, self.N))


class _FusedDeviceTrilinearOp(_DeviceTrilinearOp):
    """Device operators plus the fused whole-step kernels.

    ``forwardSensitivity``/``adjointSensitivity`` dispatch on whether the op has
    ``tangentStep``/``adjointStep``, so these live on a subclass that is only
    instantiated when the CUDA kernels are actually usable - on a device without
    double ``atomicAdd`` the base class is used and the unfused operator path
    runs instead.
    """

    def tangentStep(self, srcR, dRk, fac, dprev, m, dU, k, dt, N):
        """One fused tangent-linear advection step: ``S'@dm + dt*D'(m .* du)``
        with ``dm = srcR*dRk + fac*dprev``."""
        xp = self.xp
        du = xp.ascontiguousarray(
            xp.stack([dU[d * N:(d + 1) * N, k] for d in range(3)]))
        out = xp.zeros(self.N)
        self._run("tangentStepK",
                  (srcR, xp.ascontiguousarray(dRk), fac,
                   xp.ascontiguousarray(dprev), m, du, float(dt))
                  + self._w + self._ih + (out, self.N))
        return out

    def adjointStep(self, b, m, srcR, fac, dt, gUk, gRk, carry):
        """One fused adjoint sensitivity step, writing gU/gR/carry in place."""
        self._run("adjointStepK",
                  (self.xp.ascontiguousarray(b), m, srcR, fac, float(dt))
                  + self._w + self._ih + (gUk, gRk, carry, self.N))


class Backend:
    """Array module plus the operator implementations for one device.

    ``bk.xp`` is ``numpy`` or ``cupy``; the rest of the urOMT numerics is
    written against it.
    """

    def __init__(self, useGpu=False):
        self.isGpu = False
        if useGpu:
            if isAvailable():
                self._initGpu()
            else:
                warnings.warn(
                    "urOMT: useGPU='yes' requested but cupy/CUDA is "
                    "unavailable; "
                    "falling back to the CPU. Install with "
                    "`pip install \"pycerr[gpu]\"` and check that the CUDA "
                    "runtime matches the cupy wheel.", RuntimeWarning)
        if not self.isGpu:
            self._initCpu()

    def _initCpu(self):
        import numpy as np
        from scipy.sparse.linalg import LinearOperator, cg
        from cerr.uromt import numerics
        self.xp = np
        self.cg = _normalizeCg(cg)
        self.LinearOperator = LinearOperator
        self.trilinearOp = numerics._TrilinearOp
        self.diffusionSolver = numerics._DiffusionSolver
        self.kernels = None

    def _initGpu(self):
        import cupy
        import cupyx
        import cupyx.scipy.fft as cufft
        from cupyx.scipy.sparse.linalg import LinearOperator, cg
        self.xp = cupy
        self.cg = _normalizeCg(cg)
        self.LinearOperator = LinearOperator
        # cupy >= 14 deprecates cupyx.scatter_add in favour of cupy.add.at
        self.scatterAdd = getattr(cupy.add, "at", None) or cupyx.scatter_add
        # Fused kernels need double atomicAdd (compute capability >= 6.0); on
        # older devices stay on the array-expression path.
        self.kernels = _gpuk.module() if _gpuk.isSupported() else None
        opCls = (_FusedDeviceTrilinearOp if self.kernels is not None
                 else _DeviceTrilinearOp)
        self.trilinearOp = lambda par, x, y, z: opCls(self, par, x, y, z)
        # the device is already massively parallel: `threads` (a CPU pool
        # size) is accepted only so the call signature matches
        self.diffusionSolver = (lambda n, h, dt, sigma, threads=0:
                                _DeviceDiffusionSolver(cupy, cufft, n, h, dt,
                                                       sigma))
        self.isGpu = True

    def toHost(self, a):
        return asNumpy(a)

    def toDevice(self, a):
        return self.xp.asarray(a)

    def describe(self):
        if not self.isGpu:
            from cerr.uromt import kernels
            return "CPU (numba, %d scatter partitions)" % kernels.numAccum()
        import cupy
        props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
        return "GPU (cupy, %s)" % props["name"].decode()
