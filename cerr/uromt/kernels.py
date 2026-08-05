"""Fused, multi-threaded trilinear kernels for the urOMT solver.

The Gauss-Newton CG matvec is dominated by four trilinear operators evaluated
once per sub-step, per matvec, on every voxel of the ROI box:

===============  ===========================================================
``interp``       ``S @ f``   - gather: sample ``f`` at the departure points
``interpT``      ``S' @ f``  - scatter: the mass-conserving push-forward
``deriv``        ``D_d @ f`` - d(S@f)/d{x,y,z}, the velocity sensitivity
``derivT``       ``sum_d D_d' @ v_d`` - its adjoint
===============  ===========================================================

Written with numpy these are memory-bandwidth bound: each one walks a few dozen
length-N temporaries, so the arithmetic intensity is ~1 flop per byte moved.
Fusing each operator into a single pass over the voxels - one gather of the
eight corners, all the arithmetic in registers, one store - removes essentially
all of that traffic, and the loop is embarrassingly parallel over voxels.

The trilinear weight of a corner is the product ``wx*wy*wz``, so the sum over
the eight corners is evaluated *separably*: collapse the z axis, then y, then x.
That is both fewer flops and what makes the derivative fall out as a difference
of the partially collapsed values.

``S`` is never assembled. The gather/scatter forms are algebraically identical
to the sparse matvecs they replace (and are validated against them in
``tests/test_uromt.py``), but skip building nt CSR matrices per outer iteration.

The scatter kernels cannot race-safely accumulate into a shared output, so they
reduce into a per-partition accumulator ``acc`` of shape ``(nacc, N)`` which the
kernel then sums in a second parallel pass. :func:`accumBuffer` sizes it.
"""

import warnings

import numba
import numpy as np
from numba import get_num_threads, njit, prange

# Partitions for the scatter accumulators. More partitions means more threads on
# the scatter, but the accumulator must be zeroed and reduced every call, so the
# (nacc x N) traffic eventually dominates. Tuned on a 165k-voxel ROI.
_MAX_ACC = 8

# Default thread count when `threads` is 0/unset. The kernels and the DCT are
# memory-bandwidth bound, so throwing every core at them *loses*: on a 24-core
# box the CG matvec measured 542 ms at 1 thread, 300 ms at 4, 286 ms at 6,
# 278 ms at 8, and back up to 368 ms at 24. 6 is near-flat-optimal across the
# ROI sizes tested; raise it explicitly if your machine profiles differently.
_AUTO_THREADS = 6


def maxThreads():
    """Upper bound on threads for this process.

    Fixed by numba when it is first imported (from ``NUMBA_NUM_THREADS`` or the
    CPU count), and cannot be raised at run time.
    """
    return int(numba.config.NUMBA_NUM_THREADS)


def setNumThreads(threads):
    """Set the thread count for the urOMT kernels (the ``numThreads`` setting).

    ``threads <= 0`` selects the auto default (see ``_AUTO_THREADS``) - *not*
    every core, because these kernels are bandwidth bound and oversubscribing
    them is measurably slower. Values above the process maximum are clamped with
    a warning - numba fixes that ceiling when it is imported, so raising it
    requires setting ``NUMBA_NUM_THREADS`` in the environment *before* pyCERR is
    imported.

    Returns the effective thread count.
    """
    cap = maxThreads()
    want = int(threads or 0)
    if want <= 0:
        want = min(cap, _AUTO_THREADS)
    elif want > cap:
        warnings.warn(
            "urOMT: numThreads=%d exceeds this process's maximum of %d; "
            "using %d. "
            "Set the NUMBA_NUM_THREADS environment variable before importing "
            "pyCERR to raise the ceiling." % (want, cap, cap), RuntimeWarning)
        want = cap
    numba.set_num_threads(want)
    return want


def numAccum():
    """Number of scatter accumulator partitions, following the thread count."""
    return max(1, min(int(get_num_threads()), _MAX_ACC))


def accumBuffer(N):
    """Allocate the scatter accumulator for a length-``N`` grid."""
    return np.empty((numAccum(), int(N)), dtype=np.float64)


@njit(parallel=True, cache=True, nogil=True)
def cornerIndexKernel(posX, posY, posZ, hx, hy, hz, n1, n2, n3,
                      cols, ex, fx, ey, fy, ez, fz):
    """Locate each departure point in the grid: the eight surrounding cell
    indices and the per-axis interpolation fractions.

    Mirrors the numpy clipping exactly, including the degenerate singleton axis
    (``n = 1``), where the lower index clamps to -1 and carries weight 0.
    """
    nx = n1 - 2
    ny = n2 - 2
    nz = n3 - 2
    for i in prange(posX.shape[0]):
        gx = posX[i] / hx - 0.5
        gy = posY[i] / hy - 0.5
        gz = posZ[i] / hz - 0.5
        gx = min(max(gx, 0.0), float(n1 - 1))
        gy = min(max(gy, 0.0), float(n2 - 1))
        gz = min(max(gz, 0.0), float(n3 - 1))
        i0 = min(max(int(np.floor(gx)), 0), nx)
        j0 = min(max(int(np.floor(gy)), 0), ny)
        k0 = min(max(int(np.floor(gz)), 0), nz)
        a = gx - i0
        b = gy - j0
        c = gz - k0
        fx[i] = a
        ex[i] = 1.0 - a
        fy[i] = b
        ey[i] = 1.0 - b
        fz[i] = c
        ez[i] = 1.0 - c
        base = i0 + n1 * j0 + n1 * n2 * k0
        sy = n1
        sz = n1 * n2
        cols[0, i] = base                       # (a, b, c) -> a*4 + b*2 + c
        cols[1, i] = base + sz
        cols[2, i] = base + sy
        cols[3, i] = base + sy + sz
        cols[4, i] = base + 1
        cols[5, i] = base + 1 + sz
        cols[6, i] = base + 1 + sy
        cols[7, i] = base + 1 + sy + sz


@njit(parallel=True, cache=True, nogil=True)
def interpKernel(field, cols, ex, fx, ey, fy, ez, fz, out):
    """``out = S @ field`` - trilinear sample at the departure points."""
    for i in prange(out.shape[0]):
        ezi = ez[i]
        fzi = fz[i]
        g00 = field[cols[0, i]] * ezi + field[cols[1, i]] * fzi
        g01 = field[cols[2, i]] * ezi + field[cols[3, i]] * fzi
        g10 = field[cols[4, i]] * ezi + field[cols[5, i]] * fzi
        g11 = field[cols[6, i]] * ezi + field[cols[7, i]] * fzi
        eyi = ey[i]
        fyi = fy[i]
        a0 = g00 * eyi + g01 * fyi
        a1 = g10 * eyi + g11 * fyi
        out[i] = a0 * ex[i] + a1 * fx[i]


@njit(parallel=True, cache=True, nogil=True)
def interpTKernel(field, cols, ex, fx, ey, fy, ez, fz, acc, out):
    """``out = S' @ field`` - the mass-conserving push-forward scatter."""
    N = out.shape[0]
    nacc = acc.shape[0]
    for t in prange(nacc):                       # zero the accumulators
        for j in range(N):
            acc[t, j] = 0.0
    chunk = (N + nacc - 1) // nacc
    for t in prange(nacc):
        lo = t * chunk
        hi = min(lo + chunk, N)
        for i in range(lo, hi):
            v = field[i]
            t0 = v * ex[i]
            t1 = v * fx[i]
            eyi = ey[i]
            fyi = fy[i]
            s00 = t0 * eyi
            s01 = t0 * fyi
            s10 = t1 * eyi
            s11 = t1 * fyi
            ezi = ez[i]
            fzi = fz[i]
            acc[t, cols[0, i]] += s00 * ezi
            acc[t, cols[1, i]] += s00 * fzi
            acc[t, cols[2, i]] += s01 * ezi
            acc[t, cols[3, i]] += s01 * fzi
            acc[t, cols[4, i]] += s10 * ezi
            acc[t, cols[5, i]] += s10 * fzi
            acc[t, cols[6, i]] += s11 * ezi
            acc[t, cols[7, i]] += s11 * fzi
    for j in prange(N):                          # reduce the partitions
        s = 0.0
        for t in range(nacc):
            s += acc[t, j]
        out[j] = s


@njit(parallel=True, cache=True, nogil=True)
def derivKernel(field, cols, ex, fx, ey, fy, ez, fz, ihx, ihy, ihz, out):
    """``out[d] = D_d @ field`` - d(S@field)/d{x,y,z} per point, shape (3, N)."""
    for i in prange(out.shape[1]):
        f000 = field[cols[0, i]]
        f001 = field[cols[1, i]]
        f010 = field[cols[2, i]]
        f011 = field[cols[3, i]]
        f100 = field[cols[4, i]]
        f101 = field[cols[5, i]]
        f110 = field[cols[6, i]]
        f111 = field[cols[7, i]]
        ezi = ez[i]
        fzi = fz[i]
        g00 = f000 * ezi + f001 * fzi            # collapse z
        g01 = f010 * ezi + f011 * fzi
        g10 = f100 * ezi + f101 * fzi
        g11 = f110 * ezi + f111 * fzi
        d00 = f001 - f000
        d01 = f011 - f010
        d10 = f101 - f100
        d11 = f111 - f110
        eyi = ey[i]
        fyi = fy[i]
        a0 = g00 * eyi + g01 * fyi                # collapse y
        a1 = g10 * eyi + g11 * fyi
        dy0 = g01 - g00
        dy1 = g11 - g10
        z0 = d00 * eyi + d01 * fyi
        z1 = d10 * eyi + d11 * fyi
        exi = ex[i]
        fxi = fx[i]
        out[0, i] = (a1 - a0) * ihx               # collapse x
        out[1, i] = (dy0 * exi + dy1 * fxi) * ihy
        out[2, i] = (z0 * exi + z1 * fxi) * ihz


@njit(parallel=True, cache=True, nogil=True)
def tangentStepKernel(srcR, dRk, fac, dprev, m, du0, du1, du2, dt,
                      cols, ex, fx, ey, fy, ez, fz, ihx, ihy, ihz, acc, out):
    """One whole tangent-linear advection step, fused.

    Computes ``out = S'@dm + dt * sum_d D_d'(m .* du_d)`` with
    ``dm = srcR*dRk + fac*dprev``, i.e. everything :func:`interpTKernel` and
    :func:`derivTKernel` did plus the source term and the ``m .* du_d``
    products - in a single pass.

    Fusing matters more than the flop count suggests: the two scatters
    previously walked the same ``cols``/weight arrays (~19 MB per step) twice
    and materialized a (3, N) scratch array for ``m .* du_d``. One pass reads
    them once and needs no scratch.
    """
    N = out.shape[0]
    nacc = acc.shape[0]
    for t in prange(nacc):
        for j in range(N):
            acc[t, j] = 0.0
    chunk = (N + nacc - 1) // nacc
    for t in prange(nacc):
        lo = t * chunk
        hi = min(lo + chunk, N)
        for i in range(lo, hi):
            exi = ex[i]
            fxi = fx[i]
            eyi = ey[i]
            fyi = fy[i]
            ezi = ez[i]
            fzi = fz[i]
            # --- push-forward of the source field dm ---
            dm = srcR[i] * dRk[i] + fac[i] * dprev[i]
            p0 = dm * exi
            p1 = dm * fxi
            s00 = p0 * eyi
            s01 = p0 * fyi
            s10 = p1 * eyi
            s11 = p1 * fyi
            # --- departure-point perturbation: sum_d D_d'(dt * m * du_d) ---
            mi = dt * m[i]
            a1 = (mi * du0[i]) * ihx
            t1 = (mi * du1[i]) * ihy
            t2 = (mi * du2[i]) * ihz
            dy0 = t1 * exi
            dy1 = t1 * fxi
            z0 = t2 * exi
            z1 = t2 * fxi
            g00 = -a1 * eyi - dy0
            g01 = -a1 * fyi + dy0
            g10 = a1 * eyi - dy1
            g11 = a1 * fyi + dy1
            d00 = z0 * eyi
            d01 = z0 * fyi
            d10 = z1 * eyi
            d11 = z1 * fyi
            acc[t, cols[0, i]] += s00 * ezi + g00 * ezi - d00
            acc[t, cols[1, i]] += s00 * fzi + g00 * fzi + d00
            acc[t, cols[2, i]] += s01 * ezi + g01 * ezi - d01
            acc[t, cols[3, i]] += s01 * fzi + g01 * fzi + d01
            acc[t, cols[4, i]] += s10 * ezi + g10 * ezi - d10
            acc[t, cols[5, i]] += s10 * fzi + g10 * fzi + d10
            acc[t, cols[6, i]] += s11 * ezi + g11 * ezi - d11
            acc[t, cols[7, i]] += s11 * fzi + g11 * fzi + d11
    for j in prange(N):
        s = 0.0
        for t in range(nacc):
            s += acc[t, j]
        out[j] = s


@njit(parallel=True, cache=True, nogil=True)
def adjointStepKernel(b, m, srcR, fac, dt, cols, ex, fx, ey, fy, ez, fz,
                      ihx, ihy, ihz, gUk, gRk, carry):
    """One whole adjoint sensitivity step, fused.

    Writes, for step ``k``::

        gUk[d*N + i] = dt * m .* (D_d @ b)          (adjoint w.r.t. velocity)
        gRk          = srcR .* (S @ b)              (adjoint w.r.t. the source)
        carry        = fac  .* (S @ b)              (adjoint to rho_{k-1})

    ``S @ b`` and ``D_d @ b`` are the *same* eight-corner gather followed by the
    same separable collapse - the interpolated value falls out of the very
    reduction the derivative needs - so computing them together costs barely
    more than either alone, and neither ``S@b`` nor the (3, N) derivative array
    ever has to be materialized.
    """
    N = b.shape[0]
    for i in prange(N):
        f000 = b[cols[0, i]]
        f001 = b[cols[1, i]]
        f010 = b[cols[2, i]]
        f011 = b[cols[3, i]]
        f100 = b[cols[4, i]]
        f101 = b[cols[5, i]]
        f110 = b[cols[6, i]]
        f111 = b[cols[7, i]]
        ezi = ez[i]
        fzi = fz[i]
        g00 = f000 * ezi + f001 * fzi            # collapse z
        g01 = f010 * ezi + f011 * fzi
        g10 = f100 * ezi + f101 * fzi
        g11 = f110 * ezi + f111 * fzi
        d00 = f001 - f000
        d01 = f011 - f010
        d10 = f101 - f100
        d11 = f111 - f110
        eyi = ey[i]
        fyi = fy[i]
        a0 = g00 * eyi + g01 * fyi                # collapse y
        a1 = g10 * eyi + g11 * fyi
        dy0 = g01 - g00
        dy1 = g11 - g10
        z0 = d00 * eyi + d01 * fyi
        z1 = d10 * eyi + d11 * fyi
        exi = ex[i]
        fxi = fx[i]
        mbar = a0 * exi + a1 * fxi                # == (S @ b)[i]
        gRk[i] = srcR[i] * mbar
        carry[i] = fac[i] * mbar
        w = dt * m[i]
        gUk[i] = w * (a1 - a0) * ihx
        gUk[N + i] = w * (dy0 * exi + dy1 * fxi) * ihy
        gUk[2 * N + i] = w * (z0 * exi + z1 * fxi) * ihz


@njit(parallel=True, cache=True, nogil=True)
def derivTKernel(v, cols, ex, fx, ey, fy, ez, fz, ihx, ihy, ihz, acc, out):
    """``out = sum_d D_d' @ v[d]`` - the adjoint of :func:`derivKernel`."""
    N = out.shape[0]
    nacc = acc.shape[0]
    for t in prange(nacc):
        for j in range(N):
            acc[t, j] = 0.0
    chunk = (N + nacc - 1) // nacc
    for t in prange(nacc):
        lo = t * chunk
        hi = min(lo + chunk, N)
        for i in range(lo, hi):
            exi = ex[i]
            fxi = fx[i]
            a1 = v[0, i] * ihx                    # adjoint of the x collapse
            t1 = v[1, i] * ihy
            t2 = v[2, i] * ihz
            dy0 = t1 * exi
            dy1 = t1 * fxi
            z0 = t2 * exi
            z1 = t2 * fxi
            eyi = ey[i]
            fyi = fy[i]
            g00 = -a1 * eyi - dy0                 # adjoint of the y collapse
            g01 = -a1 * fyi + dy0
            g10 = a1 * eyi - dy1
            g11 = a1 * fyi + dy1
            d00 = z0 * eyi
            d01 = z0 * fyi
            d10 = z1 * eyi
            d11 = z1 * fyi
            ezi = ez[i]
            fzi = fz[i]
            acc[t, cols[0, i]] += g00 * ezi - d00  # adjoint of the z collapse
            acc[t, cols[1, i]] += g00 * fzi + d00
            acc[t, cols[2, i]] += g01 * ezi - d01
            acc[t, cols[3, i]] += g01 * fzi + d01
            acc[t, cols[4, i]] += g10 * ezi - d10
            acc[t, cols[5, i]] += g10 * fzi + d10
            acc[t, cols[6, i]] += g11 * ezi - d11
            acc[t, cols[7, i]] += g11 * fzi + d11
    for j in prange(N):
        s = 0.0
        for t in range(nacc):
            s += acc[t, j]
        out[j] = s
