"""Fused CUDA kernels for the urOMT device backend.

The device counterparts of :mod:`cerr.uromt.kernels`. The array-expression form
in :mod:`cerr.uromt.gpu` is correct but issues dozens of small kernels per
operator call, which left the GPU *kernel-launch-latency bound*: solve time was
nearly flat from 84k to 622k voxels. These fuse each operator - and each whole
sensitivity sub-step - into a single launch, exactly as the numba kernels do on
the CPU, and use the same separable collapse (z, then y, then x).

Corner ordering matches the CPU kernels: ``(a, b, c) -> a*4 + b*2 + c``, with
``cols`` an (8, N) C-contiguous int64 array indexed ``cols[j*N + i]``.

The scatter kernels use ``atomicAdd`` on doubles, which is native only from
compute capability 6.0. :func:`isSupported` gates on that; below it the caller
keeps the array-expression path. Note atomics make the scatter summation order
nondeterministic, so repeated runs can differ at the ~1e-16 level - this is not
new, ``cupy.add.at`` is atomic-based too.
"""

_SOURCE = r"""
extern "C" {

#define CORNERS(i)                                                            \
    const long long c0 = cols[0*N + (i)], c1 = cols[1*N + (i)],               \
                    c2 = cols[2*N + (i)], c3 = cols[3*N + (i)],               \
                    c4 = cols[4*N + (i)], c5 = cols[5*N + (i)],               \
                    c6 = cols[6*N + (i)], c7 = cols[7*N + (i)];

#define WEIGHTS(i)                                                            \
    const double exi = ex[(i)], fxi = fx[(i)], eyi = ey[(i)],                 \
                 fyi = fy[(i)], ezi = ez[(i)], fzi = fz[(i)];

/* out = S @ field : trilinear gather */
__global__ void interpK(const double* f, const long long* cols,
                        const double* ex, const double* fx, const double* ey,
                        const double* fy, const double* ez, const double* fz,
                        double* out, const long long N)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    CORNERS(i) WEIGHTS(i)
    double g00 = f[c0]*ezi + f[c1]*fzi;
    double g01 = f[c2]*ezi + f[c3]*fzi;
    double g10 = f[c4]*ezi + f[c5]*fzi;
    double g11 = f[c6]*ezi + f[c7]*fzi;
    double a0 = g00*eyi + g01*fyi;
    double a1 = g10*eyi + g11*fyi;
    out[i] = a0*exi + a1*fxi;
}

/* out = S' @ field : mass-conserving push-forward scatter */
__global__ void interpTK(const double* f, const long long* cols,
                         const double* ex, const double* fx, const double* ey,
                         const double* fy, const double* ez, const double* fz,
                         double* out, const long long N)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    CORNERS(i) WEIGHTS(i)
    double v = f[i];
    double t0 = v*exi, t1 = v*fxi;
    double s00 = t0*eyi, s01 = t0*fyi, s10 = t1*eyi, s11 = t1*fyi;
    atomicAdd(&out[c0], s00*ezi);  atomicAdd(&out[c1], s00*fzi);
    atomicAdd(&out[c2], s01*ezi);  atomicAdd(&out[c3], s01*fzi);
    atomicAdd(&out[c4], s10*ezi);  atomicAdd(&out[c5], s10*fzi);
    atomicAdd(&out[c6], s11*ezi);  atomicAdd(&out[c7], s11*fzi);
}

/* out[d] = D_d @ field, d = x,y,z (out is (3, N)) */
__global__ void derivK(const double* f, const long long* cols,
                       const double* ex, const double* fx, const double* ey,
                       const double* fy, const double* ez, const double* fz,
                       const double ihx, const double ihy, const double ihz,
                       double* out, const long long N)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    CORNERS(i) WEIGHTS(i)
    double f000 = f[c0], f001 = f[c1], f010 = f[c2], f011 = f[c3];
    double f100 = f[c4], f101 = f[c5], f110 = f[c6], f111 = f[c7];
    double g00 = f000*ezi + f001*fzi, g01 = f010*ezi + f011*fzi;
    double g10 = f100*ezi + f101*fzi, g11 = f110*ezi + f111*fzi;
    double d00 = f001 - f000, d01 = f011 - f010;
    double d10 = f101 - f100, d11 = f111 - f110;
    double a0 = g00*eyi + g01*fyi, a1 = g10*eyi + g11*fyi;
    double dy0 = g01 - g00, dy1 = g11 - g10;
    double z0 = d00*eyi + d01*fyi, z1 = d10*eyi + d11*fyi;
    out[i]       = (a1 - a0) * ihx;
    out[N + i]   = (dy0*exi + dy1*fxi) * ihy;
    out[2*N + i] = (z0*exi + z1*fxi) * ihz;
}

/* out = sum_d D_d' @ v[d] : adjoint of derivK */
__global__ void derivTK(const double* v, const long long* cols,
                        const double* ex, const double* fx, const double* ey,
                        const double* fy, const double* ez, const double* fz,
                        const double ihx, const double ihy, const double ihz,
                        double* out, const long long N)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    CORNERS(i) WEIGHTS(i)
    double a1 = v[i]*ihx, t1 = v[N + i]*ihy, t2 = v[2*N + i]*ihz;
    double dy0 = t1*exi, dy1 = t1*fxi;
    double z0 = t2*exi, z1 = t2*fxi;
    double g00 = -a1*eyi - dy0, g01 = -a1*fyi + dy0;
    double g10 =  a1*eyi - dy1, g11 =  a1*fyi + dy1;
    double d00 = z0*eyi, d01 = z0*fyi, d10 = z1*eyi, d11 = z1*fyi;
    atomicAdd(&out[c0], g00*ezi - d00);  atomicAdd(&out[c1], g00*fzi + d00);
    atomicAdd(&out[c2], g01*ezi - d01);  atomicAdd(&out[c3], g01*fzi + d01);
    atomicAdd(&out[c4], g10*ezi - d10);  atomicAdd(&out[c5], g10*fzi + d10);
    atomicAdd(&out[c6], g11*ezi - d11);  atomicAdd(&out[c7], g11*fzi + d11);
}

/* One whole tangent-linear advection step:
   out = S'@dm + dt*sum_d D_d'(m .* du_d),  dm = srcR*dRk + fac*dprev  */
__global__ void tangentStepK(const double* srcR, const double* dRk,
                             const double* fac, const double* dprev,
                             const double* m, const double* du,
                             const double dt, const long long* cols,
                             const double* ex, const double* fx,
                             const double* ey, const double* fy,
                             const double* ez, const double* fz,
                             const double ihx, const double ihy,
                             const double ihz, double* out, const long long N)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    CORNERS(i) WEIGHTS(i)
    double dm = srcR[i]*dRk[i] + fac[i]*dprev[i];
    double p0 = dm*exi, p1 = dm*fxi;
    double s00 = p0*eyi, s01 = p0*fyi, s10 = p1*eyi, s11 = p1*fyi;

    double mi = dt * m[i];
    double a1 = (mi*du[i])*ihx, t1 = (mi*du[N + i])*ihy,
           t2 = (mi*du[2*N + i])*ihz;
    double dy0 = t1*exi, dy1 = t1*fxi;
    double z0 = t2*exi, z1 = t2*fxi;
    double g00 = -a1*eyi - dy0, g01 = -a1*fyi + dy0;
    double g10 =  a1*eyi - dy1, g11 =  a1*fyi + dy1;
    double d00 = z0*eyi, d01 = z0*fyi, d10 = z1*eyi, d11 = z1*fyi;

    atomicAdd(&out[c0], s00*ezi + g00*ezi - d00);
    atomicAdd(&out[c1], s00*fzi + g00*fzi + d00);
    atomicAdd(&out[c2], s01*ezi + g01*ezi - d01);
    atomicAdd(&out[c3], s01*fzi + g01*fzi + d01);
    atomicAdd(&out[c4], s10*ezi + g10*ezi - d10);
    atomicAdd(&out[c5], s10*fzi + g10*fzi + d10);
    atomicAdd(&out[c6], s11*ezi + g11*ezi - d11);
    atomicAdd(&out[c7], s11*fzi + g11*fzi + d11);
}

/* One whole adjoint sensitivity step. S@b falls out of the same reduction
   that produces D_d@b, so neither is materialized. */
__global__ void adjointStepK(const double* b, const double* m,
                             const double* srcR, const double* fac,
                             const double dt, const long long* cols,
                             const double* ex, const double* fx,
                             const double* ey, const double* fy,
                             const double* ez, const double* fz,
                             const double ihx, const double ihy,
                             const double ihz, double* gUk, double* gRk,
                             double* carry, const long long N)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= N) return;
    CORNERS(i) WEIGHTS(i)
    double f000 = b[c0], f001 = b[c1], f010 = b[c2], f011 = b[c3];
    double f100 = b[c4], f101 = b[c5], f110 = b[c6], f111 = b[c7];
    double g00 = f000*ezi + f001*fzi, g01 = f010*ezi + f011*fzi;
    double g10 = f100*ezi + f101*fzi, g11 = f110*ezi + f111*fzi;
    double d00 = f001 - f000, d01 = f011 - f010;
    double d10 = f101 - f100, d11 = f111 - f110;
    double a0 = g00*eyi + g01*fyi, a1 = g10*eyi + g11*fyi;
    double dy0 = g01 - g00, dy1 = g11 - g10;
    double z0 = d00*eyi + d01*fyi, z1 = d10*eyi + d11*fyi;

    double mbar = a0*exi + a1*fxi;          /* == (S @ b)[i] */
    gRk[i] = srcR[i] * mbar;
    carry[i] = fac[i] * mbar;
    double w = dt * m[i];
    gUk[i]       = w * (a1 - a0) * ihx;
    gUk[N + i]   = w * (dy0*exi + dy1*fxi) * ihy;
    gUk[2*N + i] = w * (z0*exi + z1*fxi) * ihz;
}

}  /* extern "C" */
"""

_BLOCK = 256
_module = None


def isSupported():
    """True when the device can run these kernels (needs double atomicAdd)."""
    try:
        import cupy
        cc = cupy.cuda.Device().compute_capability
        return int(cc) >= 60
    except Exception:
        return False


def module():
    """The compiled RawModule (compiled once, on first use)."""
    global _module
    if _module is None:
        import cupy
        _module = cupy.RawModule(code=_SOURCE)
    return _module


def grid(n):
    """Launch configuration for ``n`` elements."""
    return ((int(n) + _BLOCK - 1) // _BLOCK,), (_BLOCK,)
