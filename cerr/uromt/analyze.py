"""urOMT post-processing - Eulerian (runEULA) and Lagrangian (runGLAD).

Ports of Parts 3 & 4 of the MATLAB urOMT ``driver_RatBrain.m`` (``runEULA.m`` /
``paramInitEULApar.m`` and ``runGLAD.m`` / ``paramInitGLADpar.m``), adapted from
the MATLAB file-based pipeline to operate in-memory on the dict returned by
:func:`cerr.uromt.solver.solveUROMT`.

The urOMT advection-diffusion model
``rho_t + div(rho v) = sigma Lap(rho) + rho r`` has an equivalent advective form
with the **effective (flux) velocity**

    v_eff = v - sigma * grad(log rho)

i.e. the OMT velocity ``v`` minus the diffusive drift. Eulerian maps summarize
the fields over time; Lagrangian pathlines integrate ``v_eff`` to trace
transport, with the Peclet number ``Pe = |v| / |sigma grad(log rho)|``
contrasting advective vs. diffusive speed.
"""

import numpy as np

_EPS = 1e-8


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _cellCenters(n, h):
    return [(np.arange(ni) + 0.5) * hi for ni, hi in zip(n, h)]


_PECLET_FLOOR_FRAC = 0.1     # Peclet denom floor as a fraction of ROI-median dif


def _pecletDenomFloor(dif, roi, frac=None):
    """Robust additive floor for the Peclet denominator |sigma grad log rho|:
    a fraction of the typical (ROI-median) diffusive speed. Without it, voxels
    in smooth-density regions (dif -> 0) produce enormous Peclet spikes (the old
    1e-8 floor let them explode), giving a salt-and-pepper / patchy map.

    The fraction is the ``peclet_floor`` setting. It is a deliberate departure
    from the reference implementation, which floors only at machine epsilon:
    because the floor is *additive* it biases Peclet downward, most strongly in
    exactly the high-Peclet voxels. Measured against the reference breast maps,
    the through-origin slope goes 0.67 at the 0.1 default, 0.98 at 0.01, and
    1.06 at 0 - so use a small value (or 0) when reproducing reference Peclet,
    and the default when you want a display-stable map.
    """
    f = _PECLET_FLOOR_FRAC if frac is None else float(frac)
    if f <= 0.0:
        return _EPS
    d = dif[roi & (dif > 0)] if roi is not None else dif[dif > 0]
    if d.size == 0:
        return _EPS
    return max(f * float(np.median(d)), _EPS)


def _gradLog(rho3, h):
    """Cell-centered grad(log rho) via central differences; returns a (3, N)
    array (components along axes 0,1,2 = row,col,slice), Fortran-flattened."""
    lr = np.log(np.maximum(rho3, 0.0) + _EPS)
    g = np.gradient(lr, h[0], h[1], h[2], edge_order=1)
    return np.stack([gi.ravel(order="F") for gi in g], axis=0)


def _stepFields(v3N, rhoN, n, h, sigma):
    """For one (velocity, density) step return effective velocity v_eff (3,N),
    advective speed |v| (N,), diffusive speed |sigma grad log rho| (N,)."""
    rho3 = rhoN.reshape(n, order="F")
    vdiff = sigma * _gradLog(rho3, h)                  # (3,N)
    vEff = v3N - vdiff
    advSpeed = np.sqrt(np.sum(v3N ** 2, axis=0))
    difSpeed = np.sqrt(np.sum(vdiff ** 2, axis=0))
    return vEff, advSpeed, difSpeed


#: Quantities that can be sampled along a pathline, colour-coded onto the
#: velocity/flux vectors, or accumulated into a Lagrangian map. Every one of
#: them is defined per (velocity, density, rate) sub-step, so a vector at a
#: voxel and the pathline segment passing through it are the SAME number.
QUANTITIES = ("speed", "effSpeed", "peclet", "rate", "rho", "flux")

QUANTITY_LABELS = {"speed": "speed |v| (mm/t)",
                   "effSpeed": "eff. speed |v_eff| (mm/t)",
                   "peclet": "Peclet (-)",
                   "rate": "rate r (1/t)",
                   "rho": "density rho (a.u.)",
                   "flux": "|flux| (a.u. mm/t)"}

#: Reductions of a quantity over a pathline (or, for vectors, over the run's
#: time intervals). ``"along"`` is the un-reduced value - per vertex for a
#: pathline, per interval for a vector.
STATS = ("mean", "median", "max", "along")


def _stepQuantities(v3N, rhoN, rN, n, h, sigma, pFloor=None,
                    keys=QUANTITIES):
    """All per-voxel quantities for ONE (velocity, density, rate) sub-step.

    Returns ``(vEff, vals)`` where ``vEff`` is the effective velocity (3,N) used
    for advection and ``vals`` maps each requested key in :data:`QUANTITIES` to
    a flat (N,) array. This is the single definition of every displayed
    quantity: :func:`runEULAIntervals` reduces these over time onto the grid and
    :func:`runGLAD` samples the same arrays along the trajectories, so the two
    displays cannot drift apart.

    ``pFloor`` is the Peclet denominator floor *value* (not the fraction);
    ``None`` uses the machine-epsilon floor, matching the reference pathline
    Peclet.
    """
    rho3 = rhoN.reshape(n, order="F")
    vdiff = sigma * _gradLog(rho3, h)                  # (3,N)
    vEff = v3N - vdiff
    adv = np.sqrt(np.sum(v3N ** 2, axis=0))
    vals = {}
    if "speed" in keys:
        vals["speed"] = adv
    if "effSpeed" in keys:
        vals["effSpeed"] = np.sqrt(np.sum(vEff ** 2, axis=0))
    if "peclet" in keys:
        dif = np.sqrt(np.sum(vdiff ** 2, axis=0))
        # `pFloor` may be a callable so a caller that needs a data-dependent
        # floor (the Eulerian maps: a fraction of the median diffusive speed)
        # gets it without recomputing grad(log rho) outside.
        fl = (_EPS if pFloor is None else
              (pFloor(dif) if callable(pFloor) else pFloor))
        vals["peclet"] = adv / (dif + fl)
    if "rate" in keys:
        vals["rate"] = np.asarray(rN, dtype=float)
    if "rho" in keys:
        vals["rho"] = np.asarray(rhoN, dtype=float)
    if "flux" in keys:
        vals["flux"] = np.sqrt(np.sum((rhoN * vEff) ** 2, axis=0))
    return vEff, vals


def _globalSteps(result):
    """Yield (g, v (3,N), rho (N,), r (N,)) for every global sub-step g over all
    intervals (g = interval*nt + k)."""
    nt = int(result["nt"])
    for t, (u, rr, rho) in enumerate(zip(result["u"], result["r"],
                                         result["rho"])):
        for k in range(nt):
            yield t * nt + k, u[:, :, k], rho[:, k], rr[:, k]


# --------------------------------------------------------------------------- #
#  Part 3: Eulerian post-processing (runEULA.m)
# --------------------------------------------------------------------------- #
def runEULA(result, maskOnly=True, pecletFloor=None):
    """Eulerian post-processing: time-averaged speed, rate, Peclet and flux maps.

    Args:
        result (dict): output of :func:`cerr.uromt.solver.solveUROMT`.
        maskOnly (bool): zero the maps outside the ROI mask.
        pecletFloor (float): override the Peclet denominator floor fraction
            (default: the run's ``peclet_floor`` setting). See
            :func:`_pecletDenomFloor`.

    Returns:
        dict ``Eul`` with flattened (N,) maps ``speed`` (mean |v|), ``rate``
        (mean r), ``peclet`` (mean |v|/|diffusion|), the mean flux vector
        ``flux`` (3,N) = per-interval time INTEGRAL sum(rho * v_eff) over each
        interval's nt sub-steps, averaged across intervals (matches the
        reference EulerFlux convention; scales with nt), their 3-D reshapes
        (``speed3``/``rate3``/``peclet3``), and grid metadata (``n``,
        ``spacing``, ``mask``, ``bbox``, ``frameScanNums``).
    """
    n = [int(v) for v in result["n"]]
    h = [float(v) for v in result["spacing"]]
    sigma = float(result.get("sigma", 0.0))
    pFloor = (result.get("pecletFloor") if pecletFloor is None
              else pecletFloor)
    N = int(np.prod(n))

    roiM = (np.asarray(result["mask"]) > 0).ravel(order="F")
    speed = np.zeros(N)
    rate = np.zeros(N)
    peclet = np.zeros(N)
    flux = np.zeros((3, N))
    nSteps = 0
    for _g, v, rho, r in _globalSteps(result):
        vEff, adv, dif = _stepFields(v, rho, n, h, sigma)
        speed += adv
        rate += r
        peclet += adv / (dif + _pecletDenomFloor(dif, roiM, pFloor))
        flux += rho * vEff
        nSteps += 1
    nt = int(result["nt"])
    if nSteps:
        speed /= nSteps
        rate /= nSteps
        peclet /= nSteps
        # flux is the per-interval time INTEGRAL (sum over an interval's nt
        # sub-steps), matching the reference EulerFlux; the scalar maps above
        # stay time-averages. Dividing by nSteps/nt = the interval count leaves
        # the mean per-interval flux integral.
        flux /= max(nSteps / nt, 1)

    if maskOnly:
        m = roiM
        speed[~m] = 0.0
        rate[~m] = 0.0
        peclet[~m] = 0.0
        flux[:, ~m] = 0.0

    Eul = dict(speed=speed, rate=rate, peclet=peclet, flux=flux,
               speed3=speed.reshape(n, order="F"),
               rate3=rate.reshape(n, order="F"),
               peclet3=peclet.reshape(n, order="F"),
               n=n, spacing=h, mask=result["mask"], bbox=result["bbox"],
               frameScanNums=result.get("frameScanNums"))
    return Eul


def runEULAIntervals(result, maskOnly=True, pecletFloor=None):
    """Per-interval Eulerian maps (time-averaged over each interval's nt
    sub-steps), as the MATLAB ``runEULA`` writes one set of maps per time
    interval. Returns lists (one entry per interval) of 3-D ROI-grid arrays.

    Keys: ``speed`` (|v|, advective), ``effSpeed`` (|v_eff|, flux velocity),
    ``rate`` (r), ``peclet`` (|v|/|diffusion|), ``flux`` (rho*v_eff, list of
    (3,*n)), ``rho`` (density); plus grid metadata.

    Correspondence with the reference MATLAB maps (verified voxel-wise against
    saved reference output using its own recovered (u, r)):

    * ``EulerS``  == ``speed``, the **raw** velocity magnitude |v| -- *not*
      ``effSpeed``. (Matches at corr 1.0000, median ratio 1.000; ``effSpeed``
      only agrees to ~0.4% here because this data is advection-dominated.)
    * ``EulerR``  == ``rate``, ``EulerPe`` == ``peclet``, ``EulerRho`` == ``rho``.
    * ``EulerFlux`` == ``flux``: both are the **sum** over an interval's ``nt``
      sub-steps (a time integral), not a time average -- unlike the scalar maps
      above, which are averages. Note this makes ``flux`` scale with ``nt``, so
      it is only comparable between runs that use the same sub-step count.
      The same convention applies to the r-weighted flux (``EulerRFlux``).
    """
    n = [int(v) for v in result["n"]]
    h = [float(v) for v in result["spacing"]]
    sigma = float(result.get("sigma", 0.0))
    pFloor = (result.get("pecletFloor") if pecletFloor is None
              else pecletFloor)
    nt = int(result["nt"])
    m = (np.asarray(result["mask"]) > 0).ravel(order="F")
    out = {k: [] for k in ("speed", "effSpeed", "rate", "peclet", "flux",
                           "rho")}
    for u, rr, rho in zip(result["u"], result["r"], result["rho"]):
        N = u.shape[1]
        acc = {k: np.zeros(N) for k in ("speed", "effSpeed", "rate", "peclet",
                                        "rho")}
        flux = np.zeros((3, N))
        for k in range(nt):
            v = u[:, :, k]
            rho_k = rho[:, k]
            # Same per-step definitions the pathlines sample (_stepQuantities),
            # so a vector coloured by a map and a pathline segment through the
            # same voxel report the same quantity. The one deliberate exception
            # is Peclet: the map floors the denominator at a fraction of the
            # median diffusive speed for display stability, the pathlines at
            # machine epsilon to match the reference - see _pecletDenomFloor.
            vEff, q = _stepQuantities(
                v, rho_k, rr[:, k], n, h, sigma,
                pFloor=lambda dif: _pecletDenomFloor(dif, m, pFloor),
                keys=("speed", "effSpeed", "peclet", "rate", "rho"))
            for key in ("speed", "effSpeed", "peclet", "rate", "rho"):
                acc[key] += q[key]
            flux += rho_k * vEff
        for k in acc:
            acc[k] /= nt
        # flux is NOT divided by nt: the reference EulerFlux is the sum over the
        # interval's nt sub-steps (a time integral), while the scalar maps are
        # time averages.
        if maskOnly:
            for k in acc:
                acc[k][~m] = 0.0
            flux[:, ~m] = 0.0
        for k in ("speed", "effSpeed", "rate", "peclet", "rho"):
            out[k].append(acc[k].reshape(n, order="F"))
        out["flux"].append(flux.reshape((3,) + tuple(n), order="F"))
    out.update(n=n, spacing=h, mask=result["mask"], bbox=result["bbox"],
               frameScanNums=result.get("frameScanNums"))
    return out


def surfaceFlux(flux3, mask, spacing, centered=False):
    """Influx / outflux / net flux through the SURFACE of ``mask``.

    ``flux`` in the maps is a flux *density* vector per voxel (``rho * v_eff``)
    and its ``|flux|`` map is unsigned - it says how fast tracer moves, not
    whether it enters or leaves. In/out only exists relative to a surface, so
    this integrates the outward normal component over the boundary of ``mask``:

        net = closed-surface integral of (rho v_eff) . n  dA

    Every boundary face - a face between a voxel inside the mask and one
    outside, plus faces of inside voxels lying on the grid border - contributes
    ``F_normal * faceArea``. Faces with a POSITIVE contribution are outflux,
    negative ones influx; both totals are returned as positive numbers.

    Args:
        flux3 (ndarray): (3, n0, n1, n2) flux vector field on the ROI grid, as
            ``runEULAIntervals()['flux'][k]``.
        mask (ndarray): (n0, n1, n2) region whose surface to integrate over.
        spacing: [row, col, slice] voxel size (mm), used for the face areas.
        centered (bool): average the two cells sharing a face instead of taking
            the inside one. **Off by default because the maps are masked**:
            ``runEULA*(maskOnly=True)`` zeroes the field outside the mask, so a
            centered average would halve every boundary face. Turn it on only
            for an unmasked field.

    Returns:
        dict with ``influx`` and ``outflux`` (positive magnitudes), ``net``
        (``outflux - influx``, i.e. OUTWARD-positive: net > 0 means the region
        is losing tracer, net < 0 means it is accumulating), and ``map3``, the
        per-voxel signed outward flux summed over each boundary voxel's faces -
        a displayable map of where tracer enters and leaves. Units are those of
        the flux density times mm^2 (a.u. mm^3 / t).
    """
    F = np.asarray(flux3, dtype=float)
    m = np.asarray(mask) > 0
    if F.ndim == 2:                         # (3, N) -> grid
        F = F.reshape((3,) + m.shape, order="F")
    h = [float(v) for v in spacing]
    areas = (h[1] * h[2], h[0] * h[2], h[0] * h[1])
    out3 = np.zeros(m.shape, dtype=float)
    inSum = outSum = 0.0

    def _add(val, where, target):
        """Accumulate one set of face contributions (signed, outward +)."""
        nonlocal inSum, outSum
        v = np.where(where, val, 0.0)
        target += v
        outSum += float(v[v > 0].sum())
        inSum += float(-v[v < 0].sum())

    for a in range(3):
        Fa = F[a]
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[a], hi[a] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        mLo, mHi = m[lo], m[hi]
        fLo, fHi = Fa[lo], Fa[hi]
        fFace = 0.5 * (fLo + fHi) if centered else None
        # inside at lo, outside at hi: outward normal is +a
        _add((fFace if centered else fLo) * areas[a], mLo & ~mHi, out3[lo])
        # inside at hi, outside at lo: outward normal is -a
        _add(-(fFace if centered else fHi) * areas[a], mHi & ~mLo, out3[hi])
        # faces on the grid border belong to the surface too
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[a], last[a] = slice(0, 1), slice(-1, None)
        first, last = tuple(first), tuple(last)
        _add(-Fa[first] * areas[a], m[first], out3[first])
        _add(Fa[last] * areas[a], m[last], out3[last])

    return {"influx": inSum, "outflux": outSum, "net": outSum - inSum,
            "map3": out3}


def intervalSurfaceFlux(ei, mask=None, centered=False):
    """Per-interval :func:`surfaceFlux` for a :func:`runEULAIntervals` result.

    Args:
        ei (dict): output of :func:`runEULAIntervals`.
        mask (ndarray): region to integrate over; defaults to the run's ROI
            mask (``ei['mask']``), which is the DILATED mask when
            ``mask_dilate`` is set - pass the undilated structure mask to get
            the flux through the drawn contour itself.
        centered (bool): see :func:`surfaceFlux`.

    Returns:
        dict of lists (one entry per interval): ``influx``, ``outflux``,
        ``net``, ``map3``.
    """
    m = ei["mask"] if mask is None else mask
    out = {k: [] for k in ("influx", "outflux", "net", "map3")}
    for f in ei.get("flux", []):
        r = surfaceFlux(f, m, ei["spacing"], centered=centered)
        for k in out:
            out[k].append(r[k])
    return out


def eulerianStats(ei, quantity="speed"):
    """Per-VOXEL ``{mean, median, max}`` of an Eulerian quantity over the run's
    time intervals, plus the per-interval list under ``'along'``.

    The vector overlay's colour-by is the Eulerian twin of the pathline one:
    a pathline reduces a quantity over its own vertices, a vector reduces the
    same quantity over time at its voxel. ``'along'`` is the un-reduced form -
    the value in the displayed interval, the vector counterpart of colouring a
    path along its length.

    Args:
        ei (dict): output of :func:`runEULAIntervals`.
        quantity (str): one of :data:`QUANTITIES`. ``'flux'`` is the magnitude
            of the interval flux vector.

    Returns:
        dict with 3-D ROI-grid arrays ``mean`` / ``median`` / ``max`` and the
        list ``along``; ``None`` when the quantity is not in ``ei``.
    """
    if quantity == "flux":
        ivl = [np.sqrt(np.sum(np.asarray(f) ** 2, axis=0))
               for f in ei.get("flux", [])]
    else:
        ivl = list(ei.get(quantity) or [])
    if not ivl:
        return None
    arr = np.asarray(ivl, dtype=float)                  # (nIvl, *n)
    return {"mean": arr.mean(0), "median": np.median(arr, 0),
            "max": arr.max(0), "along": ivl}


# --------------------------------------------------------------------------- #
#  Part 4: Lagrangian post-processing (runGLAD.m)
# --------------------------------------------------------------------------- #
def _interpolators(field3, gc):
    """RegularGridInterpolator for each component of a (k, *n) field."""
    from scipy.interpolate import RegularGridInterpolator
    return [RegularGridInterpolator(gc, field3[c], bounds_error=False,
                                    fill_value=0.0) for c in range(len(field3))]


def _stackInterpolator(fields, gc):
    """ONE interpolator over a stack of same-grid scalar fields.

    ``RegularGridInterpolator`` accepts trailing value dimensions, so stacking
    the fields into (n0, n1, n2, K) computes the interpolation weights once for
    all K channels instead of once per channel. That matters here: ``runGLAD``
    samples the 3 velocity components *and* every requested quantity at the
    same points on every sub-step, and per-channel interpolators made sampling
    all six quantities twice the cost of sampling two.
    """
    from scipy.interpolate import RegularGridInterpolator
    stack = np.stack(list(fields), axis=-1)
    return RegularGridInterpolator(gc, stack, bounds_error=False,
                                   fill_value=0.0)


def runGLAD(result, spfs=1, nEuler=5, direction=1.0, minSpeed=0.0,
            slTolVox=0.0, maxSeeds=None, seedMask=None,
            quantities=QUANTITIES):
    """Lagrangian post-processing: integrate transport pathlines of the
    effective velocity ``v_eff`` seeded in the ROI.

    Args:
        result (dict): output of :func:`cerr.uromt.solver.solveUROMT`.
        spfs (int): seed every ``spfs``-th masked voxel per axis. Note this
            thins by ``spfs**3``, so the default 2 used to yield 1/8 of the ROI.
        nEuler (int): Euler sub-steps per urOMT time sub-step.
        direction (float): +1 follows the urOMT velocity, -1 reverses it.
        minSpeed (float): stop advancing a particle where ``|v_eff| < minSpeed``.
        slTolVox (float): drop pathlines whose net displacement (voxels) is below
            this. **Default 0 = keep everything.** It used to be 1.0, which on
            real data discards the majority of seeds - 88% of them move under a
            voxel - leaving a sparse display AND biasing any accumulated map
            toward the fast survivors (measured against the reference: speed
            median ratio 0.66 at 1.0 vs 0.96 at 0). Raise it only to declutter
            a purely qualitative view.
        maxSeeds (int): optional cap on the number of seed particles, applied
            by uniform thinning. ``None`` (default) = no cap; density is then
            governed solely by ``spfs``.
        seedMask (ndarray): where to seed, on the ROI grid. ``None`` uses
            ``result['mask']``, which is the **dilated** ROI mask when
            ``mask_dilate`` is set - so with the reference's ``mask_dilate=2``
            about 23% of seeds start outside the drawn structure contour (the
            mask grows ~30%). That matches the reference, which also seeds in
            its dilated ROI mask, but it is surprising on screen; pass the
            undilated structure mask here to keep every pathline starting inside
            the contour.
        quantities: which of :data:`QUANTITIES` to sample along the
            trajectories (speed, effSpeed, peclet, rate, rho, flux). All of
            them by default, so any of them can be used to colour pathlines
            *or* vectors without re-integrating; pass a shorter tuple to save
            the sampling cost and the (M, nSample) array each one occupies.

    Returns:
        dict ``Lag`` with ``SL`` (list of (steps,3) pathlines in ROI voxel
        coords row,col,slice), ``streams`` ({quantity: (M, nSample)} samples
        along each path) with ``sstream`` / ``pestream`` kept as per-path views
        of the speed / Peclet streams, ``stats`` ({quantity: {mean, median,
        max}}), ``segVals`` (per-segment values, filled on demand by
        :func:`segmentValues`), ``startp`` (M,3), ``disp`` (M,3) and ``displen``
        (M,) net displacements (mm), ``ind_msk``, and grid metadata.
    """
    n = [int(v) for v in result["n"]]
    h = [float(v) for v in result["spacing"]]
    sigma = float(result.get("sigma", 0.0))
    dt = float(result["dt"])
    gc = _cellCenters(n, h)

    # ---- seed points: every spfs-th masked voxel (cell-centered coords) ----
    mask3 = np.asarray(result["mask"] if seedMask is None else seedMask) > 0
    seed = np.zeros_like(mask3)
    s = max(1, int(spfs))
    seed[::s, ::s, ::s] = True
    seed &= mask3
    seedMask = seed
    ri, ci, si = np.where(seedMask)
    if maxSeeds and ri.size > maxSeeds:                 # optional uniform thin
        sel = np.linspace(0, ri.size - 1, maxSeeds).astype(int)
        ri, ci, si = ri[sel], ci[sel], si[sel]
    startVox = np.stack([ri, ci, si], axis=1).astype(float)
    pos = np.stack([(ri + 0.5) * h[0], (ci + 0.5) * h[1],
                    (si + 0.5) * h[2]], axis=1)         # (M,3) physical
    M = pos.shape[0]

    # Pre-allocated history arrays (avoid a per-seed Python append loop, which
    # was O(nIntervals*nt*nEuler*M) and dominated the full-run cost): one
    # recorded position per Euler sub-step plus the seed, speed/Peclet per
    # sub-step. Vectorized assignment over all M seeds at once.
    nEulerI = int(max(1, nEuler))
    nGlob = sum(1 for _ in _globalSteps(result))
    nRec = 1 + nGlob * nEulerI
    voxHist = np.empty((nRec, M, 3))
    voxHist[0] = startVox
    # One history plane per sampled quantity, not just speed/Peclet: the whole
    # point of the sampling pass is that every displayed metric comes from the
    # SAME interpolation of the same per-step fields, so a colour-by choice in
    # the GUI is a lookup rather than a re-integration.
    # `None` means the full set, not "none": an empty sample store would leave
    # the pathline display with nothing to colour by.
    qKeys = tuple(q for q in (QUANTITIES if quantities is None else quantities)
                  if q in QUANTITIES)
    qHist = {q: np.empty((nGlob * nEulerI, M)) for q in qKeys}
    mdt = dt / float(nEulerI)
    hInv = 1.0 / np.asarray(h)

    # ---- integrate through the time-varying effective velocity ------------
    rec = 0
    for _g, v, rho, r in _globalSteps(result):
        # NB: the pathline Peclet floors at machine epsilon, matching the
        # reference. The Eulerian maps instead use the `peclet_floor`
        # setting (an additive fraction of the median diffusive speed) to
        # keep displayed maps stable, so the two are not identically
        # scaled - see _pecletDenomFloor.
        vEff, qVals = _stepQuantities(v, rho, r, n, h, sigma, keys=qKeys)
        # Velocity components AND every sampled quantity in ONE interpolator:
        # they are read at the same points, so the weights are computed once.
        interp = _stackInterpolator(
            [vEff[c].reshape(n, order="F") for c in range(3)]
            + [qVals[q].reshape(n, order="F") for q in qKeys], gc)
        for _sub in range(nEulerI):
            sample = interp(pos)                       # (M, 3 + nQuantities)
            vv = sample[:, :3]
            for j, q in enumerate(qKeys):
                qHist[q][rec] = sample[:, 3 + j]
            moving = np.sqrt(np.sum(vv ** 2, axis=1)) >= minSpeed
            pos = pos + direction * mdt * vv * moving[:, None]
            rec += 1
            voxHist[rec] = (pos - 0.5 * np.asarray(h)) * hInv   # physical->vox

    # ---- assemble, filter near-stationary pathlines -----------------------
    disp = voxHist[-1] - voxHist[0]                      # (M,3) voxels
    keep = np.where(np.linalg.norm(disp, axis=1) >= slTolVox)[0]
    SL, keepStart, dispV = [], [], []
    for m in keep:
        SL.append(voxHist[:, m, :].copy())
        keepStart.append(startVox[m])
        dispV.append(disp[m] * np.asarray(h))            # cm displacement
    # Per-quantity sample streams, (M, nSample) with one row per surviving
    # pathline. Kept as one array per quantity rather than a list per path:
    # every reduction below is then a single vectorized call (a Python loop
    # over 66k paths costs ~1.5 s per statistic).
    streams = {q: np.ascontiguousarray(qHist[q][:, keep].T) for q in qKeys}
    startp = np.asarray(keepStart) if keepStart else np.zeros((0, 3))
    dispCm = np.asarray(dispV) if dispV else np.zeros((0, 3))
    displen = (np.linalg.norm(dispCm, axis=1) if dispCm.size
               else np.zeros(0))
    # Back-compatible aliases: `sstream` / `pestream` are rows of the arrays
    # above (views, no copy), so older callers keep working unchanged.
    sstream = list(streams["speed"]) if "speed" in streams else []
    pestream = list(streams["peclet"]) if "peclet" in streams else []

    Lag = dict(SL=SL, streams=streams, sstream=sstream, pestream=pestream,
               startp=startp,
               disp=dispCm, displen=displen, ind_msk=np.arange(len(SL)),
               n=n, spacing=h, bbox=result["bbox"], mask=result["mask"],
               frameScanNums=result.get("frameScanNums"))

    # ---- display-ready derivatives, computed ONCE here --------------------
    # These depend only on the solved trajectories, never on a display option,
    # so they belong to the run rather than to a redraw: the viewer used to
    # rebuild every one of them each time any control changed (colour by,
    # opacity, grow, density, line width, timepoint), which on a 66k-path ROI
    # is seconds of pure recomputation before anything is drawn. They travel
    # with the run through planC.urOMT.
    nVert = int(SL[0].shape[0]) if SL else 0
    Lag["nVert"] = nVert
    # Per-path mean/median/max of EVERY sampled quantity - the whole colour-by
    # menu, reduced once. Each is a single vectorized reduction over an
    # (M, nSample) array; the per-path Python loop it replaces cost ~1.5 s per
    # statistic on a 66k-path ROI.
    Lag["stats"] = {q: pathSpeedStats(streams[q]) for q in qKeys}
    # Per-SEGMENT values are NOT all precomputed: each is another (M, nVert-1)
    # array (~17 MB at breast scale) and only the displayed quantity is ever
    # needed. `segmentValues` computes one on demand and caches it here, so the
    # second look-up is free.
    Lag["segVals"] = {}
    # Back-compatible aliases for the speed/Peclet-only keys the viewer and the
    # stored runs used before the colour-by generalization.
    Lag["speedStats"] = Lag["stats"].get("speed", pathSpeedStats([]))
    Lag["pecletStats"] = Lag["stats"].get("peclet", pathSpeedStats([]))
    Lag["segSpeed"] = (segmentValues(Lag, "speed") if "speed" in qKeys
                       else None)
    Lag["seedVox"] = (np.rint(startp).astype(int) if startp.size
                      else np.zeros((0, 3), dtype=int))
    Lag["dispVox"] = (np.linalg.norm(disp[keep], axis=1) if len(keep)
                      else np.zeros(0))              # net displacement, VOXELS
    return Lag


def alignSpeedToVertices(sp, nVert):
    """Pad/trim a per-sub-step speed array to one value per pathline vertex.

    ``runGLAD`` samples the speed at the position *before* each move, so there
    are ``nVert - 1`` samples for ``nVert`` recorded positions. Repeating the
    first sample keeps sample *i* aligned with the vertex it was measured at.
    """
    sp = np.asarray(sp, dtype=float).ravel()
    if sp.size == 0:
        return np.zeros(nVert)
    if sp.size >= nVert:
        return sp[:nVert]
    return np.concatenate([sp, np.repeat(sp[-1], nVert - sp.size)])


def alignStreamToVertices(arr, nVert):
    """Pad/trim an (M, nSample) sample stream to one column per pathline vertex.

    The array form of :func:`alignSpeedToVertices` - ``runGLAD`` samples every
    quantity at the position *before* each move, giving ``nVert - 1`` samples
    for ``nVert`` recorded positions, so the last column is repeated.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] == nVert:
        return arr
    if arr.shape[1] > nVert:
        return arr[:, :nVert]
    pad = np.repeat(arr[:, -1:], nVert - arr.shape[1], axis=1)
    return np.concatenate([arr, pad], axis=1)


def pathStats(Lag, quantity="speed"):
    """Per-path ``{mean, median, max}`` of ``quantity`` for a ``runGLAD`` result.

    Reads the precomputed ``Lag['stats']``; falls back to reducing the stream
    (and to the legacy ``sstream``/``pestream`` keys of runs stored before the
    per-quantity streams existed), so older stored runs keep displaying.
    """
    stats = Lag.get("stats") or {}
    if quantity in stats:
        return stats[quantity]
    st = streamOf(Lag, quantity)
    if st is None:
        return None
    out = pathSpeedStats(st)
    Lag.setdefault("stats", {})[quantity] = out
    return out


def streamOf(Lag, quantity="speed"):
    """The (M, nSample) sample stream of ``quantity``, or ``None``.

    Falls back to the legacy per-path lists (``sstream`` for speed, ``pestream``
    for Peclet) so runs stored before the generalization still colour by those
    two quantities.
    """
    st = (Lag.get("streams") or {}).get(quantity)
    if st is not None:
        return np.asarray(st)
    legacy = {"speed": "sstream", "peclet": "pestream"}.get(quantity)
    if legacy and Lag.get(legacy) is not None and len(Lag[legacy]):
        return np.asarray(Lag[legacy], dtype=float)
    return None


def segmentValues(Lag, quantity="speed"):
    """Per-SEGMENT values of ``quantity``, aligned to the drawn vertices.

    (M, nVert-1) array, computed on first request and cached in
    ``Lag['segVals']`` - one such array per quantity is ~17 MB at breast scale,
    so they are built for the displayed quantity only.
    """
    cache = Lag.setdefault("segVals", {})
    if quantity in cache:
        return cache[quantity]
    st = streamOf(Lag, quantity)
    if st is None or not st.size:
        cache[quantity] = None
        return None
    nVert = int(Lag.get("nVert") or 0)
    if not nVert:
        SL = Lag.get("SL") or []
        nVert = int(np.asarray(SL[0]).shape[0]) if len(SL) else 0
    aligned = alignStreamToVertices(st, nVert) if nVert else st
    out = (0.5 * (aligned[:, :-1] + aligned[:, 1:])
           if aligned.shape[1] >= 2 else None)
    cache[quantity] = out
    return out


def scalePathline(pts, scale):
    """Shrink (or stretch) one pathline about its seed by ``scale``.

    Keeps the seed anchored and the trajectory's shape intact, scaling only how
    far it reaches - the length equivalent of the vector overlay's arrow-length
    scale. Use it to stop long excursions from crossing the whole field of view.
    ``scale == 1`` is a no-op.
    """
    pts = np.asarray(pts, dtype=float)
    scale = float(scale)
    if scale == 1.0 or pts.shape[0] == 0:
        return pts
    return pts[0] + (pts - pts[0]) * scale


def pathSpeedStats(streams):
    """Per-path mean / median / max of a per-vertex quantity.

    ``runGLAD`` integrates every seed over the same time steps, so the arrays
    are equal length and the whole thing is one vectorized reduction. Doing it
    per path in Python costs ~1.5 s for a 66k-path ROI - enough to make the
    cheap-to-draw flat colouring slower overall than the per-vertex gradient it
    was meant to beat.
    """
    if isinstance(streams, np.ndarray) and streams.ndim == 2:
        if not streams.size:
            z = np.zeros(streams.shape[0])
            return {"mean": z, "median": z.copy(), "max": z.copy()}
        return {"mean": streams.mean(1), "median": np.median(streams, 1),
                "max": streams.max(1)}
    if not len(streams):
        z = np.zeros(0)
        return {"mean": z, "median": z.copy(), "max": z.copy()}
    n0 = len(streams[0])
    if all(len(s) == n0 for s in streams):
        arr = np.asarray(streams, dtype=float)           # (M, nVert)
        return {"mean": arr.mean(1), "median": np.median(arr, 1),
                "max": arr.max(1)}
    return {k: np.array([f(s) if len(s) else 0.0 for s in streams])
            for k, f in (("mean", np.mean), ("median", np.median),
                         ("max", np.max))}


def _segmentValues(streams):
    """Midpoint value of each pathline SEGMENT, for along-path colouring.

    Returns an (M, nVert-1) array when the paths share a vertex count (they do),
    else a list. ``None`` when there is nothing to reduce.
    """
    if not len(streams):
        return None
    n0 = len(streams[0])
    if all(len(s) == n0 for s in streams):
        if n0 < 2:
            return None
        arr = np.asarray(streams, dtype=float)
        return 0.5 * (arr[:, :-1] + arr[:, 1:])
    return [0.5 * (np.asarray(s)[:-1] + np.asarray(s)[1:])
            if len(s) > 1 else np.zeros(0) for s in streams]


def lagrangianMaps(Lag, keys=("speed", "peclet")):
    """Accumulate per-pathline samples onto the ROI grid (runGLAD -> maps).

    The reference implementation stores its Lagrangian output as whole-run grid
    maps, whereas :func:`runGLAD` returns per-pathline arrays; this bridges the
    two so Part 4 can be compared with the reference, exported, or displayed as
    a colourwash. Each voxel gets the MEAN of every pathline sample that falls
    in it (a per-voxel minimum, tried against the reference, agrees far worse:
    correlation 0.32 vs 0.53).

    Args:
        Lag (dict): output of :func:`runGLAD`.
        keys: which metrics to accumulate - any of :data:`QUANTITIES`
            sampled by :func:`runGLAD` (speed, effSpeed, peclet, rate, rho,
            flux); runs stored before per-quantity sampling only carry speed
            and Peclet.

    Returns:
        dict: ``{key: (n0, n1, n2) ndarray}``, zero where no pathline passed.
    """
    n = [int(v) for v in Lag["n"]]
    out = {}
    for key in keys:
        streams = streamOf(Lag, key)
        if streams is None:
            continue
        tot = np.zeros(n)
        cnt = np.zeros(n)
        for pl, val in zip(Lag["SL"], streams):
            val = np.asarray(val, dtype=float)
            k = min(val.size, np.asarray(pl).shape[0])
            if k == 0:
                continue
            vox = np.rint(np.asarray(pl)[:k]).astype(int)
            ok = np.all((vox >= 0) & (vox < np.asarray(n)), axis=1)
            if not ok.any():
                continue
            ix = (vox[ok, 0], vox[ok, 1], vox[ok, 2])
            np.add.at(tot, ix, val[:k][ok])
            np.add.at(cnt, ix, 1.0)
        m = cnt > 0
        arr = np.zeros(n)
        arr[m] = tot[m] / cnt[m]
        out[key] = arr
    return out
