"""urOMT result visualization helpers.

Map the per-interval urOMT velocity field (defined on the ROI bounding-box grid
in pyCERR virtual coordinates) into the napari ``vectors_dict`` consumed by
:func:`cerr.viewer.pycerr_napari.showNapari`, so the recovered fluid transport
can be overlaid on the scans/ROI as a 3-D vector field.

The vector array follows the pyCERR/DVF convention (see
``register.getDvfVectors``)::

    vectors[i, 0, :] = [rStart, cStart, sStart]    # scan voxel (row,col,slice)
    vectors[i, 1, :] = [dy, dx, dz]                # cm, virtual coords

where (dy, dx, dz) are the velocity components along (row, col, slice). The
array building has no napari/Qt dependency (testable headlessly); only
:func:`showVelocity` imports the viewer.
"""

import numpy as np


def velocityVectors(result, interval=0, step=None, subsample=2,
                    speedPctile=60.0, lengthScale=1.0, maxVectors=20000):
    """Build a napari ``vectors_dict`` for one urOMT interval's velocity field.

    Args:
        result (dict): output of :func:`cerr.uromt.solver.runUROMT`.
        interval (int): which time interval's velocity to show.
        step (int): inner sub-step index (0..nt-1); ``None`` -> time-mean.
        subsample (int): keep every ``subsample``-th voxel per axis (thinning).
        speedPctile (float): drop vectors below this percentile of |velocity|
            (over the ROI), reducing clutter. 0 keeps all.
        lengthScale (float): multiply velocity (mm/time) to set arrow length.
        maxVectors (int): hard cap; if exceeded, thin further by |velocity|.

    Returns:
        dict: ``{'vectors': (m,2,3) ndarray, 'features': {...},
        'scanNum': int}`` ready for ``showNapari(..., vectors_dict=...)``.
    """
    u = result["u"][interval]                       # (3, N, nt)
    n = [int(v) for v in result["n"]]
    N = int(np.prod(n))
    rs_, _, cs_, _, ss_, _ = result["bbox"]
    sf = float(result.get("sizeFactor", 1.0)) if result.get("doResize", 0) else 1.0

    if step is None:
        comp = u.mean(axis=2)                        # (3, N) time-mean velocity
    else:
        comp = u[:, :, int(step)]
    u0 = comp[0].reshape(n, order="F")               # row (y) velocity
    u1 = comp[1].reshape(n, order="F")               # col (x) velocity
    u2 = comp[2].reshape(n, order="F")               # slice (z) velocity

    mask = np.asarray(result["mask"]) > 0
    i1, i2, i3 = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]),
                             indexing="ij")
    keep = mask.copy()
    sub = np.zeros(n, dtype=bool)
    s = max(1, int(subsample))
    sub[::s, ::s, ::s] = True
    keep &= sub

    speed = np.sqrt(u0 ** 2 + u1 ** 2 + u2 ** 2)
    if speedPctile and speedPctile > 0:
        roiSpeed = speed[mask]
        thr = np.percentile(roiSpeed, speedPctile) if roiSpeed.size else 0.0
        keep &= speed >= thr

    idx = np.where(keep)
    nKeep = idx[0].size
    if nKeep == 0:
        return dict(vectors=np.zeros((0, 2, 3)), features={}, scanNum=0)
    if nKeep > maxVectors:                           # keep the fastest
        order = np.argsort(-speed[idx])[:maxVectors]
        idx = tuple(a[order] for a in idx)

    ri, ci, si = idx
    # ROI-local indices -> full scan voxel coordinates (account for resize)
    rStart = rs_ + ri / sf
    cStart = cs_ + ci / sf
    sStart = ss_ + si / sf
    dy = lengthScale * u0[idx]
    dx = lengthScale * u1[idx]
    dz = lengthScale * u2[idx]

    m = ri.size
    vectors = np.zeros((m, 2, 3))
    vectors[:, 0, 0] = rStart
    vectors[:, 0, 1] = cStart
    vectors[:, 0, 2] = sStart
    vectors[:, 1, 0] = dy
    vectors[:, 1, 1] = dx
    vectors[:, 1, 2] = dz

    spd = speed[idx]
    features = {"speed (mm/t)": spd,
                "|dy| row": np.abs(u0[idx]),
                "|dx| col": np.abs(u1[idx]),
                "|dz| slice": np.abs(u2[idx])}

    scanNum = 0
    fsn = result.get("frameScanNums")
    if fsn:                                           # the interval start frame
        scanNum = int(fsn[min(interval, len(fsn) - 1)])
    return dict(vectors=vectors, features=features, scanNum=scanNum)


def showVelocity(planC, result, structNum=None, interval=0, step=None,
                 subsample=2, speedPctile=60.0, lengthScale=1.0,
                 displayMode="3d"):
    """Overlay an urOMT velocity field on the scans/ROI in napari.

    Displays the interval's start-frame scan (and the ROI structure, if given)
    with the velocity vectors. Returns the napari viewer.
    """
    from cerr.viewer.pycerr_napari import showNapari        # lazy (Qt/GL)
    vd = velocityVectors(result, interval=interval, step=step,
                         subsample=subsample, speedPctile=speedPctile,
                         lengthScale=lengthScale)
    scanNum = vd.get("scanNum", 0)
    structNums = [] if structNum is None else [structNum]
    out = showNapari(planC, scan_nums=scanNum, struct_nums=structNums,
                     dose_nums=[], vectors_dict=vd, displayMode=displayMode)
    return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------------- #
#  Part 5 visualizations: Eulerian maps/flux + Lagrangian pathlines
# --------------------------------------------------------------------------- #
def _scanNumOf(meta, index=0):
    fsn = meta.get("frameScanNums")
    if fsn:
        return int(fsn[min(index, len(fsn) - 1)])
    return 0


def fieldVectors(field3N, n, bbox, mask=None, scanNum=0, subsample=2,
                 magPctile=60.0, lengthScale=1.0, sizeFactor=1.0, doResize=0,
                 featureName="magnitude", maxVectors=20000):
    """Generic ROI-grid (3,N) vector field -> napari ``vectors_dict`` (scan
    voxel coords). Used for the Eulerian flux field; mirrors
    :func:`velocityVectors`."""
    n = [int(v) for v in n]
    rs_, _, cs_, _, ss_, _ = bbox
    sf = float(sizeFactor) if doResize else 1.0
    f0 = field3N[0].reshape(n, order="F")
    f1 = field3N[1].reshape(n, order="F")
    f2 = field3N[2].reshape(n, order="F")
    keep = (np.asarray(mask) > 0) if mask is not None else np.ones(n, bool)
    sub = np.zeros(n, dtype=bool)
    st = max(1, int(subsample))
    sub[::st, ::st, ::st] = True
    keep = keep & sub
    mag = np.sqrt(f0 ** 2 + f1 ** 2 + f2 ** 2)
    if magPctile and magPctile > 0:
        roi = mag[(np.asarray(mask) > 0)] if mask is not None else mag.ravel()
        thr = np.percentile(roi, magPctile) if roi.size else 0.0
        keep &= mag >= thr
    idx = np.where(keep)
    if idx[0].size == 0:
        return dict(vectors=np.zeros((0, 2, 3)), features={}, scanNum=scanNum)
    if idx[0].size > maxVectors:
        order = np.argsort(-mag[idx])[:maxVectors]
        idx = tuple(a[order] for a in idx)
    ri, ci, si = idx
    m = ri.size
    vectors = np.zeros((m, 2, 3))
    vectors[:, 0, 0] = rs_ + ri / sf
    vectors[:, 0, 1] = cs_ + ci / sf
    vectors[:, 0, 2] = ss_ + si / sf
    vectors[:, 1, 0] = lengthScale * f0[idx]
    vectors[:, 1, 1] = lengthScale * f1[idx]
    vectors[:, 1, 2] = lengthScale * f2[idx]
    features = {featureName: mag[idx]}
    return dict(vectors=vectors, features=features, scanNum=int(scanNum))


def eulerianFluxVectors(Eul, subsample=2, magPctile=60.0, lengthScale=1.0):
    """napari ``vectors_dict`` for the Eulerian mean-flux field (Part 5)."""
    return fieldVectors(Eul["flux"], Eul["n"], Eul["bbox"], mask=Eul["mask"],
                        scanNum=_scanNumOf(Eul), subsample=subsample,
                        magPctile=magPctile, lengthScale=lengthScale,
                        featureName="flux")


def eulerianMapToScan(Eul, field="speed", scanShape=None, planC=None,
                      scanNum=None):
    """Embed an Eulerian ROI-grid map (``speed``/``rate``/``peclet``) back into
    a full scan-sized array for overlay. Returns the full-grid float array.

    When the run was computed at reduced resolution (``do_resize``), the ROI map
    is smaller than its bounding box, so it is zoomed back up to the bbox extent
    (mirrors :func:`fieldToScan`) before insertion - otherwise the assignment
    broadcasts a half-size array into the full-size bbox slice and raises."""
    from scipy.ndimage import zoom
    key = {"speed": "speed3", "rate": "rate3", "peclet": "peclet3"}[field]
    roiMap = np.asarray(Eul[key])
    rs_, re_, cs_, ce_, ss_, se_ = Eul["bbox"]
    if scanShape is None:
        if planC is not None:
            sn = _scanNumOf(Eul) if scanNum is None else scanNum
            scanShape = tuple(int(v) for v in
                              planC.scan[sn].getScanArray().shape)
        else:
            raise ValueError("provide scanShape or planC to size the full grid")
    target = (re_ - rs_, ce_ - cs_, se_ - ss_)
    if roiMap.shape != target:                        # resized run -> zoom to bbox
        roiMap = zoom(roiMap, [t / s for t, s in zip(target, roiMap.shape)],
                      order=1)
    full = np.zeros(scanShape, dtype=float)
    full[rs_:re_, cs_:ce_, ss_:se_] = roiMap
    return full


def pathlineTracks(Lag, colorBy="speed", maxTracks=2000, minVertices=3):
    """Build a napari Tracks-layer array from Lagrangian pathlines (Part 5).

    Returns ``(data, properties)`` where ``data`` is (K, 5) columns
    ``[track_id, t, row, col, slice]`` in scan voxel coords and ``properties``
    holds a per-vertex feature (``speed`` or ``peclet``) for colouring.
    """
    rs_, _, cs_, _, ss_, _ = Lag["bbox"]
    feat = {"speed": Lag["sstream"], "peclet": Lag["pestream"]}[colorBy]
    SL = Lag["SL"]
    idx = range(len(SL))
    if len(SL) > maxTracks:
        idx = np.linspace(0, len(SL) - 1, maxTracks).astype(int)
    rows = []
    vals = []
    tid = 0
    for i in idx:
        pl = np.asarray(SL[i])
        if pl.shape[0] < minVertices:
            continue
        fv = np.asarray(feat[i])
        # feat has one value per advanced vertex (path has start + those)
        fvFull = np.concatenate([fv[:1], fv]) if fv.size == pl.shape[0] - 1 \
            else fv
        fvFull = np.resize(fvFull, pl.shape[0])
        for t in range(pl.shape[0]):
            rows.append([tid, t, pl[t, 0] + rs_, pl[t, 1] + cs_,
                         pl[t, 2] + ss_])
            vals.append(fvFull[t])
        tid += 1
    data = np.asarray(rows, dtype=float) if rows else np.zeros((0, 5))
    return data, {colorBy: np.asarray(vals)}


def _scanAffine(out):
    """Recover the displayed scan layer's affine matrix from a showNapari
    return so overlays (maps, tracks) align with the affine-placed scan."""
    try:
        scanLayers = out[1] if isinstance(out, tuple) and len(out) > 1 else []
        if scanLayers:
            return scanLayers[0].affine.affine_matrix
    except Exception:  # noqa: BLE001
        pass
    return None


def fieldToScan(field3N, n, bbox, scanShape, sizeFactor=1.0, doResize=0):
    """Embed an ROI-grid (3,N) vector field into three full scan-grid arrays,
    one per component (axis 0=row/y, 1=col/x, 2=slice/z), for overlaying on the
    main viewer. Zero outside the ROI bbox; resized ROIs are zoomed to fit."""
    from scipy.ndimage import zoom
    n = [int(v) for v in n]
    rs_, re_, cs_, ce_, ss_, se_ = bbox
    target = (re_ - rs_, ce_ - cs_, se_ - ss_)
    comps = []
    for c in range(3):
        roi = np.asarray(field3N[c]).reshape(n, order="F")
        if roi.shape != target:
            roi = zoom(roi, [t / s for t, s in zip(target, roi.shape)], order=1)
        full = np.zeros(scanShape, dtype=float)
        full[rs_:re_, cs_:ce_, ss_:se_] = roi
        comps.append(full)
    return comps                                      # [fy(row), fx(col), fz(slc)]


def pathlinesToScanVox(Lag, sizeFactor=1.0, doResize=0):
    """Map ROI-voxel pathlines to full scan voxel coordinates (row,col,slice)
    and return ``(segs, vals)`` where ``segs`` is a list of (steps,3) arrays and
    ``vals`` the per-pathline mean speed (for colouring)."""
    rs_, _, cs_, _, ss_, _ = Lag["bbox"]
    off = np.array([rs_, cs_, ss_], dtype=float)
    sf = float(sizeFactor) if doResize else 1.0
    segs, vals = [], []
    for pl, sp in zip(Lag["SL"], Lag["sstream"]):
        pl = np.asarray(pl, dtype=float) / sf + off
        segs.append(pl)
        vals.append(float(np.mean(sp)) if len(sp) else 0.0)
    return segs, np.asarray(vals)


def overlayTo3D(ov, xV, yV, zV, maxArrows=1500, maxPaths=400):
    """Build 3-D urOMT overlay geometry from a cached overlay ``ov`` (the dict
    produced by ``PyCerrViewer.set_uromt_overlay``) and the scan's physical
    coordinate axes ``xV`` (col), ``yV`` (row), ``zV`` (slice).

    Returns a dict (or ``None``) with optional keys:

    * ``vectors``: ``points`` (M,3), ``vec`` (M,3, scaled so the longest arrow
      spans ~5% of the field of view, keeping arrows inside the scan), ``mag``
      (M,) raw magnitudes, ``tip`` (M,3) arrow-head points.
    * ``paths``: list of (steps,3) pathline polylines in physical coords.
    * ``scalar``: ``points`` (M,3) and ``vals`` (M,) for a colour-coded point
      cloud of an Eulerian map (speed / rate / Peclet).

    Pure NumPy (no Qt / pyvista), so the coordinate mapping and arrow scaling are
    headless-testable; the GUI 3-D renderers only consume the result.
    """
    if ov is None:
        return None
    xV = np.asarray(xV, float)
    yV = np.asarray(yV, float)
    zV = np.asarray(zV, float)
    spanFOV = max(abs(float(xV[-1] - xV[0])), abs(float(yV[-1] - yV[0])),
                  abs(float(zV[-1] - zV[0])) if zV.size > 1 else 0.0) or 1.0
    out = {}
    if "map3" in ov:                                   # scalar map point cloud
        m3 = np.asarray(ov["map3"])
        ri, ci, si = np.where(m3 != 0)
        if ri.size:
            if ri.size > maxArrows:
                sel = np.linspace(0, ri.size - 1, maxArrows).astype(int)
                ri, ci, si = ri[sel], ci[sel], si[sel]
            out["scalar"] = dict(
                points=np.column_stack([xV[ci], yV[ri], zV[si]]).astype(float),
                vals=m3[ri, ci, si].astype(float))
    if "comps" in ov:                                  # velocity / flux arrows
        cy, cx, cz = ov["comps"]                       # row(y), col(x), slice(z)
        mag = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        ri, ci, si = np.where(mag > 0)
        if ri.size:
            if ri.size > maxArrows:                     # thin uniformly
                sel = np.linspace(0, ri.size - 1, maxArrows).astype(int)
                ri, ci, si = ri[sel], ci[sel], si[sel]
            pts = np.column_stack([xV[ci], yV[ri], zV[si]]).astype(float)
            vec = np.column_stack([cx[ri, ci, si], cy[ri, ci, si],
                                   cz[ri, ci, si]]).astype(float)
            m = np.linalg.norm(vec, axis=1)
            mmax = float(m.max()) or 1.0
            vecS = vec * (0.05 * spanFOV / mmax)        # longest ~5% of FOV
            out["vectors"] = dict(points=pts, vec=vecS, mag=m, tip=pts + vecS)
    if "segs" in ov:                                    # pathlines
        segs, _vals = ov["segs"]
        order = range(len(segs))
        if len(segs) > maxPaths:
            order = np.linspace(0, len(segs) - 1, maxPaths).astype(int)
        axR = np.arange(yV.size)
        axC = np.arange(xV.size)
        axS = np.arange(zV.size)
        lines = []
        for i in order:
            pl = np.asarray(segs[i], dtype=float)
            if pl.shape[0] < 2:
                continue
            x = np.interp(pl[:, 1], axC, xV)            # col -> x
            y = np.interp(pl[:, 0], axR, yV)            # row -> y
            z = np.interp(pl[:, 2], axS, zV)            # slice -> z
            lines.append(np.column_stack([x, y, z]))
        if lines:
            out["paths"] = lines
    return out or None


def _overlayColorbar(ax, cmap, lo, hi, label, alpha=0.95):
    """Draw a compact colorbar legend for a urOMT overlay in the corner of the
    slice axes ``ax``, in axes-fraction coordinates so it survives the viewer's
    per-frame artist clearing (it uses only patches + texts, not a child axes,
    and never touches the axes aspect). Shows the metric name and its ``lo``..
    ``hi`` range with the colormap used for the overlay."""
    import matplotlib
    from matplotlib.patches import Rectangle
    getc = (matplotlib.colormaps[cmap] if hasattr(matplotlib, "colormaps")
            else matplotlib.cm.get_cmap(cmap))
    x0, y0, w, h = 0.935, 0.10, 0.03, 0.34
    nseg = 32
    for i in range(nseg):
        frac = i / (nseg - 1)
        ax.add_patch(Rectangle((x0, y0 + frac * h), w, h / nseg * 1.06,
                     transform=ax.transAxes, facecolor=getc(frac),
                     edgecolor="none", alpha=alpha, zorder=10, clip_on=False))
    ax.add_patch(Rectangle((x0, y0), w, h, transform=ax.transAxes, fill=False,
                 edgecolor="white", lw=0.6, zorder=11, clip_on=False))
    tkw = dict(transform=ax.transAxes, color="white", fontsize=7, zorder=11,
               ha="left", va="center", clip_on=False)
    ax.text(x0 + w + 0.012, y0 + h, "%.3g" % hi, **tkw)
    ax.text(x0 + w + 0.012, y0, "%.3g" % lo, **tkw)
    if lo < 0 < hi:                                   # mark zero for diverging
        ax.text(x0 + w + 0.012, y0 + h * (-lo / (hi - lo)), "0", **tkw)
    ax.text(x0, y0 + h + 0.02, label, transform=ax.transAxes, color="#e8c542",
            fontsize=7, ha="left", va="bottom", zorder=11, clip_on=False)


def drawUROMTOverlay(ax, ov, k, hV, vV, extent, slicer, hAxis, vAxis,
                     thruAxis, scanShape, alpha=0.6, subsample=None,
                     sliceTol=2.0, cmap="turbo", colorbar=True):
    """Draw a urOMT overlay (precomputed in ``ov``) onto an existing viewer
    slice ``ax`` for one orientation. Pure matplotlib (headless-testable).

    ``ov`` holds the cached full scan-grid data for the chosen view:
    ``map3`` (scalar), ``comps`` (3 vector components) or ``segs`` (pathlines),
    plus a global ``vrange`` (lo, hi) and ``label`` so the colour-coding (and the
    drawn colorbar legend) is **consistent across slices and orientations**.
    ``hAxis``/``vAxis``/``thruAxis`` are the scan-array axes (0=row,1=col,2=slc)
    that map to the horizontal / vertical / through-plane directions of this
    view; ``slicer`` slices a full scan-grid array for the current slice. The
    vector ``subsample`` defaults to 1 (one arrow per voxel); ``ov['subsample']``
    overrides it.
    """
    view = ov.get("view")
    cmName = "bwr" if view == "rate" else cmap
    lo, hi = ov.get("vrange", (None, None))
    label = ov.get("label", view or "urOMT")
    sub = int(ov.get("subsample", subsample or 1) or 1)
    drewSomething = False
    if "map3" in ov:                                  # scalar colourwash
        m = np.ma.masked_equal(slicer(ov["map3"]), 0.0)
        vmin = lo if lo is not None else None
        vmax = hi if hi is not None else None
        ax.imshow(m, cmap=cmName, extent=extent, alpha=alpha, vmin=vmin,
                  vmax=vmax, interpolation="nearest", aspect="equal", zorder=3)
        drewSomething = lo is not None
    elif "comps" in ov:                               # velocity / flux quiver
        comps = ov["comps"]
        U = slicer(comps[hAxis])
        V = slicer(comps[vAxis])
        H, Vm = np.meshgrid(hV, vV)
        Hs, Vs = H[::sub, ::sub], Vm[::sub, ::sub]
        Us, Vss = U[::sub, ::sub], V[::sub, ::sub]
        Ms = np.sqrt(Us ** 2 + Vss ** 2)
        gmax = hi if (hi is not None and hi > 0) else (
            float(Ms.max()) if Ms.size else 0.0)
        if gmax <= 0:
            return
        # Scale arrows by the GLOBAL max so arrow length is comparable across
        # slices, and so the longest possible arrow spans a small fraction of
        # the field of view (keeps arrows inside the scan).
        spanH = abs(float(hV[-1] - hV[0])) if len(hV) > 1 else 1.0
        scale = gmax / (0.08 * (spanH or 1.0))         # data units per arrow unit
        drawn = Ms > 0.02 * gmax                       # skip near-zero background
        if drawn.any():
            hT, vT = Hs[drawn], Vs[drawn]
            mT = Ms[drawn]
            # Clip arrow LENGTH at the global max (p99): a few near-boundary
            # voxels can have non-physical huge velocity (the urOMT velocity
            # degeneracy), which would otherwise shoot arrows off the grid.
            clip = np.minimum(1.0, gmax / (mT + 1e-12))
            uT = Us[drawn] * clip
            vvT = Vss[drawn] * clip
            ax.quiver(hT, vT, uT, vvT, mT, cmap=cmName, alpha=alpha,
                      angles="xy", scale_units="xy", scale=scale, pivot="tail",
                      width=0.003, clim=(0.0, gmax), zorder=4)
            # green start (tail) marker - useful when sparse but clutters a
            # dense field, so only draw below a modest arrow count. The stop is
            # shown by the arrowhead, so no separate (red) stop marker.
            if hT.size <= 800:
                ax.scatter(hT, vT, s=6, c="#39ff14", edgecolors="none",
                           alpha=min(1.0, alpha + 0.25), zorder=5)     # start
        lo, hi = 0.0, gmax
        drewSomething = True
    elif "segs" in ov:                                # pathlines near the slice
        import matplotlib
        getc = (matplotlib.colormaps[cmap]
                if hasattr(matplotlib, "colormaps")
                else matplotlib.cm.get_cmap(cmap))
        segs, vals = ov["segs"]
        vmax = hi if (hi is not None and hi > 0) else (
            float(vals.max()) if len(vals) else 1.0)
        hx = np.arange(scanShape[hAxis])
        vx = np.arange(scanShape[vAxis])
        for seg, val in zip(segs, vals):
            near = np.abs(seg[:, thruAxis] - k) <= sliceTol
            if not near.any():
                continue
            h = np.where(near, np.interp(seg[:, hAxis], hx, hV), np.nan)
            v = np.where(near, np.interp(seg[:, vAxis], vx, vV), np.nan)
            ax.plot(h, v, "-", lw=0.8, alpha=alpha,
                    color=getc(min(val / (vmax + 1e-9), 1.0)))
        lo, hi = 0.0, vmax
        drewSomething = True

    if colorbar and drewSomething and hi is not None:
        _overlayColorbar(ax, cmName, lo, hi, label)


def drawUROMTSlice(fig, result, Eul=None, Lag=None, view="speed", axis=2,
                   sliceIdx=None, bg=None, cmap="turbo", subsample=None,
                   interval=0):
    """Draw one urOMT result view on a matplotlib Figure (for the embedded Qt
    viewer; no napari/Qt dependency here so it is headless-testable).

    Args:
        fig: a matplotlib Figure (cleared and redrawn).
        result (dict): runUROMT output.
        Eul (dict): runEULA output (needed for speed/rate/peclet/flux views).
        Lag (dict): runGLAD output (needed for the pathlines view).
        view (str): 'speed' | 'rate' | 'peclet' | 'velocity' | 'flux' | 'pathlines'.
        axis (int): ROI slice axis (0=row,1=col,2=slice); the other two are in-plane.
        sliceIdx (int): slice index along ``axis`` (default: middle).
        bg (np.ndarray): grayscale background on the ROI grid (e.g. mean concentration).
        cmap (str): colormap for the overlay.
        subsample (int): vector thinning (default scales with grid size).

    Returns:
        matplotlib.axes.Axes: the populated axes.
    """
    from matplotlib.collections import LineCollection      # lazy (matplotlib)

    n = [int(v) for v in result["n"]]
    if sliceIdx is None:
        sliceIdx = n[axis] // 2
    sliceIdx = int(np.clip(sliceIdx, 0, n[axis] - 1))
    ydim, xdim = [i for i in (0, 1, 2) if i != axis]

    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])

    def slc(arr3):
        return np.take(np.asarray(arr3), sliceIdx, axis=axis)

    if bg is not None:
        ax.imshow(slc(bg), cmap="gray", origin="upper", interpolation="nearest")

    title = "%s  (axis %d, slice %d/%d)" % (view, axis, sliceIdx, n[axis] - 1)

    if view in ("speed", "rate", "peclet"):
        key = {"speed": "speed3", "rate": "rate3", "peclet": "peclet3"}[view]
        if not Eul:
            raise ValueError("Eulerian results required for the '%s' view" % view)
        mp = slc(Eul[key]).astype(float)
        cmp_ = "bwr" if view == "rate" else cmap
        im = ax.imshow(np.ma.masked_equal(mp, 0.0), cmap=cmp_, alpha=0.8,
                       origin="upper", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=view)

    elif view in ("velocity", "flux"):
        if view == "velocity":
            field = result["u"][interval].mean(axis=2)         # (3,N)
        else:
            if not Eul:
                raise ValueError("Eulerian results required for the flux view")
            field = Eul["flux"]
        comp = [field[c].reshape(n, order="F") for c in range(3)]
        U = slc(comp[xdim]).astype(float)         # in-plane x-component
        V = slc(comp[ydim]).astype(float)         # in-plane y-component
        s = subsample or max(1, min(U.shape) // 20)
        yy, xx = np.mgrid[0:U.shape[0], 0:U.shape[1]]
        mag = np.sqrt(U ** 2 + V ** 2)
        q = ax.quiver(xx[::s, ::s], yy[::s, ::s], U[::s, ::s], V[::s, ::s],
                      mag[::s, ::s], cmap=cmap, angles="xy",
                      scale_units="xy", pivot="mid")
        fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label="%s mag" % view)

    elif view == "pathlines":
        if not Lag or not Lag.get("SL"):
            raise ValueError("Lagrangian results required for the pathlines view")
        segs, vals = [], []
        for pl, sp in zip(Lag["SL"], Lag["sstream"]):
            pl = np.asarray(pl)
            segs.append(np.column_stack([pl[:, xdim], pl[:, ydim]]))
            vals.append(float(np.mean(sp)) if len(sp) else 0.0)
        lc = LineCollection(segs, cmap=cmap, linewidths=0.7)
        lc.set_array(np.asarray(vals))
        ax.add_collection(lc)
        ax.set_xlim(0, n[xdim] - 1)
        ax.set_ylim(n[ydim] - 1, 0)                # match imshow orientation
        fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04, label="mean speed")
        title = "%s (%d, projected)" % (view, len(segs))
    else:
        raise ValueError("unknown urOMT view '%s'" % view)

    ax.set_title(title, color="#e8c542", fontsize=9)
    return ax


def drawUROMT3D(fig, result, Eul=None, Lag=None, view="pathlines",
                cmap="turbo", maxItems=2000, subsample=None, elev=20, azim=-60):
    """Draw a 3-D urOMT result view on a matplotlib Figure (Axes3D) for the
    embedded Qt viewer's 3-D mode. Pure matplotlib (headless-testable).

    Coordinates are ROI voxel indices mapped x=col, y=row, z=slice.

    Args:
        view (str): 'pathlines' | 'velocity' | 'flux' | 'speed' | 'rate' | 'peclet'.
        maxItems (int): cap on pathlines / scatter points / arrows.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection  # lazy
    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    _getcmap = (matplotlib.colormaps.__getitem__
                if hasattr(matplotlib, "colormaps") else cm.get_cmap)

    n = [int(v) for v in result["n"]]
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    ax.set_xlabel("col"), ax.set_ylabel("row"), ax.set_zlabel("slc")

    if view == "pathlines":
        if not Lag or not Lag.get("SL"):
            raise ValueError("Lagrangian results required for the pathlines view")
        SL, ss = Lag["SL"], Lag["sstream"]
        idx = range(len(SL))
        if len(SL) > maxItems:
            idx = np.linspace(0, len(SL) - 1, maxItems).astype(int)
        segs, vals = [], []
        for i in idx:
            pl = np.asarray(SL[i])
            segs.append(np.column_stack([pl[:, 1], pl[:, 0], pl[:, 2]]))
            vals.append(float(np.mean(ss[i])) if len(ss[i]) else 0.0)
        lc = Line3DCollection(segs, cmap=cmap, linewidths=0.7)
        lc.set_array(np.asarray(vals))
        ax.add_collection3d(lc)
        fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.1, label="mean speed")
        title = "pathlines 3D (%d)" % len(segs)

    elif view in ("velocity", "flux", "speed", "rate", "peclet"):
        mask3 = np.asarray(result["mask"]) > 0
        ri, ci, si = np.where(mask3)
        s = subsample or max(1, int((ri.size / maxItems) ** (1 / 1)) // 1 or 1)
        if ri.size > maxItems:
            sel = np.linspace(0, ri.size - 1, maxItems).astype(int)
            ri, ci, si = ri[sel], ci[sel], si[sel]
        if view in ("speed", "rate", "peclet"):
            if not Eul:
                raise ValueError("Eulerian results required for '%s'" % view)
            arr = Eul[{"speed": "speed3", "rate": "rate3",
                       "peclet": "peclet3"}[view]]
            vals = np.asarray(arr)[ri, ci, si]
            cmp_ = "bwr" if view == "rate" else cmap
            p = ax.scatter(ci, ri, si, c=vals, cmap=cmp_, s=6, depthshade=False)
            fig.colorbar(p, ax=ax, fraction=0.03, pad=0.1, label=view)
            title = "%s 3D" % view
        else:                                   # velocity / flux arrows
            if view == "flux":
                if not Eul:
                    raise ValueError("Eulerian results required for flux")
                field = Eul["flux"]
            else:
                field = result["u"][0].mean(axis=2)
            comp = [field[c].reshape(n, order="F") for c in range(3)]
            U, V, W = (comp[1][ri, ci, si], comp[0][ri, ci, si],
                       comp[2][ri, ci, si])           # x=col,y=row,z=slc
            mag = np.sqrt(U ** 2 + V ** 2 + W ** 2)
            mx = mag.max() if mag.size else 1.0
            colors = _getcmap(cmap)(mcolors.Normalize()(mag))
            ax.quiver(ci, ri, si, U, V, W, length=2.0 / (mx + 1e-8),
                      normalize=False, colors=colors, linewidth=0.6)
            sm = cm.ScalarMappable(cmap=cmap,
                                   norm=mcolors.Normalize(0, mx))
            sm.set_array(mag)
            fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.1,
                         label="%s mag" % view)
            title = "%s 3D" % view
    else:
        raise ValueError("unknown urOMT 3D view '%s'" % view)

    ax.set_xlim(0, n[1]); ax.set_ylim(0, n[0]); ax.set_zlim(0, n[2])
    try:
        ax.set_box_aspect((n[1], n[0], n[2]))
    except Exception:  # noqa: BLE001 (older matplotlib)
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, color="#e8c542", fontsize=9)
    return ax


def showEulerian(planC, Eul, field="speed", scanNum=None, structNum=None,
                 fluxVectors=True, subsample=2, magPctile=60.0,
                 lengthScale=1.0, displayMode="3d", colormap="turbo"):
    """Overlay an Eulerian map (speed/rate/peclet) and, optionally, the mean
    flux vectors on the scan/ROI in napari (Part 5). Returns the viewer."""
    from cerr.viewer.pycerr_napari import showNapari        # lazy (Qt/GL)
    sn = _scanNumOf(Eul) if scanNum is None else scanNum
    vd = eulerianFluxVectors(Eul, subsample=subsample, magPctile=magPctile,
                             lengthScale=lengthScale) if fluxVectors else {}
    structNums = [] if structNum is None else [structNum]
    out = showNapari(planC, scan_nums=sn, struct_nums=structNums,
                     dose_nums=[], vectors_dict=vd, displayMode=displayMode)
    viewer = out[0] if isinstance(out, tuple) else out
    fullMap = eulerianMapToScan(Eul, field=field, planC=planC, scanNum=sn)
    affine = _scanAffine(out)                          # align with the scan
    kw = dict(name="Eulerian %s" % field, colormap=colormap, opacity=0.6,
              blending="additive")
    if affine is not None:
        kw["affine"] = affine
    viewer.add_image(fullMap, **kw)
    return viewer


def showLagrangian(planC, Lag, colorBy="speed", scanNum=None, structNum=None,
                   maxTracks=2000, displayMode="3d", colormap="turbo"):
    """Display Lagrangian transport pathlines as a napari Tracks layer coloured
    by speed or Peclet (Part 5). Returns the viewer."""
    from cerr.viewer.pycerr_napari import showNapari        # lazy (Qt/GL)
    sn = _scanNumOf(Lag) if scanNum is None else scanNum
    structNums = [] if structNum is None else [structNum]
    out = showNapari(planC, scan_nums=sn, struct_nums=structNums,
                     dose_nums=[], vectors_dict={}, displayMode=displayMode)
    viewer = out[0] if isinstance(out, tuple) else out
    data, props = pathlineTracks(Lag, colorBy=colorBy, maxTracks=maxTracks)
    if data.shape[0]:
        affine = _scanAffine(out)                      # align with the scan
        kw = dict(properties=props, name="urOMT pathlines", color_by=colorBy,
                  colormap=colormap)
        if affine is not None:
            kw["affine"] = affine
        try:
            viewer.add_tracks(data, **kw)
        except TypeError:                              # older napari kwargs
            kw.pop("color_by", None)
            kw.pop("colormap", None)
            viewer.add_tracks(data, **kw)
    return viewer
