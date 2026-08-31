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

# Narrow arrowhead style shared by the velocity/flux quiver and the pathline
# end arrows, so direction reads the same way in both overlays. A slim shaft
# with a small head stays legible when hundreds are drawn over an image; the
# matplotlib defaults (headwidth 3, headlength 5, relative to `width`) give a
# blunt head that blots out the underlying anatomy at this density. The head
# multipliers have been raised twice from the original 4/6/5 (1.3x, then 1.3x
# again) - narrow is right, but the original values were far too small to pick
# out at typical zoom. They are multiples of the shaft `width`, so raising the
# "line w" control enlarges them further, proportionally.
_NARROW_ARROW = dict(width=0.0022, headwidth=9.0, headlength=13.0,
                     headaxislength=11.0,
                     # matplotlib scales the WHOLE arrow down once the shaft is
                     # shorter than `minshaft` head lengths - at the default 1
                     # a pathline tail (a fraction of a voxel) rendered at 12%
                     # of the head size, which is why the heads looked absent
                     # however large the head multipliers were set.
                     minshaft=0.35, minlength=0.05)

# Base pathline stroke width. Both this and the arrow shaft above are scaled by
# ``ov['lineWidth']`` so one control thickens vectors and pathlines together.
_PATH_LINE_WIDTH = 0.9


# How the arrow HEAD follows the length scale. Head sizes are multiples of the
# shaft width - i.e. fixed on screen - so without this a 10x "length x" drew
# 10x longer arrows with the same pinprick head, and a 0.2x one drew stubs that
# were nearly all head.
#
# Growth is LINEAR in the length scale - a 2x longer arrow gets a 2x head - so
# the glyph keeps its proportions where it matters. It was briefly sqrt-damped,
# which made x2 barely change the head at all. The cap only stops the extreme
# end of the range ("length x" reaches 100, where a proportional head would
# fill the view); it does not bind at ordinary settings.
_ARROW_HEAD_SCALE = (0.4, 3.0)

# Aspect of a pathline's arrowhead: full width as a fraction of its length,
# matching the narrow quiver head (_NARROW_ARROW headwidth/headlength).
_HEAD_ASPECT = 0.7


# Largest a head may be as a fraction of the arrow it terminates. Quiver head
# dimensions are multiples of the shaft width - an AXES fraction - so they know
# nothing about how long the arrows actually are: with "length x" raised, the
# heads grew into solid triangles that swamped the field. Tying the ceiling to
# the MEDIAN drawn arrow keeps the glyph in proportion at any length scale and
# any zoom. (Pathlines have the same rule per path: `_TAIL_MAX_FRAC`.)
_HEAD_MAX_ARROW_FRAC = 0.6

# HARD ceiling on an arrowhead, as a multiple of the scan's FINEST voxel. The
# caps above are relative (a fraction of the arrow or of the path), so on a
# long arrow or a big ROI they still allow a head of any absolute size; this
# bounds it against the resolution of the data being annotated, which is the
# scale a reader judges the picture at - a fixed millimetre number is either
# huge on a fine scan or invisible on a coarse one.
#
# The viewer passes the value on the overlay as ``headMaxData`` (in DATA units,
# i.e. cm - `scan.getScanSpacing()` is cm and the ruler reads cm). This
# fallback applies only to callers that draw without one.
_HEAD_MAX_VOXELS = 2.0
_HEAD_MAX_FALLBACK_CM = 0.2

# Legibility floor for a capped quiver head: shorter than this many shaft
# widths and the "head" is narrower than the shaft that carries it, i.e. the
# arrow reads as a plain line.
#
# Where the ceiling can only be met by a head this short, the ceiling is what
# yields: a ceiling honoured by erasing the arrowhead annotates nothing, and
# the alternative - thinning the shaft to buy head multiples - would break
# "line w" (the clamp is absolute, so every line width collapses onto the same
# drawn shaft). The floor is in shaft widths, so it stays small: at the default
# line width a floored head is ~1% of the view.
_HEAD_MIN_LENGTH_MULT = 5.0


def _headCeiling(ov=None):
    """Hard head-length ceiling in data units (cm) for this overlay."""
    v = (ov or {}).get("headMaxData")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _HEAD_MAX_FALLBACK_CM
    return v if v > 0 else _HEAD_MAX_FALLBACK_CM


def capTipFraction(frac, maxLen, ceiling=None):
    """Clamp a *proportional* arrowhead (a fraction of each arrow's length, as
    the 3-D glyph renderers use) so no head exceeds the ceiling.

    ``maxLen`` is the LONGEST arrow in the scene, in data units (cm) - using
    the longest rather than the median is what makes the ceiling hard: every
    head is then at or below it, not just the typical one. ``ceiling`` defaults
    to :func:`_headCeiling` with no overlay, i.e. the fallback.
    """
    frac = float(frac)
    maxLen = float(maxLen)
    ceil = _headCeiling() if ceiling is None else float(ceiling)
    if maxLen <= 0:
        return frac
    return max(min(frac, ceil / maxLen), 1e-3)


def _capHead(kw, maxHeadData, spanH):
    """Shrink a quiver glyph so its head is at most ``maxHeadData`` long in
    DATA units, without letting the head disappear.

    Head dims are multiples of the shaft ``width`` (an axes fraction), so
    capping the head alone eventually asks for a head only one or two shaft
    widths long - which draws as a bare line segment with no arrowhead at all
    (reported on a wide FOV with fine voxels: 2 voxels of ceiling over a 30 cm
    axis leaves headlength ~1.5x width). Once the cap would take the head under
    :data:`_HEAD_MIN_LENGTH_MULT` shaft widths, the head stops there and the
    ceiling yields - a head shorter than its own shaft is no arrowhead at all.
    The shaft width is untouched, so "line w" keeps its meaning.
    """
    span = float(max(spanH, 1e-12))
    width = float(kw["width"])
    if width <= 0 or maxHeadData <= 0:
        return kw
    maxMult = max(maxHeadData / (width * span), _HEAD_MIN_LENGTH_MULT)
    if float(kw["headlength"]) <= maxMult:
        return kw
    f = maxMult / float(kw["headlength"])
    out = dict(kw)
    for key in ("headwidth", "headlength", "headaxislength"):
        out[key] = kw[key] * f
    return out


def _pathHeadPolys(tips, dirs, sizes, aspect=_HEAD_ASPECT):
    """Triangles terminating each path: (M, 3, 2) vertices.

    Drawn as polygons rather than a matplotlib quiver head because quiver head
    dimensions are GLOBAL to the collection and fixed in axes units - on real
    urOMT data (paths ~1 voxel in a 60-voxel view, head ~2.6 voxels) one head
    covered its whole path, so the display showed arrowheads and no paths.
    Sizing each head off ITS path's length is the only way to keep both
    visible, and it needs one triangle per path.

    Args:
        tips (M,2): path end points. dirs (M,2): unit direction at the tip.
        sizes (M,): head LENGTH in data units, already limited per path.
    """
    tips = np.asarray(tips, dtype=float).reshape(-1, 2)
    dirs = np.asarray(dirs, dtype=float).reshape(-1, 2)
    sizes = np.asarray(sizes, dtype=float).reshape(-1, 1)
    nrm = np.hypot(dirs[:, 0], dirs[:, 1]).reshape(-1, 1)
    u = dirs / np.where(nrm > 1e-12, nrm, 1.0)
    perp = np.column_stack([-u[:, 1], u[:, 0]])
    back = tips - u * sizes
    half = perp * sizes * (0.5 * aspect)
    return np.stack([tips, back + half, back - half], axis=1)


# Largest a pathline's head may be, as a fraction of that path's visible
# EXTENT (bounding-box diagonal). Real urOMT paths span ~1 voxel while a fixed
# head is ~2.6 voxels at line width 2, so an unbounded head covered its whole
# path and the display showed arrowheads with no paths. Extent rather than arc
# length: a tight squiggle can travel ten voxels inside a two-voxel box.
_TAIL_MAX_FRAC = 0.5

#: Length of the longest velocity/flux arrow as a fraction of the field of
#: view, before the user's length scale. Shared by the 2-D quiver and the 3-D
#: glyphs so one "length x" means the same thing in both.
_VECTOR_FOV_FRAC = 0.08


def _headScale(lengthScale):
    """Head-size multiplier for a display length scale (see
    :data:`_ARROW_HEAD_SCALE`)."""
    return float(np.clip(max(float(lengthScale), 1e-9), *_ARROW_HEAD_SCALE))


def _tailVector(pts, minLen):
    """Direction vector along the END of a path, at least ``minLen`` long.

    Walks back over the drawn vertices until the chord from that vertex to the
    tip is long enough, so the base is a point ON the path and the arrow still
    terminates the trajectory instead of striking out on its own. The final
    SEGMENT alone is often a fraction of a voxel, and matplotlib shrinks an
    arrow whose shaft is shorter than its head - which is what made the heads
    vanish. Falls back to the whole path when even that is shorter.
    """
    pts = np.asarray(pts, dtype=float)
    tip = pts[-1]
    prev = pts[-2] if pts.shape[0] > 1 else pts[-1]
    dPrev = float(np.hypot(*(tip - prev)))
    for j in range(pts.shape[0] - 2, -1, -1):
        d = tip - pts[j]
        dj = float(np.hypot(d[0], d[1]))
        if dj >= minLen:
            # Interpolate WITHIN this segment instead of stopping at the
            # vertex: vertices are sparse after display decimation, so a whole
            # step back can overshoot and let the straight arrow cover more of
            # the path than intended.
            if dj > dPrev + 1e-12:
                t = (minLen - dPrev) / (dj - dPrev)
                base = prev + np.clip(t, 0.0, 1.0) * (pts[j] - prev)
                return base, tip - base
            return pts[j], d
        prev, dPrev = pts[j], dj
    return pts[0], tip - pts[0]


def _axisCoords(vox, gridIdx, gridVals):
    """Voxel indices -> physical coordinates along one display axis.

    ``np.interp`` CLAMPS outside the grid, which for a scaled pathline means
    every vertex past the field of view collapses onto the edge: the path piles
    up on the boundary and its end arrow degenerates to zero length and
    disappears. Points inside the grid still interpolate (slice spacing need
    not be uniform); points outside are extrapolated along the edge spacing, so
    a path that leaves the view is simply clipped by the axes.
    """
    vox = np.asarray(vox, dtype=float)
    out = np.interp(vox, gridIdx, gridVals)
    if gridVals.size < 2:
        return out
    lo = vox < gridIdx[0]
    if lo.any():
        step = float(gridVals[1] - gridVals[0])
        out[lo] = gridVals[0] + (vox[lo] - gridIdx[0]) * step
    hi = vox > gridIdx[-1]
    if hi.any():
        step = float(gridVals[-1] - gridVals[-2])
        out[hi] = gridVals[-1] + (vox[hi] - gridIdx[-1]) * step
    return out


def _arrowStyle(lw, lengthScale=1.0):
    """Narrow-arrow kwargs: shaft scaled by the line-width factor, head scaled
    by the display length scale so the glyph keeps its proportions.

    ``headwidth``/``headlength``/``headaxislength`` are multiples of ``width``,
    which is an axes-fraction - so they do NOT follow the arrow's data length.
    Scaling them by :func:`_headScale` keeps the head in proportion with the
    arrow without letting it grow without bound.
    """
    kw = dict(_NARROW_ARROW)
    kw["width"] = _NARROW_ARROW["width"] * max(float(lw), 1e-3)
    k = _headScale(lengthScale)
    if k != 1.0:
        for key in ("headwidth", "headlength", "headaxislength"):
            kw[key] = _NARROW_ARROW[key] * k
    return kw

# A pathline's direction arrow is the path's OWN final segment: the quiver
# vector is that segment, so the shaft lies exactly on the path and all the
# glyph adds is a head at the tip. The head keeps a fixed pixel size because
# `_NARROW_ARROW` sizes it in multiples of the shaft width, not of the vector.
#
# It used to be a separate stick of fixed length (1.2% of the field of view)
# pointing along the last segment's DIRECTION from a base behind the tip. On a
# curved or zigzag path - which is most of them - that stick cut straight
# across the windings it was supposed to terminate, reading as an unrelated
# arrow lying over the picture, and on a short (or mid-growth) path it was
# longer than the path itself.


def velocityVectors(result, interval=0, step=None, subsample=2,
                    speedPctile=60.0, lengthScale=1.0, maxVectors=20000):
    """Build a napari ``vectors_dict`` for one urOMT interval's velocity field.

    Args:
        result (dict): output of :func:`cerr.uromt.solver.solveUROMT`.
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
    # Any quantity is embeddable, not just the three original maps: the key is
    # "<field>3" unless the field has a legacy alias.
    key = {"speed": "speed3", "rate": "rate3", "peclet": "peclet3",
           "fluxmag": "fluxmag3"}.get(field, "%s3" % field)
    if key not in Eul:
        raise KeyError("no map '%s' in this Eulerian result" % key)
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


def pathlinesToScanVox(Lag, sizeFactor=1.0, doResize=0, perVertex=False):
    """Map ROI-voxel pathlines to full scan voxel coordinates (row,col,slice).

    Returns ``(segs, vals)`` where ``segs`` is a list of (steps,3) arrays and
    ``vals`` the per-pathline **mean** speed. With ``perVertex=True`` a third
    element is returned: a list of (steps,) arrays giving the speed sampled at
    every vertex, so a path can be coloured *along its length* rather than
    flat-shaded by its mean. ``runGLAD`` records one speed per integration
    sub-step and one extra position (the seed), so the per-vertex array is
    padded at the head to match the vertex count.
    """
    rs_, _, cs_, _, ss_, _ = Lag["bbox"]
    off = np.array([rs_, cs_, ss_], dtype=float)
    sf = float(sizeFactor) if doResize else 1.0
    SL, ss = Lag["SL"], Lag["sstream"]
    # Paths share a vertex count (runGLAD integrates every seed over the same
    # steps), so map the whole set in one shot rather than per path.
    if SL and all(np.asarray(p).shape == np.asarray(SL[0]).shape for p in SL):
        arr = np.asarray(SL, dtype=float) / sf + off      # (M, nVert, 3)
        segs = list(arr)
        sp = np.asarray(ss, dtype=float)
        vals = sp.mean(1) if sp.size else np.zeros(len(SL))
        if perVertex:
            nV = arr.shape[1]
            spds = list(sp if sp.shape[1] == nV
                        else np.asarray([_alignSpeedToVertices(s, nV)
                                         for s in sp]))
            return segs, vals, spds
        return segs, vals
    segs, vals, spds = [], [], []
    for pl, sp in zip(SL, ss):
        pl = np.asarray(pl, dtype=float) / sf + off
        segs.append(pl)
        sp = np.asarray(sp, dtype=float)
        vals.append(float(sp.mean()) if sp.size else 0.0)
        if perVertex:
            spds.append(_alignSpeedToVertices(sp, pl.shape[0]))
    if perVertex:
        return segs, np.asarray(vals), spds
    return segs, np.asarray(vals)


# Re-exported from analyze, where runGLAD computes these once per run.
from cerr.uromt.analyze import (pathSpeedStats,  # noqa: E402,F401
                                alignSpeedToVertices as
                                _alignSpeedToVertices)


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


def growPathline(pts, vals, frac):
    """Truncate one pathline to the leading ``frac`` of its vertices.

    Used by the "grow" control, which animates paths outward from their seed.
    ``frac >= 1`` returns the whole path; a fraction short enough to leave a
    single vertex still returns two points, so the path stays drawable.

    ``runGLAD`` integrates every seed over the same time steps, so all pathlines
    carry the same vertex count and a vertex fraction **is** a time fraction:
    the arc length drawn at a given ``frac`` is proportional to that path's
    speed, and fast paths visibly outrun slow ones. This is deliberate - it is
    the physically meaningful animation.
    """
    n = int(np.asarray(pts).shape[0])
    if n == 0:
        return pts, vals
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac >= 1.0:
        return pts, vals
    keep = max(2, int(round(n * frac))) if n >= 2 else n
    keep = min(keep, n)
    return pts[:keep], (None if vals is None else np.asarray(vals)[:keep])


def overlayTo3D(ov, xV, yV, zV, maxArrows=None, maxPaths=None,
                lengthScale=1.0):
    """Build 3-D urOMT overlay geometry from a cached overlay ``ov`` (the dict
    produced by ``PyCerrViewer.set_uromt_overlay``) and the scan's physical
    coordinate axes ``xV`` (col), ``yV`` (row), ``zV`` (slice).

    ``lengthScale`` multiplies both the arrow length and the pathline extent
    (paths are scaled about their seed), so long excursions can be reined in
    without changing the underlying result. ``ov['grow']`` truncates paths to a
    leading time fraction and ``ov['pathSpeeds']`` colours them along their
    length, matching the 2-D overlay.

    ``ov['subsample']`` keeps one arrow per Nth voxel **in all three
    directions**; the 2-D overlay applies the same N to the two in-plane axes
    of the displayed slice, so N=1 means every voxel of that slice there and
    every voxel of the volume here.

    ``maxArrows``/``maxPaths`` default to ``None`` = **no cap**: density is
    controlled solely by ``subsample``, so what is asked for is what is drawn.
    Pass an integer to thin uniformly on top. Note an uncapped N=1 on a large
    ROI is tens of thousands of glyphs and will be slow to render - raise
    ``subsample`` rather than reaching for a cap.

    Returns a dict (or ``None``) with optional keys:

    * ``vectors``: ``points`` (M,3), ``vec`` (M,3, scaled so the longest arrow
      spans ~5% of the field of view, keeping arrows inside the scan), ``mag``
      (M,) raw magnitudes, ``tip`` (M,3) arrow-head points.
    * ``paths``: list of (steps,3) pathline polylines in physical coords, with
      ``pathVals`` (one value per vertex: the reduced statistic repeated when
      ``ov['pathColorBy']`` is mean/median/max, the per-vertex samples for
      'along'), ``pathStart`` and ``pathEnd`` (M,3) seed and end points for the
      direction markers.
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
            if maxArrows and ri.size > maxArrows:       # optional cap
                sel = np.linspace(0, ri.size - 1, maxArrows).astype(int)
                ri, ci, si = ri[sel], ci[sel], si[sel]
            out["scalar"] = dict(
                points=np.column_stack([xV[ci], yV[ri], zV[si]]).astype(float),
                vals=m3[ri, ci, si].astype(float))
    if "comps" in ov:                                  # velocity / flux arrows
        cy, cx, cz = ov["comps"]                       # row(y), col(x), slice(z)
        mag = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
        ri, ci, si = np.where(mag > 0)
        # `subsample` ("vec every N") applies in ALL THREE directions here,
        # whereas the 2-D overlay strides only the two in-plane axes of the
        # displayed slice - one arrow per Nth voxel in the volume vs per Nth
        # voxel of the slice. Anchored at index 0 modulo N, matching the 2-D
        # `[::N]` stride, so the same voxels are picked in both views.
        sub = max(1, int(ov.get("subsample", 1) or 1))
        if ri.size and sub > 1:
            keep = (ri % sub == 0) & (ci % sub == 0) & (si % sub == 0)
            ri, ci, si = ri[keep], ci[keep], si[keep]
        if ri.size:
            if maxArrows and ri.size > maxArrows:       # optional cap
                sel = np.linspace(0, ri.size - 1, maxArrows).astype(int)
                ri, ci, si = ri[sel], ci[sel], si[sel]
            pts = np.column_stack([xV[ci], yV[ri], zV[si]]).astype(float)
            vec = np.column_stack([cx[ri, ci, si], cy[ri, ci, si],
                                   cz[ri, ci, si]]).astype(float)
            m = np.linalg.norm(vec, axis=1)
            mmax = float(m.max()) or 1.0
            # Longest arrow ~8% of the FOV, times the user's length scale - the
            # SAME fraction the 2-D quiver uses, so "length x" means the same
            # thing in both views. It used to be 5% here, which made every 3-D
            # arrow 40% shorter than its 2-D counterpart at the same setting.
            vecS = (vec * lengthScale if ov.get("trueScale")
                    else vec * (_VECTOR_FOV_FRAC * spanFOV * lengthScale / mmax))
            out["vectors"] = dict(points=pts, vec=vecS, mag=m, tip=pts + vecS)
            # Colour channel, when the overlay carries a scalar map to colour
            # the arrows by (any Eulerian quantity). Sampled at the SAME voxels
            # the arrows sit on, so 3-D and 2-D colour an arrow identically;
            # length still comes from the vector magnitude.
            cMap3 = ov.get("colorMap3")
            if cMap3 is not None:
                out["vectors"]["color"] = np.asarray(
                    cMap3)[ri, ci, si].astype(float)
    if "segs" in ov:                                    # pathlines
        segs, vals = ov["segs"][0], ov["segs"][1]
        spds = ov.get("pathVertVals", ov.get("pathSpeeds"))
        # Colour-by REDUCTION, the same control the 2-D overlay obeys: a
        # statistic gives the whole path one colour, 'along path' shades it per
        # vertex. 3-D used to always shade per vertex, so switching to
        # median/mean/max changed nothing here and the render read as noise -
        # every path carrying the full colour range along its length.
        colorBy = str(ov.get("pathColorBy", "median")).lower()
        if colorBy == "along path":
            colorBy = "along"
        pathStat = (None if colorBy == "along"
                    else (ov.get("pathStat") or {}).get(colorBy))
        grow = float(ov.get("grow", 1.0))
        # `subsample` thins the SEEDS in all three directions here, matching the
        # velocity arrows; the 2-D overlay instead thins in-plane among the
        # paths seeded on the displayed slice. Same N, applied per view.
        sub = max(1, int(ov.get("subsample", 1) or 1))
        order = range(len(segs))
        if sub > 1:
            order = [i for i in order
                     if not any(int(round(segs[i][0][a])) % sub
                                for a in (0, 1, 2))]
        if maxPaths and len(order) > maxPaths:          # optional cap
            order = np.asarray(order)[
                np.linspace(0, len(order) - 1, maxPaths).astype(int)]
        axR = np.arange(yV.size)
        axC = np.arange(xV.size)
        axS = np.arange(zV.size)
        lines, lineVals, starts, ends = [], [], [], []
        for i in order:
            pl = np.asarray(segs[i], dtype=float)
            sv = (np.asarray(spds[i], dtype=float)
                  if spds is not None and i < len(spds) else None)
            pl, sv = growPathline(pl, sv, grow)
            if pl.shape[0] < 2:
                continue
            x = _axisCoords(pl[:, 1], axC, xV)          # col -> x
            y = _axisCoords(pl[:, 0], axR, yV)          # row -> y
            z = _axisCoords(pl[:, 2], axS, zV)          # slice -> z
            pts3 = scalePathline(np.column_stack([x, y, z]), lengthScale)
            lines.append(pts3)
            # One colour per path for a statistic (constant along the path, so
            # the colour does not shift while the growth animation runs), else
            # the per-vertex values; fall back to the per-path mean when an
            # overlay carries neither.
            if pathStat is not None and i < len(pathStat):
                lineVals.append(np.full(pts3.shape[0], float(pathStat[i])))
            else:
                lineVals.append(sv if sv is not None
                                else np.full(pts3.shape[0], float(vals[i])))
            starts.append(pts3[0])
            ends.append(pts3[-1])
        if lines:
            out["paths"] = lines
            out["pathVals"] = lineVals
            out["pathStart"] = np.asarray(starts)
            out["pathEnd"] = np.asarray(ends)
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
                     cmap="turbo", colorbar=True):
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
    # A signed quantity (rate r, which is a source *or* a sink) needs a
    # diverging map wherever it is displayed - as the Eulerian colourwash, or
    # as the colour-by of vectors / pathlines.
    signed = (view == "rate" or bool(ov.get("diverging")))
    cmName = "bwr" if signed else cmap
    lo, hi = ov.get("vrange", (None, None))
    label = ov.get("label", view or "urOMT")
    sub = int(ov.get("subsample", subsample or 1) or 1)
    lineW = float(ov.get("lineWidth", 1.0))
    drewSomething = False
    if "map3" in ov:                                  # scalar colourwash
        m = np.ma.masked_equal(slicer(ov["map3"]), 0.0)
        vmin = lo if lo is not None else None
        vmax = hi if hi is not None else None
        ax.imshow(m, cmap=cmName, extent=extent, alpha=alpha, vmin=vmin,
                  vmax=vmax, interpolation="bilinear", aspect="equal", zorder=3)
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
        # `scale` is data units PER arrow unit, so a bigger length scale must
        # divide it (quiver draws shorter arrows for a larger `scale`).
        lenScale = max(float(ov.get("lengthScale", 1.0)), 1e-6)
        scale = gmax / (_VECTOR_FOV_FRAC * (spanH or 1.0) * lenScale)
        # `trueScale` draws the vectors 1:1 in data units instead of scaling the
        # longest one to a fixed fraction of the field of view. A urOMT velocity
        # has no meaningful length on a centimetre axis, but a DEFORMATION does -
        # an arrow that is 3 mm long because the voxel moved 3 mm is the whole
        # point of a DVF display, and it stays comparable between scans.
        if ov.get("trueScale"):
            scale = 1.0 / lenScale
        # Colour-by: an arrow is coloured by its own magnitude unless the
        # overlay carries a scalar map to colour it with (`colorMap3`, on the
        # full scan grid - speed, effSpeed, Peclet, rate, rho or |flux|, already
        # reduced over time by mean/median/max or taken at this interval). The
        # arrow LENGTH always stays the vector magnitude; only the colour
        # channel changes, so the same voxel reads the same value as the
        # pathline segment through it.
        cMap3 = ov.get("colorMap3")
        cLo, cHi = (0.0, gmax)
        if cMap3 is not None:
            Cs = slicer(cMap3)[::sub, ::sub]
            rng = ov.get("colorRange")
            if rng is not None and rng[1] > rng[0]:
                cLo, cHi = float(rng[0]), float(rng[1])
            elif Cs.size:
                cLo, cHi = 0.0, float(np.nanmax(Cs)) or 1.0
        else:
            Cs = Ms
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
            # Cap the head against the arrows actually on screen: `scale` is
            # data units per arrow unit, so median(mT)/scale is the median
            # drawn length.
            medLen = float(np.median(mT)) / max(scale, 1e-12)
            aKw = _capHead(_arrowStyle(lineW, lenScale),
                           min(_HEAD_MAX_ARROW_FRAC * medLen,
                               _headCeiling(ov)), spanH)
            ax.quiver(hT, vT, uT, vvT, Cs[drawn], cmap=cmName, alpha=alpha,
                      angles="xy", scale_units="xy", scale=scale, pivot="tail",
                      clim=(cLo, cHi), zorder=4, **aKw)
            # (no start/stop markers - the arrowhead shows direction/stop)
        lo, hi = cLo, cHi
        drewSomething = True
    elif "segs" in ov:                                # pathlines near the slice
        import matplotlib
        from matplotlib.collections import LineCollection
        # cmName, not cmap: colouring paths by a signed quantity (rate) needs
        # the diverging map, the same one the colourwash and the colorbar use.
        getc = (matplotlib.colormaps[cmName]
                if hasattr(matplotlib, "colormaps")
                else matplotlib.cm.get_cmap(cmName))
        segs, vals = ov["segs"][0], ov["segs"][1]
        # Per-vertex values of the SELECTED quantity (speed, effSpeed, Peclet,
        # rate, rho, |flux| - whatever `pathVertVals` was filled with) colour
        # each path ALONG its length; without them we fall back to flat-shading
        # the whole path by its mean. `pathSpeeds` is the pre-generalization
        # name, still honoured for overlays built by older code.
        spds = ov.get("pathVertVals", ov.get("pathSpeeds"))
        # Colour-by: 'along' shades each path ALONG its length, which needs one
        # matplotlib Path per SEGMENT and is ~2.3x the draw cost; 'mean',
        # 'median' and 'max' give the path one colour, so it is a single
        # polyline entry (10x fewer Path objects). Statistics are over the whole
        # path, so the colour does not shift while the grow animation runs.
        colorBy = str(ov.get("pathColorBy", "median")).lower()
        if colorBy == "along path":
            colorBy = "along"
        stat = None
        if colorBy != "along":
            stat = (ov.get("pathStat") or {}).get(colorBy)
            if stat is None and spds is not None:
                stat = pathSpeedStats(spds).get(colorBy)
            if stat is None:
                stat = np.asarray(vals)
        # Colour normalization spans the overlay's vrange, which for a signed
        # quantity (rate) is symmetric about zero - so a path cannot be
        # normalized by a max alone; `nrm` maps value -> 0..1 for the colormap.
        vmin = float(lo) if (signed and lo is not None) else 0.0
        vmax = hi if (hi is not None and hi > vmin) else (
            float(np.max(stat if stat is not None else vals))
            if len(vals) else 1.0)
        span = (vmax - vmin) or 1.0

        def cnorm(x):                  # value -> 0..1 for the colormap
            return (np.asarray(x, dtype=float) - vmin) / span
        grow = float(ov.get("grow", 1.0))
        lenScale = float(ov.get("lengthScale", 1.0))
        # Direction is shown by a narrow arrowhead at the END of each path, not
        # by start/end dot markers: with one marker per path they outnumbered
        # and obscured the paths themselves, and a coloured dot competes with
        # the colour channel that already carries speed.
        # Head length in DATA units - a CEILING: a head never exceeds this on
        # screen, and on a shorter path it shrinks to `_TAIL_MAX_FRAC` of that
        # path so the curve stays visible.
        aKw = _arrowStyle(lineW, ov.get("lengthScale", 1.0))
        spanH = abs(float(hV[-1] - hV[0])) if len(hV) > 1 else 1.0
        headLen = aKw["headlength"] * aKw["width"] * spanH
        headCeil = _headCeiling(ov)
        hx = np.arange(scanShape[hAxis])
        vx = np.arange(scanShape[vAxis])
        # A path is selected by WHERE IT STARTS, and then drawn in full. Showing
        # only the in-slice portion cut every trajectory at the slice boundary,
        # which hid where the transport actually went; selecting by seed gives
        # "the trajectories of the particles that started in this slice".
        # `subsample` thins those seeds IN-PLANE (the same N the vector quiver
        # applies to the slice), so N=1 is one path per ROI voxel of the slice.
        # Select vectorized. Looping over every path just to reject 98% of them
        # cost ~18% of the draw; `pathSeedVox` is built once per overlay.
        seedVox = ov.get("pathSeedVox")
        if seedVox is None:
            seedVox = np.rint(np.array([s[0] for s in segs])).astype(int)
        keep = seedVox[:, thruAxis] == int(k)
        if sub > 1:
            keep &= (seedVox[:, hAxis] % sub == 0)
            keep &= (seedVox[:, vAxis] % sub == 0)
        sel = np.where(keep)[0]
        # Pathline vertices are heavily oversampled for display - runGLAD
        # records one per integration sub-step (33 for a 13-interval run) for
        # curves spanning a couple of voxels. Every extra vertex is another
        # matplotlib Path, which dominates the draw, so thin them for drawing
        # only; the geometry is unchanged, just sampled more coarsely.
        decim = max(1, int(ov.get("pathDecim", 1)))
        # Per-segment speeds precomputed by Part 4 (Lag['segSpeed']); usable
        # only when the full vertex set is drawn.
        segSpeed = ov.get("pathSegVals", ov.get("pathSegSpeed"))
        nFull = (len(spds[0]) if spds is not None and len(spds) else 0)
        lc, lcC, startPts, endPts, endC, headLens = [], [], [], [], [], []
        for i in sel:
            seg, val = segs[i], vals[i]
            sv = spds[i] if spds is not None and i < len(spds) else None
            seg, sv = growPathline(seg, sv, grow)
            seg = scalePathline(seg, lenScale)
            if decim > 1 and seg.shape[0] > 2:
                # keep the last vertex so the end arrow still sits on the tip
                idx = np.r_[np.arange(0, seg.shape[0] - 1, decim),
                            seg.shape[0] - 1]
                seg = seg[idx]
                sv = None if sv is None else np.asarray(sv)[idx]
            if seg.shape[0] < 2:
                continue
            h = _axisCoords(seg[:, hAxis], hx, hV)
            v = _axisCoords(seg[:, vAxis], vx, vV)
            pts = np.column_stack([h, v])      # whole path, no slice clipping
            if stat is None:                   # 'along': one entry per SEGMENT
                lc.extend(np.stack([pts[:-1], pts[1:]], axis=1))
                # per-segment speeds come from Part 4 when the path is drawn
                # whole; grow/decimation change the vertex set, so recompute
                # only then.
                if (segSpeed is not None and sv is not None
                        and len(sv) == nFull and pts.shape[0] == nFull):
                    cvals = segSpeed[i]
                elif sv is not None:
                    cvals = 0.5 * (sv[:-1] + sv[1:])
                else:
                    cvals = np.full(pts.shape[0] - 1, val)
                cn = cnorm(cvals)
                lcC.extend(cn)
                lastC = float(cn[-1]) if len(cn) else 0.0
            else:                              # one polyline entry per PATH
                lc.append(pts)
                lastC = float(cnorm(stat[i]))
                lcC.append(lastC)
            # A head at the tip, pointing along the path's tail and sized off
            # THIS path, so it terminates the trajectory without covering it.
            # The measure is the path's visible EXTENT (bounding-box diagonal),
            # not its arc length: a tight squiggle can travel ten voxels inside
            # a two-voxel box, and a head sized off the arc would bury it.
            ext2 = float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
            hLen = min(headLen, _TAIL_MAX_FRAC * ext2, headCeil)
            _base, d = _tailVector(pts, hLen)
            if float(np.hypot(d[0], d[1])) > 1e-12 and hLen > 0:
                startPts.append(pts[-1])
                endPts.append(d)
                headLens.append(hLen)
                endC.append(lastC)
        if lc:
            # autolim=False: the axes limits are already fixed by the scan
            # image extent, and letting matplotlib recompute them walks every
            # segment (get_path_collection_extents was ~16% of the draw).
            # `lc` is a LIST so both forms work: (2,2) segments for 'along',
            # (nVert,2) polylines otherwise.
            ax.add_collection(LineCollection(
                lc, colors=getc(np.clip(lcC, 0, 1)),
                linewidths=_PATH_LINE_WIDTH * lineW, alpha=alpha, zorder=4),
                autolim=False)
        if startPts:
            from matplotlib.collections import PolyCollection
            tris = _pathHeadPolys(np.asarray(startPts), np.asarray(endPts),
                                  np.asarray(headLens))
            ax.add_collection(PolyCollection(
                tris, facecolors=getc(np.clip(endC, 0, 1)), edgecolors="none",
                alpha=alpha, zorder=5), autolim=False)
        lo, hi = vmin, vmax
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
        result (dict): solveUROMT output.
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
