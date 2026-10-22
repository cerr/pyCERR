"""Smoothing of 2-D contour polylines.

``taubin_smooth``, ``chaikin_smooth`` and ``catmull_rom_smooth`` are pure-NumPy
replacements for the functions of the same name in the ``shapelysmooth``
package (v0.2.1), ported from its C++ source. Results match shapelysmooth,
except that ``catmull_rom_smooth`` uses the standard reflected phantom points
at the ends of an open line, so its first and last segments differ slightly.

``bspline_smooth`` fits a closed smoothing B-spline through a contour. It is
the per-contour step of
:func:`cerr.dataclasses.structure.getBsplineSmoothing`.

Each function takes an ``(N, 2)`` array-like of ``(x, y)`` points (extra
columns such as z are ignored) and returns an ``(M, 2)`` NumPy array. For the
shapelysmooth ports, a line whose first and last points are equal is treated
as closed, as in shapelysmooth.
"""

import numpy as np
from scipy.interpolate import splprep, splev


def _as_points(geometry):
    pts = np.asarray(geometry, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("geometry must be a sequence of (x, y) points")
    return pts[:, :2].copy()


def _is_closed(pts):
    return len(pts) > 1 and np.array_equal(pts[0], pts[-1])


def taubin_smooth(geometry, factor=0.5, mu=-0.5, steps=5):
    """Taubin smoothing.

    Each step runs two Laplacian passes: one with weight ``factor`` and one
    with weight ``mu``. Every point moves toward the midpoint of its two
    neighbours by that weight. End points of an open line stay fixed.

    Args:
        geometry: ``(N, 2)`` array-like of points.
        factor (float): Weight of the first pass (lambda). Default 0.5.
        mu (float): Weight of the second pass. Default -0.5.
        steps (int): Number of steps. Default 5.

    Returns:
        np.ndarray: Smoothed ``(N, 2)`` points.
    """
    pts = _as_points(geometry)
    n = len(pts)
    if n < 3:
        return pts
    closed = _is_closed(pts)
    for _ in range(steps):
        for w in (factor, mu):
            new = pts.copy()
            new[1:-1] = (1 - w) * pts[1:-1] + w * 0.5 * (pts[2:] + pts[:-2])
            if closed:
                new[0] = (1 - w) * pts[0] + w * 0.5 * (pts[1] + pts[-2])
                new[-1] = new[0]
            pts = new
    return pts


def chaikin_smooth(geometry, iters=5, keep_ends=True):
    """Chaikin corner-cutting smoothing.

    Applies ``iters`` rounds of corner cutting in one closed-form pass.

    Args:
        geometry: ``(N, 2)`` array-like of points.
        iters (int): Number of corner-cutting rounds. Default 5.
        keep_ends (bool): Keep the end points of an open line. Default True.

    Returns:
        np.ndarray: Smoothed points.
    """
    pts = _as_points(geometry)
    if len(pts) < 2:
        return pts
    closed = _is_closed(pts)
    if closed:
        pts = np.vstack((pts, pts[1]))
    size = len(pts)
    e = 2 ** iters
    j = np.arange(e, dtype=float)
    F = 0.5 - 0.5 / e - j * (1.0 / e - (j + 1.0) * 0.5 / e / e)
    G = 0.5 + 0.5 / e + j * (1.0 / e - (j + 1.0) / e / e)
    H = j * (j + 1.0) * 0.5 / e / e

    # Shape (size-2, e, 2): e new points for every interior vertex.
    prev, cur, nxt = pts[:-2, None], pts[1:-1, None], pts[2:, None]
    body = (F[None, :, None] * prev + G[None, :, None] * cur
            + H[None, :, None] * nxt).reshape(-1, 2)
    out = np.vstack((pts[0], body))

    if keep_ends and not closed:
        return np.vstack((out, pts[-1]))
    out[0] = 0.5 * (1 + 1 / e) * pts[0] + 0.5 * (1 - 1 / e) * pts[1]
    if closed:
        return out
    last = 0.5 * (1 - 1 / e) * pts[-2] + 0.5 * (1 + 1 / e) * pts[-1]
    return np.vstack((out, last))


def catmull_rom_smooth(geometry, alpha=0.5, subdivs=10):
    """Catmull-Rom spline smoothing.

    Adds ``subdivs - 1`` interpolated points between each pair of points.

    Args:
        geometry: ``(N, 2)`` array-like of points.
        alpha (float): 0 = uniform, 0.5 = centripetal (default), 1 = chordal.
        subdivs (int): Subdivisions per segment. Default 10.

    Returns:
        np.ndarray: Smoothed points.
    """
    pts = _as_points(geometry)
    n = len(pts)
    if n < 2:
        return pts
    if _is_closed(pts):
        ext = np.vstack((pts[-2], pts, pts[1]))
    else:
        # Phantom end points reflect the neighbouring point. shapelysmooth
        # 0.2.1 used 2*p0 + p1 and 2*p_last instead, which made the end
        # segments depend on where the line sits in space.
        ext = np.vstack((2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]))

    out = [ext[1:2]]
    frac = np.arange(1, subdivs) / subdivs
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(len(ext) - 3):
            p = ext[k:k + 4]
            d = np.linalg.norm(np.diff(p, axis=0), axis=1) ** alpha
            t = np.concatenate(([0.0], np.cumsum(d)))
            tt = (t[1] + frac * (t[2] - t[1]))[:, None]
            # Barry-Goldman pyramid
            a = [((t[i + 1] - tt) * p[i] + (tt - t[i]) * p[i + 1])
                 / (t[i + 1] - t[i]) for i in range(3)]
            b = [((t[i + 2] - tt) * a[i] + (tt - t[i]) * a[i + 1])
                 / (t[i + 2] - t[i]) for i in range(2)]
            c = ((t[2] - tt) * b[0] + (tt - t[1]) * b[1]) / (t[2] - t[1])
            out.append(c)
            out.append(p[2:3])
    return np.vstack(out)


def bspline_smooth(geometry, resolution_factor=1.0, smoothing_factor=0.1):
    """Smooth a closed contour with a periodic smoothing B-spline.

    Fits ``scipy.interpolate.splprep`` with ``per=True`` and samples the
    spline at evenly spaced parameter values. Repeated consecutive points are
    dropped before fitting, since ``splprep`` rejects them. Contours with 3 or
    fewer distinct points are returned unchanged.

    Args:
        geometry: ``(N, 2)`` array-like of points (extra columns are ignored).
        resolution_factor (float): Number of output points relative to the
            input. For example, 1.2 gives 20% more points. Default 1.0.
        smoothing_factor (float): ``splprep`` smoothing condition ``s``. 0
            means no smoothing (the spline passes through every point);
            0.1-0.15 is reasonable for contours. Default 0.1.

    Returns:
        np.ndarray: Smoothed ``(M, 2)`` points.
    """
    pts = _as_points(geometry)
    num_pts = len(pts)
    fit_pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]
    if len(fit_pts) <= 3:
        return pts
    tck, _ = splprep([fit_pts[:, 0], fit_pts[:, 1]], s=smoothing_factor, per=True)
    x, y = splev(np.linspace(0, 1, round(num_pts * resolution_factor)), tck, der=0)
    return np.column_stack((x, y))
