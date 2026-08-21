"""Benchmark the urOMT overlay draw path (no PHI, synthetic data).

Answers "where does one animation frame go?" for the urOMT display:

* the overlay REBUILD (``PyCerrViewer.set_uromt_overlay``), which is what the
  animation calls on every tick, versus
* the overlay DRAW (``viz.drawUROMTOverlay``) into one slice view, versus
* recreating the matplotlib artists each frame versus updating them in place.

What it does NOT measure is the viewer's own slice redraw - the scan image,
structure contours and locators that ``refresh_views`` repaints around the
overlay. Measurements here pointed at the overlay, thinning it made the
overlay 3x cheaper, and the real GUI showed no difference: so the remaining
per-frame cost is in that redraw, and profiling it is the next step. Instrument
``PyCerrViewer.refresh_views(only=orient)`` in a live session rather than
extending this script.

Run::

    python -m cerr.scripts.bench_uromt_overlay
    python -m cerr.scripts.bench_uromt_overlay --paths 120000 --reps 5
"""
import argparse
import time
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure                       # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import LineCollection          # noqa: E402
from matplotlib.quiver import Quiver                       # noqa: E402


def timeIt(fn, reps=3):
    """Mean ms per call, one warm-up call excluded."""
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1000.0 * (time.perf_counter() - t) / reps


def buildRun(nVox, nt=2, nIvl=3, seed=0):
    """A synthetic solved run of roughly ``nVox`` ROI voxels."""
    from cerr.uromt.analyze import runGLAD
    side = int(round((nVox / 0.5) ** (1.0 / 3.0)))
    n = (side, side, max(4, side // 2))
    N = int(np.prod(n))
    rng = np.random.default_rng(seed)
    u = [rng.normal(0, 0.4, (3, N, nt)) for _ in range(nIvl)]
    res = dict(u=u, r=[np.zeros((N, nt))] * nIvl,
               rho=[np.ones((N, nt))] * nIvl, n=list(n),
               spacing=[1.0, 1.0, 1.4], mask=np.ones(n, np.uint8),
               bbox=(0, n[0], 0, n[1], 0, n[2]),
               frameScanNums=list(range(nIvl + 1)), doResize=0,
               sizeFactor=1.0, dt=0.3, nt=nt, sigma=2e-3)
    t0 = time.perf_counter()
    Lag = runGLAD(res, spfs=1, nEuler=2)
    print("run: %d paths x %d vertices (runGLAD %.1f s)"
          % (len(Lag["SL"]), Lag["nVert"], time.perf_counter() - t0))
    return res, Lag


def viewerStub(res, Lag):
    """The overlay builder bound onto a plain object (no Qt needed)."""
    from cerr.viewer.pycerr_gui.main_window import PyCerrViewer

    class _Stub:
        refresh_views = staticmethod(lambda **k: None)
        _refresh_uromt_views = staticmethod(lambda: None)
    for name in ("set_uromt_overlay", "_uromtVectorColor",
                 "_uromtEulIntervals", "_uromtSurfaceFlux",
                 "_uromtRoiMaskToScan", "_UROMT_LABELS",
                 "_UROMT_COLOR_LABELS"):
        setattr(_Stub, name, getattr(PyCerrViewer, name))
    v = _Stub()
    v.planC = SimpleNamespace(urOMT=[SimpleNamespace(
        UROMTResult=res, UROMTLagrangian=Lag)])
    v.scan3M = np.zeros([d + 4 for d in res["n"]])
    v.uromtOverlay = None
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", type=int, default=60000,
                    help="approximate ROI voxel count (= pathline count)")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv)

    from cerr.uromt import viz
    res, Lag = buildRun(args.paths)
    v = viewerStub(res, Lag)
    shape = v.scan3M.shape
    xV = np.arange(shape[1], dtype=float)
    yV = np.arange(shape[0], dtype=float)
    ext = [xV[0], xV[-1], yV[-1], yV[0]]
    K = shape[2] // 2

    print("\n-- overlay rebuild vs draw (one slice view) --")
    for reduce_ in ("median", "along"):
        build = timeIt(lambda: v.set_uromt_overlay(
            0, view="pathlines", grow=0.6, subsample=1,
            pathColorBy=reduce_), args.reps)
        ov = v.uromtOverlay

        def draw():
            fig = Figure()
            ax = fig.add_subplot(111)
            viz.drawUROMTOverlay(ax, ov, K, xV, yV, ext,
                                 lambda m: m[:, :, K], 1, 0, 2, shape)
            return ax
        d = timeIt(draw, args.reps)
        lc = [c for c in draw().collections
              if isinstance(c, LineCollection) and not isinstance(c, Quiver)]
        print("  reduce=%-7s %6d paths on the slice | rebuild %6.1f ms | "
              "draw %6.1f ms" % (reduce_, len(lc[0].get_segments()) if lc
                                 else 0, build, d))

    print("\n-- draw vs glyph stride ('every N') --")
    for sub in (1, 2, 3, 4):
        v.set_uromt_overlay(0, view="pathlines", grow=0.6, subsample=sub,
                            pathColorBy="median")
        ov = v.uromtOverlay

        def draw():
            fig = Figure()
            ax = fig.add_subplot(111)
            viz.drawUROMTOverlay(ax, ov, K, xV, yV, ext,
                                 lambda m: m[:, :, K], 1, 0, 2, shape)
            return ax
        ms = timeIt(draw, args.reps)
        lc = [c for c in draw().collections
              if isinstance(c, LineCollection) and not isinstance(c, Quiver)]
        print("  N=%d -> %6d drawn, %6.1f ms" % (
            sub, len(lc[0].get_segments()) if lc else 0, ms))

    print("\n-- recreate vs update in place (artists) --")
    rng = np.random.default_rng(0)
    nArrow = 20000
    X, Y = rng.random(nArrow) * 50, rng.random(nArrow) * 50
    U, V = rng.normal(size=nArrow), rng.normal(size=nArrow)
    C = rng.random(nArrow)
    kw = viz._arrowStyle(2.0)

    def freshQ():
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        q = ax.quiver(X, Y, U, V, C, angles="xy", scale_units="xy", scale=1.0,
                      **kw)
        fig.canvas.draw()
        return q
    q = freshQ()
    figQ = q.figure
    print("  quiver %d: recreate %7.1f ms | set_UVC + draw %7.1f ms"
          % (nArrow, timeIt(freshQ, args.reps),
             timeIt(lambda: (q.set_UVC(U, V, C), figQ.canvas.draw()),
                    args.reps)))

    nPath, nVert = 60000, 11
    polys = [np.column_stack([np.linspace(0, 5, nVert) + rng.random() * 40,
                              np.linspace(0, 3, nVert) + rng.random() * 40])
             for _ in range(nPath)]
    cols = rng.random((nPath, 4))

    def freshLC():
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        lc = LineCollection(polys, colors=cols, linewidths=1.8)
        ax.add_collection(lc, autolim=False)
        fig.canvas.draw()
        return lc
    lc = freshLC()
    figL = lc.figure
    print("  paths %d: recreate %7.1f ms | set_segments + draw %7.1f ms | "
          "render only %7.1f ms"
          % (nPath, timeIt(freshLC, 2),
             timeIt(lambda: (lc.set_segments(polys), lc.set_color(cols),
                             figL.canvas.draw()), 2),
             timeIt(lambda: figL.canvas.draw(), 2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
