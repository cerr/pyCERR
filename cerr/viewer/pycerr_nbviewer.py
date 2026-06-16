"""
pyCERR Notebook Viewer
======================
A notebook/Colab-friendly version of the pyCERR Viewer GUI (pycerr_gui.py).

The PyQt5 desktop GUI cannot run on a headless server (no display). This module
provides the same functionality with ipywidgets + matplotlib, which renders
natively inside Jupyter, JupyterLab, VS Code notebooks and Google Colab.

Features (matching the desktop viewer where it makes sense in a notebook):
  * Axial / sagittal / coronal views with slice sliders
  * CT window presets + manual center/width
  * Base-scan colormap + opacity
  * Structure contour overlays: per-structure toggles & DICOM colors, All/None,
    adjustable contour line width, vertex dots ("Alaly dots"), and a "Center"
    button to jump all views to a structure's center of mass
  * Patient-orientation labels (L/R/A/P/S/I)
  * Dose colorwash with alpha, all 21 CERR colormaps (cerr_colormaps.py)
  * Colorbar (colormap) range + dose display range controls with a live,
    standalone colorbar (range sliders replace the desktop's draggable markers)
  * Cumulative DVH tool (cerr.dvh) + CSV export
  * Programmatic control API (set_scan/set_dose/goto_structure/... ) mirroring
    the desktop pycerr_gui, plus save_screenshot() and export_dvh()
  * Live planC access: viewer.planC

Desktop-only features that don't map to a static notebook (3D VTK view,
interactive contouring, IMRTP beams, drag interactions, registration QA) are
not included here; use the Qt viewer (pycerr_gui.py) for those.

Usage in a notebook (e.g. Google Colab):

    !pip install pyCERR ipywidgets
    # upload pycerr_nbviewer.py and cerr_colormaps.py next to your notebook

    import cerr.plan_container as pc
    from pycerr_nbviewer import showNB

    planC = pc.loadDcmDir("/content/dicom_dir")
    viewer = showNB(planC)        # interactive viewer appears in the cell

    # later, in any cell -- planC reflects everything done in/through the viewer:
    planC = viewer.planC
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display
from scipy.interpolate import RegularGridInterpolator

import cerr.contour.rasterseg as rs
from cerr import dvh as cerrDvh

try:
    from cerr_colormaps import CERR_COLORMAP_NAMES, get_cmap as cerr_get_cmap
except ImportError:  # pragma: no cover
    CERR_COLORMAP_NAMES = ["jet"]

    def cerr_get_cmap(_name):
        return plt.get_cmap("jet")

CT_WINDOW_PRESETS = {
    "--- Manual ---": None,
    "Abd/Med": (-10, 330),
    "Head": (45, 125),
    "Liver": (80, 305),
    "Lung": (-500, 1500),
    "Spine": (30, 300),
    "Vrt/Bone": (400, 1500),
    "Soft Tissue": (40, 400),
    "Brain": (40, 80),
    "PET SUV": (2.5, 5),
}

# base-scan colormaps (matches the desktop viewer's SCAN_CMAPS)
SCAN_CMAPS = ["gray", "bone", "hot", "jet", "viridis", "magma", "plasma",
              "copper", "cool", "Greens", "Reds", "Blues"]

# patient-orientation labels along pyCERR virtual axes (+x=L, +y=A, +z=I)
ORIENT_POS = {"x": "L", "y": "A", "z": "I"}
ORIENT_NEG = {"x": "R", "y": "P", "z": "S"}
# (horizontal, vertical) physical axis per view
VIEW_HV = {"ax": ("x", "y"), "sag": ("y", "z"), "cor": ("x", "z")}


def _ascending(coords, arr, axis):
    coords = np.asarray(coords, dtype=float)
    if coords.size > 1 and coords[0] > coords[-1]:
        return coords[::-1], np.flip(arr, axis=axis)
    return coords, arr


class NbViewer:
    """CERR-style slice viewer for Jupyter/Colab. Access .planC at any time."""

    def __init__(self, planC, scanNum=0, figWidth=13.0, autoDisplay=True):
        if planC is None or not planC.scan:
            raise ValueError("planC must contain at least one scan.")
        self.planC = planC
        self.scanNum = int(scanNum)
        self.maskCache = {}
        self.doseInterp = None
        self.doseMax = 0.0
        self.figWidth = figWidth

        self._build_widgets()
        self._on_scan_change(initial=True)
        if autoDisplay:
            display(self.ui)
        self._redraw()

    # ------------------------------------------------------------ widgets ---
    def _build_widgets(self):
        planC = self.planC
        lay = W.Layout(width="240px")

        self.wScan = W.Dropdown(
            options=[(f"{i}: {getattr(s.scanInfo[0], 'imageType', 'scan')}", i)
                     for i, s in enumerate(planC.scan)],
            value=self.scanNum, description="Scan:", layout=lay)
        self.wPreset = W.Dropdown(options=list(CT_WINDOW_PRESETS),
                                  value="Soft Tissue", description="Window:",
                                  layout=lay)
        self.wCenter = W.FloatText(value=40.0, description="C:", layout=lay)
        self.wWidth = W.FloatText(value=400.0, description="W:", layout=lay)
        self.wScanCmap = W.Dropdown(options=SCAN_CMAPS, value="gray",
                                    description="Scan cmap:", layout=lay)
        self.wScanAlpha = W.FloatSlider(value=1.0, min=0, max=1, step=0.05,
                                        description="Scan op:",
                                        continuous_update=False, layout=lay)

        self.wAx = W.IntSlider(description="Axial", continuous_update=False)
        self.wSag = W.IntSlider(description="Sagittal", continuous_update=False)
        self.wCor = W.IntSlider(description="Coronal", continuous_update=False)

        self.structChecks = []
        for i, st in enumerate(planC.structure):
            cb = W.Checkbox(value=True, indent=False,
                            description=f"{i}: {st.structureName}",
                            layout=W.Layout(width="220px"))
            cb.observe(self._on_change, names="value")
            self.structChecks.append(cb)
        structList = W.VBox(self.structChecks,
                            layout=W.Layout(max_height="150px",
                                            overflow_y="auto"))
        self.wAllBtn = W.Button(description="All",
                                layout=W.Layout(width="55px"))
        self.wNoneBtn = W.Button(description="None",
                                 layout=W.Layout(width="55px"))
        self.wAllBtn.on_click(lambda _b: self.set_structures_visible("all"))
        self.wNoneBtn.on_click(lambda _b: self.set_structures_visible("none"))
        self.wDots = W.Checkbox(value=False, indent=False, description="Dots",
                                layout=W.Layout(width="80px"))
        self.wLineW = W.FloatSlider(value=1.4, min=0.2, max=6.0, step=0.2,
                                    description="Line:", readout_format=".1f",
                                    continuous_update=False,
                                    layout=W.Layout(width="220px"))
        self.wGoStruct = W.Dropdown(
            options=[(f"{i}: {st.structureName}", i)
                     for i, st in enumerate(planC.structure)] or [("-", -1)],
            description="Go to:", layout=W.Layout(width="220px"))
        self.wGoBtn = W.Button(description="Center",
                               layout=W.Layout(width="70px"))
        self.wGoBtn.on_click(
            lambda _b: self.goto_structure(self.wGoStruct.value))
        structBox = W.VBox([
            W.HBox([self.wAllBtn, self.wNoneBtn, self.wDots]),
            structList, self.wLineW,
            W.HBox([self.wGoStruct, self.wGoBtn])])

        doseOpts = [("None", -1)] + [
            (f"{i}: {getattr(d, 'fractionGroupID', 'dose')}", i)
            for i, d in enumerate(planC.dose)]
        self.wDose = W.Dropdown(options=doseOpts,
                                value=(0 if planC.dose else -1),
                                description="Dose:", layout=lay)
        self.wAlpha = W.FloatSlider(value=0.45, min=0, max=1, step=0.05,
                                    description="Alpha:",
                                    continuous_update=False, layout=lay)
        default_cmap = "starinterp" if "starinterp" in CERR_COLORMAP_NAMES \
            else CERR_COLORMAP_NAMES[0]
        self.wCmap = W.Dropdown(options=CERR_COLORMAP_NAMES,
                                value=default_cmap, description="Colormap:",
                                layout=lay)
        rlay = W.Layout(width="320px")
        self.wCbarRange = W.FloatRangeSlider(
            value=[0, 1], min=0, max=1, step=0.01, readout_format=".3g",
            description="Colorbar:", continuous_update=False, layout=rlay)
        self.wDispRange = W.FloatRangeSlider(
            value=[0, 1], min=0, max=1, step=0.01, readout_format=".3g",
            description="Display:", continuous_update=False, layout=rlay)

        self.wDvhBtn = W.Button(description="Plot DVH",
                                button_style="primary")
        self.wDvhBtn.on_click(self._on_dvh)
        self.wDvhExport = W.Button(description="Export CSV")
        self.wDvhExport.on_click(self._on_dvh_export)
        self.wDvhPath = W.Text(value="dvh.csv", layout=W.Layout(width="150px"))
        self.wXhair = W.Checkbox(value=True, indent=False,
                                 description="Crosshairs",
                                 layout=W.Layout(width="110px"))
        self.wOrient = W.Checkbox(value=True, indent=False,
                                  description="Orient labels",
                                  layout=W.Layout(width="130px"))

        self.out = W.Output()       # slice views + colorbar
        self.dvhOut = W.Output()    # DVH figure

        for w in (self.wPreset, self.wCenter, self.wWidth, self.wScanCmap,
                  self.wScanAlpha, self.wAx, self.wSag, self.wCor, self.wDose,
                  self.wAlpha, self.wCmap, self.wCbarRange, self.wDispRange,
                  self.wXhair, self.wOrient, self.wDots, self.wLineW):
            w.observe(self._on_change, names="value")
        self.wScan.observe(lambda _ch: self._on_scan_change(), names="value")

        controls = W.HBox([
            W.VBox([self.wScan, self.wPreset, self.wCenter, self.wWidth,
                    self.wScanCmap, self.wScanAlpha]),
            W.VBox([W.HTML("<b>Structures</b>"), structBox]),
            W.VBox([self.wDose, self.wAlpha, self.wCmap,
                    self.wCbarRange, self.wDispRange,
                    W.HBox([self.wXhair, self.wOrient]),
                    W.HBox([self.wDvhBtn, self.wDvhExport, self.wDvhPath])]),
        ])
        sliders = W.HBox([self.wAx, self.wSag, self.wCor])
        self.ui = W.VBox([controls, sliders, self.out, self.dvhOut])

    # ------------------------------------------------------------- state ----
    def _on_scan_change(self, initial=False):
        self.scanNum = self.wScan.value
        self.maskCache.clear()
        scanObj = self.planC.scan[self.scanNum]
        self.scan3M = scanObj.getScanArray().astype(np.float32)
        self.xV, self.yV, self.zV = scanObj.getScanXYZVals()
        nR, nC, nS = self.scan3M.shape
        for w, n in ((self.wAx, nS), (self.wSag, nC), (self.wCor, nR)):
            w.unobserve(self._on_change, names="value")
            w.max = n - 1
            w.value = n // 2
            w.observe(self._on_change, names="value")
        mod = str(getattr(scanObj.scanInfo[0], "imageType", "")).upper()
        if "CT" not in mod:
            lo, hi = np.percentile(self.scan3M, [2, 98])
            self.wCenter.value, self.wWidth.value = (lo + hi) / 2, max(hi - lo, 1)
        self._build_dose_interp()
        if not initial:
            self._redraw()

    def _build_dose_interp(self):
        self.doseInterp, self.doseMax = None, 0.0
        dNum = self.wDose.value
        if dNum < 0 or dNum >= len(self.planC.dose):
            return
        d = self.planC.dose[dNum]
        dose3M = np.asarray(d.doseArray, dtype=np.float32)
        xD, yD, zD = d.getDoseXYZVals()
        yD, dose3M = _ascending(yD, dose3M, 0)
        xD, dose3M = _ascending(xD, dose3M, 1)
        zD, dose3M = _ascending(zD, dose3M, 2)
        self.doseInterp = RegularGridInterpolator(
            (yD, xD, zD), dose3M, bounds_error=False, fill_value=0.0)
        self.doseMax = float(dose3M.max())
        for w in (self.wCbarRange, self.wDispRange):
            w.unobserve(self._on_change, names="value")
            w.max = self.doseMax
            w.step = self.doseMax / 200.0
            w.value = (0.0, self.doseMax)
            w.observe(self._on_change, names="value")

    def _on_change(self, change):
        owner = change.get("owner")
        if owner is self.wPreset:
            preset = CT_WINDOW_PRESETS.get(self.wPreset.value)
            if preset:
                self.wCenter.unobserve(self._on_change, names="value")
                self.wWidth.unobserve(self._on_change, names="value")
                self.wCenter.value, self.wWidth.value = preset
                self.wCenter.observe(self._on_change, names="value")
                self.wWidth.observe(self._on_change, names="value")
        elif owner is self.wDose:
            self._build_dose_interp()
        self._redraw()

    # ----------------------------------------------------------- helpers ----
    def _struct_mask(self, n):
        if n not in self.maskCache:
            try:
                self.maskCache[n] = rs.getStrMask(n, self.planC)
            except Exception:  # noqa: BLE001
                self.maskCache[n] = None
        return self.maskCache[n]

    def _struct_color(self, n):
        col = np.asarray(self.planC.structure[n].structureColor,
                         dtype=float).ravel()
        if col.size != 3:
            return (1.0, 0.0, 0.0)
        if col.max() > 1:
            col = col / 255.0
        return tuple(np.clip(col, 0, 1))

    def _slice(self, orient):
        if orient == "ax":
            k = self.wAx.value
            return (self.scan3M[:, :, k], self.xV, self.yV,
                    lambda m: m[:, :, k], k)
        if orient == "sag":
            k = self.wSag.value
            return (self.scan3M[:, k, :].T, self.yV, self.zV,
                    lambda m: m[:, k, :].T, k)
        k = self.wCor.value
        return (self.scan3M[k, :, :].T, self.xV, self.zV,
                lambda m: m[k, :, :].T, k)

    @staticmethod
    def _draw_dots(ax, contourSet, color):
        """Alaly-style contour vertex dots; white on dark colors, black on
        light (sum of RGB < 1.5 -> white) so they stay visible."""
        dotColor = "white" if sum(color[:3]) < 1.5 else "black"
        for seg in contourSet.allsegs[0]:
            if len(seg) == 0:
                continue
            step = max(len(seg) // 150, 3)
            pts = seg[::step]
            ax.plot(pts[:, 0], pts[:, 1], linestyle="none", marker="o",
                    markersize=1.0, markerfacecolor=dotColor,
                    markeredgecolor=dotColor, markeredgewidth=0.0, zorder=12)

    @staticmethod
    def _draw_orient_labels(ax, hAxis, vAxis):
        """L/R/A/P/S/I markers at the edges, from the displayed directions."""
        x0, x1 = ax.get_xlim()
        left, right = ((ORIENT_NEG[hAxis], ORIENT_POS[hAxis]) if x1 >= x0
                       else (ORIENT_POS[hAxis], ORIENT_NEG[hAxis]))
        y0, y1 = ax.get_ylim()
        bottom, top = ((ORIENT_NEG[vAxis], ORIENT_POS[vAxis]) if y1 >= y0
                       else (ORIENT_POS[vAxis], ORIENT_NEG[vAxis]))
        kw = dict(transform=ax.transAxes, color="#e8e8e8", fontsize=9,
                  fontweight="bold", ha="center", va="center", zorder=15,
                  bbox=dict(facecolor="black", alpha=0.45, edgecolor="none",
                            pad=1.5))
        ax.text(0.04, 0.5, left, **kw)
        ax.text(0.96, 0.5, right, **kw)
        ax.text(0.5, 0.95, top, **kw)
        ax.text(0.5, 0.05, bottom, **kw)

    # ----------------------------------------------------------- drawing ----
    def render_figure(self):
        """Build and return the matplotlib figure of the three views."""
        vmin = self.wCenter.value - self.wWidth.value / 2.0
        vmax = self.wCenter.value + self.wWidth.value / 2.0
        cmap = cerr_get_cmap(self.wCmap.value)
        showDose = self.doseInterp is not None and self.wAlpha.value > 0
        cbLo, cbHi = self.wCbarRange.value
        dLo, dHi = self.wDispRange.value

        ncols = 4 if showDose else 3
        widths = [1, 1, 1] + ([0.06] if showDose else [])
        fig, axes = plt.subplots(
            1, ncols, figsize=(self.figWidth, self.figWidth / 3.1),
            facecolor="black", gridspec_kw={"width_ratios": widths})
        axes = np.atleast_1d(axes)
        checked = [i for i, cb in enumerate(self.structChecks) if cb.value]

        scanCmap = self.wScanCmap.value
        scanAlpha = self.wScanAlpha.value
        lineW = self.wLineW.value
        showDots = self.wDots.value
        titles = {"ax": "Axial", "sag": "Sagittal", "cor": "Coronal"}
        for ax, orient in zip(axes[:3], ("ax", "sag", "cor")):
            img, hV, vV, slicer, k = self._slice(orient)
            extent = [hV[0], hV[-1], vV[-1], vV[0]]
            ax.imshow(img, cmap=scanCmap, vmin=vmin, vmax=vmax, extent=extent,
                      interpolation="nearest", aspect="equal", alpha=scanAlpha)
            if showDose:
                H, V = np.meshgrid(hV, vV)
                if orient == "ax":
                    pts = (V, H, np.full_like(H, self.zV[k]))
                elif orient == "sag":
                    pts = (H, np.full_like(H, self.xV[k]), V)
                else:
                    pts = (np.full_like(H, self.yV[k]), H, V)
                doseSlc = self.doseInterp(np.stack(
                    [p.ravel() for p in pts], -1)).reshape(H.shape)
                dm = np.ma.masked_where(
                    (doseSlc < max(dLo, 1e-3)) | (doseSlc > dHi), doseSlc)
                ax.imshow(dm, cmap=cmap, extent=extent, vmin=cbLo,
                          vmax=max(cbHi, cbLo + 1e-6),
                          alpha=self.wAlpha.value,
                          interpolation="bilinear", aspect="equal")
            for n in checked:
                mask = self._struct_mask(n)
                if mask is None or mask.shape != self.scan3M.shape:
                    continue
                mslc = slicer(mask)
                if np.any(mslc):
                    color = self._struct_color(n)
                    cs = ax.contour(hV, vV, mslc.astype(float), levels=[0.5],
                                    colors=[color], linewidths=lineW)
                    if showDots:
                        self._draw_dots(ax, cs, color)
            if self.wOrient.value:
                self._draw_orient_labels(ax, *VIEW_HV[orient])
            if self.wXhair.value:
                kw = dict(color="#e8c542", lw=1.1, ls="--", alpha=0.9)
                if orient == "ax":      # sagittal x, coronal y
                    ax.axvline(self.xV[self.wSag.value], **kw)
                    ax.axhline(self.yV[self.wCor.value], **kw)
                elif orient == "sag":   # coronal y, axial z
                    ax.axvline(self.yV[self.wCor.value], **kw)
                    ax.axhline(self.zV[self.wAx.value], **kw)
                else:                   # sagittal x, axial z
                    ax.axvline(self.xV[self.wSag.value], **kw)
                    ax.axhline(self.zV[self.wAx.value], **kw)
            ax.set_title(f"{titles[orient]} - slice {k + 1}",
                         color="#e8c542", fontsize=10)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_facecolor("black")

        # crosshair intersection readout (position + scan value + dose)
        if self.wXhair.value:
            r, c, s = self.wCor.value, self.wSag.value, self.wAx.value
            x, y, z = self.xV[c], self.yV[r], self.zV[s]
            txt = (f"crosshair:  x={x:.2f}  y={y:.2f}  z={z:.2f} cm   "
                   f"scan={self.scan3M[r, c, s]:.1f}")
            if self.doseInterp is not None:
                txt += f"   dose={float(self.doseInterp((y, x, z))):.2f}"
            fig.text(0.01, 0.005, txt, color="#e8c542", fontsize=9)

        if showDose:   # standalone colorbar with display-range shading
            cax = axes[3]
            grad = np.linspace(cbHi, cbLo, 256)[:, None]
            cax.imshow(grad, cmap=cmap, aspect="auto",
                       extent=[0, 1, cbLo, cbHi], vmin=cbLo,
                       vmax=max(cbHi, cbLo + 1e-6))
            for lo, hi in ((cbLo, dLo), (dHi, cbHi)):   # dim outside display
                if hi > lo:
                    cax.axhspan(lo, hi, color="black", alpha=0.78)
            for v, c in ((dLo, "#3ad6e0"), (dHi, "#3ad6e0")):
                cax.axhline(v, color=c, lw=2)
            cax.set_xticks([])
            cax.yaxis.tick_right()
            cax.tick_params(colors="white", labelsize=8)
            cax.set_ylim(cbLo, cbHi)
            cax.set_title("Dose", color="white", fontsize=9)
        fig.tight_layout()
        return fig

    def _redraw(self):
        with self.out:
            self.out.clear_output(wait=True)
            fig = self.render_figure()
            display(fig)
            plt.close(fig)

    # --------------------------------------------------------------- DVH ----
    def _on_dvh(self, _btn=None):
        with self.dvhOut:
            self.dvhOut.clear_output(wait=True)
            doseNum = self.wDose.value
            if doseNum < 0:
                print("Select a dose first.")
                return
            fig, ax = plt.subplots(figsize=(7, 4.2))
            plotted = False
            for i, cb in enumerate(self.structChecks):
                if not cb.value:
                    continue
                try:
                    dosesV, volsV, isErr = cerrDvh.getDVH(i, doseNum, self.planC)
                    if isErr or dosesV is None or len(dosesV) == 0:
                        continue
                    bw = max(float(np.max(dosesV)) / 400.0, 1e-3)
                    bins, hist = cerrDvh.doseHist(dosesV, volsV, bw)
                    cum = np.flip(np.cumsum(np.flip(hist)))
                    ax.plot(bins, 100.0 * cum / cum[0],
                            label=self.planC.structure[i].structureName,
                            color=self._struct_color(i), lw=1.8)
                    plotted = True
                except Exception as e:  # noqa: BLE001
                    print(f"DVH failed for structure {i}: {e}")
            if plotted:
                ax.set_xlabel("Dose (Gy)"), ax.set_ylabel("Volume (%)")
                ax.set_ylim(0, 105), ax.grid(alpha=0.3), ax.legend(fontsize=8)
                display(fig)
            else:
                print("No DVH could be computed (check structure selection).")
            plt.close(fig)

    def _on_dvh_export(self, _btn=None):
        with self.dvhOut:
            self.dvhOut.clear_output(wait=True)
            try:
                path = self.export_dvh(self.wDvhPath.value)
                print(f"DVH written to {path}")
            except Exception as e:  # noqa: BLE001
                print(f"DVH export failed: {e}")

    # ------------------------------------------------------------- API ------
    # Programmatic control (mirrors the desktop pycerr_gui scripting API).
    def set_scan(self, scanNum):
        self.wScan.value = int(scanNum)

    def set_window_level(self, center, width):
        self.wPreset.value = "--- Manual ---"
        self.wCenter.value = float(center)
        self.wWidth.value = float(width)

    def set_window_preset(self, name):
        self.wPreset.value = name

    def set_scan_colormap(self, name):
        self.wScanCmap.value = name

    def set_scan_opacity(self, alpha):
        self.wScanAlpha.value = float(alpha)

    def set_dose(self, doseNum):
        self.wDose.value = -1 if doseNum is None or doseNum < 0 else int(doseNum)

    def set_dose_alpha(self, alpha):
        self.wAlpha.value = float(alpha)

    def set_dose_colormap(self, name):
        self.wCmap.value = name

    def set_structures_visible(self, which):
        if which == "all":
            want = set(range(len(self.structChecks)))
        elif which == "none":
            want = set()
        else:
            want = set(int(i) for i in which)
        for i, cb in enumerate(self.structChecks):
            cb.unobserve(self._on_change, names="value")
            cb.value = i in want
            cb.observe(self._on_change, names="value")
        self._redraw()

    def set_structure_dots(self, on):
        self.wDots.value = bool(on)

    def set_contour_linewidth(self, width):
        self.wLineW.value = float(width)

    def set_crosshairs(self, on):
        self.wXhair.value = bool(on)

    def set_orientation_labels(self, on):
        self.wOrient.value = bool(on)

    def set_slice(self, orientation, k):
        w = {"ax": self.wAx, "axial": self.wAx, "sag": self.wSag,
             "sagittal": self.wSag, "cor": self.wCor,
             "coronal": self.wCor}[str(orientation).lower()]
        w.value = int(np.clip(k, 0, w.max))

    def goto_structure(self, strNum):
        """Center the three views on a structure's center of mass."""
        if strNum is None or strNum < 0:
            return
        mask = self._struct_mask(int(strNum))
        if mask is None or mask.shape != self.scan3M.shape or not mask.any():
            return
        rows, cols, slcs = np.where(mask)            # (y, x, z)
        for w, val in ((self.wAx, slcs.mean()), (self.wSag, cols.mean()),
                       (self.wCor, rows.mean())):
            w.value = int(np.clip(round(val), 0, w.max))

    def compute_dvh(self, doseNum=None, structNums=None, num_bins=400):
        """Cumulative DVHs -> (doseAxis_Gy, {name: volPct}), each interpolated
        onto a shared 0..max dose axis."""
        if doseNum is None:
            doseNum = self.wDose.value
        if doseNum is None or doseNum < 0:
            raise ValueError("Select/pass a dose to compute DVHs.")
        if structNums is None:
            structNums = list(range(len(self.planC.structure)))
        elif isinstance(structNums, (int, np.integer)):
            structNums = [int(structNums)]
        raw, gmax = {}, 0.0
        for n in structNums:
            try:
                dosesV, volsV, isErr = cerrDvh.getDVH(n, doseNum, self.planC)
            except Exception:  # noqa: BLE001
                continue
            if isErr or dosesV is None or len(dosesV) == 0:
                continue
            bw = max(float(np.max(dosesV)) / 400.0, 1e-3)
            bins, hist = cerrDvh.doseHist(dosesV, volsV, bw)
            cum = np.flip(np.cumsum(np.flip(hist)))
            if cum[0] <= 0:
                continue
            raw[n] = (np.asarray(bins, float), 100.0 * cum / cum[0])
            gmax = max(gmax, float(bins[-1]))
        if not raw:
            raise ValueError("No DVH could be computed.")
        axis = np.linspace(0.0, gmax, int(num_bins))
        table = {self.planC.structure[n].structureName:
                 np.interp(axis, db, cp, left=cp[0], right=0.0)
                 for n, (db, cp) in raw.items()}
        return axis, table

    def export_dvh(self, path, doseNum=None, structNums=None, num_bins=400):
        """Compute cumulative DVHs and write a wide CSV (Dose(Gy), <struct>...)."""
        import csv
        axis, table = self.compute_dvh(doseNum=doseNum, structNums=structNums,
                                       num_bins=num_bins)
        names = list(table.keys())
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Dose(Gy)"] + names)
            for i in range(len(axis)):
                w.writerow([f"{axis[i]:.5g}"]
                           + [f"{table[n][i]:.4f}" for n in names])
        return path

    def save_screenshot(self, path, dpi=150):
        """Render the current three-view figure and save it to file."""
        fig = self.render_figure()
        fig.savefig(path, dpi=dpi, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        return path

    def setPlanC(self, planC):
        """Swap in a different plan container and rebuild the UI state."""
        self.planC = planC
        self.maskCache.clear()
        self._build_widgets()
        self._on_scan_change(initial=True)
        display(self.ui)
        self._redraw()

    def getPlanC(self):
        return self.planC

    def refresh(self):
        """Redraw after external edits to planC (e.g. importStructureMask)."""
        self.maskCache.clear()
        self._on_scan_change()


def showNB(planC, scanNum=0, figWidth=13.0, *,
           scan_nums=None, struct_nums=None, structNums=None,
           dose_nums=None, doseNum=None,
           windowPreset=None, windowCenter=None, windowWidth=None,
           **_ignored):
    """Display the notebook viewer for planC and return the viewer object.

    The returned object keeps a live handle: `viewer.planC` is always current,
    and `viewer.refresh()` redraws after you modify planC from other cells.

    Optional keyword arguments set the initial state and provide backward
    compatibility with the removed ``showMplNb`` / the ``showNapari``
    signatures: ``scan_nums``/``scanNum``, ``struct_nums``/``structNums``
    (visible structures), ``dose_nums``/``doseNum``, and ``windowPreset`` or
    ``windowCenter`` + ``windowWidth``. List-valued scan/dose use the first
    element; unrecognised keywords are ignored.
    """
    def _first(v, default=0):
        if isinstance(v, (list, tuple, np.ndarray)):
            return int(v[0]) if len(v) else default
        return int(v)

    scan = scan_nums if scan_nums is not None else scanNum
    v = NbViewer(planC, scanNum=_first(scan, 0), figWidth=figWidth)

    structs = struct_nums if struct_nums is not None else structNums
    if structs is not None:
        v.set_structures_visible([int(s) for s in structs])

    dose = dose_nums if dose_nums is not None else doseNum
    if dose is not None:
        dn = _first(dose, -1)
        if dn >= 0:
            v.set_dose(dn)

    if windowPreset is not None and \
            CT_WINDOW_PRESETS.get(windowPreset) is not None:
        v.set_window_preset(windowPreset)
    elif windowCenter is not None and windowWidth is not None:
        v.set_window_level(windowCenter, windowWidth)
    return v
