"""PyCerrViewer: the main viewer window."""
from cerr.viewer.pycerr_gui.common import *  # noqa: F401,F403
from cerr.viewer.pycerr_gui.slice_view import SliceView  # noqa: E402
from cerr.viewer.pycerr_gui.colorbars import DoseColorbarWidget, ScanColorbarWidget  # noqa: E402
from cerr.viewer.pycerr_gui.dialogs import (DvhDialog, ContourDialog, RegQaDialog,  # noqa: E402
                                            ScanDoseExportDialog, StructureExportDialog)
from cerr.viewer.pycerr_gui.uromt_gui import UROMTDialog  # noqa: E402
from cerr.viewer.pycerr_gui.volume3d import Volume3DDialog  # noqa: E402

# Fast contour extraction (matplotlib >= 3.6 ships contourpy); falls back to
# uncached ax.contour when unavailable.
try:
    import contourpy  # noqa: E402
    HAS_CONTOURPY = True
except ImportError:  # pragma: no cover
    HAS_CONTOURPY = False

class PyCerrViewer(QtWidgets.QMainWindow):
    def __init__(self, planC=None):
        super().__init__()
        self.setWindowTitle("pyCERR Viewer")
        self.resize(1480, 920)
        # light-gray theme on this window (also covers the embedded case where
        # a host owns the QApplication and set its own palette)
        self.setPalette(_theme_palette())
        self.setStyleSheet(_THEME_STYLESHEET)
        self.setWindowIcon(pycerr_icon())
        self.setAcceptDrops(True)    # drag-drop DICOM dirs / NIfTI / .pkl

        self.planC = None
        self.scanNum = 0
        self.doseNum = -1            # -1 = no dose displayed
        self.doseAlpha = 0.45
        # dose display mode + CERR-style isodose options
        self.doseDispMode = "colorwash"          # "colorwash" | "isodose"
        self.isodoseLevels = [10, 30, 50, 70, 90, 95, 100]
        self.isodoseUnits = "% of max"           # "% of max" | "% of Rx" | "Gy"
        self.isodoseWidth = 1.5                  # isodose line width
        self.doseInterp = None       # RegularGridInterpolator
        self.maskCache = {}          # structNum -> 3D mask
        self.slices = {w: 0 for w in ("A", "B", "C", "D")}  # winId -> slice
        self.activeWins = ["A", "B", "C"]   # windows shown by current layout
        # last slice per view type, for crosshairs & newly switched windows
        self.lastSlice = {VIEW_AXIAL: 0, VIEW_SAGITTAL: 0, VIEW_CORONAL: 0}
        self.lockViews = False       # sync slices across same-view windows
        self.windowCenter, self.windowWidth = 40.0, 400.0
        self.scanCmap = "gray"       # base-scan colormap
        self.scanAlpha = 1.0         # base-scan opacity
        self.showCrosshairs = True
        self.showOrientation = True  # L/R/A/P/S/I edge labels
        self.upsampleDisplay = False  # sinc-upsample the scan slice for display
        self.showStructDots = False  # contour vertex dots (Alaly dots)
        self.structLineWidth = 1.4   # contour line width
        self.overlayState = {}       # scanIdx -> {"on", "alpha", "cmap"}
        self.overlayCache = {}       # scanIdx -> (interp, vmin, vmax) | None
        self.wlByScan = {}           # scanIdx -> (center, width)
        self.dispByScan = {}         # scanIdx -> (cmapName, alpha)
        self.doseCache = {}          # doseIdx -> (interp, doseMax) | None
        self._pvStructCache = {}     # structNum -> pyvista surface | None
        self._pvDoseCache = {}       # doseIdx -> (isosurface, doseMax) | None
        # 2D render caches; entries self-invalidate by identity of the source
        # object (interpolator / mask), so scrolling back to a slice is free
        self._slice2dCache = {}      # (kind, idx, orient, k, scanNum) -> (interp, 2D)
        self._structSegCache = {}    # (strNum, orient, k, scanNum) -> (mask, segs)
        self._upsampleCache = {}     # (kind, orient, k, scanNum) -> upsampled 2D
        self.plane3dOpacity = 0.6    # translucency of the 3D orthogonal planes
        # per-axis overrides (CERR-style axis menu); None = "Auto" -> follow
        # the global panel selection. "dose" may also be -1 (no dose).
        self.axisSel = {w: {"scan": None, "dose": None, "structs": None}
                        for w in ("A", "B", "C", "D")}
        self._toolWindows = []       # keep IMRTP/ROE windows alive
        self.contourCtl = None       # active ContourDialog (or None)
        self.regCtl = None           # active RegQaDialog (or None)
        # IMRTP beam overlays: list of {"polylines": [Nx3,...], "color": rgb}
        self.beams = []
        # re-render 3D views shortly after slice scrolling stops
        self._timer3d = QtCore.QTimer(self)
        self._timer3d.setSingleShot(True)
        self._timer3d.timeout.connect(self._refresh_3d_views)

        self._build_menus()
        self._build_ui()
        self.statusBar().showMessage("Import a DICOM directory or NIfTI file to begin "
                                     "(File menu).")
        if planC is not None:
            self.setPlanC(planC)

    def setPlanC(self, planC):
        """Attach an existing pyCERR plan container and display it."""
        self.planC = planC
        self.wlByScan.clear()        # scan indices refer to a new plan now
        self.dispByScan.clear()
        self.overlayState.clear()
        self._reset_axis_sel()
        if planC is not None and planC.scan:
            self.after_load()
            self.statusBar().showMessage(
                f"planC loaded: {len(planC.scan)} scan(s), "
                f"{len(planC.structure)} structure(s), {len(planC.dose)} dose(s)")

    def getPlanC(self):
        """Return the current (possibly updated) plan container."""
        return self.planC

    # ----------------------------------------------- external (IMRTP) API ---
    def display_dose(self, doseNum):
        """Refresh from planC and show dose ``doseNum`` (called by IMRTP)."""
        self.after_load(keep_view=True)          # pick up newly added dose
        if self.planC is not None and 0 <= doseNum < len(self.planC.dose):
            self.doseCombo.setCurrentIndex(doseNum + 1)   # +1 for "None"
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def setBeams(self, beams):
        """Display IMRTP beam overlays (2D + 3D). ``beams`` is a list of
        {"polylines": [Nx3 arrays], "color": (r, g, b)} dicts; [] clears."""
        self.beams = list(beams) if beams else []
        self.refresh_views()

    # ------------------------------------------------ scripting/control API --
    # Public methods to drive the GUI programmatically. Each sets the relevant
    # widget(s) so the on-screen controls also reflect the change. Obtain the
    # viewer object from show()/launch(), e.g.:
    #     viewer = show(planC); viewer.set_layout("grid")
    def _window_for(self, orientation):
        """winId of the (first) window showing an orientation, or None."""
        for wid in self.activeWins:
            if self.views[wid].orientation == orientation:
                return wid
        return None

    def set_scan(self, scanNum, keep_view=False):
        """Select the base scan (index into planC.scan).

        With ``keep_view`` the current slice locators, pan/zoom and window are
        preserved instead of resetting to the image centre - used when scrubbing
        co-registered longitudinal time points (e.g. urOMT), so the locators stay
        on the structure the user navigated to."""
        scanNum = int(scanNum)
        if keep_view and scanNum != self.scanNum:
            self.scanCombo.blockSignals(True)
            self.scanCombo.setCurrentIndex(scanNum)
            self.scanCombo.blockSignals(False)
            self.scanNum = scanNum
            self.maskCache.clear()
            self._pvStructCache.clear()
            self._load_scan_geometry(reset_slices=False)
            self._populate_overlay_rows()
            if self.regCtl is not None:
                self.regCtl.sync_base(scanNum)
            self.refresh_views()
        else:
            self.scanCombo.setCurrentIndex(scanNum)

    def set_dose(self, doseNum):
        """Select the displayed dose; doseNum < 0 (or None) hides dose."""
        idx = 0 if doseNum is None or doseNum < 0 else int(doseNum) + 1
        self.doseCombo.setCurrentIndex(idx)

    def set_dose_alpha(self, alpha):
        """Dose colorwash opacity, 0..1."""
        self.alphaSlider.setValue(int(round(float(alpha) * 100)))

    def set_window_level(self, center, width):
        """Manual window center/width for the base scan."""
        self.centerSpin.setValue(float(center))
        self.widthSpin.setValue(float(width))

    # ---- urOMT result overlay on the main slice views --------------------
    _UROMT_LABELS = {"speed": "speed (mm/t)", "rate": "rate r (1/t)",
                     "peclet": "Peclet (-)", "velocity": "|v| (mm/t)",
                     "flux": "flux (a.u. mm/t)",
                     "fluxmag": "|flux| (a.u. mm/t)",
                     "pathlines": "speed (mm/t)"}

    def _uromtEulIntervals(self, index, res):
        """Per-interval Eulerian maps (cached per run), so the maps/flux overlays
        can follow the timepoint slider instead of showing one time-average."""
        from cerr.uromt.analyze import runEULAIntervals
        cache = getattr(self, "_uromtEulIvlCache", None)
        if cache is None:
            cache = self._uromtEulIvlCache = {}
        if index not in cache:
            cache[index] = runEULAIntervals(res)
        return cache[index]

    def _uromtRoiMaskToScan(self, res, scanShape):
        """ROI mask mapped onto the full scan grid (nearest-neighbour zoom for
        resized runs), used to restrict the velocity quiver to the ROI."""
        from scipy.ndimage import zoom
        m = np.asarray(res["mask"]).astype(float)
        rs_, re_, cs_, ce_, ss_, se_ = res["bbox"]
        target = (re_ - rs_, ce_ - cs_, se_ - ss_)
        if m.shape != target:
            m = zoom(m, [t / s for t, s in zip(target, m.shape)], order=0)
        full = np.zeros(scanShape, dtype=bool)
        full[rs_:re_, cs_:ce_, ss_:se_] = m > 0.5
        return full

    def set_uromt_overlay(self, index, view="speed", alpha=0.6, interval=0,
                          subsample=1):
        """Overlay a stored ``planC.urOMT[index]`` result on the 2-D scan views.

        ``view`` is one of 'speed' | 'rate' | 'peclet' (Eulerian colourwash),
        'velocity' | 'flux' (quiver) or 'pathlines'. The overlay is cached on the
        full scan grid so each slice just slices/quivers it (no popup). The maps,
        flux and velocity are all taken from the **time interval** matching the
        selected timepoint (so scrubbing the timepoint slider updates them);
        pathlines are a whole-run summary. The velocity/flux quiver is restricted
        to the ROI mask (the calculation grid). A global ``vrange`` + ``label``
        are stored so the colour-coding is consistent across slices and drives the
        dialog colorbar. ``subsample`` thins the quiver (1 = one arrow/voxel)."""
        from cerr.uromt import viz
        from cerr.uromt.analyze import runGLAD
        runs = getattr(self.planC, "urOMT", None) or []
        if index is None or index < 0 or index >= len(runs):
            return
        run = runs[index]
        res = run.UROMTResult
        Lag = run.UROMTLagrangian or None
        scanShape = self.scan3M.shape
        sf, dr = res.get("sizeFactor", 1.0), res.get("doResize", 0)
        nIv = len(res["u"])
        ivl = int(np.clip(interval, 0, max(0, nIv - 1)))
        ov = {"view": view, "alpha": float(alpha), "index": int(index),
              "subsample": max(1, int(subsample)),
              "label": self._UROMT_LABELS.get(view, view)}
        if view in ("speed", "rate", "peclet", "fluxmag"):
            ei = self._uromtEulIntervals(index, res)
            if view == "fluxmag":                      # |flux| colourwash
                fmag = np.sqrt(np.sum(np.asarray(ei["flux"][ivl]) ** 2, axis=0))
                EulI = {"fluxmag3": fmag, "bbox": res["bbox"],
                        "frameScanNums": res.get("frameScanNums")}
            else:
                EulI = {"speed3": ei["speed"][ivl], "rate3": ei["rate"][ivl],
                        "peclet3": ei["peclet"][ivl], "bbox": res["bbox"],
                        "frameScanNums": res.get("frameScanNums")}
            ov["map3"] = viz.eulerianMapToScan(EulI, field=view,
                                               scanShape=scanShape)
            nz = ov["map3"][ov["map3"] != 0]
            if view == "rate":                         # diverging, symmetric
                a = float(np.percentile(np.abs(nz), 99)) if nz.size else 1.0
                ov["vrange"] = (-a, a)
            else:
                ov["vrange"] = (0.0, float(np.percentile(nz, 99))
                                if nz.size else 1.0)
        elif view in ("velocity", "flux"):
            if view == "flux":
                ei = self._uromtEulIntervals(index, res)
                field = ei["flux"][ivl]                # (3, *n) for this interval
            else:
                field = res["u"][ivl].mean(axis=2)
            comps = viz.fieldToScan(field, res["n"], res["bbox"],
                                    scanShape, sf, dr)
            roi = self._uromtRoiMaskToScan(res, scanShape)   # ROI calc grid
            for c in comps:
                c[~roi] = 0.0
            # Winsorize the magnitude to a robust cap so the non-physical
            # boundary velocities (where the rho-weighted kinetic energy barely
            # constrains u) don't make arrows jump to huge lengths. Vectors above
            # the cap are scaled down to it, preserving direction.
            mag = np.sqrt(sum(c ** 2 for c in comps))
            nz = mag[mag > 0]
            cap = float(np.percentile(nz, 95)) if nz.size else 1.0
            if cap > 0:
                clampF = np.minimum(1.0, cap / (mag + 1e-12))
                for c in comps:
                    c *= clampF
            ov["comps"] = comps
            ov["vrange"] = (0.0, cap)
        elif view == "pathlines":
            Lag = Lag or runGLAD(res)
            ov["segs"] = viz.pathlinesToScanVox(Lag, sf, dr)
            vals = ov["segs"][1]
            ov["vrange"] = (0.0, float(np.percentile(vals, 99))
                            if len(vals) else 1.0)
        else:
            return
        self.uromtOverlay = ov
        self.refresh_views()
        if getattr(self, "_uromtDialog", None) is not None:   # update its colorbar
            self._uromtDialog._updateColorbar(ov)

    def clear_uromt_overlay(self):
        """Remove the urOMT overlay from the scan views."""
        if getattr(self, "uromtOverlay", None) is not None:
            self.uromtOverlay = None
            self.refresh_views()

    def _draw_uromt_overlay(self, winId, ax, extent, hV, vV, slicer):
        from cerr.uromt import viz
        orientation = self.views[winId].orientation
        hA, vA, tA = UROMT_AXES[orientation]
        k = self.slices[winId]
        viz.drawUROMTOverlay(ax, self.uromtOverlay, k, hV, vV, extent, slicer,
                             hA, vA, tA, self.scan3M.shape,
                             alpha=self.uromtOverlay.get("alpha", 0.6),
                             colorbar=False)   # colorbar lives in the urOMT dialog

    def _uromt_3d_geometry(self, maxArrows=1500, maxPaths=400):
        """3-D urOMT overlay geometry (vectors / pathlines) in physical coords
        from the cached overlay; delegates to :func:`cerr.uromt.viz.overlayTo3D`
        (kept there so the coordinate mapping / arrow scaling is headless
        testable)."""
        from cerr.uromt import viz
        return viz.overlayTo3D(getattr(self, "uromtOverlay", None),
                               self.xV, self.yV, self.zV,
                               maxArrows=maxArrows, maxPaths=maxPaths)

    def _add_uromt_3d_vtk(self, pl):
        """Add urOMT vectors / scalar maps / pathlines to the pyvista 3-D scene.

        Colour-coding uses the same global ``vrange`` + colormap as the dialog
        colorbar. Velocity arrows are coloured by magnitude (no start/stop sphere
        markers in 3-D - they would blanket and hide the coloured arrows)."""
        geom = self._uromt_3d_geometry()
        if geom is None:
            return
        ov = self.uromtOverlay
        lo, hi = ov.get("vrange", (None, None))
        cmap = "bwr" if ov.get("view") == "rate" else "turbo"
        # In 3-D the overlay shares the global cutting-plane opacity (the "Plane
        # opacity" slider), not the dialog's 2-D opacity spinbox.
        op = float(self.plane3dOpacity)
        clim = (lo, hi) if (lo is not None and hi is not None and hi > lo) \
            else None
        # No pyvista scalar bar: the single colour legend lives in the urOMT
        # dialog (the colour-coding here uses the same global vrange + colormap).
        if "scalar" in geom:                            # speed / rate / Peclet
            g = geom["scalar"]
            pts = pv.PolyData(g["points"])
            pts["val"] = g["vals"]
            pl.add_mesh(pts, scalars="val", cmap=cmap, clim=clim, opacity=op,
                        point_size=9, render_points_as_spheres=True,
                        show_scalar_bar=False, pickable=False,
                        name="uromt_scalar", render=False)
        if "vectors" in geom:
            g = geom["vectors"]
            pd = pv.PolyData(g["points"])
            pd["vec"] = g["vec"]
            pd["mag"] = g["mag"]
            pd.set_active_vectors("vec")
            arrows = pd.glyph(orient="vec", scale="vec", factor=1.0,
                              geom=pv.Arrow(tip_length=0.3, tip_radius=0.1,
                                            shaft_radius=0.03))
            pl.add_mesh(arrows, scalars="mag", cmap=cmap, clim=clim, opacity=op,
                        show_scalar_bar=False, pickable=False,
                        name="uromt_vec", render=False)
        if "paths" in geom:
            pts_list, conn, off = [], [], 0
            for p in geom["paths"]:
                nP = len(p)
                pts_list.append(p)
                conn.append(np.concatenate(([nP], np.arange(off, off + nP))))
                off += nP
            pd = pv.PolyData()
            pd.points = np.vstack(pts_list)
            pd.lines = np.concatenate(conn).astype(np.int64)
            pl.add_mesh(pd, color="#ffd23f", line_width=2, opacity=op,
                        pickable=False, show_scalar_bar=False,
                        name="uromt_paths", render=False)

    def _add_uromt_3d_mpl(self, ax):
        """Add urOMT vectors / pathlines to the matplotlib 3-D fallback."""
        import matplotlib
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        geom = self._uromt_3d_geometry(maxArrows=600, maxPaths=150)
        if geom is None:
            return
        ov = self.uromtOverlay
        lo, hi = ov.get("vrange", (None, None))
        op = float(self.plane3dOpacity)   # 3-D overlay shares plane opacity
        cmName = "bwr" if ov.get("view") == "rate" else "turbo"
        getc = (matplotlib.colormaps[cmName]
                if hasattr(matplotlib, "colormaps") else cm.get_cmap(cmName))

        def _norm(vals):
            vlo = lo if lo is not None else float(np.min(vals))
            vhi = hi if (hi is not None and hi > vlo) else float(np.max(vals))
            return mcolors.Normalize(vmin=vlo, vmax=max(vhi, vlo + 1e-9))

        if "scalar" in geom:                            # speed / rate / Peclet
            g = geom["scalar"]
            p = g["points"]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=getc(_norm(g["vals"])(
                g["vals"])), s=6, depthshade=False, alpha=op)
        if "vectors" in geom:
            g = geom["vectors"]
            p, v = g["points"], g["vec"]
            # colour each arrow by magnitude (no start/stop markers in 3-D: they
            # would hide the coloured arrows). A fig.colorbar is avoided here too
            # (it would accumulate an axes every refresh); the dialog carries it.
            ax.quiver(p[:, 0], p[:, 1], p[:, 2], v[:, 0], v[:, 1], v[:, 2],
                      colors=getc(_norm(g["mag"])(g["mag"])), linewidth=0.6,
                      normalize=False, alpha=op)
        for p in geom.get("paths", []):
            ax.plot3D(p[:, 0], p[:, 1], p[:, 2], color="#ffd23f", lw=0.8,
                      alpha=op)

    def set_window_preset(self, name):
        """Apply a named CT window preset (see CT_WINDOW_PRESETS)."""
        self.presetCombo.setCurrentText(name)

    def set_scan_colormap(self, name):
        self.scanCmapCombo.setCurrentText(name)

    def set_scan_opacity(self, alpha):
        self.scanAlphaSlider.setValue(int(round(float(alpha) * 100)))

    def set_orientation(self, winId, orientation):
        """Set which orientation a window (A/B/C/D) displays (or VIEW_3D)."""
        self._set_axis_view(winId, orientation)

    def set_slice(self, orientation, k):
        """Move the window showing ``orientation`` to slice index ``k``."""
        wid = self._window_for(orientation)
        if wid is not None:
            n = self._slice_count(orientation)
            self.views[wid].slider.setValue(int(np.clip(k, 0, n - 1)))

    def set_structures_visible(self, which):
        """Show structures: 'all', 'none', or a list of structure indices."""
        if which == "all":
            self._set_all_structs(True)
            return
        if which == "none":
            self._set_all_structs(False)
            return
        want = set(int(i) for i in which)
        self.structList.blockSignals(True)
        for i in range(self.structList.count()):
            it = self.structList.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in want
                             else Qt.Unchecked)
        self.structList.blockSignals(False)
        self.refresh_views()

    def goto_structure(self, strNum):
        """Center all views on a structure's center of mass."""
        self.goto_struct_center(int(strNum))

    def _view_under_cursor(self):
        """Return this window's SliceView currently under the mouse, or None."""
        w = QtWidgets.QApplication.widgetAt(QtGui.QCursor.pos())
        while w is not None and not isinstance(w, SliceView):
            w = w.parentWidget()
        if isinstance(w, SliceView) and self.views.get(w.winId) is w:
            return w
        return None

    def eventFilter(self, obj, event):
        # Arrow keys step the slice of the 2D view under the mouse (hover-based
        # navigation, independent of which widget has keyboard focus).
        if event.type() == QtCore.QEvent.KeyPress and event.key() in (
                Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            view = self._view_under_cursor()
            if view is not None and not view.is3d:
                step = 1 if event.key() in (Qt.Key_Up, Qt.Key_Right) else -1
                sl = view.slider
                sl.setValue(int(np.clip(sl.value() + step,
                                        sl.minimum(), sl.maximum())))
                return True   # consume so a focused slider/widget doesn't also move
        return super().eventFilter(obj, event)

    def set_crosshairs(self, on):
        self.actXhair.setChecked(bool(on))

    def set_orientation_labels(self, on):
        self.actOrient.setChecked(bool(on))

    def set_structure_dots(self, on):
        self.dotsChk.setChecked(bool(on))

    def set_contour_linewidth(self, width):
        self.lineWidthSpin.setValue(float(width))

    def set_lock_views(self, on):
        self.actLock.setChecked(bool(on))

    # -------------------------------------------------------- screenshots ----
    def save_screenshot(self, path, target="window", dpi=150):
        """Save a screenshot to ``path`` (PNG by file extension).

        target:
          "window" - the whole GUI window
          "views"  - just the panel of view windows
          "A"/"B"/"C"/"D" - a single view window (2D figure or 3D render)
          an orientation ("Axial"/"Sagittal"/"Coronal"/"3D") - that view
          "3d"     - the 3D view (whichever window shows it)
        Returns the path written.
        """
        target = str(target)
        QtWidgets.QApplication.processEvents()
        if target == "window":
            self.grab().save(path)
        elif target == "views":
            self.viewContainer.grab().save(path)
        else:
            wid = target if target in self.views else self._window_for(
                VIEW_3D if target.lower() == "3d" else target)
            if wid is None:
                raise ValueError(f"No view window for target {target!r}")
            v = self.views[wid]
            if v.is3d and v.vtk_widget is not None:
                v.vtk_widget.screenshot(path)
            elif v.is3d:
                v.fig.savefig(path, dpi=dpi, facecolor="black")
            else:
                v.fig.savefig(path, dpi=dpi, facecolor="black",
                              bbox_inches="tight", pad_inches=0)
        return path

    # ----------------------------------------------- registration QA / DVH --
    def start_reg_qa(self, base=None, moving=None, mode="Mirrorscope",
                     size=None, base_frac=None):
        """Open and configure the registration QA tool programmatically.

        base/moving: scan indices; mode: 'Mirrorscope'|'Sidebyside'|
        'AlternateGrid'|'Toggle'; size: mirror-box/tile size (cm);
        base_frac: Toggle-mode base weight (0..1). Returns the RegQaDialog.
        """
        if self.planC is None or len(self.planC.scan) < 2:
            raise ValueError("Registration QA needs at least two scans.")
        if self.regCtl is None or not self.regCtl.isVisible():
            RegQaDialog(self).show()       # sets self.regCtl
        self.regCtl.configure(base=base, moving=moving, mode=mode, size=size,
                              base_frac=base_frac)
        return self.regCtl

    def stop_reg_qa(self):
        """Close the registration QA tool if open."""
        if self.regCtl is not None:
            self.regCtl.close()

    def compute_dvh(self, doseNum=None, structNums=None, num_bins=400):
        """Compute cumulative DVHs. Returns (doseAxis_Gy, {name: volPct}),
        each structure interpolated onto a shared dose axis (0..global max)."""
        planC = self.planC
        if planC is None or not planC.dose:
            raise ValueError("A dose is required to compute DVHs.")
        if doseNum is None:
            doseNum = self.doseNum if self.doseNum >= 0 else 0
        if not 0 <= doseNum < len(planC.dose):
            raise ValueError(f"doseNum {doseNum} out of range.")
        if structNums is None:
            structNums = list(range(len(planC.structure)))
        elif isinstance(structNums, (int, np.integer)):
            structNums = [int(structNums)]

        raw, gmax = {}, 0.0
        for strNum in structNums:
            try:
                dosesV, volsV, isErr = cerrDvh.getDVH(strNum, doseNum, planC)
            except Exception:  # noqa: BLE001
                continue
            if isErr or dosesV is None or len(dosesV) == 0:
                continue
            binWidth = max(float(np.max(dosesV)) / 400.0, 1e-3)
            doseBinsV, volsHistV = cerrDvh.doseHist(dosesV, volsV, binWidth)
            cumVols = np.flip(np.cumsum(np.flip(volsHistV)))
            if cumVols[0] <= 0:
                continue
            raw[strNum] = (np.asarray(doseBinsV, dtype=float),
                           100.0 * cumVols / cumVols[0])
            gmax = max(gmax, float(doseBinsV[-1]))
        if not raw:
            raise ValueError("No DVH could be computed for the given "
                             "structures/dose.")
        axis = np.linspace(0.0, gmax, int(num_bins))
        table = {}
        for strNum, (db, cp) in raw.items():
            name = planC.structure[strNum].structureName
            table[name] = np.interp(axis, db, cp, left=cp[0], right=0.0)
        return axis, table

    def export_dvh(self, path, doseNum=None, structNums=None, num_bins=400):
        """Compute cumulative DVHs and write them to a CSV (wide format:
        Dose(Gy), <struct>, ...). Returns (doseAxis, {name: volPct})."""
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
        return axis, table

    def closeEvent(self, event):
        # release VTK render windows cleanly to avoid shutdown crashes
        for v in self.views.values():
            if v.vtk_widget is not None:
                try:
                    v.vtk_widget.close()
                except Exception:  # noqa: BLE001
                    pass
                v.vtk_widget = None
        super().closeEvent(event)

    # ------------------------------------------------------------------ UI --
    def _build_menus(self):
        m = self.menuBar()
        fileM = m.addMenu("&File")
        fileM.addAction("Import &DICOM directory...", self.import_dicom)
        fileM.addAction("Import &NIfTI scan...", self.import_nii_scan)
        self.actNiiDose = fileM.addAction("Import NIfTI d&ose...", self.import_nii_dose)
        self.actNiiStr = fileM.addAction("Import NIfTI &structure(s)...",
                                         self.import_nii_struct)
        fileM.addSeparator()
        exportM = fileM.addMenu("&Export")
        self.actExpScan = exportM.addAction("Scan to NIfTI...",
                                            self.export_scan_nii)
        self.actExpDose = exportM.addAction("Dose to NIfTI...",
                                            self.export_dose_nii)
        self.actExpStrNii = exportM.addAction("Structure(s) to NIfTI...",
                                              self.export_struct_nii)
        self.actExpStrDcm = exportM.addAction(
            "Structure(s) to DICOM RTSTRUCT...", self.export_struct_dicom)
        fileM.addSeparator()
        fileM.addAction("&Open planC (.pkl)...", self.open_pkl)
        fileM.addAction("&Save planC (.pkl)...", self.save_pkl)
        fileM.addSeparator()
        fileM.addAction("E&xit", self.close)

        viewM = m.addMenu("&View")
        layoutM = viewM.addMenu("&Layout")
        lgrp = QtWidgets.QActionGroup(layoutM)
        for key, label in (("single", "&One view"),
                           ("two", "&Two views side-by-side"),
                           ("default", "One large + two stacke&d"),
                           ("columns", "Three &columns"),
                           ("grid", "&Four views (2x2)")):
            act = layoutM.addAction(label)
            act.setCheckable(True)
            act.setChecked(key == "default")
            lgrp.addAction(act)
            act.triggered.connect(lambda _=False, k=key: self.set_layout(k))
        viewM.addSeparator()
        self.actXhair = viewM.addAction("Show &crosshairs")
        self.actXhair.setCheckable(True)
        self.actXhair.setChecked(True)
        self.actXhair.setShortcut("X")
        self.actXhair.toggled.connect(self.on_crosshair_toggled)
        self.actOrient = viewM.addAction("Show &orientation labels (L/R/A/P/S/I)")
        self.actOrient.setCheckable(True)
        self.actOrient.setChecked(True)
        self.actOrient.setShortcut("O")
        self.actOrient.toggled.connect(self.on_orientation_toggled)
        self.actUpsample = viewM.addAction("&Upsample display (sinc)")
        self.actUpsample.setCheckable(True)
        self.actUpsample.setChecked(False)
        self.actUpsample.setToolTip(
            "Sinc-upsample the scan slice for display to the finer of the two "
            "in-plane voxel resolutions (smoother, especially on thick-slice "
            "planes); does not change the stored data")
        self.actUpsample.toggled.connect(self.on_upsample_toggled)
        self.actLock = viewM.addAction("&Lock slices across matching views")
        self.actLock.setCheckable(True)
        self.actLock.setShortcut("L")
        self.actLock.toggled.connect(self.on_lock_toggled)
        actReset = viewM.addAction("&Reset pan/zoom", self.reset_all_views)
        actReset.setShortcut("R")
        viewM.addSeparator()
        viewM.addAction("Scan &display / window...",
                        self.show_scan_display_dialog)
        viewM.addAction("3D &Visualization (volume render)...",
                        self.show_3d_volume)

        toolsM = m.addMenu("&Tools")
        toolsM.addAction("&Contouring (draw/edit structures)...",
                         self.show_contour_dialog)
        toolsM.addAction("&DVH...", self.show_dvh_dialog)
        toolsM.addAction("Registration &QA (compare scans)...",
                         self.show_reg_dialog)
        toolsM.addSeparator()
        toolsM.addAction("&IMRTP (beamlet dose calculation)...",
                         self.show_imrtp_gui)
        toolsM.addAction("&ROE (Radiotherapy Outcomes Explorer)...",
                         self.show_roe_gui)
        toolsM.addAction("&urOMT (fluid transport on longitudinal scans)...",
                         self.show_uromt_dialog)
        toolsM.addSeparator()
        toolsM.addAction("Re&fresh from planC",
                         lambda: self.after_load(keep_view=True))

        helpM = m.addMenu("&Help")
        helpM.addAction("&Controls", self.show_controls)
        helpM.addAction("&About", self.show_about)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)
        h.setContentsMargins(2, 2, 2, 2)

        # ---------------- left control panel (CERR-style) ----------------
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(290)
        pl = QtWidgets.QVBoxLayout(panel)

        grpScan = QtWidgets.QGroupBox("Scan")
        gl = QtWidgets.QVBoxLayout(grpScan)
        self.scanCombo = QtWidgets.QComboBox()
        self.scanCombo.currentIndexChanged.connect(self.on_scan_changed)
        gl.addWidget(self.scanCombo)
        # Window/level, colormap, opacity and the scan colorbar live in a
        # separate "Scan Display" dialog so the main figure just shows the base
        # scan and fused overlays. The windowing widgets are built there (in
        # _build_scan_display_dialog) but remain accessible as self.* .
        self.scanDisplayBtn = QtWidgets.QPushButton("Scan display / window...")
        self.scanDisplayBtn.setToolTip(
            "Open the scan display dialog (window/level, colormap, opacity, "
            "colorbar) for the selected scan")
        self.scanDisplayBtn.clicked.connect(self.show_scan_display_dialog)
        gl.addWidget(self.scanDisplayBtn)
        pl.addWidget(grpScan)

        # fused scan overlays (visible only when >1 scan is loaded)
        self.grpOverlay = QtWidgets.QGroupBox("Scan overlays (fusion)")
        ovl = QtWidgets.QVBoxLayout(self.grpOverlay)
        ovl.setContentsMargins(4, 4, 4, 4)
        overlayHost = QtWidgets.QWidget()
        self.overlayLayout = QtWidgets.QVBoxLayout(overlayHost)
        self.overlayLayout.setContentsMargins(0, 0, 0, 0)
        self.overlayLayout.setSpacing(2)
        self.overlayLayout.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setMaximumHeight(150)
        scroll.setWidget(overlayHost)
        ovl.addWidget(scroll)
        self.grpOverlay.setVisible(False)
        pl.addWidget(self.grpOverlay)

        grpStr = QtWidgets.QGroupBox("Structures")
        sl = QtWidgets.QVBoxLayout(grpStr)
        btnRow = QtWidgets.QHBoxLayout()
        allBtn = QtWidgets.QPushButton("All")
        noneBtn = QtWidgets.QPushButton("None")
        allBtn.clicked.connect(lambda: self._set_all_structs(True))
        noneBtn.clicked.connect(lambda: self._set_all_structs(False))
        btnRow.addWidget(allBtn), btnRow.addWidget(noneBtn)
        self.structScanFilterChk = QtWidgets.QCheckBox("Current scan")
        self.structScanFilterChk.setToolTip(
            "List only structures associated with the current scan")
        self.structScanFilterChk.toggled.connect(self._on_struct_scan_filter)
        btnRow.addWidget(self.structScanFilterChk)
        btnRow.addStretch(1)
        sl.addLayout(btnRow)
        self.structList = QtWidgets.QListWidget()
        self.structList.itemChanged.connect(lambda *_: self.refresh_views())
        self.structList.itemDoubleClicked.connect(self.on_struct_double_click)
        self.structList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.structList.customContextMenuRequested.connect(
            self._on_struct_context_menu)
        self.structList.setToolTip(
            "Double-click a structure to center all views on it; "
            "right-click to change its color")
        # black list background so the per-structure colored names all show
        self.structList.setStyleSheet(
            "QListWidget { background: #000; }"
            "QListWidget::item:selected { background: #3a6ea5; }")
        sl.addWidget(self.structList)

        optRow = QtWidgets.QHBoxLayout()
        self.dotsChk = QtWidgets.QCheckBox("Dots")
        self.dotsChk.setToolTip("Show contour vertex dots (Alaly dots)")
        self.dotsChk.toggled.connect(self.on_struct_dots)
        optRow.addWidget(self.dotsChk)
        optRow.addWidget(QtWidgets.QLabel("Line:"))
        self.lineWidthSpin = QtWidgets.QDoubleSpinBox()
        self.lineWidthSpin.setRange(0.2, 6.0)
        self.lineWidthSpin.setSingleStep(0.2)
        self.lineWidthSpin.setValue(self.structLineWidth)
        self.lineWidthSpin.setToolTip("Contour line width")
        self.lineWidthSpin.valueChanged.connect(self.on_struct_linewidth)
        optRow.addWidget(self.lineWidthSpin)
        optRow.addStretch(1)
        sl.addLayout(optRow)
        pl.addWidget(grpStr, stretch=1)

        grpDose = QtWidgets.QGroupBox("Dose")
        dl = QtWidgets.QVBoxLayout(grpDose)
        self.doseCombo = QtWidgets.QComboBox()
        self.doseCombo.currentIndexChanged.connect(self.on_dose_changed)
        dl.addWidget(self.doseCombo)
        cmapRow = QtWidgets.QHBoxLayout()
        cmapRow.addWidget(QtWidgets.QLabel("Colormap:"))
        self.doseCmapCombo = QtWidgets.QComboBox()
        self.doseCmapCombo.addItems(DOSE_CMAP_NAMES)
        self.doseCmapCombo.setCurrentText(DEFAULT_DOSE_CMAP)
        self.doseCmapCombo.setToolTip("Dose colorwash colormap")
        self.doseCmapCombo.currentTextChanged.connect(self.on_dose_cmap)
        cmapRow.addWidget(self.doseCmapCombo, 1)
        dl.addLayout(cmapRow)
        aRow = QtWidgets.QHBoxLayout()
        aRow.addWidget(QtWidgets.QLabel("Colorwash alpha:"))
        self.alphaSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.alphaSlider.setRange(0, 100)
        self.alphaSlider.setValue(int(self.doseAlpha * 100))
        self.alphaSlider.valueChanged.connect(self.on_alpha)
        aRow.addWidget(self.alphaSlider)
        dl.addLayout(aRow)
        # display mode (colorwash vs CERR-style isodose lines) + line width
        dispRow = QtWidgets.QHBoxLayout()
        dispRow.addWidget(QtWidgets.QLabel("Display:"))
        self.doseDispCombo = QtWidgets.QComboBox()
        self.doseDispCombo.addItems(["Colorwash", "Isodose lines"])
        self.doseDispCombo.setToolTip(
            "Show the dose as a colorwash or as isodose contour lines")
        self.doseDispCombo.currentIndexChanged.connect(self.on_dose_disp_mode)
        dispRow.addWidget(self.doseDispCombo, 1)
        self.isoWidthSpin = QtWidgets.QDoubleSpinBox()
        self.isoWidthSpin.setRange(0.5, 6.0)
        self.isoWidthSpin.setSingleStep(0.5)
        self.isoWidthSpin.setValue(self.isodoseWidth)
        self.isoWidthSpin.setToolTip("Isodose line width")
        self.isoWidthSpin.valueChanged.connect(self.on_isodose_width)
        dispRow.addWidget(self.isoWidthSpin)
        dl.addLayout(dispRow)
        # isodose levels (comma separated) + units, as in MATLAB CERR
        isoRow = QtWidgets.QHBoxLayout()
        isoRow.addWidget(QtWidgets.QLabel("Levels:"))
        self.isoLevelsEdit = QtWidgets.QLineEdit(
            ", ".join(str(v) for v in self.isodoseLevels))
        self.isoLevelsEdit.setToolTip(
            "Comma-separated isodose levels, in the selected units")
        self.isoLevelsEdit.editingFinished.connect(self.on_isodose_levels)
        isoRow.addWidget(self.isoLevelsEdit, 1)
        self.isoUnitsCombo = QtWidgets.QComboBox()
        self.isoUnitsCombo.addItems(["% of max", "% of Rx", "Gy"])
        self.isoUnitsCombo.setToolTip(
            "Interpret levels as percent of the maximum dose, percent of the "
            "prescription dose, or absolute dose (Gy)")
        self.isoUnitsCombo.currentTextChanged.connect(self.on_isodose_units)
        isoRow.addWidget(self.isoUnitsCombo)
        dl.addLayout(isoRow)
        self._update_isodose_enabled()
        pl.addWidget(grpDose)

        # (3D-view controls moved out of the panel: the urOMT overlay opacity
        # slider now lives in the urOMT dialog; plane locators are always shown
        # in the "3D Cut Planes" view.)

        h.addWidget(panel)

        # ------- right: view windows (each can show any orientation) -------
        self.views = {
            "A": SliceView("A", VIEW_AXIAL),
            "B": SliceView("B", VIEW_SAGITTAL),
            "C": SliceView("C", VIEW_CORONAL),
            "D": SliceView("D", VIEW_3D),       # 4th window: 3D by default
        }
        self.views["D"].set_projection(True)
        for v in self.views.values():
            v.sliceChanged.connect(self.on_slice_changed)
            v.cursorMoved.connect(self.on_cursor_moved)
            v.viewReset.connect(self._on_view_reset)
            v.contextRequested.connect(self._show_axis_menu)
            v.rulerChanged.connect(self.on_ruler_changed)
            v.crosshairDragged.connect(self.on_crosshair_dragged)

        # Arrow keys (Up/Down/Left/Right) change the slice of whichever 2D view
        # is under the mouse, regardless of focus.
        QtWidgets.QApplication.instance().installEventFilter(self)

        # container whose contents are rebuilt by _apply_layout()
        self.viewContainer = QtWidgets.QWidget()
        self.viewLay = QtWidgets.QVBoxLayout(self.viewContainer)
        self.viewLay.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.viewContainer, stretch=1)
        self._apply_layout("default")

        # the base-scan colorbar (window/colormap/display range) lives in the
        # Scan Display dialog, built here so its widgets exist before load
        self._build_scan_display_dialog()

        # standalone dose colorbar (hidden until a dose is selected)
        self.colorbar = DoseColorbarWidget()
        self.colorbar.rangesChanged.connect(self._on_colorbar_ranges_changed)
        self.colorbar.setVisible(False)
        h.addWidget(self.colorbar)

    # ------------------------------------------------ scan display dialog ----
    def _build_scan_display_dialog(self):
        """Modeless dialog holding the base-scan window/level, colormap,
        opacity and the scan colorbar. Its widgets stay accessible as self.* so
        the existing handlers keep working; the main panel only selects the
        scan and manages fused overlays."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Scan Display")
        dlg.setModal(False)
        lay = QtWidgets.QHBoxLayout(dlg)

        ctrl = QtWidgets.QVBoxLayout()

        # scan selection (mirrors the main-panel combo)
        ctrl.addWidget(QtWidgets.QLabel("Scan:"))
        self.scanComboDlg = QtWidgets.QComboBox()
        self.scanComboDlg.currentIndexChanged.connect(self._on_dlg_scan_changed)
        ctrl.addWidget(self.scanComboDlg)

        # window preset
        wlRow = QtWidgets.QHBoxLayout()
        self.presetCombo = QtWidgets.QComboBox()
        self.presetCombo.addItems(CT_WINDOW_PRESETS.keys())
        self.presetCombo.setCurrentText("Soft Tissue")
        self.presetCombo.currentTextChanged.connect(self.on_preset)
        wlRow.addWidget(QtWidgets.QLabel("Window:"))
        wlRow.addWidget(self.presetCombo, 1)
        ctrl.addLayout(wlRow)

        # manual center / width
        cwRow = QtWidgets.QHBoxLayout()
        self.centerSpin = QtWidgets.QDoubleSpinBox()
        self.centerSpin.setRange(-5000, 50000)
        self.centerSpin.setValue(self.windowCenter)
        self.widthSpin = QtWidgets.QDoubleSpinBox()
        self.widthSpin.setRange(0.01, 100000)
        self.widthSpin.setValue(self.windowWidth)
        for s in (self.centerSpin, self.widthSpin):
            s.valueChanged.connect(self.on_manual_wl)
        cwRow.addWidget(QtWidgets.QLabel("C:"))
        cwRow.addWidget(self.centerSpin)
        cwRow.addWidget(QtWidgets.QLabel("W:"))
        cwRow.addWidget(self.widthSpin)
        ctrl.addLayout(cwRow)

        # colormap
        cmRow = QtWidgets.QHBoxLayout()
        cmRow.addWidget(QtWidgets.QLabel("Colormap:"))
        self.scanCmapCombo = QtWidgets.QComboBox()
        self.scanCmapCombo.addItems(SCAN_CMAPS)
        self.scanCmapCombo.setCurrentText(self.scanCmap)
        self.scanCmapCombo.currentTextChanged.connect(self.on_scan_cmap)
        cmRow.addWidget(self.scanCmapCombo, 1)
        ctrl.addLayout(cmRow)

        # opacity
        opRow = QtWidgets.QHBoxLayout()
        opRow.addWidget(QtWidgets.QLabel("Opacity:"))
        self.scanAlphaSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.scanAlphaSlider.setRange(0, 100)
        self.scanAlphaSlider.setValue(int(self.scanAlpha * 100))
        self.scanAlphaSlider.valueChanged.connect(self.on_scan_alpha)
        opRow.addWidget(self.scanAlphaSlider)
        ctrl.addLayout(opRow)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        # the scan colorbar (colormap-mapping & display ranges, colormap menu)
        self.scanColorbar = ScanColorbarWidget()
        self.scanColorbar.rangesChanged.connect(self._on_scan_colorbar_changed)
        lay.addWidget(self.scanColorbar)

        dlg.resize(360, 340)
        self.scanDisplayDialog = dlg

    def show_scan_display_dialog(self):
        """Open (or raise) the Scan Display dialog."""
        if self.planC is None or not self.planC.scan:
            _show_info(self, "Scan Display", "Load a scan first.")
            return
        self.scanDisplayDialog.show()
        self.scanDisplayDialog.raise_()
        self.scanDisplayDialog.activateWindow()

    def _dlg_target(self):
        """Scan edited by the Scan Display dialog (may differ from the base)."""
        idx = self.scanComboDlg.currentIndex() \
            if hasattr(self, "scanComboDlg") else self.scanNum
        nScans = len(self.planC.scan) if self.planC else 0
        return idx if 0 <= idx < nScans else self.scanNum

    def _on_dlg_scan_changed(self, idx):
        """Dialog scan combo -> edit that scan's display settings (the base
        scan shown in the views is only changed from the main panel combo)."""
        if idx < 0:
            return
        self._dlg_load_target(idx)

    def _dlg_load_target(self, t):
        """Load scan ``t``'s stored window / colormap / opacity into the Scan
        Display dialog widgets and colorbar (without re-rendering)."""
        if self.planC is None or not (0 <= t < len(self.planC.scan)):
            return
        cmap, alpha = self._scan_display(t)
        self.dispByScan.setdefault(t, (cmap, alpha))
        wl = self.wlByScan.get(t)
        lo = hi = None
        if t == self.scanNum and getattr(self, "_scanDataRange", None):
            lo, hi = self._scanDataRange
        else:
            res = self._overlay_interp(t)
            if res is not None:
                lo, hi = res[1], res[2]
        if wl is None:
            wl = ((lo + hi) / 2.0, max(hi - lo, 1.0)) if lo is not None \
                else (self.windowCenter, self.windowWidth)
            self.wlByScan[t] = wl
        for sp, val in ((self.centerSpin, wl[0]), (self.widthSpin, wl[1])):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self.scanCmapCombo.blockSignals(True)
        self.scanCmapCombo.setCurrentText(cmap)
        self.scanCmapCombo.blockSignals(False)
        self.scanAlphaSlider.blockSignals(True)
        self.scanAlphaSlider.setValue(int(round(alpha * 100)))
        self.scanAlphaSlider.blockSignals(False)
        if lo is None:
            lo, hi = wl[0] - wl[1] / 2.0, wl[0] + wl[1] / 2.0
        self.scanColorbar.setScan(wl[0], wl[1], lo, hi)
        self.scanColorbar._set_cmap(cmap)
        self.scanColorbar.update()

    def _open_scan_display_for(self, scanIdx):
        """Open the Scan Display dialog targeting the given scan."""
        self.show_scan_display_dialog()
        if hasattr(self, "scanComboDlg") \
                and self.scanComboDlg.currentIndex() != scanIdx:
            self.scanComboDlg.setCurrentIndex(scanIdx)  # -> _dlg_load_target

    def _mirror_scan_combo(self):
        """Copy the main scan combo's items/selection into the dialog combo."""
        if not hasattr(self, "scanComboDlg"):
            return
        self.scanComboDlg.blockSignals(True)
        self.scanComboDlg.clear()
        self.scanComboDlg.addItems(
            [self.scanCombo.itemText(i) for i in range(self.scanCombo.count())])
        self.scanComboDlg.setCurrentIndex(self.scanCombo.currentIndex())
        self.scanComboDlg.blockSignals(False)

    def _sync_dlg_scan_index(self):
        """Keep the dialog combo's selection in step with the main combo."""
        if hasattr(self, "scanComboDlg") \
                and self.scanComboDlg.currentIndex() != self.scanNum:
            self.scanComboDlg.blockSignals(True)
            self.scanComboDlg.setCurrentIndex(self.scanNum)
            self.scanComboDlg.blockSignals(False)

    # --------------------------------------------------------------- I/O ----
    def _busy(self, msg):
        self.statusBar().showMessage(msg)
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()

    def _done(self, msg="Ready."):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(msg)

    def import_dicom(self, path=None):
        if not path:
            path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select DICOM directory")
        if not path:
            return
        self._do_dicom_import(path, f"Importing DICOM from {path} ...",
                              f"Imported {path}")

    def import_dicom_files(self, fileList):
        """Import only the given DICOM file(s), not their whole directory -
        e.g. a dropped RTSTRUCT, so we don't pull in the rest of the folder."""
        fileList = [f for f in fileList if f]
        if not fileList:
            return
        n = len(fileList)
        self._do_dicom_import(list(fileList),
                              f"Importing {n} DICOM file(s) ...",
                              f"Imported {n} DICOM file(s)")

    def _do_dicom_import(self, source, busyMsg, doneMsg):
        """Shared DICOM import: source is a directory path or a list of file
        paths (both accepted by loadDcmDir). Reports skipped duplicates."""
        try:
            self._busy(busyMsg)
            before = ((len(self.planC.scan), len(self.planC.structure),
                       len(self.planC.dose)) if self.planC else (0, 0, 0))
            import warnings as _warnings
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                self.planC = pc.loadDcmDir(source, initplanC=self.planC or "")
            self.after_load()
            self._done(doneMsg)
            self._report_skipped_dups(caught, before)
        except Exception as e:  # noqa: BLE001
            self._done()
            _show_error(self, "Import error", str(e))

    def _report_skipped_dups(self, caught, before):
        """Surface duplicates skipped by loadDcmDir (per-category, accurate for
        both fully- and partially-duplicate imports)."""
        msgs = [str(w.message) for w in caught
                if "already exists in planC" in str(w.message)]
        if not msgs:
            return
        after = (len(self.planC.scan), len(self.planC.structure),
                 len(self.planC.dose))
        lines = []
        for i, (label, prefix) in enumerate(
                (("scan", "Scan"), ("structure", "Structure"),
                 ("dose", "Dose"))):
            skipped = sum(1 for m in msgs if m.startswith(prefix))
            if skipped == 0:
                continue
            if after[i] - before[i] == 0:
                lines.append(f"No {label}s were imported as they already "
                             f"exist in planC.")
            else:
                lines.append(f"{skipped} {label}{'' if skipped == 1 else 's'} "
                             f"already exist in planC and "
                             f"{'was' if skipped == 1 else 'were'} not "
                             f"imported.")
        if lines:
            _show_info(self, "Import", "\n".join(lines))

    # ----------------------------------------------------- drag & drop ------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if u.toLocalFile()]
        event.acceptProposedAction()
        # Dropped DICOM *files* are imported as exactly those files (e.g. a
        # single RTSTRUCT), NOT their whole folder - dropping one structure
        # file must not pull in the scan series sitting next to it. Dropped
        # *directories* still import the whole directory.
        dcmFiles, dirs, others = [], [], []
        for p in paths:
            if os.path.isdir(p):
                dirs.append(p)
            elif os.path.isfile(p) and p.lower().endswith(".dcm"):
                dcmFiles.append(p)
            else:
                others.append(p)
        # directories (scans) first, then the dropped DICOM files (structures/
        # doses that reference them), then dose/structure NIfTIs
        for d in dirs:
            self.load_path(d)
        if dcmFiles:
            self.import_dicom_files(dcmFiles)
        for p in sorted(others, key=self._drop_priority):
            self.load_path(p)

    @staticmethod
    def _drop_priority(path):
        low = os.path.basename(path).lower()
        if low.endswith((".nii", ".nii.gz")) and any(
                k in low for k in ("dose", "mask", "label", "seg", "struct",
                                   "roi")):
            return 1                  # dose/structure NIfTIs load last
        return 0

    def load_path(self, path):
        """Load a dropped/opened path: a DICOM directory, a NIfTI file
        (.nii/.nii.gz - scan, or dose/structure by name heuristic when a scan
        is loaded), a single DICOM file (only that file is imported), or a
        .pkl plan container."""
        path = str(path)
        low = path.lower()
        try:
            if os.path.isdir(path):
                self.import_dicom(path)
            elif low.endswith(".pkl"):
                self._busy("Loading planC ...")
                self.planC = pc.loadPlanCFromPkl(path)
                self.wlByScan.clear()
                self.dispByScan.clear()
                self.overlayState.clear()
                self._reset_axis_sel()
                self.after_load()
                self._done(f"Loaded {path}")
            elif low.endswith((".nii", ".nii.gz")):
                self._load_nii(path)
            elif low.endswith(".dcm"):
                self.import_dicom_files([path])
            else:
                _show_info(
                    self, "Open", f"Unsupported file type:\n{path}")
        except Exception as e:  # noqa: BLE001
            self._done()
            _show_error(self, "Load error", str(e))

    def _nii_grid_dims(self, path):
        """First three header dimensions of a NIfTI file, or None."""
        try:
            import nibabel as nib
            return tuple(int(d) for d in nib.load(path).shape[:3])
        except Exception:  # noqa: BLE001
            return None

    def _assoc_scan_for_nii(self, path):
        """Scan index to associate a NIfTI dose / label mask with: the current
        scan when its grid matches the file, else the first scan whose grid
        does (auto-suggestion); falls back to the current scan when nothing
        matches. Returns (scanIndex, note) where note is '' or a short
        message describing the auto-association."""
        dims = self._nii_grid_dims(path)
        if dims is None:
            return self.scanNum, ""

        def _matches(i):
            return sorted(self.planC.scan[i].scanArray.shape[:3]) \
                == sorted(dims)

        if _matches(self.scanNum):
            return self.scanNum, ""
        cand = [i for i in range(len(self.planC.scan)) if _matches(i)]
        if cand:
            return cand[0], f" (grid matches scan {cand[0]})"
        return self.scanNum, (" (warning: grid matches no scan; associated "
                              f"with current scan {self.scanNum})")

    def _load_nii(self, path):
        """Load a NIfTI file. With no scan loaded it can only be a scan; when
        a scan exists, ask the user whether the file is a scan, a dose or a
        segmentation (label mask), preselecting a guess from the filename."""
        if self.planC is None or not self.planC.scan:
            self._load_nii_as(path, "scan")
            return
        name = os.path.basename(path).lower()
        if "dose" in name:
            guess = "dose"
        elif any(k in name for k in ("mask", "label", "seg", "struct", "roi")):
            guess = "segmentation"
        else:
            guess = "scan"
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setWindowTitle("Import NIfTI")
        box.setText(f"Load {os.path.basename(path)} as:")
        btns = {}
        for kind in ("scan", "dose", "segmentation"):
            btns[kind] = box.addButton(kind.capitalize(),
                                       QtWidgets.QMessageBox.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(btns[guess])
        # non-modal (a modal exec_ can hang in an integrated event loop)
        box.setModal(False)
        box.setAttribute(Qt.WA_DeleteOnClose, True)

        def _clicked(btn):
            for kind, b in btns.items():
                if btn is b:
                    self._load_nii_as(path, kind)
                    return
        box.buttonClicked.connect(_clicked)
        box.show()
        box.raise_()
        box.activateWindow()

    def _load_nii_as(self, path, kind):
        """Load a NIfTI file as the given kind: 'scan', 'dose' or
        'segmentation'."""
        try:
            if kind == "dose":
                assocScan, note = self._assoc_scan_for_nii(path)
                self._busy("Loading NIfTI dose ...")
                self.planC = pc.loadNiiDose(path, assocScan, self.planC)
                self.after_load(keep_view=True)
                self._done(f"Loaded dose {path}{note}")
            elif kind == "segmentation":
                assocScan, note = self._assoc_scan_for_nii(path)
                self._busy("Loading NIfTI structure(s) ...")
                self.planC = pc.loadNiiStructure(path, assocScan, self.planC)
                self.after_load(keep_view=True)
                self._done(f"Loaded structures from {path}{note}")
            else:
                self._busy("Loading NIfTI scan ...")
                self.planC = pc.loadNiiScan(path, imageType="CT SCAN",
                                            initplanC=self.planC or "")
                self.after_load()
                # show the newly imported scan instead of resetting to scan 0
                self.scanCombo.setCurrentIndex(len(self.planC.scan) - 1)
                self._done(f"Loaded {path}")
        except Exception as e:  # noqa: BLE001
            self._done()
            _show_error(self, "Import error", str(e))

    def import_nii_scan(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select NIfTI scan", filter="NIfTI (*.nii *.nii.gz)")
        if f:
            self._load_nii_as(f, "scan")

    def import_nii_dose(self):
        if self.planC is None or not self.planC.scan:
            _show_info(self, "pyCERR", "Load a scan first.")
            return
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select NIfTI dose", filter="NIfTI (*.nii *.nii.gz)")
        if f:
            self._load_nii_as(f, "dose")

    def import_nii_struct(self):
        if self.planC is None or not self.planC.scan:
            _show_info(self, "pyCERR", "Load a scan first.")
            return
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select NIfTI label mask", filter="NIfTI (*.nii *.nii.gz)")
        if f:
            self._load_nii_as(f, "segmentation")

    def open_pkl(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open planC pickle", filter="Pickle (*.pkl)")
        if not f:
            return
        try:
            self._busy("Loading planC ...")
            self.planC = pc.loadPlanCFromPkl(f)
            self.wlByScan.clear()    # scan indices refer to a new plan now
            self.dispByScan.clear()
            self.overlayState.clear()
            self._reset_axis_sel()
            self.after_load()
            self._done(f"Loaded {f}")
        except Exception as e:  # noqa: BLE001
            self._done()
            _show_error(self, "Load error", str(e))

    def save_pkl(self):
        if self.planC is None:
            return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save planC pickle", filter="Pickle (*.pkl)")
        if not f:
            return
        try:
            import pickle
            with open(f, "wb") as fh:
                pickle.dump(self.planC, fh)
            self.statusBar().showMessage(f"Saved {f}")
        except Exception as e:  # noqa: BLE001
            _show_error(self, "Save error", str(e))

    # ------------------------------------------------------------ export ----
    def export_scan_nii(self):
        if self.planC is None or not self.planC.scan:
            _show_info(self, "Export scan", "No scan to export.")
            return
        ScanDoseExportDialog(self, "scan").show()

    def export_dose_nii(self):
        if self.planC is None or not self.planC.dose:
            _show_info(self, "Export dose", "No dose to export.")
            return
        ScanDoseExportDialog(self, "dose").show()

    def export_struct_nii(self):
        if self.planC is None or not self.planC.structure:
            _show_info(self, "Export structures", "No structures to export.")
            return
        StructureExportDialog(self, "nii").show()

    def export_struct_dicom(self):
        if self.planC is None or not self.planC.structure:
            _show_info(self, "Export structures", "No structures to export.")
            return
        StructureExportDialog(self, "dicom").show()

    # ----------------------------------------------------------- loading ----
    def _update_window_title(self):
        """Window title: 'patientName (patientID)' from the first scan when
        available, else the default 'pyCERR Viewer'."""
        name = mrn = ""
        try:
            sInfo = self.planC.scan[0].scanInfo[0]
            name = str(getattr(sInfo, "patientName", "") or "").strip()
            mrn = str(getattr(sInfo, "patientID", "") or "").strip()
        except Exception:  # noqa: BLE001
            pass
        if name and mrn:
            title = f"{name} ({mrn})"
        else:
            title = name or mrn or "pyCERR Viewer"
        self.setWindowTitle(title)

    def after_load(self, keep_view=False):
        """Refresh combo boxes / lists after planC changes."""
        if self.planC is None or not self.planC.scan:
            self.setWindowTitle("pyCERR Viewer")
            return
        self._update_window_title()
        self.maskCache.clear()
        self.overlayCache.clear()
        self.doseCache.clear()
        self._pvStructCache.clear()
        self._pvDoseCache.clear()
        self._slice2dCache.clear()
        self._structSegCache.clear()
        self._upsampleCache.clear()
        prevScan = self.scanNum if keep_view else 0

        self.scanCombo.blockSignals(True)
        self.scanCombo.clear()
        for i, s in enumerate(self.planC.scan):
            mod = getattr(s.scanInfo[0], "imageType", "scan")
            self.scanCombo.addItem(f"{i}: {mod}")
        self.scanCombo.setCurrentIndex(min(prevScan, len(self.planC.scan) - 1))
        self.scanCombo.blockSignals(False)
        self._mirror_scan_combo()    # keep the Scan Display dialog combo in step

        self.doseCombo.blockSignals(True)
        self.doseCombo.clear()
        self.doseCombo.addItem("None")
        for i, d in enumerate(self.planC.dose):
            self.doseCombo.addItem(f"{i}: {getattr(d, 'fractionGroupID', 'dose')}")
        if self.planC.dose and not keep_view:
            self.doseCombo.setCurrentIndex(1)
        self.doseCombo.blockSignals(False)

        self.scanNum = self.scanCombo.currentIndex()
        self.doseNum = self.doseCombo.currentIndex() - 1
        self._load_scan_geometry(reset_slices=not keep_view)
        self._populate_struct_list()
        self._populate_overlay_rows()
        self._build_dose_interp()
        self.refresh_views()

    def _slice_count(self, orientation):
        nR, nC, nS = self.scan3M.shape
        return {VIEW_AXIAL: nS, VIEW_SAGITTAL: nC, VIEW_CORONAL: nR}[orientation]

    def _load_scan_geometry(self, reset_slices=True):
        scanObj = self.planC.scan[self.scanNum]
        self.scan3M = scanObj.getScanArray().astype(np.float32)
        self.xV, self.yV, self.zV = scanObj.getScanXYZVals()
        nR, nC, nS = self.scan3M.shape
        if reset_slices:
            self.lastSlice = {VIEW_AXIAL: nS // 2,
                              VIEW_SAGITTAL: nC // 2,
                              VIEW_CORONAL: nR // 2}
            for wid, v in self.views.items():
                if not v.is3d:
                    self.slices[wid] = self.lastSlice[v.orientation]
                v.user_limits = None    # new geometry -> drop pan/zoom
                v._pan = None
                v._zoom_drag = None
        for orient in (VIEW_AXIAL, VIEW_SAGITTAL, VIEW_CORONAL):
            self.lastSlice[orient] = min(self.lastSlice[orient],
                                         self._slice_count(orient) - 1)
        for wid, v in self.views.items():
            if v.is3d:
                continue
            n = self._slice_count(v.orientation)
            self.slices[wid] = min(self.slices[wid], n - 1)
            v.set_range(n, self.slices[wid])

        # restore this scan's saved window; auto-window non-CT on first view
        if self.scanNum in self.wlByScan:
            self.windowCenter, self.windowWidth = self.wlByScan[self.scanNum]
        else:
            mod = str(getattr(scanObj.scanInfo[0], "imageType", "")).upper()
            if "CT" not in mod:
                lo, hi = np.percentile(self.scan3M, [2, 98])
                self.windowCenter, self.windowWidth = \
                    (lo + hi) / 2, max(hi - lo, 1)
            self.wlByScan[self.scanNum] = (self.windowCenter, self.windowWidth)
        for sp, val in ((self.centerSpin, self.windowCenter),
                        (self.widthSpin, self.windowWidth)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)

        # restore this scan's saved colormap & opacity (default gray / 1.0)
        self.scanCmap, self.scanAlpha = self.dispByScan.setdefault(
            self.scanNum, ("gray", 1.0))
        self.scanCmapCombo.blockSignals(True)
        self.scanCmapCombo.setCurrentText(self.scanCmap)
        self.scanCmapCombo.blockSignals(False)
        self.scanAlphaSlider.blockSignals(True)
        self.scanAlphaSlider.setValue(int(round(self.scanAlpha * 100)))
        self.scanAlphaSlider.blockSignals(False)

        # seed the base-scan colorbar: robust data range for the axis, current
        # window as the colormap range, current colormap.
        lo, hi = np.percentile(self.scan3M, [0.5, 99.5])
        self._scanDataRange = (float(lo), float(hi))
        if getattr(self, "scanColorbar", None) is not None:
            self.scanColorbar.setScan(self.windowCenter, self.windowWidth,
                                      lo, hi)
            self.scanColorbar._set_cmap(self.scanCmap)
            self.scanColorbar.setVisible(True)
            self.scanColorbar.update()
        self._sync_dlg_scan_index()

    def _populate_struct_list(self):
        self.structList.blockSignals(True)
        self.structList.clear()
        curUID = None
        if getattr(self, "structScanFilterChk", None) is not None \
                and self.structScanFilterChk.isChecked() \
                and 0 <= self.scanNum < len(self.planC.scan):
            curUID = self.planC.scan[self.scanNum].scanUID
        for i, st in enumerate(self.planC.structure):
            if curUID is not None and st.assocScanUID != curUID:
                continue            # filtered: not on the current scan
            item = QtWidgets.QListWidgetItem(f"{i}: {st.structureName}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, i)
            col = np.asarray(st.structureColor, dtype=float).ravel()
            rgb = (col / 255.0 if col.size == 3 and col.max() > 1
                   else col if col.size == 3 else np.array([1, 0, 0]))
            item.setForeground(QtGui.QColor.fromRgbF(*np.clip(rgb, 0, 1)))
            self.structList.addItem(item)
        self.structList.blockSignals(False)

    def _on_struct_scan_filter(self, *_):
        """Re-list structures for the current scan-filter setting."""
        if self.planC is not None and self.planC.structure:
            self._populate_struct_list()
            self.refresh_views()

    def _set_all_structs(self, on):
        self.structList.blockSignals(True)
        for i in range(self.structList.count()):
            self.structList.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.structList.blockSignals(False)
        self.refresh_views()

    # ------------------------------------------------------ scan overlays ---
    def _populate_overlay_rows(self):
        """One row per non-base scan: a visibility checkbox plus a shortcut to
        the Scan Display dialog. Colormap, opacity and window come from the
        per-scan settings there (dispByScan / wlByScan)."""
        while self.overlayLayout.count() > 1:      # keep trailing stretch
            w = self.overlayLayout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        nScans = len(self.planC.scan) if self.planC else 0
        self.grpOverlay.setVisible(nScans > 1)
        if nScans < 2:
            return
        for i, s in enumerate(self.planC.scan):
            if i == self.scanNum:
                continue
            st = self.overlayState.setdefault(i, {"on": False})
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(3)
            mod = getattr(s.scanInfo[0], "imageType", "scan")
            cb = QtWidgets.QCheckBox(f"{i}: {mod}")
            cb.setChecked(st["on"])
            cb.setToolTip("Overlay this scan on the base scan (colormap / "
                          "opacity / window from Scan display)")
            cb.toggled.connect(
                lambda on, k=i: self._on_overlay_changed(k, "on", on))
            btn = QtWidgets.QToolButton()
            btn.setText("...")
            btn.setToolTip(
                "Set this scan's colormap, opacity and window (Scan Display)")
            btn.clicked.connect(
                lambda _=False, k=i: self._open_scan_display_for(k))
            rl.addWidget(cb, 1)
            rl.addWidget(btn)
            self.overlayLayout.insertWidget(self.overlayLayout.count() - 1, row)

    def _on_overlay_changed(self, scanIdx, key, val):
        self.overlayState.setdefault(scanIdx, {"on": False})[key] = val
        if key == "on" and val:
            # first-time overlays get a fusion-friendly default display
            self.dispByScan.setdefault(scanIdx, ("hot", 0.5))
        self.refresh_views()

    def _scan_display(self, scanIdx):
        """Per-scan (cmapName, alpha) from the Scan Display settings."""
        return self.dispByScan.get(
            scanIdx, ("gray", 1.0) if scanIdx == self.scanNum else ("hot", 0.5))

    def _overlay_interp(self, scanIdx):
        """Interpolator + auto 2-98 percentile window for an overlay scan."""
        if scanIdx not in self.overlayCache:
            try:
                sObj = self.planC.scan[scanIdx]
                a3M = sObj.getScanArray().astype(np.float32)
                xs, ys, zs = sObj.getScanXYZVals()
                ys, a3M = ascending(ys, a3M, axis=0)
                xs, a3M = ascending(xs, a3M, axis=1)
                zs, a3M = ascending(zs, a3M, axis=2)
                lo, hi = np.percentile(a3M, [2, 98])
                interp = RegularGridInterpolator(
                    (ys, xs, zs), a3M, bounds_error=False, fill_value=np.nan)
                self.overlayCache[scanIdx] = (interp, float(lo),
                                              float(max(hi, lo + 1e-3)))
            except Exception:  # noqa: BLE001
                self.overlayCache[scanIdx] = None
        return self.overlayCache[scanIdx]

    def _dose_interp(self, doseIdx):
        """Cached (interpolator, doseMax) for a dose index, or None."""
        if self.planC is None or doseIdx is None or doseIdx < 0 \
                or doseIdx >= len(self.planC.dose):
            return None
        if doseIdx not in self.doseCache:
            try:
                d = self.planC.dose[doseIdx]
                dose3M = np.asarray(d.doseArray, dtype=np.float32)
                xD, yD, zD = d.getDoseXYZVals()
                yD, dose3M = ascending(yD, dose3M, axis=0)
                xD, dose3M = ascending(xD, dose3M, axis=1)
                zD, dose3M = ascending(zD, dose3M, axis=2)
                interp = RegularGridInterpolator(
                    (yD, xD, zD), dose3M, bounds_error=False, fill_value=0.0)
                self.doseCache[doseIdx] = (interp, float(dose3M.max()))
            except Exception:  # noqa: BLE001
                self.doseCache[doseIdx] = None
        return self.doseCache[doseIdx]

    def _build_dose_interp(self):
        """Point the colorbar/status readout at the globally selected dose."""
        res = self._dose_interp(self.doseNum)
        if res is None:
            self.doseInterp = None
            self.doseMax = 0.0
            self.colorbar.setVisible(False)
            return
        self.doseInterp, self.doseMax = res
        self.colorbar.setDose(self.doseMax)
        self.colorbar.setVisible(True)
        self._update_isodose_ticks()

    # ----------------------------------------------------------- callbacks --
    def on_scan_changed(self, idx):
        if self.planC is None or idx < 0:
            return
        self.scanNum = idx
        self.maskCache.clear()
        self._pvStructCache.clear()   # surfaces live on the scan grid
        self._load_scan_geometry()
        if getattr(self, "structScanFilterChk", None) is not None \
                and self.structScanFilterChk.isChecked():  # filter follows scan
            self._populate_struct_list()
        self._populate_overlay_rows()   # base scan is excluded from overlays
        if self.regCtl is not None:     # keep QA base in sync with the scan
            self.regCtl.sync_base(idx)
        self.refresh_views()

    def on_dose_changed(self, idx):
        self.doseNum = idx - 1
        self._build_dose_interp()
        self.refresh_views()

    def on_alpha(self, val):
        self.doseAlpha = val / 100.0
        self._fast_layer_style("dose")

    def _fast_layer_style(self, layer):
        """Reapply a layer's colormap, window (clim) and opacity in place on the
        persistent 2D images (cheap ``set_cmap``/``set_clim``/``set_alpha`` +
        canvas redraw), and update the 3D views in place too. Falls back to a
        per-window refresh where an in-place update isn't possible (isodose
        mode, per-axis scan override, or an unhandled 3D actor) - so changing
        the colormap/window/opacity no longer re-slices the whole scene.

        The display (cyan) mask range is NOT handled here; callers that change
        it must do a full refresh so the slice is re-masked."""
        if self.planC is None or not self.planC.scan:
            return
        if layer == "scan":
            vmin = self.windowCenter - self.windowWidth / 2.0
            vmax = self.windowCenter + self.windowWidth / 2.0
            cmap, clim, alpha = self.scanCmap, (vmin, vmax), self.scanAlpha
        else:
            cbLo, cbHi = self.colorbar.cbarRange
            cmap = self.colorbar.mplCmap
            clim, alpha = (cbLo, cbHi), self.doseAlpha
        clim = (clim[0], max(clim[1], clim[0] + 1e-6))
        for winId in self.activeWins:
            view = self.views[winId]
            if view.is3d:
                if not self._fast_3d_style(view, layer, cmap, clim, alpha):
                    self.refresh_views(only=winId)
                continue
            if layer == "scan":
                if self._axis_scan(winId) != self.scanNum:
                    continue                 # base change doesn't touch overrides
                im = getattr(view, "_scanIm", None)
                if im is None:
                    self.refresh_views(only=winId)
                    continue
                # re-mask the slice in place so the display (cyan) range updates
                # without a full re-render; the raw slice + upsample are cheap
                # (a numpy view + a cached resample).
                img, _extent, hV, vV, _slicer = self._slice_data(winId)
                img = self._upsample_for_display(
                    img, hV, vV,
                    ("base", view.orientation, self.slices[winId],
                     self.scanNum))
                im.set_data(self._apply_scan_dispmask(img))
            else:
                im = getattr(view, "_doseIm", None)
                if im is None:
                    self.refresh_views(only=winId)  # isodose / no reusable image
                    continue
                # re-mask the dose colorwash in place so the display (cyan)
                # range updates without a full re-render (cached dose slice).
                doseIdx = self._axis_dose(winId)
                doseRes = self._dose_interp(doseIdx)
                if doseRes is None:
                    self.refresh_views(only=winId)
                    continue
                doseInterp = doseRes[0]
                _img, _extent, hV, vV, _slicer = self._slice_data(winId)
                doseSlc = self._resample_slice2d("dose", doseIdx, doseInterp,
                                                 winId, hV, vV)
                dLo, dHi = self.colorbar.dispRange
                im.set_data(np.ma.masked_where(
                    (doseSlc < max(dLo, 1e-3)) | (doseSlc > dHi), doseSlc))
            im.set_cmap(cmap)
            im.set_clim(*clim)
            im.set_alpha(alpha)
            view.canvas.draw_idle()
        self._notify_volume3d_style(layer)

    def _fast_3d_style(self, view, layer, cmap, clim, alpha):
        """In-place colormap/window/opacity update for the embedded VTK
        cut-planes view. Returns False (caller re-renders the window) when an
        in-place update is not possible - including when a display (cyan) range
        is active, since the scan NaN mask / dose iso-levels then depend on it
        and must be rebuilt by a re-render."""
        if not getattr(view, "uses_vtk", False):
            return False
        dispActive = (self._scan_disp_range() if layer == "scan"
                      else self._dose_disp_range()) != (float("-inf"),
                                                        float("inf"))
        if dispActive:
            return False
        pl = getattr(view, "vtk_widget", None)
        if pl is None:
            return False
        try:
            if layer == "scan":
                actors = list(getattr(view, "_plane_actors", {}).values())
                op = float(alpha)
            else:
                doseIdx = self._axis_dose(view.winId)
                act = pl.actors.get("isodose%d" % doseIdx) \
                    if doseIdx is not None and doseIdx >= 0 else None
                actors = [act] if act is not None else []
                op = min(max(float(alpha), 0.0), 0.6)
            if not actors:
                return False
            lut = pv.LookupTable(cmap=cmap)
            lut.scalar_range = clim
            for act in actors:
                act.mapper.lookup_table = lut
                act.mapper.scalar_range = clim
                act.prop.opacity = op
            pl.render()
            return True
        except Exception:  # noqa: BLE001
            return False

    def _notify_volume3d_style(self, layer):
        """Ask the 3D volume dialog to restyle in place (colormap/window/
        opacity) without a full rebuild; falls back to its debounced rebuild.

        The scan volume honors the display range via its opacity transfer
        function (in place), but dose iso-surfaces at a masked display range
        change *geometry* (fewer levels), so a rebuild is requested instead."""
        dlg = getattr(self, "_volume3dDialog", None)
        if dlg is None:
            return
        try:
            if layer == "dose" and self._dose_disp_range() != (
                    float("-inf"), float("inf")):
                dlg.request_refresh()
            else:
                dlg.apply_style(layer)
        except Exception:  # noqa: BLE001
            self._volume3dDialog = None

    # ------------------------------------------------------ isodose options --
    def _update_isodose_enabled(self):
        iso = self.doseDispMode == "isodose"
        for w in (self.isoLevelsEdit, self.isoUnitsCombo, self.isoWidthSpin):
            w.setEnabled(iso)

    def _update_isodose_ticks(self):
        """Show the selected isodose levels as tick marks on the dose colorbar
        (isodose mode only)."""
        cb = getattr(self, "colorbar", None)
        if cb is None:
            return
        if self.doseDispMode == "isodose" and self.doseNum is not None \
                and self.doseNum >= 0 and getattr(self, "doseMax", 0) > 0:
            cb.setMarkers(self._isodose_abs_levels(self.doseNum, self.doseMax))
        else:
            cb.setMarkers([])

    def on_dose_disp_mode(self, idx):
        self.doseDispMode = "isodose" if idx == 1 else "colorwash"
        self._update_isodose_enabled()
        self._update_isodose_ticks()
        self.refresh_views()

    def on_isodose_levels(self):
        txt = self.isoLevelsEdit.text().replace(";", ",")
        try:
            levels = sorted({float(t) for t in txt.split(",") if t.strip()})
        except ValueError:
            self.statusBar().showMessage(
                "Isodose levels: enter comma-separated numbers.")
            return
        if levels != self.isodoseLevels:
            self.isodoseLevels = levels
            self._update_isodose_ticks()
            # 2D isodose lines and the 3D/cut-plane isodose surfaces both follow
            # these levels, so refresh regardless of the 2D display mode.
            self.refresh_views()

    def on_isodose_units(self, units):
        self.isodoseUnits = units
        self._update_isodose_ticks()
        self.refresh_views()

    def on_isodose_width(self, val):
        self.isodoseWidth = float(val)
        if self.doseDispMode == "isodose":
            self.refresh_views()

    def _isodose_abs_levels(self, doseIdx, doseMax):
        """Isodose levels in absolute dose units (Gy), resolving the selected
        units: absolute, % of max dose, or % of the prescription dose (falls
        back to % of max when no prescription is stored)."""
        if self.isodoseUnits == "Gy":
            return sorted(self.isodoseLevels)
        base = doseMax
        if self.isodoseUnits == "% of Rx":
            try:
                rx = float(getattr(self.planC.dose[doseIdx],
                                   "prescriptionDose", 0) or 0)
            except Exception:  # noqa: BLE001
                rx = 0.0
            if rx > 0:
                base = rx
        return sorted(lev * base / 100.0 for lev in self.isodoseLevels)

    def _draw_isodose_lines(self, ax, doseSlc, hV, vV, doseIdx, doseMax):
        """CERR-style isodose contour lines, colored by the dose colorbar's
        colormap mapping (so line colors match the colorwash/colorbar)."""
        levels = [lv for lv in self._isodose_abs_levels(doseIdx, doseMax)
                  if lv > 0]
        if not levels:
            return
        cbLo, cbHi = self.colorbar.cbarRange
        rng = max(cbHi - cbLo, 1e-6)
        cmap = self.colorbar.mplCmap
        if HAS_CONTOURPY:
            gen = contourpy.contour_generator(
                x=hV, y=vV, z=doseSlc,
                line_type=contourpy.LineType.Separate)
            for lev in levels:
                segs = gen.lines(lev)
                if not segs:
                    continue
                color = cmap(np.clip((lev - cbLo) / rng, 0.0, 1.0))
                ax.add_collection(LineCollection(
                    segs, colors=[color], linewidths=self.isodoseWidth,
                    zorder=2.5))
        else:      # fallback: matplotlib's contour
            colors = [cmap(np.clip((lev - cbLo) / rng, 0.0, 1.0))
                      for lev in levels]
            ax.contour(hV, vV, doseSlc, levels=levels, colors=colors,
                       linewidths=self.isodoseWidth, zorder=2.5)

    def on_plane_opacity(self, val):
        # opacity of the 3D urOMT result overlay (driven from the urOMT dialog)
        self.plane3dOpacity = val / 100.0
        self._refresh_3d_views()

    def on_dose_cmap(self, name):
        """Set the dose colorwash colormap from the panel combo."""
        if not name:
            return
        self.colorbar._set_cmap(name)
        self.colorbar.update()        # repaint the colorbar gradient
        self._fast_layer_style("dose")

    def _on_colorbar_ranges_changed(self):
        # keep the panel combo in sync when the colormap (or ranges) is changed
        # via the colorbar's right-click menu, then re-render.
        if hasattr(self, "doseCmapCombo") and \
                self.doseCmapCombo.currentText() != self.colorbar.cmapName:
            self.doseCmapCombo.blockSignals(True)
            self.doseCmapCombo.setCurrentText(self.colorbar.cmapName)
            self.doseCmapCombo.blockSignals(False)
        # Colormap, mapping-range (yellow) and display-mask (cyan) all update
        # in place: _fast_layer_style re-masks the 2D colorwash and restyles 3D
        # without a rebuild.
        self._fast_layer_style("dose")

    def _set_scan_display(self, t, cmap=None, alpha=None):
        """Store a colormap / opacity for scan ``t``; mirrors to the live
        base-scan state when ``t`` is the base."""
        cur = list(self._scan_display(t))
        if cmap is not None:
            cur[0] = cmap
        if alpha is not None:
            cur[1] = alpha
        self.dispByScan[t] = tuple(cur)
        if t == self.scanNum:
            self.scanCmap, self.scanAlpha = cur

    def _set_scan_window(self, t, center, width):
        """Store a window for scan ``t``; mirrors to the live base window when
        ``t`` is the base."""
        self.wlByScan[t] = (center, width)
        if t == self.scanNum:
            self.windowCenter, self.windowWidth = center, width

    def _on_scan_colorbar_changed(self):
        """React to the scan colorbar: yellow (colormap) range -> window,
        colormap-menu choice -> panel combo, cyan range -> re-mask the scan.
        Applies to the scan selected in the Scan Display dialog."""
        cb = self.scanColorbar
        t = self._dlg_target()
        # colormap picked from the colorbar's right-click menu
        if hasattr(self, "scanCmapCombo") and \
                self.scanCmapCombo.currentText() != cb.cmapName:
            self._set_scan_display(t, cmap=cb.cmapName)
            self.scanCmapCombo.blockSignals(True)
            self.scanCmapCombo.setCurrentText(cb.cmapName)
            self.scanCmapCombo.blockSignals(False)
        # colormap-mapping range (yellow handles) -> window center/width
        lo, hi = cb.cbarRange
        center, width = (lo + hi) / 2.0, max(hi - lo, 1e-6)
        wl0 = self.wlByScan.get(t, (None, None))
        if wl0[0] is None or abs(center - wl0[0]) > 1e-6 \
                or abs(width - wl0[1]) > 1e-6:
            self._set_scan_window(t, center, width)
            for sp, val in ((self.centerSpin, center), (self.widthSpin, width)):
                sp.blockSignals(True)
                sp.setValue(val)
                sp.blockSignals(False)
            self.presetCombo.blockSignals(True)
            self.presetCombo.setCurrentText("--- Manual ---")
            self.presetCombo.blockSignals(False)
        # Window (yellow), colormap and display-mask (cyan) all update in place
        # for the base scan: _fast_layer_style re-masks the 2D slice and
        # restyles the 3D views without a rebuild (the display mask doesn't
        # affect 3D). Overlay targets still take the full-refresh path.
        if t == self.scanNum:
            self._fast_layer_style("scan")
        else:
            self.refresh_views()

    def _sync_scan_colorbar_window(self):
        """Push the dialog target scan's window & colormap into the scan
        colorbar without disturbing the user's display (cyan) range."""
        cb = getattr(self, "scanColorbar", None)
        if cb is None:
            return
        t = self._dlg_target()
        center, width = self.wlByScan.get(
            t, (self.windowCenter, self.windowWidth))
        cb.setWindow(center, width)
        cmap = self._scan_display(t)[0]
        if cb.cmapName != cmap:
            cb._set_cmap(cmap)
        cb.update()

    def on_scan_cmap(self, name):
        tgt = self._dlg_target()
        self._set_scan_display(tgt, cmap=name)
        self._sync_scan_colorbar_window()
        if tgt == self.scanNum:
            self._fast_layer_style("scan")
        else:
            self.refresh_views()

    def on_scan_alpha(self, val):
        tgt = self._dlg_target()
        self._set_scan_display(tgt, alpha=val / 100.0)
        if tgt == self.scanNum:
            self._fast_layer_style("scan")   # base-scan image: update in place
        else:
            self.refresh_views()             # overlay opacity: overlays redrawn

    def on_preset(self, name):
        preset = CT_WINDOW_PRESETS.get(name)
        if preset is None:
            return
        self._set_scan_window(self._dlg_target(), *preset)
        self.centerSpin.blockSignals(True)
        self.widthSpin.blockSignals(True)
        self.centerSpin.setValue(preset[0])
        self.widthSpin.setValue(preset[1])
        self.centerSpin.blockSignals(False)
        self.widthSpin.blockSignals(False)
        self._sync_scan_colorbar_window()
        if self._dlg_target() == self.scanNum:
            self._fast_layer_style("scan")
        else:
            self.refresh_views()

    def on_manual_wl(self, *_):
        tgt = self._dlg_target()
        self._set_scan_window(tgt, self.centerSpin.value(),
                              self.widthSpin.value())
        self.presetCombo.blockSignals(True)
        self.presetCombo.setCurrentText("--- Manual ---")
        self.presetCombo.blockSignals(False)
        self._sync_scan_colorbar_window()
        if tgt == self.scanNum:
            self._fast_layer_style("scan")
        else:
            self.refresh_views()

    def on_slice_changed(self, winId, val):
        orientation = self.views[winId].orientation
        self.slices[winId] = val
        self.lastSlice[orientation] = val
        refreshed = [winId]
        if self.lockViews:   # follow with every other window of the same view
            for wid in self.activeWins:
                v = self.views[wid]
                if wid != winId and v.orientation == orientation \
                        and self.slices[wid] != val:
                    self.slices[wid] = val
                    v.set_range(v.slider.maximum() + 1, val)
                    refreshed.append(wid)
        for wid in refreshed:
            self.refresh_views(only=wid)
        # other views' crosshairs depend on this slice -> reposition (blitted:
        # repaints just the two lines instead of redrawing the whole view)
        for wid in self.activeWins:
            if wid not in refreshed and not self.views[wid].is3d:
                self._position_crosshair(self.views[wid])
                self.views[wid].blit_crosshair()
        # re-render 3D plane views shortly after scrolling settles
        if any(self.views[w].is3d for w in self.activeWins):
            self._timer3d.start(150)

    def on_lock_toggled(self, on):
        self.lockViews = on
        if not on or self.planC is None or not self.planC.scan:
            return
        # align all windows to the last-touched slice of their view type
        for wid, v in self.views.items():
            if v.is3d:
                continue
            k = self.lastSlice[v.orientation]
            if self.slices[wid] != k:
                self.slices[wid] = k
                v.set_range(v.slider.maximum() + 1, k)
        self.refresh_views()

    def on_crosshair_toggled(self, on):
        self.showCrosshairs = on
        for view in self.views.values():
            for ln in (view.xline, view.yline):
                if ln is not None:
                    ln.set_visible(on)
            view.canvas.draw_idle()

    def on_orientation_toggled(self, on):
        self.showOrientation = on
        self.refresh_views()

    def on_upsample_toggled(self, on):
        self.upsampleDisplay = bool(on)
        self._upsampleCache.clear()
        self.refresh_views()

    def reset_all_views(self):
        for view in self.views.values():
            view.user_limits = None
            view._pan = None
            view._zoom_drag = None
        self.refresh_views()

    # ----------------------------------------------------------- layouts ----
    def set_layout(self, name):
        """Switch the arrangement of view windows (View > Layout)."""
        self._apply_layout(name)
        # Keep the contouring tool working across the layout change instead of
        # blocking it: re-bind to the new layout's axial view.
        if self.contourCtl is not None and self.contourCtl.isVisible():
            self.contourCtl.rebind_after_layout()

    def _apply_layout(self, name):
        prevActive = set(self.activeWins)
        # detach all views so the old splitters can be deleted safely
        for v in self.views.values():
            v.setParent(None)
        while self.viewLay.count():
            item = self.viewLay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        V = self.views
        if name == "single":
            self.activeWins = ["A"]
            self.viewLay.addWidget(V["A"])
        elif name == "two":
            # equal-stretch halves: always the same size
            self.activeWins = ["A", "B"]
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(2)
            for wid in self.activeWins:
                rl.addWidget(V[wid], 1)
            self.viewLay.addWidget(row)
        elif name == "columns":
            # equal-stretch columns: always the same size
            self.activeWins = ["A", "B", "C"]
            row = QtWidgets.QWidget()
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(2)
            for wid in self.activeWins:
                rl.addWidget(V[wid], 1)
            self.viewLay.addWidget(row)
        elif name == "grid":
            # equal-stretch 2x2 grid: four same-size quadrants
            self.activeWins = ["A", "B", "C", "D"]
            gw = QtWidgets.QWidget()
            gl = QtWidgets.QGridLayout(gw)
            gl.setContentsMargins(0, 0, 0, 0)
            gl.setSpacing(2)
            gl.addWidget(V["A"], 0, 0)
            gl.addWidget(V["B"], 0, 1)
            gl.addWidget(V["C"], 1, 0)
            gl.addWidget(V["D"], 1, 1)
            for i in (0, 1):
                gl.setRowStretch(i, 1)
                gl.setColumnStretch(i, 1)
            self.viewLay.addWidget(gw)
        else:   # "default": one large + two stacked
            self.activeWins = ["A", "B", "C"]
            right = QtWidgets.QSplitter(Qt.Vertical)
            right.addWidget(V["B"])
            right.addWidget(V["C"])
            main = QtWidgets.QSplitter(Qt.Horizontal)
            main.addWidget(V["A"])
            main.addWidget(right)
            main.setStretchFactor(0, 3)
            main.setStretchFactor(1, 2)
            self.viewLay.addWidget(main)

        for wid, v in self.views.items():
            v.setVisible(wid in self.activeWins)

        if self.planC is not None and self.planC.scan:
            # newly shown windows join at the last slice of their view type
            for wid in set(self.activeWins) - prevActive:
                if self.views[wid].is3d:
                    continue
                self.slices[wid] = min(
                    self.lastSlice[self.views[wid].orientation],
                    self._slice_count(self.views[wid].orientation) - 1)
            self._load_scan_geometry(reset_slices=False)
            self.refresh_views()

    # ------------------------------------------- per-axis menu (CERR-style) -
    def _reset_axis_sel(self):
        for sel in self.axisSel.values():
            sel["scan"] = sel["dose"] = sel["structs"] = None

    def _axis_scan(self, orient):
        """Effective base scan index for a view (manual override or global)."""
        sel = self.axisSel[orient]["scan"]
        if sel is not None and 0 <= sel < len(self.planC.scan):
            return sel
        return self.scanNum

    def _axis_dose(self, orient):
        """Effective dose index for a view (manual, 'none', or global)."""
        sel = self.axisSel[orient]["dose"]
        if sel is None:
            return self.doseNum
        return sel    # may be -1 = no dose

    def _axis_structs(self, orient):
        """Effective visible structures for a view."""
        sel = self.axisSel[orient]["structs"]
        if sel is None:
            return self._checked_structs()
        return [i for i in sorted(sel) if i < len(self.planC.structure)]

    def _show_axis_menu(self, orient):
        """Right-click menu: choose view/scan/dose/structures for this window
        only (mirrors MATLAB CERR's per-axis View/ScanSet/DoseSet/StructSet
        menu)."""
        if self.planC is None or not self.planC.scan:
            return
        sel = self.axisSel[orient]
        menu = QtWidgets.QMenu(self)

        viewM = menu.addMenu("View")
        vgrp = QtWidgets.QActionGroup(viewM)
        for o in (VIEW_AXIAL, VIEW_SAGITTAL, VIEW_CORONAL, VIEW_3D):
            a = viewM.addAction(VIEW_DISPLAY.get(o, o))
            a.setCheckable(True)
            a.setChecked(self.views[orient].orientation == o)
            vgrp.addAction(a)
            a.triggered.connect(
                lambda _=False, w=orient, oo=o: self._set_axis_view(w, oo))

        if self.views[orient].is3d:
            # slice-plane selections don't apply to the 3D view
            menu.addSeparator()
            menu.addAction(self.actLock)
            menu.exec_(QtGui.QCursor.pos())
            return

        scanM = menu.addMenu("ScanSet")
        grp = QtWidgets.QActionGroup(scanM)
        act = scanM.addAction("Auto (panel selection)")
        act.setCheckable(True)
        act.setChecked(sel["scan"] is None)
        grp.addAction(act)
        act.triggered.connect(
            lambda _=False, o=orient: self._set_axis_scan(o, None))
        scanM.addSeparator()
        for i, s in enumerate(self.planC.scan):
            mod = getattr(s.scanInfo[0], "imageType", "scan")
            a = scanM.addAction(f"{i}: {mod}")
            a.setCheckable(True)
            a.setChecked(sel["scan"] == i)
            grp.addAction(a)
            a.triggered.connect(
                lambda _=False, o=orient, k=i: self._set_axis_scan(o, k))

        doseM = menu.addMenu("DoseSet")
        dgrp = QtWidgets.QActionGroup(doseM)
        for label, val in (("Auto (panel selection)", None), ("None", -1)):
            a = doseM.addAction(label)
            a.setCheckable(True)
            a.setChecked(sel["dose"] is val if val is None
                         else sel["dose"] == val)
            dgrp.addAction(a)
            a.triggered.connect(
                lambda _=False, o=orient, v=val: self._set_axis_dose(o, v))
        doseM.addSeparator()
        for i, d in enumerate(self.planC.dose):
            a = doseM.addAction(
                f"{i}: {getattr(d, 'fractionGroupID', 'dose')}")
            a.setCheckable(True)
            a.setChecked(sel["dose"] == i)
            dgrp.addAction(a)
            a.triggered.connect(
                lambda _=False, o=orient, k=i: self._set_axis_dose(o, k))

        structM = menu.addMenu("StructSet")
        a = structM.addAction("Auto (panel selection)")
        a.setCheckable(True)
        a.setChecked(sel["structs"] is None)
        a.triggered.connect(
            lambda _=False, o=orient: self._toggle_axis_struct(o, None))
        structM.addSeparator()
        shown = set(self._axis_structs(orient))
        for i, st in enumerate(self.planC.structure):
            a = structM.addAction(f"{i}: {st.structureName}")
            a.setCheckable(True)
            a.setChecked(i in shown)
            a.triggered.connect(
                lambda _=False, o=orient, k=i: self._toggle_axis_struct(o, k))

        menu.addSeparator()
        rulerAct = menu.addAction("Draw Ruler")
        rulerAct.setCheckable(True)
        rulerAct.setChecked(self.views[orient].ruler_mode)
        rulerAct.triggered.connect(
            lambda _=False, o=orient: self._toggle_ruler(o))
        menu.addAction(self.actLock)   # lock slices across matching views
        menu.addAction("Reset pan/zoom (this view)",
                       self.views[orient].reset_view)
        menu.exec_(QtGui.QCursor.pos())

    def _set_axis_view(self, winId, orientation):
        """Switch which orientation a view window displays (CERR SET_VIEW)."""
        view = self.views[winId]
        if view.orientation == orientation:
            return
        if self.contourCtl is not None and self.contourCtl.isVisible() \
                and self.contourCtl.axView is view:
            self.statusBar().showMessage(
                "Close the Contouring panel before changing this view.")
            return
        was3d = view.is3d
        view.orientation = orientation
        if orientation == VIEW_3D:
            view.set_projection(True)
            self.refresh_views(only=winId)
            return
        if was3d:
            view.set_projection(False)
        view.clear_ruler()             # in-plane coords no longer apply
        view.user_limits = None
        n = self._slice_count(orientation)
        k = min(self.lastSlice[orientation], n - 1)
        self.slices[winId] = k
        view.set_range(n, k)
        self.refresh_views(only=winId)

    def _set_axis_scan(self, orient, scanIdx):
        self.axisSel[orient]["scan"] = scanIdx
        self.refresh_views(only=orient)

    def _set_axis_dose(self, orient, doseIdx):
        self.axisSel[orient]["dose"] = doseIdx
        self.refresh_views(only=orient)

    def _toggle_ruler(self, orient):
        """CERR's TOGGLERULER: arm ruler mode / clear the ruler and disarm."""
        view = self.views[orient]
        if view.ruler_mode:
            view.ruler_mode = False
            view.clear_ruler()
            view.canvas.setCursor(Qt.ArrowCursor)
            self.statusBar().showMessage("Ruler off.")
        else:
            view.ruler_mode = True
            view.canvas.setCursor(Qt.CrossCursor)
            self.statusBar().showMessage(
                f"Ruler: click and drag in the {view.orientation} view to "
                "measure; right-click > Draw Ruler again to clear.")

    def on_ruler_changed(self, orient):
        view = self.views[orient]
        if view.ruler is None:
            return
        (x0, y0), (x1, y1) = view.ruler
        self.statusBar().showMessage(
            f"({x0:.2f}, {y0:.2f}) to ({x1:.2f}, {y1:.2f})   "
            f"Dist: {view.ruler_length():.3g} cm")

    def _toggle_axis_struct(self, orient, strNum):
        sel = self.axisSel[orient]
        if strNum is None:                      # back to Auto
            sel["structs"] = None
        else:
            # first manual toggle starts from the currently shown set
            cur = set(sel["structs"]) if sel["structs"] is not None \
                else set(self._checked_structs())
            cur.symmetric_difference_update({strNum})
            sel["structs"] = cur
        self.refresh_views(only=orient)

    def on_cursor_moved(self, winId, xd, yd):
        if self.planC is None:
            return
        orientation = self.views[winId].orientation
        k = self.slices[winId]
        if orientation == VIEW_AXIAL:
            x, y, z = xd, yd, self.zV[k]
        elif orientation == VIEW_SAGITTAL:
            x, y, z = self.xV[k], xd, yd
        else:
            x, y, z = xd, self.yV[k], yd
        msg = f"x={x:.2f} cm  y={y:.2f} cm  z={z:.2f} cm"
        # scan value (of the scan shown in this view)
        try:
            baseIdx = self._axis_scan(winId)
            if baseIdx == self.scanNum:
                r = int(np.argmin(np.abs(self.yV - y)))
                c = int(np.argmin(np.abs(self.xV - x)))
                s = int(np.argmin(np.abs(self.zV - z)))
                msg += f"   scan={self.scan3M[r, c, s]:.1f}"
            else:
                res = self._overlay_interp(baseIdx)
                if res is not None:
                    msg += f"   scan={float(res[0]((y, x, z))):.1f}"
        except Exception:  # noqa: BLE001
            pass
        doseRes = self._dose_interp(self._axis_dose(winId))
        if doseRes is not None:
            try:
                dv = float(doseRes[0]((y, x, z)))
                msg += f"   dose={dv:.2f}"
            except Exception:  # noqa: BLE001
                pass
        self.statusBar().showMessage(msg)

    # ----------------------------------------------------------- rendering --
    def _checked_structs(self):
        out = []
        for i in range(self.structList.count()):
            it = self.structList.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _struct_mask(self, strNum):
        if strNum not in self.maskCache:
            try:
                self.maskCache[strNum] = rs.getStrMask(strNum, self.planC)
            except Exception:  # noqa: BLE001
                self.maskCache[strNum] = None
        return self.maskCache[strNum]

    def _struct_color(self, strNum):
        col = np.asarray(self.planC.structure[strNum].structureColor,
                         dtype=float).ravel()
        if col.size != 3:
            return (1.0, 0.0, 0.0)
        if col.max() > 1:
            col = col / 255.0
        return tuple(np.clip(col, 0, 1))

    @staticmethod
    def _draw_contour_dots(ax, segs, color):
        """Plot Alaly-style vertex dots along a structure's contour segments
        (a list of (N,2) vertex arrays). Dots are drawn white on dark contour
        colors and black on light ones (sum of the contour RGB < 1.5 -> white)
        so they stay visible."""
        dotColor = "white" if sum(color[:3]) < 1.5 else "black"
        for seg in segs:
            if len(seg) == 0:
                continue
            # small, fairly fine dots: ~every 3rd vertex, capped at ~150/contour
            step = max(len(seg) // 150, 3)
            pts = seg[::step]
            ax.plot(pts[:, 0], pts[:, 1], linestyle="none", marker="o",
                    markersize=1.0, markerfacecolor=dotColor,
                    markeredgecolor=dotColor, markeredgewidth=0.0, zorder=12)

    # --------------------------------------------------- structure options --
    def on_struct_dots(self, on):
        self.showStructDots = on
        self.refresh_views()

    def on_struct_linewidth(self, val):
        self.structLineWidth = float(val)
        self.refresh_views()

    def on_struct_double_click(self, item):
        """Center all views on the double-clicked structure's centroid."""
        strNum = item.data(Qt.UserRole)
        if strNum is None:
            return
        self.goto_struct_center(strNum)

    def _on_struct_context_menu(self, pos):
        """Right-click menu on the structure list: color, center, rename,
        delete."""
        item = self.structList.itemAt(pos)
        if item is None:
            return
        strNum = item.data(Qt.UserRole)
        if strNum is None:
            return
        menu = QtWidgets.QMenu(self.structList)
        actColor = menu.addAction("Select color…")
        actCenter = menu.addAction("Go to center")
        actRename = menu.addAction("Rename…")
        menu.addSeparator()
        actDelete = menu.addAction("Delete")
        chosen = menu.exec_(self.structList.viewport().mapToGlobal(pos))
        if chosen is actColor:
            self._pick_struct_color(strNum, item)
        elif chosen is actCenter:
            self.goto_struct_center(strNum)
        elif chosen is actRename:
            self._rename_struct(strNum, item)
        elif chosen is actDelete:
            self._delete_struct(strNum)

    def _rename_struct(self, strNum, item=None):
        """Prompt for a new name and apply it to the structure."""
        if self.planC is None or not (0 <= strNum < len(self.planC.structure)):
            return
        cur = self.planC.structure[strNum].structureName
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename structure", "Structure name:",
            QtWidgets.QLineEdit.Normal, cur)
        if not ok:
            return
        name = name.strip()
        if not name or name == cur:
            return
        self.planC.structure[strNum].structureName = name
        if item is not None:
            item.setText(f"{strNum}: {name}")
        self.refresh_views()

    def _delete_struct(self, strNum):
        """Remove a structure from planC after confirmation.

        Deleting shifts every later structure's index, so all structure-indexed
        caches are cleared and the list is rebuilt."""
        if self.planC is None or not (0 <= strNum < len(self.planC.structure)):
            return
        # An active contour edit is bound to a structure index that this delete
        # would invalidate; ask the user to close it first.
        ctl = self.contourCtl
        if ctl is not None and ctl.isVisible():
            self.statusBar().showMessage(
                "Close the contouring tool before deleting a structure.")
            return
        name = self.planC.structure[strNum].structureName
        if QtWidgets.QMessageBox.question(
                self, "Delete structure",
                f"Delete structure '{strNum}: {name}'?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes:
            return
        del self.planC.structure[strNum]
        # Structure indices shifted: drop all structure-indexed caches.
        self.maskCache.clear()
        self._pvStructCache.clear()
        self._structSegCache.clear()
        self._populate_struct_list()
        self.refresh_views()

    def _pick_struct_color(self, strNum, item=None):
        """Open a color-wheel dialog and apply the chosen color to a structure.

        Colors are stored on the structure as 0-255 integer RGB triplets (the
        convention used throughout planC.structure)."""
        if self.planC is None or not (0 <= strNum < len(self.planC.structure)):
            return
        r, g, b = (int(round(c * 255)) for c in self._struct_color(strNum))
        initial = QtGui.QColor(r, g, b)
        col = QtWidgets.QColorDialog.getColor(
            initial, self, "Select structure color")
        if not col.isValid():
            return
        self.planC.structure[strNum].structureColor = \
            [col.red(), col.green(), col.blue()]
        # Recolor the list-item label to match.
        if item is not None:
            item.setForeground(QtGui.QColor.fromRgbF(
                *np.clip([col.redF(), col.greenF(), col.blueF()], 0, 1)))
        self.refresh_views()

    def goto_struct_center(self, strNum):
        """Navigate the three orthogonal views to the structure's center of
        mass (its isocenter slice in each plane)."""
        mask = self._struct_mask(strNum)
        if mask is None or mask.shape != self.scan3M.shape or not mask.any():
            self.statusBar().showMessage(
                "Structure has no voxels on the current scan grid.")
            return
        rows, cols, slcs = np.where(mask)       # (y, x, z)
        centers = {VIEW_AXIAL: int(round(slcs.mean())),
                   VIEW_SAGITTAL: int(round(cols.mean())),
                   VIEW_CORONAL: int(round(rows.mean()))}
        for orient, k in centers.items():
            n = self._slice_count(orient)
            self.lastSlice[orient] = int(np.clip(k, 0, n - 1))
        for wid, v in self.views.items():
            if v.is3d:
                continue
            k = self.lastSlice[v.orientation]
            self.slices[wid] = k
            v.set_range(self._slice_count(v.orientation), k)
        self.refresh_views()
        name = self.planC.structure[strNum].structureName
        self.statusBar().showMessage(
            f"Centered views on '{name}' (axial {centers[VIEW_AXIAL] + 1}, "
            f"sagittal {centers[VIEW_SAGITTAL] + 1}, "
            f"coronal {centers[VIEW_CORONAL] + 1}).")

    def _slice_data(self, winId):
        """Return (img2D, extent, aspect, (hAxisVals, vAxisVals), maskSlicer)."""
        orientation = self.views[winId].orientation
        k = self.slices[winId]
        if orientation == VIEW_AXIAL:
            img = self.scan3M[:, :, k]
            hV, vV = self.xV, self.yV
            slicer = lambda m: m[:, :, k]               # noqa: E731
        elif orientation == VIEW_SAGITTAL:
            img = self.scan3M[:, k, :].T                # rows=z, cols=y
            hV, vV = self.yV, self.zV
            slicer = lambda m: m[:, k, :].T             # noqa: E731
        else:  # coronal
            img = self.scan3M[k, :, :].T                # rows=z, cols=x
            hV, vV = self.xV, self.zV
            slicer = lambda m: m[k, :, :].T             # noqa: E731
        extent = [hV[0], hV[-1], vV[-1], vV[0]]
        return img, extent, hV, vV, slicer

    def _apply_scan_dispmask(self, img):
        """Hide base-scan intensities outside the scan colorbar's display
        (cyan) range. When that range spans the full axis (the default) the
        array is returned unchanged so the fast set_data path is unaffected."""
        cb = getattr(self, "scanColorbar", None)
        if cb is None:
            return img
        dLo, dHi = cb.dispRange
        eps = cb._span() * 1e-3
        if dLo <= cb.axisMin + eps and dHi >= cb.axisMax - eps:
            return img          # full range -> nothing masked
        return np.ma.masked_where((img < dLo) | (img > dHi), img)

    def _upsample_for_display(self, img, hV, vV, cacheKey):
        """Sinc-upsample a 2D slice to the finer of the two in-plane voxel
        spacings, for a smoother display. The coarser axis (or axes) is
        resampled up to the finer spacing with a polyphase windowed-sinc
        filter (scipy.signal.resample_poly); the physical extent is unchanged,
        so it is shown via the same imshow ``extent``. Results are cached per
        (kind, orientation, slice, scan). Returns ``img`` unchanged on failure
        or when already isotropic."""
        if not self.upsampleDisplay:
            return img
        hit = self._upsampleCache.get(cacheKey)
        if hit is not None:
            return hit
        out = img
        try:
            nV, nH = img.shape
            if nH >= 2 and nV >= 2:
                dh = abs(hV[-1] - hV[0]) / (nH - 1)
                dv = abs(vV[-1] - vV[0]) / (nV - 1)
                s = min(dh, dv)
                if s > 0:
                    from fractions import Fraction
                    from scipy.signal import resample_poly
                    MAXFAC = 8      # cap upsampling to keep arrays sane
                    arr = np.ascontiguousarray(img, dtype=np.float32)
                    for axis, d in ((1, dh), (0, dv)):
                        if d / s > 1.001:
                            fr = Fraction(d / s).limit_denominator(32)
                            up, dn = fr.numerator, fr.denominator
                            if up > MAXFAC * dn:
                                up, dn = MAXFAC, 1
                            arr = resample_poly(arr, up, dn, axis=axis)
                    out = arr
        except Exception:  # noqa: BLE001
            out = img
        if len(self._upsampleCache) > 96:
            self._upsampleCache.clear()
        self._upsampleCache[cacheKey] = out
        return out

    def _upsample_plane(self, img2, rowVals, colVals, cacheKey):
        """Upsample a 3D-view plane image (shape rows x cols) to the finer
        in-plane spacing and return ``(img2up, dRow, dCol)`` with the new voxel
        spacings. A no-op (original image and spacings) when upsampling is off."""
        nR, nC = img2.shape
        dRow = abs(rowVals[-1] - rowVals[0]) / (nR - 1) if nR > 1 else 1.0
        dCol = abs(colVals[-1] - colVals[0]) / (nC - 1) if nC > 1 else 1.0
        out = self._upsample_for_display(img2, colVals, rowVals, cacheKey)
        if out is not img2 and out.shape != img2.shape:
            nR2, nC2 = out.shape
            if nR2 > 1:
                dRow = abs(rowVals[-1] - rowVals[0]) / (nR2 - 1)
            if nC2 > 1:
                dCol = abs(colVals[-1] - colVals[0]) / (nC2 - 1)
        return out, dRow, dCol

    def _grid_points(self, winId, hV, vV):
        """(N,3) physical (y,x,z) points covering this view's slice plane."""
        H, V = np.meshgrid(hV, vV)
        orient = self.views[winId].orientation
        k = self.slices[winId]
        if orient == VIEW_AXIAL:
            pts = (V, H, np.full_like(H, self.zV[k]))
        elif orient == VIEW_SAGITTAL:
            pts = (H, np.full_like(H, self.xV[k]), V)
        else:
            pts = (np.full_like(H, self.yV[k]), H, V)
        return np.stack([p.ravel() for p in pts], axis=-1), H.shape

    def _resample_slice2d(self, kind, idx, interp, winId, hV, vV):
        """Cached 2D resample of a dose/overlay volume onto this view's plane.
        Entries are validated against the interpolator's identity, so they
        self-invalidate when doseCache / overlayCache are rebuilt."""
        key = (kind, idx, self.views[winId].orientation, self.slices[winId],
               self.scanNum)
        hit = self._slice2dCache.get(key)
        if hit is not None and hit[0] is interp:
            return hit[1]
        pts, shp = self._grid_points(winId, hV, vV)
        slc = interp(pts).reshape(shp).astype(np.float32)
        if len(self._slice2dCache) > 256:     # ~1 MB per 512x512 entry
            self._slice2dCache.clear()
        self._slice2dCache[key] = (interp, slc)
        return slc

    def _struct_contour_segs(self, strNum, winId, hV, vV, slicer):
        """Cached contour segments ([(N,2) arrays]) of a structure on this
        view's slice, extracted with contourpy (much cheaper than ax.contour).
        Entries are validated against the cached mask's identity."""
        mask = self._struct_mask(strNum)
        if mask is None or mask.shape != self.scan3M.shape:
            return None
        key = (strNum, self.views[winId].orientation, self.slices[winId],
               self.scanNum)
        hit = self._structSegCache.get(key)
        if hit is not None and hit[0] is mask:
            return hit[1]
        mslc = slicer(mask)
        if not np.any(mslc):
            segs = []
        else:
            gen = contourpy.contour_generator(
                x=hV, y=vV, z=mslc.astype(np.float32),
                line_type=contourpy.LineType.Separate)
            segs = gen.lines(0.5)
        if len(self._structSegCache) > 4096:  # segments are tiny; hold plenty
            self._structSegCache.clear()
        self._structSegCache[key] = (mask, segs)
        return segs

    @staticmethod
    def _clear_dynamic(ax, keep=()):
        """Remove every artist from ax except the images in `keep` (persistent
        base-scan / overlay / dose images). Lets us reuse those images via
        set_data instead of recreating them with imshow each frame - roughly
        halves the per-frame render cost."""
        if keep is None:
            keep = ()
        elif not isinstance(keep, (list, tuple, set)):
            keep = (keep,)
        for coll in list(ax.collections):
            coll.remove()
        for ln in list(ax.lines):
            ln.remove()
        for txt in list(ax.texts):
            txt.remove()
        for im in list(ax.images):
            if im not in keep:
                im.remove()
        for p in list(ax.patches):
            p.remove()

    def refresh_views(self, only=None):
        if self.planC is None or not self.planC.scan:
            return
        if only is None:           # global change -> auto-refresh the 3D volume
            self._notify_volume3d()
        targets = [only] if only else list(self.activeWins)
        vmin = self.windowCenter - self.windowWidth / 2.0
        vmax = self.windowCenter + self.windowWidth / 2.0

        for orient in targets:
            view = self.views[orient]
            if view.is3d:
                self._render_3d(view)
                continue
            ax = view.ax
            # Full extent (no active pan/zoom): re-enable autoscale so the
            # reused base-scan image's set_extent drives the limits back to the
            # whole slice. A prior zoom/pan disabled autoscale via set_xlim, so
            # without this a double-click reset would leave the view zoomed.
            ax.set_autoscale_on(view.user_limits is None)
            img, extent, hV, vV, slicer = self._slice_data(orient)
            baseIdx = self._axis_scan(orient)
            regComp = None
            if self.regCtl is not None:
                regComp = self.regCtl.compose_slice(orient, img, hV, vV)
            # Reuse persistent images via set_data (base scan, fused overlays
            # and the dose colorwash) instead of recreating them with imshow
            # every frame; clear only the rest. Explicit zorders keep the
            # stacking (base 0 < overlays 0.1 < dose 0.2 < urOMT 0.3 <
            # live-contour fill 0.4) independent of artist creation order.
            useBase = regComp is None and baseIdx == self.scanNum
            scanIm = view._scanIm if getattr(view, "_scanIm", None) is not None \
                and view._scanIm.axes is ax else None

            activeOv = []       # (ovIdx, (cmap, alpha), interp-result) to draw
            for ovIdx, st in self.overlayState.items():
                if not st["on"] or ovIdx == baseIdx \
                        or ovIdx >= len(self.planC.scan):
                    continue
                disp = self._scan_display(ovIdx)   # Scan Display settings
                if disp[1] <= 0:
                    continue
                res = self._overlay_interp(ovIdx)
                if res is None:
                    continue
                activeOv.append((ovIdx, disp, res))
            doseIdx = self._axis_dose(orient)
            doseRes = self._dose_interp(doseIdx) if self.doseAlpha > 0 else None
            drawDose = doseRes is not None and doseRes[1] > 0
            isoMode = self.doseDispMode == "isodose"
            doseIm = getattr(view, "_doseIm", None)
            if doseIm is not None and (not drawDose or isoMode
                                       or doseIm.axes is not ax):
                doseIm = None
            ovIms = {i: im for i, im in getattr(view, "_ovIms", {}).items()
                     if im.axes is ax and any(i == o[0] for o in activeOv)}
            keepIms = ([scanIm] if useBase and scanIm is not None else []) \
                + list(ovIms.values()) \
                + ([doseIm] if doseIm is not None else [])
            self._clear_dynamic(ax, keepIms)
            ax.set_facecolor("black")
            if not useBase:
                view._scanIm = None
            if regComp is not None:    # RGB composite from the QA tool
                ax.imshow(regComp, extent=extent, interpolation="nearest",
                          aspect="equal", zorder=0)
            elif useBase:
                img = self._upsample_for_display(
                    img, hV, vV,
                    ("base", view.orientation, self.slices[orient],
                     self.scanNum))
                img = self._apply_scan_dispmask(img)
                if scanIm is None:
                    view._scanIm = ax.imshow(
                        img, cmap=self.scanCmap, vmin=vmin, vmax=vmax,
                        extent=extent, interpolation="nearest",
                        aspect="equal", alpha=self.scanAlpha, zorder=0)
                else:
                    scanIm.set_data(img)
                    scanIm.set_clim(vmin, vmax)
                    scanIm.set_cmap(self.scanCmap)
                    scanIm.set_alpha(self.scanAlpha)
                    scanIm.set_extent(extent)
            else:
                # per-axis scan override: resample onto the reference grid
                res = self._overlay_interp(baseIdx)
                if res is not None:
                    interp, oLo, oHi = res
                    wl = self.wlByScan.get(baseIdx)
                    if wl is not None:
                        bvmin = wl[0] - wl[1] / 2.0
                        bvmax = wl[0] + wl[1] / 2.0
                    else:
                        bvmin, bvmax = oLo, oHi
                    bSlc = self._resample_slice2d("scan", baseIdx, interp,
                                                  orient, hV, vV)
                    ax.imshow(np.ma.masked_invalid(bSlc), cmap=self.scanCmap,
                              vmin=bvmin, vmax=bvmax, extent=extent,
                              interpolation="bilinear", aspect="equal",
                              alpha=self.scanAlpha, zorder=0)

            # ---- fused scan overlays (display settings from Scan Display) ----
            newOvIms = {}
            for ovIdx, (ovCmap, ovAlpha), res in activeOv:
                interp, oLo, oHi = res
                wl = self.wlByScan.get(ovIdx)
                if wl is not None:
                    oLo, oHi = wl[0] - wl[1] / 2.0, wl[0] + wl[1] / 2.0
                    oHi = max(oHi, oLo + 1e-6)
                ovSlc = self._resample_slice2d("scan", ovIdx, interp,
                                               orient, hV, vV)
                ovMasked = np.ma.masked_invalid(ovSlc)  # hide outside its FOV
                im = ovIms.get(ovIdx)
                if im is not None:
                    im.set_data(ovMasked)
                    im.set_cmap(ovCmap)
                    im.set_clim(oLo, oHi)
                    im.set_alpha(ovAlpha)
                    im.set_extent(extent)
                else:
                    im = ax.imshow(ovMasked, cmap=ovCmap, extent=extent,
                                   vmin=oLo, vmax=oHi, alpha=ovAlpha,
                                   interpolation="bilinear", aspect="equal",
                                   zorder=0.1)
                newOvIms[ovIdx] = im
            view._ovIms = newOvIms

            # ---- dose display: isodose lines or colorwash ----
            if drawDose and isoMode:
                view._doseIm = None
                doseInterp, _doseMax = doseRes
                doseSlc = self._resample_slice2d("dose", doseIdx, doseInterp,
                                                 orient, hV, vV)
                self._draw_isodose_lines(ax, doseSlc, hV, vV,
                                         doseIdx, _doseMax)
            elif drawDose:
                doseInterp, _doseMax = doseRes
                doseSlc = self._resample_slice2d("dose", doseIdx, doseInterp,
                                                 orient, hV, vV)
                cbLo, cbHi = self.colorbar.cbarRange     # colormap mapping
                dLo, dHi = self.colorbar.dispRange       # display (mask) range
                doseMasked = np.ma.masked_where(
                    (doseSlc < max(dLo, 1e-3)) | (doseSlc > dHi), doseSlc)
                if doseIm is not None:
                    doseIm.set_data(doseMasked)
                    doseIm.set_cmap(self.colorbar.mplCmap)
                    doseIm.set_clim(cbLo, max(cbHi, cbLo + 1e-6))
                    doseIm.set_alpha(self.doseAlpha)
                    doseIm.set_extent(extent)
                else:
                    doseIm = ax.imshow(
                        doseMasked, cmap=self.colorbar.mplCmap, extent=extent,
                        vmin=cbLo, vmax=max(cbHi, cbLo + 1e-6),
                        alpha=self.doseAlpha, interpolation="bilinear",
                        aspect="equal", zorder=0.2)
                view._doseIm = doseIm
            else:
                view._doseIm = None

            # ---- urOMT result overlay (speed/flux/pathlines on the scan) ----
            if getattr(self, "uromtOverlay", None) is not None:
                self._draw_uromt_overlay(orient, ax, extent, hV, vV, slicer)

            # ---- structure contours ----
            ctl = self.contourCtl
            editStrNum = (ctl.structNum if ctl is not None and ctl.isVisible()
                          else None)
            for strNum in self._axis_structs(orient):
                if strNum == editStrNum:
                    continue   # being edited: shown via the live overlay below
                if HAS_CONTOURPY:
                    # cached segment extraction + a single LineCollection is
                    # much cheaper than re-running ax.contour every frame
                    segs = self._struct_contour_segs(strNum, orient,
                                                     hV, vV, slicer)
                    if not segs:
                        continue
                    color = self._struct_color(strNum)
                    ax.add_collection(LineCollection(
                        segs, colors=[color],
                        linewidths=self.structLineWidth, zorder=2))
                    if self.showStructDots:
                        self._draw_contour_dots(ax, segs, color)
                else:      # fallback: matplotlib's contour (uncached)
                    mask = self._struct_mask(strNum)
                    if mask is None or mask.shape != self.scan3M.shape:
                        continue
                    mslc = slicer(mask)
                    if not np.any(mslc):
                        continue
                    color = self._struct_color(strNum)
                    cs = ax.contour(hV, vV, mslc.astype(float), levels=[0.5],
                                    colors=[color],
                                    linewidths=self.structLineWidth)
                    if self.showStructDots:
                        self._draw_contour_dots(ax, cs.allsegs[0], color)

            # ---- live contouring overlay (working mask being edited) ----
            if ctl is not None and ctl.isVisible() and ctl.mask3M is not None \
                    and ctl.mask3M.shape == self.scan3M.shape:
                cslc = slicer(ctl.mask3M)
                im = cs = None
                if np.any(cslc):
                    im = ax.imshow(
                        np.ma.masked_where(~cslc, cslc.astype(float)),
                        cmap=ListedColormap([ctl.color]), extent=extent,
                        alpha=0.35, vmin=0, vmax=1,
                        interpolation="nearest", aspect="equal", zorder=0.4)
                    cs = ax.contour(hV, vV, cslc.astype(float), levels=[0.5],
                                    colors=[ctl.color], linewidths=1.6,
                                    linestyles="--")
                if view is ctl.axView:
                    # the live updater owns the axial overlay, so each brush
                    # step removes and redraws it (no stale boundary on erase)
                    ctl._liveIm = im
                    ctl._liveContour = cs

            # ---- registration QA split/lens guides ----
            if regComp is not None:
                self.regCtl.draw_guides(view)

            # ---- IMRTP beam overlays (projected onto this view plane) ----
            if self.beams:
                self._draw_beams_2d(view)

            # ---- crosshairs from the other view types ----
            self._draw_crosshair(view)

            # ---- ruler (axes were cleared, recreate its artists) ----
            view.redraw_ruler()

            ax.set_xticks([]), ax.set_yticks([])
            if view.user_limits is not None:    # keep user pan/zoom
                ax.set_xlim(view.user_limits[0])
                ax.set_ylim(view.user_limits[1])

            # ---- patient orientation labels (L/R/A/P/S/I) ----
            if self.showOrientation:
                self._draw_orientation_labels(view)

            view.label.setText(self._view_label_text(view))
            view.canvas.draw_idle()

    # Through-plane pyCERR virtual axis for each view (column of
    # cerrToDcmTransM: 0=x/col, 1=y/row, 2=z/slice).
    _VIEW_THRU_COL = {VIEW_AXIAL: 2, VIEW_SAGITTAL: 0, VIEW_CORONAL: 1}
    # Dominant DICOM/patient axis of the through-plane direction -> anatomical
    # plane (DICOM X = L-R -> Sagittal, Y = A-P -> Coronal, Z = S-I -> Axial).
    _DCM_AXIS_PLANE = (VIEW_SAGITTAL, VIEW_CORONAL, VIEW_AXIAL)

    def _anatomical_plane(self, orient):
        """Anatomical plane name for a view, from the base scan's actual image
        orientation (so a sagittally/coronally acquired scan is labelled by the
        plane it really shows, not by its array slice axis). Falls back to the
        view's own key when the scan geometry is unavailable."""
        col = self._VIEW_THRU_COL.get(orient)
        if col is None or self.planC is None \
                or not (0 <= self.scanNum < len(self.planC.scan)):
            return orient
        try:
            M = np.asarray(self.planC.scan[self.scanNum].cerrToDcmTransM,
                           dtype=float)
            direction = M[:3, col]
            if not np.any(direction):
                return orient
            return self._DCM_AXIS_PLANE[int(np.argmax(np.abs(direction)))]
        except Exception:  # noqa: BLE001
            return orient

    def _view_label_text(self, view):
        """View title: anatomical plane, slice-plane coordinate (cm),
        scan #/modality and dose shown in this view, plus any per-axis
        overrides / lock state."""
        winId, orient = view.winId, view.orientation
        if view.is3d:
            return view.label.text()        # 3D label set by its renderer
        k = self.slices[winId]
        if orient == VIEW_AXIAL:
            axis, val = "z", self.zV[k]
        elif orient == VIEW_SAGITTAL:
            axis, val = "x", self.xV[k]
        else:
            axis, val = "y", self.yV[k]
        plane = self._anatomical_plane(orient)

        baseIdx = self._axis_scan(winId)
        mod = getattr(self.planC.scan[baseIdx].scanInfo[0], "imageType",
                      "scan")
        doseIdx = self._axis_dose(winId)
        doseStr = "no dose" if (doseIdx is None or doseIdx < 0) \
            else f"dose {doseIdx}"

        flags = []
        if self.axisSel[winId]["structs"] is not None:
            flags.append("structs*")
        if self.lockViews:
            flags.append("locked")
        extra = "  [" + ", ".join(flags) + "]" if flags else ""
        return (f"{plane}   {axis}={val:.2f} cm   |   "
                f"scan {baseIdx} ({mod})   |   {doseStr}{extra}")

    @staticmethod
    def _project_points(orient, pts):
        """Project (N,3) world (x,y,z) points to a view's (h, v) coords."""
        pts = np.asarray(pts, dtype=float)
        hAxis, vAxis = AXES_2D[orient]
        col = {"x": 0, "y": 1, "z": 2}
        return pts[:, col[hAxis]], pts[:, col[vAxis]]

    def _beam_slice_polygon(self, view, apex, corners):
        """Cross-section of the beam pyramid (apex + 4 far corners) with this
        view's slice plane, returned as an ordered (M,2) closed polygon in the
        view's (h, v) coords - or None if the slice misses the beam. Changes
        as you scroll, since the slice plane moves."""
        orient = view.orientation
        axisIdx, coords = {VIEW_AXIAL: (2, self.zV),
                           VIEW_SAGITTAL: (0, self.xV),
                           VIEW_CORONAL: (1, self.yV)}[orient]
        sv = float(coords[self.slices[view.winId]])
        edges = [(apex, corners[i]) for i in range(4)]          # lateral
        edges += [(corners[i], corners[(i + 1) % 4]) for i in range(4)]  # base
        eps = 1e-6
        pts = []
        for a, b in edges:
            da, db = a[axisIdx] - sv, b[axisIdx] - sv
            aon, bon = abs(da) < eps, abs(db) < eps
            if aon:
                pts.append(a)
            if bon:
                pts.append(b)
            if not aon and not bon and da * db < 0:    # strict crossing
                t = da / (da - db)
                pts.append(a + t * (b - a))
        if len(pts) < 2:
            return None
        h, v = self._project_points(orient, np.asarray(pts))
        hv = np.column_stack([h, v])
        # dedupe near-identical points
        keep = [hv[0]]
        for p in hv[1:]:
            if min(np.hypot(*(p - q)) for q in keep) > 1e-4:
                keep.append(p)
        hv = np.asarray(keep)
        if len(hv) < 2:
            return None
        # order as a convex polygon (cross-section of a convex solid)
        c = hv.mean(axis=0)
        order = np.argsort(np.arctan2(hv[:, 1] - c[1], hv[:, 0] - c[0]))
        ring = hv[order]
        return np.vstack([ring, ring[0]])      # close it

    def _draw_beams_2d(self, view):
        """Draw beams on this 2D view as a single LineCollection. When a beam
        carries pyramid geometry ("apex"/"corners") its per-slice cross-section
        is drawn (changes with the slice); otherwise the polylines are
        projected. Zoom is preserved (the far source never autoscales)."""
        segs, colors = [], []
        for beam in self.beams:
            color = beam.get("color", (0.2, 0.85, 0.9))
            apex = beam.get("apex")
            corners = beam.get("corners")
            if apex is not None and corners is not None:
                ring = self._beam_slice_polygon(view, np.asarray(apex),
                                                np.asarray(corners))
                if ring is not None:
                    segs.append(ring)
                    colors.append(color)
            else:
                for poly in beam["polylines"]:
                    h, v = self._project_points(view.orientation, poly)
                    segs.append(np.column_stack([h, v]))
                    colors.append(color)
        if not segs:
            return
        xl, yl = view.ax.get_xlim(), view.ax.get_ylim()
        lc = LineCollection(segs, colors=colors, linewidths=1.2, alpha=0.9,
                            zorder=13)
        view.ax.add_collection(lc, autolim=False)
        view.ax.set_xlim(xl)            # keep the scan-extent / user zoom
        view.ax.set_ylim(yl)

    # DICOM/patient axis -> (label at +axis end, label at -axis end).
    # DICOM LPS: +X = Left, +Y = Posterior, +Z = Superior.
    _DCM_AXIS_ENDS = (("L", "R"), ("P", "A"), ("S", "I"))

    def _axis_anatomy(self, axisLetter):
        """(labelAtIncreasingEnd, labelAtDecreasingEnd) for a pyCERR virtual
        axis ('x'/'y'/'z'), from the base scan's actual image orientation, so
        the L/R/A/P/S/I markers are correct for sagittally/coronally acquired
        (or oblique) scans. Falls back to the fixed L/A/I convention when the
        scan geometry is unavailable."""
        col = {"x": 0, "y": 1, "z": 2}[axisLetter]
        try:
            if self.planC is not None and 0 <= self.scanNum < len(self.planC.scan):
                M = np.asarray(self.planC.scan[self.scanNum].cerrToDcmTransM,
                               dtype=float)
                d = M[:3, col]
                if np.any(d):
                    a = int(np.argmax(np.abs(d)))
                    hi, lo = self._DCM_AXIS_ENDS[a]
                    return (hi, lo) if d[a] > 0 else (lo, hi)
        except Exception:  # noqa: BLE001
            pass
        return ORIENT_POS[axisLetter], ORIENT_NEG[axisLetter]

    def _draw_orientation_labels(self, view):
        """L/R/A/P/S/I markers at the edge midpoints of a 2D view, derived from
        the displayed axes' actual patient directions (via the base scan's
        cerrToDcmTransM), so they are correct for any acquisition orientation."""
        ax = view.ax
        hAxis, vAxis = AXES_2D[view.orientation]
        hPos, hNeg = self._axis_anatomy(hAxis)
        vPos, vNeg = self._axis_anatomy(vAxis)
        x0, x1 = ax.get_xlim()
        left, right = (hNeg, hPos) if x1 >= x0 else (hPos, hNeg)
        y0, y1 = ax.get_ylim()
        bottom, top = (vNeg, vPos) if y1 >= y0 else (vPos, vNeg)
        kw = dict(transform=ax.transAxes, color="#e8e8e8", fontsize=9,
                  fontweight="bold", ha="center", va="center", zorder=15,
                  bbox=dict(facecolor="black", alpha=0.45, edgecolor="none",
                            pad=1.5))
        ax.text(0.03, 0.5, left, **kw)
        ax.text(0.97, 0.5, right, **kw)
        ax.text(0.5, 0.96, top, **kw)
        ax.text(0.5, 0.04, bottom, **kw)

    def _refresh_3d_views(self):
        for wid in self.activeWins:
            if self.views[wid].is3d:
                self.refresh_views(only=wid)

    def _on_view_reset(self, winId):
        """Double-click reset: clear 2D pan/zoom, or restore the 3D camera to
        its default framing, then re-render the affected view."""
        view = self.views.get(winId)
        if view is not None and getattr(view, "is3d", False):
            view._vtk_cam_set = False     # re-applied by _render_3d_vtk
        self.refresh_views(only=winId)

    def _render_3d(self, view):
        if view.uses_vtk:
            self._render_3d_vtk(view)
        else:
            self._render_3d_mpl(view)

    def _notify_volume3d(self):
        """If the 3D Volume tool window is open, schedule a debounced rebuild so
        it tracks the main GUI's window/level, transparency, structure and dose
        selections automatically (no manual Refresh needed)."""
        dlg = getattr(self, "_volume3dDialog", None)
        if dlg is not None:
            try:
                dlg.request_refresh()
            except Exception:  # noqa: BLE001 (dialog may be closing)
                self._volume3dDialog = None

    def _plane_slices_3d(self):
        """Current plane indices, window range and plane label text.

        The plane names follow the base scan's actual orientation (so a
        sagittally/coronally acquired scan is labelled correctly)."""
        kA = self.lastSlice[VIEW_AXIAL]
        kS = self.lastSlice[VIEW_SAGITTAL]
        kC = self.lastSlice[VIEW_CORONAL]
        label = ("3D Cut Planes  -  planes: "
                 f"{self._anatomical_plane(VIEW_AXIAL).lower()} {kA + 1}, "
                 f"{self._anatomical_plane(VIEW_SAGITTAL).lower()} {kS + 1}, "
                 f"{self._anatomical_plane(VIEW_CORONAL).lower()} {kC + 1}")
        return kA, kS, kC, label

    def _scan_grid_geometry(self):
        """Canonical ascending scan-grid axes and the flips applied to get
        them (pyCERR arrays are (row=y, col=x, slice=z))."""
        flipR = self.yV[0] > self.yV[-1]
        flipC = self.xV[0] > self.xV[-1]
        flipS = len(self.zV) > 1 and self.zV[0] > self.zV[-1]
        xA = self.xV[::-1] if flipC else self.xV
        yA = self.yV[::-1] if flipR else self.yV
        zA = self.zV[::-1] if flipS else self.zV
        return xA, yA, zA, flipR, flipC, flipS

    @staticmethod
    def _pv_volume(arr3, xA, yA, zA):
        """pyvista ImageData from a (y, x, z) array on ascending axes."""
        nR, nC, nS = arr3.shape
        grid = pv.ImageData(
            dimensions=(nC, nR, nS),
            spacing=(float(xA[1] - xA[0]) if nC > 1 else 1.0,
                     float(yA[1] - yA[0]) if nR > 1 else 1.0,
                     float(zA[1] - zA[0]) if nS > 1 else 1.0),
            origin=(float(xA[0]), float(yA[0]), float(zA[0])))
        grid.point_data["v"] = np.transpose(
            arr3, (1, 0, 2)).ravel(order="F")     # x fastest, then y, z
        return grid

    @staticmethod
    def _resample_volume(arr, xA, yA, zA, frac):
        """Resample a (y, x, z) volume and its ascending axes to ``frac`` of
        the original resolution (linear interpolation); frac >= 1 is a no-op."""
        if frac >= 1.0:
            return arr, xA, yA, zA
        from scipy.ndimage import zoom
        out = zoom(arr, frac, order=1)
        nY, nX, nZ = out.shape
        return (out,
                np.linspace(float(xA[0]), float(xA[-1]), nX),
                np.linspace(float(yA[0]), float(yA[-1]), nY),
                np.linspace(float(zA[0]), float(zA[-1]), nZ))

    @staticmethod
    def _smallest_spacing(xA, yA, zA, shape):
        """Smallest native voxel spacing across the three (ascending) axes."""
        nR, nC, nS = shape

        def sp(a, n):
            return abs(float(a[-1]) - float(a[0])) / (n - 1) if n > 1 else 0.0
        cand = [d for d in (sp(yA, nR), sp(xA, nC), sp(zA, nS)) if d > 0]
        return min(cand) if cand else 0.0

    @staticmethod
    def _resample_volume_isotropic(vol, xA, yA, zA, targetSpacing, maxDim=512):
        """Linearly resample a (y, x, z) volume to isotropic voxels of
        ``targetSpacing`` (cm) for visualization. Each axis's sample count is
        derived from its physical extent so voxels are cubic, capped at
        ``maxDim`` per axis. Returns the volume and its (still ascending) axes;
        a no-op when already at the target sampling."""
        nR, nC, nS = vol.shape

        def cnt(a, n):
            if n <= 1:
                return n
            ext = abs(float(a[-1]) - float(a[0]))
            if ext <= 0 or targetSpacing <= 0:
                return n
            return max(2, min(int(round(ext / targetSpacing)) + 1, maxDim))
        nR2, nC2, nS2 = cnt(yA, nR), cnt(xA, nC), cnt(zA, nS)
        if (nR2, nC2, nS2) == (nR, nC, nS):
            return vol, xA, yA, zA
        from scipy.ndimage import zoom
        out = zoom(vol, (nR2 / nR, nC2 / nC, nS2 / nS), order=1)
        return (out,
                np.linspace(float(xA[0]), float(xA[-1]), out.shape[1]),
                np.linspace(float(yA[0]), float(yA[-1]), out.shape[0]),
                np.linspace(float(zA[0]), float(zA[-1]), out.shape[2]))

    def _pv_struct_mesh(self, strNum, frac=1.0):
        """Cached smoothed surface mesh of a structure mask, or None.
        ``frac < 1`` extracts the surface from a resampled mask (faster,
        coarser); results are cached per (structure, fraction)."""
        key = (strNum, round(frac, 3))
        if key not in self._pvStructCache:
            surf = None
            try:
                mask = self._struct_mask(strNum)
                if mask is not None and mask.shape == self.scan3M.shape \
                        and np.any(mask):
                    xA, yA, zA, flipR, flipC, flipS = \
                        self._scan_grid_geometry()
                    m = mask.astype(np.float32)
                    if flipR:
                        m = m[::-1, :, :]
                    if flipC:
                        m = m[:, ::-1, :]
                    if flipS:
                        m = m[:, :, ::-1]
                    m, xA, yA, zA = self._resample_volume(m, xA, yA, zA, frac)
                    grid = self._pv_volume(m, xA, yA, zA)
                    surf = grid.contour([0.5])
                    if surf.n_points:
                        surf = surf.smooth(n_iter=30, relaxation_factor=0.1)
                    else:
                        surf = None
            except Exception:  # noqa: BLE001
                surf = None
            self._pvStructCache[key] = surf
        return self._pvStructCache[key]

    def _scan_disp_range(self):
        """Scan display (cyan) range in scan units, or (-inf, inf) when the
        range spans the whole axis (i.e. no masking)."""
        cb = getattr(self, "scanColorbar", None)
        if cb is None:
            return float("-inf"), float("inf")
        lo, hi = cb.dispRange
        if lo <= cb.axisMin + cb._span() * 1e-3 \
                and hi >= cb.axisMax - cb._span() * 1e-3:
            return float("-inf"), float("inf")
        return float(lo), float(hi)

    def _dose_disp_range(self):
        """Dose display (cyan) range in dose units, or (-inf, inf) when the
        range spans the whole axis (i.e. no masking)."""
        cb = getattr(self, "colorbar", None)
        if cb is None:
            return float("-inf"), float("inf")
        lo, hi = cb.dispRange
        if lo <= cb.axisMin + cb._span() * 1e-3 \
                and hi >= cb.axisMax - cb._span() * 1e-3:
            return float("-inf"), float("inf")
        return float(lo), float(hi)

    def _pv_dose_iso(self, doseIdx, frac=1.0):
        """Cached (isodose surfaces, doseMax) for a dose index, or None.
        Surfaces are contoured at the panel's isodose levels (resolved to Gy via
        :meth:`_isodose_abs_levels`, matching the 2D isodose lines), restricted
        to the dose display (cyan) range so the 3D views honor it. ``frac < 1``
        contours a resampled dose grid. Results are cached per
        (dose, fraction, levels)."""
        # Resolve levels first so the cache invalidates when they change.
        dmaxAll = self._dose_interp(doseIdx)
        dmaxAll = dmaxAll[1] if dmaxAll is not None else 0.0
        dLo, dHi = self._dose_disp_range()
        levels = tuple(round(lv, 4)
                       for lv in self._isodose_abs_levels(doseIdx, dmaxAll)
                       if lv > 0 and dLo <= lv <= dHi)
        key = (doseIdx, round(frac, 3), levels)
        if key not in self._pvDoseCache:
            res = None
            try:
                d = self.planC.dose[doseIdx]
                dose3M = np.asarray(d.doseArray, dtype=np.float32)
                xD, yD, zD = d.getDoseXYZVals()
                yD, dose3M = ascending(yD, dose3M, axis=0)
                xD, dose3M = ascending(xD, dose3M, axis=1)
                zD, dose3M = ascending(zD, dose3M, axis=2)
                dose3M, xD, yD, zD = \
                    self._resample_volume(dose3M, xD, yD, zD, frac)
                dmax = float(dose3M.max())
                # Keep only levels within the dose range (contouring outside it
                # yields empty surfaces / VTK warnings).
                drawLevels = [lv for lv in levels if 0 < lv <= dmax]
                if dmax > 0 and drawLevels:
                    grid = self._pv_volume(dose3M, xD, yD, zD)
                    iso = grid.contour(drawLevels)
                    if iso.n_points:
                        res = (iso, dmax)
            except Exception:  # noqa: BLE001
                res = None
            self._pvDoseCache[key] = res
        return self._pvDoseCache[key]

    def _render_3d_vtk(self, view):
        """GPU (pyvista/VTK) 3D view: full-resolution textured orthogonal
        planes at the slices selected in the 2D views + locator outlines."""
        pl = view.vtk_widget
        kA, kS, kC, label = self._plane_slices_3d()
        vmin = self.windowCenter - self.windowWidth / 2.0
        vmax = self.windowCenter + self.windowWidth / 2.0
        nR, nC, nS = self.scan3M.shape

        # canonical ascending axes; flip slice images to match
        flipR = self.yV[0] > self.yV[-1]
        flipC = self.xV[0] > self.xV[-1]
        flipS = nS > 1 and self.zV[0] > self.zV[-1]
        xA = self.xV[::-1] if flipC else self.xV
        yA = self.yV[::-1] if flipR else self.yV
        zA = self.zV[::-1] if flipS else self.zV
        dx = float(xA[1] - xA[0]) if nC > 1 else 1.0
        dy = float(yA[1] - yA[0]) if nR > 1 else 1.0
        dz = float(zA[1] - zA[0]) if nS > 1 else 1.0

        def canon(img2, flip0, flip1):
            if flip0:
                img2 = img2[::-1, :]
            if flip1:
                img2 = img2[:, ::-1]
            return np.ascontiguousarray(img2)

        sn = self.scanNum
        # honor the scan display (cyan) range: values outside it become NaN and
        # are drawn transparent (nan_opacity=0), so the planes match the 2D mask.
        dLo, dHi = self._scan_disp_range()
        masked = np.isfinite(dLo) or np.isfinite(dHi)

        def mask(scalars):
            if not masked:
                return scalars
            a = np.asarray(scalars, dtype=np.float32).copy()
            a[(a < dLo) | (a > dHi)] = np.nan
            return a

        aImg = canon(self.scan3M[:, :, kA], flipR, flipC)   # (y, x)
        aImg, dyA, dxA = self._upsample_plane(aImg, yA, xA, ("3dA", kA, sn))
        nRa, nCa = aImg.shape
        axial = pv.ImageData(dimensions=(nCa, nRa, 1), spacing=(dxA, dyA, 1.0),
                             origin=(float(xA[0]), float(yA[0]),
                                     float(self.zV[kA])))
        axial.point_data["v"] = mask(aImg.ravel())          # x fastest

        sImg = canon(self.scan3M[:, kS, :], flipR, flipS)   # (y, z)
        sImg, dyS, dzS = self._upsample_plane(sImg, yA, zA, ("3dS", kS, sn))
        nRs, nSs = sImg.shape
        sag = pv.ImageData(dimensions=(1, nRs, nSs), spacing=(1.0, dyS, dzS),
                           origin=(float(self.xV[kS]), float(yA[0]),
                                   float(zA[0])))
        sag.point_data["v"] = mask(sImg.ravel(order="F"))   # y fastest

        cImg = canon(self.scan3M[kC, :, :], flipC, flipS)   # (x, z)
        cImg, dxC, dzC = self._upsample_plane(cImg, xA, zA, ("3dC", kC, sn))
        nCc, nSc = cImg.shape
        cor = pv.ImageData(dimensions=(nCc, 1, nSc), spacing=(dxC, 1.0, dzC),
                           origin=(float(xA[0]), float(self.yV[kC]),
                                   float(zA[0])))
        cor.point_data["v"] = mask(cImg.ravel(order="F"))   # x fastest

        pl.clear()      # actors only; the camera is preserved
        view._plane_actors = {}
        for orient, mesh in ((VIEW_AXIAL, axial), (VIEW_SAGITTAL, sag),
                             (VIEW_CORONAL, cor)):
            view._plane_actors[orient] = pl.add_mesh(
                mesh, cmap=self.scanCmap, clim=(vmin, max(vmax, vmin + 1e-6)),
                opacity=self.scanAlpha, nan_opacity=0.0, lighting=False,
                show_scalar_bar=False, render=False)

        # colored plane outlines (locators)
        x0, x1 = float(xA[0]), float(xA[-1])
        y0, y1 = float(yA[0]), float(yA[-1])
        z0, z1 = float(zA[0]), float(zA[-1])
        zP, xP, yP = float(self.zV[kA]), float(self.xV[kS]), float(self.yV[kC])
        edges = {
            VIEW_AXIAL: [(x0, y0, zP), (x1, y0, zP), (x1, y1, zP),
                         (x0, y1, zP), (x0, y0, zP)],
            VIEW_SAGITTAL: [(xP, y0, z0), (xP, y1, z0), (xP, y1, z1),
                            (xP, y0, z1), (xP, y0, z0)],
            VIEW_CORONAL: [(x0, yP, z0), (x1, yP, z0), (x1, yP, z1),
                           (x0, yP, z1), (x0, yP, z0)],
        }
        view._outline_actors = {}
        for orient, ptsList in edges.items():   # plane locators (always shown)
            pts = np.asarray(ptsList, dtype=float)
            poly = pv.PolyData()
            poly.points = pts
            poly.lines = np.hstack(([len(pts)], np.arange(len(pts)))).astype(
                np.int64)
            view._outline_actors[orient] = pl.add_mesh(
                poly, color=PLANE_COLORS[orient], line_width=2,
                pickable=False, show_scalar_bar=False, render=False)

        # ---- structure surfaces (follow the Structures checklist) ----
        for strNum in self._axis_structs(view.winId):
            surf = self._pv_struct_mesh(strNum)
            if surf is not None:
                pl.add_mesh(surf, color=self._struct_color(strNum),
                            opacity=0.45, smooth_shading=True,
                            pickable=False, show_scalar_bar=False,
                            name=f"struct{strNum}", render=False)

        # ---- isodose surfaces (follow the Dose combo & alpha slider) ----
        doseIdx = self._axis_dose(view.winId)
        if doseIdx is not None and doseIdx >= 0 and self.doseAlpha > 0:
            res = self._pv_dose_iso(doseIdx)
            if res is not None:
                iso, _dmax = res
                cbLo, cbHi = self.colorbar.cbarRange
                pl.add_mesh(iso, cmap=self.colorbar.mplCmap,
                            clim=(cbLo, max(cbHi, cbLo + 1e-6)),
                            opacity=min(max(self.doseAlpha, 0.0), 0.6),
                            pickable=False, show_scalar_bar=False,
                            name=f"isodose{doseIdx}", render=False)

        # ---- IMRTP beam overlays: ONE combined polyline actor (fast) ----
        if self.beams:
            pts_list, conn, cellRGB, off = [], [], [], 0
            for beam in self.beams:
                rgb = (np.asarray(beam.get("color", (0.2, 0.85, 0.9)))
                       * 255).astype(np.uint8)
                for poly in beam["polylines"]:
                    p = np.asarray(poly, dtype=float)
                    n = len(p)
                    if n < 2:
                        continue
                    pts_list.append(p)
                    conn.append(np.concatenate(([n], np.arange(off, off + n))))
                    cellRGB.append(rgb)
                    off += n
            if pts_list:
                pd = pv.PolyData()
                pd.points = np.vstack(pts_list)
                pd.lines = np.concatenate(conn).astype(np.int64)
                pd.cell_data["rgb"] = np.asarray(cellRGB, dtype=np.uint8)
                pl.add_mesh(pd, scalars="rgb", rgb=True, line_width=2,
                            pickable=False, show_scalar_bar=False,
                            name="beams", render=False)

        # ---- urOMT velocity / flux arrows + pathlines ----
        if getattr(self, "uromtOverlay", None) is not None:
            self._add_uromt_3d_vtk(pl)

        # wire the plane-drag hooks once per widget
        if pl.pick_plane is None:
            pl.pick_plane = lambda pos, v=view: self._vtk_pick_plane(v, pos)
            pl.drag_plane = lambda pos, v=view: self._vtk_drag_plane(v, pos)
            pl.end_plane_drag = lambda v=view: self._vtk_end_plane_drag(v)

        # patient-orientation triad: (re)enable after the window is realized
        # (enabling it at widget-creation time leaves it invisible)
        if self.showOrientation:
            try:
                # label the +x/+y/+z arrows by their true patient direction,
                # recomputed each render so a scan-orientation change relabels
                # the triad in place.
                triad = (self._axis_anatomy("x")[0], self._axis_anatomy("y")[0],
                         self._axis_anatomy("z")[0])
                if getattr(pl.renderer, "axes_widget", None) is None:
                    pl.add_axes(xlabel=triad[0], ylabel=triad[1],
                                zlabel=triad[2])
                    # caption text defaults to black: match each label to
                    # its axis shaft color so it reads on dark backgrounds
                    axes = pl.renderer.axes_actor
                    for cap, shaft in (
                            (axes.GetXAxisCaptionActor2D(),
                             axes.GetXAxisShaftProperty()),
                            (axes.GetYAxisCaptionActor2D(),
                             axes.GetYAxisShaftProperty()),
                            (axes.GetZAxisCaptionActor2D(),
                             axes.GetZAxisShaftProperty())):
                        tp = cap.GetCaptionTextProperty()
                        tp.SetColor(*shaft.GetColor())
                        tp.SetShadow(0)
                        tp.BoldOn()
                    view._orientTriad = triad
                elif getattr(view, "_orientTriad", None) != triad:
                    axes = pl.renderer.axes_actor
                    axes.SetXAxisLabelText(triad[0])
                    axes.SetYAxisLabelText(triad[1])
                    axes.SetZAxisLabelText(triad[2])
                    view._orientTriad = triad
                pl.renderer.axes_widget.SetEnabled(1)
            except Exception:  # noqa: BLE001
                pass
        elif getattr(pl.renderer, "axes_widget", None) is not None:
            try:
                pl.renderer.axes_widget.SetEnabled(0)
            except Exception:  # noqa: BLE001
                pass

        if not getattr(view, "_vtk_cam_set", False):
            # medical default: superior up (-z in pyCERR virtual coords),
            # viewed from the patient's anterior-superior-right
            cx, cy, cz = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
            diag = float(np.linalg.norm([x1 - x0, y1 - y0, z1 - z0])) or 1.0
            pos = (cx - 0.9 * diag,    # -x = patient right
                   cy + 1.2 * diag,    # +y = anterior
                   cz - 0.9 * diag)    # -z = superior
            pl.camera_position = [pos, (cx, cy, cz), (0.0, 0.0, -1.0)]
            pl.reset_camera()          # refit distance, keep direction/up
            view._vtk_cam_set = True
        pl.render()
        view.label.setText(label)

    # --------------------------------------------- draggable 3D planes ------
    def _vtk_display_pos(self, view, qpos):
        """Qt widget position -> VTK display coordinates (origin bottom-left,
        physical pixels)."""
        pl = view.vtk_widget
        ratio = float(pl.devicePixelRatioF())
        x = qpos.x() * ratio
        y = pl.height() * ratio - qpos.y() * ratio - 1
        return x, y

    def _vtk_pick_plane(self, view, qpos):
        """Left press in the 3D view: True (consume) if a plane was grabbed."""
        if self.planC is None or not self.planC.scan:
            return False
        planeActors = getattr(view, "_plane_actors", {})
        outlineActors = getattr(view, "_outline_actors", {})
        if not planeActors:
            return False
        try:
            import vtk
            picker = vtk.vtkCellPicker()   # geometric ray cast: reliable
            picker.SetTolerance(0.002)
        except Exception:  # noqa: BLE001
            return False
        pl = view.vtk_widget
        x, y = self._vtk_display_pos(view, qpos)
        if not picker.Pick(x, y, 0, pl.renderer):
            return False
        picked = picker.GetActor()
        # plane and outline actor maps share orientation keys: check both
        orient = next((o for o, a in planeActors.items() if a is picked),
                      None)
        if orient is None:
            orient = next((o for o, a in outlineActors.items()
                           if a is picked), None)
        if orient is None:
            return False

        axisIdx = {VIEW_AXIAL: 2, VIEW_SAGITTAL: 0, VIEW_CORONAL: 1}[orient]
        n = np.zeros(3)
        n[axisIdx] = 1.0
        wp = np.asarray(picker.GetPickPosition(), dtype=float)
        ren = pl.renderer

        def w2d(p):
            ren.SetWorldPoint(p[0], p[1], p[2], 1.0)
            ren.WorldToDisplay()
            return np.asarray(ren.GetDisplayPoint()[:2], dtype=float)

        base = w2d(wp)
        dird = w2d(wp + n) - base    # screen direction of the plane normal
        if float(np.dot(dird, dird)) < 1e-9:
            return False             # normal is perpendicular to the screen
        coords = {VIEW_AXIAL: self.zV, VIEW_SAGITTAL: self.xV,
                  VIEW_CORONAL: self.yV}[orient]
        view._drag3d = {
            "orient": orient,
            "coords": np.asarray(coords, dtype=float),
            "coord0": float(coords[self.lastSlice[orient]]),
            "base": base, "dird": dird, "tRender": 0.0,
        }
        return True

    def _vtk_drag_plane(self, view, qpos):
        st = getattr(view, "_drag3d", None)
        if st is None:
            return
        x, y = self._vtk_display_pos(view, qpos)
        d = np.array([x, y]) - st["base"]
        t = float(np.dot(d, st["dird"]) / np.dot(st["dird"], st["dird"]))
        coord = st["coord0"] + t
        k = int(np.argmin(np.abs(st["coords"] - coord)))
        orient = st["orient"]
        if k == self.lastSlice[orient]:
            return
        self._set_orientation_slice(orient, k)
        # live plane motion: re-render the 3D scene, throttled to ~25 fps
        now = time.monotonic()
        if now - st["tRender"] > 0.04:
            st["tRender"] = now
            self._timer3d.stop()
            self._refresh_3d_views()

    def _vtk_end_plane_drag(self, view):
        view._drag3d = None
        self._timer3d.stop()
        self._refresh_3d_views()

    def _set_orientation_slice(self, orient, k):
        """Move the given view type to slice k, driving the first matching
        2D window's slider (so crosshairs, lock, etc. all follow)."""
        for wid in self.activeWins:
            v = self.views[wid]
            if not v.is3d and v.orientation == orient:
                v.slider.setValue(k)     # triggers on_slice_changed
                return
        self.lastSlice[orient] = k       # no 2D window shows it: update state

    def on_crosshair_dragged(self, winId, xdata, ydata, dragV, dragH):
        """Dragging a crosshair line in one view scrolls the perpendicular
        views to the dragged in-plane location."""
        if self.planC is None or not self.planC.scan:
            return
        orient = self.views[winId].orientation
        coordArr = {VIEW_AXIAL: self.zV, VIEW_SAGITTAL: self.xV,
                    VIEW_CORONAL: self.yV}
        if dragV:                        # vertical line -> h-axis target
            tgt = CROSS_TARGET[(orient, "h")]
            arr = np.asarray(coordArr[tgt])
            self._set_orientation_slice(tgt, int(np.argmin(np.abs(arr - xdata))))
        if dragH:                        # horizontal line -> v-axis target
            tgt = CROSS_TARGET[(orient, "v")]
            arr = np.asarray(coordArr[tgt])
            self._set_orientation_slice(tgt, int(np.argmin(np.abs(arr - ydata))))
        # move the dragged view's own crosshair live (its slice is unchanged)
        view = self.views[winId]
        self._position_crosshair(view)
        view.canvas.draw_idle()

    def _render_3d_mpl(self, view):
        """3D view: the three orthogonal planes (at the slices selected in
        the 2D views) textured with the scan, plus colored plane outlines -
        the analog of CERR's 3D plane locators."""
        ax = view.ax
        ax.clear()
        kA = self.lastSlice[VIEW_AXIAL]
        kS = self.lastSlice[VIEW_SAGITTAL]
        kC = self.lastSlice[VIEW_CORONAL]
        vmin = self.windowCenter - self.windowWidth / 2.0
        vmax = self.windowCenter + self.windowWidth / 2.0
        scanCmapObj = plt.get_cmap(self.scanCmap)

        def norm(a):
            return np.clip((a - vmin) / max(vmax - vmin, 1e-6), 0, 1)

        nR, nC, nS = self.scan3M.shape
        ir = slice(None, None, max(1, nR // N3D))
        ic = slice(None, None, max(1, nC // N3D))
        is_ = slice(None, None, max(1, nS // N3D))
        xs, ys, zs = self.xV[ic], self.yV[ir], self.zV[is_]
        surf_kw = dict(shade=False, rstride=1, cstride=1, antialiased=False,
                       alpha=self.scanAlpha)        # scan opacity (top-left)

        # axial plane (constant z)
        X, Y = np.meshgrid(xs, ys)
        ax.plot_surface(X, Y, np.full_like(X, self.zV[kA]),
                        facecolors=scanCmapObj(norm(self.scan3M[ir, ic, kA])),
                        **surf_kw)
        # sagittal plane (constant x)
        Yg, Zg = np.meshgrid(ys, zs, indexing="ij")
        ax.plot_surface(np.full_like(Yg, self.xV[kS]), Yg, Zg,
                        facecolors=scanCmapObj(norm(self.scan3M[ir, kS, is_])),
                        **surf_kw)
        # coronal plane (constant y)
        Xg, Zg2 = np.meshgrid(xs, zs, indexing="ij")
        ax.plot_surface(Xg, np.full_like(Xg, self.yV[kC]), Zg2,
                        facecolors=scanCmapObj(norm(self.scan3M[kC, ic, is_])),
                        **surf_kw)

        # colored plane outlines (locators)
        x0, x1 = self.xV[0], self.xV[-1]
        y0, y1 = self.yV[0], self.yV[-1]
        z0, z1 = self.zV[0], self.zV[-1]
        zA, xS, yC = self.zV[kA], self.xV[kS], self.yV[kC]
        edges = {
            VIEW_AXIAL: ([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                         [zA] * 5),
            VIEW_SAGITTAL: ([xS] * 5, [y0, y1, y1, y0, y0],
                            [z0, z0, z1, z1, z0]),
            VIEW_CORONAL: ([x0, x1, x1, x0, x0], [yC] * 5,
                           [z0, z0, z1, z1, z0]),
        }
        for orient, (ex, ey, ez) in edges.items():   # plane locators (always)
            ax.plot3D(ex, ey, ez, color=PLANE_COLORS[orient], lw=1.6)

        for beam in self.beams:        # IMRTP beam overlays
            color = beam.get("color", (0.2, 0.85, 0.9))
            for poly in beam["polylines"]:
                p = np.asarray(poly, dtype=float)
                ax.plot3D(p[:, 0], p[:, 1], p[:, 2], color=color, lw=1.0)

        if getattr(self, "uromtOverlay", None) is not None:   # urOMT 3-D overlay
            self._add_uromt_3d_mpl(ax)

        ax.set_box_aspect((abs(x1 - x0) or 1, abs(y1 - y0) or 1,
                           abs(z1 - z0) or 1))
        ax.set_axis_off()
        view.label.setText(self._plane_slices_3d()[3])
        view.canvas.draw_idle()

    def _draw_crosshair(self, view):
        """(Re)create the crosshair artists for a freshly drawn view.

        When blitting is unsafe (macOS) the lines are non-animated so they are
        painted by the normal figure draw; elsewhere they are animated and
        blitted for a fast reposition without a full redraw."""
        kw = dict(color="#e8c542", lw=0.6, ls="--", alpha=0.7,
                  visible=self.showCrosshairs, animated=CROSSHAIR_BLIT_OK)
        view.xline = view.ax.axvline(0, **kw)
        view.yline = view.ax.axhline(0, **kw)
        self._position_crosshair(view)

    def _position_crosshair(self, view):
        """Move a view's crosshair to the other view types' last slices."""
        if view.xline is None or self.planC is None:
            return
        if view.orientation == VIEW_AXIAL:
            xv = self.xV[self.lastSlice[VIEW_SAGITTAL]]
            yv = self.yV[self.lastSlice[VIEW_CORONAL]]
        elif view.orientation == VIEW_SAGITTAL:
            xv = self.yV[self.lastSlice[VIEW_CORONAL]]
            yv = self.zV[self.lastSlice[VIEW_AXIAL]]
        else:
            xv = self.xV[self.lastSlice[VIEW_SAGITTAL]]
            yv = self.zV[self.lastSlice[VIEW_AXIAL]]
        view.xline.set_xdata([xv, xv])
        view.yline.set_ydata([yv, yv])

    # ----------------------------------------------------------- DVH tool ---
    def show_dvh_dialog(self):
        if self.planC is None or not self.planC.structure or not self.planC.dose:
            _show_info(
                self, "DVH", "A dose and at least one structure are required.")
            return
        # Non-modal (like the other tools): a modal exec_() hangs when the
        # viewer runs inside an integrated event loop (show() / %gui qt).
        dlg = DvhDialog(self.planC, self)
        self._toolWindows.append(dlg)
        dlg.show()

    def _view_with_orientation(self, orientation):
        """First *visible* view window showing the given orientation."""
        for wid in self.activeWins:
            if self.views[wid].orientation == orientation:
                return self.views[wid]
        return None

    def show_contour_dialog(self):
        """Open the CERR-style contouring tools (draw on the axial view)."""
        if self.planC is None or not self.planC.scan:
            _show_info(self, "Contouring",
                                              "Load a scan first.")
            return
        if self._view_with_orientation(VIEW_AXIAL) is None:
            _show_info(
                self, "Contouring",
                "Set one of the views to Axial first (right-click > View).")
            return
        if self.contourCtl is not None and self.contourCtl.isVisible():
            self.contourCtl.raise_()
            self.contourCtl.activateWindow()
            return
        dlg = ContourDialog(self)
        dlg.show()

    def show_reg_dialog(self):
        """Open the registration QA tool (Mirrorscope / Side-by-side /
        Alternate grid / Toggle), cf. the napari QA modes in cerr.viewer."""
        if self.planC is None or len(self.planC.scan) < 2:
            _show_info(
                self, "Registration QA",
                "At least two scans are required to compare.")
            return
        if self.regCtl is not None and self.regCtl.isVisible():
            self.regCtl.raise_()
            self.regCtl.activateWindow()
            return
        dlg = RegQaDialog(self)
        dlg.show()

    def show_imrtp_gui(self):
        """Open the IMRTP beamlet-dose GUI (cerr.imrtp.imrtp_gui) non-blocking."""
        if self.planC is None or not self.planC.scan:
            _show_info(self, "IMRTP",
                                              "Load a scan first.")
            return
        try:
            from cerr.imrtp.imrtp_gui import IMRTPGui
            win = IMRTPGui(self.planC, block=False, viewer=self)
        except Exception as e:  # noqa: BLE001
            _show_error(
                self, "IMRTP", f"Could not open the IMRTP GUI:\n{e}")
            return
        self._toolWindows.append(win)
        self.statusBar().showMessage(
            "IMRTP opened (shares this viewer's planC). After computing dose, "
            "use Tools > Refresh from planC to see it here.")

    def show_roe_gui(self):
        """Open the ROE outcomes-explorer GUI (cerr.roe.roe_gui) non-blocking."""
        try:
            from cerr.roe.roe_gui import launch as roe_launch
            win = roe_launch(self.planC)
        except Exception as e:  # noqa: BLE001
            _show_error(
                self, "ROE", f"Could not open the ROE GUI:\n{e}")
            return
        self._toolWindows.append(win)
        self.statusBar().showMessage(
            "ROE opened (shares this viewer's planC).")

    def show_uromt_dialog(self):
        """Open the urOMT (fluid transport) tool - runs the cerr.uromt pipeline
        on the longitudinal scans in planC using a structure as the ROI."""
        if self.planC is None or len(self.planC.scan) < 2:
            _show_info(self, "urOMT",
                       "urOMT needs at least two co-registered longitudinal "
                       "scans (time points) in planC.")
            return
        if not self.planC.structure:
            _show_info(self, "urOMT",
                       "Load or draw an ROI structure first.")
            return
        dlg = UROMTDialog(self)
        self._toolWindows.append(dlg)
        self._uromtDialog = dlg           # so set_uromt_overlay updates its colorbar
        dlg.show()

    def show_3d_volume(self):
        """Open the true-3D volume-visualization tool (Tools menu): GPU volume
        render of the scan with structure / dose / urOMT overlays, using the
        main viewer's transparency values."""
        if self.planC is None or not self.planC.scan:
            _show_info(self, "3D Visualization", "Load a scan first.")
            return
        if not HAS_PYVISTA:
            _show_info(self, "3D Visualization",
                       "3D volume rendering needs pyvista "
                       "(pip install pyvista pyvistaqt).")
            return
        dlg = Volume3DDialog(self)
        self._toolWindows.append(dlg)
        self._volume3dDialog = dlg        # for auto-refresh on main-GUI changes
        dlg.show()
        dlg.render_scene()

    def show_controls(self):
        _show_info(
            self, "Controls",
            "Mouse:\n"
            "  Scroll: change slice\n"
            "  Left / middle drag: pan\n"
            "  Right-drag up/down: zoom in/out\n"
            "  Drag a crosshair line: scroll the other views\n"
            "  Right-click: choose scan / dose / structures, draw ruler\n"
            "  Double-click: reset pan/zoom\n\n"
            "Keyboard:\n"
            "  Arrow keys (hovered view): change slice\n"
            "  X: toggle crosshairs     O: orientation labels\n"
            "  L: lock slices across views     R: reset pan/zoom\n\n"
            "Tools > DVH for dose-volume histograms.")

    def show_about(self):
        _show_info(
            self, "About pyCERR Viewer",
            "<b>pyCERR Viewer</b><br>"
            f"version {_pycerr_version()}<br><br>"
            "A CERR-style slice viewer built on pyCERR for visualizing "
            "scans, segmentations and radiotherapy dose.<br><br>"
            '<a href="https://github.com/cerr/pyCERR">'
            "https://github.com/cerr/pyCERR</a>",
            rich=True)


# ---------------------------------------------------------------------------#
#  DVH dialog
# ---------------------------------------------------------------------------#
