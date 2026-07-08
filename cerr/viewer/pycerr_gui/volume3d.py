"""Volume3DDialog: volume-rendered 3D view."""
from cerr.viewer.pycerr_gui.common import *  # noqa: F401,F403

class Volume3DDialog(QtWidgets.QDialog):
    """True-3D visualization (View -> 3D Visualization): a GPU volume render of
    the scan with structure surfaces, isodose surfaces and the urOMT overlay in
    one interactive scene. Scan, dose and structure opacity are set with the
    dialog's own sliders, independent of the main viewer's 2-D settings. A
    'Clip box' shows a cube around the scan volume whose faces can be dragged
    to cut the scene open and look inside. Left-drag rotates (or pans when
    'Pan mode' is checked) and scroll zooms; a triad of arrows labelled
    L (Left), A (Anterior), S (Superior) sits in the corner and the camera is
    set with Superior up. Click 'Refresh' after changing settings.
    """

    def __init__(self, viewer):
        super().__init__(viewer)
        self.viewer = viewer
        self.setWindowTitle("3D Visualization (volume render)")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(760, 700)
        self._panMode = False           # left-drag pans (else rotates)
        self._camStyle = None           # persistent trackball interactor style
        self._panObsInstalled = False
        self._clipPlanes = None         # vtkPlanes from the clip-box widget
        self._clipBounds = None         # last clip-box bounds (for restore)
        self._hasRendered = False       # keep the camera across rebuilds
        lay = QtWidgets.QVBoxLayout(self)
        self.plotter = QtInteractor(self)          # pyvista/VTK GPU widget
        lay.addWidget(self.plotter, 1)

        # per-dialog opacities - independent of the main viewer's 2D settings
        self.scanOpacity = 0.5
        self.doseOpacity = 0.5
        self.structOpacity = 0.45
        opRow = QtWidgets.QHBoxLayout()
        self.scanAlphaSlider = self._add_opacity_slider(
            opRow, "Scan:", self.scanOpacity, self._on_scan_alpha,
            "Opacity of the volume-rendered scan (this 3D view only)")
        self.doseAlphaSlider = self._add_opacity_slider(
            opRow, "Dose:", self.doseOpacity, self._on_dose_alpha,
            "Opacity of the isodose surfaces (this 3D view only)")
        self.structAlphaSlider = self._add_opacity_slider(
            opRow, "Structures:", self.structOpacity, self._on_struct_alpha,
            "Opacity of structure surfaces in this 3D view only. The 2D "
            "contour lines are unaffected and always drawn fully opaque.")
        lay.addLayout(opRow)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(
            "Left-drag rotate, scroll zoom."))
        row.addStretch(1)
        self.outlineCheck = QtWidgets.QCheckBox("Scan box")
        self.outlineCheck.setChecked(True)
        self.outlineCheck.setToolTip(
            "Show the dotted outline of the original scan volume.")
        self.outlineCheck.toggled.connect(self._on_outline_toggled)
        row.addWidget(self.outlineCheck)
        row.addWidget(QtWidgets.QLabel("Resolution:"))
        self.resCombo = QtWidgets.QComboBox()
        self.resCombo.addItems(["1/4", "1/2", "3/4", "Full"])
        self.resCombo.setCurrentText("1/2")
        self.resCombo.setToolTip(
            "Display resolution for the isotropically-resampled scan volume. "
            "'Full' uses the smallest native voxel spacing; a fraction gives "
            "that spacing divided by the fraction (e.g. '1/2' = twice the "
            "smallest spacing). Lower fractions load and update faster.")
        self.resCombo.currentTextChanged.connect(lambda *_: self.render_scene())
        row.addWidget(self.resCombo)
        self.clipCheck = QtWidgets.QCheckBox("Clip box")
        self.clipCheck.setToolTip(
            "Show a cube around the volume; drag its faces to cut the scene "
            "open and view the inside from that face. Unchecking hides the "
            "cube without changing the cut; re-enable and drag the faces "
            "back out to the dotted scan box to restore the full volume.")
        self.clipCheck.toggled.connect(self._on_clip_toggled)
        row.addWidget(self.clipCheck)
        self.panCheck = QtWidgets.QCheckBox("Pan mode")
        self.panCheck.setToolTip(
            "Left-drag pans the scene instead of rotating it "
            "(scroll still zooms). Uncheck to rotate.")
        self.panCheck.toggled.connect(self._on_pan_toggled)
        row.addWidget(self.panCheck)
        refresh = QtWidgets.QPushButton("Refresh")
        refresh.setToolTip("Rebuild the 3-D scene from the current planC and "
                           "the main viewer's transparency settings.")
        refresh.clicked.connect(self.render_scene)
        closeBtn = QtWidgets.QPushButton("Close")
        closeBtn.clicked.connect(self.close)
        row.addWidget(refresh)
        row.addWidget(closeBtn)
        lay.addLayout(row)

        # debounce: coalesce rapid main-GUI changes (W/L drag, opacity slider,
        # structure toggles) into a single rebuild shortly after they settle.
        self._refreshTimer = QtCore.QTimer(self)
        self._refreshTimer.setSingleShot(True)
        self._refreshTimer.setInterval(150)
        self._refreshTimer.timeout.connect(self.render_scene)

    @staticmethod
    def _add_opacity_slider(lay, label, value, slot, tip):
        lay.addWidget(QtWidgets.QLabel(label))
        sld = QtWidgets.QSlider(Qt.Horizontal)
        sld.setRange(0, 100)
        sld.setValue(int(round(value * 100)))
        sld.setToolTip(tip)
        sld.valueChanged.connect(slot)
        lay.addWidget(sld, 1)
        return sld

    def _on_scan_alpha(self, val):
        self.scanOpacity = val / 100.0
        self.apply_style("scan")

    def apply_style(self, layer):
        """Restyle a layer in place (colormap / window / opacity) without a
        re-resample or re-upload; rebuilds only when the actor must be added
        or removed, or on any error. Called by the main viewer when its scan/
        dose colormap, window or opacity changes."""
        pl = self.plotter
        try:
            if layer == "scan":
                actor = pl.actors.get("scan")
                if actor is None or self.scanOpacity <= 0.01:
                    self._refreshTimer.start()     # add/remove the volume
                    return
                prop = actor.GetProperty()
                prop.SetColor(self._scan_color_tf())
                prop.SetScalarOpacity(self._scan_opacity_tf())
                pl.render()
            else:
                actor = pl.actors.get("dose")
                if actor is None:
                    if self.doseOpacity > 0:
                        self._refreshTimer.start()
                    return
                v = self.viewer
                cbLo, cbHi = v.colorbar.cbarRange
                actor.mapper.lookup_table = pv.LookupTable(
                    cmap=v.colorbar.cmapName,
                    scalar_range=(cbLo, max(cbHi, cbLo + 1e-6)))
                actor.mapper.scalar_range = (cbLo, max(cbHi, cbLo + 1e-6))
                actor.GetProperty().SetOpacity(
                    min(max(self.doseOpacity, 0.0), 0.6))
                pl.render()
        except Exception:  # noqa: BLE001
            self._refreshTimer.start()             # fall back to a full rebuild

    def _clim(self):
        v = self.viewer
        vmin = v.windowCenter - v.windowWidth / 2.0
        vmax = v.windowCenter + v.windowWidth / 2.0
        return vmin, max(vmax, vmin + 1e-6)

    def _scan_opacity_ramp(self):
        """256-sample opacity ramp (0..1) over clim: the sigmoid TF scaled by
        scan opacity, with samples outside the scan display (cyan) range zeroed
        so the volume honors it."""
        vmin, vmax = self._clim()
        ramp = np.clip(pv.opacity_transfer_function("sigmoid", 256)
                       .astype(float) * float(self.scanOpacity),
                       0.0, 255.0) / 255.0
        dLo, dHi = self.viewer._scan_disp_range()
        if np.isfinite(dLo) or np.isfinite(dHi):
            scal = vmin + (vmax - vmin) * np.arange(len(ramp)) / (len(ramp) - 1)
            ramp = ramp.copy()
            ramp[(scal < dLo) | (scal > dHi)] = 0.0
        return ramp

    def _scan_opacity_tf(self):
        """vtkPiecewiseFunction from :meth:`_scan_opacity_ramp` over clim."""
        import vtk
        vmin, vmax = self._clim()
        span = vmax - vmin
        ramp = self._scan_opacity_ramp()
        otf = vtk.vtkPiecewiseFunction()
        n = len(ramp)
        for i, a in enumerate(ramp):
            otf.AddPoint(vmin + span * i / (n - 1), float(a))
        return otf

    def _scan_color_tf(self):
        """vtkColorTransferFunction from the viewer's scan colormap over clim."""
        import vtk
        vmin, vmax = self._clim()
        span = max(vmax - vmin, 1e-6)
        cmap = plt.get_cmap(self.viewer.scanCmap)
        ctf = vtk.vtkColorTransferFunction()
        n = 256
        for i in range(n):
            r, g, b, _ = cmap(i / (n - 1))
            ctf.AddRGBPoint(vmin + span * i / (n - 1),
                            float(r), float(g), float(b))
        return ctf

    def _on_dose_alpha(self, val):
        """Update the isodose-surface opacity in place when possible."""
        self.doseOpacity = val / 100.0
        pl = self.plotter
        try:
            actor = pl.actors.get("dose")
            if actor is not None:
                actor.GetProperty().SetOpacity(
                    min(max(self.doseOpacity, 0.0), 0.6))
                pl.render()
            elif self.doseOpacity > 0:
                self.render_scene()
        except Exception:  # noqa: BLE001
            self.render_scene()

    # ------------------------------------------------------------ clip box --
    def _on_clip_toggled(self, on):
        """Show/hide the draggable clip cube. The cut itself is untouched:
        hiding the cube keeps the clipped display exactly as it is, and
        re-enabling shows the cube at the same position. To restore the full
        volume, drag the cube's faces back out to the dotted scan box."""
        pl = self.plotter
        try:
            if on:
                self._add_clip_widget()
            else:
                pl.clear_box_widgets()
                pl.render()
        except Exception:  # noqa: BLE001
            pass

    def _add_clip_widget(self):
        """(Re)create the clip-box widget, restoring the last box position and
        applying its clip."""
        # add_box_widget fires the callback once with the full-size box, which
        # would overwrite the remembered position - save it first
        prevBounds = getattr(self, "_clipBounds", None)
        self.plotter.clear_box_widgets()     # never accumulate stale widgets
        wdg = self.plotter.add_box_widget(
            callback=self._on_clip_box, use_planes=True,
            rotation_enabled=False, factor=1.0, color="white",
            pass_widget=True, bounds=self._scan_bounds())
        if prevBounds is not None:
            import vtk
            wdg.PlaceWidget(prevBounds)
            planes = vtk.vtkPlanes()
            wdg.GetPlanes(planes)
            self._on_clip_box(planes, wdg)
        self.plotter.render()

    def _on_clip_box(self, planes, widget=None):
        """Box-widget callback: crop the scene at the box faces. The box is
        confined to the original scan bounds (the dotted reference box) - a
        face dragged outside snaps back to the scan face. The box geometry is
        remembered so it can be restored after being hidden or rebuilt."""
        try:
            if widget is not None:
                import vtk
                pd = vtk.vtkPolyData()
                widget.GetPolyData(pd)
                b = list(pd.GetBounds())
                sb = self._scan_bounds()
                if sb is not None:
                    clamped = [max(b[0], sb[0]), min(b[1], sb[1]),
                               max(b[2], sb[2]), min(b[3], sb[3]),
                               max(b[4], sb[4]), min(b[5], sb[5])]
                    for i in (0, 2, 4):     # degenerate axis: full extent
                        if clamped[i + 1] <= clamped[i]:
                            clamped[i], clamped[i + 1] = sb[i], sb[i + 1]
                    if not np.allclose(clamped, b):
                        widget.PlaceWidget(clamped)
                        planes = vtk.vtkPlanes()
                        widget.GetPlanes(planes)
                        b = clamped
                self._clipBounds = list(b)
        except Exception:  # noqa: BLE001
            pass
        self._clipPlanes = planes
        self._apply_clip_planes(planes)

    def _scan_bounds(self):
        """Bounds of the ORIGINAL scan volume (also the clip-box limits)."""
        try:
            xA, yA, zA = self.viewer._scan_grid_geometry()[:3]
            self._scanBounds = [float(xA[0]), float(xA[-1]),
                                float(yA[0]), float(yA[-1]),
                                float(zA[0]), float(zA[-1])]
        except Exception:  # noqa: BLE001
            pass
        return getattr(self, "_scanBounds", None)

    def _on_outline_toggled(self, on):
        """Show/hide the dotted original-scan outline without a rebuild."""
        try:
            if on:
                self._add_scan_outline()
            else:
                self.plotter.remove_actor("scan_outline")
            self.plotter.render()
        except Exception:  # noqa: BLE001
            pass

    def _add_scan_outline(self):
        """Dotted outline of the ORIGINAL scan volume, shown as a size
        reference (never clipped by the clip box)."""
        if not self.outlineCheck.isChecked():
            return
        try:
            b = self._scan_bounds()
            if b is None:
                return
            # dotted: sample points along the 12 box edges
            corners = np.array(
                [[b[i], b[2 + j], b[4 + k]]
                 for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            edges = [(0, 1), (2, 3), (4, 5), (6, 7),    # z edges
                     (0, 2), (1, 3), (4, 6), (5, 7),    # y edges
                     (0, 4), (1, 5), (2, 6), (3, 7)]    # x edges
            t = np.linspace(0.0, 1.0, 36)[:, None]
            pts = np.vstack([corners[a] + (corners[c] - corners[a]) * t
                             for a, c in edges])
            self.plotter.add_points(
                pts, color="#bbbbbb", point_size=2.0,
                render_points_as_spheres=False, pickable=False,
                name="scan_outline")
        except Exception:  # noqa: BLE001
            pass

    def _apply_clip_planes(self, planes):
        """Apply (or clear, with None) the clip-box planes to all mappers -
        the scan volume, structure surfaces and isodose surfaces alike.

        vtkBoxWidget.GetPlanes returns face planes with outward normals, but
        mapper clipping planes cut away the half-space behind the normal, so
        the normals are flipped to keep what is inside the box."""
        try:
            import vtk
            coll = None
            if planes is not None:
                coll = vtk.vtkPlaneCollection()
                for i in range(planes.GetNumberOfPlanes()):
                    p = planes.GetPlane(i)
                    n = p.GetNormal()
                    q = vtk.vtkPlane()
                    q.SetOrigin(p.GetOrigin())
                    q.SetNormal(-n[0], -n[1], -n[2])
                    coll.AddItem(q)
            clipNames = ("scan", "struct", "dose", "uromt")
            for name, actor in list(self.plotter.actors.items()):
                # only cut the data actors; the dotted reference box, the
                # widget's own handles and other helpers stay visible
                if name == "scan_outline" \
                        or not str(name).startswith(clipNames):
                    continue
                mapper = getattr(actor, "GetMapper", lambda: None)()
                if mapper is None:
                    continue
                if coll is None:
                    mapper.RemoveAllClippingPlanes()
                else:
                    mapper.SetClippingPlanes(coll)
            self.plotter.render()
        except Exception:  # noqa: BLE001
            pass

    def request_refresh(self):
        """Schedule a (debounced) rebuild - called by the main viewer when the
        window/level, transparency, structure or dose selection changes."""
        if self.isVisible():
            self._refreshTimer.start()

    def _on_struct_alpha(self, val):
        """Set the 3D structure-surface opacity. Updates the existing surface
        actors in place (cheap) rather than rebuilding the whole scene; falls
        back to a full rebuild if the actors are not present yet."""
        self.structOpacity = val / 100.0
        pl = self.plotter
        try:
            actors = pl.actors
            updated = False
            for strNum in self.viewer._checked_structs():
                actor = actors.get(f"struct{strNum}")
                if actor is not None:
                    actor.GetProperty().SetOpacity(self.structOpacity)
                    updated = True
            if updated:
                pl.render()
            else:
                self.render_scene()
        except Exception:  # noqa: BLE001
            self.render_scene()

    def _on_pan_toggled(self, on):
        """Toggle left-drag between pan and rotate.

        Subclassing the interactor style and overriding ``OnLeftButtonDown`` did
        not work (VTK does not dispatch button virtuals to Python overrides),
        and ``enable_image_style`` maps left-drag to window/level, not pan. So we
        keep a normal trackball-camera style and add a high-priority observer on
        the interactor's LeftButtonPress: when pan mode is on it calls
        ``StartPan`` before the default handler runs, so the default
        ``StartRotate`` early-returns (a state is already set) and left-drag
        pans. Middle-drag still pans, scroll still zooms."""
        self._panMode = bool(on)
        try:
            import vtk
            iren = getattr(self.plotter, "iren", None)
            iren = getattr(iren, "interactor", iren)
            if iren is None:
                return
            if getattr(self, "_camStyle", None) is None:
                self._camStyle = vtk.vtkInteractorStyleTrackballCamera()
            # ensure our style is the active one (pyvista may swap it on render)
            iren.SetInteractorStyle(self._camStyle)
            if not getattr(self, "_panObsInstalled", False):
                def _press(_obj, _evt):
                    if self._panMode:
                        self._camStyle.StartPan()

                def _release(_obj, _evt):
                    if self._panMode:
                        self._camStyle.EndPan()
                # priority 10 => runs before the style's own handler
                iren.AddObserver("LeftButtonPressEvent", _press, 10.0)
                iren.AddObserver("LeftButtonReleaseEvent", _release, 10.0)
                self._panObsInstalled = True
        except Exception:  # noqa: BLE001
            pass

    def _add_orientation_marker(self):
        """Corner triad of arrows labelled by patient direction (e.g. L/A/S for
        an axial scan).

        Arrows point along the pyCERR virtual axes; each label comes from the
        base scan's actual orientation, so the triad is correct for any
        acquisition. The Z arrow is flipped (scale z by -1) so it points to
        world -z, and is labelled with that axis's -z-end direction. The widget
        lives on the interactor and survives ``clear()``; the labels are
        refreshed in place when the scan orientation changes."""
        pl = self.plotter
        # label each arrow by its true patient direction; the Z arrow is
        # flipped to world -z, so it gets the label at the -z end.
        xl = self.viewer._axis_anatomy("x")[0]
        yl = self.viewer._axis_anatomy("y")[0]
        zl = self.viewer._axis_anatomy("z")[1]
        triad = (xl, yl, zl)
        if getattr(self, "_orientMarker", None) is not None:
            if getattr(self, "_orientTriad", None) == triad:
                return
            axes = getattr(self, "_orientAxes", None)
            if axes is not None:                 # relabel the vtkAxesActor
                axes.SetXAxisLabelText(xl)
                axes.SetYAxisLabelText(yl)
                axes.SetZAxisLabelText(zl)
            else:                                # fallback triad -> re-add
                try:
                    pl.add_axes(xlabel=xl, ylabel=yl, zlabel=zl, color="white")
                except Exception:  # noqa: BLE001
                    pass
            self._orientTriad = triad
            return
        try:
            import vtk
            axes = vtk.vtkAxesActor()
            axes.SetXAxisLabelText(xl)
            axes.SetYAxisLabelText(yl)
            axes.SetZAxisLabelText(zl)
            flip = vtk.vtkTransform()
            flip.Scale(1.0, 1.0, -1.0)          # Z arrow -> world -z
            axes.SetUserTransform(flip)
            # lighting off so the negative-scale normal flip can't darken arrows
            for prop in (axes.GetXAxisShaftProperty(), axes.GetXAxisTipProperty(),
                         axes.GetYAxisShaftProperty(), axes.GetYAxisTipProperty(),
                         axes.GetZAxisShaftProperty(), axes.GetZAxisTipProperty()):
                prop.LightingOff()
            for cap in (axes.GetXAxisCaptionActor2D(),
                        axes.GetYAxisCaptionActor2D(),
                        axes.GetZAxisCaptionActor2D()):
                cap.GetCaptionTextProperty().SetColor(1.0, 1.0, 1.0)
                cap.GetCaptionTextProperty().BoldOn()
            iren = getattr(pl, "iren", None)
            iren = getattr(iren, "interactor", iren)
            marker = vtk.vtkOrientationMarkerWidget()
            marker.SetOrientationMarker(axes)
            marker.SetInteractor(iren)
            marker.SetViewport(0.0, 0.0, 0.22, 0.22)
            marker.EnabledOn()
            marker.InteractiveOff()
            self._orientMarker = marker          # keep a reference alive
            self._orientAxes = axes              # for in-place relabelling
            self._orientTriad = triad
        except Exception:  # noqa: BLE001
            try:                                 # fallback: simple labelled triad
                pl.add_axes(xlabel=xl, ylabel=yl, zlabel=zl, color="white")
                self._orientMarker = True
                self._orientAxes = None
                self._orientTriad = triad
            except Exception:  # noqa: BLE001
                pass

    def closeEvent(self, event):
        if getattr(self.viewer, "_volume3dDialog", None) is self:
            self.viewer._volume3dDialog = None
        super().closeEvent(event)

    def _res_frac(self):
        """Selected resolution as a fraction of the native grid (1.0=Full)."""
        return {"1/4": 0.25, "1/2": 0.5, "3/4": 0.75}.get(
            self.resCombo.currentText(), 1.0)

    def render_scene(self):
        """(Re)build the 3-D scene, showing a busy cursor and blocking
        interaction with the dialog while the rebuild runs."""
        QtWidgets.QApplication.setOverrideCursor(Qt.BusyCursor)
        self.setEnabled(False)
        QtWidgets.QApplication.processEvents()   # show the busy state now
        try:
            self._build_scene()
        finally:
            self.setEnabled(True)
            QtWidgets.QApplication.restoreOverrideCursor()

    def _build_scene(self):
        """(Re)build the 3-D scene from the current planC + viewer transparency.
        Every actor is wrapped defensively so a failure of one (e.g. the GPU
        volume mapper) still leaves the rest of the scene usable."""
        v = self.viewer
        pl = self.plotter
        cam = None
        try:
            if self._hasRendered:      # keep the user's view across rebuilds
                cam = pl.camera_position
        except Exception:  # noqa: BLE001
            cam = None
        try:
            pl.clear()
            pl.set_background("black")
        except Exception:  # noqa: BLE001
            return
        resFrac = self._res_frac()
        # ---- scan as a GPU volume; opacity ramp scaled by the scan opacity ---
        try:
            if float(self.scanOpacity) > 0.01:
                xA, yA, zA, fR, fC, fS = v._scan_grid_geometry()
                scan = v.scan3M.astype(np.float32)
                if fR:
                    scan = scan[::-1, :, :]
                if fC:
                    scan = scan[:, ::-1, :]
                if fS:
                    scan = scan[:, :, ::-1]
                # Always render on isotropic voxels. 'Full' resolution = the
                # smallest native voxel spacing; a fraction f gives spacing
                # (smallest / f), e.g. '1/2' -> twice the smallest spacing.
                sMin = v._smallest_spacing(xA, yA, zA, scan.shape)
                if sMin > 0:
                    scan, xA, yA, zA = v._resample_volume_isotropic(
                        scan, xA, yA, zA, sMin / resFrac)
                grid = v._pv_volume(scan, xA, yA, zA)
                vmin = v.windowCenter - v.windowWidth / 2.0
                vmax = v.windowCenter + v.windowWidth / 2.0
                # sigmoid opacity ramp x scan opacity, zeroed outside the scan
                # display (cyan) range so the volume honors it (0..1 -> 0..255).
                op = self._scan_opacity_ramp() * 255.0
                pl.add_volume(grid, scalars="v", cmap=v.scanCmap,
                              clim=(vmin, max(vmax, vmin + 1e-6)), opacity=op,
                              shade=False, show_scalar_bar=False, name="scan")
        except Exception:  # noqa: BLE001
            pass
        # ---- structure surfaces (the panel's structure checklist) ------------
        try:
            for strNum in v._checked_structs():
                surf = v._pv_struct_mesh(strNum, resFrac)
                if surf is not None:
                    pl.add_mesh(surf, color=v._struct_color(strNum),
                                opacity=self.structOpacity, smooth_shading=True,
                                pickable=False, show_scalar_bar=False,
                                name=f"struct{strNum}")
        except Exception:  # noqa: BLE001
            pass
        # ---- isodose surfaces (panel dose, this dialog's dose opacity) -------
        try:
            if v.doseNum is not None and v.doseNum >= 0 \
                    and self.doseOpacity > 0:
                res = v._pv_dose_iso(v.doseNum, resFrac)
                if res is not None:
                    iso, _dmax = res
                    cbLo, cbHi = v.colorbar.cbarRange
                    pl.add_mesh(iso, cmap=v.colorbar.mplCmap,
                                clim=(cbLo, max(cbHi, cbLo + 1e-6)),
                                opacity=min(max(float(self.doseOpacity),
                                                0.0), 0.6),
                                pickable=False, show_scalar_bar=False,
                                name="dose")
        except Exception:  # noqa: BLE001
            pass
        # ---- IMRTP beam overlays (one combined polyline actor) ---------------
        try:
            if getattr(v, "beams", None):
                pts_list, conn, cellRGB, off = [], [], [], 0
                for beam in v.beams:
                    rgb = (np.asarray(beam.get("color", (0.2, 0.85, 0.9)))
                           * 255).astype(np.uint8)
                    for poly in beam["polylines"]:
                        p = np.asarray(poly, dtype=float)
                        if len(p) < 2:
                            continue
                        pts_list.append(p)
                        conn.append(np.concatenate(
                            ([len(p)], np.arange(off, off + len(p)))))
                        cellRGB.append(rgb)
                        off += len(p)
                if pts_list:
                    pd = pv.PolyData()
                    pd.points = np.vstack(pts_list)
                    pd.lines = np.concatenate(conn).astype(np.int64)
                    pd.cell_data["rgb"] = np.asarray(cellRGB, dtype=np.uint8)
                    pl.add_mesh(pd, scalars="rgb", rgb=True, line_width=2,
                                pickable=False, show_scalar_bar=False,
                                name="beams")
        except Exception:  # noqa: BLE001
            pass
        # ---- urOMT overlay (reuse the embedded-3D builder) -------------------
        try:
            if getattr(v, "uromtOverlay", None) is not None:
                v._add_uromt_3d_vtk(pl)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._add_scan_outline()
            self._add_orientation_marker()
            if cam is not None:        # rebuild: restore the previous camera
                pl.camera_position = cam
            else:                      # first render: frame the scene
                pl.reset_camera()
                # pyCERR virtual +z = Inferior, so Superior is -z: orient the
                # camera with Superior up (else the scene is upside-down S-I).
                try:
                    pl.set_viewup((0.0, 0.0, -1.0), reset=False, render=False)
                except TypeError:                # older pyvista signature
                    pl.set_viewup((0.0, 0.0, -1.0))
                except Exception:  # noqa: BLE001
                    pl.camera.up = (0.0, 0.0, -1.0)
            self._hasRendered = True
            # keep the interaction mode in sync with the Pan-mode checkbox
            self._on_pan_toggled(self.panCheck.isChecked())
            # a rebuild drops the clip-cube's visuals (a stale widget object
            # can linger in box_widgets): recreate it at the stored position,
            # which also re-applies the clip to the rebuilt actors. When the
            # cube is hidden the cut still persists - re-apply it directly.
            if self.clipCheck.isChecked():
                self._add_clip_widget()
            elif self._clipPlanes is not None:
                self._apply_clip_planes(self._clipPlanes)
            pl.render()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------#
#  Embedded (matplotlib/Qt) urOMT result viewer - renders a stored
#  planC.urOMT[idx] run (Eulerian maps, velocity / flux vectors, Lagrangian
#  pathlines) on ROI slices inside the Qt GUI, without launching napari.
# ---------------------------------------------------------------------------#
