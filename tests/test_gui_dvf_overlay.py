"""Tests for the deformation-vector-field (DVF) display in pyCERR-GUI.

Two layers are covered:

* :func:`cerr.registration.register.getDvfGrid`, which samples a ``Deform`` on a
  regular grid - the grid twin of ``getDvfVectors``, which returns the same
  vectors scattered for Napari's ``Vectors`` layer. Exercised against the
  bundled radiomics phantom with a synthetic, analytically known DVF, so the
  tests are fully offline.
* ``PyCerrViewer.set_dvf_overlay`` / ``_draw_dvf_overlay``, the Registration QA
  overlay itself, driven with a lightweight stand-in ``self`` (no QApplication)
  onto a plain matplotlib axes.
"""
import os
import types

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cerr import datasets  # noqa: E402
from cerr import plan_container as pc  # noqa: E402
from cerr.registration import register  # noqa: E402
from cerr.registration.register import getDvfGrid  # noqa: E402

sitk = pytest.importorskip("SimpleITK")

phantom_dir = os.path.join(os.path.dirname(datasets.__file__),
                           'radiomics_phantom_dicom', 'pat_1')

#: the constant LPS displacement (mm) the fixture field carries
SHIFT_MM = (5.0, -3.0, 2.0)


def _writeConstantDvf(scanObj, dvfFile, shiftMM=SHIFT_MM):
    """A DVF that displaces every voxel by the same LPS-mm vector.

    A constant field makes every assertion below analytic: interpolation cannot
    change it, so whatever the sampler returns is exactly what the coordinate
    transform did to it - and its median IS that vector, which is what the
    median-subtraction test needs.
    """
    numRows, numCols, numSlcs = scanObj.getScanSize()
    xV, yV, zV = scanObj.getScanXYZVals()
    spacing = [10 * abs(xV[1] - xV[0]), 10 * abs(yV[0] - yV[1]),
               10 * abs(zV[1] - zV[0])]
    origin = [float(v) for v in scanObj.scanInfo[0].imagePositionPatient]

    dvfArr = np.zeros((numSlcs, numRows, numCols, 3), dtype=np.float32)
    for comp, val in enumerate(shiftMM):
        dvfArr[..., comp] = val
    img = sitk.GetImageFromArray(dvfArr, isVector=True)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])   # LPS-aligned
    sitk.WriteImage(img, dvfFile)


@pytest.fixture(scope="module")
def planCWithDvf(tmp_path_factory):
    planC = pc.loadDcmDir(phantom_dir)
    dvfFile = os.path.join(str(tmp_path_factory.mktemp("dvf")), "dvf.nii.gz")
    _writeConstantDvf(planC.scan[0], dvfFile)
    return pc.loadNiiVf(dvfFile, 0, planC)


@pytest.fixture(scope="module")
def deformNum(planCWithDvf):
    return len(planCWithDvf.deform) - 1


# ----------------------------------------------------------------- sampling --
def test_native_resolution_matches_the_scan_grid(planCWithDvf, deformNum):
    planC = planCWithDvf
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    comps, axes, info = getDvfGrid(planC.deform[deformNum], planC, 0,
                                   outputResV=[0, 0, 0])
    assert comps[0].shape == (len(yV), len(xV), len(zV))
    np.testing.assert_allclose(axes[0], xV, atol=1e-6)
    np.testing.assert_allclose(axes[1], yV, atol=1e-6)
    np.testing.assert_allclose(info["resolution"],
                               planC.scan[0].getScanSpacing(), atol=1e-6)


def test_requested_resolution_sets_the_grid_spacing(planCWithDvf, deformNum):
    planC = planCWithDvf
    comps, axes, info = getDvfGrid(planC.deform[deformNum], planC, 0,
                                   outputResV=[0.4, 0.4, 0.4])
    np.testing.assert_allclose(info["resolution"], [0.4, 0.4, 0.4])
    assert abs(axes[0][1] - axes[0][0]) == pytest.approx(0.4)
    assert abs(axes[1][1] - axes[1][0]) == pytest.approx(0.4)
    assert abs(axes[2][1] - axes[2][0]) == pytest.approx(0.4)
    assert comps[0].shape == (len(axes[1]), len(axes[0]), len(axes[2]))


def test_axes_keep_pycerr_orientation(planCWithDvf, deformNum):
    """x ascending, y DESCENDING, z ascending - pyCERR's convention, not
    Napari's row/col/slice frame."""
    _comps, axes, _info = getDvfGrid(planCWithDvf.deform[deformNum],
                                     planCWithDvf, 0, outputResV=[0.4] * 3)
    assert axes[0][1] > axes[0][0]
    assert axes[1][1] < axes[1][0]
    assert axes[2][1] > axes[2][0]


def test_auto_resolution_respects_the_point_budget(planCWithDvf, deformNum):
    planC = planCWithDvf
    comps, _axes, info = getDvfGrid(planC.deform[deformNum], planC, 0,
                                    outputResV=None, maxPoints=2000)
    assert comps[0].size <= 2000
    native = planC.scan[0].getScanSpacing()
    # coarsened isotropically from the native spacing
    factors = [r / n for r, n in zip(info["resolution"], native)]
    assert factors[0] == pytest.approx(factors[1])
    assert factors[0] == pytest.approx(factors[2])


def test_constant_field_is_recovered_in_virtual_cm(planCWithDvf, deformNum):
    """A constant LPS-mm shift must come back as the same constant vector in
    pyCERR virtual cm - same magnitude, and rotated the way the scan is."""
    comps, _axes, info = getDvfGrid(planCWithDvf.deform[deformNum],
                                    planCWithDvf, 0, outputResV=[0.4] * 3)
    valid = info["valid"]
    assert valid.any()
    mag = np.sqrt(sum(np.asarray(c, dtype=float) ** 2 for c in comps))
    expected = np.linalg.norm(SHIFT_MM) / 10.0        # mm -> cm
    np.testing.assert_allclose(mag[valid], expected, rtol=1e-4)
    # the phantom is axial HFS: virtual x is DICOM x, virtual y is -y
    np.testing.assert_allclose(np.abs(comps[1][valid]),
                               abs(SHIFT_MM[0]) / 10.0, rtol=1e-4)
    np.testing.assert_allclose(np.abs(comps[0][valid]),
                               abs(SHIFT_MM[1]) / 10.0, rtol=1e-4)
    np.testing.assert_allclose(np.abs(comps[2][valid]),
                               abs(SHIFT_MM[2]) / 10.0, rtol=1e-4)


def test_grid_agrees_with_getdvfvectors(planCWithDvf, deformNum):
    """The grid sampler and the scattered one are two views of one field: at
    the same resolution they must report the same displacements."""
    planC = planCWithDvf
    res = [0.4, 0.4, 0.4]
    comps, _axes, info = getDvfGrid(planC.deform[deformNum], planC, 0,
                                    outputResV=res)
    vectors = register.getDvfVectors(planC.deform[deformNum], planC, 0,
                                     outputResV=res)
    # getDvfVectors carries (yDeform, xDeform, zDeform) in the same virtual cm
    scattered = np.linalg.norm(vectors[:, 1, :], axis=1)
    gridMag = np.sqrt(sum(np.asarray(c, dtype=float) ** 2 for c in comps))
    dense = gridMag[info["valid"]]
    assert dense.size == np.isfinite(scattered).sum()
    np.testing.assert_allclose(np.median(dense),
                               np.nanmedian(scattered), rtol=1e-4)


def test_median_subtraction_removes_a_bulk_shift(planCWithDvf, deformNum):
    """The fixture field IS a pure bulk shift, so subtracting its median must
    leave nothing - which is exactly what the control is for."""
    planC = planCWithDvf
    comps, _axes, info = getDvfGrid(planC.deform[deformNum], planC, 0,
                                    outputResV=[0.4] * 3, subtractMedian=True)
    valid = info["valid"]
    mag = np.sqrt(sum(np.asarray(c, dtype=float) ** 2 for c in comps))
    np.testing.assert_allclose(mag[valid], 0.0, atol=1e-6)
    assert np.linalg.norm(info["median"]) == pytest.approx(
        np.linalg.norm(SHIFT_MM) / 10.0, rel=1e-4)


def test_structure_and_surface_restrict_the_vectors(planCWithDvf, deformNum):
    planC = planCWithDvf
    if not planC.structure:
        pytest.skip("phantom has no structures")
    whole = getDvfGrid(planC.deform[deformNum], planC, 0,
                       outputResV=[0.3] * 3)[2]["nVectors"]
    solid = getDvfGrid(planC.deform[deformNum], planC, 0,
                       outputResV=[0.3] * 3, structNum=0)[2]
    surf = getDvfGrid(planC.deform[deformNum], planC, 0,
                      outputResV=[0.3] * 3, structNum=0, surfFlag=True)[2]
    assert 0 < surf["nVectors"] < solid["nVectors"] < whole


# ------------------------------------------------------------------ overlay --
mw = pytest.importorskip("cerr.viewer.pycerr_gui.main_window")
from cerr.viewer.pycerr_gui.common import (  # noqa: E402
    VIEW_AXIAL, VIEW_SAGITTAL, VIEW_CORONAL)

PyCerrViewer = mw.PyCerrViewer


def _stubViewer(planC, slices=None):
    """A stand-in ``self`` carrying only what the DVF overlay methods read."""
    xV, yV, zV = planC.scan[0].getScanXYZVals()
    nR, nC, nS = planC.scan[0].getScanSize()
    views = {w: types.SimpleNamespace(orientation=o, is3d=False)
             for w, o in (("A", VIEW_AXIAL), ("B", VIEW_SAGITTAL),
                          ("C", VIEW_CORONAL))}
    v = types.SimpleNamespace(
        planC=planC, scanNum=0, dvfOverlay=None, uromtOverlay=None,
        xV=xV, yV=yV, zV=zV, views=views,
        slices=(slices or {"A": nS // 2, "B": nC // 2, "C": nR // 2}),
        _dvfGridCache={},
        _DVF_COLOR_LABELS=PyCerrViewer._DVF_COLOR_LABELS,
        _DVF_COLOR_COMPS=PyCerrViewer._DVF_COLOR_COMPS,
        _refresh_uromt_views=lambda: None,
        _uromtHeadCeiling=lambda: 0.2)
    for name in ("set_dvf_overlay", "clear_dvf_overlay", "_dvf_slice_grid",
                 "_draw_dvf_overlay", "_uromt_3d_geometry"):
        setattr(v, name, getattr(PyCerrViewer, name).__get__(v))
    return v


def _quivers(ax):
    from matplotlib.quiver import Quiver
    return [c for c in ax.collections if isinstance(c, Quiver)]


def _drawn(viewer, winId="A"):
    """The Quiver artists the overlay puts on a fresh axes for one view."""
    fig, ax = plt.subplots()
    hV, vV = viewer.xV, viewer.yV
    viewer._draw_dvf_overlay(winId, ax, [hV[0], hV[-1], vV[-1], vV[0]])
    fig.canvas.draw()
    q = _quivers(ax)
    plt.close(fig)
    return q


def _shaftLengths(q):
    """Length of each drawn arrow in data units (``scale_units='xy'`` puts the
    quiver's own U/V straight into the axes' coordinates)."""
    return np.hypot(q.U, q.V) / q.scale


def test_overlay_builds_and_draws_a_quiver(planCWithDvf, deformNum):
    v = _stubViewer(planCWithDvf)
    ov = v.set_dvf_overlay(deformNum, resolution=[0.4] * 3)
    assert ov is not None and ov["view"] == "dvf"
    np.testing.assert_allclose(ov["resolution"], [0.4] * 3)
    assert v.dvfOverlay is ov
    for winId in ("A", "B", "C"):
        assert len(_drawn(v, winId)) == 1, "no arrows in view %s" % winId
    v.clear_dvf_overlay()
    assert v.dvfOverlay is None


def test_true_scale_draws_arrows_at_the_real_displacement(planCWithDvf,
                                                          deformNum):
    """At true scale the drawn arrow must be as long as the displacement it
    stands for - that is the whole claim the checkbox makes."""
    v = _stubViewer(planCWithDvf)
    v.set_dvf_overlay(deformNum, resolution=[0.5] * 3, trueScale=True)
    q = _drawn(v)[0]
    # in-plane (x, y) part of the constant field, in cm
    expected = np.hypot(SHIFT_MM[0], SHIFT_MM[1]) / 10.0
    drawn = _shaftLengths(q)
    assert drawn.size > 10
    np.testing.assert_allclose(np.median(drawn), expected, rtol=1e-3)


def test_length_scale_multiplies_the_drawn_arrows(planCWithDvf, deformNum):
    v = _stubViewer(planCWithDvf)

    def extent(scale):
        v.set_dvf_overlay(deformNum, resolution=[0.5] * 3, trueScale=True,
                          lengthScale=scale)
        return float(np.median(_shaftLengths(_drawn(v)[0])))

    assert extent(2.0) == pytest.approx(2 * extent(1.0), rel=1e-3)


def test_subsample_thins_the_drawn_arrows(planCWithDvf, deformNum):
    v = _stubViewer(planCWithDvf)
    v.set_dvf_overlay(deformNum, resolution=[0.2] * 3, subsample=1)
    dense = len(_drawn(v)[0].get_offsets())
    v.set_dvf_overlay(deformNum, resolution=[0.2] * 3, subsample=3)
    thin = len(_drawn(v)[0].get_offsets())
    assert thin < dense / 4          # thinned in both in-plane directions


def test_colour_channels_are_in_mm(planCWithDvf, deformNum):
    """The colour scale reports millimetres - the unit a registration is judged
    in, and the one the Napari layer's features use - while the geometry stays
    in the viewer's centimetres."""
    v = _stubViewer(planCWithDvf)
    ov = v.set_dvf_overlay(deformNum, resolution=[0.5] * 3, colorBy="length")
    assert ov["label"] == "length (mm)"
    assert ov["colorRange"][1] == pytest.approx(
        10.0 * ov["vrange"][1], rel=1e-6)      # p99 == max on a constant field
    assert not ov.get("diverging")

    ov = v.set_dvf_overlay(deformNum, resolution=[0.5] * 3, colorBy="dz")
    assert ov["colorRange"][1] == pytest.approx(abs(SHIFT_MM[2]), rel=1e-3)


def test_signed_component_colouring_is_symmetric_and_diverging(planCWithDvf,
                                                               deformNum):
    v = _stubViewer(planCWithDvf)
    ov = v.set_dvf_overlay(deformNum, resolution=[0.5] * 3,
                           colorBy="dz_signed")
    assert ov["diverging"] is True
    lo, hi = ov["colorRange"]
    assert lo == pytest.approx(-hi)
    assert hi == pytest.approx(abs(SHIFT_MM[2]), rel=1e-3)
    assert ov["label"].startswith("dz")


def test_median_subtraction_is_reported_on_the_label(planCWithDvf, deformNum):
    v = _stubViewer(planCWithDvf)
    ov = v.set_dvf_overlay(deformNum, resolution=[0.5] * 3,
                           subtractMedian=True)
    # a pure bulk shift leaves nothing behind, and the label says why
    assert ov["label"].endswith("- median")
    assert ov["vrange"][1] < 1e-6
    assert np.linalg.norm(ov["info"]["median"]) > 0


def test_arrow_length_is_never_clipped_by_the_colour_range(planCWithDvf,
                                                           deformNum):
    """``vrange`` scales/clips arrow LENGTH, so it must carry the true maximum
    displacement even though the colour range is a robust percentile."""
    v = _stubViewer(planCWithDvf)
    ov = v.set_dvf_overlay(deformNum, resolution=[0.5] * 3)
    mag = np.sqrt(sum(np.asarray(c, dtype=float) ** 2 for c in ov["comps"]))
    assert ov["vrange"][1] == pytest.approx(float(mag.max()))


def test_off_grid_slice_falls_back_to_the_nearest_sampled_plane(planCWithDvf,
                                                                deformNum):
    """The sampling grid is coarser than the scan, so most slices fall between
    sampled planes; the overlay shows the nearest one rather than blinking off.
    """
    planC = planCWithDvf
    nS = planC.scan[0].getScanSize()[2]
    v = _stubViewer(planC, slices={"A": 0, "B": 0, "C": 0})
    v.set_dvf_overlay(deformNum, resolution=[0.5] * 3)
    for k in range(min(6, nS)):
        v.slices["A"] = k
        assert len(_drawn(v, "A")) == 1, "slice %d drew nothing" % k


def test_3d_geometry_uses_the_overlays_own_axes(planCWithDvf, deformNum):
    """The DVF grid is its own resampling, so its 3-D points must come from
    ``ov['axes']`` - reading the scan axes would misplace every arrow."""
    v = _stubViewer(planCWithDvf)
    ov = v.set_dvf_overlay(deformNum, resolution=[0.5] * 3)
    geom = v._uromt_3d_geometry(ov=ov)
    assert geom is not None and "vectors" in geom
    pts = geom["vectors"]["points"]
    xg, yg, zg = ov["axes"]
    assert np.isin(pts[:, 0], xg).all()
    assert np.isin(pts[:, 1], yg).all()
    assert np.isin(pts[:, 2], zg).all()


def test_missing_or_invalid_field_index_is_a_no_op(planCWithDvf):
    v = _stubViewer(planCWithDvf)
    assert v.set_dvf_overlay(99) is None
    assert v.set_dvf_overlay(-1) is None
    assert v.dvfOverlay is None


# ------------------------------------------------------- field-type filter --
def test_only_vector_field_deforms_are_offered(planCWithDvf, deformNum):
    """A planC.deform may be a rigid transform or a set of B-spline
    coefficients, neither of which has per-voxel displacements. Those must not
    be offered as something to draw, and must not be drawable if asked for."""
    from cerr.dataclasses import deform as cerrDeform

    field = planCWithDvf.deform[deformNum]
    assert field.deformOutFileType in cerrDeform.DVF_FILE_TYPES
    assert cerrDeform.hasDvfMatrix(field)

    rigid = cerrDeform.Deform()
    rigid.deformOutFileType = "rigid"
    rigid.deformParams = {"rigidMatrix": np.eye(4).tolist()}
    assert not cerrDeform.hasDvfMatrix(rigid)

    bspline = cerrDeform.Deform()          # plastimatch coefficients: no field
    bspline.deformOutFileType = ""
    assert not cerrDeform.hasDvfMatrix(bspline)

    # the overlay refuses one even when a script asks for it by index
    planCWithDvf.deform.append(rigid)
    try:
        v = _stubViewer(planCWithDvf)
        assert v.set_dvf_overlay(len(planCWithDvf.deform) - 1) is None
        assert v.dvfOverlay is None
        assert v.set_dvf_overlay(deformNum, resolution=[0.5] * 3) is not None
    finally:
        planCWithDvf.deform.pop()
