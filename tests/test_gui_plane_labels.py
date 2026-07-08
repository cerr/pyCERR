"""Headless tests for orientation-aware view labels.

Covers ``PyCerrViewer._anatomical_plane``, which names each 2D view by the
anatomical plane it actually shows (from the base scan's ``cerrToDcmTransM``),
so a sagittally/coronally acquired scan is labelled correctly rather than by
its array slice axis. The method is pure geometry, exercised with a lightweight
stand-in ``self`` (no QApplication).
"""
import types
import numpy as np
import pytest

mw = pytest.importorskip("cerr.viewer.pycerr_gui.main_window")
PyCerrViewer = mw.PyCerrViewer
from cerr.viewer.pycerr_gui.common import (  # noqa: E402
    VIEW_AXIAL, VIEW_SAGITTAL, VIEW_CORONAL)


def _fake(cerrToDcm3x3):
    M = np.eye(4)
    M[:3, :3] = np.asarray(cerrToDcm3x3, dtype=float)
    scan = types.SimpleNamespace(cerrToDcmTransM=M)
    planC = types.SimpleNamespace(scan=[scan])
    return types.SimpleNamespace(
        planC=planC, scanNum=0,
        _VIEW_THRU_COL=PyCerrViewer._VIEW_THRU_COL,
        _DCM_AXIS_PLANE=PyCerrViewer._DCM_AXIS_PLANE,
        _DCM_AXIS_ENDS=PyCerrViewer._DCM_AXIS_ENDS)


def _planes(fake):
    m = PyCerrViewer._anatomical_plane
    return {v: m(fake, v) for v in (VIEW_AXIAL, VIEW_SAGITTAL, VIEW_CORONAL)}


def test_axial_acquisition_identity_labels():
    # virtual x->DICOM X (L), y->Y (A/P), z->Z (S/I): standard axial layout.
    fake = _fake([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    p = _planes(fake)
    assert p[VIEW_AXIAL] == VIEW_AXIAL
    assert p[VIEW_SAGITTAL] == VIEW_SAGITTAL
    assert p[VIEW_CORONAL] == VIEW_CORONAL


def test_sagittal_acquisition_relabels():
    # Matches the real Sag T2 test data: x->DICOM Y, y->DICOM Z, z->DICOM X.
    fake = _fake([[0, 0, 10], [10, 0, 0], [0, 10, 0]])
    p = _planes(fake)
    assert p[VIEW_AXIAL] == VIEW_SAGITTAL     # slice axis is L-R -> sagittal
    assert p[VIEW_SAGITTAL] == VIEW_CORONAL
    assert p[VIEW_CORONAL] == VIEW_AXIAL


def test_coronal_acquisition_relabels():
    # x->DICOM X (L-R), y->DICOM Z (S-I), z->DICOM Y (A-P): coronal stack.
    fake = _fake([[10, 0, 0], [0, 0, 10], [0, 10, 0]])
    p = _planes(fake)
    assert p[VIEW_AXIAL] == VIEW_CORONAL      # slice axis is A-P -> coronal
    assert p[VIEW_SAGITTAL] == VIEW_SAGITTAL
    assert p[VIEW_CORONAL] == VIEW_AXIAL


def test_negative_and_oblique_directions():
    # Sign is irrelevant (abs), and the dominant axis wins for oblique scans.
    fake = _fake([[-10, 0, 0], [0, -9, 3], [0, 2, -10]])
    p = _planes(fake)
    assert p[VIEW_AXIAL] == VIEW_AXIAL         # z mostly along DICOM Z
    assert p[VIEW_SAGITTAL] == VIEW_SAGITTAL   # x mostly along DICOM X
    assert p[VIEW_CORONAL] == VIEW_CORONAL     # y mostly along DICOM Y


def test_missing_geometry_falls_back():
    fake = types.SimpleNamespace(
        planC=None, scanNum=0,
        _VIEW_THRU_COL=PyCerrViewer._VIEW_THRU_COL,
        _DCM_AXIS_PLANE=PyCerrViewer._DCM_AXIS_PLANE)
    assert PyCerrViewer._anatomical_plane(fake, VIEW_AXIAL) == VIEW_AXIAL


# --- orientation markers (L/R/A/P/S/I) -------------------------------------
def _anat(fake, axisLetter):
    return PyCerrViewer._axis_anatomy(fake, axisLetter)


def test_axis_anatomy_real_axial():
    # pyCERR axial: virtual x->+X (L), y->-Y (A), z->-Z (I) (y,z negated).
    fake = _fake([[10, 0, 0], [0, -10, 0], [0, 0, -10]])
    assert _anat(fake, "x") == ("L", "R")
    assert _anat(fake, "y") == ("A", "P")
    assert _anat(fake, "z") == ("I", "S")


def test_axis_anatomy_sagittal():
    # Real Sag T2 data: x->+Y (P), y->+Z (S), z->+X (L).
    fake = _fake([[0, 0, 10], [10, 0, 0], [0, 10, 0]])
    assert _anat(fake, "x") == ("P", "A")
    assert _anat(fake, "y") == ("S", "I")
    assert _anat(fake, "z") == ("L", "R")


def test_axis_anatomy_sign_flip():
    # Flipping an axis direction swaps its two end labels.
    fake = _fake([[-10, 0, 0], [0, 10, 0], [0, 0, 10]])
    assert _anat(fake, "x") == ("R", "L")     # now -X (Right) at the + end
    assert _anat(fake, "y") == ("P", "A")
    assert _anat(fake, "z") == ("S", "I")


def test_axis_anatomy_falls_back_without_geometry():
    from cerr.viewer.pycerr_gui.common import ORIENT_POS, ORIENT_NEG
    fake = types.SimpleNamespace(
        planC=None, scanNum=0, _DCM_AXIS_ENDS=PyCerrViewer._DCM_AXIS_ENDS)
    for a in ("x", "y", "z"):
        assert _anat(fake, a) == (ORIENT_POS[a], ORIENT_NEG[a])


# --- 3D cut-planes label ----------------------------------------------------
def test_cut_planes_label_follows_orientation():
    fake = _fake([[0, 0, 10], [10, 0, 0], [0, 10, 0]])   # sagittal acquisition
    fake.lastSlice = {VIEW_AXIAL: 5, VIEW_SAGITTAL: 6, VIEW_CORONAL: 7}
    fake._anatomical_plane = lambda o: PyCerrViewer._anatomical_plane(fake, o)
    _, _, _, label = PyCerrViewer._plane_slices_3d(fake)
    # window keyed Axial slices the L-R axis here -> a sagittal plane, etc.
    assert "sagittal 6" in label
    assert "coronal 7" in label
    assert "axial 8" in label


def test_cut_planes_label_axial_unchanged():
    fake = _fake([[10, 0, 0], [0, -10, 0], [0, 0, -10]])   # real axial
    fake.lastSlice = {VIEW_AXIAL: 0, VIEW_SAGITTAL: 1, VIEW_CORONAL: 2}
    fake._anatomical_plane = lambda o: PyCerrViewer._anatomical_plane(fake, o)
    _, _, _, label = PyCerrViewer._plane_slices_3d(fake)
    assert "axial 1" in label
    assert "sagittal 2" in label
    assert "coronal 3" in label
