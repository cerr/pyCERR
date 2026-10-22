"""Tests for ``cerr.utils.image_proc.resizeScanAndMask``.

The resized scan is imported into planC with the grid the function returns,
so that grid must give the physical position of every output voxel. A linear
intensity ramp in x and y is resized; padding, cropping and linear
interpolation all reproduce a linear function exactly, so the output must equal
the ramp evaluated at the returned grid wherever the output covers the input.
"""
import numpy as np
import pytest

from cerr import plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.dataclasses import structure as cerrStr
from cerr.utils.image_proc import resizeScanAndMask

N_ROWS, N_COLS, N_SLCS = 41, 46, 5
DX, DY, DZ = 0.1, 0.12, 0.3


def _ramp(xV, yV, zV):
    xM, yM, zM = np.meshgrid(xV, yV, zV)          # (rows, cols, slices)
    return 3.0 * xM + 5.0 * yM + 7.0 * zM


def _grid():
    xV = -2.0 + np.arange(N_COLS) * DX
    yV = 1.5 - np.arange(N_ROWS) * DY               # y decreases down the rows
    zV = -0.6 + np.arange(N_SLCS) * DZ
    return xV, yV, zV


def _ellipse(xV, yV, zV):
    xM, yM, _ = np.meshgrid(xV, yV, zV)
    return ((xM + 0.3) / 1.2) ** 2 + ((yM - 0.2) / 1.0) ** 2 < 1


@pytest.mark.parametrize("method, outSize", [
    ("padorcrop3d", [64, 64]),     # pad both
    ("padorcrop3d", [30, 33]),     # crop both
    ("padorcrop3d", [52, 31]),     # pad rows, crop cols
    ("padorcrop3d", [41, 46]),     # unchanged
    ("pad3d", [50, 51]),           # odd differences
    ("unpad3d", [40, 45]),
])
def test_pad_crop_grid_matches_data(method, outSize):
    gridS = _grid()
    scan3M = _ramp(*gridS)
    out3M, _, (xO, yO, zO) = resizeScanAndMask(scan3M, None, gridS,
                                               outSize + [N_SLCS], method)
    assert out3M.shape == (outSize[0], outSize[1], N_SLCS)
    assert len(xO) == outSize[1] and len(yO) == outSize[0] and len(zO) == N_SLCS
    np.testing.assert_allclose(np.diff(xO), DX)
    np.testing.assert_allclose(np.diff(yO), -DY)
    # Voxels copied from the input hold the ramp value of their grid position
    inside = ((xO >= gridS[0][0] - 1e-9) & (xO <= gridS[0][-1] + 1e-9))[None, :, None] \
        & ((yO <= gridS[1][0] + 1e-9) & (yO >= gridS[1][-1] - 1e-9))[:, None, None]
    inside = np.broadcast_to(inside, out3M.shape)
    np.testing.assert_allclose(out3M[inside], _ramp(xO, yO, zO)[inside], atol=1e-9)
    # Padded voxels hold the scan minimum
    assert np.all(out3M[~inside] == scan3M.min())


def test_pad_then_crop_round_trip():
    gridS = _grid()
    scan3M = _ramp(*gridS)
    mask4M = _ellipse(*gridS)[..., None]
    pad3M, padMask4M, padGrid = resizeScanAndMask(scan3M, mask4M, gridS, [57, 60, N_SLCS],
                                                  'padorcrop3d')
    back3M, backMask4M, backGrid = resizeScanAndMask(pad3M, padMask4M, padGrid,
                                                     [N_ROWS, N_COLS, N_SLCS], 'padorcrop3d')
    np.testing.assert_array_equal(back3M, scan3M)
    np.testing.assert_array_equal(backMask4M, mask4M)
    for a, b in zip(backGrid, gridS):
        np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("method", ["bilinear", "bicubic"])
@pytest.mark.parametrize("outSize", [[20, 23], [82, 92], [30, 70]])
def test_interpolation_grid_matches_data(method, outSize):
    gridS = _grid()
    scan3M = _ramp(*gridS)
    out3M, _, (xO, yO, zO) = resizeScanAndMask(scan3M, None, gridS,
                                               outSize + [N_SLCS], method)
    assert out3M.shape == (outSize[0], outSize[1], N_SLCS)
    # Same physical extent: the outer pixel edges coincide
    np.testing.assert_allclose(xO[0] - (xO[1] - xO[0]) / 2, gridS[0][0] - DX / 2)
    np.testing.assert_allclose(xO[-1] + (xO[1] - xO[0]) / 2, gridS[0][-1] + DX / 2)
    np.testing.assert_allclose(yO[0] - (yO[1] - yO[0]) / 2, gridS[1][0] + DY / 2)
    assert yO[1] < yO[0]                                  # still decreasing
    # Interior values equal the ramp at the returned coordinates. The edge
    # pixels are excluded: interpolation there reflects the image boundary.
    expect3M = _ramp(xO, yO, zO)
    m = 6
    np.testing.assert_allclose(out3M[m:-m, m:-m, :], expect3M[m:-m, m:-m, :],
                               rtol=0, atol=0.02)


def test_interpolation_keeps_integer_scan_values():
    gridS = _grid()
    scan3M = (_ramp(*gridS) * 100).astype(np.int16)
    out3M, _, _ = resizeScanAndMask(scan3M, None, gridS, [20, 23, N_SLCS], 'bilinear')
    assert abs(out3M.mean() - scan3M.mean()) < 0.05 * abs(scan3M).max()


def test_single_structure_mask_and_aspect():
    gridS = _grid()
    scan3M = _ramp(*gridS)
    mask4M = _ellipse(*gridS)[..., None]
    for method in ["bilinear", "nearest"]:
        out3M, outMask4M, _ = resizeScanAndMask(scan3M, mask4M, gridS, [64, 64, N_SLCS], method)
        assert outMask4M.shape == (64, 64, N_SLCS, 1)
        assert set(np.unique(outMask4M)) <= {0, 1}
    out3M, outMask4M, (xO, yO, _) = resizeScanAndMask(scan3M, mask4M, gridS, [64, 64, N_SLCS],
                                                      'bilinear', preserveAspectFlag=True)
    # Square padding first: equal physical spacing along rows and cols of the square
    assert out3M.shape[:2] == (64, 64)
    np.testing.assert_allclose(abs(xO[1] - xO[0]), N_COLS * DX / 64)
    np.testing.assert_allclose(abs(yO[1] - yO[0]), N_COLS * DY / 64)
    out3M, _, _ = resizeScanAndMask(scan3M, None, gridS, [60, 60, N_SLCS], 'padorcrop3d',
                                    preserveAspectFlag=True)
    assert out3M.shape == (60, 60, N_SLCS)


def test_resized_segmentation_maps_back_onto_original_scan(tmp_path):
    """The AI pipeline pattern: resize, segment, import on the resized scan,
    copy the structure back to the original scan."""
    # Build a planC from NIfTI so the original scan has full geometry
    import SimpleITK as sitk
    gridS = _grid()
    img = sitk.GetImageFromArray(np.transpose(_ramp(*gridS), (2, 0, 1)).astype(np.float32))
    img.SetSpacing((DX * 10, DY * 10, DZ * 10))
    niiFile = str(tmp_path / 'scan.nii.gz')
    sitk.WriteImage(img, niiFile)
    planC = pc.loadNiiScan(niiFile, 'CT SCAN')
    scan3M = planC.scan[0].getScanArray()
    gridS = planC.scan[0].getScanXYZVals()
    xM, yM, _ = np.meshgrid(*gridS)
    xC, yC = gridS[0].mean(), gridS[1].mean()
    ref3M = ((xM - xC) / 1.2) ** 2 + ((yM - yC) / 1.0) ** 2 < 1
    planC = pc.importStructureMask(ref3M, 0, 'ref', planC)

    for method, size in [('bilinear', [96, 80]), ('padorcrop3d', [64, 64]),
                         ('padorcrop3d', [30, 36])]:
        out3M, outMask4M, outGrid = resizeScanAndMask(scan3M, ref3M[..., None], gridS,
                                                      size + [scan3M.shape[2]], method)
        planC = pc.importScanArray(out3M, *outGrid, 'CT', 0, planC)
        procScanNum = len(planC.scan) - 1
        planC = pc.importStructureMask(outMask4M[..., 0] > 0, procScanNum, 'seg', planC)
        planC = cerrStr.copyToScan(len(planC.structure) - 1, 0, planC)
        back3M = rs.getStrMask(len(planC.structure) - 1, planC)
        dice = 2 * (back3M & ref3M).sum() / (back3M.sum() + ref3M.sum())
        assert dice > 0.97, (method, size, dice)
