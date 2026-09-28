"""
Checks that the vectorized rasterseg.polyFill reproduces the original
scanline implementation pixel for pixel.
"""

import numpy as np
import pytest
from cerr.contour.rasterseg import polyFill


def reference_poly_fill(rowV, colV, xSize, ySize):
    """Original loop-based polyFill (pyCERR <= 2.2.1), kept as the reference."""
    result = np.zeros((xSize, ySize))
    pointCount = len(rowV)
    edgeList = np.zeros((pointCount, 4))
    for point in range(pointCount):
        p1x, p1y = rowV[point], colV[point]
        p2x, p2y = rowV[(point + 1) % pointCount], colV[(point + 1) % pointCount]
        if p1y <= p2y:
            edgeList[point, :] = [p1x, p1y, p2x, p2y]
        else:
            edgeList[point, :] = [p2x, p2y, p1x, p1y]
    minY = int(np.ceil(np.min(edgeList[:, 1])))
    maxY = int(np.floor(np.max(edgeList[:, 3])))
    for y in range(minY, maxY + 1):
        indV = (edgeList[:, 1] <= y) & (edgeList[:, 3] >= y)
        activeEdges = edgeList[indV, :]
        drawlist = np.empty(0, dtype=int)
        togglelist = np.empty(0, dtype=float)
        for edge in range(activeEdges.shape[0]):
            p1x, p1y, p2x, p2y = activeEdges[edge, :]
            if p1y == p2y:
                if p1x > p2x:
                    drawlist = np.append(drawlist, np.arange(int(np.ceil(p2x)), int(np.floor(p1x)) + 1))
                else:
                    drawlist = np.append(drawlist, np.arange(int(np.ceil(p1x)), int(np.floor(p2x)) + 1))
            elif p2y == y and p2x == round(p2x):
                drawlist = np.append(drawlist, int(p2x))
            elif p2y != y:
                invslope = float(p2x - p1x) / float(p2y - p1y)
                togglelist = np.append(togglelist, p1x + (invslope * (y - p1y)))
        togglelistRound = np.round(togglelist)
        ind_snap = np.abs(togglelistRound - togglelist) < 1e-6
        togglelist[ind_snap] = togglelistRound[ind_snap]
        togglelist.sort()
        for i in range(0, len(togglelist), 2):
            x1, x2 = int(np.ceil(togglelist[i])), int(np.floor(togglelist[i + 1]))
            result[x1:x2 + 1, y] = 1
        result[drawlist, y] = 1
    return result


def star_polygon(rng, center, rMax, nPts, integer=False):
    ang = np.sort(rng.uniform(0, 2 * np.pi, nPts))
    rad = rng.uniform(0.3 * rMax, rMax, nPts)
    rowV = center[0] + rad * np.cos(ang)
    colV = center[1] + rad * np.sin(ang)
    if integer:
        rowV, colV = np.round(rowV), np.round(colV)
    return rowV, colV


def assert_same(rowV, colV, xSize, ySize):
    rowV, colV = np.asarray(rowV, dtype=float), np.asarray(colV, dtype=float)
    inside = rowV.min() >= 0 and colV.min() >= 0 and rowV.max() < xSize - 1 and colV.max() < ySize - 1
    if inside:
        expected = reference_poly_fill(rowV, colV, xSize, ySize)
    else:
        # The reference raises past the far edge and wraps negative indices;
        # polyFill clips. Compare against the reference on a padded canvas.
        pad = int(np.ceil(max(-rowV.min(), -colV.min(), rowV.max() - xSize,
                              colV.max() - ySize, 0))) + 2
        expected = reference_poly_fill(rowV + pad, colV + pad, xSize + 2 * pad,
                                       ySize + 2 * pad)[pad:pad + xSize, pad:pad + ySize]
    np.testing.assert_array_equal(polyFill(rowV, colV, xSize, ySize), expected)


@pytest.mark.parametrize("integer", [False, True])
def test_random_polygons(integer):
    rng = np.random.default_rng(0 if integer else 1)
    for _ in range(300):
        nPts = int(rng.integers(3, 60))
        rowV, colV = star_polygon(rng, (64, 64), rng.uniform(2, 60), nPts, integer)
        assert_same(rowV, colV, 128, 128)


def test_axis_aligned_and_degenerate():
    # square with flat edges on integer scanlines
    assert_same([10, 10, 20, 20], [10, 30, 30, 10], 64, 64)
    # half-integer square
    assert_same([10.5, 10.5, 20.5, 20.5], [10.5, 30.5, 30.5, 10.5], 64, 64)
    # triangle with integer vertices
    assert_same([10, 20, 30, 20], [5, 10, 5, 1], 50, 50)
    # sub-pixel polygon
    assert_same([5.2, 5.4, 5.3], [5.1, 5.2, 5.4], 16, 16)
    # single point and a line
    assert_same([5.0], [5.0], 16, 16)
    assert_same([2.0, 12.0], [3.0, 3.0], 16, 16)


def test_polygon_extending_past_image():
    rng = np.random.default_rng(2)
    for _ in range(100):
        rowV, colV = star_polygon(rng, (rng.uniform(40, 60), rng.uniform(40, 60)), 30, 25)
        assert_same(rowV, colV, 64, 64)
        # crossing the top/left border (negative indices)
        rowV, colV = star_polygon(rng, (rng.uniform(0, 10), rng.uniform(15, 50)), 12, 25)
        assert_same(rowV, colV, 64, 64)


def test_multi_segment_contour():
    # two disjoint polygons concatenated into one vertex list
    rowV = np.r_[10.3, 10.3, 20.7, 20.7, 30.2, 40.8, 35.5]
    colV = np.r_[10.2, 20.6, 20.6, 10.2, 30.1, 30.9, 45.4]
    assert_same(rowV, colV, 64, 64)
