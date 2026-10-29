"""
Fast radiomics features for an ROI with a small region removed.

Removing a small box of voxels from an ROI changes the texture matrices only
near that box. ``LocalFeatureUpdater`` computes the matrices once for the whole
ROI and, for each box, updates only the part of each matrix the box touches.
The updated matrices go through pyCERR's own ``*ToScalarFeatures`` functions,
so the features are the same as a full recompute with
``ibsi1.calcRadiomicsForImgType`` (to float32 rounding).

This makes occlusion / leave-region-out analyses practical: thousands of
"ROI minus a patch" feature sets cost a fraction of a full recompute each.

How each matrix is updated:

* GLCM, NGTDM, NGLDM: the matrix is computed on a small box around the patch,
  with and without the patch. The difference is exactly the change, because
  everything else in the box is counted the same way both times.
  Box margin = voxel offset (GLCM) or 2 x neighborhood radius (NGTDM, NGLDM).
* GLSZM: only zones that contain patch voxels are relabeled (a zone without
  patch voxels cannot change, and removing voxels never merges zones).
* GLRLM: only runs that contain patch voxels are recounted, per direction.
* First-order: recomputed from the remaining voxel values.

Exactness needs the gray levels to stay the same. ``featuresWithoutBox``
returns ``None`` (the caller should recompute in full) when the box would change
the quantization (bins derived from the ROI's own min/max) or would remove every
voxel of the highest gray level (which changes the matrix size).

Example:
    Features of the ROI with a 3x3x3 box removed::

        from cerr.radiomics import ibsi1, preprocess
        from cerr.radiomics.local_update import LocalFeatureUpdater
        from cerr.utils.mask import computeBoundingBox

        scan3M, mask3M, morph3M, gridS, paramS, _ = \\
            preprocess.preProcessForRadiomics(scanNum, structNum, paramS, planC)
        r0, r1, c0, c1, s0, s1, _ = computeBoundingBox(mask3M)
        crop = (slice(r0, r1 + 1), slice(c0, c1 + 1), slice(s0, s1 + 1))
        updater = LocalFeatureUpdater(scan3M[crop], mask3M[crop], gridS, paramS)
        box = (slice(10, 13), slice(20, 23), slice(5, 8))   # in cropped indices
        featDict = updater.featuresWithoutBox(box)          # None -> recompute in full
        flatS = ibsi1.createFlatFeatureDict(featDict, 'original', 'feature', '3D')
"""

import numpy as np
from skimage.measure import label as skLabel

from cerr.radiomics import (ibsi1, first_order, gray_level_cooccurence, run_length, size_zone,
                            neighbor_gray_level_dependence, neighbor_gray_tone)

TEXTURE_CLASSES = ["firstOrder", "glcm", "glrlm", "glszm", "gldm", "gtdm"]


def glcmCounts(quantized3M, offsetsM, nL, perDirection):
    """Raw symmetric co-occurrence counts with the same layout as
    ``gray_level_cooccurence.calcCooccur`` (before its normalization).

    Args:
        quantized3M (np.ndarray(dtype=int)): quantized image, 0 outside the ROI.
        offsetsM (np.ndarray(dtype=int)): direction offsets (N x 3).
        nL (int): number of gray levels.
        perDirection (bool): True for one column per offset (cooccurType 2),
            False for a single merged column (cooccurType 1).

    Returns:
        np.ndarray: (nL*nL) x (N or 1) co-occurrence counts.
    """
    R, C, S = quantized3M.shape
    countsM = np.zeros((nL * nL, len(offsetsM) if perDirection else 1))
    for k, (dr, dc, ds) in enumerate(offsetsM):
        a = quantized3M[max(0, -dr):R - max(0, dr), max(0, -dc):C - max(0, dc), max(0, -ds):S - max(0, ds)]
        b = quantized3M[max(0, dr):R - max(0, -dr), max(0, dc):C - max(0, -dc), max(0, ds):S - max(0, -ds)]
        validM = (a > 0) & (b > 0)
        if not validM.any():
            continue
        indV = (a[validM].astype(np.int64) - 1) * nL + (b[validM].astype(np.int64) - 1)
        cntV = np.bincount(indV, minlength=nL * nL).astype(float)
        cntV += cntV.reshape(nL, nL).T.ravel()
        countsM[:, k if perDirection else 0] += cntV
    return countsM


def glcmNormalize(countsM):
    """Normalize co-occurrence counts per column, as ``calcCooccur`` does (float32)."""
    m = countsM.astype(np.float32)
    colSum = m.sum(axis=0)
    colSum[colSum == 0] = 1
    return m / colSum


def _growBox(boxSlices, marginV, shapeV):
    return tuple(slice(max(0, s.start - m), min(n, s.stop + m)) for s, m, n in zip(boxSlices, marginV, shapeV))


def _relativeTo(boxSlices, outerSlices):
    return tuple(slice(s.start - o.start, s.stop - o.start) for s, o in zip(boxSlices, outerSlices))


class LocalFeatureUpdater:
    """Whole-ROI texture matrices for one image, updated when a box is removed.

    Args:
        img3M (np.ndarray): image (original or filtered), cropped to the ROI's
            bounding box, as passed to ``ibsi1.calcRadiomicsForImgType``.
        mask3M (np.ndarray(dtype=bool)): ROI mask (after re-segmentation), same shape.
        gridS (dict): grid from ``preprocess.preProcessForRadiomics`` (uses ``PixelSpacingV``).
        paramS (dict): radiomics settings.

    Shape features are not handled here: they depend on the morphological mask
    and are unchanged when only the intensity/texture ROI changes.
    """

    def __init__(self, img3M, mask3M, gridS, paramS):
        (self.foOffset, self.foBinWidth, self.foBinNum, self.cooccurType, self.rlmType,
         self.szmDir, self.patchRad, self.diffThresh, offsetsM, self.minClip, self.maxClip,
         self.binNum, self.binWidth) = ibsi1.getRadiomicsSettings(paramS)
        self.offsetsM = np.asarray(offsetsM, dtype=int).reshape(-1, 3)
        featClassS = paramS["featureClass"]
        self.classes = [c for c in TEXTURE_CLASSES
                        if c in featClassS and featClassS[c]["featureList"] != {}]

        mask3M = np.asarray(mask3M, dtype=bool)
        vol3M = np.array(img3M, dtype=float)
        # As in calcRadiomicsForImgType: quantizing sets NaN outside the mask and
        # clips intensities in place; first-order features use that same array.
        q3M = ibsi1.getQuantizedVolume(vol3M, paramS, mask3M, self.binNum,
                                       self.minClip, self.maxClip, self.binWidth)
        self.vol3M = vol3M
        self.mask3M = mask3M
        self.q3M = np.asarray(q3M).astype(np.int64)
        self.nL = int(self.q3M.max())
        self.nVox = int(mask3M.sum())
        self.voxelVol = np.prod(gridS["PixelSpacingV"]) * 1000
        self.levelCountV = np.bincount(self.q3M[mask3M], minlength=self.nL + 1)

        # Does quantization depend on the ROI's own min/max?
        hasTexture = "texture" in paramS["settings"]
        self.minDependent = hasTexture and self.minClip is None and \
            (self.binWidth is not None or self.binNum is not None)
        self.maxDependent = hasTexture and self.maxClip is None and self.binNum is not None
        roiV = vol3M[mask3M]
        self.vMin, self.vMax = np.nanmin(roiV), np.nanmax(roiV)
        self.nAtMin, self.nAtMax = int((roiV == self.vMin).sum()), int((roiV == self.vMax).sum())

        q, nL = self.q3M, self.nL
        if "firstOrder" in self.classes:
            # Sorted ROI values and each voxel's rank, so a box can be removed
            # without re-sorting (first-order stats do not depend on voxel order)
            order = np.argsort(roiV, kind="stable")
            self.foSortedV = roiV[order]
            rankV = np.empty(len(roiV), dtype=np.int64)
            rankV[order] = np.arange(len(roiV))
            self.foRank3M = np.zeros(mask3M.shape, dtype=np.int64)
            self.foRank3M[mask3M] = rankV
        if "glcm" in self.classes:
            self.glcmCountsM = glcmCounts(q, self.offsetsM, nL, self.cooccurType == 2)
            self.glcmMarginV = [int(np.abs(self.offsetsM).max())] * 3
        if "gldm" in self.classes or "gtdm" in self.classes:
            self.nbMarginV = [2 * int(r) for r in self.patchRad]
        if "gldm" in self.classes:
            self.ngldmM = neighbor_gray_level_dependence.calcNGLDM(q, self.patchRad, nL, self.diffThresh)
        if "gtdm" in self.classes:
            s, p, nvc = neighbor_gray_tone.calcNGTDM(q, self.patchRad, nL)
            self.ngtdmS, self.ngtdmCount, self.ngtdmNvc = s, np.round(p * nvc), nvc
        if "glrlm" in self.classes:
            self._initRlm()
        if "glszm" in self.classes:
            self._initSzm()

    # --- GLRLM: which run each voxel belongs to, per direction ---------------
    def _initRlm(self):
        q = self.q3M
        rlm = run_length.calcRLM(q.astype(float), self.offsetsM, self.nL, self.rlmType)
        self.rlmL = [np.array(m, dtype=float) for m in (rlm if self.rlmType == 2 else [rlm])]
        self.rlmPad = pad = max(2, int(np.abs(self.offsetsM).max()) + 1)
        qPad = np.pad(q, pad).astype(np.int32)
        self.rlmShape = qPad.shape
        self.rlmFlatV = flatV = qPad.ravel()
        stridesV = np.array(qPad.strides) // qPad.itemsize
        roiIndV = np.flatnonzero(flatV)
        self.rlmSteps, self.runStartL, self.runLenL = [], [], []
        for off in self.offsetsM:
            step = int(np.dot(off, stridesV))
            startV = roiIndV[flatV[roiIndV - step] != flatV[roiIndV]]
            runStartV = np.full(flatV.size, -1, dtype=np.int64)
            runLenV = np.zeros(flatV.size, dtype=np.int32)
            levV = flatV[startV]
            curV = startV.copy()
            lenV = np.ones(len(startV), dtype=np.int32)
            runStartV[startV] = startV
            activeV = np.arange(len(startV))
            while activeV.size:
                nextV = curV[activeV] + step
                contV = flatV[nextV] == levV[activeV]
                endedV = activeV[~contV]
                runLenV[startV[endedV]] = lenV[endedV]
                activeV = activeV[contV]
                curV[activeV] = nextV[contV]
                lenV[activeV] += 1
                runStartV[curV[activeV]] = startV[activeV]
            self.rlmSteps.append(step)
            self.runStartL.append(runStartV)
            self.runLenL.append(runLenV)

    def _rlmWithout(self, boxIndV):
        pad = self.rlmPad
        boxFlatV = np.ravel_multi_index(tuple(i + pad for i in boxIndV), self.rlmShape)
        removedV = np.zeros(self.rlmFlatV.size, dtype=bool)
        removedV[boxFlatV] = True
        mats = [m.copy() for m in self.rlmL]
        for k, step in enumerate(self.rlmSteps):
            M = mats[k if self.rlmType == 2 else 0]
            for start in np.unique(self.runStartL[k][boxFlatV]):
                runLen = int(self.runLenL[k][start])
                lev = int(self.rlmFlatV[start])
                keepV = ~removedV[start + step * np.arange(runLen)]
                M[lev - 1, runLen - 1] -= 1
                if keepV.any():
                    d = np.diff(np.concatenate(([0], keepV.astype(np.int8), [0])))
                    segLenV = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
                    np.add.at(M[lev - 1], segLenV - 1, 1)
        return mats if self.rlmType == 2 else mats[0]

    # --- GLSZM: zone labels ---------------------------------------------------
    def _initSzm(self):
        q = self.q3M
        self.szmM = np.asarray(size_zone.calcSZM(q, self.nL, self.szmDir))
        if self.szmDir == 1:
            lab3M = skLabel(q, background=0, connectivity=3)
        else:  # 2D zones per slice, with labels unique across slices
            lab3M = np.zeros(q.shape, dtype=np.int64)
            nextLab = 0
            for k in range(q.shape[2]):
                lab2M = skLabel(q[:, :, k], background=0, connectivity=2).astype(np.int64)
                lab2M[lab2M > 0] += nextLab
                nextLab = max(nextLab, int(lab2M.max()))
                lab3M[:, :, k] = lab2M
        self.zoneLab3M = lab3M
        labV = lab3M.ravel()
        self.zoneOrderV = np.argsort(labV, kind="stable")
        self.zoneStartV = np.concatenate(([0], np.cumsum(np.bincount(labV))))

    def _szmWithout(self, boxSlices):
        M = self.szmM.astype(np.int64).copy()
        boxLab3M = self.zoneLab3M[boxSlices]
        conn = 3 if self.szmDir == 1 else 2
        for z in np.unique(boxLab3M[boxLab3M > 0]):
            voxV = self.zoneOrderV[self.zoneStartV[z]:self.zoneStartV[z + 1]]
            r, c, s = np.unravel_index(voxV, self.q3M.shape)
            lev = int(self.q3M[r[0], c[0], s[0]])
            M[lev - 1, len(voxV) - 1] -= 1
            inBoxV = ((r >= boxSlices[0].start) & (r < boxSlices[0].stop) &
                      (c >= boxSlices[1].start) & (c < boxSlices[1].stop) &
                      (s >= boxSlices[2].start) & (s < boxSlices[2].stop))
            r, c, s = r[~inBoxV], c[~inBoxV], s[~inBoxV]
            if r.size == 0:
                continue
            zone3M = np.zeros((r.max() - r.min() + 1, c.max() - c.min() + 1, s.max() - s.min() + 1), dtype=bool)
            zone3M[r - r.min(), c - c.min(), s - s.min()] = True
            if conn == 2:
                zone3M = zone3M[:, :, 0]
            newSizeV = np.bincount(skLabel(zone3M, background=0, connectivity=conn).ravel())[1:]
            np.add.at(M[lev - 1], newSizeV[newSizeV > 0] - 1, 1)
        usedV = np.flatnonzero(M.any(axis=0))
        return M[:, :usedV[-1] + 1 if usedV.size else 0]

    # --- features with a box removed -------------------------------------------
    def featuresWithoutBox(self, boxSlices):
        """Texture/first-order features with the voxels in boxSlices removed from the ROI.

        Args:
            boxSlices (tuple of slice): (rows, cols, slices) in the cropped array.

        Returns:
            dict or None: feature dictionary laid out like
            ``ibsi1.calcRadiomicsForImgType`` (without ``'shape'``), or None when
            a local update would not be exact; recompute in full in that case.
        """
        q, nL = self.q3M, self.nL
        boxMask3M = self.mask3M[boxSlices]
        if self._needsFullRecompute(boxSlices, boxMask3M):
            return None

        numVoxels = self.nVox - int(boxMask3M.sum())
        boxIndV = tuple(ix + s.start for ix, s in zip(np.nonzero(boxMask3M), boxSlices))
        featDict = {}

        if "firstOrder" in self.classes:
            valsV = np.delete(self.foSortedV, self.foRank3M[boxSlices][boxMask3M])
            featDict["firstOrder"] = first_order.stats(valsV, self.voxelVol, self.foOffset,
                                                       self.foBinWidth, self.foBinNum)

        if "glcm" in self.classes:
            outer = _growBox(boxSlices, self.glcmMarginV, q.shape)
            qFull = q[outer]
            qRed = qFull.copy()
            qRed[_relativeTo(boxSlices, outer)] = 0
            perDir = self.cooccurType == 2
            countsM = self.glcmCountsM - glcmCounts(qFull, self.offsetsM, nL, perDir) \
                + glcmCounts(qRed, self.offsetsM, nL, perDir)
            featDict["glcm"] = gray_level_cooccurence.cooccurToScalarFeatures(glcmNormalize(countsM))

        if "glrlm" in self.classes:
            n = numVoxels * self.offsetsM.shape[0] if self.rlmType == 1 else numVoxels
            featDict["glrlm"] = run_length.rlmToScalarFeatures(self._rlmWithout(boxIndV), n)

        if "glszm" in self.classes:
            featDict["glszm"] = size_zone.szmToScalarFeatures(self._szmWithout(boxSlices), numVoxels)

        if "gldm" in self.classes or "gtdm" in self.classes:
            featDict.update(self._neighborhoodFeatures(boxSlices, numVoxels))
        return featDict

    def _needsFullRecompute(self, boxSlices, boxMask3M):
        """True when a local update would not be exact: an empty box, or a box
        that changes the gray-level bins or removes the highest gray level."""
        if not boxMask3M.any():
            return True
        boxValV = self.vol3M[boxSlices][boxMask3M]
        if self.minDependent and (boxValV == self.vMin).sum() == self.nAtMin:
            return True
        if self.maxDependent and (boxValV == self.vMax).sum() == self.nAtMax:
            return True
        boxLevV = self.q3M[boxSlices][boxMask3M]
        return (boxLevV == self.nL).sum() == self.levelCountV[self.nL]

    def _neighborhoodFeatures(self, boxSlices, numVoxels):
        """NGLDM and NGTDM: difference of the matrices on a box around the patch."""
        q, nL = self.q3M, self.nL
        outer = _growBox(boxSlices, self.nbMarginV, q.shape)
        qFull = q[outer]
        qRed = qFull.copy()
        qRed[_relativeTo(boxSlices, outer)] = 0
        featDict = {}
        if "gldm" in self.classes:
            sFull = neighbor_gray_level_dependence.calcNGLDM(qFull, self.patchRad, nL, self.diffThresh)
            sRed = neighbor_gray_level_dependence.calcNGLDM(qRed, self.patchRad, nL, self.diffThresh)
            s = self.ngldmM.astype(np.int64) - sFull.astype(np.int64) + sRed.astype(np.int64)
            featDict["gldm"] = neighbor_gray_level_dependence.ngldmToScalarFeatures(
                s.astype(self.ngldmM.dtype), numVoxels)
        if "gtdm" in self.classes:
            sF, pF, nF = neighbor_gray_tone.calcNGTDM(qFull, self.patchRad, nL)
            sR, pR, nR = neighbor_gray_tone.calcNGTDM(qRed, self.patchRad, nL)
            cntF = np.round(pF * nF) if nF else np.zeros_like(pF)
            cntR = np.round(pR * nR) if nR else np.zeros_like(pR)
            nvc = self.ngtdmNvc - nF + nR
            featDict["gtdm"] = neighbor_gray_tone.ngtdmToScalarFeatures(
                self.ngtdmS - sF + sR, (self.ngtdmCount - cntF + cntR) / nvc, nvc)
        return featDict
