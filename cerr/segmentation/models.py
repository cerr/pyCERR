"""Functions to apply (non-AI) segmentation models."""

import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import logging
from cerr import plan_container as pc
from cerr.dataclasses import scan as cerrScn
from cerr.dataclasses import structure as cerrStr

logger = logging.getLogger(__name__)

MODELS = {'SCRR':'parotidSCRRFn'}

def parotidSCRRFn(planC, scanNum, paramDict):
    """This routine extracts Stem Cell Rich Regions (SCRR) from parotid gland masks.

    Uses anatomical landmarks (parotid, masseter, mandible) to identify
    and extract regions of interest containing stem cell populations
    (Steenbakkers et al. 2021). For each side:
        - level: the slice of that parotid's center of gravity;
        - junction of that parotid, masseter and the mandible on that slice: the parotid
          voxels closest to touching both the masseter and the mandible (smallest larger
          distance, in mm, to the two), and their center if several are equally close;
        - SCRR center: the junction shifted 0.5 cm lateral and 0.5 cm posterior;
        - SCRR: the ellipsoid of semi-axes 1 cm (anterior-posterior, left-right) and 2 cm
          (cranial-caudal) around the center, within the parotid.
    
    Args:
        planC (cerr.plan_container.PlanC): pyCERR's plan container object
        scanNum (int): index of scan from planC.scan
        paramDict (dict, optional): dictionary specifying model parameters.
                Valid keys:
                structNameDict (dict): (Required) Mapping of expected structures to their names in planC.structure.
                                    Required keys: "left_parotid", "right_parotid", "left_masseter",
                                    "right_masseter", "mandible"
                                    structNameDict = {'left_parotid': 'Left Parotid', 'right_parotid': 'Right Parotid',
                                         'left_masseter': 'Left masseter', 'right_masseter': 'Right masseter',
                                         'mandible': 'Mandible'}
                marginsMm (dict, optional): semi-axes of the ellipsoid in mm, with keys "x" (left-right),
                                    "y" (anterior-posterior) and "z" (cranial-caudal).
                shiftMm (dict, optional): shift of the center from the junction in mm, per side ("L", "R"),
                                    with keys "x" (+ toward the patient's left), "y" (+ posterior) and "z"
                                    (not applied), i.e. LPS directions.

    Returns:
        mask4M (np.ndarray, 4D)          : Stack of output 3D binary masks.
        labelDict (dict)                 : Dictionary mapping output structure labels with indices in map4M.
                                           {"outStructure":label}
                                           outStructureMask3M = mask4M[:,:,:,label-1]
        planC (cerr.plan_container.PlanC): pyCERR's plan container object with SCRR structures added

    Author: Viktor Rogowski, PhD student, Medical Physicist
    Skåne University Hospital, Lund, Sweden and Lund University, Lund, Sweden
    """
    # Default parameters
    defaultParamDict = {'marginsMm': {"x": 10.0, "y": 10.0, "z": 20.0},
                        # 0.5 cm lateral and 0.5 cm posterior (LPS: +x left, +y posterior)
                        'shiftMm': {
                            "L": {"x": 5.0, "y": 5.0, "z": 0.0},
                            "R": {"x": -5.0, "y": 5.0, "z": 0.0},
                        },
                        'side': "L",
                        'pixelValue': 1
    }

    if paramDict is None:
        paramDict = defaultParamDict
    else:
        for key in defaultParamDict:
            if key not in paramDict:
                paramDict[key] = defaultParamDict[key]

    structNameDict = paramDict["structNameDict"]
    marginsMm = paramDict["marginsMm"]
    shiftMm = paramDict["shiftMm"]

    labelDict = {"leftSCRR": 1, "rightSCRR": 2}
    outputNames = list(labelDict.keys())

    # Helper functions
    def _extractSingleSCRR(parotidMask3M, masseterMask3M, mandibleMask3M,
                           spacing, side, marginsMm, shiftMm, pixelValue=1):
        """This routine extracts SCRR for a single side (left or right).

        Args:
            parotidMask3M (numpy.ndarray): 3D binary mask of this side's parotid in LPS orientation
            masseterMask3M (numpy.ndarray): 3D binary mask of this side's masseter in LPS orientation
            mandibleMask3M (numpy.ndarray): 3D binary mask of the mandible in LPS orientation
            spacing (tuple): voxel spacing (x, y, z) in mm of the LPS-oriented masks
            side (str): "L" for left, "R" for right
            marginsMm (dict): ellipsoid semi-axes in mm, keys "x" (left-right), "y" (anterior-posterior), "z"
            shiftMm (dict): center shift in mm per side ("L", "R"), keys "x" (+ left), "y" (+ posterior)
            pixelValue (int): value to assign to extracted SCRR voxels

        Returns:
            numpy.ndarray: 3D mask of extracted SCRR region
        """
        # Level: the slice of this parotid's center of gravity
        centerGravity = _calculateCenterOfGravity(parotidMask3M)

        # Junction of this parotid, its masseter and the mandible on that slice. The mandible
        # isn't split into sides: the part nearest this parotid is on the same side.
        intersection = _findJunction(parotidMask3M, masseterMask3M, mandibleMask3M,
                                     centerGravity[-1], spacing)

        # Convert margins from mm to voxels. Masks are in LPS orientation, indexed (row, col, slc):
        # rows run anterior-posterior (y), columns left-right (x), slices cranial-caudal (z).
        marginsVoxels = (
            marginsMm["y"] / spacing[1],
            marginsMm["x"] / spacing[0],
            marginsMm["z"] / spacing[2],
        )

        # Apply centroid shift (x: + toward the patient's left; y: + posterior)
        shift = shiftMm[side]
        shiftedCentroid = intersection + np.array([
            shift["y"] / spacing[1],
            shift["x"] / spacing[0],
            0,
        ])

        # Create ellipsoid volume and extract overlap with parotid
        ellipsoidVolume = _createVolume(shiftedCentroid, parotidMask3M.shape, marginsVoxels)
        scrr = _extractOverlappingVolume(ellipsoidVolume, parotidMask3M)

        if pixelValue != 1:
            scrr[scrr == 1] = pixelValue
            logger.info(f"Assigned pixel value {pixelValue} to extracted SCRR")

        logger.info(f"Extracted SCRR for side {side}")

        return scrr


    def _getMatchingInds(namesList, planC):
        """This routine finds structure indices in planC matching the given names.

        Args:
            namesList (list): list of structure name strings to match
            planC (cerr.plan_container.PlanC): pyCERR's plan container object

        Returns:
            list: list of matching structure indices from planC.structure
        """
        allStructNames = [s.structureName for s in planC.structure]
        matchingInds = []
        for strName in namesList:
            inds = cerrStr.getMatchingIndex(strName, allStructNames, "exact")
            if len(inds) > 0:
                matchingInds.extend(inds)
        return matchingInds


    def _getLpsMask(inds, planC):
        """This routine extracts a combined binary mask in LPS orientation for given structure indices.

        Args:
            inds (list or int): structure index or list of indices from planC.structure
            planC (cerr.plan_container.PlanC): pyCERR's plan container object

        Returns:
            numpy.ndarray: 3D binary mask in LPS orientation (row, col, slc)
            str: original DICOM orientation string
            tuple: voxel spacing (x, y, z) in mm of the LPS-oriented mask
        """
        if isinstance(inds, (int, float)):
            inds = [inds]

        scanNum = cerrScn.getScanNumFromUID(
            planC.structure[inds[0]].assocScanUID, planC
        )
        scanSize = planC.scan[scanNum].getScanSize()
        mask3M = np.zeros(scanSize, dtype=bool)

        originalOrient = None
        for strNum in inds:
            sitkStructImage = planC.structure[strNum].getSitkImage(planC)
            originalOrient = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
                sitkStructImage.GetDirection()
            )
            lpsImage = sitk.DICOMOrient(sitkStructImage, "LPS")
            mask3M = mask3M | np.transpose(
                sitk.GetArrayFromImage(lpsImage), (1, 2, 0)
            )

        return mask3M, originalOrient, lpsImage.GetSpacing()


    def _getMasks(structNameDict, planC):
        """This routine returns the masks the SCRR is derived from, in LPS orientation:
        the left and right parotids and masseters, and the mandible.

        Args:
            structNameDict (dict): structure names (see parotidSCRRFn), keyed by
                "left_parotid", "right_parotid", "left_masseter", "right_masseter" and
                "mandible".
            planC (cerr.plan_container.PlanC): pyCERR's plan container object

        Returns:
            dict: 3D binary masks keyed as structNameDict
            str: original DICOM orientation string
            tuple: voxel spacing (x, y, z) in mm of the LPS-oriented masks
            int: index of a parotid in planC.structure (for its scan)

        Raises:
            ValueError: a structure isn't found in planC
        """
        def inds(key):
            found = _getMatchingInds([structNameDict[key]], planC)
            if not found:
                raise ValueError(f"No matching structure found for {key}: {structNameDict[key]!r}")
            return found

        masks, originalOrient = {}, None
        for key in ("left_parotid", "right_parotid", "left_masseter", "right_masseter", "mandible"):
            masks[key], originalOrient, spacing = _getLpsMask(inds(key), planC)
        return masks, originalOrient, spacing, inds("left_parotid")[0]


    def _reorientMask(mask3M, originalOrient, planC, scanNum):
        """This routine reorients a mask from LPS orientation back to the original scan orientation.

        Args:
            mask3M (numpy.ndarray): 3D mask in LPS orientation (row, col, slc)
            originalOrient (str): original DICOM orientation string
            planC (cerr.plan_container.PlanC): pyCERR's plan container object
            scanNum (int): index of scan from planC.scan

        Returns:
            numpy.ndarray: 3D mask in original orientation (row, col, slc)
        """
        sitkImage = sitk.GetImageFromArray(np.transpose(mask3M, (2, 0, 1)))
        sitkImage.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        sitkImage = sitk.DICOMOrient(sitkImage, originalOrient)
        reorientedMask3M = sitk.GetArrayFromImage(sitkImage)
        reorientedMask3M = np.transpose(reorientedMask3M, (1, 2, 0))

        if cerrScn.flipSliceOrderFlag(planC.scan[scanNum]):
            reorientedMask3M = np.flip(reorientedMask3M, axis=2)

        return reorientedMask3M


    def _calculateCenterOfGravity(contour):
        """This routine calculates the center of gravity (centroid) of a 3D contour.

        Args:
            contour (numpy.ndarray): 3D binary mask array

        Returns:
            numpy.ndarray: (row, col, slc) of the centroid, rounded down to whole voxels

        Raises:
            ValueError: if contour is empty
        """
        if not np.any(contour):
            raise ValueError("Cannot calculate center of gravity for empty contour")

        indices = np.indices(contour.shape)
        totalMass = np.sum(contour)
        center = np.array([np.sum(indices[axis] * contour) / totalMass for axis in range(3)])

        return center.astype(int)


    def _findJunction(parotidMask3M, masseterMask3M, mandibleMask3M, zSlice, spacing):
        """This routine finds the junction of the parotid, masseter and mandible on a slice.

        The junction is where the three structures meet: the parotid voxels closest to
        touching both the masseter and the mandible, i.e. with the smallest larger distance
        (in mm) to the two. Where several voxels are equally close (e.g. along a shared
        boundary), their center is used.

        Args:
            parotidMask3M (numpy.ndarray): 3D parotid mask in LPS orientation (row, col, slc)
            masseterMask3M (numpy.ndarray): 3D masseter mask in LPS orientation
            mandibleMask3M (numpy.ndarray): 3D mandible mask in LPS orientation
            zSlice (int): slice to find the junction on
            spacing (tuple): voxel spacing (x, y, z) in mm of the LPS-oriented masks

        Returns:
            numpy.ndarray: [row, col, slc] of the junction (row and col may be fractional)

        Raises:
            ValueError: if a structure is empty on this slice
        """
        parotid2M = parotidMask3M[:, :, zSlice].astype(bool)
        masseter2M = masseterMask3M[:, :, zSlice].astype(bool)
        mandible2M = mandibleMask3M[:, :, zSlice].astype(bool)
        for name, mask2M in (("Parotid", parotid2M), ("Masseter", masseter2M), ("Mandible", mandible2M)):
            if not mask2M.any():
                raise ValueError(f"{name} mask is empty on this slice")

        # Distance (mm) from each voxel to the nearest masseter and mandible voxel; rows run
        # anterior-posterior (y spacing), columns left-right (x spacing)
        sampling = (spacing[1], spacing[0])
        distMasseter2M = ndimage.distance_transform_edt(~masseter2M, sampling=sampling)
        distMandible2M = ndimage.distance_transform_edt(~mandible2M, sampling=sampling)
        closeness2M = np.where(parotid2M, np.maximum(distMasseter2M, distMandible2M), np.inf)
        junctionCoords = np.argwhere(np.isclose(closeness2M, closeness2M.min()))
        junction = junctionCoords.mean(axis=0)
        logger.info(f"Junction at {junction} on slice {zSlice}, {closeness2M.min():.1f} mm from "
                    f"masseter and mandible ({len(junctionCoords)} voxel(s))")

        return np.append(junction, zSlice)


    def _createVolume(center, shape, margins):
        """This routine creates an ellipsoid volume around a center point.

        Args:
            center (numpy.ndarray): [row, col, slc] of the ellipsoid center
            shape (tuple): shape of the output volume (rows, cols, slcs)
            margins (tuple): semi-axes lengths (row, col, slc) in voxels

        Returns:
            numpy.ndarray: binary volume with ellipsoid region set to 1
        """
        rowMargin, colMargin, slcMargin = margins

        rowCoords, colCoords, slcCoords = np.indices(shape)

        rowCentered = rowCoords - center[0]
        colCentered = colCoords - center[1]
        slcCentered = slcCoords - center[2]

        mask = ((rowCentered / rowMargin) ** 2 + (colCentered / colMargin) ** 2
                + (slcCentered / slcMargin) ** 2) <= 1

        logger.info(f"Created ellipsoid volume with center at {center} and margins {margins}")

        return mask.astype(int)


    def _extractOverlappingVolume(volume, contour):
        """This routine extracts the overlapping region between two volumes.

        Args:
            volume (numpy.ndarray): first binary volume
            contour (numpy.ndarray): second binary volume

        Returns:
            numpy.ndarray: binary volume with overlapping region set to 1
        """
        overlappingVolume = np.logical_and(volume, contour).astype(int)

        return overlappingVolume

    # Get masks in LPS orientation: left and right parotids and masseters, and the mandible
    # spacing: voxel spacing (x, y, z) in mm of the LPS-oriented masks, matching their axes
    masks, originalOrient, spacing, parotidInd = _getMasks(structNameDict, planC)

    # Derive scanNum from structure if not provided
    if scanNum is None:
        scanNum = cerrScn.getScanNumFromUID(planC.structure[parotidInd].assocScanUID, planC)

    # Extract left parotid SCRR, centred at the level of the left parotid's centre of gravity
    leftSCRR = _extractSingleSCRR(
        masks["left_parotid"], masks["left_masseter"], masks["mandible"],
        spacing, "L", marginsMm, shiftMm, pixelValue=1
    )

    # Extract right parotid SCRR, centred at the level of the right parotid's centre of gravity
    rightSCRR = _extractSingleSCRR(
        masks["right_parotid"], masks["right_masseter"], masks["mandible"],
        spacing, "R", marginsMm, shiftMm, pixelValue=2
    )

    # Reorient from LPS to original orientation
    leftSCRRmask3M = _reorientMask(leftSCRR, originalOrient, planC, scanNum)
    rightSCRRmask3M = _reorientMask(rightSCRR, originalOrient, planC, scanNum)
    mask4M = np.stack((leftSCRRmask3M, rightSCRRmask3M), axis=3)

    # Import masks to planC
    planC = pc.importStructureMask(leftSCRRmask3M > 0, scanNum, outputNames[0], planC)
    planC = pc.importStructureMask(rightSCRRmask3M > 0, scanNum, outputNames[1], planC)

    logger.info("Extracted left and right parotid SCRR structures")

    return mask4M, labelDict, planC


def run(modelName, planC, scanNum, paramDict=None):
    """Runs segmentation model specified through name and any user-input parameters required.

    Args:
        modelName (str): name of the model. See MODELS keys for valid options.
        planC (cerr.plan_container.PlanC): pyCERR's plan container object
        scanNum (int): index of scan from planC.scan
        paramDict (dict, optional): dictionary specifying model-specific parameters.

    Returns:
        mask4M (np.ndarray, 4D)          : Stack of output 3D binary masks.
        labelDict (dict)                 : Dictionary mapping output structure labels with indices in mask4M.
                                           {"outStructure":label}
                                           outStructureMask3M = mask4M[:,:,:,label-1]
        planC (cerr.plan_container.PlanC): pyCERR's plan container object with SCRR structures added
    """
    modelFn = MODELS[modelName]
    mask4M, labelDict, planC = eval(modelFn)(planC, scanNum, paramDict)

    return mask4M, labelDict, planC