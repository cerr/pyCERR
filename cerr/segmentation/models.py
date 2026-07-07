"""Functions to apply (non-AI) segmentation models."""

import SimpleITK as sitk
import numpy as np
import logging
from cerr import plan_container as pc
from cerr.dataclasses import scan as cerrScn
from cerr.dataclasses import structure as cerrStr

logger = logging.getLogger(__name__)

MODELS = {'SCRR':'parotidSCRRFn'}

def parotidSCRRFn(planC, scanNum, paramDict):
    """This routine extracts Stem Cell Rich Regions (SCRR) from parotid gland masks.

    Uses anatomical landmarks (parotid, masseter, mandible) to identify
    and extract regions of interest containing stem cell populations.
    
    Args:
        planC (cerr.plan_container.PlanC): pyCERR's plan container object
        scanNum (int): index of scan from planC.scan
        paramDict (dict, optional): dictionary specifying model parameters.
                Valid keys:
                structNameDict (dict): (Required) Mapping of expected structures to their names in planC.structure.
                                    Required keys: "parotid", "masseter", "mandible"
                                    structNameDict = {'parotid': ['Left Parotid', 'Right Parotid'],
                                         'masseter': ['Left masseter','Right masseter'],
                                         'mandible': ['Mandible']}
                marginsMm (dict, optional): margin values in mm with keys "x", "y", "z"
                shiftMm (dict, optional): centroid shift values with keys "L" and "R"

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
                        'shiftMm': {
                            "L": {"x": 5.0, "y": 5.0, "z": 0.0},
                            "R": {"x": 5.0, "y": -5.0, "z": 0.0},
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
            parotidMask3M (numpy.ndarray): 3D binary mask of parotid in LPS orientation
            masseterMask3M (numpy.ndarray): 3D binary mask of masseter in LPS orientation
            mandibleMask3M (numpy.ndarray): 3D binary mask of mandible in LPS orientation
            spacing (numpy.ndarray): voxel spacing (x, y, z) in mm
            side (str): "L" for left, "R" for right
            marginsMm (dict): margin values in mm with keys "x", "y", "z"
            shiftMm (dict): centroid shift values with keys "L" and "R"
            pixelValue (int): value to assign to extracted SCRR voxels

        Returns:
            numpy.ndarray: 3D mask of extracted SCRR region
        """
        # Find center of gravity of parotid
        centerGravity = _calculateCenterOfGravity(parotidMask3M)

        # Split mandible into left and right at the z-slice of center of gravity
        leftMandibleMask, rightMandibleMask = _splitStructure2D(mandibleMask3M[:, :, centerGravity[-1]])

        # Select appropriate mandible side
        mandibleSide = leftMandibleMask if side == "L" else rightMandibleMask

        # Find intersection point of parotid, masseter, and mandible
        intersection = _findIntersection(parotidMask3M, mandibleSide, masseterMask3M, centerGravity)

        # Convert margins from mm to voxels
        marginsVoxels = (
            marginsMm["x"] / spacing[0],
            marginsMm["y"] / spacing[1],
            marginsMm["z"] / spacing[2],
        )

        # Apply centroid shift
        shift = shiftMm[side]
        shiftedCentroid = intersection + np.array([
            shift["x"] / spacing[0],
            shift["y"] / spacing[1],
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

        return mask3M, originalOrient


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
            numpy.ndarray: (x, y, z) coordinates of the centroid

        Raises:
            ValueError: if contour is empty
        """
        if not np.any(contour):
            raise ValueError("Cannot calculate center of gravity for empty contour")

        indices = np.indices(contour.shape)
        totalMass = np.sum(contour)
        center = np.array([np.sum(indices[axis] * contour) / totalMass for axis in range(3)])

        return center.astype(int)


    def _splitStructure2D(mask):
        """This routine splits a 2D mask into left and right halves at the midline.

        Used for splitting mandible into left and right components.

        Args:
            mask (numpy.ndarray): 2D binary mask array

        Returns:
            tuple: (leftMask, rightMask) as numpy arrays
        """
        height, width = mask.shape
        splitCol = width // 2

        leftMask = mask.copy()
        leftMask[:, :splitCol] = 0

        rightMask = mask.copy()
        rightMask[:, splitCol:] = 0

        return leftMask, rightMask


    def _findClosestPointToStructures(parotidMask, masseterMask, mandibleMask):
        """This routine finds the point with minimum total distance to all three structures.

        Prioritizes points within the parotid mask.

        Args:
            parotidMask (numpy.ndarray): 2D binary mask of parotid
            masseterMask (numpy.ndarray): 2D binary mask of masseter
            mandibleMask (numpy.ndarray): 2D binary mask of mandible

        Returns:
            numpy.ndarray: 2D coordinates [x, y] of the optimal point

        Raises:
            ValueError: if any mask is empty
        """
        parotidCoords = np.argwhere(parotidMask)
        masseterCoords = np.argwhere(masseterMask)
        mandibleCoords = np.argwhere(mandibleMask)

        if len(parotidCoords) == 0:
            raise ValueError("Parotid mask is empty on this slice")
        if len(masseterCoords) == 0:
            raise ValueError("Masseter mask is empty on this slice")
        if len(mandibleCoords) == 0:
            raise ValueError("Mandible mask is empty on this slice")

        minDistances = []
        for point in parotidCoords:
            distToMasseter = np.min(np.linalg.norm(masseterCoords - point, axis=1))
            distToMandible = np.min(np.linalg.norm(mandibleCoords - point, axis=1))
            totalDist = distToMasseter + distToMandible
            minDistances.append(totalDist)

        bestIdx = np.argmin(minDistances)

        return parotidCoords[bestIdx]


    def _findIntersection(parotidMask, mandibleMask, masseterMask, centerGravity):
        """This routine finds the intersection point of three anatomical structures.

        If no intersection exists, finds the point with minimum total distance
        to all three structures, preferring points within the parotid mask.

        Args:
            parotidMask (numpy.ndarray): 3D parotid gland mask
            mandibleMask (numpy.ndarray): 2D mandible mask (single slice)
            masseterMask (numpy.ndarray): 3D masseter muscle mask
            centerGravity (numpy.ndarray): center of gravity coordinates [x, y, z]

        Returns:
            numpy.ndarray: 3D coordinates [x, y, z] of the intersection or closest point
        """
        zSlice = centerGravity[-1]

        parotidSlice = parotidMask[:, :, zSlice].astype(bool)
        masseterSlice = masseterMask[:, :, zSlice].astype(bool)

        intersection = np.logical_and(
            np.logical_and(parotidSlice, masseterSlice), mandibleMask
        )

        intersectionCoords = np.argwhere(intersection)

        if intersectionCoords.size > 0:
            logger.info(f"Intersection found at coordinates: {intersectionCoords}")
            result = intersectionCoords[0]
        else:
            logger.info("No intersection found. Calculating closest point to all three structures...")
            result = _findClosestPointToStructures(
                parotidSlice, masseterSlice, mandibleMask
            )

        return np.append(result, zSlice)


    def _createVolume(center, shape, margins):
        """This routine creates an ellipsoid volume around a center point.

        Args:
            center (numpy.ndarray): 3D coordinates [x, y, z] of ellipsoid center
            shape (tuple): shape of the output volume
            margins (tuple): semi-axes lengths (x, y, z) in voxels

        Returns:
            numpy.ndarray: binary volume with ellipsoid region set to 1
        """
        xMargin, yMargin, zMargin = margins

        xCoords, yCoords, zCoords = np.indices(shape)

        xCentered = xCoords - center[0]
        yCentered = yCoords - center[1]
        zCentered = zCoords - center[2]

        mask = ((xCentered / xMargin) ** 2 + (yCentered / yMargin) ** 2 + (zCentered / zMargin) ** 2) <= 1

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

    # Get structure indices
    parotidInds = _getMatchingInds(structNameDict["parotid"], planC)
    masseterInds = _getMatchingInds(structNameDict["masseter"], planC)
    mandibleInd = _getMatchingInds(structNameDict["mandible"], planC)

    if not parotidInds:
        raise ValueError(
            f"No matching structures found for parotid names: {structNameDict['parotid']}"
        )
    if not masseterInds:
        raise ValueError(
            f"No matching structures found for masseter names: {structNameDict['masseter']}"
        )
    if not mandibleInd:
        raise ValueError(
            f"No matching structure found for mandible name: {structNameDict['mandible']}"
        )

    # Derive scanNum from structure if not provided
    if scanNum is None:
        scanNum = cerrScn.getScanNumFromUID(
            planC.structure[parotidInds[0]].assocScanUID, planC
        )
    spacing = planC.scan[scanNum].getScanSpacing() * 10     # Get voxel spacing in mm

    # Get masks in LPS orientation
    parotidMask3M, originalOrient = _getLpsMask(parotidInds, planC)
    masseterMask3M, _ = _getLpsMask(masseterInds, planC)
    mandibleMask3M, _ = _getLpsMask(mandibleInd, planC)

    # Extract left parotid SCRR
    leftSCRR = _extractSingleSCRR(
        parotidMask3M, masseterMask3M, mandibleMask3M,
        spacing, "L", marginsMm, shiftMm, pixelValue=1
    )

    # Extract right parotid SCRR
    rightSCRR = _extractSingleSCRR(
        parotidMask3M, masseterMask3M, mandibleMask3M,
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