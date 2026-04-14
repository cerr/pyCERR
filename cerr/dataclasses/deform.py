"""structure module.

Ths deform module defines metadata for deformation (REG).
The metadata are attributes of the Deform class.
This module also defines routines for transforming and
accessing the Deformation Vector Field metadata in CERR coordinate system.

"""

from dataclasses import dataclass, field
import numpy as np
import os
from pydicom import dcmread
from cerr.dataclasses import scan as scn
from cerr.utils import uid
import json

def get_empty_list():
    """Return an empty list.

    Used as a ``default_factory`` for dataclass fields that require a mutable
    list default.

    Returns:
        list: An empty list ``[]``.
    """
    return []

def get_empty_np_array():
    """Return an empty 3-D NumPy array with shape (0, 0, 0).

    Used as a ``default_factory`` for dataclass fields that require a mutable
    NumPy array default.

    Returns:
        np.ndarray: A zero-element array with shape ``(0, 0, 0)``.
    """
    return np.empty((0,0,0))

@dataclass
class Deform:
    baseScanUID: str = ""
    movScanUID: str = ""
    algorithm: str = ""
    algorithmParams: dict = field(default_factory=dict)
    deformParams: dict = field(default_factory=dict)
    deformUID: str = ""
    registrationTool: str = ""
    deformOutFileType: str = ""
    deformOutFilePath: str = ""
    dvfMatrix: np.ndarray = field(default_factory=get_empty_np_array)
    xOffset: float = 0.0
    yOffset: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    imageOrientationPatient: np.array = field(default_factory=get_empty_np_array)
    imagePositionPatientV: np.array = field(default_factory=get_empty_np_array)
    zValuesV: np.ndarray = field(default_factory=get_empty_np_array)
    Image2PhysicalTransM: np.ndarray = field(default_factory=get_empty_np_array)
    Image2VirtualPhysicalTransM: np.ndarray = field(default_factory=get_empty_np_array)
    cerrToDcmTransM:  np.ndarray = field(default_factory=get_empty_np_array)


    def convertDcmToCerrVirtualCoords(self):
        """Compute and store coordinate-system transformation matrices for the DVF.

        Builds the affine mapping from DICOM image indices to DICOM physical
        (patient) coordinates (``Image2PhysicalTransM``) and to pyCERR's virtual
        physical coordinate system (``Image2VirtualPhysicalTransM``), accounting
        for the possibility that CERR's slice ordering is the reverse of DICOM's.
        Also populates ``xOffset``, ``yOffset``, and ``cerrToDcmTransM`` (the
        transformation that converts pyCERR xyz coordinates in cm back to DICOM
        physical coordinates in mm).

        The method operates entirely on the instance attributes already set
        (``imageOrientationPatient``, ``imagePositionPatientV``, ``dvfMatrix``,
        ``dx``, ``dy``) and updates the following attributes in-place:

        Attributes set:
            xOffset (float): X-coordinate of the DVF volume centre in the CERR
                coordinate system (cm).
            yOffset (float): Y-coordinate of the DVF volume centre in the CERR
                coordinate system (cm, sign-flipped relative to DICOM column
                direction).
            Image2PhysicalTransM (np.ndarray): 4×4 affine from DICOM image
                indices to DICOM physical coordinates (cm).
            Image2VirtualPhysicalTransM (np.ndarray): 4×4 affine from DICOM
                image indices to pyCERR virtual physical coordinates (cm).
            cerrToDcmTransM (np.ndarray): 4×4 matrix converting pyCERR xyz
                (cm) to DICOM physical coordinates (mm).
        """
        # Construct DICOM Affine transformation matrix
        # To construct DICOM affine transformation matrix it is necessary to figure out
        # whether CERR slice direction matches DICOM to get the position of the 1st slice
        # according to DICOM convention. Since slices are sorted according to decreasing order of
        # dot product between ImagePositionPatient and ImageOrientationPatient.
        #
        # Determining order of scanArray slices
        # If (slice_normal . ipp_2nd_slice - slice_normal . ipp_1st_slice) > 0,
        # then DICOM slice order is reverse of CERR scanArray and scanInfo.
        # i.e. the 1st slice in DICOM will correspond to the last slice in
        # scanArray and the last element in scanInfo.
        # Compute slice normal
        dcmImgOri = self.imageOrientationPatient
        dcmImgOri = dcmImgOri.reshape(6,1)
        # slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
        #        - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]
        # slice_normal = slice_normal.reshape((1,3))
        # zDiff = np.matmul(slice_normal, self.scanInfo[1].imagePositionPatient) - np.matmul(slice_normal, self.scanInfo[0].imagePositionPatient)
        # ippDiffV = self.scanInfo[1].imagePositionPatient - self.scanInfo[0].imagePositionPatient

        if flipSliceOrderFlag(self): # np.all(np.sign(zDiff) < 0):
            pos1V = self.imagePositionPatientV[-1,:] / 10  # cm
            pos2V = self.imagePositionPatientV[-2,:] / 10  # cm
        else:
            pos1V = self.imagePositionPatientV[0,:] / 10  # cm
            pos2V = self.imagePositionPatientV[1,:] / 10  # cm

        deltaPosV = pos2V - pos1V
        pixelSpacing = [self.dx, self.dy]

        # Transformation for DICOM Image to DICOM physical coordinates
        # Pt coordinate to DICOM image coordinate mapping
        # Based on ref: https://nipy.org/nibabel/dicom/dicom_orientation.html
        position_matrix = np.hstack((np.matmul(dcmImgOri.reshape(3, 2,order="F"),np.diag(pixelSpacing)),
                                    np.array([[deltaPosV[0], pos1V[0]], [deltaPosV[1], pos1V[1]], [deltaPosV[2], pos1V[2]]])))

        position_matrix = np.vstack((position_matrix, np.array([0, 0, 0, 1])))

        positionMatrixInv = np.linalg.inv(position_matrix)
        self.Image2PhysicalTransM = position_matrix

        # Get DICOM x,y,z coordinates of the center voxel.
        # This serves as the reference point for the image volume.
        sizV = self.dvfMatrix.shape
        xyzCtrV = position_matrix * np.array([(sizV[1] - 1) / 2, (sizV[0] - 1) / 2, 0, 1])
        self.xOffset = np.sum(np.matmul(np.transpose(dcmImgOri[:3,:]), xyzCtrV[:3]))
        self.yOffset = -np.sum(np.matmul(np.transpose(dcmImgOri[3:,:]), xyzCtrV[:3]))  # (-)ve since CERR y-coordinate is opposite of column vector.

        xs, ys, zs = self.getDVFXYZVals()
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        slice_distance = zs[1] - zs[0]
        # Transformation for DICOM Image to CERR physical coordinates
        # DICOM 1st slice is CERR's last slice (i.e. zs[-1]
        if flipSliceOrderFlag(self): #np.all(np.sign(zDiff) < 0):
            virPosMtx = np.array([[dx, 0, 0, xs[0]], [0, dy, 0, ys[0]], [0, 0, -slice_distance, zs[-1]], [0, 0, 0, 1]])
        else:
            virPosMtx = np.array([[dx, 0, 0, xs[0]], [0, dy, 0, ys[0]], [0, 0, slice_distance, zs[0]], [0, 0, 0, 1]])
        self.Image2VirtualPhysicalTransM = virPosMtx

        # Construct transformation matrix to convert cerr-xyz to dicom-xyz
        self.cerrToDcmTransM = np.matmul(self.Image2PhysicalTransM, np.linalg.inv(self.Image2VirtualPhysicalTransM))
        self.cerrToDcmTransM[:,:3] = self.cerrToDcmTransM[:,:3] * 10 # cm to mm

    def getDVFXYZVals(self):
        """Compute the x, y, and z coordinate vectors for the DVF grid.

        Derives spatial coordinate arrays from the stored grid offsets
        (``xOffset``, ``yOffset``), voxel spacings (``dx``, ``dy``), and
        per-slice z-values (``zValuesV``) based on the dimensions of
        ``dvfMatrix``.

        Returns:
            tuple: A 3-tuple ``(xvals, yvals, zvals)`` where

            - **xvals** (*np.ndarray*): 1-D array of x-coordinates (cm) for
              each column of the DVF grid, increasing left-to-right.
            - **yvals** (*np.ndarray*): 1-D array of y-coordinates (cm) for
              each row of the DVF grid, decreasing top-to-bottom (CERR
              convention).
            - **zvals** (*np.ndarray*): 1-D array of z-coordinates (cm) for
              each slice, taken directly from ``self.zValuesV``.
        """
        numRows, numCols, numSlcs, _ = self.dvfMatrix.shape
        numCols = numCols  -1
        numRows = numRows - 1

        # Calculate xVals
        xvals = np.arange(self.xOffset - (numCols * self.dx) / 2,
                  self.xOffset + (numCols * self.dx) / 2 + self.dx,
                  self.dx)

        # Calculate yVals (flipped left-right)
        yvals = np.arange(self.yOffset + (numRows * self.dy) / 2,
                  self.yOffset - (numRows * self.dy) / 2 - self.dy,
                  -self.dy)

        # Extract zValues from the scanStruct dictionary or object
        zvals = self.zValuesV

        return (xvals,yvals,zvals)

    def getDeformDict(self):
        """Return a shallow copy of the Deform instance's attribute dictionary.

        Returns:
            dict: A dictionary mapping each attribute name to its current value
            for this ``Deform`` instance.
        """
        deformDict = self.__dict__.copy()
        return deformDict

def flipSliceOrderFlag(deform):
    """Determine whether the slice ordering in the Deform object is reversed relative to DICOM.

    Computes the slice normal from the image orientation cosines and projects
    consecutive ``imagePositionPatient`` vectors onto it.  A negative dot-product
    difference indicates that DICOM slices are stored in the opposite order from
    pyCERR's internal convention.

    Args:
        deform (Deform): A ``Deform`` dataclass instance whose
            ``imageOrientationPatient`` (shape ``(6,)``) and
            ``imagePositionPatientV`` (shape ``(N, 3)``) attributes are
            already populated.

    Returns:
        bool: ``True`` if the slice order should be flipped (i.e. DICOM stores
        slices in descending z-order relative to pyCERR), ``False`` otherwise.
    """
    dcmImgOri = deform.imageOrientationPatient
    slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
           - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]
    slice_normal = slice_normal.reshape((1,3))
    zDiff = np.matmul(slice_normal, deform.imagePositionPatientV[1,:]) - np.matmul(slice_normal, deform.imagePositionPatientV[0,:])
    ippDiffV = deform.imagePositionPatientV[1,:] - deform.imagePositionPatientV[0,:]
    return np.all(np.sign(zDiff) < 0)
