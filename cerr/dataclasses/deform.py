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
    return []
def get_empty_np_array():
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
        deformDict = self.__dict__.copy()
        return deformDict

def flipSliceOrderFlag(deform):
    dcmImgOri = deform.imageOrientationPatient
    slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
           - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]
    slice_normal = slice_normal.reshape((1,3))
    zDiff = np.matmul(slice_normal, deform.imagePositionPatientV[1,:]) - np.matmul(slice_normal, deform.imagePositionPatientV[0,:])
    ippDiffV = deform.imagePositionPatientV[1,:] - deform.imagePositionPatientV[0,:]
    return np.all(np.sign(zDiff) < 0)
