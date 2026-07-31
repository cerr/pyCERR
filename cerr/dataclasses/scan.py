"""scan module.

This module defines pyCERR data object for images (CT, MR, PT, US).
Metadata can be imported from various file formats such as DICOM, NifTi.
It also provides methods to transform the Scan object to other formats such NifTi, SimpleITK
and for converting images to real world units and SUV calculation.

"""

from dataclasses import dataclass, field
import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
import cerr.dataclasses.scan_info as scn_info
import SimpleITK as sitk
import datetime
import json
import os
import re
import warnings

def get_empty_list():
    """Return an empty list, used as a default factory for dataclass fields.

    Returns:
        list: An empty list.
    """
    return []

def get_empty_np_array():
    """Return an empty 3-D numpy array, used as a default factory for dataclass fields.

    Returns:
        np.ndarray: A zero-element array with shape (0, 0, 0).
    """
    return np.empty((0,0,0))

@dataclass
class Scan:
    """This class defines data object for volumetric images such as CT, MR, PET or derived image type.

    Attributes:
        scanArray (np.ndarray): numpy array for the image.
        scanType (str): Type of scan. e.g. 'CT SCAN'
        scanInfo (cerr.dataclasses.scan_info.ScanInfo): scan_info object containing metadata for each scan slice
        scanUID (str): unique identifier for each scan.
        assocDeformUID (str): optional, UID of associated deformation object that was used to generate this scan.
        assocTextureUID (str): optional, UID of associated texture object that was used to generate this scan.
        assocBaseScanUID (str): optional, UID of associated base scan in the deformation that was used to generate this scan.
        assocMovingScanUID (str): optional, UID of associated moving scan in the deformation that was used to generate this scan.
        Image2PhysicalTransM (np.ndarray): Transformation matrix to convert pyCERR row,col,slc to DICOM physical coordinates.
        Image2VirtualPhysicalTransM (np.ndarray): Transformation matrix to convert pyCERR's scan row,col,slc to pyCERR virtual coordinates.
        cerrToDcmTransM (np.ndarray): Transformation matrix to convert pyCERR virtual x,y,z coordinates to DICOM physical coordinates.

    """

    scanArray: np.ndarray = field(default_factory=get_empty_np_array)
    scanType: str = ''
    scanInfo: scn_info.ScanInfo = field(default_factory=list)
    uniformScanInfo: scn_info.UniformScanInfo = field(default_factory=get_empty_list)
    scanArraySuperior: np.ndarray = field(default_factory=get_empty_np_array)
    scanArrayInferior: np.ndarray = field(default_factory=get_empty_np_array)
    thumbnails: np.ndarray = field(default_factory=get_empty_np_array)
    transM: np.ndarray = field(default_factory=get_empty_np_array)
    scanUID: str = ''
    assocDeformUID: str = ''
    assocTextureUID: str = ''
    assocBaseScanUID: str = ''
    assocMovingScanUID: str = ''
    Image2PhysicalTransM: np.ndarray = field(default_factory=get_empty_np_array)
    Image2VirtualPhysicalTransM: np.ndarray = field(default_factory=get_empty_np_array)
    cerrToDcmTransM:  np.ndarray = field(default_factory=get_empty_np_array)

    def __getitem__(self, key):
        """Return the value of the named attribute.

        Args:
            key (str): Attribute name to retrieve.

        Returns:
            Any: Value of the requested attribute.
        """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Set the value of the named attribute.

        Args:
            key (str): Attribute name to set.
            value (Any): Value to assign to the attribute.
        """
        return setattr(self, key, value)

    class json_serialize(json.JSONEncoder):
        def default(self, obj):
            """Serialize a Scan instance to a JSON-compatible dictionary.

            Args:
                obj (Any): Object to serialize.  When the object is an instance
                    of :class:`Scan` its ``scanUID`` is returned; all other types
                    fall back to an empty string.

            Returns:
                str | dict: ``{'scan': obj.scanUID}`` for Scan instances,
                    otherwise an empty string.
            """
            if isinstance(obj, Scan):
                return {'scan':obj.scanUID}
            return "" #json.JSONEncoder.default(self, obj)

    def getScanArray(self):
        """ Routine to obtain image in the units defined in planC.scan[scanNum].scanInfo[slcNum].imageUnits
        Returns:
             np.ndarray: CTOffset is added to to scanArray such that the resulting array is in
                         real world units such as HU, SUV
        """

        scan3M = self.scanArray - self.scanInfo[0].CTOffset
        return scan3M

    def getNiiAffine(self):
        """ Routine for affine transformation of pyCERR scan object for storing in NifTi format

        Returns:
            np.ndarray: 3x3 affine matrix
        """
        # https://neurostars.org/t/direction-orientation-matrix-dicom-vs-nifti/14382/2
        affine3M = self.Image2PhysicalTransM.copy()
        affine3M[0,:] = -affine3M[0,:] * 10 #nii row is reverse of dicom, cm to mm
        affine3M[1,:] = -affine3M[1,:] * 10 #nii col is reverse of dicom, cm to mm
        affine3M[2,:] = affine3M[2,:] * 10 # cm to mm
        return affine3M

    def saveNii(self, niiFileName):
        """ Routine to save pyCERR Scan object to NifTi file

        Args:
            niiFileName (str): File name including the full path to save the pyCERR scan object to NifTi file.

        Returns:
            int: 0 when NifTi file is written successfully.
        """

        img = self.getSitkImage()
        sitk.WriteImage(img, niiFileName)

        return 0

        # affine3M = self.getNiiAffine()
        # scan3M = self.getScanArray()
        # scan3M = np.moveaxis(scan3M,[0,1],[1,0])
        # #scan3M = np.flip(scan3M,axis=[0,1]) # negated affineM to take care of reverse row/col compared to dicom
        # # Determine whether CERR slice order matches DICOM
        # # dcmImgOri = self.scanInfo[0].imageOrientationPatient
        # # slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
        # #        - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]
        # # slice_normal = slice_normal.reshape((1,3))
        # # zDiff = np.matmul(slice_normal, self.scanInfo[1].imagePositionPatient) - np.matmul(slice_normal, self.scanInfo[0].imagePositionPatient)
        # # ippDiffV = self.scanInfo[1].imagePositionPatient - self.scanInfo[0].imagePositionPatient
        # if flipSliceOrderFlag(self): #np.all(np.sign(zDiff) < 0):
        #     scan3M = np.flip(scan3M,axis=2) # CERR slice ordering is opposite of DICOM
        # img = nib.Nifti1Image(scan3M, affine3M)
        # success = nib.save(img, niiFileName)
        # return success

    def getSitkImage(self):
        """ Routine to convert pyCERR Scan object to SimpleITK Image object

        Returns:
            sitk.Image: SimpleITK Image

        """

        #sitkArray = np.moveaxis(self.getScanArray(),[0,1,2],[1,2,0])
        sitkArray = np.transpose(self.getScanArray(), (2, 0, 1)) # z,y,x order
        # CERR slice ordering is opposite of DICOM
        if flipSliceOrderFlag(self):
            sitkArray = np.flip(sitkArray, axis = 0)
        originXyz = list(np.matmul(self.Image2PhysicalTransM, np.asarray([0,0,0,1]).T)[:3] * 10)
        xV, yV, zV = self.getScanXYZVals()
        dx = np.abs(xV[1] - xV[0]) * 10
        dy = np.abs(yV[1] - yV[0]) * 10
        dz = np.abs(zV[1] - zV[0]) * 10
        spacing = [dx, dy, dz]
        img_ori = self.scanInfo[0].imageOrientationPatient
        slice_normal = img_ori[[1,2,0]] * img_ori[[5,3,4]] \
                       - img_ori[[2,0,1]] * img_ori[[4,5,3]]
        # Get row-major directions for ITK
        dir_cosine_mat = np.hstack((img_ori.reshape(3,2,order="F"),slice_normal.reshape(3,1)))
        direction = dir_cosine_mat.reshape(9,order='C')
        img = sitk.GetImageFromArray(sitkArray)
        img.SetOrigin(originXyz)
        img.SetSpacing(spacing)
        img.SetDirection(direction)
        return img


    def getScanXYZVals(self):
        """ Routine to obtain pyCERR scan object's x,y,z grid coordinates. The coordinates are in pyCERR's
        virtual coordinate system.

        Returns:
            tuple: x, y, z coordinates corresponding to the columns, rows, slices of scan voxels

        """
        scan_info = self.scanInfo[0]
        sizeDim1 = scan_info.sizeOfDimension1-1
        sizeDim2 = scan_info.sizeOfDimension2-1

        # Add gridUnits/2 to the last value to account for numerical noise

        # Calculate xVals
        xvals = np.arange(scan_info.xOffset - (sizeDim2 * scan_info.grid2Units) / 2,
                  scan_info.xOffset + (sizeDim2 * scan_info.grid2Units) / 2 + scan_info.grid2Units/2,
                  scan_info.grid2Units)

        # Calculate yVals (flipped left-right)
        yvals = np.arange(scan_info.yOffset + (sizeDim1 * scan_info.grid1Units) / 2,
                  scan_info.yOffset - (sizeDim1 * scan_info.grid1Units) / 2 - scan_info.grid1Units/2,
                  -scan_info.grid1Units)

        # Extract zValues from the scanStruct dictionary or object
        zvals = np.asarray([si.zValue for si in self.scanInfo])

        return (xvals,yvals,zvals)

    def getScanSize(self):
        """ Routine to get scan dimensions.

        Returns:
            np.array:  numRows, numCols, numSlcs of pyCERR scan object

        """
        numRows, numCols, numSlcs = self.scanInfo[0].sizeOfDimension1, self.scanInfo[0].sizeOfDimension2, \
                                    len(self.scanInfo)
        return np.asarray([numRows, numCols, numSlcs])


    def getUniformScanSize(self):
        """ Return the size of the uniformized scan.

        Returns:
            np.array:  numRows, numCols, numSlcs

        """
        uniformScanInfo = self.uniformScanInfo
        scanInfo = self.scanInfo[0]

        # No. slices in uniformized set.
        nCTSlices = abs(uniformScanInfo.sliceNumSup - uniformScanInfo.sliceNumInf) + 1;
        #Use scan access function in case of remote variables.
        scanArraySup = self.scanArraySuperior
        scanArrayInf = self.scanArrayInferior
        nSupSlices = scanArraySup.shape[2]
        if scanArraySup.size==0:
            nSupSlices = 0
        nInfSlices = scanArrayInf.shape[2]
        if scanArrayInf.size==0:
            nInfSlices = 0
        zSize = nCTSlices + nSupSlices + nInfSlices
        xSize = scanInfo[0].sizeOfDimension2
        ySize = scanInfo[0].sizeOfDimension1

        return ySize, xSize, zSize


    def getScanOrientation(self):
        """ Routine to get orientation of sacn w.r.t. patient.

        Returns:
            str: 3-character String representing the orientation of Scans's row, column and slice.

        """

        orientPos = ['L', 'P', 'S']
        orientNeg = ['R', 'A', 'I']
        flipDict = {}
        for i in range(len(orientPos)):
            flipDict[orientPos[i]] = orientNeg[i]
            flipDict[orientNeg[i]] = orientPos[i]
        img_ori = self.scanInfo[0].imageOrientationPatient
        img_ori = img_ori.reshape(6,1)
        slice_normal = img_ori[[1,2,0]] * img_ori[[5,3,4]] \
                       - img_ori[[2,0,1]] * img_ori[[4,5,3]]
        slice_normal = slice_normal.reshape((1,3))
        # img_ori = np.vstack((img_ori, slice_normal.reshape((3,1))))
        # dir_cosine_mat = img_ori.reshape(3, 3,order="F")
        # itk_direction = dir_cosine_mat.reshape(9, order="C")
        itk_direction = getITKDirection(self)
        itk_orient_str = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(itk_direction)
        zDiff = np.matmul(slice_normal, self.scanInfo[1].imagePositionPatient) - np.matmul(slice_normal, self.scanInfo[0].imagePositionPatient)
        ippDiffV = self.scanInfo[1].imagePositionPatient - self.scanInfo[0].imagePositionPatient
        if np.all(np.sign(zDiff) < 0):
            # cerr slice direction is opposite to ITK/DICOM order. Hence, flip.
            zOri = flipDict[itk_orient_str[-1]]
        else:
            # cerr slice direction is opposite to ITK/DICOM order
            zOri = itk_orient_str[-1]
        orientString = itk_orient_str[:2]
        orientString = orientString + zOri
        return orientString

    def getScanSpacing(self):
        """ Routine to get voxel spacing in cm.

        Returns:
            np.array: 3-element array containing dx, dy, dz of scan

        """

        x_vals_v, y_vals_v, z_vals_v = self.getScanXYZVals()
        if y_vals_v[0] > y_vals_v[1]:
            y_vals_v = np.flip(y_vals_v)
        dx = abs(np.median(np.diff(x_vals_v)))
        dy = abs(np.median(np.diff(y_vals_v)))
        dz = abs(np.median(np.diff(z_vals_v)))
        spacing_v = np.array([dx, dy, dz])
        return spacing_v


    def convertDcmToCerrVirtualCoords(self):
        """Routine to get scan from DICOM to pyCERR virtual coordinates. More information
        about virtual coordinates is on the Wiki https://github.com/cerr/pyCERR/wiki/Coordinate-system
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
        dcmImgOri = self.scanInfo[0].imageOrientationPatient
        dcmImgOri = dcmImgOri.reshape(6,1)
        # slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
        #        - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]
        # slice_normal = slice_normal.reshape((1,3))
        # zDiff = np.matmul(slice_normal, self.scanInfo[1].imagePositionPatient) - np.matmul(slice_normal, self.scanInfo[0].imagePositionPatient)
        # ippDiffV = self.scanInfo[1].imagePositionPatient - self.scanInfo[0].imagePositionPatient
        if len(self.scanInfo) < 2:
            # Single-slice scan: no neighbouring slice to difference, so derive
            # the slice spacing from the nominal slice thickness projected along
            # the slice normal (row cosine x column cosine).
            info1 = self.scanInfo[0]
            pos1V = info1.imagePositionPatient / 10  # cm
            ori = info1.imageOrientationPatient
            sliceNormal = ori[[1, 2, 0]] * ori[[5, 3, 4]] \
                - ori[[2, 0, 1]] * ori[[4, 5, 3]]
            sliceThickness = info1.sliceThickness
            if sliceThickness is None or np.isnan(sliceThickness) \
                    or sliceThickness == 0:
                sliceThickness = 1.0  # mm fallback when (0018,0050) is absent
            deltaPosV = sliceNormal * (sliceThickness / 10)  # cm
        else:
            if flipSliceOrderFlag(self): # np.all(np.sign(zDiff) < 0):
                info1 = self.scanInfo[-1]
                info2 = self.scanInfo[-2]
            else:
                info1 = self.scanInfo[0]
                info2 = self.scanInfo[1]
            pos1V = info1.imagePositionPatient / 10  # cm
            pos2V = info2.imagePositionPatient / 10  # cm
            deltaPosV = pos2V - pos1V
        pixelSpacing = [info1.grid2Units, info1.grid1Units]

        # Transformation for DICOM Image to DICOM physical coordinates
        # Pt coordinate to DICOM image coordinate mapping
        # Based on ref: https://nipy.org/nibabel/dicom/dicom_orientation.html
        position_matrix = np.hstack((np.matmul(dcmImgOri.reshape(3, 2,order="F"),np.diag(pixelSpacing)),
                                    np.array([[deltaPosV[0], pos1V[0]], [deltaPosV[1], pos1V[1]], [deltaPosV[2], pos1V[2]]])))

        position_matrix = np.vstack((position_matrix, np.array([0, 0, 0, 1])))

        #positionMatrixInv = np.linalg.inv(position_matrix)
        self.Image2PhysicalTransM = position_matrix

        # Get DICOM x,y,z coordinates of the center voxel.
        # This serves as the reference point for the image volume.
        sizV = self.scanArray.shape
        xyzCtrV = position_matrix * np.array([(sizV[1] - 1) / 2, (sizV[0] - 1) / 2, 0, 1])
        xOffset = np.sum(np.matmul(np.transpose(dcmImgOri[:3,:]), xyzCtrV[:3]))
        yOffset = -np.sum(np.matmul(np.transpose(dcmImgOri[3:,:]), xyzCtrV[:3]))  # (-)ve since CERR y-coordinate is opposite of column vector.

        for i in range(len(self.scanInfo)):
            self.scanInfo[i].xOffset = xOffset
            self.scanInfo[i].yOffset = yOffset

        xs, ys, zs = self.getScanXYZVals()
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        if len(zs) > 1:
            slice_distance = zs[1] - zs[0]
        else:
            slice_distance = float(np.linalg.norm(deltaPosV))  # single slice
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


    def convertDcmToRealWorldUnits(self, opts={}):
        """ Routine to convert pixel array from DICOM storage units to real world units.

        Args:
            opts (dict): Dictionary of options to convert to real world units. Currrently, only one option
             if supported - importMRPreciseValueFlag (yes or no) to specify whether to convert MR image from
              Philips scanner to precise values.

        """

        importMRPreciseValueFlag = 'no'
        if 'importMRPreciseValueFlag' in opts:
            importMRPreciseValueFlag = opts['importMRPreciseValueFlag']

        # Apply ReScale Intercept and Slope
        scanArray3M = np.zeros(self.scanArray.shape, dtype=np.float32)
        numSlcs = self.scanArray.shape[2]
        rescaleSlopeV = np.ones(numSlcs)
        realWorldImageFlag = False

        for slcNum in range(numSlcs):
            rescaleSlope = self.scanInfo[slcNum].rescaleSlope
            rescaleIntrcpt = self.scanInfo[slcNum].rescaleIntercept
            realWorldValueSlope = self.scanInfo[slcNum].realWorldValueSlope
            realWorldValueIntercept = self.scanInfo[slcNum].realWorldValueIntercept
            realWorldMeasurCodeMeaning = self.scanInfo[slcNum].realWorldMeasurCodeMeaning
            philipsImageUnits = self.scanInfo[slcNum].philipsImageUnits
            rescaleType = self.scanInfo[slcNum].rescaleType
            manufacturer = self.scanInfo[slcNum].manufacturer

            if manufacturer.lower() in ['philips'] and \
                    realWorldValueSlope is not None and \
                    not np.isnan(realWorldValueSlope) and \
                    self.scanInfo[slcNum].imageType.lower() in ['mr scan'] and \
                    realWorldMeasurCodeMeaning is not None:
                realWorldImageFlag = True
                scanArray3M[:, :, slcNum] = \
                    self.scanArray[:, :, slcNum] * realWorldValueSlope + realWorldValueIntercept
                self.scanInfo[slcNum].imageUnits = realWorldMeasurCodeMeaning
            else:
                scanArray3M[:, :, slcNum] = \
                    self.scanArray[:, :, slcNum] * rescaleSlope + rescaleIntrcpt
                if len(self.scanInfo[slcNum].imageUnits) == 0 and \
                        self.scanInfo[slcNum].imageType.lower() not in ['pt scan', 'nm scan']:
                    self.scanInfo[slcNum].imageUnits = rescaleType

            rescaleSlopeV[slcNum] = rescaleSlope

        minScanVal = np.min(scanArray3M)
        ctOffset = max(0, -minScanVal)
        scanArray3M += ctOffset

        # Decommissioned conversion to unsigned int. Need to update logic to handle various data types - dicom, nii etc.
        # minScanVal = np.min(scanArray3M)
        # maxScanVal = np.max(scanArray3M)
        # if not realWorldImageFlag and not np.any(np.abs(rescaleSlopeV - 1) > np.finfo(float).eps * 1e5):
        #     # Convert to uint if rescale slope is not 1
        #     if minScanVal >= -32768 and maxScanVal <= 32767:
        #         scanArray3M = scanArray3M.astype(np.uint16)
        #     else:
        #         scanArray3M = scanArray3M.astype(np.uint32)

        for slcNum in range(numSlcs):
            self.scanInfo[slcNum].CTOffset = ctOffset

        self.scanArray = scanArray3M

        # Convert Philips MR to precise values
        if self.scanInfo[slcNum].imageType.lower() == 'mr scan' and \
                importMRPreciseValueFlag.lower() == 'yes':
            # Ref: Chenevert, Thomas L., et al. "Errors in quantitative image analysis due to platform-dependent image scaling."
            manufacturer = self.scanInfo[0].manufacturer
            if 'philips' in manufacturer.lower() and \
                    self.scanInfo[0].scaleSlope is not None and \
                    not realWorldImageFlag:
                scaleSlope = self.scanInfo[0].scaleSlope
                self.scanArray = self.scanArray.astype(np.float32) / (rescaleSlope * scaleSlope)

    def convertToSUV(self, suvType='BW'):
        """ Routine to convert pixel array for PET scan from DICOM storage to SUV

        The conversion follows the SUV computation strategy standardized by the
        IBSI-SUV manual (https://oncoray.github.io/suv_computation/suv.html).
        Stored values are first brought to a body-weight SUV, which is
        independent of the time point the values and the dose were corrected to.
        Images already normalized by the scanner (Units 'GML' or 'CM2ML') are
        re-scaled from the scanner's normalization to body weight rather than
        being used as-is. A different normalization can be requested via suvType.

        Args:
            suvType (str): type of SUV to produce. Defaults to 'BW' (body weight),
             regardless of the normalization the scanner applied. Supported options are
             'BW', 'BSA', 'LBM', 'LBMJAMES128', 'LBMJANMA', 'IBW' and 'AS_STORED'.
             'AS_STORED' keeps the normalization the scanner already applied, i.e. the
             SUV Type (0054,1006) of images stored as 'GML'/'CM2ML'; images stored as an
             activity concentration have no applied normalization and yield 'BW'. Because
             it inherits the scanner's protocol, 'AS_STORED' may return a different
             normalization per series and should not be used to pool a cohort.
             None and '' are rejected: an absent option is not a request to skip
             normalization, and silently treating it as either choice would be ambiguous.

        Raises:
            ValueError: if suvType is None, '', or not one of the supported types.

        Note:
            The scan is left unmodified and a warning is issued when the DICOM
            attributes required for the conversion are missing or inconsistent.

        """

        scan3M = self.scanArray
        headerS = self.scanInfo
        suv3M = np.zeros(scan3M.shape)
        numSlcs = scan3M.shape[2]

        # SUV Type (0054,1006) describes the normalization already applied to
        # the stored values; it is not necessarily the type being requested.
        storedSuvType = str(headerS[0].suvType).upper()
        if storedSuvType == '':
            storedSuvType = 'BW'
        # Body weight is the default target since SUVbw is independent of the
        # normalization the scanner happened to use. None/'' are rejected rather
        # than coerced: they cannot be distinguished from an unset option, and
        # guessing between 'BW' and 'AS_STORED' would silently change the values.
        if suvType is None or str(suvType).strip() == '':
            raise ValueError(
                "suvType must be one of " + ', '.join(sorted(SUV_TYPES)) +
                "; None and '' are ambiguous. Use 'BW' for body weight (the "
                "default) or 'AS_STORED' to keep the scanner's normalization.")
        suvType = str(suvType).strip().upper()
        if suvType not in SUV_TYPES:
            raise ValueError(f"Unsupported suvType '{suvType}'. Supported types are "
                             + ', '.join(sorted(SUV_TYPES)) + '.')
        # 'AS_STORED' resolves per frame, since the applied normalization is a
        # property of the stored units rather than of the request.
        asStored = suvType == 'AS_STORED'

        warnMsgs = []
        sliceUnits = [''] * numSlcs
        sliceSuvTypes = [''] * numSlcs
        for slcNum in range(numSlcs):
            headerSlcS = headerS[slcNum]
            imgM = scan3M[:, :, slcNum] - headerSlcS.CTOffset
            imgUnits = str(headerSlcS.imageUnits).upper()

            # Resolve the normalization requested for this frame. Only 'GML' and
            # 'CM2ML' carry a scanner-applied normalization to inherit; an
            # activity concentration has none, so 'AS_STORED' yields body weight.
            if asStored:
                if imgUnits == 'CM2ML':
                    frameSuvType = 'BSA'
                elif imgUnits == 'GML':
                    frameSuvType = storedSuvType
                else:
                    frameSuvType = 'BW'
            else:
                frameSuvType = suvType

            # Step 1: bring the frame to either an activity concentration in
            # Bq/ml, or directly to a body-weight SUV when the scanner already
            # normalized the values.
            activityConcM = None   # Bq/ml
            suvBwM = None          # g/ml
            suvDirectM = None      # already in the requested normalization
            if imgUnits == 'CNTS':
                activityScaleFactor = headerSlcS.philipsActivityConcentrationScaleFactor
                suvScaleFactor = headerSlcS.philipsSUVScaleFactor
                if activityScaleFactor != "":
                    activityConcM = imgM * activityScaleFactor
                elif suvScaleFactor != "":
                    suvBwM = imgM * suvScaleFactor
                else:
                    warnings.warn('SUV computation for Units CNTS requires a Philips '
                                  'activity concentration or SUV scale factor.')
                    return
            elif imgUnits in ['BQML', 'BQCC']:
                activityConcM = imgM
            elif imgUnits in ['KBQCC', 'KBQML']:
                activityConcM = imgM * 1000
            elif imgUnits in ['GML', 'CM2ML']:
                # Values are already normalized. Re-scale to body weight using
                # the normalization the scanner applied, per SUV Type.
                storedType = 'BSA' if imgUnits == 'CM2ML' else storedSuvType
                if frameSuvType == storedType:
                    # Already in the requested normalization. Pass the values
                    # through instead of dividing and re-applying the same
                    # factor, so the conversion needs no patient attributes.
                    suvDirectM = imgM
                else:
                    weightG = getSuvNormalizationFactor('BW', headerSlcS)
                    storedFactor = getSuvNormalizationFactor(storedType, headerSlcS)
                    if weightG is None or storedFactor is None or storedFactor <= 0:
                        warnings.warn('Patient attributes required to convert stored '
                                      f'{storedType} values to SUV are missing or invalid.')
                        return
                    suvBwM = imgM * weightG / storedFactor
            else:
                warnings.warn("'SUV calculation is supported only for imageUnits BQML and CNTS'")
                return

            # Step 2: when starting from an activity concentration, decay-correct
            # the administered dose to the same time point as the voxel values.
            if activityConcM is not None:
                injectedDose = headerSlcS.injectedDose
                if not isinstance(injectedDose, (int, float)) or injectedDose <= 0:
                    warnings.warn('Radionuclide Total Dose is missing or non-positive; '
                                  'SUV cannot be computed.')
                    return
                if str(headerSlcS.imageType).upper() == 'NM SCAN':
                    injectedDose = injectedDose * 1e6  # Convert MBq to Bq
                # Some PT IODs record the dose in MBq rather than Bq.
                if injectedDose < 1e5:
                    injectedDose = injectedDose * 1e6

                decayRefDateTime, errMsg = getDecayReferenceDateTime(headerSlcS)
                if errMsg:
                    warnings.warn(errMsg)
                    return
                if decayRefDateTime is None:
                    # Decay Correction 'ADMIN': dose already matches the values.
                    doseDecayed = injectedDose
                else:
                    admDateTime, warnMsg = getAdministrationDateTime(headerSlcS,
                                                                    decayRefDateTime)
                    if warnMsg:
                        warnMsgs.append(warnMsg)
                    if admDateTime is None:
                        warnings.warn(warnMsg)
                        return
                    halfLife = headerSlcS.halfLife
                    if not isinstance(halfLife, (int, float)) or halfLife <= 0:
                        warnings.warn('Radionuclide Half Life is missing or non-positive; '
                                      'SUV cannot be computed.')
                        return
                    uptakeSecs = (decayRefDateTime - admDateTime).total_seconds()
                    doseDecayed = injectedDose * np.exp(-np.log(2) * uptakeSecs / halfLife)

                weightG = getSuvNormalizationFactor('BW', headerSlcS)
                if weightG is None:
                    warnings.warn("Patient's Weight is missing or non-positive; "
                                  'SUV cannot be computed.')
                    return
                suvBwM = activityConcM * weightG / doseDecayed

            # Step 3: re-normalize the body-weight SUV if another type was asked for.
            if suvDirectM is not None:
                # Passed through unchanged; already in frameSuvType.
                suvM = suvDirectM
                imageUnits = 'CM2ML' if frameSuvType == 'BSA' else 'GML'
            elif frameSuvType == 'BW':
                suvM = suvBwM
                imageUnits = 'GML'
            else:
                weightG = getSuvNormalizationFactor('BW', headerSlcS)
                factor = getSuvNormalizationFactor(frameSuvType, headerSlcS)
                if weightG is None or factor is None:
                    warnings.warn(f'Patient attributes required for SUV type {frameSuvType} '
                                  'are missing or invalid.')
                    return
                suvM = suvBwM * factor / weightG
                imageUnits = 'CM2ML' if frameSuvType == 'BSA' else 'GML'

            suv3M[:, :, slcNum] = suvM
            sliceUnits[slcNum] = imageUnits
            sliceSuvTypes[slcNum] = frameSuvType

        for warnMsg in set(warnMsgs):
            warnings.warn(warnMsg)

        # Commit only once every slice converted, so that an early return above
        # leaves the scan and its metadata consistently unconverted.
        for slcNum in range(numSlcs):
            self.scanInfo[slcNum].imageUnits = sliceUnits[slcNum]
            # Record the normalization actually applied, so 'AS_STORED' resolves
            # to a concrete type rather than leaving the sentinel in metadata.
            self.scanInfo[slcNum].suvType = sliceSuvTypes[slcNum]
            # scanArray now holds SUV directly; the storage offset no longer applies.
            self.scanInfo[slcNum].CTOffset = 0
        self.scanArray = suv3M

        return

    def getScanDict(self):
        """ Routine to get dictionary representation of scan metadata

        Returns:
            dict: fields of the dictionary are attributes of the Scan object.

        """
        scanDict = self.__dict__.copy()
        sInfoList = []
        for sInfo in scanDict['scanInfo']:
            sInfoDict = sInfo.__dict__.copy()
            sInfoList.append(sInfoDict)
        scanDict['scanInfo'] = sInfoList
        return scanDict


    def getDcmScanInfo(self) -> list:
        """This routine return a list of scanInfo from scan object. The order corresponds to scanArray slice dimension

        Returns:
            list: list of scanInfo dictionaries

        """

        scnDict = self.getScanDict()
        scanInfoList = scnDict['scanInfo']
        cerrSpecificKeyList = ['xOffset', 'yOffset',  'CTScale', 'distrustAbove',
                                'imageSource', 'transferProtocol', 'studyNumberOfOrigin',
                               'tapeOfOrigin', 'scanFileName', 'unitNumber', 'CTAir', 'CTWater',
                               'zValue','imageNumber','caseNumber','CTOffset','scanType',
                               'numberRepresentation', 'numberOfDimensions', 'voxelThickness',
                               'headInOut', 'positionInScan', 'patientAttitude', 'frameAcquisitionDuration',
                               'frameReferenceDateTime', 'scanNumber', 'scanID', 'patientBirthDate',
                               'scaleSlope', 'scaleIntercept', 'siteOfInterest','scanDate', 'patientPosition',
                               'philipsImageUnits', 'philipsRescaleSlope', 'philipsRescaleIntercept']
        for scanInfoDict in scanInfoList:
            for cerrKey in cerrSpecificKeyList:
                del scanInfoDict[cerrKey]

        # Create a new dictionary with DICOM names for keys
        scanDir = os.path.dirname(__file__)
        mappingFile = os.path.join(scanDir, 'dcm_cerr_name_map.json')
        with open(mappingFile, 'r') as nameMapFile:
            dcmCerrNameMap = json.load(nameMapFile)

        dcmNameScanInfoList = []
        for scanInfoDict in scanInfoList:
            dcmNameScanInfoDict = dict('')
            for key in scanInfoDict.keys():
                if key in dcmCerrNameMap:
                    dcmNameScanInfoDict[dcmCerrNameMap[key]] = scanInfoDict[key]
                    if isinstance(dcmNameScanInfoDict[dcmCerrNameMap[key]], np.ndarray):
                        dcmNameScanInfoDict[dcmCerrNameMap[key]] = dcmNameScanInfoDict[dcmCerrNameMap[key]].tolist()
            dcmNameScanInfoDict["PixelSpacing"] = [scanInfoDict['grid2Units'], scanInfoDict['grid1Units']]
            dcmNameScanInfoList.append(dcmNameScanInfoDict)

        return dcmNameScanInfoList #scanInfoList


def flipSliceOrderFlag(scan):
    """ Routine to determine slice order for determining the origin for conversion to NifTi and SimpleITK formats.

    Args:
        scan (cerr.dataclasses.scan.Scan): pyCERR scan object

    Returns:
        bool: True when dot product of slice normal and imagePositionPatient increases with slice order

    """

    # A single-slice scan has no slice order to flip.
    if len(scan.scanInfo) < 2:
        return False

    dcmImgOri = scan.scanInfo[0].imageOrientationPatient
    slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
           - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]
    slice_normal = slice_normal.reshape((1,3))
    zDiff = np.matmul(slice_normal, scan.scanInfo[1].imagePositionPatient) - np.matmul(slice_normal, scan.scanInfo[0].imagePositionPatient)
    ippDiffV = scan.scanInfo[1].imagePositionPatient - scan.scanInfo[0].imagePositionPatient
    return np.all(np.sign(zDiff) < 0)

def getITKDirection(scan):
    """

    Args:
        scan (cerr.dataclasses.scan.Scan): pyCERR scan object

    Returns:
        np.ndarray: 9-element array of direction cosines of row, column and slice w.r.t. patient.
    """

    img_ori = scan.scanInfo[0].imageOrientationPatient
    img_ori = img_ori.reshape(6,1)
    slice_normal = img_ori[[1,2,0]] * img_ori[[5,3,4]] \
                   - img_ori[[2,0,1]] * img_ori[[4,5,3]]
    slice_normal = slice_normal.reshape((1,3))
    img_ori = np.vstack((img_ori, slice_normal.reshape((3,1))))
    dir_cosine_mat = img_ori.reshape(3, 3,order="F")
    itk_direction = dir_cosine_mat.reshape(9, order="C")
    return itk_direction


# ---------------------------------------------------------------------------
# SUV computation helpers
#
# The routines below implement the SUV computation strategy described in the
# IBSI-SUV manual (https://oncoray.github.io/suv_computation/suv.html), which
# standardizes how DICOM PET images are converted to standardized uptake values.
# ---------------------------------------------------------------------------

# SUV normalizations accepted by Scan.convertToSUV. 'AS_STORED' is a sentinel
# rather than a normalization: it resolves per frame to whichever type the
# scanner already applied. None and '' are deliberately excluded as ambiguous.
SUV_TYPES = frozenset({'BW', 'BSA', 'LBM', 'LBMJAMES128', 'LBMJANMA', 'IBW',
                       'AS_STORED'})

# Half-life threshold (s) below which a radionuclide is treated as short-lived,
# i.e. the administration date may be inferred from the decay-correction date.
SHORT_HALF_LIFE_THRESH = 41400
# Tolerance (s) for a decay-reference datetime preceding administration.
UPTAKE_TOLERANCE_SECS = -3600

_DCM_DT_RE = re.compile(r'^(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?(\.\d+)?([+-]\d{4})?$')
_DCM_TM_RE = re.compile(r'^(\d{2})(\d{2})?(\d{2})?(\.\d+)?([+-]\d{4})?$')


def parseDcmDateTime(dtStr):
    """Parse a DICOM DT (date-time) string into a python datetime.

    Any timezone offset suffix is ignored: within a PET series all datetimes
    share the same offset, so dropping it leaves their differences unchanged.

    Args:
        dtStr (str): DICOM DT string, e.g. '20250101110000.000000+0100'.

    Returns:
        datetime.datetime | None: parsed datetime, or None when unparseable.
    """
    if dtStr is None:
        return None
    match = _DCM_DT_RE.match(str(dtStr).strip())
    if match is None:
        return None
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    hour = int(match.group(4)) if match.group(4) else 0
    minute = int(match.group(5)) if match.group(5) else 0
    sec = int(match.group(6)) if match.group(6) else 0
    frac = float(match.group(7)) if match.group(7) else 0.0
    try:
        return datetime.datetime(year, month, day, hour, minute, sec) \
            + datetime.timedelta(seconds=frac)
    except ValueError:
        return None


def parseDcmTimeOfDay(tmStr):
    """Parse a DICOM TM (time) string into seconds since midnight.

    Args:
        tmStr (str): DICOM TM string, e.g. '110000.000000'.

    Returns:
        float | None: seconds elapsed since midnight, or None when unparseable.
    """
    if tmStr is None:
        return None
    match = _DCM_TM_RE.match(str(tmStr).strip())
    if match is None:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    sec = int(match.group(3)) if match.group(3) else 0
    frac = float(match.group(4)) if match.group(4) else 0.0
    return hour * 3600 + minute * 60 + sec + frac


def combineDcmDateAndTime(dateStr, timeStr):
    """Combine separate DICOM DA and TM strings into a datetime.

    Args:
        dateStr (str): DICOM DA string (YYYYMMDD).
        timeStr (str): DICOM TM string (HHMMSS.FFFFFF).

    Returns:
        datetime.datetime | None: combined datetime, or None when either part is unparseable.
    """
    if not dateStr or not timeStr:
        return None
    secs = parseDcmTimeOfDay(timeStr)
    if secs is None:
        return None
    dateOnly = parseDcmDateTime(str(dateStr).strip() + '000000')
    if dateOnly is None:
        return None
    return dateOnly + datetime.timedelta(seconds=secs)


def getAverageCountRateTime(frameDurationSecs, halfLife):
    """Return the average count rate time (T_ave) for a PET frame.

    The measured counts of a frame of duration T represent the average activity
    over that frame, which corresponds to a single time point T_ave after the
    frame started:  T_ave = (1/lambda) * ln( lambda*T / (1 - exp(-lambda*T)) ).

    Args:
        frameDurationSecs (float): Actual Frame Duration (0018,1242) in seconds.
        halfLife (float): Radionuclide Half Life (0018,1075) in seconds.

    Returns:
        float: T_ave in seconds; 0 when the frame duration is unavailable.
    """
    # ScanInfo defaults these to '' rather than NaN when the tags are absent.
    if not isinstance(frameDurationSecs, (int, float)) or frameDurationSecs <= 0 \
            or not isinstance(halfLife, (int, float)) or halfLife <= 0:
        return 0.0
    decayConst = np.log(2) / halfLife
    x = decayConst * frameDurationSecs
    if x < 1e-9:
        # Negligible decay over the frame: T_ave tends to the frame midpoint.
        return frameDurationSecs / 2
    return np.log(x / (1 - np.exp(-x))) / decayConst


def getDecayReferenceDateTime(headerSlcS):
    """Return the decay-correction reference datetime for one PET frame.

    This is the time point the stored voxel values correspond to, and the time
    point the administered dose must be decay-corrected to. It is determined
    from Decay Correction (0054,1102) as follows:

    - 'ADMIN': values are corrected to the administration time, so no dose
      correction is needed and None is returned.
    - 'START': values are corrected to the scan start datetime, resolved in
      order of decreasing reliability: (1) the vendor private scan start
      datetime (Siemens 0071,1022 / GE 0009,100D) or Decay Correction DateTime
      (0018,9701); (2) the Acquisition DateTime when it equals the Series
      DateTime; (3) back-computation from the frame timing attributes, using
      t_acq - dt for GE and t_acq + T_ave - dt for other manufacturers.
    - 'NONE': values are not decay corrected, so they correspond to the frame
      measurement time, t_acq + T_ave.

    Args:
        headerSlcS (cerr.dataclasses.scan_info.ScanInfo): scanInfo for the frame.

    Returns:
        tuple: (datetime.datetime | None, str) reference datetime and an error
            message. A None datetime with an empty message means no dose
            correction is needed ('ADMIN'); a non-empty message means the
            reference datetime could not be determined.
    """
    decayCorrection = str(headerSlcS.petDecayCorrection).upper()
    if decayCorrection == 'ADMIN':
        # Dose already corresponds to the administration time.
        return None, ''

    halfLife = headerSlcS.halfLife

    # Enhanced PET replaces Decay Correction (0054,1102) with Decay Corrected
    # (0018,9758), which makes the reference datetime explicit.
    decayCorrected = str(headerSlcS.petDecayCorrected).upper()
    if decayCorrected == 'YES':
        scanStart = parseDcmDateTime(headerSlcS.petDecayCorrectionDateTime)
        if scanStart is None:
            return None, 'Decay Correction DateTime is required for decay-corrected ' \
                         'Enhanced PET images but is absent or unparseable.'
        return scanStart, ''
    if decayCorrected == 'NO':
        # Values are not decay corrected, so they occurred at the frame
        # measurement time given by the Frame Reference DateTime.
        frameRefDateTime = parseDcmDateTime(headerSlcS.frameReferenceDateTime)
        if frameRefDateTime is not None:
            return frameRefDateTime, ''
        frameAcqDateTime = parseDcmDateTime(headerSlcS.frameAcquisitionDateTime)
        frameAcqDuration = headerSlcS.frameAcquisitionDuration
        frameAcqDuration = frameAcqDuration / 1000 \
            if isinstance(frameAcqDuration, (int, float)) and frameAcqDuration != '' else None
        if frameAcqDateTime is None or frameAcqDuration is None:
            return None, 'Frame Reference DateTime, or Frame Acquisition DateTime with ' \
                         'Frame Acquisition Duration, is required for Enhanced PET images ' \
                         'that are not decay corrected.'
        return frameAcqDateTime + datetime.timedelta(
            seconds=getAverageCountRateTime(frameAcqDuration, halfLife)), ''

    acqDateTime = combineDcmDateAndTime(headerSlcS.acquisitionDate,
                                        headerSlcS.acquisitionTime)
    seriesDateTime = combineDcmDateAndTime(headerSlcS.seriesDate,
                                           headerSlcS.seriesTime)
    frameDuration = headerSlcS.actualFrameDuration
    frameDuration = frameDuration / 1000 if isinstance(frameDuration, (int, float)) \
        and frameDuration != '' else None
    tAve = getAverageCountRateTime(frameDuration, halfLife)

    if decayCorrection == 'NONE':
        # Values occurred at the frame measurement time.
        if acqDateTime is None or frameDuration is None:
            return None, 'Acquisition DateTime and Actual Frame Duration are required to ' \
                         'determine the measurement time of non-decay-corrected images.'
        return acqDateTime + datetime.timedelta(seconds=tAve), ''

    # Decay Correction is 'START' (or unspecified, which is treated as 'START').
    # 1. Vendor private scan start datetime / Decay Correction DateTime.
    for privateDateTime in (headerSlcS.petDecayCorrectionDateTime,
                            headerSlcS.siemensPETDecayCorrectionDateTime,
                            headerSlcS.gePETDecayCorrectionDateTime):
        scanStart = parseDcmDateTime(privateDateTime)
        if scanStart is not None:
            return scanStart, ''

    # 2. Acquisition DateTime when it agrees with the Series DateTime.
    if acqDateTime is not None and seriesDateTime is not None \
            and acqDateTime == seriesDateTime:
        return acqDateTime, ''

    # 3. Back-compute the scan start from the frame timing attributes.
    frameRefTime = headerSlcS.frameReferenceTime
    if acqDateTime is not None and isinstance(frameRefTime, (int, float)) \
            and frameRefTime != '':
        manufacturer = str(headerSlcS.manufacturer).upper()
        isGE = 'GE' in manufacturer.split() or manufacturer.startswith('GE ')
        if isGE:
            # GE corrects to one frame reference time before Acquisition DateTime.
            offsetSecs = -frameRefTime / 1000
        elif frameDuration is not None:
            offsetSecs = tAve - frameRefTime / 1000
        else:
            # T_ave cannot be evaluated without the Actual Frame Duration.
            return None, 'Actual Frame Duration is required to back-compute the scan ' \
                         'start datetime; SUV cannot be computed reliably.'
        return acqDateTime + datetime.timedelta(seconds=offsetSecs), ''

    # Fall back to the Series DateTime when nothing better is available.
    if seriesDateTime is None:
        return None, 'The decay-correction reference datetime cannot be determined.'
    return seriesDateTime, ''


def getAdministrationDateTime(headerSlcS, decayRefDateTime):
    """Return the radiopharmaceutical administration datetime for one PET frame.

    Implements the IBSI-SUV strategy for reconciling the administration datetime
    with the decay-correction reference datetime, which is needed because one or
    both date components are frequently altered during post-processing:

    - Radiopharmaceutical Start DateTime (0018,1078) is trusted when the
      resulting uptake time is plausible, i.e. no earlier than one hour before
      the reference datetime and shorter than two half-lives.
    - Otherwise (or when only Radiopharmaceutical Start Time (0018,1072) is
      available) the date of the reference datetime is combined with the
      administration time of day, provided the radionuclide is short-lived.
      A day is subtracted when the uptake would otherwise be negative, which
      covers uptake periods spanning midnight.

    Args:
        headerSlcS (cerr.dataclasses.scan_info.ScanInfo): scanInfo for the frame.
        decayRefDateTime (datetime.datetime): decay-correction reference datetime.

    Returns:
        tuple: (datetime.datetime | None, str) administration datetime and a
            warning message ('' when none).
    """
    halfLife = headerSlcS.halfLife
    injDateTime = parseDcmDateTime(headerSlcS.injectionDateTime)

    hasHalfLife = isinstance(halfLife, (int, float)) and halfLife > 0
    if injDateTime is not None and hasHalfLife:
        uptakeSecs = (decayRefDateTime - injDateTime).total_seconds()
        if UPTAKE_TOLERANCE_SECS <= uptakeSecs < 2 * halfLife:
            # Both date components are consistent; use the datetime as stored.
            return injDateTime, ''

    # The stored dates disagree; the administration date can only be inferred
    # for short-lived radionuclides, where uptake cannot span a whole day.
    injTimeOfDay = parseDcmTimeOfDay(headerSlcS.injectionTime)
    if injTimeOfDay is None:
        return None, 'Radiopharmaceutical administration datetime is unavailable.'
    if not hasHalfLife or halfLife >= SHORT_HALF_LIFE_THRESH:
        return None, 'Radiopharmaceutical administration date is inconsistent with the ' \
                     'decay-correction datetime and the radionuclide is not short-lived; ' \
                     'SUV cannot be computed reliably.'

    warnMsg = 'Radiopharmaceutical administration date is missing or inconsistent with ' \
              'the decay-correction datetime; assuming administration on the same date.'
    admDateTime = decayRefDateTime.replace(hour=0, minute=0, second=0, microsecond=0) \
        + datetime.timedelta(seconds=injTimeOfDay)
    refTimeOfDay = (decayRefDateTime
                    - decayRefDateTime.replace(hour=0, minute=0, second=0,
                                               microsecond=0)).total_seconds()
    if refTimeOfDay - injTimeOfDay < UPTAKE_TOLERANCE_SECS:
        # Uptake period spans midnight: the tracer was given the previous day.
        admDateTime -= datetime.timedelta(days=1)
        warnMsg = 'Radiopharmaceutical administration date is missing or inconsistent ' \
                  'with the decay-correction datetime; assuming the uptake period ' \
                  'spans midnight.'
    return admDateTime, warnMsg


def getSuvNormalizationFactor(suvType, headerSlcS):
    """Return the SUV normalization factor, in grams, for the requested SUV type.

    Args:
        suvType (str): 'BW', 'BSA', 'LBM', 'LBMJAMES128', 'LBMJANMA' or 'IBW'.
        headerSlcS (cerr.dataclasses.scan_info.ScanInfo): scanInfo for the frame.

    Returns:
        float | None: normalization factor in grams (cm^2 for 'BSA'), or None
            when the required patient attributes are missing.

    Note:
        For 'BSA' the returned value is BSA in cm^2, matching the SUVbsa
        definition Ac * BSA(m^2) * 1e4 / dose.
    """
    weightKg = headerSlcS.patientWeight
    if not isinstance(weightKg, (int, float)) or weightKg <= 0:
        return None
    if weightKg >= 1000:
        # Patient's Weight is occasionally recorded in grams.
        weightKg = weightKg / 1000

    suvType = str(suvType).upper()
    if suvType == 'BW':
        return weightKg * 1000

    # The remaining normalizations additionally require the patient's height.
    heightM = headerSlcS.patientSize
    if not isinstance(heightM, (int, float)) or heightM <= 0:
        return None
    heightCm = heightM * 100
    sex = str(headerSlcS.patientSex).upper()

    def factorForSex(isMale):
        if suvType == 'LBM':  # lean body mass by Morgan
            return 1.10 * weightKg - 120 * (weightKg / heightCm) ** 2 if isMale \
                else 1.07 * weightKg - 148 * (weightKg / heightCm) ** 2
        if suvType == 'LBMJAMES128':  # lean body mass by James / Morgan
            return 1.10 * weightKg - 128 * (weightKg / heightCm) ** 2 if isMale \
                else 1.07 * weightKg - 148 * (weightKg / heightCm) ** 2
        if suvType == 'LBMJANMA':  # lean body mass by Janmahasatian
            bmi = weightKg / heightM ** 2
            return (9270 * weightKg) / (6680 + 216 * bmi) if isMale \
                else (9270 * weightKg) / (8780 + 244 * bmi)
        if suvType == 'IBW':  # ideal body weight
            return 48.0 + 1.06 * (heightCm - 152) if isMale \
                else 45.5 + 0.91 * (heightCm - 152)
        return None

    if suvType == 'BSA':
        # Du Bois formula, in m^2, scaled to cm^2.
        return 0.007184 * heightCm ** 0.725 * weightKg ** 0.425 * 1e4

    if sex == 'M':
        factorKg = factorForSex(True)
    elif sex == 'F':
        factorKg = factorForSex(False)
    elif sex == 'O':
        # No consensus exists for Patient's Sex 'O'; the mean of the
        # sex-specific factors is a reasonable compromise.
        maleFactor, femaleFactor = factorForSex(True), factorForSex(False)
        factorKg = None if maleFactor is None or femaleFactor is None \
            else (maleFactor + femaleFactor) / 2
    else:
        # Absent or non-conformant Patient's Sex: the factor is sex-dependent,
        # so it cannot be determined.
        factorKg = None

    return None if factorKg is None else factorKg * 1000


def dcm_hhmmss(time_str):
    """Parse a DICOM time string (HHMMSS) into its components and total seconds.

    Args:
        time_str (str): DICOM-format time string with at least 6 characters in
            HHMMSS order (fractional seconds are accepted but currently ignored).

    Returns:
        tuple: A 5-element tuple ``(totSec, hh, mm, ss, fract)`` where
            ``totSec`` (int) is the total number of seconds since midnight,
            ``hh`` (int) is hours, ``mm`` (int) is minutes, ``ss`` (int) is
            seconds, and ``fract`` is ``None`` (reserved for future use).
    """
    hh = int(time_str[0:2])
    mm = int(time_str[2:4])
    ss = int(time_str[4:6])
    fract = None  # You can implement this part if needed

    totSec = hh * 3600 + mm * 60 + ss
    return totSec, hh, mm, ss, fract

def dcm_to_np_date(dateStr):
    """Convert a DICOM date string (YYYYMMDD) to a numpy datetime64 object.

    Args:
        dateStr (str): DICOM-format date string, exactly 8 characters in
            YYYYMMDD order.

    Returns:
        np.datetime64 | None: A ``numpy.datetime64`` with day precision when
            ``dateStr`` is 8 characters long, otherwise ``None``.
    """
    dateObj = None
    if len(dateStr) == 8:
        dateObj = np.datetime64(dateStr[:4] + "-" + dateStr[4:6] + "-" + dateStr[6:], 'D')
    return dateObj

def get_slice_position(scan_info_item):
    """Extract the z-position from an enumerated scan-info pair for sort key use.

    Args:
        scan_info_item (tuple): A ``(index, ScanInfo)`` pair as produced by
            ``enumerate``.  The second element must have a ``zValue`` attribute.

    Returns:
        float: The ``zValue`` of the scan-info slice, in centimetres.
    """
    return scan_info_item[1].zValue

def populateScanInfoFields(s_info, ds):
    """

    Args:
        s_info (cerr.dataclasses.scan_info.ScanInfo): pyCERR's scanInfo object for storing metadata per slice.
        ds (pydicom.dataset.Dataset): pydicom dataset object

    Returns:
        cerr.dataclasses.scan_info.ScanInfo: scanInfo object with attributes populated from metadata from input ds.

    """
    s_info.frameOfReferenceUID = ds.FrameOfReferenceUID
    s_info.imageType = ds.Modality
    if not "SCAN" in s_info.imageType.upper():
        s_info.imageType = s_info.imageType + " SCAN"
    if hasattr(ds,"SeriesDescription"): s_info.seriesDescription = ds.SeriesDescription
    if hasattr(ds,"ManufacturerModelName"): s_info.scannerType = ds.ManufacturerModelName
    if hasattr(ds,"Manufacturer"): s_info.manufacturer = ds.Manufacturer
    s_info.scanFileName = ds.filename
    s_info.sopInstanceUID = ds.SOPInstanceUID
    s_info.sopClassUID = ds.SOPClassUID
    s_info.seriesInstanceUID = ds.SeriesInstanceUID
    s_info.studyInstanceUID = ds.StudyInstanceUID
    s_info.bitsAllocated = ds.BitsAllocated
    s_info.bitsStored = ds.BitsStored
    s_info.pixelRepresentation = ds.PixelRepresentation
    s_info.sizeOfDimension1 = ds.Rows
    s_info.sizeOfDimension2 = ds.Columns

    if hasattr(ds,"PatientName"): s_info.patientName = str(ds.PatientName)
    if hasattr(ds,"PatientID"): s_info.patientID = ds.PatientID
    if hasattr(ds,"AcquisitionDate"): s_info.acquisitionDate = ds.AcquisitionDate
    if hasattr(ds,"AcquisitionTime"): s_info.acquisitionTime = ds.AcquisitionTime
    if hasattr(ds,"SeriesDate"): s_info.seriesDate = ds.SeriesDate
    if hasattr(ds,"SeriesTime"): s_info.seriesTime = ds.SeriesTime
    if hasattr(ds,"StudyDate"): s_info.studyDate = ds.StudyDate
    if hasattr(ds,"StudyTime"): s_info.studyTime = ds.StudyTime
    if hasattr(ds,"StudyDescription"): s_info.studyDescription = ds.StudyDescription
    if hasattr(ds,"PatientWeight"): s_info.patientWeight = ds.PatientWeight
    if hasattr(ds,"PatientSize"): s_info.patientSize = ds.PatientSize
    if ("0010","1022") in ds: s_info.patientBmi = ds["0010","1022"].value
    if hasattr(ds,"PatientSex"): s_info.patientSex = ds.PatientSex
    if hasattr(ds,"SeriesType"): s_info.petSeriesType = ds.SeriesType
    if hasattr(ds,"Units"): s_info.imageUnits = ds.Units
    if hasattr(ds,"CountsSource"): s_info.petCountSource = ds.CountsSource
    if hasattr(ds,"NumberOfSlices"): s_info.petNumSlices = ds.NumberOfSlices
    if hasattr(ds,"DecayCorrection"): s_info.petDecayCorrection = ds.DecayCorrection
    if hasattr(ds,"CorrectedImage"): s_info.petCorrectedImage = ds.CorrectedImage
    # Decay Corrected (0018,9758) replaces Decay Correction (0054,1102) in Enhanced PET
    if ("0018","9758") in ds: s_info.petDecayCorrected = ds["0018","9758"].value
    # Frame timing attributes needed to derive the decay-correction reference datetime.
    # These are type 2/3 attributes, so they may be present but empty.
    if ("0054","1300") in ds and ds["0054","1300"].value is not None:
        s_info.frameReferenceTime = float(ds["0054","1300"].value)
    if ("0018","1242") in ds and ds["0018","1242"].value is not None:
        s_info.actualFrameDuration = float(ds["0018","1242"].value)
    if ("0054","1321") in ds and ds["0054","1321"].value is not None:
        s_info.decayFactor = float(ds["0054","1321"].value)
    if ("0054","1006") in ds: s_info.suvType = ds["0054","1006"].value
    if hasattr(ds,"WindowCenter"): s_info.windowCenter = ds.WindowCenter
    if hasattr(ds,"WindowWidth"): s_info.windowWidth = ds.WindowWidth

    if ("2005","140B") in ds: s_info.philipsImageUnits = ds["2005","140B"].value
    if ("2005","140A") in ds: s_info.philipsRescaleSlope = ds["2005","140A"].value
    if ("2005","1409") in ds: s_info.philipsRescaleIntercept = ds["2005","1409"].value

    if hasattr(ds,"PatientIdentityRemoved"): s_info.patientIdentityRemoved = ds.PatientIdentityRemoved
    if hasattr(ds,"DeidentificationMethod"): s_info.deIdentificationMethod = ds.DeidentificationMethod
    if hasattr(ds,"DeidentificationMethodCodeSequence"):
        for deIdMethod in ds.DeidentificationMethodCodeSequence:
            methodStr = deIdMethod.CodeValue + ': ' + deIdMethod.CodeMeaning
            s_info.deidentificationMethodDescription = np.append(s_info.deidentificationMethodDescription, methodStr)
    if ("0018","0010") in ds: s_info.contrastBolusAgent = ds["0018","0010"].value

    return s_info

def populateRealWorldFields(s_info, perFrameSeq):
    """

    Args:
        s_info (cerr.dataclasses.scan_info.ScanInfo): pyCERR's scanInfo object for storing metadata per slice.
        perFrameSeq (pydicom.dataset.Dataset): pydicom dataset object or ds.PerFrameFunctionalGroupsSequence
        for multiFrameFlg images.

    Returns:
        cerr.dataclasses.scan_info.ScanInfo: scanInfo object with attributes populated from metadata from input ds.

    """

    if 'RealWorldValueMappingSequence' in perFrameSeq:
        RealWorldValueMappingSeq = perFrameSeq.RealWorldValueMappingSequence[0]
        if hasattr(RealWorldValueMappingSeq,'RealWorldValueSlope'):
            s_info.realWorldValueSlope = RealWorldValueMappingSeq.RealWorldValueSlope
        if hasattr(RealWorldValueMappingSeq,'RealWorldValueIntercept'):
            s_info.realWorldValueIntercept = RealWorldValueMappingSeq.RealWorldValueIntercept
        if ("0040","08EA") in RealWorldValueMappingSeq:
            if ("0008","0100") in RealWorldValueMappingSeq["0040","08EA"][0]:
                s_info.realWorldMeasurCodeMeaning = RealWorldValueMappingSeq["0040","08EA"][0]["0008","0100"].value
            elif ("0008","0119") in RealWorldValueMappingSeq["0040","08EA"][0]:
                s_info.realWorldMeasurCodeMeaning = RealWorldValueMappingSeq["0040","08EA"][0]["0008","0119"].value
            if ("0008","0104") in RealWorldValueMappingSeq["0040","08EA"][0]:
                s_info.realWorldMeasurCodeMeaning = RealWorldValueMappingSeq["0040","08EA"][0]["0008","0104"].value
    return s_info

def populateRadiopharmaFields(s_info, seq):
    """

    Args:
        s_info (cerr.dataclasses.scan_info.ScanInfo): pyCERR's scanInfo object for storing metadata per slice.
        seq (pydicom.dataset.Dataset): dataset containing radiopharma metadata for PET scan.

    Returns:
        cerr.dataclasses.scan_info.ScanInfo: scanInfo object with attributes populated from metadata from input ds.

    """
    # populate radiopharma info
    if ("0054","0016") in seq:
        radiopharmaInfoSeq = seq["0054","0016"].value[0]
        if hasattr(radiopharmaInfoSeq,"RadiopharmaceuticalStartDateTime"):
            s_info.injectionDateTime = radiopharmaInfoSeq.RadiopharmaceuticalStartDateTime
            s_info.injectionDate = radiopharmaInfoSeq.RadiopharmaceuticalStartDateTime[:8]
            s_info.injectionTime = radiopharmaInfoSeq.RadiopharmaceuticalStartDateTime[8:]
        elif hasattr(radiopharmaInfoSeq,"RadiopharmaceuticalStartTime"):
            s_info.injectionTime = radiopharmaInfoSeq.RadiopharmaceuticalStartTime
        # Both are type 3 and may be absent or empty in non-conformant files.
        if getattr(radiopharmaInfoSeq, 'RadionuclideTotalDose', None) is not None:
            s_info.injectedDose = float(radiopharmaInfoSeq.RadionuclideTotalDose)
        if getattr(radiopharmaInfoSeq, 'RadionuclideHalfLife', None) is not None:
            s_info.halfLife = float(radiopharmaInfoSeq.RadionuclideHalfLife)
        if ("7053","1009") in seq: s_info.philipsActivityConcentrationScaleFactor = seq["7053","1009"].value
        if ("0018", "9701") in seq: s_info.petDecayCorrectionDateTime = seq["0018", "9701"].value
        if ("0071","1022") in seq: s_info.siemensPETDecayCorrectionDateTime = seq["0071","1022"].value # Siemens
        if ("0009","100D") in seq: s_info.gePETDecayCorrectionDateTime = seq["0009","100D"].value # GE
        if ("7053","1000") in seq: s_info.philipsSUVScaleFactor = seq["7053","1000"].value
    return s_info


def getFunctionalGroupItem(perFrameSeq, sharedSeq, seqKeyword):
    """Return the first item of a functional-group macro sequence for a frame.

    Enhanced multi-frame IODs (e.g. Enhanced PET Image, Enhanced CT/MR) may
    store a given functional-group macro either in the Per-Frame Functional
    Groups Sequence (0020,9111 item) when it varies frame-to-frame, or in the
    Shared Functional Groups Sequence (5200,9229 item) when it is constant for
    all frames. Per DICOM, a macro must appear in exactly one of the two. This
    helper looks in the per-frame group first and falls back to the shared
    group, so attributes such as Image Orientation (Patient) are imported
    correctly regardless of where the encoder placed them.

    Args:
        perFrameSeq (pydicom.dataset.Dataset): item of PerFrameFunctionalGroupsSequence for a frame.
        sharedSeq (pydicom.dataset.Dataset | None): item of SharedFunctionalGroupsSequence, or None.
        seqKeyword (str): keyword of the functional-group sequence, e.g. 'PlaneOrientationSequence'.

    Returns:
        pydicom.dataset.Dataset | None: first item of the matching sequence, or None when absent.
    """
    for seq in (perFrameSeq, sharedSeq):
        if seq is not None and seqKeyword in seq:
            fgSeq = getattr(seq, seqKeyword)
            if len(fgSeq) > 0:
                return fgSeq[0]
    return None

def parseScanInfoFields(ds, multiFrameFlg=False) -> (scn_info.ScanInfo, Dataset.pixel_array, str):
    """

    Args:
        ds (pydicom.dataset.Dataset): Dataset object read from DICOM file
        multiFrameFlg (bool): True when dataset is multiFrame image, otherwise False.

    Returns:
        cerr.dataclasses.scan_info.ScanInfo: scanInfo object with attributes populated from metadata from input ds.

    """
    #numberOfFrames = ds.NumberOfFrames.real
    # s_info.frameOfReferenceUID = ds.FrameOfReferenceUID
    #s_info.seriesDescription = ds.SeriesDescription
    if not multiFrameFlg: #numberOfFrames == 1:
        scan_info = scn_info.ScanInfo()
        scan_info = populateScanInfoFields(scan_info, ds)
        if hasattr(ds,'RescaleSlope'): scan_info.rescaleSlope = ds.RescaleSlope
        if hasattr(ds,'RescaleIntercept'): scan_info.rescaleIntercept = ds.RescaleIntercept
        if hasattr(ds,'RescaleType'): scan_info.rescaleType = ds.RescaleType
        if ("2005","100E") in ds: scan_info.scaleSlope = ds["2005","100E"].value
        if ("2005","100D") in ds: scan_info.scaleIntercept = ds["2005","100D"].value

        scan_info = populateRealWorldFields(scan_info, ds)

        scan_info.grid1Units = ds.PixelSpacing[1] / 10
        scan_info.grid2Units = ds.PixelSpacing[0] / 10
        scan_info.sliceThickness = ds.SliceThickness / 10
        scan_info.imageOrientationPatient = np.array(ds.ImageOrientationPatient)
        scan_info.imagePositionPatient = np.array(ds.ImagePositionPatient)
        slice_normal = scan_info.imageOrientationPatient[[1,2,0]] * scan_info.imageOrientationPatient[[5,3,4]] \
                       - scan_info.imageOrientationPatient[[2,0,1]] * scan_info.imageOrientationPatient[[4,5,3]]
        scan_info.zValue = - np.sum(slice_normal * scan_info.imagePositionPatient) / 10

        bVal1 = ("0043", "1039") # GE
        bVal2 = ("0018", "9087") # Philips
        bVal3 = ("0019", "100C") # Siemens
        temporalPos = ("0020", "0100")
        triggerTime = ("0018", "1060")
        TR = ("0018", "0080")
        FA = ("0018", "1314")
        MFS = ("0018", "0087")  # MagneticFieldStrength
        if bVal1 in ds: scan_info.bValue = ds["0043", "1039"].value
        if bVal2 in ds: scan_info.bValue = ds["0018", "9087"].value
        if bVal3 in ds: scan_info.bValue = ds["0019", "100C"].value
        if temporalPos in ds: scan_info.temporalPositionIdentifier = ds["0020", "0100"].value
        if triggerTime in ds: scan_info.triggerTime = ds["0018", "1060"].value
        if TR in ds: scan_info.repetitionTime = float(ds["0018", "0080"].value)
        if FA in ds: scan_info.flipAngle = float(ds["0018", "1314"].value)
        if MFS in ds: scan_info.magneticFieldStrength = float(ds["0018", "0087"].value) 
        scan_info = populateRadiopharmaFields(scan_info, ds)

    else:
        numberOfFrames = ds.NumberOfFrames.real
        sharedSeq = None
        if 'SharedFunctionalGroupsSequence' in ds and len(ds.SharedFunctionalGroupsSequence) > 0:
            sharedSeq = ds.SharedFunctionalGroupsSequence[0]
        scan_info = np.empty(numberOfFrames, dtype=scn_info.ScanInfo)
        for iFrame in range(numberOfFrames):
            s_info = scn_info.ScanInfo()
            s_info = populateScanInfoFields(s_info, ds)
            if 'PerFrameFunctionalGroupsSequence' in ds:
                perFrameSeq = ds.PerFrameFunctionalGroupsSequence[iFrame]
                # Real World Value Mapping may be per-frame or shared (e.g. Enhanced PET).
                s_info = populateRealWorldFields(s_info, perFrameSeq)
                if 'RealWorldValueMappingSequence' not in perFrameSeq and sharedSeq is not None:
                    s_info = populateRealWorldFields(s_info, sharedSeq)

                # Pixel Value Transformation (rescale slope/intercept/type). Read from the
                # per-frame group, falling back to the shared group when constant across frames.
                PixelValueTransformSeq = getFunctionalGroupItem(
                    perFrameSeq, sharedSeq, 'PixelValueTransformationSequence')
                if PixelValueTransformSeq is not None:
                    if hasattr(PixelValueTransformSeq,'RescaleSlope'): s_info.rescaleSlope = PixelValueTransformSeq.RescaleSlope
                    if hasattr(PixelValueTransformSeq,'RescaleIntercept'): s_info.rescaleIntercept = PixelValueTransformSeq.RescaleIntercept
                    if hasattr(PixelValueTransformSeq,'RescaleType'): s_info.rescaleType = PixelValueTransformSeq.RescaleType
                    if ("2005","100E") in PixelValueTransformSeq: s_info.scaleSlope = PixelValueTransformSeq["2005","100E"].value
                    if ("2005","100D") in PixelValueTransformSeq: s_info.scaleIntercept = PixelValueTransformSeq["2005","100D"].value

                # Plane Position (Patient) - per-frame in Enhanced PET, shared fallback otherwise.
                planePositionSeq = getFunctionalGroupItem(
                    perFrameSeq, sharedSeq, 'PlanePositionSequence')
                if planePositionSeq is not None and 'ImagePositionPatient' in planePositionSeq:
                    s_info.imagePositionPatient = np.array(planePositionSeq.ImagePositionPatient)

                # Plane Orientation (Patient) - typically shared in Enhanced PET, per-frame otherwise.
                planeOrientationSeq = getFunctionalGroupItem(
                    perFrameSeq, sharedSeq, 'PlaneOrientationSequence')
                if planeOrientationSeq is not None and 'ImageOrientationPatient' in planeOrientationSeq:
                    s_info.imageOrientationPatient = np.array(planeOrientationSeq.ImageOrientationPatient)

                if s_info.imageOrientationPatient.size == 6 and s_info.imagePositionPatient.size == 3:
                    slice_normal = s_info.imageOrientationPatient[[1,2,0]] * s_info.imageOrientationPatient[[5,3,4]] \
                                   - s_info.imageOrientationPatient[[2,0,1]] * s_info.imageOrientationPatient[[4,5,3]]
                    s_info.zValue = - np.sum(slice_normal * s_info.imagePositionPatient) / 10

                # Frame VOI LUT (window/level) - per-frame with shared fallback.
                frameVOISeq = getFunctionalGroupItem(perFrameSeq, sharedSeq, 'FrameVOILUTSequence')
                if frameVOISeq is not None:
                    if hasattr(frameVOISeq, 'WindowWidth'): s_info.windowWidth = float(frameVOISeq.WindowWidth)
                    if hasattr(frameVOISeq, 'WindowCenter'): s_info.windowCenter = float(frameVOISeq.WindowCenter)

                # Frame Content carries the per-frame timing needed to establish the
                # decay-correction reference datetime of an Enhanced PET frame.
                frameContentSeq = getFunctionalGroupItem(perFrameSeq, sharedSeq,
                                                         'FrameContentSequence')
                if frameContentSeq is not None:
                    if ("0018","9074") in frameContentSeq:
                        s_info.frameAcquisitionDateTime = frameContentSeq["0018","9074"].value
                    if ("0018","9151") in frameContentSeq:
                        s_info.frameReferenceDateTime = frameContentSeq["0018","9151"].value
                    if ("0018","9220") in frameContentSeq and \
                            frameContentSeq["0018","9220"].value is not None:
                        s_info.frameAcquisitionDuration = float(frameContentSeq["0018","9220"].value)
            else: # NM scans
                s_info = populateRealWorldFields(s_info, ds)
                if 'DetectorInformationSequence' in ds:
                    imagePositionPatientStart = np.array(ds.DetectorInformationSequence[0].ImagePositionPatient)
                    s_info.imageOrientationPatient = np.array(ds.DetectorInformationSequence[0].ImageOrientationPatient)
                    sliceSpacing = ds.SpacingBetweenSlices * iFrame
                    slice_normal = s_info.imageOrientationPatient[[1,2,0]] * s_info.imageOrientationPatient[[5,3,4]] \
                                   - s_info.imageOrientationPatient[[2,0,1]] * s_info.imageOrientationPatient[[4,5,3]]
                    s_info.imagePositionPatient = imagePositionPatientStart + slice_normal * sliceSpacing
                    s_info.zValue = - np.sum(slice_normal * s_info.imagePositionPatient) / 10

            # Pixel Measures (spacing, slice thickness) - shared in Enhanced PET,
            # but allow a per-frame override and fall back to top-level tags.
            pixelMeasuresSeq = getFunctionalGroupItem(
                perFrameSeq if 'PerFrameFunctionalGroupsSequence' in ds else None,
                sharedSeq, 'PixelMeasuresSequence')
            if pixelMeasuresSeq is not None:
                PixelSpacing = pixelMeasuresSeq.PixelSpacing
                if hasattr(pixelMeasuresSeq, 'SliceThickness') and \
                        isinstance(pixelMeasuresSeq.SliceThickness, (float, int)):
                    s_info.sliceThickness = pixelMeasuresSeq.SliceThickness / 10
                elif hasattr(pixelMeasuresSeq, 'SpacingBetweenSlices') and \
                        isinstance(pixelMeasuresSeq.SpacingBetweenSlices, (float, int)):
                    s_info.sliceThickness = pixelMeasuresSeq.SpacingBetweenSlices / 10
            else:
                PixelSpacing = ds.PixelSpacing
                if isinstance(ds.SliceThickness, (float, int)):
                    s_info.sliceThickness = ds.SliceThickness / 10
                elif 'SpacingBetweenSlices' in ds and isinstance(ds.SpacingBetweenSlices, (float, int)):
                    s_info.sliceThickness = ds.SpacingBetweenSlices / 10

            s_info.grid1Units = PixelSpacing[1] / 10
            s_info.grid2Units = PixelSpacing[0] / 10

            # PET Units (0054,1001) is absent from Enhanced PET Image (Units is a per-frame
            # concept there); derive the storage units from the Pixel Value Transformation
            # Rescale Type (e.g. BQML) so SUV conversion has the correct input units.
            if s_info.imageType.upper() in ['PT SCAN', 'NM SCAN'] and \
                    len(s_info.imageUnits) == 0 and len(s_info.rescaleType) > 0:
                s_info.imageUnits = s_info.rescaleType


            # MR-specific tags
            TR = ("0018", "0080")
            FA = ("0018", "1314")
            MFS = ("0018", "0087")
            if TR in ds: s_info.repetitionTime = float(ds["0018", "0080"].value)
            if FA in ds: s_info.flipAngle = float(ds["0018", "1314"].value)
            if MFS in ds: s_info.magneticFieldStrength = float(ds["0018", "0087"].value)
            s_info = populateRadiopharmaFields(s_info, ds)

            scan_info[iFrame] = s_info

    return (scan_info, ds.pixel_array, ds.SeriesInstanceUID)

def loadSortedScanInfo(file_list):
    """

    Args:
        file_list (list): list of files to read into pyCERR's Scan object

    Returns:
        cerr.daatclasses.scan.Scan: pyCERR scan object containing metadata from the file_list.

    """
    scan = Scan()
    #scan_info = [] #scn_info.ScanInfo()
    #scan_array = []
    scan_array = [] #np.empty(len(file_list))
    scan_info = np.empty(len(file_list),dtype=scn_info.ScanInfo)
    count = 0
    multiFrameFlag = False
    for file in file_list:
        ds = dcmread(file)
        if np.any(ds.Modality == np.array(["CT","PT", "MR", "NM", "US", "OT"])): #hasattr(ds, "pixel_array"):
            if len(file_list) == 1 and 'NumberOfFrames' in ds:
                multiFrameFlag = True
                si_pixel_data = parseScanInfoFields(ds, multiFrameFlag)
                scan_array = np.transpose(si_pixel_data[1], (1,2,0))
                scan_info = si_pixel_data[0]
                count = len(scan_info)
            else:
                si_pixel_data = parseScanInfoFields(ds)
                #scan_info.append(si_pixel_data[0])
                #scan_array.append(si_pixel_data[1])
                scan_info[count] = si_pixel_data[0]
                if not isinstance(scan_array, np.ndarray) and not scan_array:
                    imgSiz = list(si_pixel_data[1].shape)
                    imgSiz.append(len(file_list))
                    scan_array = np.empty(imgSiz, dtype=np.float32)
                scan_array[:,:,count] = si_pixel_data[1]
                count += 1

    if count < scan_array.shape[2]:
        scan_array = np.delete(scan_array,np.arange(count,scan_array.shape[2]),axis=2)
        scan_info = np.delete(scan_info,np.arange(count,scan_array.shape[2]),axis=0)

    # Filter out duplicate SOP Instances
    if np.any(ds.Modality == np.array(["CT","PT", "MR", "NM", "US", "OT"])) and not multiFrameFlag:
        allSOPs = [s.sopInstanceUID for s in scan_info]
        uniqSOPs, uniqInds = np.unique(allSOPs, return_index=True)
        duplicateIDs = list(set(range(len(scan_info))) - set(uniqInds))
        scan_array = np.delete(scan_array,duplicateIDs,axis=2)
        scan_info = np.delete(scan_info,duplicateIDs,axis=0)

    #sorted_indices = scan_info.sort(key=get_slice_position, reverse=False)
    sort_index = [i for i,x in sorted(enumerate(scan_info),key=get_slice_position, reverse=False)]
    #scan_array = np.array(scan_array)
    #scan_array = np.moveaxis(scan_array,[0,1,2],[2,0,1])
    #scan_info = np.array(scan_info)
    scan_info = scan_info[sort_index]
    scan_array = scan_array[:,:,sort_index]
    scan_info = scn_info.deduce_voxel_thickness(scan_info)
    scan.scanInfo = scan_info
    scan.scanArray = scan_array
    scan.scanUID = "CT." + si_pixel_data[2]
    return scan


def parseScanInfoFromDB(scanObj, scanInfoList):
    """Assign scanInfo from the list of dictionaries to scanObj by matching zValue per slice

    Args:
        scanObj (cerr.dataclasses.scan.Scan): pyCERR's Scan object whose scanInfo needs to be populated
        scanInfoList (list): list of dictionaries with fields corresponding to scanInfo

    Returns:
       0 when field assignment is successful

    """

    # Create a new dictionary with DICOM names for keys
    scanDir = os.path.dirname(__file__)
    mappingFile = os.path.join(scanDir, 'dcm_cerr_name_map.json')
    with open(mappingFile, 'r') as nameMapFile:
        dcmCerrNameMap = json.load(nameMapFile)
    cerrDcmNameMap = {v: k for k, v in dcmCerrNameMap.items()}

    zValsScanV = [s.zValue for s in scanObj.scanInfo]
    # Get z-values for scanInfoList and assign scanInfo
    numberOfFrames = len(scanInfoList) #ds.NumberOfFrames.real
    for iFrame in range(numberOfFrames):

        imageOri = np.array(scanInfoList[iFrame]['ImageOrientationPatient'])
        imagePos = np.array(scanInfoList[iFrame]['ImagePositionPatient'])
        slice_normal = imageOri[[1,2,0]] * imageOri[[5,3,4]] \
                       - imageOri[[2,0,1]] * imageOri[[4,5,3]]
        zValue = - np.sum(slice_normal * imagePos) / 10
        indSlc = np.argmin((zValsScanV - zValue)**2)
        fieldNames = scanInfoList[iFrame].keys()
        for fieldName in fieldNames:
            if fieldName in cerrDcmNameMap:
                cerrField = cerrDcmNameMap[fieldName]
                if hasattr(scanObj.scanInfo[indSlc], cerrField):
                    setattr(scanObj.scanInfo[indSlc], cerrField, scanInfoList[iFrame][fieldName])
        scanObj.scanInfo[indSlc].imageOrientationPatient = imageOri
        scanObj.scanInfo[indSlc].imagePositionPatient = imagePos
        scanObj.scanInfo[indSlc].grid1Units = scanInfoList[iFrame]['PixelSpacing'][1]
        scanObj.scanInfo[indSlc].grid2Units = scanInfoList[iFrame]['PixelSpacing'][0]

    return 0


def getScanNumFromUID(assocScanUID,planC) -> int:
    """

    Args:
        assocScanUID (str): UID of scan.
        planC (cerr.plan_container.planC): pyCERR's plan container object.

    Returns:
        int: index within planC.scan that matches input assocScanUID.
    """

    uid_list = [s.scanUID for s in planC.scan]
    if assocScanUID in uid_list:
        return uid_list.index(assocScanUID)
    else:
        return None

def getCERRScanArrayFromITK(itkImage, assocScanNum, planC):
    """ This routine returns a numpy array in pyCERR coordinate system (orientation) from a SimpleITK Image.

    Args:
        itkImage (SimpleITK.Image): SimpleITK's Image object
        assocScanNum (int): Scan index to associate orientation of itkImage in pyCERR.
        planC (cerr.planC_container.planC): pyCERR's plan container object.

    Returns:
        np.ndarray: array in CERR virtual coordinates.

    """
    if isinstance(itkImage, sitk.Image):
        itkImage = sitk.GetArrayFromImage(itkImage)
    cerrArray = np.transpose(itkImage, (1, 2, 0))
    # flip slices in CERR z-slice order which increases from head to toe
    if flipSliceOrderFlag(planC.scan[assocScanNum]):
        cerrArray = np.flip(cerrArray, axis=2)
    return cerrArray
