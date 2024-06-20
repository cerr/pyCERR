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
import nibabel as nib
import SimpleITK as sitk
import json

def get_empty_list():
    return []
def get_empty_np_array():
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
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    class json_serialize(json.JSONEncoder):
        def default(self, obj):
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

    def get_nii_affine(self):
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

    def save_nii(self, niiFileName):
        """ Routine to save pyCERR Scan object to NifTi file

        Args:
            niiFileName (str): File name including the full path to save the pyCERR scan object to NifTi file.

        Returns:
            int: 0 when NifTi file is written successfully.
        """

        affine3M = self.get_nii_affine()
        scan3M = self.getScanArray()
        scan3M = np.moveaxis(scan3M,[0,1],[1,0])
        #scan3M = np.flip(scan3M,axis=[0,1]) # negated affineM to take care of reverse row/col compared to dicom
        # Determine whether CERR slice order matches DICOM
        # dcmImgOri = self.scanInfo[0].imageOrientationPatient
        # slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
        #        - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]
        # slice_normal = slice_normal.reshape((1,3))
        # zDiff = np.matmul(slice_normal, self.scanInfo[1].imagePositionPatient) - np.matmul(slice_normal, self.scanInfo[0].imagePositionPatient)
        # ippDiffV = self.scanInfo[1].imagePositionPatient - self.scanInfo[0].imagePositionPatient
        if flipSliceOrderFlag(self): #np.all(np.sign(zDiff) < 0):
            scan3M = np.flip(scan3M,axis=2) # CERR slice ordering is opposite of DICOM
        img = nib.Nifti1Image(scan3M, affine3M)
        success = nib.save(img, niiFileName)
        return success

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

        # Calculate xVals
        xvals = np.arange(scan_info.xOffset - (sizeDim2 * scan_info.grid2Units) / 2,
                  scan_info.xOffset + (sizeDim2 * scan_info.grid2Units) / 2 + scan_info.grid2Units,
                  scan_info.grid2Units)

        # Calculate yVals (flipped left-right)
        yvals = np.arange(scan_info.yOffset + (sizeDim1 * scan_info.grid1Units) / 2,
                  scan_info.yOffset - (sizeDim1 * scan_info.grid1Units) / 2 - scan_info.grid1Units,
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

        positionMatrixInv = np.linalg.inv(position_matrix)
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
        scanArray3M = np.zeros(self.scanArray.shape, dtype=float)
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

            if 'philips' in manufacturer.lower() and \
                    realWorldValueSlope is not None and \
                    not np.isnan(realWorldValueSlope) and \
                    self.scanInfo[slcNum].imageType.lower() == 'mr scan' and \
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
                self.scanArray = self.scanArray.astype(float) / (rescaleSlope * scaleSlope)

    def convert_to_suv(self,suvType="BW"):
        """ Routine to convert pixel array for PET scan from DICOM storage to SUV

        Args:
            suvType (str): optional, type of SUV. When not specified, the suvType is read from DICOM if available.
             When not specified and not available in DIOCM, a default value of 'BW' is used. Currently supported
             options are 'BW', 'BSA', 'LBM', 'LBMJANMA'

        """

        scan3M = self.scanArray
        headerS = self.scanInfo
        scanSiz = scan3M.shape
        suv3M = np.zeros(scanSiz)

        numSlcs = scan3M.shape[2]
        acqTimeV = np.empty(numSlcs,dtype=float)
        for slcNum in range(numSlcs):
            headerSlcS = headerS[slcNum]
            if headerSlcS.acquisitionTime:
                acqTimeV[slcNum] = dcm_hhmmss(headerSlcS.acquisitionTime)[0]
        seriesTime = dcm_hhmmss(headerSlcS.seriesTime)[0]
        seriesDate = np.nan
        if headerSlcS.seriesDate:
            seriesDate = dcm_to_np_date(headerSlcS.seriesDate)
        injectionDate = np.nan
        if headerSlcS.injectionDate:
            injectionDate = dcm_to_np_date(headerSlcS.injectionDate)
        acqStartTime = np.nan
        if not np.any(np.isnan(acqTimeV)):
            acqStartTime = np.min(acqTimeV)

        for slcNum in range(scan3M.shape[2]):
            headerSlcS = headerS[slcNum]
            imgM = scan3M[:, :, slcNum] - headerSlcS.CTOffset

            imgUnits = headerSlcS.imageUnits
            imgMUnits = imgM.copy()
            if imgUnits == 'CNTS':
                activityScaleFactor = headerSlcS.petActivityConctrScaleFactor
                imgMUnits = imgMUnits * activityScaleFactor
                imgMUnits = imgMUnits * 1000  # Bq/L
            elif imgUnits in ['BQML', 'BQCC']:
                imgMUnits = imgMUnits * 1000  # Bq/L
            elif imgUnits in ['KBQCC', 'KBQML']:
                imgMUnits = imgMUnits * 1e6  # Bq/L
            else:
                #raise ValueError('SUV calculation is supported only for imageUnits BQML and CNTS')
                import warnings
                warnings.warn("'SUV calculation is supported only for imageUnits BQML and CNTS'")
                return

            decayCorrection = headerSlcS.decayCorrection
            if decayCorrection == 'START':
                scantime = seriesTime
                if not np.isnan(acqStartTime) and acqStartTime < scantime:
                    scantime = acqStartTime
            elif decayCorrection == 'ADMIN':
                scantime = dcm_hhmmss(headerSlcS.injectionTime)
            elif decayCorrection == 'NONE':
                scantime = np.nan
            elif len(headerSlcS.petDecayCorrectionDateTime) > 8:
                scantime = dcm_hhmmss(headerSlcS.petDecayCorrectionDateTime[8:])[0]
            else:
                scantime = np.nan

            # Start Time for Radiopharmaceutical Injection
            injection_time = dcm_hhmmss(headerSlcS.injectionTime)[0]

            if not np.isnan(seriesDate) and not np.isnan(injectionDate):
                date_diff = seriesDate - injectionDate
                if date_diff < 5: # check whether it is a reasonable value
                    injection_time = injection_time - date_diff.item().total_seconds()

            # Half Life for Radionuclide
            half_life = headerSlcS.halfLife

            # Total dose injected for Radionuclide
            injected_dose = headerSlcS.injectedDose

            # Modality
            modality = headerSlcS.imageType
            if modality.upper() == 'NM SCAN':
                injected_dose = injected_dose * 1e6  # Convert MBq to Bq

            # Fix issue where IOD is PT and injected_dose units are in MBq
            if injected_dose < 1e5:
                injected_dose = injected_dose * 1e6

            # Calculate the decay
            # The injected dose used to calculate suvM is corrected for the decay that
            # occurs between the time of injection and the time of scan.
            # decayFactor = e^(t1-t2/halflife)
            if decayCorrection.upper() == 'NONE':
                decay = 1
            else:
                decay = np.exp(-np.log(2) * (scantime - injection_time) / half_life)

            # Calculate the dose decayed during procedure
            injected_dose_decay = injected_dose * decay  # in Bq

            # Patient Weight
            ptWeight = headerSlcS.patientWeight

            # Calculate SUV based on type
            # reference: http://dicom.nema.org/medical/Dicom/2017e/output/chtml/part16/sect_CID_85.html
            # SUVbw and SUVbsa equations are taken from Kim et al. Journal of Nuclear Medicine. Volume 35, No. 1, January 1994. pp 164-167.
            suvType = suvType.upper()
            if suvType == 'BW':  # Body Weight
                suvM = imgMUnits * ptWeight / injected_dose_decay  # pt weight in grams
                imageUnits = 'GML'
            elif suvType == 'BSA':  # body surface area
                # Patient height
                # (BSA in m2) = [(weight in kg)^0.425 * (height in cm)^0.725 * 0.007184].
                # SUV-bsa = (PET image Pixels) * (BSA in m2) * (10000 cm2/m2) / (injected dose).
                ptHeight = headerSlcS.patientSize  # units of meter
                bsaMm = ptWeight**0.425 * (ptHeight * 100)**0.725 * 0.007184
                suvM = imgMUnits * bsaMm / injected_dose_decay
                imageUnits = 'CM2ML'
            elif suvType == 'LBM':  # lean body mass by James method
                ptGender = headerSlcS.patientSex
                ptHeight = headerSlcS.patientSize
                if ptGender.upper() == 'M':
                    # LBM in kg = 1.10 * (weight in kg) - 120 * [(weight in kg) / (height in cm)]^2.
                    lbmKg = 1.10 * ptWeight - 120 * (ptWeight / (ptHeight * 100))**2
                else:
                    # if gender == female
                    # LBM in kg = 1.07 * (weight in kg) - 148 * [(weight in kg) / (height in cm)]^2.
                    lbmKg = 1.07 * ptWeight - 148 * (ptWeight / (ptHeight * 100))**2
                suvM = imgMUnits * lbmKg / injected_dose_decay
                imageUnits = 'GML'
            elif suvType == 'LBMJAMES128':  # lean body mass by James method
                imageUnits = 'GML'
            elif suvType == 'LBMJANMA':  # lean body mass by Janmahasatian method
                ptHeight = headerSlcS['patientSize']
                bmi = (ptWeight * 2.20462 / (ptHeight * 39.3701)**2) * 703
                ptGender = headerSlcS['patientSex']
                if ptGender.upper() == 'M':
                    lbmKg = (9270 * ptWeight) / (6680 + 216 * bmi)  # male
                else:
                    lbmKg = (9270 * ptWeight) / (8780 + 244 * bmi)  # female
                suvM = imgMUnits * lbmKg / injected_dose_decay
                imageUnits = 'GML'
            elif suvType == 'IBW':  # ideal body weight
                imageUnits = 'GML'

            suv3M[:, :, slcNum] = suvM
            self.scanInfo[slcNum].imageUnits = imageUnits
            self.scanInfo[slcNum].suvType = suvType

        self.scanArray = suv3M

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

def flipSliceOrderFlag(scan):
    """ Routine to determine slice order for determining the origin for conversion to NifTi and SimpleITK formats.

    Args:
        scan (cerr.dataclasses.scan.Scan): pyCERR scan object

    Returns:
        bool: True when dot product of slice normal and imagePositionPatient increases with slice order

    """

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

def dcm_hhmmss(time_str):
    hh = int(time_str[0:2])
    mm = int(time_str[2:4])
    ss = int(time_str[4:6])
    fract = None  # You can implement this part if needed

    totSec = hh * 3600 + mm * 60 + ss
    return totSec, hh, mm, ss, fract

def dcm_to_np_date(dateStr):
    dateObj = None
    if len(dateStr) == 8:
        dateObj = np.datetime64(dateStr[:4] + "-" + dateStr[4:6] + "-" + dateStr[6:], 'D')
    return dateObj

def get_slice_position(scan_info_item):
    return scan_info_item[1].zValue

def populate_scan_info_fields(s_info, ds):
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
    if hasattr(ds,"CorrectedImage"): s_info.correctedImage = ds.CorrectedImage
    if hasattr(ds,"DecayCorrection"): s_info.decayCorrection = ds.DecayCorrection
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
    if ("0054","1006") in ds: s_info.suvType = ds["0054","1006"].value
    if hasattr(ds,"WindowCenter"): s_info.windowCenter = ds.WindowCenter
    if hasattr(ds,"WindowWidth"): s_info.windowWidth = ds.WindowWidth

    if ("2005","140B") in ds: s_info.philipsImageUnits = ds["2005","140B"].value
    if ("2005","140A") in ds: s_info.philipsRescaleSlope = ds["2005","140A"].value
    if ("2005","1409") in ds: s_info.philipsRescaleIntercept = ds["2005","1409"].value

    return s_info

def populate_real_world_fields(s_info, perFrameSeq):
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

def populate_radiopharma_fields(s_info, seq):
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
            s_info.injectionDate = radiopharmaInfoSeq.RadiopharmaceuticalStartDateTime[:8]
            s_info.injectionTime = radiopharmaInfoSeq.RadiopharmaceuticalStartDateTime[8:]
        elif hasattr(radiopharmaInfoSeq,"RadiopharmaceuticalStartTime"):
            s_info.injectionTime = radiopharmaInfoSeq.RadiopharmaceuticalStartTime
        s_info.injectedDose = float(radiopharmaInfoSeq.RadionuclideTotalDose)
        s_info.halfLife = float(radiopharmaInfoSeq.RadionuclideHalfLife)
        if ("7053","1009") in seq: s_info.petActivityConctrScaleFactor = seq["7053","1009"].value
        if ("0018", "9701") in seq: s_info.petDecayCorrectionDateTime = seq["0018", "9701"].value
    return s_info

def parse_scan_info_fields(ds, multiFrameFlg=False) -> (scn_info.ScanInfo, Dataset.pixel_array, str):
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
        scan_info = populate_scan_info_fields(scan_info, ds)
        if hasattr(ds,'RescaleSlope'): scan_info.rescaleSlope = ds.RescaleSlope
        if hasattr(ds,'RescaleIntercept'): scan_info.rescaleIntercept = ds.RescaleIntercept
        if hasattr(ds,'RescaleType'): scan_info.rescaleType = ds.RescaleType
        if ("2005","100E") in ds: scan_info.scaleSlope = ds["2005","100E"].value
        if ("2005","100D") in ds: scan_info.scaleIntercept = ds["2005","100D"].value

        scan_info = populate_real_world_fields(scan_info, ds)

        scan_info.grid1Units = ds.PixelSpacing[1] / 10
        scan_info.grid2Units = ds.PixelSpacing[0] / 10
        scan_info.sliceThickness = ds.SliceThickness / 10
        scan_info.imageOrientationPatient = np.array(ds.ImageOrientationPatient)
        scan_info.imagePositionPatient = np.array(ds.ImagePositionPatient)
        slice_normal = scan_info.imageOrientationPatient[[1,2,0]] * scan_info.imageOrientationPatient[[5,3,4]] \
                       - scan_info.imageOrientationPatient[[2,0,1]] * scan_info.imageOrientationPatient[[4,5,3]]
        scan_info.zValue = - np.sum(slice_normal * scan_info.imagePositionPatient) / 10

        scan_info = populate_radiopharma_fields(scan_info, ds)

    else:
        numberOfFrames = ds.NumberOfFrames.real
        scan_info = np.empty(numberOfFrames, dtype=scn_info.ScanInfo)
        for iFrame in range(numberOfFrames):
            s_info = scn_info.ScanInfo()
            s_info = populate_scan_info_fields(s_info, ds)
            perFrameSeq = ds.PerFrameFunctionalGroupsSequence[iFrame]
            s_info = populate_real_world_fields(s_info, perFrameSeq)

            PixelSpacing = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
            s_info.grid1Units = PixelSpacing[1] / 10
            s_info.grid2Units = PixelSpacing[0] / 10
            s_info.sliceThickness = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].SliceThickness / 10

            if 'PixelValueTransformationSequence' in perFrameSeq:
                PixelValueTransformSeq = perFrameSeq.PixelValueTransformationSequence[0]
                if hasattr(PixelValueTransformSeq,'RescaleSlope'): s_info.rescaleSlope = PixelValueTransformSeq.RescaleSlope
                if hasattr(PixelValueTransformSeq,'RescaleIntercept'): s_info.rescaleIntercept = PixelValueTransformSeq.RescaleIntercept
                if ("2005","100E") in PixelValueTransformSeq: s_info.scaleSlope = PixelValueTransformSeq["2005","100E"].value
                if ("2005","100D") in PixelValueTransformSeq: s_info.scaleIntercept = PixelValueTransformSeq["2005","100D"].value

            s_info.imagePositionPatient = np.array(perFrameSeq.PlanePositionSequence[0].ImagePositionPatient)
            s_info.imageOrientationPatient = np.array(perFrameSeq.PlaneOrientationSequence[0].ImageOrientationPatient)
            slice_normal = s_info.imageOrientationPatient[[1,2,0]] * s_info.imageOrientationPatient[[5,3,4]] \
                           - s_info.imageOrientationPatient[[2,0,1]] * s_info.imageOrientationPatient[[4,5,3]]
            s_info.zValue = - np.sum(slice_normal * s_info.imagePositionPatient) / 10

            if 'FrameVOILUTSequence' in perFrameSeq:
                s_info.windowWidth = float(perFrameSeq.FrameVOILUTSequence[0].WindowWidth)
                s_info.windowCenter = float(perFrameSeq.FrameVOILUTSequence[0].WindowCenter)

            s_info = populate_radiopharma_fields(s_info, ds)

            scan_info[iFrame] = s_info

    return (scan_info, ds.pixel_array, ds.SeriesInstanceUID)

def load_sorted_scan_info(file_list):
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
        if np.any(ds.Modality == np.array(["CT","PT", "MR"])): #hasattr(ds, "pixel_array"):
            if len(file_list) == 1 and 'NumberOfFrames' in ds:
                multiFrameFlag = True
                si_pixel_data = parse_scan_info_fields(ds, multiFrameFlag)
                scan_array = np.transpose(si_pixel_data[1], (1,2,0))
                scan_info = si_pixel_data[0]
                count = len(scan_info)
            else:
                si_pixel_data = parse_scan_info_fields(ds)
                #scan_info.append(si_pixel_data[0])
                #scan_array.append(si_pixel_data[1])
                scan_info[count] = si_pixel_data[0]
                if not isinstance(scan_array, np.ndarray) and not scan_array:
                    imgSiz = list(si_pixel_data[1].shape)
                    imgSiz.append(len(file_list))
                    scan_array = np.empty(imgSiz)
                scan_array[:,:,count] = si_pixel_data[1]
                count += 1

    if count < scan_array.shape[2]:
        scan_array = np.delete(scan_array,np.arange(count,scan_array.shape[2]),axis=2)
        scan_info = np.delete(scan_info,np.arange(count,scan_array.shape[2]),axis=0)

    # Filter out duplicate SOP Instances
    if np.any(ds.Modality == np.array(["CT","PT", "MR"])) and not multiFrameFlag:
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
