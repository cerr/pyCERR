from dataclasses import dataclass, field
from typing import List
import numpy as np
import os
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
    cerrDcmSliceDirMatch: bool = False

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
        scan3M = self.scanArray - self.scanInfo[0].CTOffset
        return scan3M

    def get_nii_affine(self):
        # https://neurostars.org/t/direction-orientation-matrix-dicom-vs-nifti/14382/2
        affine3M = self.Image2PhysicalTransM.copy()
        affine3M[0,:] = -affine3M[0,:] * 10 #nii row is reverse of dicom, cm to mm
        affine3M[1,:] = -affine3M[1,:] * 10 #nii col is reverse of dicom, cm to mm
        affine3M[2,:] = affine3M[2,:] * 10 # cm to mm
        return affine3M


    def save_nii(self, niiFileName):
        affine3M = self.get_nii_affine()
        scan3M = self.scanArray - self.scanInfo[0].CTOffset
        scan3M = np.moveaxis(scan3M,[0,1],[1,0])
        #scan3M = np.flip(scan3M,axis=[0,1]) # negated affineM to take care of reverse row/col compared to dicom
        if not self.cerrDcmSliceDirMatch:
            scan3M = np.flip(scan3M,axis=2)
        img = nib.Nifti1Image(scan3M, affine3M)
        success = nib.save(img, niiFileName)
        return success

    def getSitkImage(self):
        sitkArray = np.moveaxis(self.getScanArray(),[0,1,2],[1,2,0])
        if not self.cerrDcmSliceDirMatch:
            sitkArray = np.flip(sitkArray, axis = 0)
        originXyz = list(np.matmul(self.Image2PhysicalTransM, np.asarray([0,0,0,1]).T)[:3] * 10)
        xV, yV, zV = self.getScanXYZVals()
        dx = np.abs(xV[1] - xV[0])
        dy = np.abs(xV[1] - xV[0])
        dz = np.abs(xV[1] - xV[0])
        spacing = list([dx, dy, dz] * 10)
        img_ori = self.scanInfo[0].imageOrientationPatient
        slice_normal = img_ori[[1,2,0]] * img_ori[[5,3,4]] \
                       - img_ori[[2,0,1]] * img_ori[[4,5,3]]
        direction = list(np.hstack((img_ori,slice_normal)))
        img = sitk.GetImageFromArray(sitkArray)
        img.SetOrigin(originXyz)
        img.SetSpacing(spacing)
        img.SetDirection(direction)
        return img


    def getScanXYZVals(self):

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

    def getScanSpacing(self):
        x_vals_v, y_vals_v, z_vals_v = self.getScanXYZVals()
        if y_vals_v[0] > y_vals_v[1]:
            y_vals_v = np.flip(y_vals_v)
        dx = abs(np.median(np.diff(x_vals_v)))
        dy = abs(np.median(np.diff(y_vals_v)))
        dz = abs(np.median(np.diff(z_vals_v)))
        spacing_v = np.array([dx, dy, dz])
        return spacing_v

    def isCerrSliceOrderMatchDcm(self):
        img_ori = self.scanInfo[0].imageOrientationPatient
        img_ori = img_ori.reshape(6,1)
        slice_normal = img_ori[[1,2,0]] * img_ori[[5,3,4]] \
                       - img_ori[[2,0,1]] * img_ori[[4,5,3]]
        lastCerrZ = self.scanInfo[-1].zValue
        firstCerrZ = self.scanInfo[0].zValue
        ippFirst = self.scanInfo[0].imagePositionPatient
        ippLast = self.scanInfo[0].imagePositionPatient
        ippDistFirstSlc = np.sum(slice_normal * ippFirst.reshape(slice_normal.shape))
        ippDistLastSlc = np.sum(slice_normal * ippLast.reshape(slice_normal.shape))
        cerrDcmSliceDirMatch = np.sign(ippDistLastSlc - ippDistFirstSlc) == np.sign(lastCerrZ - firstCerrZ)
        self.cerrDcmSliceDirMatch = cerrDcmSliceDirMatch
        return cerrDcmSliceDirMatch

    def convertDcmToCerrVirtualCoords(self):

        # Compute slice normal
        img_ori = self.scanInfo[0].imageOrientationPatient
        img_ori = img_ori.reshape(6,1)
        slice_normal = img_ori[[1,2,0]] * img_ori[[5,3,4]] \
                       - img_ori[[2,0,1]] * img_ori[[4,5,3]]

        # # Calculate the distance of ‘ImagePositionPatient’ along the slice direction cosine
        # numSlcs = len(self.scanInfo)
        # distV = np.zeros(numSlcs)
        # for slcNum in range(numSlcs):
        #     ipp = self.scanInfo[slcNum].imagePositionPatient
        #     distV[slcNum] = np.sum(slice_normal * ipp.reshape(slice_normal.shape))

        # Construct DICOM Affine transformation matrix
        # CERR scanArray and scanInfo are sorted in reverse direction of imagePositionPatient
        # along imageOrieintationPatient. This results in order of CERR scanArray slices to match
        # DICOM slice direction for orientations such as HFP and opposite to DICOM slice direction
        # for orientations such as HFS. To construct DICOM affine transformation matrix it is
        # necessary to figure out whether CERR slice direction matches DICOM to get the position of
        # the 1st slice according to DICOM convention.
        cerrDcmSliceDirMatch = self.isCerrSliceOrderMatchDcm()
        if cerrDcmSliceDirMatch:
            info1 = self.scanInfo[1]
            info2 = self.scanInfo[2]
        else:
            info1 = self.scanInfo[-1]
            info2 = self.scanInfo[-2]

        pos1V = info1.imagePositionPatient / 10  # cm
        pos2V = info2.imagePositionPatient / 10  # cm
        deltaPosV = pos2V - pos1V
        pixelSpacing = [info1.grid2Units, info1.grid1Units]

        # Pt coordinate to DICOM image coordinate mapping
        # Based on ref: https://nipy.org/nibabel/dicom/dicom_orientation.html
        position_matrix = np.hstack((np.matmul(img_ori.reshape(3, 2,order="F"),np.diag(pixelSpacing)),
                                    np.array([[deltaPosV[0], pos1V[0]], [deltaPosV[1], pos1V[1]], [deltaPosV[2], pos1V[2]]])))

        position_matrix = np.vstack((position_matrix, np.array([0, 0, 0, 1])))

        positionMatrixInv = np.linalg.inv(position_matrix)
        self.Image2PhysicalTransM = position_matrix

        # Get DICOM x,y,z coordinates of the center voxel.
        # This serves as the reference point for the image volume.
        sizV = self.scanArray.shape
        xyzCtrV = position_matrix * np.array([(sizV[1] - 1) / 2, (sizV[0] - 1) / 2, 0, 1])
        xOffset = np.sum(np.matmul(np.transpose(img_ori[:3,:]), xyzCtrV[:3]))
        yOffset = -np.sum(np.matmul(np.transpose(img_ori[3:,:]), xyzCtrV[:3]))  # (-)ve since CERR y-coordinate is opposite of column vector.

        for i in range(len(self.scanInfo)):
            self.scanInfo[i].xOffset = xOffset
            self.scanInfo[i].yOffset = yOffset

        xs, ys, zs = self.getScanXYZVals()
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        if cerrDcmSliceDirMatch:
            slice_distance = (info2.zValue - info1.zValue) # 2nd - 1st
            virPosMtx = np.array([[dx, 0, 0, xs[0]], [0, dy, 0, ys[0]], [0, 0, slice_distance, zs[0]], [0, 0, 0, 1]])
        else: # Apply reflection transform along slices
            slice_distance = (info1.zValue - info2.zValue) # last - secondFromLast
            virPosMtx = np.array([[dx, 0, 0, xs[0]], [0, dy, 0, ys[0]], [0, 0, -slice_distance, zs[-1]], [0, 0, 0, 1]])
        self.Image2VirtualPhysicalTransM = virPosMtx

        # Construct transformation matrix to convert cerr-xyz to dicom-xyz
        self.cerrToDcmTransM = np.matmul(self.Image2PhysicalTransM, np.linalg.inv(self.Image2VirtualPhysicalTransM))
        self.cerrToDcmTransM[:,:3] = self.cerrToDcmTransM[:,:3] * 10 # cm to mm


    def convertDcmToRealWorldUnits(self):

        store_scan_as_mr_philips_precise_value = "yes"

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
            manufacturer = self.scanInfo[slcNum].manufacturer

            if 'philips' in manufacturer.lower() and \
                    realWorldValueSlope is not None and \
                    not np.isnan(realWorldValueSlope) and \
                    self.scanInfo[slcNum].imageType.lower() == 'mr scan' and \
                    realWorldMeasurCodeMeaning is not None and \
                    philipsImageUnits.lower() not in ['no units','normalized']:
                realWorldImageFlag = True
                scanArray3M[:, :, slcNum] = \
                    self.scanArray[:, :, slcNum] * realWorldValueSlope + realWorldValueIntercept
            else:
                scanArray3M[:, :, slcNum] = \
                    self.scanArray[:, :, slcNum] * rescaleSlope + rescaleIntrcpt

            rescaleSlopeV[slcNum] = rescaleSlope

        minScanVal = np.min(scanArray3M)
        ctOffset = max(0, -minScanVal)
        scanArray3M += ctOffset
        minScanVal = np.min(scanArray3M)
        maxScanVal = np.max(scanArray3M)

        if not realWorldImageFlag and not np.any(np.abs(rescaleSlopeV - 1) > np.finfo(float).eps * 1e5):
            # Convert to uint if rescale slope is not 1
            if minScanVal >= -32768 and maxScanVal <= 32767:
                scanArray3M = scanArray3M.astype(np.uint16)
            else:
                scanArray3M = scanArray3M.astype(np.uint32)

        for slcNum in range(numSlcs):
            self.scanInfo[slcNum].CTOffset = ctOffset

        self.scanArray = scanArray3M

        # Apply scale slope & intercept for Philips data if not realWorldValue
        if self.scanInfo[slcNum].imageType.lower() == 'mr scan' and \
                store_scan_as_mr_philips_precise_value.lower() == 'yes':
            # Ref: Chenevert, Thomas L., et al. "Errors in quantitative image analysis due to platform-dependent image scaling."
            manufacturer = self.scanInfo[0].manufacturer
            if 'philips' in manufacturer.lower() and \
                    self.scanInfo[0].scaleSlope is not None and \
                    not realWorldImageFlag:
                scaleSlope = self.scanInfo[0].scaleSlope
                self.scanArray = self.scanArray.astype(np.float32) / (rescaleSlope * scaleSlope)

    def convert_to_suv(self,suvType="BW"):

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
            seriesDate = np.datetime64(headerSlcS.seriesDate, 'D')
        injectionDate = np.nan
        if headerSlcS.injectionDate:
            injectionDate = np.datetime64(headerSlcS.injectionDate, 'D')
        acqStartTime = np.nan
        if not np.any(np.isnan(acqTimeV)):
            acqStartTime = np.min(acqTimeV)

        for slcNum in range(scan3M.shape[2]):
            headerSlcS = headerS[slcNum]
            imgM = scan3M[:, :, slcNum] - headerSlcS.CTOffset

            imgUnits = headerSlcS.imageUnits
            imgMUnits = imgM.copy()
            if imgUnits == 'CNTS':
                activityScaleFactor = headerSlcS.petActivityConcentrationScaleFactor
                imgMUnits = imgMUnits * activityScaleFactor
                imgMUnits = imgMUnits * 1000  # Bq/L
            elif imgUnits in ['BQML', 'BQCC']:
                imgMUnits = imgMUnits * 1000  # Bq/L
            elif imgUnits in ['KBQCC', 'KBQML']:
                imgMUnits = imgMUnits * 1e6  # Bq/L
            else:
                raise ValueError('SUV calculation is supported only for imageUnits BQML and CNTS')

            decayCorrection = headerSlcS.decayCorrection
            if decayCorrection == 'START':
                scantime = seriesTime
                if not np.isnan(acqStartTime) and acqStartTime < scantime:
                    scantime = acqStartTime
            elif decayCorrection == 'ADMIN':
                scantime = dcm_hhmmss(headerSlcS.injectionTime)
            elif decayCorrection == 'NONE':
                scantime = np.nan
            else:
                scantime = dcm_hhmmss(headerSlcS.petDecayCorrectionDateTime[8:])

            # Start Time for Radiopharmaceutical Injection
            injection_time = dcm_hhmmss(headerSlcS.injectionTime)[0]

            if not np.isnan(seriesDate) and not np.isnan(injectionDate):
                date_diff = seriesDate - injectionDate
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

        self.scanArray = suv3M

def dcm_hhmmss(time_str):
    hh = int(time_str[0:2])
    mm = int(time_str[2:4])
    ss = int(time_str[4:6])
    fract = None  # You can implement this part if needed

    totSec = hh * 3600 + mm * 60 + ss
    return totSec, hh, mm, ss, fract

def get_slice_position(scan_info_item):
    return scan_info_item[1].zValue

def parse_scan_info_fields(ds) -> (scn_info.ScanInfo,Dataset.pixel_array):
    s_info = scn_info.ScanInfo()
    s_info.frameOfReferenceUID = ds.FrameOfReferenceUID
    #s_info.seriesDescription = ds.seriesDescription
    if hasattr(ds,'RescaleSlope'): s_info.rescaleSlope = ds.RescaleSlope
    if hasattr(ds,'RescaleIntercept'): s_info.rescaleIntercept = ds.RescaleIntercept
    if ("2005","100E") in ds: s_info.scaleSlope = ds["2005","100E"].value
    if ("2005","100D") in ds: s_info.scaleIntercept = ds["2005","100D"].value
    if hasattr(ds,'RealWorldValueSlope'): s_info.realWorldValueSlope = ds.RealWorldValueSlope
    if ("0040","9096") in ds:
        s_info.realWorldValueIntercept = ds["0040","9096"][0]["0040","9224"].value
        s_info.realWorldValueSlope = ds["0040","9096"][0]["0040","9225"].value
        if ("0040","08EA") in ds["0040","9096"][0]:
            s_info.realWorldMeasurCodeMeaning = ds["0040","9096"][0]["0040","08EA"][0]["0008","0119"].value

    if ("2005","140B") in ds: s_info.philipsImageUnits = ds["2005","140B"].value
    if ("2005","140A") in ds: s_info.philipsRescaleSlope = ds["2005","140A"].value
    if ("2005","1409") in ds: s_info.philipsRescaleIntercept = ds["2005","1409"].value
    s_info.grid1Units = ds.PixelSpacing[1] / 10
    s_info.grid2Units = ds.PixelSpacing[0] / 10
    s_info.sizeOfDimension1 = ds.Rows
    s_info.sizeOfDimension2 = ds.Columns
    s_info.imageOrientationPatient = np.array(ds.ImageOrientationPatient)
    s_info.imagePositionPatient = np.array(ds.ImagePositionPatient)
    slice_normal = s_info.imageOrientationPatient[[1,2,0]] * s_info.imageOrientationPatient[[5,3,4]] \
                   - s_info.imageOrientationPatient[[2,0,1]] * s_info.imageOrientationPatient[[4,5,3]]
    s_info.zValue = - np.sum(slice_normal * s_info.imagePositionPatient) / 10
    #s_info.xOffset
    #s_info.yOffset
    #s_info.CTAir
    #s_info.CTWater
    s_info.sliceThickness = ds.SliceThickness / 10
    #s_info.siteOfInterest
    #s_info.unitNumber = ds.ManufacturerModelName
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

    #bValue
    if hasattr(ds,"PatientName"): s_info.patientName = ds.PatientName
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
    if ("0054","1006") in ds: s_info.suvType = ds["0054","1006"].value
    if hasattr(ds,"SeriesType"): s_info.petSeriesType = ds.SeriesType
    if hasattr(ds,"Units"): s_info.imageUnits = ds.Units
    if hasattr(ds,"CountsSource"): s_info.petCountSource = ds.CountsSource
    if hasattr(ds,"NumberOfSlices"): s_info.petNumSlices = ds.NumberOfSlices
    if hasattr(ds,"DecayCorrection"): s_info.petDecayCorrection = ds.DecayCorrection
    if hasattr(ds,"CorrectedImage"): s_info.petCorrectedImage = ds.CorrectedImage
    if hasattr(ds,"WindowCenter"): s_info.windowCenter = ds.WindowCenter
    if hasattr(ds,"WindowWidth"): s_info.windowWidth = ds.WindowWidth

    # populate radiopharma info
    if ("0054","0016") in ds:
        radiopharmaInfoSeq = ds["0054","0016"].value[0]
        if hasattr(radiopharmaInfoSeq,"RadiopharmaceuticalStartDateTime"):
            s_info.injectionDate = radiopharmaInfoSeq.RadiopharmaceuticalStartDateTime[:8]
            s_info.injectionTime = radiopharmaInfoSeq.RadiopharmaceuticalStartDateTime[8:]
        elif hasattr(radiopharmaInfoSeq,"RadiopharmaceuticalStartTime"):
            s_info.injectionTime = radiopharmaInfoSeq.RadiopharmaceuticalStartTime
        s_info.injectedDose = radiopharmaInfoSeq.RadionuclideTotalDose
        s_info.halfLife = radiopharmaInfoSeq.RadionuclideHalfLife
        if ("7053","1009") in ds: s_info.petActivityConctrScaleFactor = ds["7053","1009"].value

    return (s_info,ds.pixel_array,ds.SeriesInstanceUID)

def load_sorted_scan_info(file_list):
    scan = Scan()
    #scan_info = [] #scn_info.ScanInfo()
    #scan_array = []
    scan_array = [] #np.empty(len(file_list))
    scan_info = np.empty(len(file_list),dtype=scn_info.ScanInfo)
    count = 0
    for file in file_list:
        ds = dcmread(file)
        if np.any(ds.Modality == np.array(["CT","PT", "MR"])): #hasattr(ds, "pixel_array"):
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

def import_nii(file_list, planC):
    pass

def import_array(scan3M, xV, yV, zV, modality, assocScanNum, planC):
    scan = Scan()
    scan_info = [] #scn_info.ScanInfo()
    scan_array = []
    siz = scan3M.shape
    for slc in range(siz[2]):
        #ds = dcmread(file)
        #si_pixel_data = parse_scan_info_fields(ds)
        si_pixel_data = (0,0)
        scan_info.append(si_pixel_data[0])
        scan_array.append(si_pixel_data[1])
    #sorted_indices = scan_info.sort(key=get_slice_position, reverse=False)
    sort_index = [i for i,x in sorted(enumerate(scan_info),key=get_slice_position, reverse=False)]
    scan_array = np.array(scan_array)
    scan_array = np.moveaxis(scan_array,[0,1,2],[2,0,1])
    scan_info = np.array(scan_info)
    scan_info = scan_info[sort_index]
    scan_array = scan_array[:,:,sort_index]
    scan_info = scn_info.deduce_voxel_thickness(scan_info)
    scan.scanInfo = scan_info
    scan.scanArray = scan_array
    scan.scanUID = "CT." + si_pixel_data[2]
    return scan

def getScanNumFromUID(assocScanUID,planC) -> int:
    uid_list = [s.scanUID for s in planC.scan]
    if assocScanUID in uid_list:
        return uid_list.index(assocScanUID)
    else:
        return None

