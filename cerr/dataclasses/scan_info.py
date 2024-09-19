"""scan_info module.

The scan_info module defines metadata for images (CT, PT, MR, US).
The metadata are attributes of the ScanInfo class. This metadata is used to generate
physical grid coordinates and for conversion of raw image to real world units.

"""

from dataclasses import dataclass, field
import numpy as np

def get_empty_list():
    return []
def get_empty_np_array():
    return np.empty((0,0,0))

@dataclass
class ScanInfo:
    """This class defines data object for a scan slice.

    Attributes:
        imageType (str): Type of scan. MR SCAN, CT SCAN, PT SCAN.
        patientName (str): Patient's name
        patientID (str): Patieint's ID
        patientBirthDate (str): Patient's date of birth
        CTOffset (float): Offset to add to scanArray to get image in scanUnits (e.g. HU). This is required so as to store
            only (+)ve values in scanArray.
        rescaleSlope (float): m in the equation Output units = m*SV + b. Slope to transform image from storage to
            real world units as per (0028,1053) tag.
        rescaleIntercept (float): b in the equation Output units = m*SV + b. Read from (0028,1052) tag
        rescaleType (str): Specifies the output units of Rescale Slope (0028,1053) and Rescale Intercept (0028,1052)
        scaleSlope (float): Private tag (2005,100E) used for conversion to MR precise values
        scaleIntercept (float): Private tag (2005,100D)
        realWorldValueSlope (float): The Slope value in relationship between stored values (SV) and the
            Real World Values as per (0040,9225) tag.
        realWorldValueIntercept (float): The Intercept value in relationship between stored values (SV) and
            the Real World values as per (0040,9224) tag.
        realWorldMeasurCodeMeaning (str): Real World code value as per (0008,0100) tag
        philipsImageUnits (str): Private tag containing image units as per (2005,140B)
        philipsRescaleSlope (float): Private tag containing rescale slope as per (2005,140A)
        philipsRescaleIntercept (float): Private tag containing rescale intercept as per (2005,1409)
        grid1Units (float): delta y of the scan grid in CERR virtual coordinates
        grid2Units (float): delta x of the scan grid in CERR virtual coordinates
        bitsAllocated (int): Number of bits allocated for each pixel sample as per (0028,0100)
        bitsStored (int): Number of bits stored for each pixel sample as per (0028,0101)
        pixelRepresentation (int): Data representation of the pixel samples as per (0028,0103)
            0000H - unsigned integer, 0001H - 2's complement
        numberOfDimensions (int): Number of scan dimensions
        sizeOfDimension1 (int): Number of rows of scanArray
        sizeOfDimension2 (int): Number of columns of scanArray
        zValue (float): z-coordinate of the slice in CERR virtual coordinate system
        xOffset (float): x-offset of the center of scanArray in CERR virtual coordinates
        yOffset (float): y-offset of the center of scanArray in CERR virtual coordinates
        sliceThickness (float): Nominal slice thickness as per (0018,0050).
        voxelThickness (float): Physical spacing between the next and the previous slice in CERR virtual coordinates.
        seriesDescription (str): Series description as per (0008,103E)
        studyDescription (str): Study description as per (0008,1030)
        scannerType (str): Manufacturer model name
        manufacturer (str): Scanner Mmanufacturer
        scanFileName (str): Location of DICOM file from which metadata was read
        bValue (float): b-value of MR scan
        acquisitionDate (str): Acquisition date
        acquisitionTime (str): Acquisition time
        patientWeight (float): Patient's weight
        patientSize (float): Patient's size
        patientBmi (float): Patient's BMI
        patientSex (str): Patient's gender
        injectionTime (str): The actual time of radiopharmaceutical administration to the patient for imaging purposes.
        injectionDate (str): The actual date of radiopharmaceutical administration to the patient for imaging purposes.
        injectedDose (float):  The radiopharmaceutical dose administered to the patient measured in MegaBecquerels (MBq)
            at the Radiopharmaceutical Start DateTime (0018,1078).
        halfLife (float): The radionuclide half life, in seconds, that was used in the correction of this image.
        imageUnits (str): realWorldMeasurCodeMeaning when available or rescaleType.
        suvType (str): The type of SUV stored in scanArray as per (0054,1006) tag.
        petCountSource (str): The primary source of counts as per (0054,1002).  EMISSION or TRANSMISSION
        petSeriesType (str): A multi-valued indicator of the type of Series as per (0054,1000).
        petActivityConctrScaleFactor (float): Used to convert the pixel data from counts to Activity Concentration (in Bq/ml) as per (7053, 1009 tag
        petNumSlices (int): The number of slices in each separate volume as per (0054,0081)
        petDecayCorrectionDateTime (str): The date and time to which all frames in this Image were decay corrected as per (0018,9701)
        decayCorrection (float): Whether Decay (DECY) correction has been applied to image. YES or NO.
        correctedImage (float): One or more values that indicate which, if any, corrections have been applied to the image as per (0028,0051)
        seriesDate (str): Series date
        seriesTime (str): Series yime
        studyDate (str): Study date
        studyTime (str): Study time
        studyInstanceUID (str): Study Instance UID
        seriesInstanceUID (str): Series Instance UID of image volume
        sopInstanceUID (str): SOP Instance UID of image frame
        sopClassUID (str): SOP Class UID of image frame
        frameOfReferenceUID (str): Frame of Reference UID
        imageOrientationPatient (np.array): Direction cosine of dose row and column with patient coordinate system.
        imagePositionPatient (np.array): x,y,z coordinate of the top left voxel of the scan volume.
        windowCenter (float): Window center used for visualization
        windowWidth (float): Window width used for visualization
        temporalPositionIndex (float): Temporal position in the dynamic sequence from the FrameContentSequence.
        triggerTime (float): Time, in msec, between peak of the R wave and the peak of the echo produced.
        frameAcquisitionDuration (float): Duration of Frame acquisition from the FrameContentSequence.
        frameReferenceDateTime (str): FrameReferenceDateTime from the FrameContentSequence

    """

    imageNumber: float = 0.0
    imageType: str = ''
    caseNumber: int = 0
    patientName: str = ''
    patientID: str = ''
    patientBirthDate: str = ''
    scanType: str = ''
    CTOffset: float = 0.0
    rescaleSlope: float = 1.0
    rescaleIntercept: float = 0.0
    rescaleType: str = ''
    scaleSlope: float = np.nan
    scaleIntercept: float = np.nan
    realWorldValueSlope: float = np.nan
    realWorldValueIntercept: float = np.nan
    realWorldMeasurCodeMeaning: str = ''
    philipsImageUnits: str = ''
    philipsRescaleSlope: float = np.nan
    philipsRescaleIntercept: float = np.nan
    grid1Units: float = 0.0
    grid2Units: float = 0.0
    numberRepresentation: int = 0
    bitsAllocated: int = np.nan
    bitsStored: int = np.nan
    pixelRepresentation: int = np.nan
    numberOfDimensions: int = 3
    sizeOfDimension1: int = 512
    sizeOfDimension2: int = 512
    zValue: float = 0.0
    xOffset: float = 0.0
    yOffset: float = 0.0
    CTAir: float = 0.0
    CTWater: float = 0.0
    sliceThickness: float = 0.0
    voxelThickness: float = 1.0
    siteOfInterest: str = ''
    unitNumber: int = 0
    seriesDescription: str = ''
    studyDescription: str = ''
    scannerType: str = ''
    manufacturer: str = ''
    scanFileName: str = ''
    headInOut: str = ''
    positionInScan: float = ''
    patientAttitude: str = ''
    bValue: float = 0.0
    acquisitionDate: str = ''
    acquisitionTime: str = ''
    patientWeight: float = '' #np.NAN
    patientSize: float = '' #np.NAN
    patientBmi: float = '' #np.NAN
    patientSex: str = ''
    radiopharmaInfoS: float = 0.0
    injectionTime: str = ''
    injectionDate: str = ''
    injectedDose: float = '' #np.NAN
    halfLife: float = '' #np.NAN
    imageUnits: str = ''
    suvType: str = ''
    petCountSource: str = ''
    petSeriesType: str = ''
    petActivityConctrScaleFactor: float = '' #np.NAN
    petNumSlices: int = '' #np.NAN
    petPrimarySourceOfCounts: str = ''
    petDecayCorrectionDateTime: str = ''
    decayCorrection: float = '' #np.NAN
    correctedImage: float = '' #np.NAN
    seriesDate: str = ''
    seriesTime: str = ''
    studyDate: str = ''
    studyTime: str = ''
    tapeOfOrigin: str = ''
    studyNumberOfOrigin: int = 0
    scanID: str = ''
    scanNumber: int = 0
    scanDate: str = ''
    CTScale: float = 0.0
    distrustAbove: float = '' #np.NAN
    imageSource: str = ''
    transferProtocol: str = ''
    studyInstanceUID: str = ''
    seriesInstanceUID: str = ''
    sopInstanceUID: str = ''
    sopClassUID: str = ''
    frameOfReferenceUID: str = ''
    patientPosition: str = ''
    imageOrientationPatient: np.array = field(default_factory=get_empty_np_array)
    imagePositionPatient: np.array = field(default_factory=get_empty_np_array)
    windowCenter: float = '' #np.NAN
    windowWidth: float = '' #np.NAN
    temporalPositionIndex: float = '' #np.NAN
    triggerTime: float = ''
    frameAcquisitionDuration: float = '' #np.NAN
    frameReferenceDateTime: str = ''

@dataclass
class UniformScanInfo:
    sliceNumSup: int = 0
    sliceNumInf: int = 0
    supInfScansCreated: int = 0
    minCTValue: float = 0.0
    maxCTValue: float = 0.0
    firstZValue: float = 0.0

def deduce_voxel_thickness(scan_info) -> ScanInfo:
    zValsV = np.array([s.zValue for s in scan_info])
    if len(zValsV) == 1:
        voxel_thickness = 1
    else:
        thick_fwd = np.diff(zValsV)
        thick_bkw = - np.diff(zValsV[::-1])
        thick_bkw = thick_bkw[::-1]
        voxel_thickness = np.zeros(len(zValsV), dtype = float)
        voxel_thickness[1:-1] = (thick_fwd[1:] + thick_bkw[:-1]) / 2
        voxel_thickness[0] = thick_fwd[0]
        voxel_thickness[-1] = thick_fwd[-1]
    for slc in range(len(scan_info)):
        scan_info[slc].voxelThickness = voxel_thickness[slc]
    return scan_info
