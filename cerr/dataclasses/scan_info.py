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
    """
    imageType (str):
    patientName (str):
    patientID (str):
    patientBirthDate (str):
    scanType (str):
    CTOffset (float):
    rescaleSlope (float):
    rescaleIntercept (float):
    rescaleType (str):
    scaleSlope (float):
    scaleIntercept (float):
    realWorldValueSlope (float):
    realWorldValueIntercept (float):
    realWorldMeasurCodeMeaning (str):
    philipsImageUnits (str):
    philipsRescaleSlope (float):
    philipsRescaleIntercept (float):
    grid1Units (float):
    grid2Units (float):
    numberRepresentation (int):
    bitsAllocated (int):
    bitsStored (int):
    pixelRepresentation (int):
    numberOfDimensions (int):
    sizeOfDimension1 (int):
    sizeOfDimension2 (int):
    zValue (float):
    xOffset (float):
    yOffset (float):
    CTAir (float):
    CTWater (float):
    sliceThickness (float):
    voxelThickness (float):
    siteOfInterest (str):
    unitNumber (int):
    seriesDescription (str):
    studyDescription (str):
    scannerType (str):
    manufacturer (str):
    scanFileName (str):
    headInOut (str):
    positionInScan (float):
    bValue (float):
    acquisitionDate (str):
    acquisitionTime (str):
    patientWeight (float):
    patientSize (float):
    patientBmi (float):
    patientSex (str):
    radiopharmaInfoS (float):
    injectionTime (str):
    injectionDate (str):
    injectedDose (float):
    halfLife (float):
    imageUnits (str):
    suvType (str):
    petCountSource (str):
    petSeriesType (str):
    petActivityConctrScaleFactor (float):
    petNumSlices (int):
    petIsDecayCorrected (str):
    petPrimarySourceOfCounts (str):
    petDecayCorrectionDateTime (str):
    decayCorrection (float):
    correctedImage (float):
    seriesDate (str):
    seriesTime (str):
    studyDate (str):
    studyTime (str):
    tapeOfOrigin (str):
    studyNumberOfOrigin (int):
    scanID (str):
    scanDate (str):
    CTScale (float):
    transferProtocol (str):
    studyInstanceUID (str):
    seriesInstanceUID (str):
    sopInstanceUID (str):
    sopClassUID (str):
    frameOfReferenceUID (str):
    patientPosition (str):
    imageOrientationPatient (np.array)
    imagePositionPatient (np.array):
    windowCenter (float):
    windowWidth (float):
    temporalPositionIndex (float):
    frameAcquisitionDuration (float):
    frameReferenceDateTime (str):
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
    scaleSlope: float = np.NAN
    scaleIntercept: float = np.NAN
    realWorldValueSlope: float = np.NAN
    realWorldValueIntercept: float = np.NAN
    realWorldMeasurCodeMeaning: str = ''
    philipsImageUnits: str = ''
    philipsRescaleSlope: float = np.NAN
    philipsRescaleIntercept: float = np.NAN
    grid1Units: float = 0.0
    grid2Units: float = 0.0
    numberRepresentation: int = 0
    bitsAllocated: int = np.NAN
    bitsStored: int = np.NAN
    pixelRepresentation: int = np.NAN
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
    petIsDecayCorrected: str = ''
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
    LRflippedToMatchPACS: int = 0
    APflippedToMatchPACS: int = 0
    SIflippedToMatchPACS: int = 0
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
