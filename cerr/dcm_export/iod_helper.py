# Module to populate iod elements common between different modalities
#
# APA, 4/3/2023

import SimpleITK as sitk
import os
from datetime import datetime
import numpy as np
from pydicom import dcmread
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ImplicitVRLittleEndian
import scipy.io as sio
import argparse
from random import randint


def get_file_meta(dataType) -> FileMetaDataset:
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 202
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    if dataType == "CT":
        dataClassUID = "1.2.840.10008.5.1.4.1.1.2"
    elif dataType == "MR":
        dataClassUID = "1.2.840.10008.5.1.4.1.1.4"
    elif dataType == "SC":
        dataClassUID = "1.2.840.10008.5.1.4.1.1.7"
    elif dataType == "RTSTRUCT":
        dataClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    elif dataType == "REG":
        dataClassUID = "1.2.840.10008.5.1.4.1.1.66.3"
    elif dataType == "RTDOSE":
        dataClassUID = "1.2.840.10008.5.1.4.1.1.481.2"
    else:
        raise Exception("Unsupported data type")

    file_meta.MediaStorageSOPClassUID = dataClassUID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    #file_meta.ImplementationClassUID = "1.2.246.352.70.2.1.160.3"
    return file_meta


def add_equipment_tags(ds: FileDataset, equipDict):
    dt = datetime.now()
    ds.Manufacturer = "pyCERR"
    ds.ManufacturerModelName = "pyCERR"
    ds.InstitutionName = "MSKCC"
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    return ds

def add_study_tags(ds: FileDataset, studyDict):
    dt = datetime.now()
    ds.StudyDate = studyDict["StudyDate"]
    ds.StudyTime = studyDict["StudyTime"]
    ds.StudyDescription = studyDict["StudyDescription"]
    ds.StudyInstanceUID = studyDict["StudyInstanceUID"]
    ds.StudyID = studyDict["StudyID"]

    return ds

def add_series_tags(ds: FileDataset, seriesDict):
    dt = datetime.now()
    ds.Modality = seriesDict['Modality']
    ds.SeriesDate = dt.strftime("%Y%m%d")
    ds.SeriesTime = dt.strftime("%H%M%S.%f")
    ds.SeriesDescription = seriesDict["SeriesDescription"]
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = seriesDict["SeriesNumber"] #str(randint(9010, 9900))

    return ds

def add_patient_tags(ds: FileDataset, patDict):
    ds.PatientName = patDict["PatientName"]
    ds.PatientID = patDict["PatientID"]
    ds.PatientBirthDate = patDict["PatientBirthDate"]
    ds.PatientSex = patDict["PatientSex"]
    ds.PatientAge = patDict["PatientAge"]
    ds.PatientSize = patDict["PatientSize"]
    ds.PatientWeight = patDict["PatientWeight"]
    return ds

def add_content_tags(ds: FileDataset, contentDict):
    dt = datetime.now()
    ds.ContentCreatorName = ''
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S.%f")
    ds.ContentDescription = contentDict['ContentDescription'] #AI REGISTRATION'
    ds.ContentLabel = contentDict['ContentLabel'] #"REGISTRATION"

    return ds


def add_sop_common_tags(ds: FileDataset):
    dt = datetime.now()
    ds.SpecificCharacterSet = "ISO_IR 192"  # "ISO_IR 100"
    ds.InstanceCreationDate = dt.strftime("%Y%m%d")
    ds.InstanceCreationTime = dt.strftime("%H%M%S.%f")
    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    # Set values already defined in the file meta
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

    return ds


def add_general_image_tags(ds: FileDataset):
    # ds.InstanceNumber # 0020,0013
    pass

def add_image_plane_tags(ds: FileDataset):
    # ds.PixelSpacing # 0028,0030 (mm)
    # ds.ImageOrientation
    # ds.ImagePosition
    # ds.WindowCenter # optional, for scan
    # ds.WindowWidth # optional, for scan
    # ds.SliceThickness # optional, for scan
    pass

def add_image_pixel_tags(ds: FileDataset):
    pass


def add_ref_FOR_tags(ds_refFOR: Sequence):
    #ds_refFOR.FrameOfReference
    #s_refFOR.RTReferencedStudySequence = Sequence()
    pass


def add_structure_set_tags(ds: FileDataset, structureDict):
    # https://dicom.nema.org/dicom/2013/output/chtml/part03/sect_A.19.html
    dt = datetime.now()
    ds.StructureSetLabel = structureDict['StructureSetLabel'] # 3006,0002 Structure Set Label
    ds.StructureSetTime = dt.strftime("%H%M%S.%f") # 3006,0008 Structure Set Date
    ds.StructureSetDate = dt.strftime("%Y%m%d") # 3006,0009 Structure Set Time
    ds.StructureSetDescription = structureDict['StructureSetDescription'] # 3006,0006 Structure Set Description
    ds.InstanceNumber = structureDict['InstanceNumber'] # 3006,0013 Instance Number
    ds.ReferencedFrameOfReferenceSequence = Sequence()
    ds.StructureSetROISequence = Sequence()
    ds.ROIContourSequence = Sequence()
    ds.RTROIObservationsSequence = Sequence()

    return ds

