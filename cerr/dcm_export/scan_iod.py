# Module to export scan (CT, MR, PT, ...) to DICOM image series
#
# APA, 7/3/2026

import os
from datetime import datetime
import numpy as np
from pydicom.uid import generate_uid
from pydicom.dataset import FileDataset
from random import randint
from cerr.dcm_export import iod_helper

org_root = '1.3.6.1.4.1.9590.100.1.2.'

# Modalities with a native image storage SOP class; anything else (e.g.
# filtered/derived scans) is exported as Secondary Capture
standardModalities = ["CT", "MR", "PT", "US", "NM"]


def getScanModality(scanObj):
    """Return the DICOM modality string (e.g. 'CT', 'MR', 'PT') for a pyCERR Scan object."""
    imageType = scanObj.scanInfo[0].imageType
    modality = imageType.upper().replace("SCAN", "").strip()
    if modality == "":
        modality = "CT"
    return modality


def getDcmTagVals(scanNum, planC, seriesOpts={}):

    sInfo = planC.scan[scanNum].scanInfo[0]
    pat_tags = {"PatientName": sInfo.patientName,
                "PatientID": sInfo.patientID,
                "PatientBirthDate": sInfo.patientBirthDate,
                "PatientSex": sInfo.patientSex,
                "PatientAge": "",
                "PatientSize": sInfo.patientSize,
                "PatientWeight": sInfo.patientWeight
                }

    study_tags = {"StudyDate": sInfo.studyDate,
                  "StudyTime": sInfo.studyTime,
                  "StudyDescription": sInfo.studyDescription,
                  "StudyInstanceUID": sInfo.studyInstanceUID,
                  "StudyID": ""
                  }

    dt = datetime.now()
    seriesDescription = seriesOpts.get('seriesDescription', sInfo.seriesDescription)
    seriesDate = seriesOpts.get('SeriesDate', sInfo.seriesDate or dt.strftime("%Y%m%d"))
    seriesTime = seriesOpts.get('SeriesTime', sInfo.seriesTime or dt.strftime("%H%M%S.%f"))
    series_tags = {
        'Modality': getScanModality(planC.scan[scanNum]),
        'SeriesDate': seriesDate,
        'SeriesTime': seriesTime,
        'SeriesDescription': seriesDescription,
        'SeriesInstanceUID': sInfo.seriesInstanceUID,
        'SeriesNumber': str(randint(9010, 9900))
    }

    return pat_tags, study_tags, series_tags


def getScaledSlice(scanObj, slcNum):
    """Convert a scan slice from pyCERR intensity values back to stored DICOM pixel values.

    Returns:
        tuple: (pixel_array, rescaleSlope, rescaleIntercept, pixelRepresentation)
    """
    sInfo = scanObj.scanInfo[slcNum]
    slc = np.asarray(scanObj.scanArray[:, :, slcNum], dtype=float) - sInfo.CTOffset

    intSlc = np.round(slc)
    isIntLike = np.max(np.abs(slc - intSlc)) < 1e-4
    int16Info = np.iinfo(np.int16)
    if isIntLike and intSlc.min() >= int16Info.min and intSlc.max() <= int16Info.max:
        # Store values directly as signed 16-bit integers
        return intSlc.astype(np.int16), 1.0, 0.0, 1

    # Non-integer or out-of-range data (e.g. PET SUV): scale to unsigned 16-bit
    minVal, maxVal = float(slc.min()), float(slc.max())
    slope = (maxVal - minVal) / 65530 if maxVal > minVal else 1.0
    stored = np.round((slc - minVal) / slope).astype(np.uint16)
    return stored, slope, minVal, 0


def create(scanNum, dirPath, planC, seriesOpts={}):
    """Export a pyCERR scan to a DICOM image series (one file per slice).

    Original Study, Series, FrameOfReference and SOPInstance UIDs from import
    are preserved when available, so that exported RTSTRUCT/RTDOSE objects
    referencing this scan remain valid.

    Args:
        scanNum (int): Index of scan in planC.scan.
        dirPath (str): Existing directory to write the slice files to.
        planC (cerr.plan_container.PlanC): pyCERR's plan container object.
        seriesOpts (dict): Optional 'seriesDescription', 'SeriesDate', 'SeriesTime'.

    Returns:
        List[str]: Paths of the DICOM files written.
    """

    scanObj = planC.scan[scanNum]
    modality = getScanModality(scanObj)
    isDerived = modality not in standardModalities
    pat_tags, study_tags, series_tags = getDcmTagVals(scanNum, planC, seriesOpts)
    if isDerived:
        # Export as Secondary Capture (derived) images
        scanType = modality
        modality = "OT"
        series_tags['Modality'] = "OT"
        if not series_tags['SeriesDescription']:
            series_tags['SeriesDescription'] = scanType
    if not series_tags['SeriesInstanceUID']:
        series_tags['SeriesInstanceUID'] = generate_uid(prefix=org_root)
    if not study_tags['StudyInstanceUID']:
        study_tags['StudyInstanceUID'] = generate_uid(prefix=org_root)

    numSlcs = len(scanObj.scanInfo)
    filePaths = []
    for slcNum in range(numSlcs):
        sInfo = scanObj.scanInfo[slcNum]

        file_meta = iod_helper.getFileMeta("SC" if isDerived else modality)
        if sInfo.sopInstanceUID:
            file_meta.MediaStorageSOPInstanceUID = sInfo.sopInstanceUID
        fileName = os.path.join(dirPath, f"{modality}_{slcNum:04d}.dcm")
        ds = FileDataset(fileName, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds = iod_helper.addSOPCommonTags(ds)
        ds = iod_helper.addPatientTags(ds, pat_tags)
        ds = iod_helper.addStudyTags(ds, study_tags)
        ds = iod_helper.addSeriesTags(ds, series_tags)
        # addSeriesTags generates a fresh SeriesInstanceUID; restore series UID
        ds.SeriesInstanceUID = series_tags['SeriesInstanceUID']
        ds = iod_helper.addEquipmentTags(ds, {})

        # General/Image module
        ds.InstanceNumber = slcNum + 1
        if isDerived:
            # SC Equipment / derived image attributes
            ds.ImageType = ['DERIVED', 'SECONDARY']
            ds.ConversionType = 'WSD'
            ds.DerivationDescription = f"{scanType} derived image generated by pyCERR"
        ds.AcquisitionDate = sInfo.acquisitionDate
        ds.AcquisitionTime = sInfo.acquisitionTime
        ds.PatientPosition = sInfo.patientPosition
        ds.FrameOfReferenceUID = sInfo.frameOfReferenceUID

        # Image plane module (pyCERR grid units are cm; DICOM uses mm)
        ds.ImagePositionPatient = [float(v) for v in sInfo.imagePositionPatient]
        ds.ImageOrientationPatient = [float(v) for v in sInfo.imageOrientationPatient]
        ds.PixelSpacing = [float(sInfo.grid2Units) * 10, float(sInfo.grid1Units) * 10]
        if sInfo.sliceThickness:
            ds.SliceThickness = float(sInfo.sliceThickness) * 10

        # Image pixel module
        pixels, slope, intercept, pixelRep = getScaledSlice(scanObj, slcNum)
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = pixels.shape[0]
        ds.Columns = pixels.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = pixelRep
        ds.RescaleSlope = slope
        ds.RescaleIntercept = intercept
        if modality == "CT":
            ds.RescaleType = "HU"
        if sInfo.windowCenter not in ("", None):
            ds.WindowCenter = float(sInfo.windowCenter)
        if sInfo.windowWidth not in ("", None):
            ds.WindowWidth = float(sInfo.windowWidth)
        ds.PixelData = pixels.tobytes()

        ds.save_as(fileName)
        filePaths.append(fileName)

    print(f"Wrote {numSlcs} {modality} files to {dirPath}")
    return filePaths
