# Module to export treatment plan (Beams) to RTPLAN DICOM
#
# APA, 7/3/2026

from datetime import datetime
import numpy as np
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from random import randint
from pydicom.uid import generate_uid
from cerr.dcm_export import iod_helper

org_root = '1.3.6.1.4.1.9590.100.1.2.'


def _isValid(val):
    """True when a dataclass field holds a real value (not nan/empty)."""
    if val is None:
        return False
    if isinstance(val, str):
        return val != ""
    if isinstance(val, float):
        return not np.isnan(val)
    if isinstance(val, np.ndarray):
        return val.size > 0
    return True


def _setIfValid(ds, tagName, val):
    if _isValid(val):
        if isinstance(val, np.ndarray):
            val = val.tolist()
        setattr(ds, tagName, val)


def getDoseRefSeq(beamsObj):
    seq = Sequence()
    for doseRef in np.atleast_1d(beamsObj.DoseReferenceSequence):
        dsRef = Dataset()
        dsRef.DoseReferenceNumber = len(seq) + 1
        dsRef.DoseReferenceStructureType = "SITE"
        _setIfValid(dsRef, 'DoseReferenceType', doseRef.DoseReferenceType)
        _setIfValid(dsRef, 'DoseReferenceUID', doseRef.DoseReferenceUID)
        _setIfValid(dsRef, 'ReferencedROINumber', doseRef.ReferencedROINumber)
        _setIfValid(dsRef, 'DeliveryMaximumDose', doseRef.DeliveryMaximumDose)
        _setIfValid(dsRef, 'TargetPrescriptionDose', doseRef.TargetPrescriptionDose)
        seq.append(dsRef)
    return seq


def getFractionGroupSeq(beamsObj):
    seq = Sequence()
    for fxGrp in np.atleast_1d(beamsObj.FractionGroupSequence):
        dsFx = Dataset()
        dsFx.FractionGroupNumber = fxGrp.FractionGroupNumber if fxGrp.FractionGroupNumber else len(seq) + 1
        dsFx.NumberOfFractionsPlanned = fxGrp.NumberOfFractionsPlanned
        dsFx.NumberOfBeams = fxGrp.NumberOfBeams
        dsFx.NumberOfBrachyApplicationSetups = fxGrp.NumberOfBrachyApplicationSetups
        refBeamSeq = fxGrp.RefBeamSeq
        if _isValid(refBeamSeq):
            dsRefBeam = Dataset()
            dsRefBeam.ReferencedBeamNumber = refBeamSeq.ReferencedBeamNumber
            _setIfValid(dsRefBeam, 'BeamMeterset', refBeamSeq.BeamMeterset)
            dsFx.ReferencedBeamSequence = Sequence([dsRefBeam])
        seq.append(dsFx)
    return seq


def getLimitingDeviceSeq(limDevSeqList, includePositions):
    """Build BeamLimitingDeviceSequence or BeamLimitingDevicePositionSequence items."""
    seq = Sequence()
    for limDev in np.atleast_1d(limDevSeqList):
        dsLim = Dataset()
        dsLim.RTBeamLimitingDeviceType = limDev.RTBeamLimitingDeviceType
        if _isValid(limDev.LeafJawPositions):
            if includePositions:
                dsLim.LeafJawPositions = [float(v) for v in limDev.LeafJawPositions]
            else:
                dsLim.NumberOfLeafJawPairs = int(len(limDev.LeafJawPositions) / 2)
        seq.append(dsLim)
    return seq


def getControlPointSeq(beamSeq):
    seq = Sequence()
    for ctrPt in np.atleast_1d(beamSeq.ControlPointSequence):
        dsCtr = Dataset()
        dsCtr.ControlPointIndex = int(ctrPt.ControlPointIndex)
        _setIfValid(dsCtr, 'NominalBeamEnergy', ctrPt.NominalBeamEnergy)
        _setIfValid(dsCtr, 'GantryAngle', ctrPt.GantryAngle)
        _setIfValid(dsCtr, 'GantryRotationDirection', ctrPt.GantryRotationDirection)
        _setIfValid(dsCtr, 'BeamLimitingDeviceAngle', ctrPt.BeamLimitingDeviceAngle)
        _setIfValid(dsCtr, 'BeamLimitingDeviceRotationDirection', ctrPt.BeamLimitingDeviceRotationDirection)
        _setIfValid(dsCtr, 'PatientSupportAngle', ctrPt.PatientSupportAngle)
        _setIfValid(dsCtr, 'TableTopEccentricAngle', ctrPt.TableTopEccentricAngle)
        _setIfValid(dsCtr, 'TableTopEccentricRotationDirection', ctrPt.TableTopEccentricRotationDirection)
        _setIfValid(dsCtr, 'SourceToSurfaceDistance', ctrPt.SourceToSurfaceDistance)
        _setIfValid(dsCtr, 'CumulativeMetersetWeight', ctrPt.CumulativeMetersetWeight)
        if _isValid(ctrPt.IsocenterPosition):
            dsCtr.IsocenterPosition = [float(v) for v in ctrPt.IsocenterPosition]
        if _isValid(ctrPt.BeamLimitingDevicePositionSequence):
            dsCtr.BeamLimitingDevicePositionSequence = \
                getLimitingDeviceSeq(ctrPt.BeamLimitingDevicePositionSequence, includePositions=True)
        seq.append(dsCtr)
    return seq


def getBeamSeq(beamsObj):
    seq = Sequence()
    for beam in np.atleast_1d(beamsObj.BeamSequence):
        dsBeam = Dataset()
        dsBeam.BeamNumber = int(beam.BeamNumber)
        dsBeam.BeamName = beam.BeamName
        _setIfValid(dsBeam, 'BeamDescription', beam.BeamDescription)
        dsBeam.BeamType = beam.BeamType
        dsBeam.RadiationType = beam.RadiationType
        dsBeam.TreatmentDeliveryType = beam.TreatmentDeliveryType
        _setIfValid(dsBeam, 'Manufacturer', beam.Manufacturer)
        dsBeam.SourceAxisDistance = float(beam.SourceAxisDistance)
        dsBeam.NumberOfWedges = int(beam.NumberOfWedges)
        dsBeam.NumberOfBoli = int(beam.NumberOfBoli)
        dsBeam.NumberOfCompensators = int(beam.NumberOfCompensators)
        dsBeam.NumberOfBlocks = int(beam.NumberOfBlocks)
        dsBeam.NumberOfControlPoints = int(beam.NumberOfControlPoints)
        if _isValid(beam.BeamLimitingDevicePositionSeq):
            dsBeam.BeamLimitingDeviceSequence = \
                getLimitingDeviceSeq(beam.BeamLimitingDevicePositionSeq, includePositions=False)
        if _isValid(beam.ControlPointSequence):
            dsBeam.ControlPointSequence = getControlPointSeq(beam)
        seq.append(dsBeam)
    return seq


def create(planNum, filePath, planC, seriesOpts={}):
    """Export a pyCERR Beams object to an RTPLAN DICOM file.

    The original SOPInstanceUID from import is preserved when available, so
    that exported RTDOSE objects referencing this plan remain valid.

    Args:
        planNum (int): Index of plan in planC.beams.
        filePath (str): Output DICOM file path.
        planC (cerr.plan_container.PlanC): pyCERR's plan container object.
        seriesOpts (dict): Optional 'seriesDescription', 'SeriesDate', 'SeriesTime'.
    """

    beamsObj = planC.beams[planNum]

    # load_beams stores PatientName under the lowercase 'patientName' attribute
    patientName = getattr(beamsObj, 'patientName', '') or beamsObj.PatientName
    pat_tags = {"PatientName": patientName,
                "PatientID": beamsObj.PatientID,
                "PatientBirthDate": beamsObj.PatientBirthDate,
                "PatientSex": beamsObj.PatientSex,
                "PatientAge": "",
                "PatientSize": "",
                "PatientWeight": ""
                }

    dt = datetime.now()
    series_tags = {
        'Modality': 'RTPLAN',
        'SeriesDate': seriesOpts.get('SeriesDate', dt.strftime("%Y%m%d")),
        'SeriesTime': seriesOpts.get('SeriesTime', dt.strftime("%H%M%S.%f")),
        'SeriesDescription': seriesOpts.get('seriesDescription', "Generated by pyCERR"),
        'SeriesInstanceUID': generate_uid(prefix=org_root),
        'SeriesNumber': str(randint(9010, 9900))
    }

    file_meta = iod_helper.getFileMeta('RTPLAN')
    if beamsObj.SOPInstanceUID:
        file_meta.MediaStorageSOPInstanceUID = beamsObj.SOPInstanceUID
    ds = FileDataset(filePath, {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds = iod_helper.addSOPCommonTags(ds)
    ds = iod_helper.addPatientTags(ds, pat_tags)
    ds = iod_helper.addSeriesTags(ds, series_tags)
    ds = iod_helper.addEquipmentTags(ds, {})

    # Study module: RTPLAN import does not retain study tags; generate a study
    ds.StudyDate = beamsObj.RTPlanDate or dt.strftime("%Y%m%d")
    ds.StudyTime = beamsObj.RTPlanTime or dt.strftime("%H%M%S.%f")
    ds.StudyInstanceUID = generate_uid(prefix=org_root)
    ds.StudyID = ""

    # RT General Plan module
    ds.RTPlanLabel = beamsObj.RTPlanLabel
    ds.RTPlanDate = beamsObj.RTPlanDate
    ds.RTPlanTime = beamsObj.RTPlanTime
    ds.RTPlanGeometry = beamsObj.RTPlanGeometry or "PATIENT"
    _setIfValid(ds, 'PrescriptionDescription', beamsObj.PrescriptionDescription)
    _setIfValid(ds, 'ApprovalStatus', beamsObj.ApprovalStatus)

    if _isValid(beamsObj.ReferencedStructureSetSequence):
        refSeq = Sequence()
        for ref in np.atleast_1d(beamsObj.ReferencedStructureSetSequence):
            dsRef = Dataset()
            dsRef.ReferencedSOPClassUID = ref.ReferencedSOPClassUID
            dsRef.ReferencedSOPInstanceUID = ref.ReferencedSOPInstanceUID
            refSeq.append(dsRef)
        ds.ReferencedStructureSetSequence = refSeq

    if _isValid(beamsObj.ReferencedDoseSequence):
        refSeq = Sequence()
        for ref in np.atleast_1d(beamsObj.ReferencedDoseSequence):
            dsRef = Dataset()
            dsRef.ReferencedSOPClassUID = ref.ReferencedSOPClassUID
            dsRef.ReferencedSOPInstanceUID = ref.ReferencedSOPInstanceUID
            refSeq.append(dsRef)
        ds.ReferencedDoseSequence = refSeq

    if _isValid(beamsObj.PatientSetupSequence):
        setupSeq = Sequence()
        for setup in np.atleast_1d(beamsObj.PatientSetupSequence):
            dsSetup = Dataset()
            dsSetup.PatientSetupNumber = setup.PatientSetupNumber
            dsSetup.PatientPosition = setup.PatientPosition
            setupSeq.append(dsSetup)
        ds.PatientSetupSequence = setupSeq

    if _isValid(beamsObj.DoseReferenceSequence):
        ds.DoseReferenceSequence = getDoseRefSeq(beamsObj)
    if _isValid(beamsObj.FractionGroupSequence):
        ds.FractionGroupSequence = getFractionGroupSeq(beamsObj)
    if _isValid(beamsObj.BeamSequence):
        ds.BeamSequence = getBeamSeq(beamsObj)

    print("Writing RTPLAN file ...", filePath)
    ds.save_as(filePath)
    print("File saved.")
