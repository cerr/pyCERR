from dataclasses import dataclass, field
from typing import List
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
class Beams:
    PatientName: str = ""
    PatientID: str = ""
    PatientBirthDate: str = ""
    PatientSex: str = ""
    Manufacturer: str = ""
    ManufacturerModelName: str = ""
    AcquisitionGroupLength: int = 0
    RelationshipGroupLength: int = 0
    ImagePresentationGroupLength: int = 0
    PixelPaddingValue: float = 0
    PlanGroupLength: int = 0
    RTPlanLabel: str = ""
    RTPlanDate: str = ""
    RTPlanTime: str = ""
    RTPlanGeometry: str = ""
    TreatmentSites: np.array = field(default_factory=get_empty_np_array)
    PrescriptionDescription: str = ""
    DoseReferenceSequence: np.array = field(default_factory=get_empty_np_array)
    FractionGroupSequence: np.array = field(default_factory=get_empty_np_array)
    BeamSequence: np.array = field(default_factory=get_empty_np_array)
    PatientSetupSequence: np.array = field(default_factory=get_empty_np_array)
    ReferencedRTGroupLength: int = 0
    ReferencedStructureSetSequence: np.array = field(default_factory=get_empty_np_array)
    ReferencedDoseSequence: np.array = field(default_factory=get_empty_np_array)
    ReviewGroupLength: int = 0
    ApprovalStatus: str = ""
    ReviewDate: str = ""
    ReviewTime: str = ""
    ReviewerName: str = ""
    SOPInstanceUID: str = ""
    BeamUID: str = ""

@dataclass
class ReferenceSeq:
    ReferencedSOPClassUID: str = ""
    ReferencedSOPInstanceUID: str = ""

@dataclass
class PatientSetupSeq:
    PatientSetupNumber: int = 0
    PatientPosition: str = ""

@dataclass
class BeamLimitingDevicePositionSeq:
    RTBeamLimitingDeviceType: str = ""
    LeafJawPositions: np.array = field(default_factory=get_empty_np_array)

@dataclass
class ControlPointSequence:
    BeamLimitingDevicePositionSequence: np.array = field(default_factory=get_empty_np_array)
    ControlPointIndex: int = 0
    NominalBeamEnergy: float = 0
    GantryAngle: float = 0
    GantryRotationDirection: str = ""
    BeamLimitingDeviceAngle: float = 0
    BeamLimitingDeviceRotationDirection: str = ""
    PatientSupportAngle: float = 0
    TableTopEccentricAngle: float = 0
    TableTopEccentricRotationDirection: str = ""
    IsocenterPosition: np.array = field(default_factory=get_empty_np_array)
    SourceToSurfaceDistance: float = 0
    CumulativeMetersetWeight: float = 0

@dataclass()
class RefBeamSeq:
    ReferencedBeamNumber: int = 0
    BeamMeterset: float = 0

@dataclass
class FractionGroupSeq:
    FractionGroupNumber: int = 0
    NumberOfFractionsPlanned: int = 0
    NumberOfBeams: int = 0
    NumberOfBrachyApplicationSetups: int = 0
    RadiationType: str = ""
    RefBeamSeq: np.array = field(default_factory=get_empty_np_array)

@dataclass
class BeamSeq:
    Manufacturer: str = ""
    BeamName: str = ""
    BeamType: str = ""
    BeamDescription: str = ""
    BeamNumber: int = 0
    SourceAxisDistance: float = 0
    BeamLimitingDevicePositionSeq: np.array = field(default_factory=get_empty_np_array)
    RadiationType: str = ""
    TreatmentDeliveryType: str = ""
    NumberOfWedges: float = 0
    NumberOfBoli: float = 0
    NumberOfCompensators: float = 0
    NumberOfBlocks: float = 0
    NumberOfControlPoints: float = 0
    ControlPointSequence: np.array = field(default_factory=get_empty_np_array)

    class json_serialize(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Beams):
                return {'beams':obj.BeamUID}
            return "" #json.JSONEncoder.default(self, obj)


def load_beams(file_list):
    beams_list = []
    for file in file_list:
        ds = dcmread(file)
        if ds.Modality == "RTPLAN":
            beams_meta = Beams()
            beams_meta.patientName = ds.PatientName
            beams_meta.PatientID = ds.PatientID
            beams_meta.PatientBirthDate = ds.PatientBirthDate
            beams_meta.PatientSex = ds.PatientSex
            beams_meta.SOPInstanceUID = ds.SOPInstanceUID
            ref_dose_seq_list = np.array([],dtype=ReferenceSeq)
            ref_str_seq_list = np.array([],dtype=ReferenceSeq)
            if hasattr(ds,"ReferencedStructureSetSequence"):
                #ref_str_seq_list = np.array([],dtype=ReferenceSeq)
                refStrSetSeq = ReferenceSeq()
                refStrSetSeq.ReferencedSOPClassUID = ds.ReferencedStructureSetSequence[0].ReferencedSOPClassUID
                refStrSetSeq.ReferencedSOPInstanceUID = ds.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
                ref_str_seq_list = np.append(ref_str_seq_list,refStrSetSeq)
                beams_meta.ReferencedStructureSetSequence = ref_str_seq_list
            if hasattr(ds,"ReferencedDoseSequence"):
                #ref_dose_seq_list = np.array([],dtype=ReferenceSeq)
                refDoseSetSeq = ReferenceSeq()
                refDoseSetSeq.ReferencedSOPClassUID = ds.ReferencedDoseSequence[0].ReferencedSOPClassUID
                refDoseSetSeq.ReferencedSOPInstanceUID = ds.ReferencedDoseSequence[0].ReferencedSOPInstanceUID
                ref_dose_seq_list = np.append(ref_dose_seq_list,refDoseSetSeq)
                beams_meta.ReferencedDoseSequence = ref_dose_seq_list
            if hasattr(ds,"PatientSetupSequence"):
                patient_setup_seq_list = np.array([],dtype=PatientSetupSeq)
                patientSetupSeq = PatientSetupSeq()
                patientSetupSeq.PatientPosition = ds.PatientSetupSequence[0].PatientPosition
                patientSetupSeq.PatientSetupNumber = ds.PatientSetupSequence[0].PatientSetupNumber
                ref_dose_seq_list = np.append(patient_setup_seq_list,patientSetupSeq)
                beams_meta.PatientSetupSequence = patient_setup_seq_list
            if hasattr(ds,"FractionGroupSequence"):
                fx_grp_list = np.array([])
                for fx_grp in ds.FractionGroupSequence:
                    f = FractionGroupSeq()
                    f.NumberOfBeams = fx_grp.NumberOfBeams
                    f.NumberOfFractionsPlanned = fx_grp.NumberOfFractionsPlanned
                    f.NumberOfBeams = fx_grp.NumberOfBeams
                    f.NumberOfBrachyApplicationSetups = fx_grp.NumberOfBrachyApplicationSetups
                    if hasattr(fx_grp,"RadiationType"): f.RadiationType = fx_grp.RadiationType
                    ref_beam_seq = RefBeamSeq()
                    if hasattr(fx_grp,"ReferencedBeamSequence"):
                        if hasattr(fx_grp.ReferencedBeamSequence[0],"BeamMeterset"):
                            ref_beam_seq.BeamMeterset = fx_grp.ReferencedBeamSequence[0].BeamMeterset
                        ref_beam_seq.ReferencedBeamNumber = fx_grp.ReferencedBeamSequence[0].ReferencedBeamNumber
                        f.RefBeamSeq = ref_beam_seq
                    # add code to read ReferencedBrachyApplicationSetupSequence
                    fx_grp_list = np.append(fx_grp_list,f)
                beams_meta.FractionGroupSequence = fx_grp_list
            if hasattr(ds,"BeamSequence"):
                beam_item_list = np.array([],dtype=BeamSeq)
                num_beams = len(ds.BeamSequence)
                for beam in ds.BeamSequence:
                    bs = BeamSeq()
                    bs.Manufacturer: beam.Manufacturer
                    bs.BeamName: beam.BeamName
                    bs.BeamType: beam.BeamType
                    bs.BeamDescription: beam.BeamDescription
                    bs.BeamNumber: beam.BeamNumber
                    bs.SourceAxisDistance: beam.SourceAxisDistance
                    limit_sevice_seq = np.array([])
                    for limitDevSeq in beam.BeamLimitingDeviceSequence:
                        seq = BeamLimitingDevicePositionSeq()
                        seq.RTBeamLimitingDeviceType = limitDevSeq.RTBeamLimitingDeviceType
                        if hasattr(limitDevSeq, "LeafJawPositions"):
                            seq.LeafJawPositions = limitDevSeq.LeafJawPositions
                        limit_sevice_seq = np.append(limit_sevice_seq,seq)
                    bs.BeamLimitingDevicePositionSeq = limit_sevice_seq
                    bs.RadiationType = beam.RadiationType
                    bs.TreatmentDeliveryType = beam.TreatmentDeliveryType
                    bs.NumberOfWedges = beam.NumberOfWedges
                    bs.NumberOfBoli = beam.NumberOfBoli
                    bs.NumberOfCompensators = beam.NumberOfCompensators
                    bs.NumberOfBlocks = beam.NumberOfBlocks
                    bs.NumberOfControlPoints = beam.NumberOfControlPoints
                    ctr_pt_seq = np.array([])
                    for ctr_pt in beam.ControlPointSequence:
                        c = ControlPointSequence()
                        beam_lim_pos_seq_list = np.array([])
                        if hasattr(ctr_pt, "BeamLimitingDevicePositionSequence"):
                            for lim_seq in ctr_pt.BeamLimitingDevicePositionSequence:
                                beam_lim_pos_seq = BeamLimitingDevicePositionSeq()
                                beam_lim_pos_seq.RTBeamLimitingDeviceType = lim_seq.RTBeamLimitingDeviceType
                                beam_lim_pos_seq.LeafJawPositions = lim_seq.LeafJawPositions
                                beam_lim_pos_seq_list = np.append(beam_lim_pos_seq_list,beam_lim_pos_seq)
                        c.BeamLimitingDevicePositionSequence = beam_lim_pos_seq_list
                        c.ControlPointIndex = ctr_pt.ControlPointIndex
                        if hasattr(ctr_pt, "NominalBeamEnergy"):
                            c.NominalBeamEnergy = ctr_pt.NominalBeamEnergy
                        if hasattr(ctr_pt, "GantryAngle"):
                            c.GantryAngle = ctr_pt.GantryAngle
                        if hasattr(ctr_pt, "GantryRotationDirection"):
                            c.GantryRotationDirection = ctr_pt.GantryRotationDirection
                        if hasattr(ctr_pt, "BeamLimitingDeviceAngle"):
                            c.BeamLimitingDeviceAngle = ctr_pt.BeamLimitingDeviceAngle
                        if hasattr(ctr_pt, "BeamLimitingDeviceRotationDirection"):
                            c.BeamLimitingDeviceRotationDirection = ctr_pt.BeamLimitingDeviceRotationDirection
                        if hasattr(ctr_pt, "PatientSupportAngle"):
                            c.PatientSupportAngle = ctr_pt.PatientSupportAngle
                        if hasattr(ctr_pt, "TableTopEccentricAngle"):
                            c.TableTopEccentricAngle = ctr_pt.TableTopEccentricAngle
                        if hasattr(ctr_pt, "TableTopEccentricRotationDirection"):
                            c.TableTopEccentricRotationDirection = ctr_pt.TableTopEccentricRotationDirection
                        if hasattr(ctr_pt, "IsocenterPosition"):
                            c.IsocenterPosition = ctr_pt.IsocenterPosition
                        if hasattr(ctr_pt, "SourceToSurfaceDistance"):
                            c.SourceToSurfaceDistance = ctr_pt.SourceToSurfaceDistance
                        if hasattr(ctr_pt, "CumulativeMetersetWeight"):
                            c.CumulativeMetersetWeight = ctr_pt.CumulativeMetersetWeight
                        ctr_pt_seq = np.append(ctr_pt_seq,c)
                    bs.ControlPointSequence = ctr_pt_seq
                    beam_item_list = np.append(beam_item_list,bs)
                beams_meta.BeamSequence = beam_item_list
            beams_meta.ReferencedDoseSequence = ref_dose_seq_list
            if hasattr(ds,"ReviewerName"): beams_meta.ReviewerName = ds.ReviewerName
            if hasattr(ds,"ReviewDate"): beams_meta.ReviewDate = ds.ReviewDate
            if hasattr(ds,"ReviewTime"): beams_meta.ReviewTime = ds.ReviewTime
            if hasattr(ds,"ApprovalStatus"): beams_meta.ApprovalStatus = ds.ApprovalStatus
            if hasattr(ds,"PixelPaddingValue"): beams_meta.PixelPaddingValue = ds.PixelPaddingValue
            if hasattr(ds,"Manufacturer"): beams_meta.Manufacturer = ds.Manufacturer
            if hasattr(ds,"ManufacturerModelName"): beams_meta.ManufacturerModelName = ds.ManufacturerModelName
            beams_meta.RTPlanLabel = ds.RTPlanLabel
            beams_meta.RTPlanDate = ds.RTPlanDate
            beams_meta.RTPlanDate = ds.RTPlanTime
            beams_meta.RTPlanGeometry = ds.RTPlanGeometry
            if hasattr(ds,"TreatmentSites"): beams_meta.TreatmentSites = ds.TreatmentSites
            if hasattr(ds,"PrescriptionDescription"): beams_meta.PrescriptionDescription = ds.PrescriptionDescription



            beams_meta.BeamUID = uid.createUID("beams")

            beams_list = np.append(beams_list, beams_meta)

    return beams_list
