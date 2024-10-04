"""beams module.

The beams module defines metadata for RT Plan (RTPLAN).
The metadata are attributes of the Beams class.

"""

from dataclasses import dataclass, field
import numpy as np
from pydicom import dcmread
from cerr.utils import uid
import json

def get_empty_list():
    return []
def get_empty_np_array():
    return np.empty((0,0,0))

@dataclass
class Beams:
    """This class defines data object for Beams in an RTPlan. The metadata is populated from DICOM.
    Attributes:
        PatientName (str): Patient's name
        PatientID (str): Patient's ID
        PatientBirthDate (str): Patient's birth date
        PatientSex (str): Patient's gender
        Manufacturer (str): Equipment manufacturer
        ManufacturerModelName (str): Equipment model
        PixelPaddingValue (float): Pixel padding value as defined in (0028,0120) DICOM tag
        RTPlanLabel (str): User-defined label for treatment plan as defined by (300A,0002) DICOM tag
        RTPlanDate (str): Date treatment plan was last modified as defined by (300A,0006) DICOM tag
        RTPlanTime (str): Time treatment plan was last modified as defined by (300A,0007) DICOM tag
        RTPlanGeometry (str): Describes whether RT Plan is based on patient or treatment device geometry (300A,000C) tag
        TreatmentSites (np.array): A free-text label describing the anatomical treatment site. (3010,0077) tag
        PrescriptionDescription (str): User-defined description of treatment prescription as defined in (300A,000E)
        FractionGroupSequence (np.array): Sequence of Fraction Groups as per (300A,0070) DICOM tag
        BeamSequence (np.array): Sequence of treatment beams for current RT Plan as per (300A,00B0) DICOM tag
        PatientSetupSequence (np.array): Sequence of patient setup data for current plan as per (300A,0180) DICOM tag
        ReferencedStructureSetSequence (np.array): The RT Structure Set on which the RT Plan is based as per (300C,0060) DICOM tag
        ReferencedDoseSequence (np.array): Sequence of Dose References.
        ApprovalStatus (str): Approval status at the time the SOP Instance was created as per (300E,0002) DICOM tag.
        ReviewDate (str): Date plan was reviewed
        ReviewTime (str): Time plan was reviewed
        ReviewerName (str): Reviewer name
        SOPInstanceUID (str): SOP Instance UID of the Plan
        BeamUID (str): pyCERR's UID of RTPLAN

    """

    PatientName: str = ""
    PatientID: str = ""
    PatientBirthDate: str = ""
    PatientSex: str = ""
    Manufacturer: str = ""
    ManufacturerModelName: str = ""
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
    """This class defines referenced sequence.
    Attributes:
        ReferencedSOPClassUID (str): Referenced SOP class UID
        ReferencedSOPInstanceUID (str): Referenced SOP instance UID

    """
    ReferencedSOPClassUID: str = ""
    ReferencedSOPInstanceUID: str = ""

@dataclass
class PatientSetupSeq:
    """This class defines patient setup sequence.
    Attributes:
        PatientSetupNumber (int): Identification number of the Patient Setup as per (300A,0182) DICOM tag
        PatientPosition (str): Patient position descriptor relative to the equipment as per (0018,5100) DICOM tag

    """

    PatientSetupNumber: int = 0
    PatientPosition: str = ""

class DoseReferenceSeq:
    """This class defines sequence of dose references.
    Attributes:
        DoseReferenceType (str): Type of dose reference (TARGET, ORGAN_AT_RISK)
        DoseReferenceUID (str): Unique identifier for dose reference
        ReferencedROINumber (str): Unique identifier for associated ROI
        DeliveryMaximumDose (float): Max dose (Gy) that can be delivered to the dose reference
        TargetPrescriptionDose (float): Prescribed dose (Gy) to dose reference if DoseReferenceType is TARGET.

    """
    DoseReferenceType: str = ""
    DoseReferenceUID: str = ""
    ReferencedROINumber: str = ""
    DeliveryMaximumDose: float = float('nan')
    TargetPrescriptionDose: float = float('nan')


@dataclass
class BeamLimitingDevicePositionSeq:
    """This class defines data model for Sequence of beam limiting device (collimator) jaw or leaf (element) positions.
    Attributes:
        RTBeamLimitingDeviceType (str): Type of beam limiting device (collimator) as per (300A,00B8) DICOM tag
        LeafJawPositions (np.array): Positions of beam limiting device (collimator) leaf (element) or jaw pairs (in mm)
         in IEC BEAM LIMITING DEVICE coordinate axis appropriate to RT Beam Limiting Device Type as defined in (300A,011C) DICOM tag

    """
    RTBeamLimitingDeviceType: str = ""
    LeafJawPositions: np.array = field(default_factory=get_empty_np_array)

@dataclass
class ControlPointSequence:
    """This class defines data model for the Sequence of machine configurations describing treatment beam.
    Attributes:
        BeamLimitingDevicePositionSequence (np.array): Sequence of beam limiting device (collimator) jaw or leaf (element)
            positions as per (300A,011A) DICOM tag.
        ControlPointIndex (int): Index of current Control Point, starting at 0 for first Control Point as per (300A,0112) DICOM tag.
        NominalBeamEnergy (float): Nominal Beam Energy at control point (MV/MeV) as per (300A,0114) DICOM tag
        GantryAngle (float): Gantry angle of radiation source, i.e., orientation of IEC GANTRY coordinate system with respect to
            IEC FIXED REFERENCE coordinate system (degrees) as per (300A,011E) DICOM tag.
        GantryRotationDirection (str): Direction of Gantry Rotation when viewing gantry from isocenter,
            for segment following Control Point as per (300A,011F) DICOM tag.
        BeamLimitingDeviceAngle (float): Beam Limiting Device angle, i.e., orientation of IEC BEAM LIMITING DEVICE coordinate
            system with respect to IEC GANTRY coordinate system (degrees) as per (300A,0120) DICOM tag.
        BeamLimitingDeviceRotationDirection (str): Direction of Beam Limiting Device Rotation when viewing beam limiting device
            (collimator) from radiation source, for segment following Control Point as per (300A,0121) DICOM tag
        PatientSupportAngle (float): Patient Support angle, i.e., orientation of IEC PATIENT SUPPORT (turntable) coordinate
            system with respect to IEC FIXED REFERENCE coordinate system (degrees) as per (300A,0122) DICOM tag
        TableTopEccentricAngle (float): Table Top (non-isocentric) angle, i.e., orientation of IEC TABLE TOP ECCENTRIC coordinate
            system with respect to IEC PATIENT SUPPORT coordinate system (degrees) as per (300A,0125) DICOM tag.
        TableTopEccentricRotationDirection (str):
        IsocenterPosition (np.array): Direction of Table Top Eccentric Rotation when viewing table from above,
            for segment following Control Point as per 	(300A,0126) DICOM tag
        SourceToSurfaceDistance (float): Source to Patient Surface (skin) distance (mm) as per (300A,0130) DICOM tag
        CumulativeMetersetWeight (float): Cumulative weight to current control point as per (300A,0134) DICOM tag

    """

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
    """This class defies data model for the sequence of Beams in current Fraction Group contributing to dose
    as per (300C,0004) DICOM tag.
    Attributes:
        ReferencedBeamNumber (int): Uniquely identifies Beam specified by Beam Number as per (300C,0006) DICOM tag
        BeamMeterset (float): Meterset duration over which image is to be acquired, specified in Monitor units (MU)
            as per (3002,0032) DICOM tag
    """
    ReferencedBeamNumber: int = 0
    BeamMeterset: float = 0


@dataclass
class FractionGroupSeq:
    """This class defines data model for Sequence of Fraction Groups in current Fraction Scheme as per (300A,0070) DICOM tag.
    Attributes:
        FractionGroupNumber (int): Identification number of the Fraction Group as per (300A,0071)
        NumberOfFractionsPlanned (int): Total number of treatments (Fractions) prescribed for current Fraction Group as per (300A,0078)
        NumberOfBeams (int): Number of Beams in current Fraction Group as per (300A,0080)
        NumberOfBrachyApplicationSetups (int): Number of Brachy Application Setups in current Fraction Group as per (300A,00A0)
        RadiationType (str): Particle type of Beam as per (300A,00C6)
        RefBeamSeq (np.array): Sequence of Beams in current Fraction Group contributing to dose as per (300C,0004)

    """
    FractionGroupNumber: int = 0
    NumberOfFractionsPlanned: int = 0
    NumberOfBeams: int = 0
    NumberOfBrachyApplicationSetups: int = 0
    RadiationType: str = ""
    RefBeamSeq: np.array = field(default_factory=get_empty_np_array)

@dataclass
class BeamSeq:
    """This class defines data model for Sequence of treatment beams for current RT Plan as per (300A,00B0) DICOM tag.
    Attributes:
        Manufacturer (str): Manufacturer of the equipment to be used for beam delivery as per (0008,0070)
        BeamName (str): primary beam identifier (often referred to as "field identifier") as per (300A,00C2)
        BeamType (str): Motion characteristic of Beam as per (300A,00C4)
        BeamDescription (str): User-defined description for Beam as per (300A,00C3)
        BeamNumber (int): Identification number of the Beam as per (300A,00C0)
        SourceAxisDistance (float): Radiation source to Gantry rotation axis distance of the equipment that is to be
            used for beam delivery (mm) as per (300A,00B4)
        BeamLimitingDevicePositionSeq (np.array): Sequence of beam limiting device (collimator) jaw or leaf (element) sets
            as per (300A,00B6)
        RadiationType (str): Particle type of Beam as per (300A,00C6)
        TreatmentDeliveryType (str): Delivery Type of treatment as per (300A,00CE)
        NumberOfWedges (float): Number of wedges associated with current Beam as per (300A,00D0)
        NumberOfBoli (float): Number of boli associated with current Beam as per (300A,00ED)
        NumberOfCompensators (float): Number of compensators associated with current Beam as per (300A,00E0)
        NumberOfBlocks (float): Number of shielding blocks associated with Beam as per (300A,00F0)
        NumberOfControlPoints (float): Number of control points in Beam as per (300A,0110)
        ControlPointSequence (np.array): Sequence of machine configurations describing treatment beam as per (300A,0111)

    """

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
    """This routine parses a list of DICOM files and imports metadata from RTPLAN modality into a list of
    pyCERR's Beams objects

    Args:
        file_list (List[str]): List of DICOM file paths.

    Returns:
        List[cerr.dataclasses.beams.Beams]: List of pyCERR's Beam objects.

    """

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
            if hasattr(ds,"DoseReferenceSequence"):
                dose_ref_seq_list = np.array([], dtype=DoseReferenceSeq)
                doseRefSeq = DoseReferenceSeq()
                doseRefSeq.DoseReferenceType = ds.DoseReferenceSequence[0].DoseReferenceType
                doseRefSeq.DoseReferenceUID = ds.DoseReferenceSequence[0].DoseReferenceUID
                # Optional tags
                optTags = ['ReferencedROINumber','DeliveryMaximumDose','TargetPrescriptionDose']
                for tag in optTags:
                    if hasattr(ds.DoseReferenceSequence[0],tag):
                        setattr(doseRefSeq, tag, getattr(ds.DoseReferenceSequence[0], tag))#Type 1C
                dose_ref_seq_list = np.append(dose_ref_seq_list, doseRefSeq)
                beams_meta.DoseReferenceSequence = dose_ref_seq_list
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
