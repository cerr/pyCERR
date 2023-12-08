"""
This module defines the container class PlanC and methods to import metadata from DICOM and NifTI formats.
"""

from dataclasses import dataclass, field
from typing import List
from cerr.dataclasses import scan as scn
import cerr.plan_container as pc
from cerr.dataclasses import structure as structr
from cerr.dataclasses import dose as rtds
from cerr.dataclasses import beams as bms
import pickle
import pandas as pd
import numpy as np
from cerr.contour import rasterseg as rs
import json
#import sys
#import scipy.io as sio
#import SimplaITK as sitk

def get_empty_list():
    return []

@dataclass
class PlanC:
    scan: List[scn.Scan] = field(default_factory=get_empty_list) #scan.Scan()
    structure: List[structr.Structure] = field(default_factory=get_empty_list) #structure.Structure()
    dose: List[rtds.Dose] = field(default_factory=get_empty_list) #dose.Dose()
    beams: List[bms.Beams] = field(default_factory=get_empty_list) #beams.Beams()

    def addScan(self, new_scan) -> scan:
        self.scan.append(new_scan)

    class json_serialize(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, scn.Scan):
                return {'scan':obj.scanUID}
            elif isinstance(obj, rtds.Dose):
                return {'dose': obj.doseUID}
            elif isinstance(obj, structr.Structure):
                return {'structure':obj.strUID}
            return "" #json.JSONEncoder.default(self, obj)


def load_dcm_dir(dcm_dir, initplanC = ''):
    """
    This routine imports metadata from DICOM directory and sub-directories into an instance of PlanC.
    INPUTS -
        dcm_dir - absolute path to directory containing dicom files
        initplanC - An instance of PlanC to add the metadata. If not specified, metadata is added to an empty PlanC instance
    OUTPUT - An instance of PlanC
    """
    # pc.PlanC is the container to hold various dicom objects
    # Parse dcm_dir an extract a map of CT, RTSTRUCT, RTDOSE etc files to pass to populate_planC_field routine
    df_img = parse_dcm_dir(dcm_dir)
    #pt_groups = df_img.groupby(by=["PatientName","PatientID","Modality"])
    # Ignore fileName column from grouping
    if not isinstance(initplanC, pc.PlanC):
        planC = pc.PlanC()
    else:
        planC = initplanC
    pt_groups = df_img.groupby(by=df_img.columns.to_list()[:-1],dropna=False)
    pt_groups.size()
    for group_name,group_content in pt_groups:
        print(group_name)
        d = group_content.to_dict()
        files = group_content.iloc[:,-1]
        modality = group_content.iloc[0,4]
        if np.any(modality == np.array(["CT","PT", "MR"])):
            # populate scan attributes
            scan_meta = populate_planC_field('scan', files)
            planC.scan.extend(scan_meta)
        elif modality == "RTSTRUCT":
            # populate structure attributes
            struct_meta = populate_planC_field('structure', files)
            planC.structure.extend(struct_meta)
        elif modality == "RTPLAN":
            # populate beams attributes
            beams_meta = populate_planC_field('beams', files)
            planC.beams.extend(beams_meta)
        elif modality == "RTDOSE":
            # populate dose attributes
            dose_meta = populate_planC_field('dose', files)
            planC.dose.extend(dose_meta)
        else:
            print(d["Modality"][0]+ " not supported")

    # Convert structure coordinates to CERR's virtual coordinates
    for str_num,struct in enumerate(planC.structure):
        planC.structure[str_num].convertDcmToCerrVirtualCoords(planC)
        #planC.structure[str_num].generate_rastersegs(planC) # this calls polyFill
        planC.structure[str_num].rasterSegments = rs.generate_rastersegs(planC.structure[str_num],planC)

    # Convert dose coordinates to CERR's virtual coordinates
    for dose_num,dose in enumerate(planC.dose):
        planC.dose[dose_num].convertDcmToCerrVirtualCoords(planC)

    return planC

    # Save planC to file
    #sio.savemat(save_file, {'planC': planC})
    #with open(save_file, 'wb') as pickle_file:
    #    pickle.dump(planC, pickle_file)

def populate_planC_field(field_name, file_list):
    if field_name == "scan":
        scan_meta = []
        scan_meta.append(scn.load_sorted_scan_info(file_list))
        scan_meta[0].convertDcmToCerrVirtualCoords()
        scan_meta[0].convertDcmToRealWorldUnits()
        if scan_meta[0].scanInfo[0].imageType == "PT SCAN":
            scan_meta[0].convert_to_suv("BW")
        return scan_meta

    elif field_name == "structure":
        struct_meta = structr.load_structure(file_list)
        return struct_meta

    elif field_name == "dose":
        dose_meta = rtds.load_dose(file_list)
        return dose_meta

    elif field_name == "beams":
        beams_meta = bms.load_beams(file_list)
        return beams_meta

def load_planC_from_pkl(file_name=""):
    file_name = r"C:\Users\aptea\PycharmProjects\pycerr\src\pycerr\tcga-ba-4074.mat"
    # Load planC from file
    #planC = sio.loadmat(file_name)
    with open(file_name, 'rb') as pickle_file:
        planC = pickle.load(pickle_file)
    return planC

def save_scan_to_nii(scan_num, nii_file_name, planC):
    pass

def load_nii_scan(nii_file_name, planC = pc.PlanC):
    pass

def load_nii_structure(nii_file_name, assocScanNum, planC = pc.PlanC, labels_dict = {}):
    struct_meta = structr.import_nii(nii_file_name,assocScanNum,planC,labels_dict)
    numOrigStructs = len(planC.structure)
    planC.structure.extend(struct_meta)
    numStructs = len(planC.structure)
    # Convert structure coordinates to CERR's virtual coordinates
    for str_num in range(numOrigStructs,numStructs):
        planC.structure[str_num].convertDcmToCerrVirtualCoords(planC)
        planC.structure[str_num].rasterSegments = rs.generate_rastersegs(planC.structure[str_num],planC)

    return planC


def load_nii_dose(nii_file_name, planC = pc.PlanC):
    pass

def parse_dcm_dir(dcm_dir):
    from pydicom.misc import is_dicom
    from pydicom import dcmread
    import os
    import numpy as np
    from pydicom.tag import Tag

    # Patient Name, Ptient ID, StudyInstanceUID, SeriesInstanceUID,
    # Modality, b-value * 3, temporalIndex, trigger, numSlices
    tag_list = [Tag(0x0010,0x0010), Tag(0x0010,0x0010), Tag(0x0020,0x000D), Tag(0x0020,0x000E),
        Tag(0x0008, 0x0060), Tag(0x0043,0x1039), Tag(0x0018,0x9087), Tag(0x0019,0x100C),
        Tag(0x0020,0x0100), Tag(0x0018,0x1060), Tag(0x0021,0x104F)]
    tag_heading = ["PatientName","PatientID","StudyInstanceUID","SeriesInstanceUID",
                   "Modality","bValue1","bValue2","bValue3","TemporalPosition",
                   "TriggerTime","NumSlices","FilePath"]
    img_meta = []
    for root, _, files in os.walk(dcm_dir):
        for i,file in enumerate(files):
            file_path = os.path.join(root, file)
            if is_dicom(file_path):
                ds = dcmread(file_path,specific_tags=tag_list)
            tag_vals = [ds[t].value if t in ds else "" for t in tag_list]
            tag_vals.append(ds.filename)
            img_meta.append(tag_vals)
    df = pd.DataFrame(img_meta,columns=tag_heading)
    return df

