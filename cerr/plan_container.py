"""
This module defines the container class PlanC and methods to import metadata from DICOM and NifTI formats.
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List

import SimpleITK as sitk
import numpy as np
import pandas as pd
from pydicom.uid import generate_uid
from skimage import measure

import cerr.dataclasses.scan_info as scn_info
import cerr.plan_container as pc
from cerr.contour import rasterseg as rs
from cerr.dataclasses import beams as bms
from cerr.dataclasses import dose as rtds
from cerr.dataclasses import scan as scn
from cerr.dataclasses import structure as structr
from cerr.dataclasses.structure import Contour
from cerr.utils import uid

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
        if modality in ["CT","PT", "MR"]:
            # populate scan attributes
            scan_meta = populate_planC_field('scan', files)
            planC.scan.extend(scan_meta)
        elif modality in ["RTSTRUCT", "SEG"]:
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

def load_nii_scan(nii_file_name, imageType = "CT SCAN", initplanC = ''):
    if not isinstance(initplanC, pc.PlanC):
        planC = pc.PlanC()
    else:
        planC = initplanC
    reader = sitk.ImageFileReader()
    reader.SetFileName(nii_file_name)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    image = reader.Execute()
    # Get numpy array for scan
    scanArray3M = sitk.GetArrayFromImage(image)
    scanArray3M = np.moveaxis(scanArray3M,[0,1,2],[2,0,1])
    #Construct position matrix from ITK Image
    pos1V = np.asarray(image.TransformIndexToPhysicalPoint((0,0,0))) / 10
    pos2V = np.asarray(image.TransformIndexToPhysicalPoint((0,0,1))) / 10
    deltaPosV = pos2V - pos1V
    pixelSpacing = np.asarray(image.GetSpacing()[:2]) / 10
    img_ori = np.array(image.GetDirection())
    dir_cosine_mat = img_ori.reshape(3, 3,order="C")
    pixelSiz = image.GetSpacing()
    dcmImgOri = dir_cosine_mat.reshape(9,order='F')[:6]
    original_orient_str = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_ori)
    slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
                   - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]



    # # Transformation for DICOM Image to DICOM physical coordinates
    # # Pt coordinate to DICOM image coordinate mapping
    # # Based on ref: https://nipy.org/nibabel/dicom/dicom_orientation.html
    # position_matrix = np.hstack((np.matmul(dir_cosine_mat[:,:2],np.diag(pixelSpacing)),
    #                             np.array([[deltaPosV[0], pos1V[0]], [deltaPosV[1], pos1V[1]], [deltaPosV[2], pos1V[2]]])))
    #
    # position_matrix = np.vstack((position_matrix, np.array([0, 0, 0, 1])))

    org_root = '1.3.6.1.4.1.9590.100.1.2.' # to assign
    forUID = generate_uid(prefix=org_root)
    studyInstanceUID = generate_uid(prefix=org_root)
    seriesInstanceUID = generate_uid(prefix=org_root)
    scan = scn.Scan()
    siz = scanArray3M.shape
    scan_info = np.empty(siz[2],dtype=scn_info.ScanInfo) #scn_info.ScanInfo()
    #pixelSiz = image.GetSpacing()
    currentDate = str(date.today())
    now = datetime.now()
    currentTime = now.strftime("%H:%M:%S")
    count = 0
    for slc in range(siz[2]):
        s_info = scn_info.ScanInfo()
        s_info.frameOfReferenceUID = forUID
        s_info.imagePositionPatient = np.asarray(image.TransformIndexToPhysicalPoint((0,0,slc)))
        s_info.imageOrientationPatient = dcmImgOri
        s_info.grid1Units = pixelSpacing[1]
        s_info.grid2Units = pixelSpacing[0]
        s_info.sizeOfDimension1 = siz[0]
        s_info.sizeOfDimension2 = siz[1]
        s_info.zValue = - np.sum(slice_normal * s_info.imagePositionPatient) / 10
        s_info.imageType = imageType
        s_info.seriesInstanceUID = seriesInstanceUID
        s_info.studyInstanceUID = studyInstanceUID
        s_info.studyDescription = ''
        s_info.seriesDate = currentDate
        s_info.seriesTime = currentTime
        s_info.studyNumberOfOrigin = ''
        #scan_info.append(s_info)
        scan_info[count] = s_info
        count += 1
    original_orient_str = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_ori)
    sort_index = [i for i,x in sorted(enumerate(scan_info),key=scn.get_slice_position, reverse=False)]
    scan_info = scan_info[sort_index]
    scanArray3M = scanArray3M[:,:,sort_index]
    scan_info = scn_info.deduce_voxel_thickness(scan_info)
    scan.scanInfo = scan_info
    scan.scanArray = scanArray3M
    scan.scanUID = "CT." + seriesInstanceUID
    scan.convertDcmToCerrVirtualCoords()
    scan.convertDcmToRealWorldUnits()
    planC.scan.append(scan)
    return planC


def load_nii_structure(nii_file_name, assocScanNum, planC, labels_dict = {}):
    struct_meta = structr.import_nii(nii_file_name,assocScanNum,planC,labels_dict)
    numOrigStructs = len(planC.structure)
    planC.structure.extend(struct_meta)
    numStructs = len(planC.structure)
    # Convert structure coordinates to CERR's virtual coordinates
    for str_num in range(numOrigStructs,numStructs):
        planC.structure[str_num].convertDcmToCerrVirtualCoords(planC)
        planC.structure[str_num].rasterSegments = rs.generate_rastersegs(planC.structure[str_num],planC)

    return planC


def load_nii_dose(nii_file_name, planC):
    pass

def import_scan_array(scan3M, xV, yV, zV, modality, assocScanNum, planC):
    org_root = '1.3.6.1.4.1.9590.100.1.2.' # to create seriesInstanceUID
    seriesInstanceUID = generate_uid(prefix=org_root)
    scan = scn.Scan()
    scan_info = [] #scn_info.ScanInfo()
    siz = scan3M.shape
    # Get DICOM ImagePositionPatient i.e. x,y,z of 1at voxel
    forUID = planC.scan[assocScanNum].scanInfo[0].frameOfReferenceUID
    dcmImgOri = planC.scan[assocScanNum].scanInfo[0].imageOrientationPatient
    studyInstanceUID = planC.scan[assocScanNum].scanInfo[0].studyInstanceUID
    studyDescription = planC.scan[assocScanNum].scanInfo[0].studyDescription
    studyDate = planC.scan[assocScanNum].scanInfo[0].studyDate
    studyTime = planC.scan[assocScanNum].scanInfo[0].studyTime
    studyNumberOfOrigin = planC.scan[assocScanNum].scanInfo[0].studyNumberOfOrigin
    currentDate = str(date.today())
    now = datetime.now()
    currentTime = now.strftime("%H:%M:%S")
    dx = xV[1] - xV[0]
    dy = yV[0] - yV[1]
    for slc in range(siz[2]):
        cerrImgPatPos = [xV[0], yV[0], zV[slc], 1]
        dcmImgPos = np.matmul(planC.scan[assocScanNum].cerrToDcmTransM, cerrImgPatPos)[:3]
        s_info = scn_info.ScanInfo()
        s_info.frameOfReferenceUID = forUID
        s_info.imagePositionPatient = dcmImgPos
        s_info.imageOrientationPatient = dcmImgOri
        s_info.grid1Units = dy
        s_info.grid2Units = dx
        s_info.zValue = zV[slc]
        s_info.sizeOfDimension1 = siz[0]
        s_info.sizeOfDimension2 = siz[1]
        s_info.imageType = modality
        s_info.seriesInstanceUID = seriesInstanceUID
        s_info.studyInstanceUID = studyInstanceUID
        s_info.studyDescription = studyDescription
        s_info.studyDate = studyDate
        s_info.studyTime = studyTime
        s_info.seriesDate = currentDate
        s_info.seriesTime = currentTime
        s_info.studyNumberOfOrigin = studyNumberOfOrigin
        scan_info.append(s_info)
    scan_info = scn_info.deduce_voxel_thickness(scan_info)
    scan.scanInfo = scan_info
    scan.scanArray = scan3M
    scan.scanUID = "CT." + seriesInstanceUID
    scan.scanType = modality
    scan.convertDcmToCerrVirtualCoords()
    #scan.convertDcmToRealWorldUnits()
    planC.scan.append(scan)
    return planC

def import_structure_mask(mask3M, assocScanNum, structName, structNum, planC):
    # Pad mask to account for boundary edges
    paddedMask3M = mask3M.astype(int)
    paddedMask3M = np.pad(paddedMask3M, ((1,1),(1,1),(0,0)), 'constant', constant_values = 0)
    dt = datetime.now()
    if isinstance(structNum,int):
        struct_meta = planC.structure[structNum]
    else:
        struct_meta = structr.Structure()
        structNum = len(planC.structure)
        struct_meta.structureColor = structr.getColorForStructNum(structNum)
        struct_meta.strUID = uid.createUID("structure")
        struct_meta.structSetSopInstanceUID = generate_uid()
        struct_meta.assocScanUID = planC.scan[assocScanNum].scanUID
    struct_meta.structureFileFormat = "NPARRAY"
    struct_meta.structureName = structName
    struct_meta.dateWritten = dt.strftime("%Y%m%d")
    struct_meta.roiNumber = ""
    contour_list = np.empty((0),Contour)
    numSlcs = len(planC.scan[assocScanNum].scanInfo)
    dim = paddedMask3M.shape
    for slc in range(dim[2]):
        if not np.any(paddedMask3M[:,:,slc]):
            continue
        contours = measure.find_contours(paddedMask3M[:,:,slc] == 1, 0.5)
        if scn.flipSliceOrderFlag(planC.scan[assocScanNum]):
            slcNum = numSlcs - slc - 1
        else:
            slcNum = slc
        # _,_,niiZ = image.TransformIndexToPhysicalPoint((0,0,slc))
        # ind = np.where((zDicomV - niiZ)**2 < slcMatchTol)
        # if len(ind[0]) == 1:
        #     ind = ind[0][0]
        # else:
        #     raise Exception('No matching slices found.')
        sopClassUID = planC.scan[assocScanNum].scanInfo[slc].sopClassUID
        sopInstanceUID = planC.scan[assocScanNum].scanInfo[slc].sopInstanceUID

        num_contours = len(contours)
        for iContour, contour in enumerate(contours):
            contObj = Contour()
            segment = np.empty((contour.shape[0],3))
            colV = contour[:, 1] - 1 # - 1 to account for padding
            rowV = contour[:, 0] - 1 # - 1 to account for padding
            slcV = np.ones_like(rowV) * slcNum
            ptsM = np.asarray((colV,rowV,slcV))
            ptsM = np.vstack((ptsM, np.ones((1, ptsM.shape[1]))))
            ptsM = np.matmul(planC.scan[assocScanNum].Image2PhysicalTransM, ptsM)[:3,:].T
            contObj.segments = ptsM
            contObj.referencedSopInstanceUID = sopInstanceUID
            contObj.referencedSopClassUID = sopClassUID
            #contour_list.append(contObj)
            contour_list = np.append(contour_list,contObj)
        struct_meta.contour = contour_list
#     struct_meta = structr.load_nii_structure(nii_file_name,assocScanNum,planC,labels_dict)
    numOrigStructs = len(planC.structure)
    planC.structure.append(struct_meta)
    #str_num = len(planC.structure) - 1
    planC.structure[structNum].convertDcmToCerrVirtualCoords(planC)
    planC.structure[structNum].rasterSegments = rs.generate_rastersegs(planC.structure[structNum],planC)
    return planC


def parse_dcm_dir(dcm_dir):
    from pydicom.misc import is_dicom
    from pydicom import dcmread
    import os
    from pydicom.tag import Tag
    from pydicom import dataelem

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
                #tag_vals = [ds[t].value if t in ds else "" for t in tag_list]
                tag_vals = []
                for t in tag_list:
                    if t in ds:
                        val = ds[t].value
                    if isinstance(val, dataelem.MultiValue):
                        val = tuple(val)
                    tag_vals.append(val)
                tag_vals.append(ds.filename)
                img_meta.append(tag_vals)
    df = pd.DataFrame(img_meta,columns=tag_heading)
    return df

