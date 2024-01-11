from dataclasses import dataclass, field
from typing import List
import numpy as np
import os
from pydicom import dcmread
import cerr.dataclasses.scan as scn
from cerr.utils import uid
import cerr.contour.rasterseg as rs
import nibabel as nib
import SimpleITK as sitk
from skimage import measure
from datetime import datetime
from pydicom.uid import generate_uid
import json
from cerr.radiomics.preprocess import imgResample3D
import cerr.plan_container as pc

def get_empty_list():
    return []
def get_empty_np_array():
    return np.empty((0,0,0))

@dataclass
class Structure:
    roiNumber: int = 0
    patientName: str = ""
    structureName: str = ""
    ROIInterpretedType: str = ""
    structureFormat: str = ""
    numberOfScans: int = 0
    maximumNumberScans: int = 0
    maximumPointsPerSegment: int = 0
    maximumSegmentsPerScan: int = 0
    structureEdition: str = ""
    writer: str = ""
    dateWritten: str = ""
    structureColor: str = ""
    structureDescription: str = ""
    roiGenerationAlgorithm: str = ""
    roiGenerationDescription: str = ""
    studyNumberOfOrigin: str = ""
    contour: List = field(default_factory=get_empty_np_array)
    rasterSegments: np.ndarray = field(default_factory=get_empty_np_array)
    DSHPoints: np.ndarray = field(default_factory=get_empty_np_array)
    orientationOfStructure: str = ""
    transferProtocol: str = ""
    visible: bool = True
    associatedScan: str = ""
    strUID: str = ""
    assocScanUID: str = ""
    structSetSopInstanceUID: str = ""
    rasterized: bool = False
    referencedFrameOfReferenceUID: str = ""
    referencedSeriesUID: str = ""
    structureFileFormat: str = ""

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    class json_serialize(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Structure):
                return {'structure':obj.strUID}
            return "" #json.JSONEncoder.default(self, obj)


    def save_nii(self,niiFileName,planC):
        str_num = getStructNumFromUID(self.strUID, planC)
        scan_num = scn.getScanNumFromUID(self.assocScanUID,planC)
        affine3M = planC.scan[scan_num].get_nii_affine()
        mask3M = rs.getStrMask(str_num,planC)
        mask3M = np.moveaxis(mask3M,[0,1],[1,0])
        # https://neurostars.org/t/direction-orientation-matrix-dicom-vs-nifti/14382/2
        # dcmImgOri = planC.scan[scan_num].scanInfo[0].imageOrientationPatient
        # slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
        #        - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]
        # zDiff = np.matmul(slice_normal, planC.scan[scan_num].scanInfo[1].imagePositionPatient) \
        #         - np.matmul(slice_normal, planC.scan[scan_num].scanInfo[0].imagePositionPatient)
        # ippDiffV = planC.scan[scan_num].scanInfo[1].imagePositionPatient - planC.scan[scan_num].scanInfo[0].imagePositionPatient
        if scn.flipSliceOrderFlag(planC.scan[scan_num]): # np.all(np.sign(zDiff) < 0):
            #if not planC.scan[scan_num].isCerrSliceOrderMatchDcm():
            mask3M = np.flip(mask3M,axis=2)
        str_img = nib.Nifti1Image(mask3M.astype('uint16'), affine3M)
        nib.save(str_img, niiFileName)

    def convertDcmToCerrVirtualCoords(self,planC):
        assocScanUID = self.assocScanUID
        scan_num = scn.getScanNumFromUID(assocScanUID,planC)
        im_to_virtual_phys_transM = planC.scan[scan_num].Image2VirtualPhysicalTransM
        im_to_phys_transM = planC.scan[scan_num].Image2PhysicalTransM
        position_matrix_inv = np.linalg.inv(im_to_phys_transM)
        _,_,zValsV = planC.scan[scan_num].getScanXYZVals()
        isRTSTRUCT = self.structureFileFormat in ["RTSTRUCT", "NPARRAY"]
        scan_sop_inst_list = np.array([scan_info.sopInstanceUID for scan_info in planC.scan[scan_num].scanInfo])
        numSlcs = len(planC.scan[scan_num].scanInfo)
        for ctr_num,contour in enumerate(self.contour):
            if np.any(contour.segments):
                if isRTSTRUCT:
                    tempa = np.hstack((contour.segments, np.ones((contour.segments.shape[0], 1))))
                    tempa = np.matmul(position_matrix_inv, tempa.T)
                    # Round the z-coordinates (slice) of 'tempa'
                    tempa[2, :] = np.round(tempa[2, :])
                else:
                    slcNum = np.argwhere(scan_sop_inst_list == contour.referencedSopInstanceUID)
                    slcNum = slcNum[0,0]
                    if scn.flipSliceOrderFlag(planC.scan[scan_num]):
                        slcNum = numSlcs - slcNum - 1
                    #tempa = np.hstack((contour.segments, np.ones((contour.segments.shape[0], 1))))
                    #tempa[:, 2] = slcNum
                    #tempa = np.matmul(im_to_phys_transM, tempa.T)
                    tempa = contour.segments
                    tempa[:,2] = slcNum
                    tempa = np.hstack((tempa, np.ones((contour.segments.shape[0], 1)))).T

                tempb = np.matmul(im_to_virtual_phys_transM, tempa)
                tempb = tempb[:3,:].T
                self.contour[ctr_num].segments = tempb

        # Sort contours per CT slices
        dcm_contour_list = self.contour
        contr_sop_inst_list = np.array([c.referencedSopInstanceUID for c in dcm_contour_list])
        scan_info = planC.scan[scan_num].scanInfo
        #scan_sop_inst_list = np.array([s.sopInstanceUID for s in scan_info])
        num_slices = len(planC.scan[scan_num].scanInfo)
        contour_list = np.empty(num_slices,Contour)

        for slc_num in range(num_slices):
            contour_list[slc_num] = []
            scan_sop_inst = planC.scan[scan_num].scanInfo[slc_num].sopInstanceUID
            if scan_sop_inst != "":
                seg_matches = np.where(contr_sop_inst_list == scan_sop_inst)
            else:
                seg_matches = [[]]
            if scan_sop_inst == "" or len(seg_matches[0]) == 0:
                matches = []
                for iCtr,ctr in enumerate(dcm_contour_list):
                    if np.all((ctr.segments[:,2] - zValsV[slc_num])**2 < 1e-5):
                        matches.append(iCtr)
                seg_matches = (np.asarray(matches,dtype='int64'),)
            segments = []
            for seg_num in seg_matches[0]: # 0 for 1-d array. we only care about the 1st dim
                segment = Segment()
                segment.points = dcm_contour_list[seg_num].segments
                segments.append(segment)
            if np.any(segments):
                contour = Contour()
                contour.segments = segments
                contour.referencedSopInstanceUID = dcm_contour_list[seg_num].referencedSopInstanceUID
                contour.referencedSopClassUID = dcm_contour_list[seg_num].referencedSopClassUID
                contour_list[slc_num] = contour
        self.contour = contour_list
        return self

    def getStructureAssociatedScan(self, planC):
        """
        Returns associated scan index for structure object based on the scan UID associated with
        the structure.
        """

        # Preallocate memory
        scanUID = [None] * len(planC.scan)

        # Get all associated scan UID from structures
        allAssocScanUID = [struct['assocScanUID'] for struct in planC.structure]

        # Get the scan UID from the scan field under planC
        scanUID = [scan['scanUID'] for scan in planC.scan]

        # Get associated scan UID's
        assocScanUID = self['assocScanUID']

        # Match all the UID to check which scan the structure belongs to.
        assocScansV = []
        for uid in assocScanUID:
            ind = next((i for i, x in enumerate(scanUID) if x == uid), 0)
            assocScansV.append(ind)

        return assocScansV

@dataclass
class Contour:
    #sopInstanceUID: str = ""
    #sopClassUID: str = ""
    referencedSopInstanceUID: str = ""
    referencedSopClassUID: str = ""
    segments: np.array = field(default_factory=get_empty_np_array)

@dataclass
class Segment:
    points: np.ndarray = field(default_factory=get_empty_np_array)

def parse_contours(contour_seq):
    num_contours = len(contour_seq)
    contour_list = np.empty(num_contours,Contour)
    for ctr_num,contr in enumerate(contour_seq):
        sop_instance_uid = ""
        sop_class_uid = ""
        if  hasattr(contr,"ContourImageSequence"):
            ref_seq = contr.ContourImageSequence #
            sop_instance_uid = ref_seq[0].ReferencedSOPInstanceUID
            sop_class_uid = ref_seq[0].ReferencedSOPClassUID
        geometry_type = contr.ContourGeometricType
        num_pts = contr.NumberOfContourPoints
        contour_data = contr.ContourData
        contour_data = np.transpose(np.reshape(contour_data,(3,num_pts),order="F"))
        contour_data = contour_data / 10
        if not np.array_equal(contour_data[0,:],contour_data[-1,:]):
            contour_data = np.vstack((contour_data,contour_data[0,:]))
        contour = Contour()
        contour.referencedSopClassUID = sop_class_uid
        contour.referencedSopInstanceUID = sop_instance_uid
        contour.segments = contour_data
        contour_list[ctr_num] = contour
    return contour_list

def parse_structure_fields(roi_contour,str_roi) -> Structure:
    struct = Structure()
    struct.roiNumber = str_roi.ROINumber #getattr(ds,'ROINumber',"")
    struct.structureName = str_roi.ROIName
    struct.numberOfScans = len(len(roi_contour.ContourSequence)) # number of scan slices
    #structureColor
    struct.roiGenerationAlgorithm = str_roi.ROIGenerationAlgorithm
    if ("3006","0038") in str_roi:
        struct.roiGenerationDescription  = str_roi["3006","0038"].value
    struct.contour = parse_contours(roi_contour.ContourSequence)
    return struct

def load_structure(file_list):
    struct_list = []
    for file in file_list:
        ds = dcmread(file)
        if ds.Modality == "RTSTRUCT":
            roi_contour_seq = ds.ROIContourSequence
            str_roi_seq = ds.StructureSetROISequence
            num_structs = len(roi_contour_seq)
            for str_num, (roi_contour,str_roi) in enumerate(zip(roi_contour_seq,str_roi_seq)):
                struct_meta = Structure() #parse_structure_fields(roi_contour_seq,str_roi_seq)
                struct_meta.patientName = ds.PatientName
                struct_meta.writer = ds.Manufacturer
                struct_meta.dateWritten = ds.StructureSetDate
                if hasattr(ds,"SeriesDescription"): struct_meta.structureDescription = ds.SeriesDescription
                struct_meta.roiNumber = str_roi.ROINumber #getattr(ds,'ROINumber',"")
                struct_meta.structureName = str_roi.ROIName
                if not hasattr(roi_contour,"ContourSequence"):
                    continue
                struct_meta.numberOfScans = len(roi_contour.ContourSequence) # number of scan slices
                struct_meta.strUID = uid.createUID("structure")
                struct_meta.structSetSopInstanceUID =  ds.SOPInstanceUID
                struct_meta.structureFileFormat = "RTSTRUCT"
                #structureColor
                struct_meta.roiGenerationAlgorithm = str_roi.ROIGenerationAlgorithm
                if ("3006","0038") in str_roi:
                    struct_meta.roiGenerationDescription  = str_roi["3006","0038"].value
                struct_meta.contour = parse_contours(roi_contour.ContourSequence)
                ref_FOR_uid = str_roi.ReferencedFrameOfReferenceUID
                ref_FOR_seq = ds.ReferencedFrameOfReferenceSequence
                for ref_FOR in ref_FOR_seq:
                    if ref_FOR.FrameOfReferenceUID == ref_FOR_uid:
                        struct_meta.assocScanUID = "CT." + ref_FOR.RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
                        # Associated scan found. Break out of for loop
                        break
                struct_list.append(struct_meta)
        elif ds.Modality == "SEG":
            # Read segmentation mask
            mask3M = ds.pixel_array
            mask3M = np.transpose(mask3M,[1,2,0])
            numStructs = len(ds.SegmentSequence)
            if hasattr(ds.PerFrameFunctionalGroupsSequence[0], 'SegmentIdentificationSequence'):
                refSegNums = np.array([segId.SegmentIdentificationSequence[0].ReferencedSegmentNumber for segId in ds.PerFrameFunctionalGroupsSequence])
            else:
                numFrames = len(ds.SharedFunctionalGroupsSequence[0].DerivationImageSequence[0].SourceImageSequence)
                refSegNums = np.array([segId.ReferencedSegmentNumber for segId in ds.SharedFunctionalGroupsSequence[0].SegmentIdentificationSequence] * numFrames)
            for strNum in range(numStructs):
                struct_meta = Structure() #parse_structure_fields(roi_contour_seq,str_roi_seq)
                struct_meta.structureFileFormat = "SEG"
                struct_meta.patientName = ds.PatientName
                struct_meta.writer = ds.Manufacturer
                struct_meta.dateWritten = ds.SeriesDate
                if hasattr(ds,"SeriesDescription"): struct_meta.structureDescription = ds.SeriesDescription
                struct_meta.roiNumber = ds.SegmentSequence[strNum].SegmentNumber
                struct_meta.structureName = ds.SegmentSequence[strNum].SegmentLabel
                struct_meta.numberOfScans = len(ds.PerFrameFunctionalGroupsSequence) # number of scan slices
                struct_meta.strUID = uid.createUID("structure")
                struct_meta.structSetSopInstanceUID =  ds.SOPInstanceUID
                if hasattr(ds.SegmentSequence[strNum], 'SegmentAlgorithmType'):
                    struct_meta.roiGenerationAlgorithm = ds.SegmentSequence[strNum].SegmentAlgorithmType
                if hasattr(ds.SegmentSequence[strNum], 'SegmentAlgorithmName'):
                    struct_meta.roiGenerationDescription  = ds.SegmentSequence[strNum].SegmentAlgorithmName
                if hasattr(ds.SegmentSequence[strNum], 'RecommendedDisplayCIELabValue'):
                    struct_meta.structureColor = ds.SegmentSequence[strNum].RecommendedDisplayCIELabValue
                ref_FOR_uid = ds.FrameOfReferenceUID
                refSeriesInstanceUID = ds.ReferencedSeriesSequence[0].SeriesInstanceUID
                struct_meta.assocScanUID = "CT." + refSeriesInstanceUID
                # Segment number
                perFrameSegNum = np.argwhere(refSegNums == ds.SegmentSequence[strNum].SegmentNumber)
                perFrameSegNum = perFrameSegNum[:,0]
                contour_list = np.empty((0),Contour)
                for frameNum in perFrameSegNum:
                    if hasattr(ds.PerFrameFunctionalGroupsSequence[0], 'DerivationImageSequence'):
                        sopInstanceUID = ds.PerFrameFunctionalGroupsSequence[frameNum].DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
                        sopClassUID = ds.PerFrameFunctionalGroupsSequence[frameNum].DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPClassUID
                    else:
                        sopInstanceUID = ds.SharedFunctionalGroupsSequence[0].DerivationImageSequence[0].SourceImageSequence[frameNum].ReferencedSOPInstanceUID
                        sopClassUID = ds.SharedFunctionalGroupsSequence[0].DerivationImageSequence[0].SourceImageSequence[frameNum].ReferencedSOPClassUID

                    mask2D = mask3M[:,:,frameNum]
                    if not np.any(mask2D):
                        continue

                    contours = measure.find_contours(mask2D == 1, 0.5)
                    num_contours = len(contours)
                    for iContour, contour in enumerate(contours):
                        contObj = Contour()
                        segment = np.empty((contour.shape[0],3))
                        colV = contour[:, 1]
                        rowV = contour[:, 0]
                        slcV = np.ones_like(rowV) # just a paceholder. This is assigned later on based on SOPInstance UID match
                        ptsM = np.asarray((colV,rowV,slcV))
                        ptsM = np.vstack((ptsM, np.ones((1, ptsM.shape[1]))))
                        #ptsM = np.matmul(planC.scan[assocScanNum].Image2PhysicalTransM, ptsM)[:3,:].T
                        contObj.segments = ptsM[:3,:].T
                        contObj.referencedSopInstanceUID = sopInstanceUID
                        contObj.referencedSopClassUID = sopClassUID
                        #contour_list.append(contObj)
                        contour_list = np.append(contour_list,contObj)
                struct_meta.contour = contour_list

                struct_list.append(struct_meta)

    return struct_list

def import_nii(file_list, assocScanNum, planC, labels_dict = {}):
    if isinstance(file_list,str) and os.path.exists(file_list):
        file_list = [file_list]
    struct_list = []
    for file in file_list:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        image = reader.Execute()
        img_ori = image.GetDirection()
        original_orient_str = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_ori)
        #image = sitk.DICOMOrient(image,"LPS")
        niiSegArray3M = sitk.GetArrayFromImage(image)
        niiSegArray3M = np.moveaxis(niiSegArray3M,[0,1,2],[2,0,1])
        siz = niiSegArray3M.shape
        # No need to check whether slice direction is reversed as we match based on dicom z values later on
        origin = list(image.GetOrigin())
        orient_nii = np.asarray(image.GetDirection())
        orient_nii.reshape(3, 3,order="C")
        orient_nii = np.reshape(orient_nii, (3,3), order = "C")
        dcmImgOrient = orient_nii.reshape(9,order='F')[:6]
        scanOrientV = planC.scan[assocScanNum].scanInfo[0].imageOrientationPatient
        if np.max((dcmImgOrient - scanOrientV)**2) > 1e-5:
            raise Exception("nii file orientation does not match the associated scan")
        slice_normal = dcmImgOrient[[1,2,0]] * dcmImgOrient[[5,3,4]] \
               - dcmImgOrient[[2,0,1]] * dcmImgOrient[[4,5,3]]
        #s_info.zValue = - np.sum(slice_normal * s_info.imagePositionPatient) / 10
        structZvalsV = np.empty((siz[2],1))
        for slc in range(siz[2]):
            imagePositionPatient = np.asarray(image.TransformIndexToPhysicalPoint((0,0,slc)))
            structZvalsV[slc] = - np.sum(slice_normal * imagePositionPatient) / 10


        dim = np.asarray(image.GetSize())
        res = np.asarray(image.GetSpacing())
        xV, yV, zV = planC.scan[assocScanNum].getScanXYZVals()
        cerrToDcmTransM = planC.scan[assocScanNum].cerrToDcmTransM
        dcmIm2PhysTransM = planC.scan[assocScanNum].Image2PhysicalTransM
        zDicomV = np.empty((len(zV),1))
        all_labels = np.unique(niiSegArray3M)
        all_labels = all_labels[all_labels != 0]
        if len(labels_dict) == 0:
            for label in all_labels:
                labels_dict[label] = "Label " + str(label)
        slcMatchTol = 1e-6
        dt = datetime.now()
        for i in range(len(zV)):
            zDicomV[i] = np.matmul(cerrToDcmTransM, np.asarray((xV[0],yV[0],zV[i], 1)).T)[2]
        for label in labels_dict.keys():
            struct_meta = Structure()
            struct_meta.structureName = labels_dict[label]
            struct_meta.dateWritten = dt.strftime("%Y%m%d")
            struct_meta.roiNumber = ""
            #struct_meta.numberOfScans = len(roi_contour.ContourSequence) # number of scan slices
            struct_meta.strUID = uid.createUID("structure")
            struct_meta.structSetSopInstanceUID = generate_uid()
            struct_meta.assocScanUID = planC.scan[assocScanNum].scanUID
            contour_list = np.empty((0),Contour)
            for slc in range(dim[2]):
                if not np.any(niiSegArray3M[:,:,slc]):
                    continue
                contours = measure.find_contours(niiSegArray3M[:,:,slc] == label, 0.5)
                ind = np.where((zV - structZvalsV[slc])**2 < slcMatchTol)
                #_,_,niiZ = image.TransformIndexToPhysicalPoint((0,0,slc))
                #ind = np.where((zDicomV - niiZ)**2 < slcMatchTol)
                #ipp = image.TransformIndexToPhysicalPoint((0,0,slc))
                #niiZ = - np.sum(slice_normal * ipp) / 10
                #ind = np.where((zV - niiZ)**2 < slcMatchTol)
                if len(ind[0]) == 1:
                    ind = ind[0][0]
                else:
                    raise Exception('No matching slices found.')
                sopClassUID = planC.scan[assocScanNum].scanInfo[ind].sopClassUID
                sopInstanceUID = planC.scan[assocScanNum].scanInfo[ind].sopInstanceUID

                num_contours = len(contours)
                for iContour, contour in enumerate(contours):
                    contObj = Contour()
                    segment = np.empty((contour.shape[0],3))
                    colV = contour[:, 1]
                    rowV = contour[:, 0]
                    slcV = np.ones_like(rowV) * slc
                    ptsM = np.asarray((colV,rowV,slcV))
                    ptsM = np.vstack((ptsM, np.ones((1, ptsM.shape[1]))))
                    ptsM = np.matmul(planC.scan[assocScanNum].Image2PhysicalTransM, ptsM)[:3,:].T
                    contObj.segments = ptsM
                    contObj.referencedSopInstanceUID = sopInstanceUID
                    contObj.referencedSopClassUID = sopClassUID
                    #contour_list.append(contObj)
                    contour_list = np.append(contour_list,contObj)
                struct_meta.contour = contour_list

            struct_list.append(struct_meta)

    return struct_list


def create_from_mask(mask3M, assocScanNum, planC):

    pass

def copyToScan(structNum, scanNum, planC):
    # Get associated scan number for structNum
    origScanNum = scn.getScanNumFromUID(planC.structure[structNum].assocScanUID, planC)
    mask3M = rs.getStrMask(structNum,planC)
    # Get x,y,z, grid for the original scan
    xOrigV, yOrigV, zOrigV = planC.scan[origScanNum].getScanXYZVals()
    # Get x,y,z grid for the new scan
    xNewV, yNewV, zNewV = planC.scan[scanNum].getScanXYZVals()
    # Interpolate mask from original scan to the new scan
    newMask3M = imgResample3D(mask3M.astype(float), xOrigV, yOrigV, zOrigV, xNewV, yNewV, zNewV, 'sitkLinear') >= 0.5
    structName = planC.structure[structNum].structureName
    planC = pc.import_structure_mask(newMask3M, scanNum, structName, planC)
    return planC


def getStructNumFromUID(assocStrUID, planC) -> int:
    uid_list = [s.strUID for s in planC.structure]
    if assocStrUID in uid_list:
        return uid_list.index(assocStrUID)
    else:
        return None

def get_struct_num_from_sop_instance_uid(assocStrUID,planC) -> int:
    uid_list = [s.structSetSopInstanceUID for s in planC.structure]
    if assocStrUID in uid_list:
        return uid_list.index(assocStrUID)
    else:
        return None
