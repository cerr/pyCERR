"""structure module.

Ths structure module defines metadata for segmentation (RTSTRUCT, SEG).
The metadata are attributes of the Structre class.
This module also defines routines for transforming and
accessing the Structure metadata in CERR coordinate system and
to convert images to real world units.

"""

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
import cerr.utils.mask as maskUtils
import warnings


def get_empty_list():
    return []
def get_empty_np_array():
    return np.empty((0,0,0))

@dataclass
class Structure:
    """This class defines data object for volumetric segmentation. The metadata can be populated from DICOM, NifTi and
    numpy arrays.

    Attributes:
        patientName (str): Patient's name.
        structureName (str): Structure's name.
        ROIInterpretedType (str): maps to DICOM tag (3006,00A4).
        numberOfScans (int): Number of scan slices containing segmentation.
        dateWritten (str): Date structure was created. Corresponds to DICOM tag (3006,0008) StructureSetDate.
        structureColor (List): rgb triplet representing the color for this structure
        structureDescription (str): Description of the structure. DICOM SeriesDescription.
        roiGenerationAlgorithm (str) = Type of algorithm used to generate ROI. DICOM tag (3006,0036).
                                        Permitted values are AUTOMATIC, SEMIAUTOMATIC and MANUAL
        roiGenerationDescription (str): User-defined description of technique used to generate ROI. DICOM tag (3006,0038).
        contour (List): List of contours including segmentation x,y,z CERR virtual coordinates per scan slice.
                        The ith entry in the list corresponds to the ith slice of the associated scanArray.
        rasterSegments (np.ndarray):  Numpy array of size numSegments x 10. The columns of this array are
                                      z-value, y-value, x segment start, x segment stop, x increment, slice, row,
                                      column start, column stop, voxel thickness for that slice.
                                      Each row represents a scan segment.
        strUID (str): unique identifier for structure object
        assocScanUID (str): unique identifier for the scan associated with the structure object.
        structSetSopInstanceUID: str = ""
        referencedFrameOfReferenceUID (str): UID for frame of reference
        referencedSeriesUID (str): UID of structure series i.e. DICOM SeriesInstanceUID
        structureFileFormat (str): File format from which structure's metadata was populated.
                                   Permitted values are "RTSTRUCT", "NPARRAY", "NIFTI".
    """

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
    structureColor: List = field(default_factory=get_empty_np_array)
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

    def save_nii(self,niiFileName,planC):
        """ Routine to save pyCERR Structure object to NifTi file

        Args:
            niiFileName (str): File name including the full path to save the pyCERR scan object to NifTi file.
            planC (cerr.plan_container.PlanC): pyCERR plan container object.

        Returns:
            int: 0 when NifTi file is written successfully.
        """

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
        """Routine to convert x,y,z coordinates of segmentation from DICOM to pyCERR virtual coordinates. More information
            about virtual coordinates is on the Wiki https://github.com/cerr/pyCERR/wiki/Coordinate-system
        """

        assocScanUID = self.assocScanUID
        scan_num = scn.getScanNumFromUID(assocScanUID,planC)
        im_to_virtual_phys_transM = planC.scan[scan_num].Image2VirtualPhysicalTransM
        im_to_phys_transM = planC.scan[scan_num].Image2PhysicalTransM
        position_matrix_inv = np.linalg.inv(im_to_phys_transM)
        _,_,zValsV = planC.scan[scan_num].getScanXYZVals()
        isPhysicalCoords = self.structureFileFormat in ["RTSTRUCT", "NPARRAY", "NIFTI"]
        scan_sop_inst_list = np.array([scan_info.sopInstanceUID for scan_info in planC.scan[scan_num].scanInfo])
        numSlcs = len(planC.scan[scan_num].scanInfo)
        for ctr_num,contour in enumerate(self.contour):
            if np.any(contour.segments):
                if isPhysicalCoords:
                    tempa = np.hstack((contour.segments, np.ones((contour.segments.shape[0], 1))))
                    tempa = np.matmul(position_matrix_inv, tempa.T)
                    # Round the z-coordinates (slice) of 'tempa'
                    tempa[2, :] = np.round(tempa[2, :])
                else: # image coords
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
        Args:
            planC (cerr.plan_container.PlanC): pyCERR's plan container object

        Returns:
            int: associated scan index for structure object based on the scan UID associated with
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
        assocScans = [i for i, x in enumerate(scanUID) if x == assocScanUID]

        return assocScans[0]
    
    def getSitkImage(self, planC):
        """ Routine to convert pyCERR Structure object to SimpleITK Image object

        Returns:
            sitk.Image: SimpleITK Image with value of 1 assigned to segmented pixels

        """

        assocScanNum = scn.getScanNumFromUID(self.assocScanUID, planC)
        mask3M = rs.getStrMask(self, planC)
        sitkArray = np.transpose(mask3M.astype(int), (2, 0, 1)) # z,y,x order
        # CERR slice ordering is opposite of DICOM
        if scn.flipSliceOrderFlag(planC.scan[assocScanNum]):
            sitkArray = np.flip(sitkArray, axis = 0)
        originXyz = list(np.matmul(planC.scan[assocScanNum].Image2PhysicalTransM, np.asarray([0,0,0,1]).T)[:3] * 10)
        xV, yV, zV = planC.scan[assocScanNum].getScanXYZVals()
        dx = np.abs(xV[1] - xV[0]) * 10
        dy = np.abs(yV[1] - yV[0]) * 10
        dz = np.abs(zV[1] - zV[0]) * 10
        spacing = [dx, dy, dz]
        img_ori = planC.scan[assocScanNum].scanInfo[0].imageOrientationPatient
        slice_normal = img_ori[[1,2,0]] * img_ori[[5,3,4]] \
                       - img_ori[[2,0,1]] * img_ori[[4,5,3]]
        # Get row-major directions for ITK
        dir_cosine_mat = np.hstack((img_ori.reshape(3,2,order="F"),slice_normal.reshape(3,1)))
        direction = dir_cosine_mat.reshape(9,order='C')
        img = sitk.GetImageFromArray(sitkArray)
        img.SetOrigin(originXyz)
        img.SetSpacing(spacing)
        img.SetDirection(direction)
        return img

    def getStructDict(self):
        """ Routine to get dictionary representation of structure metadata

        Returns:
            dict: fields of the dictionary are attributes of the Structure object.

        """

        structDict = self.__dict__.copy()
        contourList = []
        for ctr in structDict['contour']:
            if ctr:
                ctrDict = ctr.__dict__.copy()
                segList = []
                for seg in ctrDict['segments']:
                    segList.append(seg.__dict__.copy())
                ctrDict['segments'] = segList
                contourList.append(ctrDict)
            else:
                contourList.append([])
        structDict['contour'] = contourList
        return structDict

@dataclass
class Contour:
    """This class defines data object for storing segmented contours. The metadata can be populated from DICOM, NifTi and
    numpy arrays.

    Attributes:
        referencedSopInstanceUID (str): Instance UID of associated image slice.
        referencedSopClassUID (str): Class UID of associated image slice.
        segments (np.array): array of segments.

    """

    referencedSopInstanceUID: str = ""
    referencedSopClassUID: str = ""
    segments: np.array = field(default_factory=get_empty_np_array)

@dataclass
class Segment:
    """This class defines data object for storing contour segments.

    Attributes:
        points (numpy.ndarray): (n X 3) array containing x,y,z coordinates of the segment in pyCERR virtual coordinate system.

    """

    points: np.ndarray = field(default_factory=get_empty_np_array)

class jsonSerializeSegment(json.JSONEncoder):
    def default(self, segObj):
        segDict = {}
        if not segObj:
            segDict['points'] = ''
        if isinstance(segObj, Segment):
            segDict['points'] = segObj.points.tolist()
            return segDict
        else:
            type_name = segObj.__class__.__name__
            raise TypeError("Unexpected type {0}".format(type_name))

class jsonSerializeContour(json.JSONEncoder):
    def default(self, ctrObj):
        ctrDict = {}
        if isinstance(ctrObj, Contour):
            ctrDict['referencedSopClassUID'] = ctrObj.referencedSopClassUID
            ctrDict['referencedSopInstanceUID'] = ctrObj.referencedSopInstanceUID
            segList = []
            for seg in ctrObj.segments:
                segList.append(json.dumps(seg, cls=jsonSerializeSegment))
            ctrDict['segments'] = segList
            return ctrDict
        else:
            type_name = ctrObj.__class__.__name__
            raise TypeError("Unexpected type {0}".format(type_name))

fieldsList = ['structureName', 'patientName', 'assocScanUID', 'strUID',
              'ROIInterpretedType', 'structureColor',
              'roiGenerationDescription', 'roiGenerationAlgorithm',
              'structSetSopInstanceUID', 'referencedFrameOfReferenceUID',
              'structureFileFormat']

class jsonSerializeStruct(json.JSONEncoder):
    def default(self, strObj):
        strDict = {}
        if isinstance(strObj, Structure):
            for fld in fieldsList:
                strDict[fld] = getattr(strObj, fld)
            ctrList = []
            for ctr in strObj.contour:
                ctrList.append(json.dumps(ctr, cls=jsonSerializeContour))
            strDict['contour'] = ctrList
            return strDict
        else:
            type_name = strObj.__class__.__name__
            raise TypeError("Unexpected type {0}".format(type_name))

def getJsonList(structNumV, planC):
    if isinstance(structNumV, (int, float)):
        structNumV = [structNumV]
    strList = []
    for strNum in structNumV:
        strObj = planC.structure[strNum]
        strList.append(json.dumps(strObj, ensure_ascii=False, indent=4, cls=jsonSerializeStruct))
    return strList

def saveJson(structNumV, jsonFileName, planC):
    """
    Args:
        structNumV (List): List of structure indices to export to JSON format.
        jsonFileName (str): JSON file name.
        planC (cerr.plan_container.PlanC): pyCERR's plan container object

    Returns:
        None
    """

    strList = getJsonList(structNumV, planC)
    with open(jsonFileName, 'w', encoding='utf-8') as f:
        json.dump(strList, f, ensure_ascii=False, indent=4)

def importJson(planC, strList=None, jsonFileName=None):
    """

    Args:
        planC (cerr.plan_container.PlanC): pyCERR's plan container object.
        strList (list of structures): (optional) list of structure metadata imported from json.
                                      Required when jsonFileName is None.
        jsonFileName: (optional) JSON file name containing structure metadata.
                      Required when strList is None.

    Returns:
        cerr.plan_container.PlanC: pyCERR's plan container object.
    """

    if jsonFileName:
        with open(jsonFileName, 'r', encoding='utf-8') as f:
            strList = json.load(f)
    elif strList:
        pass
    else:
        return 'Provide filepath or strList'
    strUIDs = [s.strUID for s in planC.structure]
    for strJsonObj in strList:
        strJsonObj = json.loads(strJsonObj)
        if strJsonObj['strUID'] in strUIDs:
            warnings.warn("Structure " + strJsonObj['strUID'] + " not imported from JSON as it already exists in planC")
            continue
        strObj = Structure()
        for fld in fieldsList:
            setattr(strObj, fld, strJsonObj[fld])
        ctrList = strJsonObj['contour']
        num_contours = len(ctrList)
        contour_list = np.empty(num_contours,Contour)
        for ctr_num, ctr in enumerate(ctrList):
            ctr = json.loads(ctr)
            if not ctr:
                contour_list[ctr_num] = []
                continue
            ctrObj = Contour()
            segList = ctr['segments']
            segments = []
            for seg in segList:
                seg = json.loads(seg)
                segment = Segment()
                segment.points = np.asarray(seg['points'])
                segments.append(segment)
            ctrObj.segments = segments
            ctrObj.referencedSopClassUID = ctr['referencedSopClassUID']
            ctrObj.referencedSopInstanceUID = ctr['referencedSopInstanceUID']
            contour_list[ctr_num] = ctrObj
        strObj.contour = contour_list
        planC.structure.append(strObj)
        planC.structure[-1].rasterSegments = rs.generate_rastersegs(planC.structure[-1],planC)

    return planC


def parse_contours(contour_seq):
    """This routine parses the ContourSequence metadata from DICOM and returns a list of pyCERR Contour objects.

    Args:
        contour_seq (pydicom.dataset.Dataset): Pydicom Dataset object for ContourSequence, DICOM tag (3006,0040).

    Returns:
        List[cerr.dataclasses.structure.Contour]: list of pyCERR Contour objects
    """

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


def load_structure(file_list):
    """This routine parses a list of DICOM files and imports metadata from RTSTRUCT and SEG modalities
    to a list of pyCERR's Structure objects
    .

    Args:
        file_list (List[str]): List of DICOM file paths.

    Returns:
        List[cerr.dataclasses.structure.Structure]: List of pyCERR's Structure objects.

    """

    struct_list = []
    for file in file_list:
        ds = dcmread(file)
        if ds.Modality == "RTSTRUCT":
            roi_contour_seq = ds.ROIContourSequence
            str_roi_seq = ds.StructureSetROISequence
            roi_obs_seq = ds.RTROIObservationsSequence
            ctrSeqRefRoiNums = np.array([roiCtr.ReferencedROINumber.real for roiCtr in roi_contour_seq])
            obsSeqRefRoiNums = np.array([roiObs.ReferencedROINumber.real for roiObs in roi_obs_seq])
            num_structs = len(roi_contour_seq)
            #for str_num, (roi_contour,str_roi) in enumerate(zip(roi_contour_seq,str_roi_seq)):
            for str_num, str_roi in enumerate(str_roi_seq):
                struct_meta = Structure() #parse_structure_fields(roi_contour_seq,str_roi_seq)
                struct_meta.patientName = str(ds.PatientName)
                struct_meta.writer = ds.Manufacturer
                struct_meta.dateWritten = ds.StructureSetDate
                if hasattr(ds,"SeriesDescription"): struct_meta.structureDescription = ds.SeriesDescription
                struct_meta.roiNumber = str_roi.ROINumber.real
                struct_meta.structureName = str_roi.ROIName
                struct_meta.roiGenerationAlgorithm = str_roi.ROIGenerationAlgorithm
                if ("3006","0038") in str_roi:
                    struct_meta.roiGenerationDescription  = str_roi["3006","0038"].value
                ref_FOR_uid = str_roi.ReferencedFrameOfReferenceUID.name
                struct_meta.referencedFrameOfReferenceUID = ref_FOR_uid

                # Find the matching ROIContourSequence element
                indCtrSeq = ctrSeqRefRoiNums == struct_meta.roiNumber
                indObsSeq = obsSeqRefRoiNums == struct_meta.roiNumber
                roi_contour = roi_contour_seq[np.where(indCtrSeq)[0][0]]
                roi_obs = roi_obs_seq[np.where(indObsSeq)[0][0]]

                if not hasattr(roi_contour,"ContourSequence"):
                    continue
                struct_meta.contour = parse_contours(roi_contour.ContourSequence)
                struct_meta.numberOfScans = len(roi_contour.ContourSequence) # number of scan slices
                if hasattr(roi_contour, "ROIDisplayColor"):
                    colorTriplet = roi_contour.ROIDisplayColor
                    struct_meta.structureColor  = [int(val) for val in colorTriplet]
                else:
                    struct_meta.structureColor = getColorForStructNum(str_num)
                struct_meta.ROIInterpretedType = roi_obs.RTROIInterpretedType
                struct_meta.strUID = uid.createUID("structure")
                struct_meta.structSetSopInstanceUID =  ds.SOPInstanceUID.name
                struct_meta.structureFileFormat = "RTSTRUCT"
                #structureColor
                # struct_meta.roiGenerationAlgorithm = str_roi.ROIGenerationAlgorithm
                # if ("3006","0038") in str_roi:
                #     struct_meta.roiGenerationDescription  = str_roi["3006","0038"].value
                # struct_meta.contour = parse_contours(roi_contour.ContourSequence)
                # ref_FOR_uid = str_roi.ReferencedFrameOfReferenceUID
                ref_FOR_seq = ds.ReferencedFrameOfReferenceSequence
                for ref_FOR in ref_FOR_seq:
                    if ref_FOR.FrameOfReferenceUID.name == ref_FOR_uid:
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
                #numFrames = len(ds.SharedFunctionalGroupsSequence[0].DerivationImageSequence[0].SourceImageSequence)
                numFrames = len(ds.PerFrameFunctionalGroupsSequence)
                refSegNums = np.array([segId.ReferencedSegmentNumber for segId in ds.SharedFunctionalGroupsSequence[0].SegmentIdentificationSequence] * numFrames)
            for strNum in range(numStructs):
                struct_meta = Structure() #parse_structure_fields(roi_contour_seq,str_roi_seq)
                struct_meta.structureFileFormat = "SEG"
                struct_meta.patientName = str(ds.PatientName)
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
                struct_meta.structureColor = getColorForStructNum(strNum)
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
    """This routine imports segmentation from a list of nii files into planC.

    Args:
        file_list (List or str): List of nii file paths or a string containing path for a single file.
        assocScanNum (int): index of scan in planC to associate the segmentation.
        planC (cerr.plan_container.PlanC): pyCERR's plan container object.
        labels_dict (dict): dictionary of index to structure name mapping. e.g. {1: 'GTV', 2: 'Lung_total'}

    Returns:
        cerr.plan_container.PlanC: pyCERR's plan container object
    """

    if isinstance(file_list,str) and os.path.exists(file_list):
        file_list = [file_list]
    struct_list = []
    numStructs = len(planC.structure)
    numAdded = 0
    for file in file_list:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        image = reader.Execute()

        # Get mask on scan grid
        scanImage = planC.scan[assocScanNum].getSitkImage()
        extrapVal = 0
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(scanImage)
        resample.SetInterpolator(getattr(sitk,'sitkNearestNeighbor'))
        resample.SetDefaultPixelValue(extrapVal)
        resampMaskImage = resample.Execute(image)
        maskOnScan3M = scn.getCERRScanArrayFromITK(resampMaskImage, assocScanNum, planC)

        all_labels = np.unique(maskOnScan3M)
        all_labels = all_labels[all_labels != 0]
        if len(labels_dict) == 0:
            for label in all_labels:
                labels_dict[label] = "Label " + str(label)

        for label in labels_dict.keys():
            planC = import_structure_mask(maskOnScan3M == label, assocScanNum, labels_dict[label], None, planC)

    return planC


def import_structure_mask(mask3M, assocScanNum, structName, structNum, planC):
    """

    Args:
        mask3M (np.ndarray): binary mask for segmentation which is of the same shape as the associated scan
        assocScanNum (int): index of scan object within planC.scan to associate the structure
        structName (str): Name of the structure
        structNum (int or None): None to add new structure or index of structure object within planC.structure to replace
        planC (cerr.plan_container.PlanC): pyCERR's container object

    Returns:
        cerr.plan_container.PlanC: pyCERR's container object with updated planC.structure attribute
    """

    # Pad mask to account for boundary edges
    paddedMask3M = mask3M.astype(int)
    paddedMask3M = np.pad(paddedMask3M, ((1,1),(1,1),(0,0)), 'constant', constant_values = 0)
    dt = datetime.now()
    if isinstance(structNum,(int,float)):
        struct_meta = planC.structure[structNum]
    else:
        struct_meta = Structure()
        structNum = len(planC.structure)
        struct_meta.structureColor = getColorForStructNum(structNum)
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
    if structNum == numOrigStructs:
        planC.structure.append(struct_meta)
    #str_num = len(planC.structure) - 1
    planC.structure[structNum].convertDcmToCerrVirtualCoords(planC)
    planC.structure[structNum].rasterSegments = rs.generate_rastersegs(planC.structure[structNum],planC)
    return planC


def getColorForStructNum(structNum):
    """This routine returns the rgb color triplet to assign to a new structure object.

    Args:
        structNum (int): index of structure object in planC.structure

    Returns:
        List: rgb triplet
    """

    colorMat = np.array([[ 230,   161,     0],
           [0,   230,     0],
           [230,     0,     0],
           [ 0,   207,   207],
           [172,     0,   172],
           [172,   172,     0],
           [138,   172,   230],
           [184,    57,    57],
           [230,     0,   230],
           [172,   115,     0],
           [ 0,   230,   115],
           [230,   115,   230],
           [115,   230,     0],
           [69,     0,   207],
           [0,   161,    69],
           [161,    69,     0],
           [0,   184,   207],
           [161,     0,   184],
           [207,   138,     0],
           [76,   151,   230],
           [230,    76,    76],
           [230,     0,   207],
           [207,   115,     0],
           [0,   230,    92],
           [207,   138,   207],
           [138,   207,     0],
           [161,    92,   184],
           [138,   207,    46]])
    # cycle colors
    colorIndex = np.mod(structNum, colorMat.shape[0])
    return list(colorMat[colorIndex,:])


def copyToScan(structNum, scanNum, planC):
    """This routine copies structure object at index structNum to scan objact at index scanNum in planC.scan.

    Args:
        structNum (int): index of structure object in planC.structure
        scanNum (int): index of scan object in planC.scan
        planC (cerr.plan_container.PlanC): pyCERR's planc ontainer object

    Returns:
        cerr.plan_container.PlanC): updated planC with new planC.structure element associated with scanNum

    """

    # Get associated scan number for structNum
    origScanNum = scn.getScanNumFromUID(planC.structure[structNum].assocScanUID, planC)
    mask3M = rs.getStrMask(structNum,planC)
    # Get x,y,z, grid for the original scan
    xOrigV, yOrigV, zOrigV = planC.scan[origScanNum].getScanXYZVals()
    # Get x,y,z grid for the new scan
    xNewV, yNewV, zNewV = planC.scan[scanNum].getScanXYZVals()
    # Interpolate mask from original scan to the new scan
    extrapVal = 0
    newMask3M = imgResample3D(mask3M.astype(float), xOrigV, yOrigV, zOrigV, xNewV, yNewV, zNewV, 'sitkLinear',extrapVal) >= 0.5
    structName = planC.structure[structNum].structureName
    structNum = None
    planC = import_structure_mask(newMask3M, scanNum, structName, structNum, planC)
    return planC


def getStructNumFromUID(assocStrUID, planC) -> int:
    """This routine returns the index of the planC.structure element corresponding to the input pyCERR's structure UID

    Args:
        assocStrUID (str): UID of the structure object
        planC (cerr.plan_container.PlanC): pyCERR's plan container object

    Returns:
        int: Index of the planC.structure element corresponding to the input UID
             None when there is no matching element in planC.structure corresponding to the input UID.
    """

    uid_list = [s.strUID for s in planC.structure]
    if assocStrUID in uid_list:
        return uid_list.index(assocStrUID)
    else:
        return None

def getStructNumFromSOPInstanceUID(assocStrUID,planC) -> int:
    """This routine returns the index of the planC.structure element corresponding to the input SOP Instance UID

    Args:
        assocStrUID (str): SOP Instance UID of the structure object
        planC (cerr.plan_container.PlanC): pyCERR's plan container object

    Returns:
        int: Index of the planC.structure element corresponding to the input UID
             None when there is no matching element in planC.structure corresponding to the input SOP Instance UID.
    """

    uid_list = [s.structSetSopInstanceUID for s in planC.structure]
    if assocStrUID in uid_list:
        return uid_list.index(assocStrUID)
    else:
        return None

def calcIsocenter(strNum, planC):
    """This routine calculates the isocenter of the input structure index in planC.structure

    Args:
        strNum (int): Index of structure object in planC.structure
        planC (cerr.plan_container.PlanC): pyCERR's plan container

    Returns:
        List: x,y,z coordinates in pyCERR's virtual coordinate system for the isocenter.
    """

    assocScanNum = scn.getScanNumFromUID(planC.structure[strNum].assocScanUID, planC)
    mask3M = rs.getStrMask(strNum,planC)
    rV, cV, sV = np.where(mask3M)
    midSliceInd = int(np.round(sV.mean()))
    midRowInd = int(np.round(rV.mean()))
    midColInd = int(np.round(cV.mean()))
    # store isocenter to update viewer to display the central slice later on
    xV, yV, zV = planC.scan[assocScanNum].getScanXYZVals()
    isocenter = [xV[midColInd], yV[midRowInd], zV[midSliceInd]]
    return isocenter

def getMatchingIndex(structName, strList, matchCriteria='exact'):
    """This routine returns the index of element/s from the list of structure names that match the input name.

    Args:
        structName (str): Structure name to find a match
        strList: List of structure names
        matchCriteria: Criteria used to find the match
            'EXACT' - returns indices of exact matches
            'FIRSTCHARS' - returns indices where first characters of elements in the list match input structName

    Returns:
        List: list of matching indices from input strList
    """

    if matchCriteria.upper() == 'EXACT':
        indMatchV = [i for i, s in enumerate(strList) if s.lower() == structName.lower()]
    elif matchCriteria.upper() == 'FIRSTCHARS':
        indMatchV = [i for i, s in enumerate(strList) if s.lower().startswith(structName.lower())]
    else:
        # implementation for 'contains' match criteria
        indMatchV = []
        for i, s in enumerate(strList):
            if structName.lower() in s.lower():
                indMatchV.append(i)
    return indMatchV

def getContourPolygons(strNum, planC, rcsFlag=False):
    """This routine returns the list of polygonal coordinates for all the segments of input structutre.

    Args:
        strNum (int): index of structure element in planC.structure
        planC (cerr.plan_container.PlanC): pyCERR's plan container object
        rcsFlag (bool): optional, flag to return polygonal coordinates in row,col,slc units.
                        By default, the polygonal coordinates are returned in physical units of cm.

    Returns:
        list: list of nx3 arrays corresponding to polygonal segments, where n is the number of points in that segment,
            the columns of each array are x,y,z coordinates in physical units of cm or r,c,s units.

    """
    assocScanNum = scn.getScanNumFromUID(planC.structure[strNum].assocScanUID, planC)
    numSlcs = len(planC.structure[strNum].contour)
    polygons = []
    for slc in range(numSlcs):
        if planC.structure[strNum].contour[slc]:
            for seg in planC.structure[strNum].contour[slc].segments:
                if rcsFlag:
                    rowV, colV = rs.xytom(seg.points[:,0], seg.points[:,1],slc,planC, assocScanNum)
                    pts = np.array((rowV, colV, slc*np.ones_like(rowV)), dtype=np.float64).T
                else:
                    pts = seg.points
                polygons.append(pts)
    return polygons

def getClosedMask(structNum, structuringElementSizeCm, planC, saveFlag=False,\
              replaceFlag=None, procSructName=None):
    """
    Function for morphological closing and hole-filling for binary masks

    Args:
        structNum : int for index of structure in planC.
        structuringElementSizeCm : float for size of structuring element for closing in cm
        planC: pyCERR plan container object.
        saveFlag: [optional, default=False] bool flag for saving processed mask to planC.
        replaceFlag: [optional, default=False] bool flag for replacing input mask with
                    processed mask in planC.
        procSructName: [optional, default=None] string for output structure name.
                      Original structure name is used if None.
    Returns:
        filledMask3M: np.ndarray(dtype=bool) for filled mask.
        planC: pyCERR plan container object.
    """

    # Get binary mask of structure
    mask3M = rs.getStrMask(structNum,planC)

    # Get mask resolution
    assocScanNum = scn.getScanNumFromUID(planC.structure[structNum].assocScanUID,\
                                              planC)
    inputResV = planC.scan[assocScanNum].getScanSpacing()

    filledMask3M = maskUtils.closeMask(mask3M,inputResV,structuringElementSizeCm)

    # # Create structuring element
    # structuringElement = createStructuringElement(structuringElementSizeCm,\
    #                                               inputResV, dimensions=3)
    #
    # # Apply morphological closing
    # closedMask3M = morphologicalClosing(mask3M, structuringElement)
    #
    # # Fill any remaining holes
    # filledMask3M = fillHoles(closedMask3M)

    # Save to planC
    if saveFlag:
        if procSructName is None:
            procSructName = planC.structure[structNum].structureName

        assocScanNum = scn.getScanNumFromUID(planC.structure[structNum].assocScanUID,\
                                                  planC)
        newStructNum = None
        if replaceFlag:
            # Delete structNum
            #del planC.structure[structNum]
            newStructNum = structNum
        #pc.import_structure_mask(filledMask3M, assocScanNum, procSructName, planC)
        planC = import_structure_mask(filledMask3M, assocScanNum, procSructName, newStructNum, planC)


    return filledMask3M, planC

def getLargestConnComps(structNum, numConnComponents, planC=None, saveFlag=None,\
                        replaceFlag=None, procSructName=None):
    """
    Function to retain 'N' largest connected components in input binary mask

    Args:
        structNum: int for index of structure in planC
                   (OR) np.ndarray(dtype=bool) 3D binary mask.
        structuringElementSizeCm: float for desired size of structuring element for
                                  morphological closing in cm.
        planC: [optional, default=None] pyCERR plan container object.
        saveFlag: [optional, default=False] bool flag for importing filtered mask
                  to planC if set to True.
        replaceFlag: [optional, default=False] bool flag for replacing
                     input mask with processed mask to planC if set to True.
        procSructName: [optional, default=None] string for output structure name.
                      Original structure name is used if empty.

    Returns:
        maskOut3M: np.ndarray(dtype=bool) filtered binary mask.
        planC: pyCERR plan container object.

    """


    if np.isscalar(structNum):
        # Get binary mask of structure
        mask3M = rs.getStrMask(structNum,planC)
    else:
        # Input is binary structure mask
        mask3M = structNum

    maskOut3M = maskUtils.largestConnComps(mask3M, numConnComponents)

    # if np.sum(mask3M) > 1:
    #     #Extract connected components
    #     labeledArray, numFeatures = label(mask3M, structure=np.ones((3, 3, 3)))
    #
    #     # Sort by size
    #     ccSiz = [len(labeledArray[labeledArray == i]) for i in range(1, numFeatures + 1)]
    #     rankV = np.argsort(ccSiz)[::-1]
    #     if len(rankV) > numConnComponents:
    #         selV = rankV[:numConnComponents]
    #     else:
    #         selV = rankV[:]
    #
    #     # Return N largest
    #     maskOut3M = np.zeros_like(mask3M, dtype=bool)
    #     for n in selV:
    #         idxV = labeledArray == n + 1
    #         maskOut3M[idxV] = True
    # else:
    #     maskOut3M = mask3M

    if planC is not None and saveFlag and np.isscalar(structNum):
        if procSructName is None:
            procSructName = planC.structure[structNum].structureName

        assocScanNum = scn.getScanNumFromUID(planC.structure[structNum].assocScanUID,\
                                                  planC)
        newStructNum = None
        if replaceFlag:
            # Delete structNum
            #del planC.structure[structNum]
            newStructNum = structNum
        planC = import_structure_mask(maskOut3M, assocScanNum, procSructName, newStructNum, planC)

    return maskOut3M, planC


def getLabelMap(strNumV, planC, labelDict=None, dim=3):
    """
    Function to create label map for user-specified structures.

    Args:
        strNumV (list) : Structure indices to be exported.
        planC (plan_container.planC): pyCERR plan_container object.
        labelDict (dict): [optional, default={}] dictionary mapping indices with structure names.
        dim (int): [optional, default=3] int indicating dimensions of output label map.
                   When set to 3, returns 3D label map np.array(dtype=int).
                   When set to 4, returns 4D array of binary masks (required for overlapping structures).

    Returns:
       labelMap3M: np.ndarray(dtype=int) for label map.
    """

    if labelDict is None:
        labelDict = {}
        # Assign default labels
        for idx in range(len(strNumV)):
            strName = planC.structure[strNumV[idx]].structureName
            labelDict[idx + 1] = strName

    allLabels = labelDict.keys()

    if dim == 3:
        assocScan = planC.structure[strNumV[0]].getStructureAssociatedScan(planC)
        shape = planC.scan[assocScan].getScanSize()
        labelMap = np.zeros(shape, dtype=int)
        for strNum in strNumV:
            strName = planC.structure[strNum].structureName
            strLabel = [label for label in allLabels if labelDict[label] == strName]
            if isinstance(strLabel, str):
                strLabel = int(strLabel)
            mask3M = rs.getStrMask(strNum, planC)
            if np.any(np.logical_and(mask3M, labelMap > 0)):
                raise Exception("Overlapping structures encountered. Please set dim=4.")
            labelMap[mask3M] = strLabel
    elif dim == 4:
        labelMap = np.array(getMaskList(strNumV, planC, labelDict=labelDict))
    else:
        raise ValueError("Invalid input. Dim must be 3 or 4.")

    return labelMap

def getMaskList(strNumV, planC, labelDict=None):
    """
        Function to create list of binary masks of user-specified structures.

        Args:
            strNumV: list of structure indices to be exported.
            planC: pyCERR plan_container object.
            labelDict: [optional, default={}] dictionary mapping indices with structure names.

        Returns:
           maskList: list(dtype=bool) of binary masks.
    """

    if labelDict is None or len(labelDict)==0:
        labelDict = {}
        # Assign default labels
        for idx in range(len(strNumV)):
            strName = planC.structure[strNumV[idx]].structureName
            labelDict[idx + 1] = strName

    maskList = []
    allLabels = list(labelDict.keys())
    for idx in range(len(strNumV)):
        scanNum = scn.getScanNumFromUID(planC.structure[strNumV[idx]].assocScanUID, planC)
        mask3M = rs.getStrMask(strNumV[idx], planC)
        strName = planC.structure[strNumV[idx]].structureName
        strLabel = [label for label in allLabels if labelDict[label] == strName][0]
        if isinstance(strLabel, str):
            strLabel = int(strLabel)
        mask3M = np.moveaxis(mask3M, [0, 1], [1, 0])
        if scn.flipSliceOrderFlag(planC.scan[scanNum]):
            mask3M = np.flip(mask3M, axis=2)
        maskList.insert(strLabel-1, mask3M)

    return maskList
