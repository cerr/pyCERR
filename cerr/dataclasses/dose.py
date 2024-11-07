"""dose module.

The dose module defines metadata for an RTDOSE object.
The metadata are attributes of the Dose class.
This module also defines routines for transforming and
accessing the Dose metadata in CERR coordinate system.

"""

import json
from dataclasses import dataclass, field
from math import sqrt
import numpy as np
from pydicom import dcmread
from cerr.dataclasses import scan as scn
from cerr.dataclasses import structure
from cerr.utils import uid
from cerr.utils.interp import finterp3
import nibabel as nib
import json
import os
import SimpleITK as sitk
from cerr.radiomics.preprocess import imgResample3D


def get_empty_list():
    return []
def get_empty_np_array():
    return np.empty((0,0,0))

@dataclass
class Dose:
    """This class defines data object for RTDose. The metadata is populated from DICOM.

    Attributes:
        patientName (str): Patient's name
        doseType (str): Type of dose as per (3004,0004). Values can be PHYSICAL, EFFECTIVE or ERROR
        doseSummationType (str): Type of dose summation as per (3004,000A)
        refBeamNumber (int): Referenced beam number from ReferencedRTPlanSequence
        refFractionGroupNumber (int): Referenced Fraction Group number from ReferencedRTPlanSequence
        numberMultiFrameImages (int): Number of image frames
        doseUnits (str): Units used to describe dose. GY or RELATIVE
        doseScale (float): Scaling factor that when multiplied by the dose grid data found in Pixel Data (7FE0,0010) Attribute
            of the Image Pixel Module, yields grid doses in the dose units as specified by Dose Units (3004,0002).
        fractionGroupID (str): Fraction Group ID from Referenced RTPLAN
        imagePositionPatient (np.array): x,y,z coordinate of the top left voxel of the dose volume.
        imageOrientationPatient (np.array): Direction cosine of dose row and column with patient coordinate system.
        sizeOfDimension1 (int): Number of columns of doseArray
        sizeOfDimension2 (int): Number of rows of doseArray
        sizeOfDimension3 (int): Number of slices of doseArray
        coord1OFFirstPoint (float): x-coordinate of dose in CERR virtual coordinates
        coord2OFFirstPoint (float): y-coordinate of dose in CERR virtual coordinates
        horizontalGridInterval (float): delta x of dose in CERR virtual coordinates
        verticalGridInterval (float): delta y of dose in CERR virtual coordinates
        writer (str): Equipment Manufacturer for RTDOSE delivery.
        dateWritten (str): Study Date.
        studyInstanceUID (str): Study Instance UID of dose.
        xcoordOfNormaliznPoint (float): x-ccordinate of normalization point
        ycoordOfNormaliznPoint (float): y-ccordinate of normalization point
        zcoordOfNormaliznPoint (float): z-ccordinate of normalization point
        doseAtNormaliznPoint (float): dose at normalization point
        coord3OfFirstPoint (float): z-coordinate of dose in CERR virtual coordinates
        doseArray (np.array): 3D volume for RTODSE in doseUnits
        zValues (np.array): z-coordinates of doseArray in CERR virtual coordinate system.
        delivered (str): whether the dose was delivered.
        transM (np.array): transformation matrix to transform dose.
        doseUID (str): unique identifier of dose.
        assocScanUID (str): associated scan's unique identifier
        assocBeamUID (str): associated RTPLAN's unique identifier
        frameOfReferenceUID (str): Frame of Reference UID
        refRTPlanSopInstanceUID (str): SOP Instance UID of associated RTPLAN
        refStructSetSopInstanceUID (str): SOP Instance UID of referenced RTSTRUCT
        prescriptionDose (float): Prescription dose
        doseOffset (float): offset value to add to doseArray
        Image2PhysicalTransM (np.ndarray): Transformation matrix to convert pyCERR's dose row,col,slc to DICOM physical coordinates.
        cerrDcmSliceDirMatch (bool): Flag whether pyCERR slice order matches DICOM.

    """

    caseNumber: int = 0
    patientName: str = ""
    doseNumber: int = 0
    doseType: str = ""
    doseSummationType: str = ""
    refBeamNumber: int = 0
    refFractionGroupNumber: int = 0
    numberMultiFrameImages: int = 0
    doseUnits: str = ""
    doseScale: float = 1
    fractionGroupID: str = ""
    numberOfTx: int = 0
    orientationOfDose: str = ""
    imagePositionPatient : np.array = field(default_factory=get_empty_np_array)
    imageOrientationPatient: np.array = field(default_factory=get_empty_np_array)
    numberRepresentation: int = 0
    numberOfDimensions: int = 0
    sizeOfDimension1: int = 0
    sizeOfDimension2: int = 0
    sizeOfDimension3: int = 0
    coord1OFFirstPoint: float = 0
    coord2OFFirstPoint: float = 0
    horizontalGridInterval: float = 0
    verticalGridInterval: float = 0
    doseDescription: str = ""
    doseEdition: str = ""
    unitNumber: int = 0
    writer: str = ""
    dateWritten: str = ""
    planNumberOfOrigin: int = 0
    planEditionOfOrigin: str = ""
    studyNumberOfOrigin: int = 0
    studyInstanceUID: str = ""
    versionNumberOfProgram: str = ""
    xcoordOfNormaliznPoint: float = np.nan
    ycoordOfNormaliznPoint: float = np.nan
    zcoordOfNormaliznPoint: float = np.nan
    doseAtNormaliznPoint: float = np.nan
    doseError: float = np.nan
    coord3OfFirstPoint: float = np.nan
    depthGridInterval: float = np.nan
    planIDOfOrigin: str = ""
    doseArray: np.array = field(default_factory=get_empty_np_array)
    zValues: np.array = field(default_factory=get_empty_np_array)
    delivered: str = ""
    cachedColor: str = ""
    cachedTime: str = ""
    numCachedSlices: int = 0
    transferProtocol: str = ""
    associatedScan: int = np.nan
    transM: np.array = field(default_factory=get_empty_np_array)
    doseUID: str = ""
    assocScanUID: str = ""
    assocBeamUID: str = ""
    frameOfReferenceUID: str = ""
    refRTPlanSopInstanceUID: str = ""
    refStructSetSopInstanceUID: str = ""
    prescriptionDose: float = 0
    doseOffset: float = 0
    Image2PhysicalTransM: np.array = field(default_factory=get_empty_np_array)
    cerrDcmSliceDirMatch: bool = False

    class json_serialize(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Dose):
                return {'dose':obj.doseUID}
            return "" #json.JSONEncoder.default(self, obj)


    def getNiiAffine(self):
        """ Routine for affine transformation of pyCERR dose object for storing in NifTi format

        Returns:
            np.ndarray: 3x3 affine matrix
        """

        doseAffine3M = self.Image2PhysicalTransM.copy()
        # nii row and col are reverse of dicom, convert cm to mm
        doseAffine3M[0,:] = -doseAffine3M[0,:] * 10 # nii row is reverse of dicom, cm to mm
        doseAffine3M[1,:] = -doseAffine3M[1,:] * 10 # nii col is reverse of dicom, cm to mm
        doseAffine3M[2,:] = doseAffine3M[2,:] * 10 # cm to mm
        return doseAffine3M

    def getSitkImage(self):

        sitkArray = np.transpose(self.doseArray, (2, 0, 1)) # z,y,x order
        if not self.cerrDcmSliceDirMatch:
            sitkArray = np.flip(sitkArray, axis = 0)

        originXyz = list(np.matmul(self.Image2PhysicalTransM, np.asarray([0,0,0,1]).T)[:3] * 10)
        xV, yV, zV = self.getDoseXYZVals()
        dx = np.abs(xV[1] - xV[0]) * 10
        dy = np.abs(yV[1] - yV[0]) * 10
        dz = np.abs(zV[1] - zV[0]) * 10
        spacing = [dx, dy, dz]
        img_ori = self.imageOrientationPatient
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

    def saveNii(self, niiFileName):
        """ Routine to save pyCERR Dose object to NifTi file

        Args:
            niiFileName (str): File name including the full path to save the pyCERR dose object to NifTi file.

        Returns:
            int: 0 when NifTi file is written successfully.
        """

        img = self.getSitkImage()
        sitk.WriteImage(img, niiFileName)

        # # https://neurostars.org/t/direction-orientation-matrix-dicom-vs-nifti/14382/2
        # doseArray = self.doseArray
        # doseArray = np.moveaxis(doseArray,[0,1],[1,0])
        # #doseArray = np.flip(doseArray,axis=[0,1])
        # if not self.cerrDcmSliceDirMatch:
        #     doseArray = np.flip(doseArray,2)
        # doseAffine3M = self.getNiiAffine()
        # dose_img = nib.Nifti1Image(doseArray, doseAffine3M)
        # nib.save(dose_img, niiFileName)

    def getImage2PhysicalTransM(self, assocScanNum, planC):

        imgOrientation = self.imageOrientationPatient
        imagePositionPatient = self.imagePositionPatient / 10
        spacing = [-self.verticalGridInterval, self.horizontalGridInterval]

        zV = self.zValues
        deltaDosePos = [0, 0, (zV[0] - zV[1])/10, 1]
        dcmDosePtPos = np.matmul(planC.scan[assocScanNum].cerrToDcmTransM, deltaDosePos)[:3]

        dcmImgOrientation = imgOrientation.reshape(6,1)
        image2PhysicalTransM = np.hstack((np.matmul(dcmImgOrientation.reshape(3, 2, order="F"), np.diag(spacing)),
                                          np.array([[dcmDosePtPos[0], imagePositionPatient[0]],
                                                    [dcmDosePtPos[1], imagePositionPatient[1]],
                                                    [dcmDosePtPos[2], imagePositionPatient[2]]])))
        image2PhysicalTransM = np.vstack((image2PhysicalTransM, np.array([0, 0, 0, 1])))

        return image2PhysicalTransM


    def convertDcmToCerrVirtualCoords(self, planC):
        """Routine to get scan from DICOM to pyCERR virtual coordinates. More information
            about virtual coordinates is on the Wiki https://github.com/cerr/pyCERR/wiki/Coordinate-system
        """

        # Get CERR z-ccordiinate for dose slices based on associated scan's virtual coordinate transform.

        # Get coord1OFFirstPoint,coord2OFFirstPoint, horizontalGridInterval, verticalGridInterval
        # from coordinates of left-top corner (0,0) voxel of doseArray

        #sort_index = [i for i,x in sorted(enumerate(scan_info),key=get_slice_position, reverse=False)]
        #sorted_index = np.argsort(self.zValues)
        #scan_array = np.array(scan_array)
        #scan_array = np.moveaxis(scan_array,[0,1,2],[2,0,1])
        #scan_info = np.array(scan_info)
        #scan_info = scan_info[sort_index]
        #scan_array = scan_array[:,:,sort_index]

        # Get associated scan number and structure number from planC
        assoc_str_num = None
        assoc_scan_num = None
        for plan in planC.beams:
            if self.refRTPlanSopInstanceUID == plan.SOPInstanceUID:
                if len(plan.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID) > 0:
                    self.refStructSetSopInstanceUID = plan.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
                    assoc_str_num = structure.getStructNumFromSOPInstanceUID(self.refStructSetSopInstanceUID,planC)
                    break
        if self.assocScanUID == "" and assoc_str_num != None:
            assocScanUID = planC.structure[assoc_str_num].assocScanUID
            assoc_scan_num = scn.getScanNumFromUID(assocScanUID,planC)
            if planC.scan[assoc_scan_num].scanInfo[0].frameOfReferenceUID == self.frameOfReferenceUID:
                self.assocScanUID = assocScanUID
        elif self.assocScanUID == "": # associate based on frame of reference UID
            for scan_num, scan in enumerate(planC.scan):
                if scan.scanInfo[0].frameOfReferenceUID == self.frameOfReferenceUID:
                    assoc_scan_num = scan_num
                    self.assocScanUID = scan.scanUID
                    break
        else:
            assoc_scan_num = scn.getScanNumFromUID(self.assocScanUID,planC)


        im_to_phys_transM = planC.scan[assoc_scan_num].Image2PhysicalTransM
        im_to_virtual_phys_transM = planC.scan[assoc_scan_num].Image2VirtualPhysicalTransM
        position_matrix_inv = np.linalg.inv(im_to_phys_transM)
        dose_size = self.doseArray.shape
        dose_image_coords = np.array([[0, 0, 0, 1], [1, 1, 0, 1]])
        dose_dcm_phys_coords = np.matmul(self.Image2PhysicalTransM,dose_image_coords.T)
        scan_image_coords = np.matmul(position_matrix_inv, dose_dcm_phys_coords)
        scan_phys_coords = np.matmul(im_to_virtual_phys_transM, scan_image_coords)
        self.coord1OFFirstPoint = scan_phys_coords[0,0]
        self.coord2OFFirstPoint = scan_phys_coords[1,0]
        self.horizontalGridInterval = scan_phys_coords[0,1] - scan_phys_coords[0,0]
        self.verticalGridInterval = scan_phys_coords[1,1] - scan_phys_coords[1,0]

        # Get zValue for dose slices in CERR coordinate system
        dose_image_coords = np.zeros((4,dose_size[2]),int)
        dose_image_coords[2,:] = np.arange(dose_size[2])
        dose_image_coords[3,:] = np.ones((1,dose_size[2]))
        dose_dcm_phys_coords = np.matmul(self.Image2PhysicalTransM,dose_image_coords)
        scan_image_coords = np.matmul(position_matrix_inv, dose_dcm_phys_coords)
        scan_phys_coords = np.matmul(im_to_virtual_phys_transM, scan_image_coords)
        self.zValues = scan_phys_coords[2,:]
        #xV,yV,zV = planC.scan[assoc_scan_num].getScanXYZVals()
        #self.cerrDcmSliceDirMatch = np.sign(self.zValues[-1] - self.zValues[0]) == np.sign(zV[-1]-zV[0])
        #if not self.cerrDcmSliceDirMatch:
        # Order doseArray in decreasing order of iP . imgOri
        self.cerrDcmSliceDirMatch = True
        if self.zValues[1] < self.zValues[0]:
            self.cerrDcmSliceDirMatch = False
            self.doseArray = np.flip(self.doseArray,axis=2)
            self.zValues = np.flip(self.zValues,axis=0)
        return self

    def getDoseXYZVals(self):
        """ Routine to obtain pyCERR dose object's x,y,z grid coordinates. The coordinates are in pyCERR's
        virtual coordinate system.

        Returns:
            tuple: x, y, z coordinates corresponding to the columns, rows, slices of scan voxels

        """
        xValsV = np.arange(self.coord1OFFirstPoint,
                           self.sizeOfDimension1*self.horizontalGridInterval + self.coord1OFFirstPoint,
                            self.horizontalGridInterval)
        yValsV = np.arange(self.coord2OFFirstPoint,
                           self.sizeOfDimension2*self.verticalGridInterval + self.coord2OFFirstPoint,
                           self.verticalGridInterval)
        zVals = self.zValues

        return xValsV,yValsV,zVals

    def getDoseAt(self,xV,yV,zV):
        """ Routine to obtain dose at input x,y,z grid coordinates. The coordinates are in pyCERR's
        virtual coordinate system.

        Returns:
            tuple (np.array): An array of dose values.

        """

        xVD, yVD, zVD = self.getDoseXYZVals()
        delta = 1e-8
        zVD[0] = zVD[0] - 1e-3
        zVD[-1] = zVD[-1] + 1e-3
        xFieldV = np.asarray([xVD[0] - delta, xVD[1] - xVD[0], xVD[-1] + delta])
        yFieldV = np.asarray([yVD[0] + delta, yVD[1] - yVD[0], yVD[-1] - delta])
        zFieldV = np.asarray(zVD)
        doseV = finterp3(xV,yV,zV,self.doseArray,xFieldV,yFieldV,zFieldV)
        return doseV

    def getAssociatedBeamNum(self, planC):
        """Routine to obtain index of planC.beams that generated this RTDOSE

        Args:
            planC (cerr.plan_container.PlanC): pyCERR's plan container object

        Returns:
            int: index of planC.beams

        """

        beamsUidList = [b.SOPInstanceUID for b in planC.beams]
        if self.refRTPlanSopInstanceUID in beamsUidList:
            return beamsUidList.index(self.refRTPlanSopInstanceUID)
        else:
            return None

def loadDose(file_list):
    """

    Args:
        file_list (list): list of files to read into pyCERR's Dose object

    Returns:
        List[cerr.daatclasses.dose.Dose]: List whose elements are pyCERR Dose objects containing metadata from file_list.

    """

    dose_list = []
    for file in file_list:
        ds = dcmread(file)
        if ds.Modality == "RTDOSE":
            dose_meta = Dose() #parse_structure_fields(roi_contour_seq,str_roi_seq)
            dose_meta.patientName = ds.PatientName
            if hasattr(ds,"Manufacturer"): dose_meta.writer = ds.Manufacturer
            dose_meta.dateWritten = ds.StudyDate
            dose_meta.doseType = ds.DoseType
            dose_meta.doseSummationType = ds.DoseSummationType
            dose_meta.frameOfReferenceUID = ds.FrameOfReferenceUID
            if hasattr(ds,"ReferencedRTPlanSequence"):
                dose_meta.refRTPlanSopInstanceUID = ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID
                if hasattr(ds.ReferencedRTPlanSequence[0],"ReferencedFractionGroupSequence"):
                    if hasattr(ds.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0],
                               "ReferencedBeamSequence"):
                        numBeams = ds.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].\
                            ReferencedBeamSequence
                        dose_meta.refFractionGroupNumber = ds.ReferencedRTPlanSequence[0].\
                            ReferencedFractionGroupSequence[0].ReferencedFractionGroupNumber
                        if numBeams > 0:
                            dose_meta.refBeamNumber = ds.ReferencedRTPlanSequence[0].\
                                ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].\
                                ReferencedBeamNumber

            dose_meta.numberMultiFrameImages = ds.NumberOfFrames
            dose_meta.doseUnits = ds.DoseUnits
            if np.any(dose_meta.doseUnits.upper() == np.array(['GY', 'GYS', 'GRAYS', 'GRAY'])):
                dose_meta.doseUnits = "GRAYS"
            dose_meta.doseScale = ds.DoseGridScaling
            # to do - get fractionGroupID based on RTPLAN when available
            dose_meta.fractionGroupID = dose_meta.doseSummationType + "(" + dose_meta.doseUnits + ")"

            gridFrameOffVec = ds.GridFrameOffsetVector
            img_ori = np.array(ds.ImageOrientationPatient)
            img_ori = img_ori.reshape(6,1)
            ipp = np.array(ds.ImagePositionPatient)
            slice_normal = img_ori[[1,2,0]] * img_ori[[5,3,4]] \
                           - img_ori[[2,0,1]] * img_ori[[4,5,3]]
            slice_normal = slice_normal.reshape((1,3))
            if gridFrameOffVec[0] == 0:
                doseZstart = np.matmul(slice_normal, ipp)
                doseZValuesV = (doseZstart + gridFrameOffVec)
            else:
                doseZValuesV = gridFrameOffVec # as per DICOM documentation, this case is valid only for HFS [1,0,0,0,1,0]
            doseZValuesV = doseZValuesV / 10
            dose_meta.zValues = -doseZValuesV

            # build image to physical units transformation matrix for dose

            # Compute slice normal
            img_ori = np.array(ds.ImageOrientationPatient)
            img_ori = img_ori.reshape(6,1)
            ipp = np.array(ds.ImagePositionPatient) / 10
            slice_normal = img_ori[[1,2,0]] * img_ori[[5,3,4]] \
                           - img_ori[[2,0,1]] * img_ori[[4,5,3]]
            vec3 = slice_normal * (doseZValuesV[1]-doseZValuesV[0])
            pixelSpacing = [ds.PixelSpacing[0]/10, ds.PixelSpacing[1]/10]
            position_matrix_dose = np.hstack((np.matmul(img_ori.reshape(3, 2, order="F"), np.diag(pixelSpacing)),
                                           np.array([[vec3[0,0], ipp[0]],[vec3[1,0], ipp[1]], [vec3[2,0], ipp[2]]])))
            position_matrix_dose = np.vstack((position_matrix_dose, np.array([0, 0, 0, 1])))
            dose_meta.Image2PhysicalTransM = position_matrix_dose

            dose_meta.verticalGridInterval = -pixelSpacing[0]
            dose_meta.horizontalGridInterval = pixelSpacing[1]

            dose_meta.sizeOfDimension1 = ds.Columns
            dose_meta.sizeOfDimension2 = ds.Rows
            dose_meta.sizeOfDimension3 = ds.NumberOfFrames

            # to do - get refStructSetSopInstanceUID based on RTPLAN when available

            dose_meta.doseArray = np.moveaxis(ds.pixel_array,[0,1,2],[2,0,1]) \
                                  * ds.DoseGridScaling

            # #dose_meta.refStructSetSopInstanceUID
            dose_meta.doseUID = uid.createUID("dose")

            dose_list.append(dose_meta)

    return dose_list


def importNii(file_list, assocScanNum, planC):
    """This routine imports RT dose distributions from a list of nii files into planC.

    Args:
        file_list (List or str): List of nii file paths or a string containing path for a single file.
        assocScanNum (int): index of scan in planC to associate the segmentation.
        planC (cerr.plan_container.PlanC): pyCERR's plan container object.

    Returns:
        cerr.plan_container.PlanC: pyCERR's plan container object
    """

    if isinstance(file_list,str) and os.path.exists(file_list):
        file_list = [file_list]
    for file in file_list:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        image = reader.Execute()

        # Assign direction
        direction = planC.scan[assocScanNum].getScanOrientation()
        dirFlipDict = {'L': 'R',
                       'R': 'L',
                       'S': 'I',
                       'I': 'S',
                       'P': 'A',
                       'A': 'P'
                        }
        if scn.flipSliceOrderFlag(planC.scan[0]):
            direction = direction[:2] + dirFlipDict[direction[2]]
        image = sitk.DICOMOrient(image,direction)
        numAxes = len(image.GetSize())
        if numAxes == 4:
            Exception('4-D dose not supported')
        # Get numpy array for scan
        axisOff = numAxes - 3
        doseArray3M = sitk.GetArrayFromImage(image)
        doseArray3M = np.moveaxis(doseArray3M,[axisOff+0,axisOff+1,axisOff+2],[axisOff+2,axisOff+0,axisOff+1])
        siz = doseArray3M.shape

        #Construct position matrix from ITK Image
        #pos1V = np.asarray(image.TransformIndexToPhysicalPoint((0,0,0))) / 10
        #pos2V = np.asarray(image.TransformIndexToPhysicalPoint((0,0,1))) / 10
        #deltaPosV = pos2V - pos1V
        pixelSpacing = np.asarray(image.GetSpacing()[:2]) / 10
        img_ori = np.array(image.GetDirection())
        dir_cosine_mat = img_ori.reshape(numAxes, numAxes, order="C")
        dir_cosine_mat = dir_cosine_mat[:3,:3]
        #pixelSiz = image.GetSpacing()
        dcmImgOri = dir_cosine_mat.reshape(9,order='F')[:6]
        #original_orient_str = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_ori)
        slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
                       - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]
        slice_normal = slice_normal[:,None]

        doseZValuesV = []
        for slc in range(siz[axisOff+2]):
            if numAxes == 3:
                imgPatPos = np.asarray(image.TransformIndexToPhysicalPoint((0,0,slc)))
            doseZValuesV.append(- np.sum(slice_normal * imgPatPos) / 10)

        ipp = np.asarray(image.TransformIndexToPhysicalPoint((0,0,0))) / 10
        vec3 = - slice_normal * (doseZValuesV[1]-doseZValuesV[0]) # -ve since doseZValuesV have been negated above
        position_matrix_dose = np.hstack((np.matmul(dcmImgOri.reshape(3, 2, order="F"), np.diag(pixelSpacing)),
                                       np.array([[vec3[0,0], ipp[0]],[vec3[1,0], ipp[1]], [vec3[2,0], ipp[2]]])))
        position_matrix_dose = np.vstack((position_matrix_dose, np.array([0, 0, 0, 1])))

        dose_meta = Dose()

        dose_meta.Image2PhysicalTransM = position_matrix_dose

        dose_meta.verticalGridInterval = -pixelSpacing[0]
        dose_meta.horizontalGridInterval = pixelSpacing[1]

        dose_meta.sizeOfDimension1 = siz[1]
        dose_meta.sizeOfDimension2 = siz[0]
        dose_meta.sizeOfDimension3 = siz[2]

        # to do - get refStructSetSopInstanceUID based on RTPLAN when available

        dose_meta.doseArray = doseArray3M

        # #dose_meta.refStructSetSopInstanceUID
        dose_meta.doseUID = uid.createUID("dose")

        dose_meta.assocScanUID = planC.scan[assocScanNum].scanUID

        dose_meta.convertDcmToCerrVirtualCoords(planC)

        planC.dose.append(dose_meta)

    return planC



def getDoseNumFromUID(assocDoseUID,planC) -> int:
    """

    Args:
        assocDoseUID (str): doseUID
        planC (cerr.plan_container.PlanC): pyCERR's plan container object

    Returns:
        int: index of planC.dose matching input assocDoseUID
    """

    uid_list = [s.doseUID for s in planC.dose]
    if assocDoseUID in uid_list:
        return uid_list.index(assocDoseUID)
    else:
        return None

def getPrescriptionDose(doseIdx, planC):
    """

            Args:
                doseIdx (int): Index of dose in planC
                planC (cerr.plan_container.PlanC): pyCERR's plan container object

            Returns:
                float: Prescribed dose (Gy)
    """
    planIdx = planC.dose[doseIdx].getAssociatedBeamNum(planC)
    doseRefSeq = planC.beams[planIdx].DoseReferenceSequence[0]
    if getattr(doseRefSeq,'DoseReferenceType')=='TARGET' and hasattr(doseRefSeq,'TargetPrescriptionDose'):
        RxDose = planC.beams[planIdx].DoseReferenceSequence[0].TargetPrescriptionDose
    elif hasattr(doseRefSeq,'DeliveryMaximumDose'):
        RxDose = planC.beams[planIdx].DoseReferenceSequence[0].DeliveryMaximumDose
    else:
        raise AttributeError("TargetPrescriptionDose not found.")
    return RxDose


def getFrxSize(doseIdx, planC):
    """

        Args:
            doseIdx (int): Index of dose in planC
            planC (cerr.plan_container.PlanC): pyCERR's plan container object

        Returns:
            float: Fraction size of delivered dose
        """

    # Read no. fractions
    planIdx = planC.dose[doseIdx].getAssociatedBeamNum(planC)
    numFractions = planC.beams[planIdx].FractionGroupSequence[0].NumberOfFractionsPlanned

    # Read target prescription
    RxDose = getPrescriptionDose(doseIdx, planC)

    #Calc. fraction size
    inputFrxSize = RxDose / numFractions

    return inputFrxSize


def fractionSizeCorrect(dose, stdFrxSize, abRatio, planC=None, inputFrxSize = None):
    """
        Function to convert input dose to equivalent in specified fraction size.

        Args:
                       dose (np.ndarray): Dose array  OR
                                   (int): Index to dose in planC
                        stdFrxSize (int): Desired output fraction size
                        abRatio (float) : Radiosensitivity in Gy
            planC (plan_container.planC): [optional, default=None] pyCERR's plan container object.
                                          Required when dose index specified via `dose` input.
                      inputFrxSize (int): [optional, default=None] Input fraction size.
                                          Required when dose array directly specified via `dose` input.

        Returns:
            correctedDose: Equivalent dose in specified fraction size.

        """
    if isinstance(dose, int):
        beamSOPInstances = [planC.beams[beamNum].SOPInstanceUID for beamNum in planC.beams]
        ReferencedSOPInstanceUID = planC.dose[dose].refRTPlanSopInstanceUID
        planIdx = beamSOPInstances.index(ReferencedSOPInstanceUID)
        numFractions = planC.beams(planIdx).FractionGroupSequence.NumberOfFractionsPlanned
        doseArray = planC.dose[dose].doseArray
        inputFrxSize = doseArray/numFractions
    else:
        doseArray = dose

    correctedDose = doseArray * (inputFrxSize + abRatio)/(stdFrxSize + abRatio)

    return correctedDose


def fractionNumCorrect(dose, stdFrxNum, abRatio, planC=None, inputFrxNum = None):
    """
        Function to convert input dose to equivalent in specified fraction size.

        Args:
                       dose (np.ndarray): Dose array  OR
                                   (int): Index to dose in planC
                        stdFrxNum (int): Desired output no. fractions.
                        abRatio (float) : Radiosensitivity in Gy
            planC (plan_container.planC): [optional, default=None] pyCERR's plan container object.
                                          Required when dose index specified via `dose` input.
                      inputFrxNum (int): [optional, default=None] Input no. fractions.
                                          Required when dose array directly specified via `dose` input.

        Returns:
            correctedDose: Equivalent dose in specified no. fractions.

        """
    if isinstance(dose, int):
        beamSOPInstances = [planC.beams[beamNum].SOPInstanceUID for beamNum in planC.beams]
        ReferencedSOPInstanceUID = planC.dose[dose].refRTPlanSopInstanceUID
        planIdx = beamSOPInstances.index(ReferencedSOPInstanceUID)
        inputFrxNum = planC.beams(planIdx).FractionGroupSequence.NumberOfFractionsPlanned
        doseArray = planC.dose[dose].doseArray
    else:
        doseArray = dose

    Na = inputFrxNum
    Nb = stdFrxNum
    a = Na
    b = Na * Nb * abRatio
    c = -doseArray * (b + doseArray * Nb)
    correctedDose = (-b + sqrt(b ^ 2 - 4 * a * c)) / (2 * a)

    return correctedDose

def sum(doseIndV, planC, fxCorrectDict={}):
    """

    Args:
                      doseIndV (list) : Indices of doses to be summed
          planC (plan_container.planC): pyCERR's plan container object.
                  fxCorrectDict (dict): Dictionary specifying correctionType and parameters for fractionation correction.

    Returns:
                  sumDose (np.ndarray): Summed dose
                  refGrid (list) : Coordinates of output (combined) dose grid [xOutV, yOutV, zOutV]

    """

    frxCorrectFlag = False
    fnHandle = None
    if len(fxCorrectDict) > 0:
        frxCorrectFlag = True
        # Identify fractionation correction method
        methodDict = {"fractionNum": fractionNumCorrect,
                      "fractionSize": fractionSizeCorrect}
        correctionType = fxCorrectDict.pop('correctionType')
        fnHandle = methodDict[correctionType]

    # Create shared grid from max extents of all dose grids
    numDose = len(doseIndV)
    origGridList = []
    minExtentsM = np.zeros((numDose, 3))
    maxExtentsM = np.zeros((numDose, 3))
    resM = np.zeros((numDose, 3))
    for doseNum in range(numDose):
        xV, yV, zV = planC.dose[doseIndV[doseNum]].getDoseXYZVals()
        origGridList.append((xV, yV, zV))
        minExtentsM[doseNum, :] = np.array([xV[0], yV[0], zV[0]])
        maxExtentsM[doseNum, :] = np.array([xV[-1], yV[-1], zV[-1]])
        dx = abs(np.median(np.diff(xV)))
        dy = -abs(np.median(np.diff(yV)))
        dz = abs(np.median(np.diff(zV)))
        resM[doseNum, :] = np.array([dx, dy, dz])    #Replace with dx, dy, dz
    minExtentsV = np.array([np.min(minExtentsM[:, 0]), np.max(minExtentsM[:, 1]),
                            np.min(minExtentsM[:, 2])])
    maxExtentsV = np.array([np.max(maxExtentsM[:, 0]), np.min(maxExtentsM[:, 1]),
                            np.max(maxExtentsM[:, 2])])
    outResV = np.min(resM, axis=0)
    numPoints = ((maxExtentsV - minExtentsV)/outResV).astype(int) + 1
    xOutV = np.linspace(minExtentsV[0], maxExtentsV[0], num=numPoints[0], endpoint=True)
    yOutV = np.linspace(minExtentsV[1], maxExtentsV[1], num=numPoints[1], endpoint=True)
    zOutV = np.linspace(minExtentsV[2], maxExtentsV[2], num=numPoints[2], endpoint=True)

    # Sum doses
    summedDose3M = np.zeros((numPoints[1], numPoints[0], numPoints[2]))
    for doseNum in doseIndV:

        # Get dose array and grid extents
        doseArray = planC.dose[doseNum].doseArray
        doseGrid = origGridList[doseNum]
        gridMatchFlag = ((doseGrid[0] == xOutV).all() and
                         (doseGrid[1] == yOutV).all() and
                         (doseGrid[2] == zOutV).all())

        # Fraction correct
        if frxCorrectFlag:
            # Get fraction size
            frxSize = getFrxSize(doseNum, planC)
            # Fractionation correction
            fxCorrectDict['inputFrxSize'] = frxSize
            doseArray = fnHandle(doseArray, **fxCorrectDict)

        # Resample to shared grid
        if gridMatchFlag:
            summedDose3M += doseArray
        else:
            summedDose3M += imgResample3D(doseArray, doseGrid[0], doseGrid[1], doseGrid[2],
                                          xOutV, yOutV, zOutV, 'sitkLinear', 0)

    return summedDose3M, (xOutV, yOutV, zOutV)