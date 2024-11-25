"""
This module defines the container class PlanC and methods to import metadata from DICOM and NifTI formats.
"""

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List
import warnings
import SimpleITK as sitk
import numpy as np
import pandas as pd
from pydicom.uid import generate_uid
import h5py
import nibabel as nib
import cerr.dataclasses.scan_info as scn_info
from cerr.utils import uid
from cerr.contour import rasterseg as rs
from cerr.dataclasses import beams as bms
from cerr.dataclasses import dose as rtds
from cerr.dataclasses import scan as scn
from cerr.dataclasses import deform as dfrm
from cerr.dataclasses import structure as structr
from cerr.dataclasses.structure import Contour, getLabelMap, getMaskList
from cerr.dataclasses import header as headr

def get_empty_list():
    return []

@dataclass
class PlanC:
    """Plan Container for imaging metadata. Each Attribute of this class holds metadata for a particular type
    of imaging object. Additional attributes can be added to this class without breaking the functionality
    from other attributes.

    Attributes:
        header (cerr.dataclasses.header.Header): Header storing information about date of creation, version etc.
        scan (List[cerr.dataclasses.scan.Scan]): List of pyCERR's scan objects for modalities such as CT, MR, PT or derived types.
        structure (List[cerr.dataclasses.structure.Structure]): List of pyCERR's structure objects for modalities such as
            RTSTRUCT, SEG or user defined segmentation masks.
        dose (List[cerr.dataclasses.dose.Dose]): List of pyCERR's dose objects for RTDOSE modality.
        beams (List[cerr.dataclasses.beams.Beams]): List of pyCERR's beams objects for RTPLAN modality.
        deform: (List[cerr.dataclasses.deform.Deform]): List of pyCERR's deformable image registration objects.

    """

    header: headr.Header = None
    scan: List[scn.Scan] = field(default_factory=get_empty_list) #scan.Scan()
    structure: List[structr.Structure] = field(default_factory=get_empty_list) #structure.Structure()
    dose: List[rtds.Dose] = field(default_factory=get_empty_list) #dose.Dose()
    beams: List[bms.Beams] = field(default_factory=get_empty_list) #beams.Beams()
    deform: List[dfrm.Deform] = field(default_factory=get_empty_list) #beams.Beams()

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

def getSortedItems(itemList):
    if not itemList:
        return itemList
    itemList = np.array(itemList)
    return itemList[np.argsort([int(item[5:]) for item in itemList])].tolist()

def addToH5Grp(h5Grp,structDict,key):
    if isinstance(structDict[key], (str)):
        h5Grp.attrs[key] = np.string_(structDict[key])
    elif isinstance(structDict[key], (int, float, np.number)):
        h5Grp.attrs[key] = structDict[key]
    elif isinstance(structDict[key], (list, np.ndarray)):
        dt = np.array(structDict[key]).dtype
        shp = np.array(structDict[key]).shape
        h5Grp.create_dataset(key, shp, dtype=dt, data=structDict[key],
                             compression="gzip", compression_opts=4)
    return h5Grp


def saveToH5(planC, h5File, scanNumV=[], structNumV=[], doseNumV=[], deformNumV=[]):
    """Routine to serialize planC to H5 file

    Args:
        planC (cerr.plan_container.PlanC): pyCERR's plan container object
        h5File (str): File path/name to save planC
        scanNumV: Indices for scan objects from planC.scan to export to h5 file
        structNumV: Indices for structure objects from planC.structure to export to h5 file
        doseNumV: Indices for dose objects from planC.dose to export to h5 file
        deformNumV: Indices for deformation objects from planC.deform to export to h5 file

    Returns:

    """
    dt = datetime.now()
    planC.header.dateLastSaved = dt.strftime("%Y%m%d")
    with h5py.File(h5File, 'w') as f:
        planCGrp = f.create_group('planC')
        headerGrp = planCGrp.create_group('header')
        scanGrp = planCGrp.create_group('scan')
        structGrp = planCGrp.create_group('structure')
        doseGrp = planCGrp.create_group('dose')
        deformGrp = planCGrp.create_group('deform')
        headerGrp = saveH5Header(headerGrp, planC)
        scanGrp = saveH5Scan(scanGrp, scanNumV, planC)
        structGrp = saveH5Structure(structGrp, structNumV, planC)
        doseGrp = saveH5Dose(doseGrp, doseNumV, planC)
        deformGrp = saveH5Deform(deformGrp, deformNumV, planC)
    return 0

def saveNiiStructure(niiFileName, strNumV, planC, labelDict=None, dim=3):
    """
    Function to export a pyCERR's structure objects to NIfTi format mask/label map.

    Args:
        niiFileName: string specifying path to output NIfTI file.
        strNumV: list of structure indices to be exported.
        planC: pyCERR plan_container object.
        labelDict: [optional, default=None] dictionary mapping indices with structure names
        dim: [optional, default=3]. 3: writes a 3D array to nii, 4: writes 4D array to nii.

    Returns:
        0 on successful export.
    """
    if not isinstance(strNumV, list):
        strNumV = [strNumV]

    if dim == 3:
        # Export label map
        maskOut = getLabelMap(strNumV, planC, labelDict)
    elif dim == 4:
        # Export stack of binary masks
        # Required for overlapping structures
        maskList = getMaskList(strNumV, planC, labelDict=None)
        maskOut = np.array(maskList)
    else:
        raise ValueError("Invalid input. dim must be 3 (label map) or 4 (stack of binary masks)")

    maskOut = np.moveaxis(maskOut,[0,1],[1,0])

    assocScan = planC.structure[strNumV[0]].getStructureAssociatedScan(planC)

    # https://neurostars.org/t/direction-orientation-matrix-dicom-vs-nifti/14382/2
    if scn.flipSliceOrderFlag(planC.scan[assocScan]): # np.all(np.sign(zDiff) < 0):
        #if not planC.scan[scan_num].isCerrSliceOrderMatchDcm():
        maskOut = np.flip(maskOut,axis=2)

    affine3M = planC.scan[assocScan].getNiiAffine()
    strImg = nib.Nifti1Image(maskOut.astype('uint16'), affine3M)
    nib.save(strImg, niiFileName)

    return 0

def loadFromH5(h5File, initplanC=''):
    """Routine to load contents of H5 file into planC

    Args:
        h5File (str): File path/name of .h5 file containing matadata from planC
        initplanC (cerr.plan_container.PlanC): pyCERR's plan container object

    Returns:
        cerr.plan_container.PlanC: pyCERR's plan container object with metadata imported from input H5 file.

    """

    if not isinstance(initplanC, PlanC):
        planC = PlanC(header=headr.Header()) #pc.PlanC()
    else:
        planC = initplanC
        if not isinstance(planC.header, headr.Header):
            planC.header = headr.Header()

    with h5py.File(h5File, 'r') as f:
        if 'header' in f['planC']:
            headerGrp = f['planC']['header']
            planC = loadH5Header(headerGrp, planC)
        if 'scan' in f['planC']:
            scanGrp = f['planC']['scan']
            planC = loadH5Scan(scanGrp, planC)
        if 'structure' in f['planC']:
            structGrp = f['planC']['structure']
            planC = loadH5Strucutre(structGrp, planC)
        if 'dose' in f['planC']:
            doseGrp = f['planC']['dose']
            #planC = loadH5Dose(doseGrp, planC)
    return planC

def saveH5Header(headerGrp, planC):
    # Routine to write header attributes from planC to H5 group
    headerDict = planC.header.__dict__.copy()
    keys = list(headerDict.keys())
    for key in keys:
        headerGrp = addToH5Grp(headerGrp,headerDict,key)
    return headerGrp

def saveH5Scan(scanGrp, scanNumV, planC):
    # Routine to write scan attributes from planC to H5 group
    scnCount = 0
    for scanNum in scanNumV:
        scnDict = planC.scan[scanNum].getScanDict()
        itemGrpName = 'Item_' + str(scnCount)
        scnCount += 1
        scnItem = scanGrp.create_group(itemGrpName)
        keys = list(scnDict.keys())
        keys.remove('scanInfo')
        for key in keys:
            scnItem = addToH5Grp(scnItem,scnDict,key)
        # populate contour group
        scnInfoGrp = scnItem.create_group('scanInfo')
        sInfoCount = 0
        for sInfo in scnDict['scanInfo']:
            sInfoItemName = 'Item_' + str(sInfoCount)
            sInfoCount += 1
            sInfoItem = scnInfoGrp.create_group(sInfoItemName)
            keys = list(sInfo.keys())
            for key in keys:
                sInfoItem = addToH5Grp(sInfoItem,sInfo,key)
    return scanGrp


def saveH5Dose(structGrp, structNumV, planC):
    # Routine to write dose attributes from planC to H5 group
    pass

def saveH5Deform(deformGrp, deformNumV, planC):
    # Routine to write deform attributes from planC to H5 group
    deformCount = 0
    for deformNum in deformNumV:
        deformDict = planC.deform[deformNum].getDeformDict()
        itemGrpName = 'Item_' + str(deformCount)
        deformCount += 1
        deformItem = deformGrp.create_group(itemGrpName)
        keys = list(deformDict.keys())
        for key in keys:
            deformItem = addToH5Grp(deformItem,deformDict,key)
    return deformGrp

def saveH5Structure(structGrp, structNumV, planC):
    # Routine to write structure attributes from planC to H5 group
    strCount = 0
    for structNum in structNumV:
        structDict = planC.structure[structNum].getStructDict()
        itemGrpName = 'Item_' + str(strCount)
        strCount += 1
        structItem = structGrp.create_group(itemGrpName)
        keys = list(structDict.keys())
        keys.remove('contour')
        for key in keys:
            structItem = addToH5Grp(structItem,structDict,key)
        # populate contour group
        ctrGrp = structItem.create_group('contour')
        ctrCount = 0
        for ctr in structDict['contour']:
            ctrItemName = 'Item_' + str(ctrCount)
            ctrItem = ctrGrp.create_group(ctrItemName)
            if ctr:
                keys = list(ctr.keys())
                keys.remove('segments')
                for key in keys:
                    ctrItem = addToH5Grp(ctrItem,ctr,key)
                # Add segments
                segCount = 0
                for seg in ctr['segments']:
                    segGrp = ctrItem.create_group('segments')
                    segItemName = 'Item_' + str(segCount)
                    segItem = segGrp.create_group(segItemName)
                    segCount += 1
                    keys = list(seg.keys())
                    for key in keys:
                        segItem = addToH5Grp(segItem, seg, key)
            ctrCount += 1

    return structGrp

def readAttribsAndDsets(obj, h5Grp, excludeKeys=[]):
    # Routine to read object attributes into H5 group
    structFields = list(obj.__dict__.keys())
    for key in excludeKeys:
        structFields.remove(key)
    # Attributes
    h5StructAttribs = list(h5Grp.attrs.keys())
    for structAttrib in structFields:
        if structAttrib in h5StructAttribs:
            attribVal = h5Grp.attrs[structAttrib]
            if isinstance(attribVal, np.bytes_):
                attribVal = attribVal.decode('UTF-8')
            setattr(obj, structAttrib, attribVal)
    # Datasets
    h5StructDsets = list(h5Grp.keys())
    for structAttrib in structFields:
        if structAttrib in h5StructDsets:
            setattr(obj, structAttrib, h5Grp[structAttrib][:])

    return obj


def loadH5Header(headerGrp, planC):
    headerObj = headr.Header()
    planC.header = headr.Header()
    headerObj = readAttribsAndDsets(headerObj, headerGrp)
    planC.header.dateLastSaved = headerObj.dateLastSaved
    planC.header.dateCreated = headerObj.dateCreated
    return planC

def loadH5Scan(scanGrp, planC):
    scanUIDs = [s.scanUID for s in planC.scan]
    scanFieldToExclude = ['scanInfo']
    scanItems = scanGrp.keys()
    # Sort scanItems in order item_1, item_2,...
    scanItems = getSortedItems(list(scanItems))
    for scanItem in scanItems:
        scanObj = scn.Scan()
        if scanGrp[scanItem].attrs['scanUID'] in scanUIDs:
            warnings.warn("Scan " + scanGrp[scanItem].attrs['scanUID'] + " not imported from H5 as it already exists in planC")
            continue
        # populate structure field
        scanObj = readAttribsAndDsets(scanObj, scanGrp[scanItem], scanFieldToExclude)
        # populate contour field
        h5sInfoList = scanGrp[scanItem]['scanInfo'].keys()
        sInfoList = []
        # Sort h5sInfoList in order item_1, item_2,...
        h5sInfoList = getSortedItems(list(h5sInfoList))
        for siItem in h5sInfoList:
            siH5 = scanGrp[scanItem]['scanInfo'][siItem]
            siObj = scn_info.ScanInfo()
            siObj = readAttribsAndDsets(siObj, siH5)
            sInfoList.append(siObj)
        scanObj.scanInfo = sInfoList
        planC.scan.append(scanObj)

    return planC

def loadH5Deform(deformGrp, planC):
    deformUIDs = [d.deformUID for d in planC.deform]
    deformItems = deformGrp.keys()
    # Sort deformItems in order item_1, item_2,...
    scanItems = getSortedItems(list(deformItems))
    for deformItem in deformItems:
        deformObj = dfrm.Deform()
        if deformGrp[deformItem].attrs['deformUID'] in deformUIDs:
            warnings.warn("Deform " + deformGrp[deformItem].attrs['deformUID'] + " not imported from H5 as it already exists in planC")
            continue
        # populate deform fields
        deformObj = readAttribsAndDsets(deformObj, deformGrp[deformItem])
        planC.deform.append(deformObj)

    return planC


def loadH5Strucutre(structGrp, planC):
    strUIDs = [s.strUID for s in planC.structure]
    strFieldToExclude = ['contour']
    ctrFieldToExclude = ['segments']
    structItems = structGrp.keys()
    # Sort structItems in order item_1, item_2,...
    structItems = getSortedItems(list(structItems))
    for strItem in structItems:
        structObj = structr.Structure()
        if structGrp[strItem].attrs['strUID'] in strUIDs:
            warnings.warn("Structure " + structGrp[strItem].attrs['strUID'] + " not imported from H5 as it already exists in planC")
            continue
        # populate structure field
        structObj = readAttribsAndDsets(structObj, structGrp[strItem], strFieldToExclude)
        # populate contour field
        h5CtrList = structGrp[strItem]['contour'].keys()
        planCtrList = []
        # Sort h5CtrList in order item_1, item_2,...
        h5CtrList = getSortedItems(list(h5CtrList))
        for ctrItem in h5CtrList:
            ctr = structGrp[strItem]['contour'][ctrItem]
            if ctr.keys():
                ctrObj = Contour()
                ctrObj = readAttribsAndDsets(ctrObj, ctr, ctrFieldToExclude)
                segList = ctr['segments'].keys()
                planSegList = []
                # Sort segList in order item_1, item_2,...
                segList = getSortedItems(list(segList))
                for segItem in segList:
                    seg = ctr['segments'][segItem]
                    segObj = structr.Segment()
                    segObj = readAttribsAndDsets(segObj, seg)
                    planSegList.append(segObj)
                ctrObj.segments = planSegList
                planCtrList.append(ctrObj)
            else:
                planCtrList.append([])
        structObj.contour = planCtrList

        # Append to existing structure list in planc
        planC.structure.append(structObj)

    return planC


def loadDcmDir(dcmDir, opts={}, initplanC=''):
    """This routine imports metadata from DICOM directory and sub-directories into an instance of PlanC.

    Args:
        dcmDir (str): absolute path to directory containing dicom files
        opts (dict): dictionary of import options. Currently supported options are:
                     'suvType': Choose from 'BW', 'BSA', 'LBM', 'LBMJANMA'
                    e.g.  opts = {'suvType': 'LBM'}
        initplanC (PlanC): An instance of PlanC to add the metadata. If not specified, metadata is added to an empty PlanC instance

    Returns:
        PlanC: An instance of PlanC with metadata populated from DICOM files in dcm_dir

    """

    import os
    if not os.path.isdir(dcmDir):
        raise FileNotFoundError(dcmDir + 'is not a valid directory path')
    # pc.PlanC is the container to hold various dicom objects
    # Parse dcm_dir an extract a map of CT, RTSTRUCT, RTDOSE etc files to pass to populate_planC_field routine
    df_img = parseDcmHeader(dcmDir)
    #pt_groups = df_img.groupby(by=["PatientName","PatientID","Modality"])
    # Ignore fileName column from grouping
    if not isinstance(initplanC, PlanC):
        planC = PlanC(header=headr.Header()) #pc.PlanC()
    else:
        planC = initplanC
        if not isinstance(planC.header, headr.Header):
            planC.header = headr.Header()

    numOrigStructs = len(planC.structure)
    numOrigDoses = len(planC.dose)
    pt_groups = df_img.groupby(by=df_img.columns.to_list()[:-1],dropna=False)
    pt_groups.size()
    for group_name,group_content in pt_groups:
        print(group_name)
        d = group_content.to_dict()
        files = group_content.iloc[:,-1]
        modality = group_content.iloc[0,4]
        if modality in ["CT","PT", "MR"]:
            # populate scan attributes
            scan_meta = populatePlanCField('scan', files, opts)
            planC.scan.extend(scan_meta)
        elif modality in ["RTSTRUCT", "SEG"]:
            # populate structure attributes
            struct_meta = populatePlanCField('structure', files)
            planC.structure.extend(struct_meta)
        elif modality == "RTPLAN":
            # populate beams attributes
            beams_meta = populatePlanCField('beams', files)
            planC.beams.extend(beams_meta)
        elif modality == "RTDOSE":
            # populate dose attributes
            dose_meta = populatePlanCField('dose', files)
            planC.dose.extend(dose_meta)
        else:
            print("Modality " + modality + " not supported")

    # Fix duplicate scanUID in case of multiple volumes belong to the same series
    scanUIDList = np.asarray([s.scanUID for s in planC.scan])
    uniqScanUIDList = np.unique(scanUIDList)
    for uid in uniqScanUIDList:
        ids = np.where(scanUIDList == uid)[0]
        if len(ids) > 1:
            cnt = 1
            for id in ids:
                planC.scan[id].scanUID = planC.scan[id].scanUID + '.' + str(cnt)
                cnt += 1

    # Fix assocScanUID for structures
    assocScanUIDList = [s.assocScanUID for s in planC.structure]
    for structNum, uid in enumerate(assocScanUIDList):
        ids = np.where(scanUIDList == uid)[0]
        if len(ids) > 1:
            ctrList = planC.structure[structNum].contour
            sopInstanceUIDList = [c.referencedSopInstanceUID for c in ctrList if not isinstance(c,list)]
            for id in ids:
                scanSopInstanceUIDList = [s.sopInstanceUID for s in planC.scan[id].scanInfo]
                if np.sum([s in scanSopInstanceUIDList for s in sopInstanceUIDList]) >= len(sopInstanceUIDList)/2:
                    planC.structure[structNum].assocScanUID = planC.scan[id].scanUID
                    break

    # Fix assocScanUID for doses
    # assocScanUIDList = [s.assocScanUID for s in planC.dose]
    # for doseNum, uid in enumerate(assocScanUIDList):
    #     ids = np.where(scanUIDList == uid)[0]
    #     #if len(ids) > 1:
    #     planC.dose[doseNum].assocScanUID = planC.scan[ids].scanUID

    numStructs = len(planC.structure)
    numDoses = len(planC.dose)

    # Remove structures not associated with any scan
    scanUIDs = [s.scanUID for s in planC.scan]

    # Convert structure coordinates to CERR's virtual coordinates
    structsToRemove = []
    for str_num in range(numOrigStructs,numStructs):
        if planC.structure[str_num].assocScanUID not in scanUIDs:
            structsToRemove.append(str_num)
            continue
        planC.structure[str_num].convertDcmToCerrVirtualCoords(planC)
        #planC.structure[str_num].generate_rastersegs(planC) # this calls polyFill
        planC.structure[str_num].rasterSegments = rs.generateRastersegs(planC.structure[str_num], planC)
    for str_num in range(len(structsToRemove)-1,-1,-1):
        del(planC.structure[structsToRemove[str_num]])

    # Convert dose coordinates to CERR's virtual coordinates
    for dose_num in range(numOrigDoses,numDoses):
        planC.dose[dose_num].convertDcmToCerrVirtualCoords(planC)
        # Assign fraction group ID from the referenced RTPLAN
        beamNum = planC.dose[dose_num].getAssociatedBeamNum(planC)
        if beamNum is not None:
            planC.dose[dose_num].fractionGroupID = planC.beams[beamNum].RTPlanLabel
        assocScanNum = scn.getScanNumFromUID(planC.dose[dose_num].assocScanUID, planC)
        planC.dose[dose_num].cerrToDcmTransM = planC.scan[assocScanNum].cerrToDcmTransM

    return planC

    # Save planC to file
    #sio.savemat(save_file, {'planC': planC})
    #with open(save_file, 'wb') as pickle_file:
    #    pickle.dump(planC, pickle_file)

def populatePlanCField(field_name, file_list, opts={}):
    if field_name == "scan":
        scan_meta = []
        scan_meta.append(scn.loadSortedScanInfo(file_list))
        scan_meta[0].convertDcmToCerrVirtualCoords()
        scan_meta[0].convertDcmToRealWorldUnits(opts)
        if scan_meta[0].scanInfo[0].imageType == "PT SCAN":
            suvType = 'BW'
            if 'suvType' in opts:
                suvType = opts['suvType']
            elif scan_meta[0].scanInfo[0].suvType:
                suvType = scan_meta[0].scanInfo[0].suvType
            scan_meta[0].convertToSUV(suvType)
        return scan_meta

    elif field_name == "structure":
        struct_meta = structr.loadStructure(file_list)
        return struct_meta

    elif field_name == "dose":
        dose_meta = rtds.loadDose(file_list)
        return dose_meta

    elif field_name == "beams":
        beams_meta = bms.load_beams(file_list)
        return beams_meta

def loadPlanCFromPkl(file_name=""):
    file_name = r"C:\Users\aptea\PycharmProjects\pycerr\src\pycerr\tcga-ba-4074.mat"
    # Load planC from file
    #planC = sio.loadmat(file_name)
    with open(file_name, 'rb') as pickle_file:
        planC = pickle.load(pickle_file)
    return planC

def loadNiiScan(nii_file_name, imageType ="CT SCAN", direction='', initplanC=''):
    """This routine imports scan from NifTi file into planC

    Args:
        nii_file_name (str): path of NifTi file containing the scan
        imageType (str): typte of scan. e.g. 'CT SCAN', 'MR SCAN', 'PT SCAN'
        direction (str): optional, the desired orientation of scan. e.g. HFS
        initplanC (cerr.plan_container.PlanC): optional, pyCERR's plan container object to append the scan

    Returns:
        cerr.plan_container.PlanC: pyCERR's plan container object with scan imported to planC.scan[-1]

    """
    if not isinstance(initplanC, PlanC):
        planC = PlanC(header=headr.Header())
    else:
        planC = initplanC
        if not isinstance(planC.header, headr.Header):
            planC.header = headr.Header()
    reader = sitk.ImageFileReader()
    reader.SetFileName(nii_file_name)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    image = reader.Execute()
    # Assign direction
    if direction:
        image = sitk.DICOMOrient(image,direction)
    numAxes = len(image.GetSize())
    # Get numpy array for scan
    axisOff = numAxes - 3
    scanArray3M = sitk.GetArrayFromImage(image)
    scanArray3M = np.moveaxis(scanArray3M,[axisOff+0,axisOff+1,axisOff+2],[axisOff+2,axisOff+0,axisOff+1])
    #Construct position matrix from ITK Image
    #pos1V = np.asarray(image.TransformIndexToPhysicalPoint((0,0,0))) / 10
    #pos2V = np.asarray(image.TransformIndexToPhysicalPoint((0,0,1))) / 10
    #deltaPosV = pos2V - pos1V
    pixelSpacing = np.asarray(image.GetSpacing()[:2]) / 10
    img_ori = np.array(image.GetDirection())
    dir_cosine_mat = img_ori.reshape(numAxes, numAxes,order="C")
    dir_cosine_mat = dir_cosine_mat[:3,:3]
    #pixelSiz = image.GetSpacing()
    dcmImgOri = dir_cosine_mat.reshape(9,order='F')[:6]
    #original_orient_str = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_ori)
    slice_normal = dcmImgOri[[1,2,0]] * dcmImgOri[[5,3,4]] \
                   - dcmImgOri[[2,0,1]] * dcmImgOri[[4,5,3]]

    # # Transformation for DICOM Image to DICOM physical coordinates
    # # Pt coordinate to DICOM image coordinate mapping
    # # Based on ref: https://nipy.org/nibabel/dicom/dicom_orientation.html
    # position_matrix = np.hstack((np.matmul(dir_cosine_mat[:,:2],np.diag(pixelSpacing)),
    #                             np.array([[deltaPosV[0], pos1V[0]], [deltaPosV[1], pos1V[1]], [deltaPosV[2], pos1V[2]]])))
    #
    # position_matrix = np.vstack((position_matrix, np.array([0, 0, 0, 1])))

    if numAxes == 3:
        numScans = 1
    elif numAxes == 4:
        numScans = scanArray3M.shape[0]
    siz = scanArray3M.shape
    org_root = '1.3.6.1.4.1.9590.100.1.2.'
    for scanNum in range(numScans):
        scan = scn.Scan()
        forUID = generate_uid(prefix=org_root)
        studyInstanceUID = generate_uid(prefix=org_root)
        seriesInstanceUID = generate_uid(prefix=org_root)
        scan_info = np.empty(siz[-1],dtype=scn_info.ScanInfo) #scn_info.ScanInfo()
        #pixelSiz = image.GetSpacing()
        currentDate = str(date.today())
        now = datetime.now()
        currentTime = now.strftime("%H:%M:%S")
        count = 0
        for slc in range(siz[axisOff+2]):
            if numAxes == 3:
                imgPatPos = np.asarray(image.TransformIndexToPhysicalPoint((0,0,slc)))
            elif numAxes == 4:
                imgPatPos = np.asarray(image.TransformIndexToPhysicalPoint((0,0,slc,scanNum)))[:3]
            s_info = scn_info.ScanInfo()
            s_info.frameOfReferenceUID = forUID
            s_info.imagePositionPatient = imgPatPos
            s_info.imageOrientationPatient = dcmImgOri
            s_info.grid1Units = pixelSpacing[1]
            s_info.grid2Units = pixelSpacing[0]
            s_info.sizeOfDimension1 = siz[0+axisOff]
            s_info.sizeOfDimension2 = siz[1+axisOff]
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
        #original_orient_str = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_ori)
        sort_index = [i for i,x in sorted(enumerate(scan_info),key=scn.get_slice_position, reverse=False)]
        scan_info = scan_info[sort_index]
        scan_info = scn_info.deduce_voxel_thickness(scan_info)
        scan.scanInfo = scan_info
        if numAxes == 3:
            scan.scanArray = scanArray3M[:,:,sort_index]
        elif numAxes == 4:
            scan.scanArray = scanArray3M[scanNum,:,:,:][:,:,sort_index]
        scan.scanUID = "CT." + seriesInstanceUID
        scan.convertDcmToCerrVirtualCoords()
        scan.convertDcmToRealWorldUnits()
        planC.scan.append(scan)
    return planC

def loadNiiDose(nii_file_name, assocScanNum, planC, fractionGroupID = "RT-dose"):
    planC = rtds.importNii(nii_file_name, assocScanNum, planC)
    planC.dose[-1].fractionGroupID = fractionGroupID
    return planC

def loadNiiStructure(nii_file_name, assocScanNum, planC, labels_dict = {}):
    """This routine imports segmentation from NifTi file into planC

    Args:
        nii_file_name (str): path of NifTi file containing the structure/s
        assocScanNum (int): index of scan from planC.scan associated with segmentation
        planC (cerr.plan_container.PlanC): optional, pyCERR's plan container object to append structure/s
        labels_dict (dict): optional, dictionary mapping labels to structure name. e.g. {1:'GTV', 2:'Lung_total'}

    Returns:
        cerr.plan_container.PlanC: pyCERR's plan container object with structure/s imported to planC.structure

    """

    planC = structr.importNii(nii_file_name, assocScanNum, planC, labels_dict)
    # struct_meta = structr.import_nii(nii_file_name,assocScanNum,planC,labels_dict)
    # numOrigStructs = len(planC.structure)
    # planC.structure.extend(struct_meta)
    # numStructs = len(planC.structure)
    # # Convert structure coordinates to CERR's virtual coordinates
    # for str_num in range(numOrigStructs,numStructs):
    #     planC.structure[str_num].convertDcmToCerrVirtualCoords(planC)
    #     planC.structure[str_num].rasterSegments = rs.generate_rastersegs(planC.structure[str_num],planC)
    return planC


def loadNiiVf(dvf_file, baseScanNum, planC):
    """This routine loads deformation vector field from file into planC

    Args:
        dvf_file (str): path to file containing the DFV
        baseScanNum (int): index of scan from planC.scan
        planC (cerr.plan_container.PlanC): pyCERR's plan container object to append vector field

    Returns:
        cerr.plan_container.PlanC: pyCERR's plan container object with vector field imported to planC.deform

    """

    # Get image direction of the baseScanNum
    scanOrientV = planC.scan[baseScanNum].scanInfo[0].imageOrientationPatient
    slice_normal = scanOrientV[[1,2,0]] * scanOrientV[[5,3,4]] \
           - scanOrientV[[2,0,1]] * scanOrientV[[4,5,3]]
    scanOrientV = np.hstack((scanOrientV, slice_normal))
    scanOrientMat = np.asarray(scanOrientV)
    scanOrientMat.reshape(3, 3,order="F")
    dcmDirection = scanOrientMat.reshape(9,order='C')

    #origin = list(image.GetOrigin())
    #orient_nii = np.asarray(image.GetDirection())
    #orient_nii.reshape(3, 3,order="C")
    #orient_nii = np.reshape(orient_nii, (3,3), order = "C")
    #dcmImgOrient = orient_nii.reshape(9,order='F')[:6]
    #if np.max((dcmImgOrient - scanOrientV)**2) > 1e-5:
    #    #scanOrientV
    #    #scanDirection = []
    #    #image.SetDirection(scanDirection)
    #    raise Exception("nii file orientation does not match the associated scan")


    # Read 4d dvf from nii file, re-orienting to baseScanNum
    reader = sitk.ImageFileReader()
    reader.SetFileName(dvf_file)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    image = reader.Execute()
    img_ori = image.GetDirection()
    image.SetDirection(dcmDirection)
    #original_orient_str = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_ori)
    #image = sitk.DICOMOrient(image,"LPS") # temp
    dvf_matrix = sitk.GetArrayFromImage(image)
    origin = list(image.GetOrigin())
    orient = np.array(image.GetDirection()[:6])
    dim = list(image.GetSize())
    res = list(image.GetSpacing())

    deform = dfrm.Deform()
    deform.doseUID = uid.createUID("deform")

    dvf_matrix = np.moveaxis(dvf_matrix,[0,1,2,3],[2,0,1,3])
    siz = dvf_matrix.shape
    numSlcs = siz[2]
    imagePositionPatientV = np.zeros((numSlcs,3))
    zValuesV = np.zeros(numSlcs)
    for slc in range(numSlcs):
        imagePositionPatientV[slc,:] = np.asarray(image.TransformIndexToPhysicalPoint((0,0,slc)))
        zValuesV[slc] = - np.sum(slice_normal * imagePositionPatientV[slc,:]) / 10
    sort_index = np.argsort(zValuesV)
    zValuesV = zValuesV[sort_index]
    imagePositionPatientV = imagePositionPatientV[sort_index,:]
    dvf_matrix = dvf_matrix[:,:,sort_index,:]

    deform.dvfMatrix = dvf_matrix
    deform.imagePositionPatientV = imagePositionPatientV
    deform.zValuesV = zValuesV
    deform.imageOrientationPatient = orient
    deform.dx = res[0] / 10
    deform.dy = res[1] / 10

    # Convert DVF to CERR virtual
    deform.convertDcmToCerrVirtualCoords()
    planC.deform.append(deform)

    return planC


def importScanArray(scan3M, xV, yV, zV, modality, assocScanNum, planC):
    """This routine imports a scan from numpy array into planC

    Args:
        scan3M (numpy.ndarray):
        xV (numpy.ndarray):
        yV (numpy.ndarray):
        zV (numpy.ndarray):
        modality (str):
        assocScanNum (int):
        planC (cerr.plan_container.PlanC):

    Returns:
        cerr.plan_container.PlanC: pyCERR's plan container object with scan imported to planC.scan

    """


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
    currentDate = date.today().strftime('%Y%m%d')
    now = datetime.now()
    currentTime = now.strftime('%H%M%S')
    dx = xV[1] - xV[0]
    dy = yV[0] - yV[1]
    for slc in range(siz[2]):
        if xV.ndim == 2 and yV.ndim==2:
            #xV is a num_cols x num slices matrix
            #yV is a num_rows x num_slices matrix
            #with x,y values varying across slices
            dx = xV[1,slc] - xV[0,slc]
            dy = yV[0,slc] - yV[1,slc]
            cerrImgPatPos = [xV[0,slc], yV[0,slc], zV[slc], 1]
        else:
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


def importDoseArray(dose3M, xV, yV, zV, planC, assocScanNum, doseInfo=None):
    """This routine imports rt dose from numpy array into planC

    Args:
        dose3M (numpy.ndarray): 3D dose array
        xV (numpy.ndarray): x-coordinates of dose grid
        yV (numpy.ndarray): y-coordinates of dose grid
        zV (numpy.ndarray): z-coordinates of dose grid
        planC (cerr.plan_container.PlanC): pyCERR's plan container object
        assocScanNum (int): Index of scan associated with dose
        doseInfo [Optional (dict), default:None]: Dictionary specifying 'fractionGroupID', and 'units'.

    Returns:
        cerr.plan_container.PlanC: pyCERR's plan container object with scan imported to planC.scan

    """

    #Initialize dose info
    dose_meta = rtds.Dose()
    currentDate = date.today().strftime('%Y%m%d')
    opt_info = dose_meta.__dict__.keys()

    # Populate assoc. scan UID
    assocScanUID = ""
    if assocScanNum is not None:
        assocScanUID = planC.scan[assocScanNum].scanUID
    dose_meta.doseUID = uid.createUID("dose")
    dose_meta.assocScanUID = assocScanUID

    # Get grid spacing
    dx = abs(np.median(np.diff(xV)))
    dy = -abs(np.median(np.diff(yV)))
    size = dose3M.shape

    # Populate dose metadata
    dose_meta.imageType = 'DOSE'
    dose_meta.dateWritten = currentDate
    #dose_meta.frameOfReferenceUID =  ?
    dose_meta.doseArray = dose3M
    dose_meta.zValues = zV
    dose_meta.verticalGridInterval = dy
    dose_meta.horizontalGridInterval = dx
    dose_meta.sizeOfDimension1 = size[1]
    dose_meta.sizeOfDimension2 = size[0]
    dose_meta.sizeOfDimension3 = size[2]
    cerrDosePatPos = [xV[0], yV[0], zV[-1], 1]
    dcmDosePtPos = np.matmul(planC.scan[assocScanNum].cerrToDcmTransM, cerrDosePatPos)[:3]
    dose_meta.imageOrientationPatient = planC.scan[assocScanNum].scanInfo[0].imageOrientationPatient
    dose_meta.imagePositionPatient = dcmDosePtPos
    dose_meta.Image2PhysicalTransM = dose_meta.getImage2PhysicalTransM(assocScanNum, planC)
    dose_meta.convertDcmToCerrVirtualCoords(planC)


    if isinstance(doseInfo, dict):
        input_keys = doseInfo.keys()
        for input_key in input_keys:
            if input_key in opt_info:
                setattr(dose_meta, input_key, doseInfo[input_key])

    planC.dose.append(dose_meta)

    return planC


def importStructureMask(mask3M, assocScanNum, structName, planC, structNum=None):
    """

    Args:
        mask3M (numpy.ndarray): binary mask containing segmentation
        assocScanNum (int): index of scan from planC.scan associated with segmentation
        structName (str): name of structure
        planC (cerr.plan_container.PlanC): pyCERR's plan container object
        structNum (int or None): optional, index of structure object within planC.structure to replace

    Returns:
        cerr.plan_container.PlanC: pyCERR's plan container object with structure imported to planC.structure

    """

    planC = structr.importStructureMask(mask3M, assocScanNum, structName, planC, structNum)
    return planC


def parseDcmHeader(dcm_dir):
    """This routine parses header from DICOM files in the input directory

    Args:
        dcm_dir (str): path to directory containing DICOM files

    Returns:
        pandas.DataFrame: Data frame containing header values of DICOM files. The header attributes consist of
        "PatientName","PatientID","StudyInstanceUID","SeriesInstanceUID","Modality",
        "bValue","TemporalPosition","TriggerTime","NumSlices","FilePath"

    """

    from pydicom.misc import is_dicom
    from pydicom import dcmread
    import os
    from pydicom.tag import Tag
    from pydicom import dataelem

    # Patient Name, Ptient ID, StudyInstanceUID, SeriesInstanceUID,
    # Modality, b-value * 3, temporalIndex, trigger, numSlices, EchoTime
    tag_list = [Tag(0x0010,0x0010), Tag(0x0010,0x0010), Tag(0x0020,0x000D), Tag(0x0020,0x000E),
        Tag(0x0008, 0x0060), Tag(0x0043,0x1039), Tag(0x0018,0x9087), Tag(0x0019,0x100C),
        Tag(0x0020,0x0100), Tag(0x0018,0x1060), Tag(0x0021,0x104F), Tag(0x0018,0x0081)]
    tag_heading = ["PatientName","PatientID","StudyInstanceUID","SeriesInstanceUID",
                   "Modality","bValue1","bValue2","bValue3","TemporalPosition",
                   "TriggerTime","NumSlices","EchoTime","FilePath"]
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
                        else: #isinstance(val, pydicom.valuerep.PersonName):
                            val = str(val)
                    else:
                        val = ''
                    tag_vals.append(val)
                tag_vals.append(ds.filename)
                img_meta.append(tag_vals)
    df = pd.DataFrame(img_meta,columns=tag_heading)
    return df

