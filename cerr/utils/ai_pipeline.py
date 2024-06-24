"""
This module defines routines useful to build and deploy AI models
"""

import os
import random
import numpy as np
from datetime import datetime
import cerr.dataclasses.scan as cerrScn

def createSessionDir(sessionPath, inputDicomPath, inputSubDirs=None):
    """
    Function to create a directory to write temporary files when deploying AI models.
    inputDicomPath

    Args:
        sessionPath (string): Desired location of session directory
        inputDicomPath (string): Path to DICOM input data. The session directory is assigned a unique name
                                 derived in aprt from the DICOM folder name.
        inputSubDirs (list): [optional, default=None] Create subdirectories for inputs other than scans.

    Returns:
        modInputPath (string): Path to directory containing data input to AI model
        modOutputPath (string): Path to directory containing data output by AI model
    """
    
    if inputDicomPath[-1] == os.sep:
        _, folderNam = os.path.split(inputDicomPath[:-1])
    else:
        _, folderNam = os.path.split(inputDicomPath)

    dateTimeV = datetime.now().timetuple()[:6]
    randStr = f"{random.random() * 1000:.3f}"
    sessionDir = f"session{folderNam}{dateTimeV[3]}{dateTimeV[4]}{dateTimeV[5]}{randStr}"
    fullSessionPath = os.path.join(sessionPath, sessionDir)

    # Create default sub-directories
    if not os.path.exists(sessionPath):
        os.mkdir(sessionPath)
    os.mkdir(fullSessionPath)
    cerrPath = os.path.join(fullSessionPath, "dataCERR")
    os.mkdir(cerrPath)
    modInputPath = os.path.join(fullSessionPath, "inputNii")
    os.mkdir(modInputPath)
    modOutputPath = os.path.join(fullSessionPath, "outputNii")
    os.mkdir(modOutputPath)
    AIoutputPath = os.path.join(fullSessionPath, "AIoutput")
    os.mkdir(AIoutputPath)

    # Create optional input sub-dirs
    if inputSubDirs is not None:
        for subDir in inputSubDirs:
            os.mkdir(os.path.join(modInputPath,subDir))

    return modInputPath, modOutputPath


def getAssocFilteredScanNum(scanNumV,planC):
    """
    Function to return index of filtered scan derived from original input scan.

    Args:
        scanNumV (list): Original scan indices in planC
        planC (plan_container.planC): pyCERR's plan container object

    Returns:
        filtScanNumV (list): Filtered scan indices in planC created from input list
                             of scans.
    """

    scanUIDs = [planC.scan[scanNum].scanUID for scanNum in scanNumV]

    filtScanNumV = np.full(len(scanNumV), np.nan)  # Initialize

    for nScan in range(len(planC.scan)):
        baseScanUID = planC.scan[nScan].assocBaseScanUID
        assocIdxV = [uid == baseScanUID for uid in scanUIDs]

        if any(assocIdxV):
            filtScanNumV[np.where(scanNumV == np.where(assocIdxV))] = nScan

    filtScanNumV = filtScanNumV[~np.isnan(filtScanNumV)]
    return filtScanNumV


def getAssocWarpedScanNum(scanNumV,planC):
    """
        Function to return index of deformed scan derived from original input scan.

        Args:
            scanNumV (list): Original scan indices in planC
            planC (plan_container.planC): pyCERR's plan container object

        Returns:
            warpedScanNumV (list): Warped scan indices in planC created from input list
                                   of scans.
    """
    assocMovScanUIDs = [planC.scan[scanNum].assocMovingScanUID for scanNum in scanNumV]

    warpedScanNumV = np.full(len(scanNumV),np.nan)
    for nScan in range(len(scanNumV)):
        movScanUID = planC.scan[scanNumV(nScan)].scanUID
        assocIdxV = [uid == movScanUID for uid in assocMovScanUIDs]
        if any(assocIdxV):
            warpedScanNumV[nScan] = np.where(assocIdxV)

    warpedScanNumV = warpedScanNumV[~np.isnan(warpedScanNumV)]
    return warpedScanNumV

def getAssocResampledScanNum(scanNumV,planC):
    """
        Function to return index of resampled scan derived from original input scan.

        Args:
            scanNumV (list): Original scan indices in planC
            planC (plan_container.planC): pyCERR's plan container object

        Returns:
            warpedScanNumV (list): Resampled scan indices in planC created from input list
                                   of scans.
    """

    scanUIDs = [planC.scan[scanNum].scanUID for scanNum in scanNumV]
    resampScanNumV = np.full(len(scanNumV),np.nan)

    for nScan in range(len(planC.scan)):
        baseScanUID = planC.scan[nScan].assocBaseScanUID
        assocIdxV = [uid == baseScanUID for uid in scanUIDs]
        if any(assocIdxV):
            resampScanNumV[scanNumV==np.where(assocIdxV)] = nScan


    resampScanNumV = resampScanNumV[~np.isnan(resampScanNumV)]
    return resampScanNumV

def getScanNumFromIdentifier(idDict,planC,origFlag:bool = False):
    """ 
    Function to retrieve index of scan with metadata matching user-input identifier(s).

    Args:
        idDict (dictionary): Scan identifiers specified in keys specifying,
                             with corresponding values specifying expected quantity.
                             Supported identifiers include: 'imageType', 'seriesDescription', 'scanNum', 'scanType',
                             'seriesDate' (may be" first" or "last"), 'studyDate' (may be" first" or "last"),
                             and 'assocStructure' (use structure name to identify associated scans. Set to 'none'
                             to select scans with no associated structures)
        planC (plan_container.planC): pyCERR's plan container object
        origFlag (bool): [optional, default:False] Flag to ignore 'warped', 'resampled' or 'filtered'
                         scans.

    Returns:
        scanNumV (np.array) : Scan indices matching specified identifier(s).
    """

    # Get no. scans
    numScan = len(planC.scan)

    # Read list of identifiers
    identifierC = list(idDict.keys())

    # Filter reserved fields
    resFieldsC = ['warped', 'filtered', 'resampled']
    for resField in resFieldsC:
        if resField in identifierC:
            identifierC.remove(resField)

    matchIdxV = np.ones(numScan, dtype=bool)

    # Loop over identifiers
    for identifier in identifierC:
        matchValC = idDict[identifier]

        # Match against metadata in planC
        if identifier == 'imageType':
            imTypeC = [x.scanInfo[0].imageType for x in planC.scan]
            idV = np.array([matchValC.lower() == imType.lower() for imType in imTypeC])

        elif identifier == 'seriesDescription':
            seriesDescC = [str(x.scanInfo[0].seriesDescription) for x in planC.scan]
            idV = np.array([matchValC in seriesDesc for seriesDesc in seriesDescC])

        elif identifier == 'scanType':
            scanTypeC = [x.scanType for x in planC.scan]
            idV = np.array([matchValC.lower() == scanType.lower() for scanType in scanTypeC])

        elif identifier == 'scanNum':
            idV = np.array([scanNum in matchValC for scanNum in range(1, numScan + 1)])

        elif identifier == 'seriesDate':
            seriesDateTimesC = [datetime.strptime(x.scanInfo[0].seriesDate + ':' + \
            x.scanInfo[0].seriesTime.split('.')[0], '%Y%m%d:%H%M%S') for x in planC.scan]

            sortedSeriesDateTimesC = sorted(seriesDateTimesC)
            idV = np.zeros(numScan,dtype='bool')
            if matchValC == 'first':
                idV[np.array(seriesDateTimesC.index(sortedSeriesDateTimesC[0]))] = True
            elif matchValC == 'last':
                idV[np.array(seriesDateTimesC.index(sortedSeriesDateTimesC[-1]))] = True
            else:
                raise ValueError(f"seriesDate value '{matchValC}' is not supported.")

        elif identifier == 'studyDate':
            studyDateTimesC = [datetime.strptime(x.scanInfo[0].studyDate + ':' + \
            x.scanInfo[0].studyTime.split('.')[0], '%Y%m%d:%H%M%S') for x in planC.scan]

            sortedStudyDateTimesC = sorted(studyDateTimesC)
            idV = np.zeros(numScan,dtype='bool')
            if matchValC == 'first':
                idV[np.array(studyDateTimesC.index(sortedStudyDateTimesC[0]))] = True
            elif matchValC == 'last':
                idV[np.array(studyDateTimesC.index(sortedStudyDateTimesC[-1]))] = True
            else:
                raise ValueError(f"studyDate value '{matchValC}' is not supported.")

        elif identifier == 'assocStructure':
            if matchValC == 'none':
                strAssocScanV = np.unique([struc.associatedScan for struc in planC.structures])
                idV = ~np.isin(range(1, numScan + 1), strAssocScanV)
            else:
                idV = np.ones(numScan, dtype=bool)
                scanNumV = np.arange(1, numScan + 1)
                for matchVal in matchValC:
                    strListC = [str(struc.structureName) for struc in planC.structures]
                    strNum = strListC.index(matchVal)
                    matchScan = cerrScn.getScanNumFromUID(planC.structures[strNum].assocScanUID, planC)
                    idV &= np.isin(scanNumV, matchScan)

        else:
            raise ValueError(f"Identifier '{identifier}' not supported.")

        matchIdxV &= idV

    # Return matching scan nos.
    scanNumV = np.nonzero(matchIdxV)[0]

    if not origFlag:
        if 'filtered' in idDict and idDict['filtered'] and idDict:
            scanNumV = getAssocFilteredScanNum(scanNumV, planC)
        if 'warped' in idDict and idDict['warped'] and idDict:
            scanNumV = getAssocWarpedScanNum(scanNumV, planC)
        if 'resampled' in idDict and idDict['resampled'] and idDict:
            scanNumV = getAssocResampledScanNum(scanNumV, planC)

    return scanNumV
