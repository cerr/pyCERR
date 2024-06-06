"""
Function required to build/deploy IA models via pyCERR.
"""
import os
import random
import numpy as np
from datetime import datetime
import cerr.dataclasses.scan as cerrScn

def createSessionDir(sessionPath, inputDicomPath):
    # Create session directory for temporary files

    if inputDicomPath[-1] == os.sep:
        _, folderNam = os.path.split(inputDicomPath[:-1])
    else:
        _, folderNam = os.path.split(inputDicomPath)

    dateTimeV = datetime.now().timetuple()[:6]
    randStr = f"{random.random() * 1000:.3f}"
    sessionDir = f"session{folderNam}{dateTimeV[3]}{dateTimeV[4]}{dateTimeV[5]}{randStr}"
    fullSessionPath = os.path.join(sessionPath, sessionDir)

    # Create sub-directories
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

    return modInputPath, modOutputPath


def getAssocFilteredScanNum(scanNumV,planC):
    """

    Args:
        scanNumV (list): list of scan indices in planC
        planC: pyCERR's plan container object

    Returns:
        list: list of filtered scan indices in planC created from input list of scan numbers

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
    Return warped scan created from input scanNum
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
    Return resampled scan created from input scanNum
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
    '''
    Return index of scan with metadata matching user-input identifier(s).
    Supported identifiers:
    :param idDict:  Parameter dictionary with keys specifying identifiers, and values holding expected quantity.
                    Supported identifiers include: 'imageType', 'seriesDescription', 'scanNum', 'scanType',
                    'seriesDate' (may be" first" or "last"), 'studyDate' (may be" first" or "last"),
                    and 'assocStructure' (use structure name to identify associated scans. Set to 'none'
                    to select scans with no associated structures)
    :param planC
    ----- Optional-----
    :param origFlag: Set to True to ignore 'warped', 'resampled' or 'filtered' scans (default:False).
    :return: scanNumV : Vector of scan indices matching specified identifier(s).
    '''

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
