"""
 Get scan no. with associated metadata matching supplied list of identifiers.
"""
import numpy as np
import datetime
from cerr.dataclasses.structure import getStructureAssociatedScan


def getScanNumForIdentifier(idS,planC,origFlag):

    # Get no. scans
    numScan = len(planC.scan)

    # Read list of identifiers
    identifierC = list(idS.keys())

    # Filter reserved fields
    resFieldsC = ['warped', 'filtered', 'resampled']
    for resField in resFieldsC:
        if resField in identifierC:
            identifierC.remove(resField)

    matchIdxV = np.ones(numScan, dtype=bool)

    # Loop over identifiers
    for identifier in identifierC:
        matchValC = idS[identifier]

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
            if matchValC == 'first':
                idV = np.array([seriesDateTimesC.index(sortedSeriesDateTimesC[0])])
            elif matchValC == 'last':
                idV = np.array([seriesDateTimesC.index(sortedSeriesDateTimesC[-1])])
            else:
                raise ValueError(f"seriesDate value '{matchValC}' is not supported.")

        elif identifier == 'studyDate':
            studyDateTimesC = [datetime.strptime(x.scanInfo[0].studyDate + ':' + \
            x.scanInfo[0].studyTime.split('.')[0], '%Y%m%d:%H%M%S') for x in planC.scan]

            sortedStudyDateTimesC = sorted(studyDateTimesC)
            if matchValC == 'first':
                idV = np.array([studyDateTimesC.index(sortedStudyDateTimesC[0])])
            elif matchValC == 'last':
                idV = np.array([studyDateTimesC.index(sortedStudyDateTimesC[-1])])
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
                    matchScan = getStructureAssociatedScan(planC.structures[strNum], planC)
                    idV &= np.isin(scanNumV, matchScan)

        else:
            raise ValueError(f"Identifier '{identifier}' not supported.")

        matchIdxV &= idV

    # Return matching scan nos.
    scanNumV = np.nonzero(matchIdxV)[0] #+ 1

    if not origFlag:
        if 'filtered' in idS and idS['filtered'] and idS:
            scanNumV = getAssocFilteredScanNum(scanNumV, planC)
        if 'warped' in idS and idS['warped'] and idS:
            scanNumV = getAssocWarpedScanNum(scanNumV, planC)
        if 'resampled' in idS and idS['resampled'] and idS:
            scanNumV = getAssocResampledScanNum(scanNumV, planC)

    return scanNumV


def getAssocFilteredScanNum(scanNumV,planC):
    """
    Return filtered scan created from input scanNum
    """
    pass

def getAssocWarpedScanNum(scanNumV,planC):
    """
    Return warped scan created from input scanNum
    """
    pass

def getAssocResampledScanNum(scanNumV,planC):
    """
    Return resampled scan created from input scanNum
    """
    pass
