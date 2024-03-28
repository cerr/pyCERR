"""
 This script compares scalar radiomic features extracted from a lung CT
 scan using settings from IBSI1 against a reference standard.

 Dataset: https://github.com/theibsi/data_sets/tree/master/ibsi_1_ct_radiomics_phantom
 Configurations: https://ibsi.readthedocs.io/en/latest/05_Reference_data_sets.html#configurations
"""

import os
import numpy as np
import pandas as pd
from cerr import plan_container
from cerr.radiomics import ibsi1

# Define paths to data and configurations
currPath = os.path.abspath(__file__)
cerrPath = os.path.join(os.path.dirname(os.path.dirname(currPath)),'cerr')
dataPath = os.path.join(cerrPath, 'datasets', 'radiomics_phantom_dicom', 'PAT1')
settingsPath = os.path.join(cerrPath, 'datasets','radiomics_settings', 'IBSIsettings','IBSI1')

def loadData(datasetDir):
    """ Load DICOM data to CERR archive"""

    # Import DICOM data
    planC = plan_container.load_dcm_dir(datasetDir)

    return planC

def dispDiff(diffValsV,tolFeatV,refV,featList):
    """ Report on differences in feature values """
    violationV = diffValsV > tolFeatV
    diffS = ''
    pctDiffS = ''
    numFeats = len(diffValsV)
    if not any(violationV):
        print('Success! ' + str(numFeats) + '/' + str(numFeats) + ' match IBSI.')
        print('-------------')
    else:
        idxV = np.where(violationV)[0]
        print('The following features differ:')
        diffS = dict(zip([featList[idx] for idx in idxV], [diffValsV[idx] for idx in idxV]))
        print(diffS)
        print('Percentage difference:')
        pctDiffS = dict(zip([featList[idx] for idx in idxV], [diffValsV[idx]/refV[idx]*100 for idx in idxV]))
        print(pctDiffS)
        print('-------------')
    return diffS, pctDiffS

def getRefFeatureVals(cerrFeatS, refFeatNames, refValsV, tolV):
    """ Indicate if features match reference, otherwise display differences."""
    cerrFeatList = list(cerrFeatS.keys())
    numFeat = len(cerrFeatList)

    if numFeat == 0:
        raise Exception('Feature calculation failed.')

    # Loop over radiomic features computed with pyCERR
    diffFeatV = []
    refV = []
    cerrV = []
    tolFeatV = []
    ibsiFeatList = []

    for featIdx in range(numFeat):

        featName = cerrFeatList[featIdx]
        sepIdx = featName.find('_')

        # Find matching reference feature value
        matchName = featName[sepIdx+1:]
        if matchName in refFeatNames:
            matchIdx = refFeatNames.index(matchName)
            refV.append(refValsV[matchIdx])
            tolFeatV.append(tolV[matchIdx])
            cerrV.append(cerrFeatS[featName].astype(float))
            diffFeatV.append(cerrV[-1] - refV[-1])
            ibsiFeatList.append(matchName)

    diffFeatV = np.asarray(diffFeatV)
    refV = np.asarray(refV)
    cerrV = np.asarray(cerrV)
    tolFeatV = np.asarray(tolFeatV)

    return refV, cerrV, diffFeatV, tolFeatV, ibsiFeatList

def run_tests():
    """ Compute radiomics features for IBSI-1 configurations """

    # Load data
    planC = loadData(dataPath)
    scanNum = 0
    structNum = 0

    # Feature extraction settings
    configList = ['A1','A2','C1','C2']

    #A1: 2.5D, texture calc. combine across directions | A2: 2.5D, texture calc. per direction
    #B1: 2.5D, texture calc. combine across directions | B2: 2.5D, texture calc. per direction
    #C1: 3D, texture calc. combine across directions   | C2: 3D, texture calc. per direction

    # Loop over settings
    for idx in range(len(configList)):
        config = configList[idx]
        print('Testing setting ' + config)
        # Read filter settings
        settingsFile = os.path.join(settingsPath, 'IBSI1ID' + config + '.json')

        # Calc. radiomics features
        calcFeatS, diagS = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)
        #imgType = list(calcFeatS.keys())[0].split('_')[0]

        # Read reference feature values
        fileName = 'IBSI1_CERR_features_config'+ str(config[:-1]) +'.csv'
        refFile = os.path.join(cerrPath, 'datasets', 'referenceValuesForTests',\
                               'IBSI1',fileName)
        refData = pd.read_csv(refFile)
        refFeatNames = list(refData['tag'][6:])
        tolV = np.array(refData['tolerance'][6:])
        refValsV = np.array(refData['benchmark_value'][6:])

        # Get cerr and reference values
        refV, cerrV, diffFeatV, tolFeatV, ibsiFeatList = getRefFeatureVals(calcFeatS, refFeatNames, refValsV, tolV)

        dispDiff(diffFeatV,tolFeatV,refV,ibsiFeatList)

        for i in range(len(refV)):
            np.testing.assert_allclose(refV[i][0], cerrV[i][0], rtol=0, atol=tolFeatV[i][0])


if __name__ == "__main__":
    run_tests()
