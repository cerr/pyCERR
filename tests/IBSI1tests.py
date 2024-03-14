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

# Read reference feature values
refFile = os.path.join(cerrPath, 'datasets', 'referenceValuesForTests', 'IBSI1',
                       'IBSI1_CERR_features.csv')
refData = pd.read_csv(refFile)
refFeatNames = list(refData['tag'][6:])
tolV = np.array(refData['tolerance_compare'][6:])
refValsV = np.array(refData['benchmark_value_compare'][6:])
# Note: The columns "benchmark_value_compare" and "tolerance_comapre" contain results in
# units of cm (rather than mm) where applicable, to simplify comparison with pyCERR.

def loadData(datasetDir):
    """ Load DICOM data to CERR archive"""

    # Import DICOM data
    planC = plan_container.load_dcm_dir(datasetDir)

    return planC

def dispDiff(diffValsV,tolFeatV,featList):
    """ Report on differences in feature values """
    violationV = diffValsV > tolFeatV
    if not any(violationV):
            print('Success! Results match reference std.')
            print('-------------')
    else:
        idxV = np.where(violationV)[0]
        print('First-order features differ:')
        diffS = dict(zip([featList[idx] for idx in idxV], [diffValsV[idx] for idx in idxV]))
        print(diffS)
        print('-------------')


def compareVals(cerrFeatS, refFeatNames):
    """ Indicate if features match reference, otherwise display differences."""
    cerrFeatList = list(cerrFeatS.keys())
    numFeat = len(cerrFeatList)

    # Loop over radiomic features computed with pyCERR
    diffFeatV = np.zeros((numFeat,1))
    ibsiFeatList = [None]*numFeat
    tolFeatV = np.zeros((numFeat,1))
    for featIdx in range(numFeat):

        featName = cerrFeatList[featIdx]
        sepIdx = featName.find('_')

        # Find matching reference feature value
        matchName = featName[sepIdx+1:]
        matchName = matchName.replace('_3D','')
        matchIdx = refFeatNames.index(matchName) if matchName in refFeatNames else None
        ibsiFeatList[featIdx] = matchName

        # Computer deviation from reference value
        if matchIdx is None:
            diffFeatV[featIdx] = float("nan")
        else:
            refVal = refValsV[matchIdx]
            tolFeatV[featIdx] = tolV[matchIdx]
            diffFeatV[featIdx] = cerrFeatS[featName].astype(float) - refVal

    dispDiff(diffFeatV,tolFeatV,ibsiFeatList)

def run_tests():
    """ Compute radiomics features for IBSI-1 configurations """

    # Load data
    planC = loadData(dataPath)
    scanNum = 0
    structNum = 0

    # Feature extraction settings
    configList = ['A1','A2','C1','C2']

    # Loop over settings
    for idx in range(len(configList)):
        config = configList[idx]
        print('Testing setting ' + config)
        # Read filter settings
        settingsFile = os.path.join(settingsPath, 'IBSI1ID' + config + '.json')

        # Calc. radiomics features
        calcFeatS, diagS = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)
        #imgType = list(calcFeatS.keys())[0].split('_')[0]

        # Compare to reference std
        compareVals(calcFeatS, refFeatNames)


if __name__ == "__main__":
    run_tests()
