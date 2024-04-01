"""
 This script compares scalar radiomic features extracted from a filtered CT
 using data and  settings from IBSI2-phase 2 against a reference standard.

 Dataset: https://github.com/theibsi/data_sets/tree/master/ibsi_2_ct_radiomics_phantom
 Configurations: https://www.overleaf.com/project/5da9e0b82f399f0001ad3970
"""

import os
import numpy as np
import pandas as pd
from cerr import plan_container
from cerr.radiomics import ibsi1

# Paths to data and settings
currPath = os.path.abspath(__file__)
cerrPath = os.path.join(os.path.dirname(os.path.dirname(currPath)), 'cerr')
dataPath = os.path.join(cerrPath, 'datasets', 'IBSIradiomicsDICOM', 'IBSI2Phase2')
settingsPath = os.path.join(cerrPath, 'datasets', 'radiomics_settings', 'IBSIsettings', 'IBSI2Phase2')

# Read reference values
refFile = os.path.join(cerrPath, 'datasets', 'referenceValuesForTests', 'IBSI2Phase2',
                       'IBSIphase2-2_pyCERR_features.csv')
refData = pd.read_csv(refFile)
refColNames = list(refData.head())[4:]
# refFeatNames = refData['feature_tag'][5:]

# Features to compare
diagList = ['Number of voxels in intensity ROI-mask before interpolation', \
            'Number of voxels in intensity ROI-mask after interpolation and resegmentation', \
            'Mean intensity in intensity ROI-mask after interpolation and resegmentation', \
            'Max intensity in intensity ROI-mask after interpolation and resegmentation', \
            'Min intensity in intensity ROI-mask after interpolation and resegmentation']
featList = ['mean', 'var', 'skewness', 'kurtosis', 'median', 'min', 'P10', 'P90', 'max', \
            'interQuartileRange', 'range', 'meanAbsDev', 'robustMeanAbsDev', 'medianAbsDev', \
            'coeffVariation', 'coeffDispersion', 'energy', 'rms']

# List required output fields
outFieldC = ['mean', 'var', 'skewness', 'kurtosis', 'median', 'min', 'P10', \
             'P90', 'max', 'interQuartileRange', 'range', 'meanAbsDev', \
             'robustMeanAbsDev', 'medianAbsDev', 'coeffVariation', \
             'coeffDispersion', 'energy', 'rms']
numStats = len(outFieldC)

diagFieldC = ['NumVoxOrig', 'numVoxelsInterpReseg', \
              'MeanIntensityInterpReseg', 'MaxIntensityInterpReseg', \
              'MinIntensityInterpReseg']


def loadData(niiDir):
    """ Import data to plan container"""

    # Import DICOM data
    scanFile = os.path.join(niiDir, 'phantom.nii')
    maskFile = os.path.join(niiDir, 'mask.nii')
    scanNum = 0
    planC = plan_container.load_nii_scan(scanFile, "CT SCAN")
    planC = plan_container.load_nii_structure(maskFile, scanNum, planC)

    return planC


def dispDiff(diffValsV, type):
    """ Report on differences in feature values """
    tol = 1e-4
    if np.max(np.abs(diffValsV)) < tol:
        if type == 'feat':
            print('Success! Results match reference std.')
            print('-------------')
    else:
        checkV = np.abs(diffValsV) > tol
        idxV = np.where(checkV)[0]
        if type == 'diag':
            print('Diagnostic features differ:')
            diffS = dict(zip([diagList[idx] for idx in idxV], [diffValsV[idx] for idx in idxV]))
            print(diffS)
        elif type == 'feat':
            print('First-order features differ:')
            diffS = dict(zip([featList[idx] for idx in idxV], [diffValsV[idx] for idx in idxV]))
            print(diffS)
            print('-------------')


def compareVals(imType, calcFeatS, diagS, refValsV):
    """ Indicate if features match reference, otherwise display differences."""

    # Extract diagnostic features computed with pyCERR
    calcDiagV = np.array(list(diagS.values()))
    # Extract radiomic features computed with pyCERR
    calcFeatV = np.array([calcFeatS[imType + '_firstOrder_' + key+ '_3D'] for key in featList])
    # Extract reference diagnositc & radiomic features
    refDiagV = refValsV[0:5]
    refFeatV = refValsV[5:]

    # Compare pyCERR calculations to reference std
    diffDiagV = calcDiagV - refDiagV
    diffFeatV = calcFeatV - refFeatV

    dispDiff(diffDiagV, 'diag')
    dispDiff(diffFeatV, 'feat')

    np.testing.assert_almost_equal(calcDiagV, refDiagV, decimal=4)
    np.testing.assert_almost_equal(calcFeatV, refFeatV, decimal=4)


def test_phase2():
    """ Calc. radiomics features using IBSI-2 phase-2 configurations """

    # Load data
    planC = loadData(dataPath)
    scanNum = 0
    structNum = 0

    configList = ['1a','1b','2a','2b','3a','3b','4a','4b','5a','5b']
    # TBD:'6a', '6b'. Wavelets not currently supported.

    # Loop over configurations
    for idx in range(len(configList)):
        config = configList[idx]
        colID = refColNames[idx]

        print('Testing setting ' + config)
        # Read filter settings
        settingsFile = os.path.join(settingsPath, 'IBSIPhase2-2ID' + config + '.json')

        # Calc. radiomics features
        calcFeatS, diagS = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)

        imType = list(calcFeatS.keys())[0].split('_')[0]

        # Compare to reference std
        refValsV = np.array(refData[colID])
        compareVals(imType, calcFeatS, diagS, refValsV)


if __name__ == "__main__":
    test_phase2()
