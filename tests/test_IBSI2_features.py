"""
 This script compares scalar radiomic features extracted from a filtered CT
 using data and  settings from IBSI2-phase 2 against a reference standard.

 Dataset: https://github.com/theibsi/data_sets/tree/master/ibsi_2_ct_radiomics_phantom
 Configurations: https://www.overleaf.com/project/5da9e0b82f399f0001ad3970
"""

import os
import sys
import numpy as np
import pandas as pd
from cerr import plan_container
from cerr.radiomics import ibsi1

def loadData(niiDir):
    """ Import data to plan container"""

    # Import DICOM data
    scanFile = os.path.join(niiDir, 'phantom.nii')
    maskFile = os.path.join(niiDir, 'mask.nii')
    scanNum = 0
    planC = plan_container.load_nii_scan(scanFile, "CT SCAN")
    planC = plan_container.load_nii_structure(maskFile, scanNum, planC)

    return planC

def dispDiff(pctDiffFeatV, ibsiFeatList):
    """ Report on differences in feature values """
    tol = 1
    if np.max(np.abs(pctDiffFeatV)) < tol:
        print('Success! Results match reference std.')
        print('-------------')
    else:
        checkV = np.abs(pctDiffFeatV) > tol
        idxV = np.where(checkV)[0]
        print('IBSI2 features differ:')
        diffS = dict(zip([ibsiFeatList[idx] for idx in idxV], [pctDiffFeatV[idx] for idx in idxV]))
        print(diffS)
        print('-------------')

def getRefFeatureVals(cerrFeatS, refValsV):
    """ Indicate if features match reference, otherwise display differences."""

    cerrFeatList = list(cerrFeatS.keys())
    numFeat = len(cerrFeatList)

    if numFeat == 0:
        raise Exception('Feature calculation failed.')

    # Loop over radiomic features computed with pyCERR
    diffFeatV = []
    refV = []
    cerrV = []
    ibsiFeatList = []

    for featIdx in range(numFeat):

        featName = cerrFeatList[featIdx]
        sepIdxV = [idx for idx, s in enumerate(featName) if '_' in s]

        # Find matching reference feature value
        if len(sepIdxV)==0:
            matchName = featName
        else:
            matchName = featName[sepIdxV[-2]+1:]

        if matchName in refFeatNames:
            matchIdx = refFeatNames.index(matchName)
            refV.append(refValsV[matchIdx])
            cerrV.append(float(cerrFeatS[featName]))
            diffFeatV.append((cerrV[-1] - refV[-1])*100/(refV[-1] + sys.float_info.epsilon))  # pct diff
            #diffFeatV.append(cerrV[-1] - refV[-1]) # abs diff
            ibsiFeatList.append(matchName)

    pctDiffFeatV = np.asarray(diffFeatV)
    refV = np.asarray(refV)
    cerrV = np.asarray(cerrV)

    return refV, cerrV, pctDiffFeatV, ibsiFeatList

def test_calc_features(config):
    print('Testing config '+config)
    scanNum = 0
    structNum = 0
    settingsFile = os.path.join(settingsPath, 'IBSIPhase2-2ID' + config + '.json')
    tol = 10^-4

    calcFeatS, diagS = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)
    cerrFeatS = {**diagS, **calcFeatS}
    refValsV = np.array(refData[config])
    refV, cerrV, pctDiffFeatV, ibsiFeatList = getRefFeatureVals(cerrFeatS, refValsV)
    dispDiff(pctDiffFeatV,ibsiFeatList)
    for i in range(len(refV)):
        np.testing.assert_allclose(refV[i], cerrV[i], rtol=tol, atol=tol)
        #np.testing.assert_allclose(refV[i], cerrV[i], rtol=0.01) #For comparison with Matlab CERR

def test_stats_original():
    config = '1a'
    test_calc_features(config)

def test_stats_resampled():
    config = '1b'
    test_calc_features(config)

def test_stats_mean_2d():
    config = '2a'
    test_calc_features(config)

def test_stats_mean_3d():
    config = '2b'
    test_calc_features(config)

def test_stats_LoG_2d():
    config = '3a'
    test_calc_features(config)

def test_stats_LoG_3d():
    config = '3b'
    test_calc_features(config)

def test_stats_rot_inv_laws_energy_2d():
    config = '4a'
    test_calc_features(config)

def test_stats_rot_inv_laws_energy_3d():
    config = '4b'
    test_calc_features(config)

# def test_stats_gabor_2d():
#     config = '5a'
#     test_calc_features(config)
#
# def test_stats_gabor_25d():
#    config = '5b'
#    test_calc_features(config)

def test_phase2():
    """ Calc. radiomics features using IBSI-2 phase-2 configurations """

    test_stats_original()

    test_stats_resampled()

    test_stats_mean_2d()

    test_stats_mean_3d()

    test_stats_LoG_2d()

    test_stats_LoG_3d()

    test_stats_rot_inv_laws_energy_2d()

    test_stats_rot_inv_laws_energy_3d()

    #Commented out due to long runtime
    #test_stats_gabor_2d()

    #test_stats_gabor_25d()


if __name__ == "__main__":

    # Paths to data and settings
    currPath = os.path.abspath(__file__)
    cerrPath = os.path.join(os.path.dirname(os.path.dirname(currPath)), 'cerr')
    dataPath = os.path.join(cerrPath, 'datasets', 'IBSIradiomicsDICOM', 'IBSI2Phase2')
    settingsPath = os.path.join(cerrPath, 'datasets', 'radiomics_settings', 'IBSIsettings', 'IBSI2Phase2')

    # Read reference values
    #--- For comparison with matlab CERR ----
    #refFile = os.path.join(cerrPath, 'datasets', 'referenceValuesForTests', 'IBSI2Phase2',
    #                   'IBSIphase2-2_CERR_features.csv')
    #---- Test to ensure pyCERR calculations are consistent ----
    refFile = os.path.join(cerrPath, 'datasets', 'referenceValuesForTests', 'IBSI2Phase2',
                       'IBSIphase2-2_pyCERR_features.csv')
    refData = pd.read_csv(refFile)
    refColNames = list(refData.head())[4:]
    refFeatNames = list(refData['feature_tag'])

    # Load data
    planC = loadData(dataPath)
    test_phase2()
