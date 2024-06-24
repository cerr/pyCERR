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

# Paths to data and settings
currPath = os.path.abspath(__file__)
cerrPath = os.path.join(os.path.dirname(os.path.dirname(currPath)), 'cerr')
dataPath = os.path.join(cerrPath, 'datasets', 'ibsi_radiomics_dicom', 'ibsi2_phase_2')
settingsPath = os.path.join(cerrPath, 'datasets', 'radiomics_settings',\
                            'ibsi_settings', 'ibsi2_phase_2')

# Read reference values
#--- For comparison with matlab CERR ----
#refFile = os.path.join(cerrPath, 'datasets', 'reference_values_for_tests', 'ibsi2_phase_2',
#                   'ibsi_phase2_2_cerr_features.csv')
#---- Test to ensure pyCERR calculations are consistent ----
refFile = os.path.join(cerrPath, 'datasets', 'reference_values_for_tests',\
                       'ibsi2_phase_2','ibsi_phase2_2_pycerr_features.csv')
refData = pd.read_csv(refFile)
refColNames = list(refData.head())[4:]
refFeatNames = list(refData['feature_tag'])

def load_data(niiDir):
    """ Import data to plan container"""

    # Import DICOM data
    scanFile = os.path.join(niiDir, 'phantom.nii')
    maskFile = os.path.join(niiDir, 'mask.nii')
    scanNum = 0
    planC = plan_container.load_nii_scan(scanFile, "CT SCAN")
    planC = plan_container.load_nii_structure(maskFile, scanNum, planC)

    return planC

# Load data
planC = load_data(dataPath)

def disp_diff(pctDiffFeatV, ibsiFeatList):
    """ Report on differences in feature values """
    tol = 1
    numFeats = len(pctDiffFeatV)
    if np.max(np.abs(pctDiffFeatV)) < tol:
        print('Success! ' + str(numFeats) + '/' + str(numFeats) +\
              ' match IBSI reference std.')
        print('-------------')
    else:
        checkV = np.abs(pctDiffFeatV) > tol
        idxV = np.where(checkV)[0]
        print('IBSI2 features differ:')
        diffS = dict(zip([ibsiFeatList[idx] for idx in idxV],\
                         [pctDiffFeatV[idx] for idx in idxV]))
        print(diffS)
        print('-------------')

def get_ref_feature_vals(cerrFeatS, refValsV):
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
        if 'diag' in featName:
            sepIdxV = []
        else:
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
            diffFeatV.append((cerrV[-1] - refV[-1])*100/(refV[-1] +\
                             sys.float_info.epsilon))  # pct diff
            #diffFeatV.append(cerrV[-1] - refV[-1]) # abs diff
            ibsiFeatList.append(matchName)

    pctDiffFeatV = np.asarray(diffFeatV)
    refV = np.asarray(refV)
    cerrV = np.asarray(cerrV)

    return refV, cerrV, pctDiffFeatV, ibsiFeatList

def run_config(config):
    print('Testing config '+config)
    scanNum = 0
    structNum = 0
    settingsFile = os.path.join(settingsPath, 'ibsi_phase2_2_id_' + config + '.json')
    assertTol = 0.0001

    calcFeatS, diagS = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)
    cerrFeatS = {**diagS, **calcFeatS}
    refValsV = np.array(refData[config])
    refV, cerrV, pctDiffFeatV, ibsiFeatList = get_ref_feature_vals(cerrFeatS, refValsV)
    disp_diff(pctDiffFeatV,ibsiFeatList)
    for i in range(len(refV)):
        np.testing.assert_allclose(refV[i], cerrV[i], atol=assertTol)
        #np.testing.assert_allclose(refV[i], cerrV[i], rtol=0.01) #For comparison with Matlab CERR

def test_stats_original():
    config = '1a'
    run_config(config)

def test_stats_resampled():
    config = '1b'
    run_config(config)

def test_stats_mean_2d():
    config = '2a'
    run_config(config)

def test_stats_mean_3d():
    config = '2b'
    run_config(config)

def test_stats_LoG_2d():
    config = '3a'
    run_config(config)

def test_stats_LoG_3d():
    config = '3b'
    run_config(config)

def test_stats_rot_inv_laws_energy_2d():
    config = '4a'
    run_config(config)

def test_stats_rot_inv_laws_energy_3d():
    config = '4b'
    run_config(config)

# def test_stats_gabor_2d():
#     config = '5a'
#     run_config(config)
#
# def test_stats_gabor_25d():
#    config = '5b'
#    run_config(config)

def run_phase2():
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
    run_phase2()
