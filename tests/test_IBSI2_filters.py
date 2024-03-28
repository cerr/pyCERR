"""
 This script compares convolutional filter response maps using synthetic data and filter settings from IBSI2-phase 1
 against a reference standard.

 Dataset: https://github.com/theibsi/data_sets/tree/master/ibsi_2_digital_phantom
 Configurations: https://www.overleaf.com/project/5da9e0b82f399f0001ad3970
"""

import os
import numpy as np
import scipy.io
from cerr import plan_container
from cerr.radiomics import textureUtils
import matplotlib.pyplot as plt

currPath = os.path.abspath(__file__)
cerrPath = os.path.join(os.path.dirname(os.path.dirname(currPath)),'cerr')
dataPath = os.path.join(cerrPath,'datasets','IBSIradiomicsDICOM','IBSI2Phase1')
settingsPath = os.path.join(cerrPath, 'datasets','radiomics_settings', 'IBSIsettings','IBSI2Phase1')
refPath = os.path.join(cerrPath,'datasets','referenceValuesForTests','IBSI2Phase1')

def loadData(synthDataset):
    """ Load CERR archive, return scan and mask(all ones)"""

    print('Loading dataset ' + synthDataset)
    datasetDir = os.path.join(dataPath,synthDataset)

    # Import DICOM data
    planC = plan_container.load_dcm_dir(datasetDir)

    # Extract scan
    scan3M = planC.scan[0].scanArray - planC.scan[0].scanInfo[0].CTOffset
    scan3M = scan3M.astype(float)

    # Create mask (all ones)
    scanSizeV = np.shape(scan3M)
    mask3M = np.ones(scanSizeV, dtype=bool)

    return scan3M, mask3M, planC

def compareMaps(calcMap3M, refMapName):
    """ Indicate if maps match reference, otherwise summarize differences."""

    # Load reference results
    refMapPath = os.path.join(refPath, refMapName)
    contentS = scipy.io.loadmat(refMapPath)
    refMap3M = contentS['scan3M'].astype(float)

    # Compute difference between pyCERR calculations & reference values
    diffMap3M = (calcMap3M - refMap3M)

    np.testing.assert_almost_equal(calcMap3M, refMap3M, decimal=4)

    # Report on differences
    __, __, nSlcs = np.shape(diffMap3M)
    if np.max(np.abs(diffMap3M)) < 1e-4:
        print('Success! Results match reference std.')
    else:
        print('Results differ:')
        diff_v = [np.min(diffMap3M), np.mean(diffMap3M), np.median(diffMap3M), np.max(diffMap3M)]
        print('5th percentile, mean, median, 95th percentile of differences:')
        print(diff_v)

        fig, axis = plt.subplots(3)
        midSlc = int(np.round(nSlcs / 2))
        im1 = axis[0].imshow(refMap3M[:, :, midSlc])
        cBar1 = plt.colorbar(im1, ax=axis[0])
        im2 = axis[1].imshow(calcMap3M[:, :, midSlc])
        cBar2 = plt.colorbar(im2, ax=axis[1])
        im3 = axis[2].imshow(diffMap3M[:, :, midSlc])
        cBar3 = plt.colorbar(im3, ax=axis[2])
        plt.show()
    print('-------------')


def test_phase1():
    """ Generate maps using IBSI-2 phase-1 configurations """

    ###########################
    # 1. Mean filter
    ###########################
    __, mask3M, planC = loadData('checkerboard')
    scanNum = 0

    # Config. a1
    print('Testing setting 1.a.1')
    config = '1a1'
    refMapName = '1a1.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. a2
    print('Testing setting 1.a.2')
    config = '1a2'
    refMapName = '1a2.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # a3
    print('Testing setting 1.a.3')
    config = '1a3'
    refMapName = '1a3.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # a4
    print('Testing setting 1.a.4')
    config = '1a4'
    refMapName = '1a4.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    #Config. b1
    print('Testing setting 1.b.1')
    __, mask3M, planC = loadData('impulse')
    config = '1b1'
    refMapName = '1b1.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    ########################
    # 2. LoG filter
    ########################
    # a
    print('Testing setting 2.a')
    __, mask3M, planC = loadData('impulse')
    config = '2a'
    refMapName = '2a.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)


    # Config. b
    print('Testing setting 2.b')
    __, mask3M, planC = loadData('checkerboard')
    config = '2b'
    refMapName = '2b.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. c
    print('Testing setting 2.c')
    config = '2c'
    refMapName = '2c.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    ######################
    # 3. Laws' filters
    #####################

    # Config. a1
    print('Testing setting 3.a.1')
    __, mask3M, planC = loadData('impulse')
    config = '3a1'
    refMapName = '3a1.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. a2
    print('Testing setting 3.a.2')
    config = '3a2'
    refMapName = '3a2.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. a3
    print('Testing setting 3.a.3')
    config = '3a3'
    refMapName = '3a3.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. b1
    print('Testing setting 3.b.1')
    __, mask3M, planC = loadData('checkerboard')
    config = '3b1'
    refMapName = '3b1.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. b2
    print('Testing setting 3.b.2')
    config = '3b2'
    refMapName = '3b2.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. b3
    print('Testing setting 3.b.3')
    config = '3b3'
    refMapName = '3b3.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)


    # Config. c1
    print('Testing setting 3.c.1')
    config = '3c1'
    refMapName = '3c1.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. c2
    print('Testing setting 3.c.2')
    config = '3c2'
    refMapName = '3c2.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. c3
    print('Testing setting 3.c.3')
    config = '3c3'
    refMapName = '3c3.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)


    ######################
    # 4. Gabor filter
    #####################

    # Config. a1
    print('Testing setting 4.a.1')
    __, mask3M, planC = loadData('impulse')
    config = '4a1'
    refMapName = '4a1.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. a2
    print('Testing setting 4.a.2')
    config = '4a2'
    refMapName = '4a2.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. b1
    print('Testing setting 4.b.1')
    __, mask3M, planC = loadData('sphere')
    config = '4b1'
    refMapName = '4b1.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

    # Config. b2
    print('Testing setting 4.b.2')
    config = '4b2'
    refMapName = '4b2.mat'

    # Read filter settings
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    # Calc. filter response
    planC = textureUtils.generateTextureMapFromPlanC(planC, scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()
    # Compare to reference std
    compareMaps(responseMap3M, refMapName)

if __name__ == "__main__":
    test_phase1()
