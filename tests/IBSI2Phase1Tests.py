"""
 This scripts compares convolutional filter response maps on synthetic data and configurations from IBSI2-phase 1
 against a reference standard.

 Dataset: https://github.com/theibsi/data_sets/tree/master/ibsi_2_digital_phantom
 Configurations: https://www.overleaf.com/project/5da9e0b82f399f0001ad3970
"""

import os
import numpy as np
import scipy.io
from cerr import plan_container
from cerr.radiomics import preprocess, filters
import matplotlib.pyplot as plt

currPath = os.path.abspath(__file__)
dataPath = os.path.join(os.path.dirname(currPath),'dataForTests','IBSI2Phase1')
settingsPath = os.path.join(os.path.dirname(currPath),'settingsForTests','IBSI2Phase1')
refPath = os.path.join(os.path.dirname(currPath),'referenceValuesForTests','IBSI2Phase1')

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


def run_tests_phase1():
    """ Generate maps using IBSI-2 phase-1 configurations """

    # 1. Mean filter
    __, mask3M, planC = loadData('checkerboard')
    scanNum = 0

    # a1
    print('Testing setting 1.a.1')
    config = '1a1'

    # Path to settings file
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)

    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS, planC)
    paddedResponseS = filters.processImage(filterTypes[0],procScan3M, procMask3M, filterParamS)
    paddedResponse3M = paddedResponseS['mean']
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '1a1.mat'
    compareMaps(responseMap3M, refMapName)

    # a2
    print('Testing setting 1.a.2')
    config = '1a2'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS, planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    paddedResponse3M = paddedResponseS['mean']
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '1a2.mat'
    compareMaps(responseMap3M, refMapName)

    # a3
    print('Testing setting 1.a.3')
    config = '1a3'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    paddedResponse3M = paddedResponseS['mean']
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '1a3.mat'
    compareMaps(responseMap3M, refMapName)

    # a4
    print('Testing setting 1.a.4')
    config = '1a4'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    paddedResponse3M = paddedResponseS['mean']
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '1a4.mat'
    compareMaps(responseMap3M, refMapName)

    # b1
    print('Testing setting 1.b.1')
    __, mask3M, planC = loadData('impulse')

    config = '1b1'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    paddedResponse3M = paddedResponseS['mean']
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '1b1.mat'
    compareMaps(responseMap3M, refMapName)

    # 2. LoG filter
    # a
    print('Testing setting 2.a')
    __, mask3M, planC = loadData('impulse')
    scanNum = 0

    config = '2a'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    paddedResponse3M = paddedResponseS['LoG']
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '2a.mat'
    compareMaps(responseMap3M, refMapName)

    # b
    print('Testing setting 2.b')
    __, mask3M, planC = loadData('checkerboard')

    config = '2b'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    paddedResponse3M = paddedResponseS['LoG']
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '2b.mat'
    compareMaps(responseMap3M, refMapName)

    # c
    print('Testing setting 2.c')
    config = '2c'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    paddedResponse3M = paddedResponseS['LoG']
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '2c.mat'
    compareMaps(responseMap3M, refMapName)

    # 3. Laws' filters
    # a1
    print('Testing setting 3.a.1')
    __, mask3M, planC = loadData('impulse')
    scanNum = 0

    config = '3a1'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    lawsType = list(paddedResponseS.keys())[0]
    paddedResponse3M = paddedResponseS[lawsType]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '3a1.mat'
    compareMaps(responseMap3M, refMapName)

    # a2
    print('Testing setting 3.a.2')
    config = '3a2'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    lawsType = list(paddedResponseS.keys())[0]
    paddedResponse3M = paddedResponseS[lawsType]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '3a2.mat'
    compareMaps(responseMap3M, refMapName)

    # a3
    print('Testing setting 3.a.3')
    config = '3a3'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    lawsType = list(paddedResponseS.keys())[0]
    paddedResponse3M = paddedResponseS[lawsType]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '3a3.mat'
    compareMaps(responseMap3M, refMapName)

    # b1
    print('Testing setting 3.b.1')
    __, mask3M, planC = loadData('checkerboard')
    scanNum = 0

    config = '3b1'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    lawsType = list(paddedResponseS.keys())[0]
    paddedResponse3M = paddedResponseS[lawsType]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '3b1.mat'
    compareMaps(responseMap3M, refMapName)

    # b2
    print('Testing setting 3.b.2')
    config = '3b2'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    lawsType = list(paddedResponseS.keys())[0]
    paddedResponse3M = paddedResponseS[lawsType]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '3b2.mat'
    compareMaps(responseMap3M, refMapName)

    # b3
    print('Testing setting 3.b.3')
    config = '3b3'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    lawsType = list(paddedResponseS.keys())[0]
    paddedResponse3M = paddedResponseS[lawsType]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '3b3.mat'
    compareMaps(responseMap3M, refMapName)

    # 4. Gabor filter
    # a1
    print('Testing setting 4.a.1')
    __, mask3M, planC = loadData('impulse')
    scanNum = 0

    config = '4a1'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    gaborKey = list(paddedResponseS.keys())[0]
    paddedResponse3M = paddedResponseS[gaborKey]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '4a1.mat'
    compareMaps(responseMap3M, refMapName)

    # a2
    print('Testing setting 4.a.2')
    config = '4a2'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    gaborKey = list(paddedResponseS.keys())[0]
    paddedResponse3M = paddedResponseS[gaborKey]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '4a2.mat'
    compareMaps(responseMap3M, refMapName)

    # b1
    print('Testing setting 4.b.1')
    __, mask3M, planC = loadData('sphere')
    scanNum = 0

    config = '4b1'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    gaborKey = list(paddedResponseS.keys())[0]
    paddedResponse3M = paddedResponseS[gaborKey]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '4b1.mat'
    compareMaps(responseMap3M, refMapName)

    # b2
    print('Testing setting 4.b.2')
    config = '4b2'
    settingsFile = os.path.join(settingsPath,'IBSIPhase2-1ID'+ config + '.json')
    paramS, filterTypes = filters.loadSettingsFromFile(settingsFile, scanNum, planC)
    filterParamS = paramS['imageType'][filterTypes[0]]
    paddingS = paramS['settings']['padding'][0]

    procScan3M, procMask3M, __, __, __ = preprocess.preProcessForRadiomics(scanNum, mask3M, paramS,planC)
    paddedResponseS = filters.processImage(filterTypes[0], procScan3M, procMask3M, filterParamS)
    gaborKey = list(paddedResponseS.keys())[0]
    paddedResponse3M = paddedResponseS[gaborKey]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, paddingS['method'], paddingS['size'])
    refMapName = '4b2.mat'
    compareMaps(responseMap3M, refMapName)

if __name__ == "__main__":
    run_tests_phase1()
