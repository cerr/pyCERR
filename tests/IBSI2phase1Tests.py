"""
 This scripts compares convolutional filter response maps on synthetic data and configurations from IBSI2-phase 1
 against a reference standard.

 Dataset: https://github.com/theibsi/data_sets/tree/master/ibsi_2_digital_phantom
 Configurations: https://www.overleaf.com/project/5da9e0b82f399f0001ad3970
"""

import os
import numpy as np
import scipy.io
import sys
from cerr import plan_container
from cerr.radiomics import preprocess, filters
import matplotlib.pyplot as plt

currPath = os.path.abspath(__file__)
dataPath = os.path.join(os.path.dirname(currPath),'dataForTests','IBSI2Phase1')
refPath = os.path.join(os.path.dirname(currPath),'referenceValuesForTests','IBSI2Phase1')

def loadData(synthDataset):
    """ Load CERR archive, return scan and mask(all ones)"""

    print('Loading dataset ' + synthDataset)
    # Replace with path to DICOM data for IBSI phase 2
    datasetDir = os.path.join(dataPath,synthDataset)

    # Import DICOM data
    #empty_planC = plan_container.PlanC()
    planC = plan_container.load_dcm_dir(datasetDir)
    # Extract scan
    scan3M = planC.scan[0].scanArray - planC.scan[0].scanInfo[0].CTOffset
    scan3M = scan3M.astype(float)
    # scan3M = np.flip(scan3M, axis=2) # flip SI

    # Create mask (all ones)
    scanSizeV = np.shape(scan3M)
    mask3M = np.ones(scanSizeV)

    return scan3M, mask3M, planC


def compareMaps(calcMap3M, refMapName):
    """ Indicate if maps match reference, otherwise summarize differences."""

    # Replace with path to reference results (.mat) for IBSI phase 2
    refMapPath = os.path.join(refPath, refMapName)
    contentS = scipy.io.loadmat(refMapPath)
    refMap3M = contentS['scan3M'].astype(float)

    # refMap3M = np.transpose(refMap3M,(2,1,0))
    diffMap3M = (calcMap3M - refMap3M) #*100./(refMap3M + sys.float_info.epsilon)
    __, __, nSlcs = np.shape(diffMap3M)
    if np.max(np.abs(diffMap3M)) < 1e-5:
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
    scan3M, mask3M, __ = loadData('checkerboard')

    # a1
    print('Testing setting 1.a.1')
    padMethod = 'padzeros'
    padSizeV = np.array([7, 7, 7])
    kernelSizeV = np.array([15, 15, 15])
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.meanFilter(procScan3M, kernelSizeV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '1a1.mat'
    compareMaps(responseMap3M, refMapName)

    # a2
    print('Testing setting 1.a.2')
    padMethod = 'nearest'
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.meanFilter(procScan3M, kernelSizeV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '1a2.mat'
    compareMaps(responseMap3M, refMapName)

    # a3
    print('Testing setting 1.a.3')
    padMethod = 'periodic'
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.meanFilter(procScan3M, kernelSizeV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '1a3.mat'
    compareMaps(responseMap3M, refMapName)

    # a4
    print('Testing setting 1.a.4')
    padMethod = 'mirror'
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.meanFilter(procScan3M, kernelSizeV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '1a4.mat'
    compareMaps(responseMap3M, refMapName)

    # b1
    print('Testing setting 1.b.1')
    scan3M, mask3M, __ = loadData('impulse')
    padMethod = 'padzeros'
    padSizeV = np.array([7, 7])
    kernelSizeV = np.array([15, 15])
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.meanFilter(procScan3M, kernelSizeV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '1b1.mat'
    compareMaps(responseMap3M, refMapName)

    # 2. LoG filter

    # a1
    print('Testing setting 2.a.1')
    scan3M, mask3M, planC = loadData('impulse')
    voxelSizemmV = planC.scan[0].getScanSpacing() * 10

    padMethod = 'padzeros'
    padSizeV = np.array([6, 6, 6])
    cutoffmmV = np.array([12, 12, 12])
    sigmammV = np.array([3, 3, 3])
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.LoGFilter(procScan3M, sigmammV, cutoffmmV, voxelSizemmV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '2a1.mat'
    compareMaps(responseMap3M, refMapName)

    # b1
    print('Testing setting 2.b.1')
    scan3M, mask3M, planC = loadData('checkerboard')
    voxelSizemmV = planC.scan[0].getScanSpacing() * 10

    padMethod = 'mirror'
    padSizeV = np.array([10, 10, 10])
    cutoffmmV = np.array([20, 20, 20])
    sigmammV = np.array([5, 5, 5])
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.LoGFilter(procScan3M, sigmammV, cutoffmmV, voxelSizemmV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '2b1.mat'
    compareMaps(responseMap3M, refMapName)

    # c1
    print('Testing setting 2.c.1')
    padMethod = 'mirror'
    padSizeV = np.array([10, 10])
    cutoffmmV = np.array([20, 20])
    sigmammV = np.array([5, 5])
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.LoGFilter(procScan3M, sigmammV, cutoffmmV, voxelSizemmV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '2c1.mat'
    compareMaps(responseMap3M, refMapName)

    # 3. Laws' filters
    # a1
    print('Testing setting 3.a.1')
    scan3M, mask3M, __ = loadData('impulse')

    padMethod = 'padzeros'
    padSizeV = np.array([2, 2, 2])
    filterDim = 'E5L5S5'
    direction = '3d'
    normFlag = True
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    responseS = filters.lawsFilter(procScan3M, direction, filterDim, normFlag)
    paddedResponse3M = responseS[filterDim]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '3a1.mat'
    compareMaps(responseMap3M, refMapName)

    # a2
    print('Testing setting 3.a.2')
    rot = {'aggregationMethod': 'max', 'dim': '3d'}
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.rotationInvariantLawsFilter(procScan3M, direction, filterDim, normFlag, rot)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '3a2.mat'
    compareMaps(responseMap3M, refMapName)

    # a3
    print('Testing setting 3.a.3')
    lawsPadMethod = 'padzeros'
    lawsPadSizeV = np.array([2, 2, 2])
    energyKernelSizeV = np.array([15, 15, 15])
    energyPadSizeV = np.array([7, 7, 7])
    energyPadMethod = 'padzeros'
    rot = {'aggregationMethod': 'max', 'dim': '3d'}
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, lawsPadMethod, lawsPadSizeV)
    paddedResponse3M = filters.rotationInvariantLawsEnergyFilter(procScan3M, direction, filterDim, normFlag,
                                                                 lawsPadSizeV,
                                                                 energyKernelSizeV, energyPadSizeV,
                                                                 energyPadMethod, rot)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, lawsPadSizeV, lawsPadSizeV)
    refMapName = '3a3.mat'
    compareMaps(responseMap3M, refMapName)

    #  b1
    print('Testing setting 3.b.1')
    scan3M, mask3M, __ = loadData('checkerboard')
    padMethod = 'mirror'
    padSizeV = np.array([2, 1, 2])
    filterDim = 'E3W5R5'
    direction = '3d'
    normFlag = True
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    response = filters.lawsFilter(procScan3M, direction, filterDim, normFlag)
    paddedResponse3M = response[filterDim]
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '3b1.mat'
    compareMaps(responseMap3M, refMapName)

    # b2
    print('Testing setting 3.b.2')
    padMethod = 'mirror'
    padSizeV = np.array([2, 2, 2])
    rot = {'aggregationMethod': 'max', 'dim': '3d'}
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.rotationInvariantLawsFilter(procScan3M, direction, filterDim, normFlag, rot)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '3b2.mat'
    compareMaps(responseMap3M, refMapName)

    # b3
    print('Testing setting 3.b.3')
    padMethod = 'mirror'
    padSizeV = np.array([2, 2, 2])
    energyKernelSizeV = np.array([15, 15, 15])
    energyPadSizeV = np.array([7, 7, 7])
    energyPadMethod = 'mirror'
    rot = {'aggregationMethod': 'max', 'dim': '3d'}
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M = filters.rotationInvariantLawsEnergyFilter(procScan3M, direction, filterDim, normFlag,
                                                                 padSizeV, energyKernelSizeV,
                                                                 energyPadSizeV, energyPadMethod, rot)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '3b3.mat'
    compareMaps(responseMap3M, refMapName)

    # 4. Gabor filter
    # a1
    print('Testing setting 4.a.1')
    scan3M, mask3M, planC = loadData('impulse')
    voxelSizemmV = planC.scan[0].getScanSpacing() * 10

    sigmamm = 10
    wavelengthmm = 4
    spatialAspectRatio = 0.5
    orientation = 60
    sigma = sigmamm / voxelSizemmV[0]
    wavelength = wavelengthmm / voxelSizemmV[0]

    padMethod = 'padzeros'
    padSizeV = np.array([41, 41, 0])

    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M, __ = filters.gaborFilter(procScan3M, sigma, wavelength, spatialAspectRatio,
                                               orientation, paddingV=padSizeV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '4a1.mat'
    compareMaps(responseMap3M, refMapName)

    # a2
    print('Testing setting 4.a.2')
    orientation = np.array([45, 90, 135, 180, 225, 270, 315, 360])
    sigma = sigmamm / voxelSizemmV[0]
    wavelength = wavelengthmm / voxelSizemmV[0]
    padMethod = 'padzeros'
    padSizeV = np.array([41, 41, 41])
    agg = {'orientation': 'average', 'plane': 'average'}
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M, __ = filters.gaborFilter3d(procScan3M, agg, sigma, wavelength, spatialAspectRatio,
                                                 orientation, paddingV=padSizeV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '4a2.mat'
    compareMaps(responseMap3M, refMapName)

    # b1
    print('Testing setting 4.b.1')
    scan3M, mask3M, planC = loadData('sphere')
    voxelSizemmV = planC.scan[0].getScanSpacing() * 10

    sigmamm = 20
    wavelengthmm = 8
    spatialAspectRatio = 2.5
    orientation = 225
    sigma = sigmamm / voxelSizemmV[0]
    wavelength = wavelengthmm / voxelSizemmV[0]

    # calc. radius
    padMethod = 'mirror'
    padSizeV = np.array([32, 32, 0])
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M, __ = filters.gaborFilter(procScan3M, sigma, wavelength, spatialAspectRatio,
                                               orientation, paddingV=padSizeV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '4b1.mat'
    compareMaps(responseMap3M, refMapName)

    # b2
    print('Testing setting 4.b.2')
    orientation = [22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5, 360]
    sigma = sigmamm / voxelSizemmV[0]
    wavelength = wavelengthmm / voxelSizemmV[0]

    # calc. radius
    padMethod = 'mirror'
    padSizeV = np.array([32, 32, 32])

    agg = {'orientation': 'average', 'plane': 'average'}
    procScan3M, __, __ = preprocess.padScan(scan3M, mask3M, padMethod, padSizeV)
    paddedResponse3M, __ = filters.gaborFilter3d(procScan3M, agg, sigma, wavelength, spatialAspectRatio,
                                                 orientation, paddingV=padSizeV)
    responseMap3M = preprocess.unpadScan(paddedResponse3M, padMethod, padSizeV)
    refMapName = '4b2.mat'
    compareMaps(responseMap3M, refMapName)

if __name__ == "__main__":
    run_tests_phase1()
