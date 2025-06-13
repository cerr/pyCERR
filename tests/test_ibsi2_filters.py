"""
 This script compares convolutional filter response maps using synthetic data and filter settings from IBSI2-phase 1
 against a reference standard.

 Dataset: https://github.com/theibsi/data_sets/tree/master/ibsi_2_digital_phantom
 Configurations: https://www.overleaf.com/project/5da9e0b82f399f0001ad3970
"""

import os
import numpy as np

from cerr import plan_container as pc
from cerr.radiomics import texture_utils
import matplotlib.pyplot as plt

currPath = os.path.abspath(__file__)
cerrPath = os.path.join(os.path.dirname(os.path.dirname(currPath)),'cerr')
dataPath = os.path.join(cerrPath,'datasets','ibsi_radiomics_dicom','ibsi2_phase_1')
settingsPath = os.path.join(cerrPath, 'datasets','radiomics_settings',\
                            'ibsi_settings','ibsi2_phase_1')
refPath = os.path.join(cerrPath,'datasets','reference_values_for_tests','ibsi2_phase_1')

def list_tests():
          print('pyCERR provides the following tests:\n '
                '• test_mean_filters_3d\n '
                '• test_mean_filt_2d\n ' 
                '• test_LoG_filters_3d\n '
                '• test_LoG_filter_2d\n ' 
                '• test_laws_filters_3d\n ' 
                '• test_laws_filter_2d\n '
                '• test_rot_inv_laws_filters_3d\n '
                '• test_rot_inv_laws_filter_2d\n '
                '• test_rot_inv_laws_energy_filters_3d\n '
                '• test_rot_inv_laws_energy_filter_2d\n '
                '• test_gabor_filters_25d\n '
                '• test_gabor_filters_2d\n '
                '• test_wavelet_filters_3d\n '
                '• test_rot_inv_wavelet_filters_3d\n\n'
                'and a wrapper to run all available tests: \n'
                '• run_ibsi_image_filters\n\n' 
                'Optional inputs:\n '
                'savePath: Write NIfTI response maps to location (default:None)\n '
                'vis     : If True, display IBSI consensus map, pyCERR-computed response map, and difference map (default: False).')
def load_data(synthDataset):
    """ Load CERR archive, return scan and mask(all ones)"""

    print('Loading dataset ' + synthDataset)
    datasetDir = os.path.join(dataPath,synthDataset)

    # Import DICOM data
    planC = pc.loadDcmDir(datasetDir)

    # Extract scan
    scan3M = planC.scan[0].scanArray - planC.scan[0].scanInfo[0].CTOffset
    scan3M = scan3M.astype(float)

    # Create mask (all ones)
    scanSizeV = np.shape(scan3M)
    mask3M = np.ones(scanSizeV, dtype=bool)

    return scan3M, mask3M, planC

def plot_slice(data1, data2, data3, slice_idx):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plt.subplots_adjust(right=0.85)
    titles = ['IBSI consensus', 'pyCERR output', 'Difference']
    # Clear each axis, plot the selected slice, and add a colorbar
    for ax, vol, title in zip(axes, (data1, data2, data3), titles):
        ax.clear()
        im = ax.imshow(vol[:, :, slice_idx], cmap='gray', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title } — slice {slice_idx}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.canvas.draw_idle()


def compare_maps(calcMap3M, refMapName, vis=False):
    """ Indicate if maps match reference, otherwise summarize differences."""

    # Load reference results
    refMapPath = os.path.join(refPath, refMapName)
    planC = pc.loadNiiScan(refMapPath)
    refMap3M = np.flip(planC.scan[0].getScanArray(),2)

    # Compute difference between pyCERR calculations & reference values
    diffMap3M = calcMap3M - refMap3M
    np.testing.assert_allclose(calcMap3M, refMap3M, rtol=1, atol=2)

    # Report on differences
    __, __, nSlcs = np.shape(diffMap3M)
    if np.max(np.abs(diffMap3M)) < 1e-4:
        print('Success! Results match reference std.')
    else:
        print('Results differ within tolerance:')
        diff_v = [np.min(diffMap3M), np.mean(diffMap3M),\
                  np.median(diffMap3M), np.max(diffMap3M)]
        print('min, mean, median, max of differences:')
        print(diff_v)

    if vis:
        midSlc = int(np.round(nSlcs / 2))
        plot_slice(refMap3M, calcMap3M, diffMap3M, midSlc)
    print('-------------')


def run_test(planC, mask3M, config, savePath=None, vis=False):

    scanNum = 0
    settingsFile = os.path.join(settingsPath,'ibsi_phase2_1_id_'+\
                                config + '.json')
    refMapName = config+'-ValidCRM.nii'

    #Calc. filter response
    planC = texture_utils.generateTextureMapFromPlanC(planC,\
                          scanNum, mask3M, settingsFile)
    filtIdx = len(planC.scan)-1
    responseMap3M = planC.scan[filtIdx].getScanArray()

    if savePath is not None:
        outFile = os.path.join(savePath, config + '.nii.gz')
        planC.scan[filtIdx].saveNii(outFile)

    # Compare to reference std
    compare_maps(responseMap3M, refMapName, vis=vis)

def test_mean_filters_3d(savePath=None, vis=False):
    __, mask3M, planC = load_data('checkerboard')

    print('Testing setting 1.a.1')
    config = '1a1'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

    print('Testing setting 1.a.2')
    config = '1a2'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

    print('Testing setting 1.a.3')
    config = '1a3'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

    print('Testing setting 1.a.4')
    config = '1a4'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_mean_filt_2d(savePath=None, vis=False):
    print('Testing setting 1.b.1')
    __, mask3M, planC = load_data('impulse')

    config = '1b1'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_LoG_filters_3d(savePath=None, vis=False):
    print('Testing setting 2.a')
    __, mask3M, planC = load_data('impulse')
    config = '2a'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

    print('Testing setting 2.b')
    __, mask3M, planC = load_data('checkerboard')
    config = '2b'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_LoG_filter_2d(savePath=None, vis=False):
    print('Testing setting 2.c')
    __, mask3M, planC = load_data('checkerboard')
    config = '2c'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_laws_filters_3d(savePath=None, vis=False):
    # Config. a1
    print('Testing setting 3.a.1')
    __, mask3M, planC = load_data('impulse')

    config = '3a1'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

    print('Testing setting 3.b.1')
    __, mask3M, planC = load_data('checkerboard')
    config = '3b1'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_laws_filter_2d(savePath=None, vis=False):
    # Config. c1
    print('Testing setting 3.c.1')
    __, mask3M, planC = load_data('checkerboard')
    config = '3c1'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_rot_inv_laws_filters_3d(savePath=None, vis=False):

    print('Testing setting 3.a.2')
    __, mask3M, planC = load_data('impulse')
    config = '3a2'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

    print('Testing setting 3.b.2')
    __, mask3M, planC = load_data('checkerboard')
    config = '3b2'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_rot_inv_laws_filter_2d(savePath=None, vis=False):
    print('Testing setting 3.c.2')
    __, mask3M, planC = load_data('checkerboard')
    config = '3c2'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_rot_inv_laws_energy_filters_3d(savePath=None, vis=False):
    print('Testing setting 3.a.3')
    __, mask3M, planC = load_data('impulse')
    config = '3a3'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

    print('Testing setting 3.b.3')
    __, mask3M, planC = load_data('checkerboard')
    config = '3b3'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_rot_inv_laws_energy_filter_2d(savePath=None, vis=False):
    print('Testing setting 3.c.3')
    __, mask3M, planC = load_data('checkerboard')
    config = '3c3'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_gabor_filters_2d(savePath=None, vis=False):
    print('Testing setting 4.a.1')
    __, mask3M, planC = load_data('impulse')
    config = '4a1'
    run_test(planC, mask3M, config, vis=vis)

    print('Testing setting 4.b.1')
    __, mask3M, planC = load_data('sphere')
    config = '4b1'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_gabor_filters_25d(savePath=None, vis=False):
    print('Testing setting 4.a.2')
    __, mask3M, planC = load_data('impulse')
    config = '4a2'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

    print('Testing setting 4.b.2')
    __, mask3M, planC = load_data('sphere')
    config = '4b2'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_wavelet_filters_3d(savePath=None, vis=False):
    print('Testing setting 5.a.1')
    __, mask3M, planC = load_data('impulse')
    config = '5a1'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

    print('Testing setting 6.a.1')
    __, mask3M, planC = load_data('sphere')
    config = '6a1'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def test_rot_inv_wavelet_filters_3d(savePath=None, vis=False):
    print('Testing setting 5.a.2')
    __, mask3M, planC = load_data('impulse')
    config = '5a2'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

    print('Testing setting 6.a.2')
    __, mask3M, planC = load_data('sphere')
    config = '6a2'
    run_test(planC, mask3M, config, savePath=savePath, vis=vis)

def run_ibsi_image_filters(savePath=None, vis=False):
    """ Generate maps using IBSI-2 phase-1 configurations """

    test_mean_filters_3d(savePath=savePath, vis=vis)
    test_mean_filt_2d(savePath=savePath, vis=vis)

    test_LoG_filters_3d(savePath=savePath, vis=vis)
    test_LoG_filter_2d(savePath=savePath, vis=vis)

    test_laws_filters_3d(savePath=savePath, vis=vis)
    test_laws_filter_2d(savePath=savePath, vis=vis)
    test_rot_inv_laws_filters_3d(savePath=savePath, vis=vis)
    test_rot_inv_laws_filter_2d(savePath=savePath, vis=vis)

    test_rot_inv_laws_energy_filters_3d(savePath=savePath, vis=vis)
    test_rot_inv_laws_energy_filter_2d(savePath=savePath, vis=vis)

    test_gabor_filters_25d(savePath=savePath, vis=vis)
    test_gabor_filters_2d(savePath=savePath, vis=vis)

    test_wavelet_filters_3d(savePath=savePath, vis=vis)
    test_rot_inv_wavelet_filters_3d(savePath=savePath, vis=vis)

if __name__ == "__main__":
    savePath=None
    visFlag=False
    run_ibsi_image_filters(savePath=savePath,vis=visFlag)
