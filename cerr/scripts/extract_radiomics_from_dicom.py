from cerr import plan_container as pc


def calc_feat_diff(matS, pyS, featType):
    featDiffV = []
    for key in matS[featType].keys():
        featDiffV.append(np.mean((matS[featType][key] - pyS[key]) / (matS[featType][key] + np.finfo(float).eps) * 100))
    return featDiffV, np.asarray(list(matS[featType].keys()))


if __name__ == "__main__":


    import cerr.dataclasses.structure as cerrStr
    import numpy as np

    dcm_dir = r'\\pensmphDEASYLAB\deasylab1\Maria\HNC_NormalTissue_pCT\Trismus_PrePostSegGuideline\38173490'
    planC = pc.loadDcmDir(dcm_dir)


    # dataset for radiomics calculation
    dcm_dir = r"L:\Data\RTOG0617\DICOM\NSCLC-Cetuximab_flat_dirs\0617-381906_09-09-2000-87022"
    dcm_dir = r'M:\Data\soft_tissue_sarcoma_DrBozzo\expanded_images\T1\RIA_16-1123_000_000001'

    moving_path = r"M:\Lopa\HN_NIFTI_files\ANTs\data_HN_CT_img_original\moving\0_patient_00042740_Week2_CT.nii.gz"
    moving_mask_path = r"M:\Lopa\HN_NIFTI_files\ANTs\data_HN_CT_img_original\moving_OAR\0_patient_00042740_Week2_CT.nii.gz"
    planC = pc.loadNiiScan(moving_path, 'CT SCAN')
    planC = pc.loadNiiStructure(moving_mask_path,0,planC)
    pc.saveNiiStructure(r'M:\Lopa\HN_NIFTI_files\ANTs\data_HN_CT_img_original\moving_OAR\test.nii.gz', [0,1,2,3,4,5], planC)

    import time
    t = time.time()
    planC = pc.loadDcmDir(dcm_dir)
    elapsed = time.time() - t
    print(f"Time spent to read dicom into planC = {elapsed / 60:.2f} minutes")

    # from cerr.radiomics import shape
    # from cerr.contour import rasterseg as rs
    # mask3M = rs.getStrMask(0, planC)
    # xV, yV, zV = planC.scan[0].getScanXYZVals()
    # shapeS = shape.compute_shape_features(mask3M,xV,yV,zV)

    # Radiomics
    from cerr.radiomics import ibsi1
    scanNum = 0
    structNum = 0
    settingsFile = r"\\vpensmph\deasylab2\Aditya\Radiomics_features\original_settings.json"
    settingsFile = r"M:\Data\soft_tissue_sarcoma_DrBozzo\extracted_features_AI\settings\pyCERR_compatible\original_settings.json"
    shapeS, firstOrderFeatS, glcmFeatS, rlmFeatS, szmFeatS, ngldmFeatS, ngtdmFeatS = \
        ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)


    # Compare feature values from Matlab CERR
    from scipy import io as sio
    import os
    import numpy as np

    matDir = r'M:\Aditya\Radiomics_features\cerr_featurss'
    shapeMat = os.path.join(matDir, 'shape.mat')
    firstOrderMat = os.path.join(matDir, 'first.mat')
    glcmMat = os.path.join(matDir, 'glcm.mat')
    rlmMat = os.path.join(matDir, 'rlm.mat')
    szmMat = os.path.join(matDir, 'szm.mat')
    ngldmMat = os.path.join(matDir, 'ngldm.mat')
    ngtdmMat = os.path.join(matDir, 'ngtdm.mat')

    cerr_shapeS = sio.loadmat(shapeMat, simplify_cells=True)
    diffV, keys = calc_feat_diff(cerr_shapeS, shapeS, 'shapeS')
    print('==========  Shape  ===========')
    print(keys[np.abs(diffV) > 1])

    cerr_firstS = sio.loadmat(firstOrderMat, simplify_cells=True)
    diffV, keys = calc_feat_diff(cerr_firstS, firstOrderFeatS, 'firstOrderS')
    print('==========  First Order  ===========')
    print(keys[np.abs(diffV) > 1])

    cerr_glcmS = sio.loadmat(glcmMat, simplify_cells=True)
    cerr_glcmS['glcmFeatureS']['diffAvg'] = cerr_glcmS['glcmFeatureS']['dissimilarity']
    diffV, keys = calc_feat_diff(cerr_glcmS, glcmFeatS, 'glcmFeatureS')
    print('==========  GLCM  ===========')
    print(keys[np.abs(diffV) > 1])

    cerr_rlmS = sio.loadmat(rlmMat, simplify_cells=True)
    diffV, keys = calc_feat_diff(cerr_rlmS, rlmFeatS, 'rlmFeatureS')
    print('==========  RLM  ===========')
    print(keys[np.abs(diffV) > 1])

    cerr_szmS = sio.loadmat(szmMat, simplify_cells=True)
    diffV, keys = calc_feat_diff(cerr_szmS, szmFeatS, 'szmFeatureS')
    print('==========  SZM  ===========')
    print(keys[np.abs(diffV) > 1])

    cerr_ngldmS = sio.loadmat(ngldmMat, simplify_cells=True)
    diffV, keys = calc_feat_diff(cerr_ngldmS, ngldmFeatS, 'ngldmFeatS')
    print('==========  NGLDM  ===========')
    print(keys[np.abs(diffV) > 1])

    cerr_ngtdmS = sio.loadmat(ngtdmMat, simplify_cells=True)
    diffV, keys = calc_feat_diff(cerr_ngtdmS, ngtdmFeatS, 'ngtdmFeatS')
    print('==========  NGTDM  ===========')
    print(keys[np.abs(diffV) > 1])

    # ====== Examples ======

    # Apply filter
    # from cerr.radiomics import filters
    # sobel3M = filters.sobelFilt(volToEval)
    # zScore3M = (volToEval - meanVal) / sigmaVal
    # scanITKObj = scn.getSITKImage(volToEval,xValsV,yValsV,zValsV,orientV)
    # scanCERR3M,imgOriV,imgPosV = scn.getCERRScan(scanITKObj)
    # biasFieldCorrected3M
    # Or any user defined filter


    # Crop scan to extents of mask
    from cerr.utils.mask import computeBoundingBox
    from cerr.dataclasses import scan as scn
    from cerr.contour import rasterseg as rs
    import numpy as np

    mask3M = rs.getStrMask(structNum, planC)
    scanNum = scn.getScanNumFromUID(planC.structure[structNum].assocScanUID, planC)
    scan3M = planC.scan[scanNum].scanArray - planC.scan[scanNum].scanInfo[0].CTOffset
    (rmin, rmax, cmin, cmax, smin, smax, _) = computeBoundingBox(mask3M)
    croppedScan3M = scan3M[rmin:rmax + 1, cmin:cmax + 1, smin:smax + 1]
    croppedMask3M = mask3M[rmin:rmax + 1, cmin:cmax + 1, smin:smax + 1]
    croppedScan3M[~croppedMask3M] = np.nan

    # Convert structure to binary mask
    import cerr.contour.rasterseg as rs

    str_num = 5
    mask3M = rs.getStrMask(str_num, planC)
    str_name = planC.structure[str_num].structureName

    # Get DVH
    from cerr import dvh

    dosesV, volsV, isErr = dvh.getDVH(0, 0, planC)
    binWidth = 0.025
    doseBinsV, volHistV = dvh.doseHist(dosesV, volsV, binWidth)
    percent = 70
    dvh.MOHx(doseBinsV, volHistV, percent)

    # Show scan, structure and dose overlay
    from cerr import viewer as vwr
    import numpy as np

    num_structs = len(planC.structure)
    str_num_list = np.arange(num_structs)
    scan_num = [0]
    dose_num = [0]  # 0
    str_num_list = np.arange(num_structs)  # [0]
    viewer, scan_layer, dose_layer, labels_layer = vwr.show_scan_struct_dose(scan_num, str_num_list, dose_num, planC)

    # vwr.show_scan_dose(0,0,46,planC)
