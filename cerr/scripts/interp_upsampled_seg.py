from cerr import plan_container as pc
import cerr.contour.rasterseg as rs
import cerr.dataclasses.structure as strct
from cerr.utils.mask import computeBoundingBox
from cerr.radiomics.preprocess import imgResample3D, getResampledGrid
from cerr.dataclasses import structure as cerrStruct
import os
from cerr import plan_container as pc
import SimpleITK as sitk
import numpy as np
import cerr.dataclasses.scan as cerrScn
import cerr.viewer as vwr

if __name__ == "__main__":

    # bug
    scanNii = r'L:\Aditya\HN_T2_radiomics\orig_nii_data\HN415D\Wk1\12-BOGGIA^GLENN_AX_T1_FS_W_20180525081459.nii'
    segNii = r'L:\Aditya\HN_T2_radiomics\orig_nii_data\HN415D\Wk1\12-BOGGIA^GLENN_AX_T1_FS_W_20180525081459_BD.nii'
    scanNii = r'S:\Amita_Dave_Collaboration_2019\30Gy_Abdalla\HN409D\Wk0\6-GONNELLO^ROBERT_W_Axial_T2_20191104082319.nii'
    segNii = r'S:\Amita_Dave_Collaboration_2019\30Gy_Abdalla\HN409D\Wk0\6-GONNELLO^ROBERT_W_Axial_T2_20191104082319_JH.nii'
    scanNii = r'S:\Amita_Dave_Collaboration_2019\30Gy_Abdalla\HN522D\Wk1\301-STURNIOLO^FRANK_M_Axial_T2_FS_STRAIGHT_20191206160536.nii.gz'
    segNii = r'S:\Amita_Dave_Collaboration_2019\30Gy_Abdalla\HN522D\Wk1\301-STURNIOLO^FRANK_M_Axial_T2_FS_STRAIGHT_20191206160536_JH.nii.gz'
    scanNii = r'L:\Aditya\HN_T2_radiomics\orig_nii_data\HN468D\Wk0\301-CALIDONNA_STEVEN_M_Axial_T2_FS_STRAIGHT_20190910180104.nii.gz'
    segNii = r'L:\Aditya\HN_T2_radiomics\orig_nii_data\HN468D\Wk0\301-CALIDONNA_STEVEN_M_Axial_T2_FS_STRAIGHT_20190910180104_new_JH.nii.gz'
    planC = pc.loadNiiScan(scanNii, 'MR SCAN')
    planC = pc.loadNiiStructure(segNii, 0, planC, {1: 'GTV'})

    # Import multiframe dicom
    par_maps_dcm_dir = r"\\vpensmph\DeasyLab1\Aditya\HN_T1_radiomics\test"
    ct_dcm_dir = r"\\vpensmph\DeasyLab2\Pierre\FDG_CT_for_Hypoxia_Nancy\35407384\9653349\D2013_11_26\CT\S0002_CT_H&N"

    ct_dcm_dir = r"\\vpensmph\DeasyLab2\Pierre\FDG_CT_for_Hypoxia_Nancy\35407384\9650539\D2013_12_09\CT\S0002_CT_for_150min"
    pt_dcm_dir = r"\\vpensmph\DeasyLab2\Pierre\FDG_CT_for_Hypoxia_Nancy\35407384\9650539\D2013_12_09\PT\S0003_FMISO_150_180"
    planC = pc.loadDcmDir(ct_dcm_dir)
    planC = pc.loadDcmDir(pt_dcm_dir, planC)
    planC = pc.loadDcmDir(par_maps_dcm_dir, planC)

    scanNum = [0]
    strNum = [0]
    doseNum = []
    vect_dict = {}
    displayMode = '2d'
    viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
            vwr.showNapari(scanNum, strNum, doseNum, vect_dict, planC, displayMode)

    # Write pixel spacing to csv
    dataDir = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL'
    all_pat_dirs = []
    for d in os.scandir(dataDir):
        dirPath = d.path
        _, id = os.path.split(dirPath)
        if id[:2] == "HN":
            for f in os.scandir(dirPath):
                filePath = f.path
                _, fname = os.path.split(filePath)
                fname = fname.lower()
                if "_t2_" in fname and "_as" not in fname:
                    all_pat_dirs.append(filePath)

    featList = []
    for pt_file in all_pat_dirs:
        print("Data dir :" + pt_file)

        planC = pc.loadNiiScan(pt_file, 'MR SCAN')

        x,y,z = planC.scan[0].getScanXYZVals()

        featDict = {}
        featDict['ID'] = pt_file
        featDict['L-R (dx, mm)'] = (x[1]-x[0]) * 10
        featDict['A-P (dy, mm)'] = (y[0]-y[1]) * 10
        featDict['S-I (dz, mm)'] = (z[1]-z[0]) * 10
        featList.append(featDict)

    import csv
    csvFileName = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\img_resolutions.csv'
    with open(csvFileName, 'a', newline='') as csvfile:
        flatFeatDict = featList[0]
        fieldnames = flatFeatDict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for flatFeatDict in featList:
            writer.writerow(flatFeatDict)


        from cerr.radiomics.preprocess import imgResample3D

        # Assume CT scan is at index 0 and PET scan is at index 1
        ctScanNum = 0
        ptScanNum = 1

        # Get grid for CT scan
        ctXValsV, ctYValsV, ctZValsV = planC.scan[ctScanNum].getScanXYZVals()

        # Get PET Scan array
        ptScan3M = planC.scan[ptScanNum].getScanArray()
        ptXValsV, ptYValsV, ptZValsV = planC.scan[ctScanNum].getScanXYZVals()

        #Resample scan
        scanInterpMethod = "sitkLinear"
        ptScanOnCTgrid3M = imgResample3D(ptScan3M, ptXValsV, ptYValsV, ptZValsV,\
                                ctXValsV, ctYValsV, ctZValsV, scanInterpMethod)


        # Voxel-wise Intensities within a structure
        import cerr.contour.rasterseg as rs
        structNum = 0
        mask3M = rs.getStrMask(structNum, planC)
        ctScan3M = planC.scan[ctScanNum].getScanArray()
        ctV = ctScan3M[mask3M]
        ptV = ptScanOnCTgrid3M[mask3M]


        # Store interpolated PET scan in planC to visualize
        modality = "PT SCAN"
        assocScanNum = ctScanNum
        planC = pc.importScanArray(ptScanOnCTgrid3M, ctXValsV, ctYValsV, ctZValsV, modality, assocScanNum, planC)

    # orig_nii_scan = r'L:\Abdalla\Thyroid_MR_T2\HN209D\Img_N4.nii'
    # resamp_nii_scan = r'L:\Abdalla\Thyroid_MR_T2\HN209D\Img_N4\Img_N4_smore4.nii'
    # orig_nii_str = r'L:\Abdalla\Thyroid_MR_T2\HN209D\Seg.nii'
    # outputStrNiiName = r'L:\Abdalla\Thyroid_MR_T2\HN209D\Img_N4\smoreInterpStruct.nii.gz'

    origNiiScan = r'L:\Abdalla\Thyroid_MR_T2\HN209D\Img_N4.nii'
    origNiiStr = r'L:\Abdalla\Thyroid_MR_T2\HN209D\Seg.nii'
    resampNiiScan = r'L:\Abdalla\Thyroid_MR_T2\HN209D\Img_N4\Img_N4_smore4.nii'
    outputNiiStr = r'L:\Abdalla\Thyroid_MR_T2\HN209D\Img_N4\smoreInterpStruct.nii.gz'

    # Import DICOM metadata to planC
    planC = pc.loadNiiScan(origNiiScan)
    planC = pc.loadNiiScan(resampNiiScan, planC)
    assocScanNum = 0
    planC = pc.loadNiiStructure(origNiiStr, assocScanNum, planC, labels_dict={1: 'tumor'})

    from cerr.contour import rasterseg as rs
    mask3M = rs.getStrMask(0, planC)

    # Copy segmentation from scan the original scan to the new scan
    structNum = 0
    scanNum = 1
    planC = cerrStruct.copyToScan(structNum, scanNum, planC)

    from skimage import measure
    newStrNum = len(planC.structure)-1
    mask3M = rs.getStrMask(newStrNum, planC)
    labeledMask3M, numLabels = measure.label(mask3M, return_num = True)
    for label in range(1,numLabels+1):
        maskForLabel3M = labeledMask3M == label
        strName = 'comp ' + str(label)
        planC = cerrStruct.import_mask(maskForLabel3M, scanNum, strName, planC)

    # Show scan, structure and dose overlay in 3D
    from cerr import viewer as vwr
    import numpy as np
    scanNumList = [1]
    doseNumList = []
    strNumList = [1]
    displayMode = '2d' # 'path' or 'surface'
    viewer, scan_layer, dose_layer, struct_layer = \
        vwr.show_scan_struct_dose(scanNumList, strNumList, doseNumList, planC, displayMode)


    # Export scan, structure to nii
    import os
    scanNum = 1
    scanNiiName = r'M:\Data\CERR_test_datasets\nii_test\scan.nii.gz'
    planC.scan[scanNum].saveNii(scanNiiName)

    structNum = len(planC.structure)-1 # assume the structure to export is the last one in the list
    planC.structure[structNum].saveNii(outputNiiStr, planC)



