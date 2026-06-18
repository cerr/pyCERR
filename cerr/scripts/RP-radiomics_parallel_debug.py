import os
import cerr.plan_container as pc
from cerr.radiomics import ibsi1
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from cerr.contour import rasterseg as rs
from cerr.radiomics import texture_utils
from cerr.dataclasses import structure as cerrStr

# Paths
dicom_root = r"\\vpensmph\deasylab1\Maria\LA-NSCLC_Durva_N230\N_65_ValidationCohort\DCM\RDRPRS_PET-CT"
nifti_root = r"\\vpensmph\deasylab1\Maria\LA-NSCLC_Durva_N230\N_65_ValidationCohort\structsForRP"
wavelete_texture_settings = r'\\vpensmph\DeasyLab3\Leyla\codes\Lung-RP-Project\radiomics-setting\wavelet_first_order_settings_HHL.json'

# Features settings
feature_settings = r'\\vpensmph\DeasyLab3\Leyla\codes\Lung-RP-Project\radiomics-setting\ibsi1_apa.json'

# Texture settings
textureSettingsDir = r'\\vpensmph\deasylab3\Leyla\codes\Lung-RP-Project\radiomics-setting'
textureSettingsFiles = ['wavelet_first_order_settings_HHH.json','wavelet_first_order_settings_HHL.json',
                        'wavelet_first_order_settings_HLL.json','wavelet_first_order_settings_LHL.json',
                        'wavelet_first_order_settings_LLH.json','wavelet_first_order_settings_LLL.json',]

# Features save file
ibsiCsvFile = r"\\vpensmph\deasylab3\Leyla\codes\Lung-RP-Project\radiomics_results_wavelet_parallel_test_apa.csv"

# --- Per-study worker ---
def process_study(study_id):
    results = []

    study_nifti_dir = os.path.join(nifti_root, study_id)
    study_dicom_dir = os.path.join(dicom_root, study_id)

    if not os.path.isdir(study_nifti_dir) or not os.path.exists(study_dicom_dir):
        return f"⚠ Skipping study {study_id}: missing NIfTI or DICOM"

    try:
        planC = pc.loadDcmDir(study_dicom_dir)
        scanNum = planC.structure[0].getStructureAssociatedScan(planC)
        #niiFileName = os.path.join(r'\\vpensmph\DeasyLab3\Leyla\codes\Lung-RP-Project',study_id+'_scan.nii.gz')
        #planC = pc.loadNiiScan(niiFileName,'CT SCAN')
        #scanNum = 0
        ROIs_added = []
        scanSize = planC.scan[scanNum].getScanSize()
        maskUnion3M = np.zeros(scanSize, dtype=bool)
        # Load all NIfTI ROIs
        for nii_file in os.listdir(study_nifti_dir):
            print(f"Processing study {study_id}...")
            if nii_file.lower().endswith(".nii.gz"):
                nii_path = os.path.join(study_nifti_dir, nii_file)
                ROI_added = os.path.splitext(os.path.splitext(nii_file)[0])[0]
                strucLabel = 1  # label representing the structure foreground

                planC = pc.loadNiiStructure(
                    nii_path,
                    assocScanNum=scanNum,
                    planC=planC,
                    labels_dict={strucLabel: ROI_added}
                )
                structNum = len(planC.structure) - 1
                ROIs_added.append((ROI_added, structNum))
                mask3M = rs.getStrMask(structNum, planC)
                maskUnion3M |= mask3M
        strNames = [s.structureName for s in planC.structure]
        print(f"structures for study ID '{study_id}': ", strNames)
        print(f"Added ROIs for study ID '{study_id}':  ", ROIs_added)

        # Add maskUnion3M to planC
        planC = pc.importStructureMask(maskUnion3M, scanNum, 'bounding_box', planC)
        bboxStructNum = len(planC.structure) - 1

        textureScansNamInd = []
        # Compute texture for the combined ROI bounding box
        for textureFile in textureSettingsFiles:
            textrSetting = os.path.join(textureSettingsDir, textureFile)
            planC = texture_utils.generateTextureMapFromPlanC(
                planC,
                scanNum,
                bboxStructNum,
                textrSetting)
            textureScansNamInd.append((textureFile[:-5],len(planC.scan)-1))

        # Extract radiomics for each ROI and IBSI setting
        for roi_name, structNum in ROIs_added:
            print(f"Extracting features for ROI '{roi_name}' (structNum={structNum})...")
            for textrStr, scanNum in textureScansNamInd:
                planC = cerrStr.copyToScan(structNum, scanNum, planC)
                radiomicsStrNum = len(planC.structure) - 1
                print(f"   ➡ Extracting features from {textrStr}")
                try:
                    featDict, _ = ibsi1.computeScalarFeatures(
                        scanNum=scanNum,
                        structNum=radiomicsStrNum,
                        settingsFile=feature_settings,
                        planC=planC
                    )
                    # update the feature names
                    dictKeys = featDict.keys()
                    for key in list(dictKeys):
                       #newKey = key.replace('original',textrStr)
                       newKey = key.replace('original','')
                       featDict[newKey] = featDict.pop(key)

                    row = {
                        "study_id": study_id,
                        "roi_name": roi_name,
                        "textureType": textrStr
                    }
                    row.update(featDict)
                    results.append(row)
                except Exception as e:
                    print(f"❌ Error extracting features for {study_id}, ROI {roi_name}, {textrStr}: {e}")

        return results
    except Exception as e:
        return [f"❌ Error processing study {study_id}: {e}"]


# --- Main runner ---
if __name__ == '__main__':
    study_ids = [d for d in os.listdir(nifti_root)[:2] if os.path.isdir(os.path.join(nifti_root, d))]
    all_results = []

    for study_id in study_ids:
        study_result = process_study(study_id.zfill(8))
        all_results.extend(study_result)

    # with ProcessPoolExecutor() as executor:
    #     for study_result in executor.map(process_study, study_ids):
    #         if isinstance(study_result, list):
    #             all_results.extend(study_result)
    #         else:
    #             print(study_result)

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(ibsiCsvFile, index=False)
    print("✅ Radiomics extraction completed in parallel.")
