from cerr import plan_container as pc
import cerr.contour.rasterseg as rs
import cerr.dataclasses.structure as strct
from cerr.utils.mask import computeBoundingBox
from cerr.radiomics.preprocess import imgResample3D, getResampledGrid

if __name__ == "__main__":
    # Get path to an example dataset included with CERR
    from cerr import datasets
    import os
    dcm_dir = os.path.join(os.path.dirname(datasets.__file__),'radiomics_phantom_dicom','PAT1')

    # Import dicom to planC
    dcm_dir = r"M:\Aditya\Cornell_AI_Imaging_course\lung_dicom_5pts\R01-001"
    dcm_dir = r"M:\Data\DCE_OMT\dicom_WeiHuang_and_TCIA\BreastChemo1\BreastChemo1\BC1_V1_concatenated"
    from cerr import plan_container as pc
    planC = pc.loadDcmDir(dcm_dir)

    settingsFile = r"\\vpensmph\deasylab2\Aditya\Cornell_AI_Imaging_course\settings\original_settings.json"
    from cerr.radiomics import ibsi1
    scanNum = 0
    structNum = 0
    featDict = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)



    # Show scan, structure and dose overlay in 3D
    from cerr import viewer as vwr
    scanNum = [0]
    doseNum = []
    strNum = []
    displayMode = '2d' # 'path' or 'surface'
    viewer, scan_layer, dose_layer, struct_layer = \
            vwr.show_scan_struct_dose(scanNum, strNum, doseNum, planC, displayMode)
