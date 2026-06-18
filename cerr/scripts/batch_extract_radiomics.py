import os
from cerr import datasets
from cerr import plan_container as pc
from cerr.radiomics import ibsi1
from scipy import io as sio
import numpy as np

if __name__ == "__main__":

    import os
    from cerr import plan_container as pc
    import SimpleITK as sitk
    import numpy as np
    from scipy import stats
    import cerr.dataclasses.scan as cerrScn
    from cerr.contour import rasterseg as rs
    from cerr.radiomics import ibsi1

    dicomDir = r'L:\Aditya\Resident_Informatics_Rotation\radiomics\lung_dicom_5pts'
    featureSaveDir = r'C:\software\pyCERR-Notebooks_master\pyCERR-Notebooks\radiomics_features'
    ibsi1Settings = r'C:\software\pyCERR-Notebooks_master\pyCERR-Notebooks\radiomics_settings\ibsi1.json'
    ibsi2Settings = r'C:\software\pyCERR-Notebooks_master\pyCERR-Notebooks\radiomics_settings\ibsi2.json'
    ibsiCsvFile = os.path.join(featureSaveDir, 'ibsi_1_2.csv')

    featList = []
    id_dict = {}
    for d in os.scandir(dicomDir):

        _, id = os.path.split(d)
        print("Data dir: " + id)

        # Load data into planC
        #imgFile = os.path.join(d, 'Img.nii.gz')
        #segFile = os.path.join(d, 'Seg.nii.gz')
        #planC = pc.load_nii_scan(imgFile, 'MR SCAN')
        #planC = pc.load_nii_structure(segFile,0,planC,{1: 'GTV'})
        planC = pc.loadDcmDir(d)

        # Record ID
        id_dict['id'] = id

        # Compute radiomics features
        scanNum = 0
        structNum = 0
        featDictIbsi1, _ = ibsi1.computeScalarFeatures(scanNum, structNum, ibsi1Settings, planC)
        ibsi1FeatDict = {**id_dict, **featDictIbsi1}
        #featDictIbsi2, _ = ibsi1.computeScalarFeatures(scanNum, structNum, ibsi2Settings, planC)
        #ibsi1FeatDict = {**id_dict, **featDictIbsi1, **featDictIbsi2}
        featList.append(ibsi1FeatDict)

    ibsi1.writeFeaturesToFile(featList, ibsiCsvFile)








    phantom_dir = os.path.join(os.path.dirname(datasets.__file__),'radiomics_phantom_dicom')
    pat_names = ['PAT1', 'PAT2', 'PAT3', 'PAT4']
    all_pat_dirs = [os.path.join(phantom_dir, pat) for pat in pat_names]
    #settingsFile = r"\\vpensmph\deasylab2\Aditya\Radiomics_features\original_settings.json"
    settingsFile = r"\\vpensmph\deasylab2\Aditya\Radiomics_features\ibsi1_test.json"
    csvFileName = r"\\vpensmph\deasylab2\Aditya\Radiomics_features\test_feats.csv"
    featList = []
    id = {}
    for pt_dir in all_pat_dirs[:1]:
        print("Data dir :" + pt_dir)
        planC = pc.loadDcmDir(pt_dir)
        scanNum = 0
        structNum = 0
        featDict = ibsi1.computeScalarFeatures(scanNum, structNum, settingsFile, planC)
        id['id'] = pt_dir
        featDict = {**id, **featDict}
        featList.append(featDict)
    ibsi1.writeFeaturesToFile(featList, csvFileName)


    # ====== Examples ======

    # Apply filter
    # from cerr.radiomics import filters
    # sobel3M = filters.sobelFilt(volToEval)
    # zScore3M = (volToEval - meanVal) / sigmaVal
    # scanITKObj = scn.getSITKImage(volToEval,xValsV,yValsV,zValsV,orientV)
    # scanCERR3M,imgOriV,imgPosV = scn.getCERRScan(scanITKObj)
    # biasFieldCorrected3M
    # Or any user defined filter

