import pandas as pd

from cerr import plan_container as pc
from cerr.registration import register
from cerr import viewer as vwr
from cerr.utils import mask
import cerr.contour.rasterseg as rs
import numpy as np
import os
from cerr.dataclasses import structure as cerrStr

def calc_feat_diff(matS, pyS, featType):
    featDiffV = []
    for key in matS[featType].keys():
        featDiffV.append(np.mean((matS[featType][key] - pyS[key]) / (matS[featType][key] + np.finfo(float).eps) * 100))
    return featDiffV, np.asarray(list(matS[featType].keys()))


if __name__ == "__main__":

    # Seven Bridges
    # ! mkdir /sbgenomics/workspace/15515009/
    # ! unzip /sbgenomics/project-files/WCMC/15515009.zip -d /sbgenomics/workspace/15515009/
    # ! pip install "pyCERR[napari]@git+https://github.com/cerr/pyCERR"from cerr import plan_container as pc
    # dcmDir = r'/sbgenomics/workspace/15515009'
    # planC = pc.loadDcmDir(dcmDir)
    # from cerr import viewer as vwr
    # vwr.showMplNb(planC, scan_nums=[0], struct_nums=[20,21,22,23,24,25,26,27,28,29,30], dose_nums=[0], windowPreset='Abd/Med')

    #dataDir = r'C:\Users\aptea\Downloads'
    #ctNiiFullFile = os.path.join(dataDir, ctNiiFile)
    #planC = pc.loadNiiScan(ctNiiFullFile,'CT SCAN')
    #parotidSegNiiFullFile = os.path.join(dataDir, parotidSegFile)
    #chewMuscSegNiiFullFile = os.path.join(dataDir, chewMuscSegFile)
    # from cerr import viewer as vwr
    # viewer =  vwr.showNapari(planC,assocScanNum,[leftSCRRIndex, rightSCRRIndex])


    # dataDir = r'N:\data\lymph_node_segmentation\dicom\35207081'
    # planC = pc.loadDcmDir(dataDir)
    #
    # niiFileName = r'N:\data\lymph_node_segmentation\4dStruct.nii.gz'
    # labelDict = {}
    # labelDict['Branch_1'] = 1
    # labelDict['Branch_2'] = 2
    # labelDict['Branch_4'] = 3
    # labelDict['Branch_5'] = 4
    # labelDict['Branch_6'] = 6
    # labelDict['Branch_7'] = 7
    # labelDict['Heart_Base'] = 8
    # # Saven as 4d array
    # pc.saveNiiStructure(niiFileName, labelDict, planC, strNumV=[0,1,2,3,4,5,6], dim=4)
    # # Read nii file containing 4d array
    # planC = pc.loadNiiStructure(niiFileName, 0, planC, labelDict)


    baseScanDir = r'L:\Data\RTOG0617\DICOM\NSCLC-Cetuximab_flat_dirs\0617-381906_09-09-2000-87022'
    baseScanDir = r'N:\data\cardiac_toxicity\Breast_N9000\00001593\1_LtBreast'
    #baseScanDir = r'M:\Data\CERR_test_datasets\CT_RTSTRUCT_RTDOSE_KOKE_Thorax_Test_Dose_Extract_from_OjalaJarkko'





    #baseScanDir = r'M:\Data\HNSCC_fuller_TCIA\DICOM\HNSCC\HNSCC-01-0001\12-01-1998-PETCT HEAD  NECK CA-92442\2-CT Atten Cor Head IN-25068'
    planC = pc.loadDcmDir(baseScanDir)
    #baseScanDir = r'M:\Data\HNSCC_fuller_TCIA\DICOM\HNSCC\HNSCC-01-0001\12-01-1998-PETCT HEAD  NECK CA-92442\4-PET AC-13575'
    #planC = pc.loadDcmDir(baseScanDir, {}, planC)

    from cerr.viewer.pycerr_gui import launch, show

    # --- Blocking (plain scripts) -------------------------------------
    # planC = launch(planC)          # opens viewer; returns when window closes
    #print(len(planC.structure))    # includes anything imported in the GUI
    # launch() mutates planC in place, so your original variable is also current.

    # --- Non-blocking (IPython / Jupyter: run `%gui qt` first) --------
    # %gui qt5
    viewer = show(planC)


    #from cerr import plan_container as pc
    from cerr.imrtp import IMRTPGui
    #planC = pc.loadDcmDir(r"path\to\dicom")
    IMRTPGui(planC)
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [0,1,2], [0], {}, '2d')


    import os
    from cerr import plan_container as pc
    from cerr.dcm_export import rtstruct_iod

    dataDir = r'H:\Public\Jeho\Sample Thymus Contours'
    ptList = os.listdir(dataDir)
    dirToImportList = []
    lungsList = []
    for id in ptList:
        ptDirPath = os.path.join(dataDir, id)
        rtStructFileName = os.path.join(ptDirPath, 'thymus_segmentation.dcm')
        if os.path.exists(rtStructFileName):
            continue
        planC = pc.loadDcmDir(ptDirPath)

        if len(planC.scan) == 1:
            # Read nii seg
            for root, dirs, files in os.walk(ptDirPath):
                for name in files:
                    if 'thymus' in name:
                        planC = pc.loadNiiStructure(os.path.join(ptDirPath,name),0,planC,{'thymus': 1})
            # export rtstruct
            rtstruct_iod.create(structNumV = [0], filePath = rtStructFileName, planC = planC, seriesOpts = {'SeriesDescription':'Exported from pyCERR'})




    dcmDir = r'L:\Maria\LA-NSCLC_Durva_N230\NewN_50_TotN230\pCT\DCM\00665532'
    planC = pc.loadDcmDir(dcmDir)
    from cerr.roe import launch
    win = launch(planC)


    # Planning CTs
    outputDir = r'L:\Maria\LA-NSCLC_Durva_N230\ventilation_features'


    def savePlanningData(outDir, planC):
        try:
            planningCTfile = os.path.join(outDir, 'planningCT.nii.gz')
            lungsFile = os.path.join(outDir, 'planningCTLungMask.nii.gz')
            doseFile = os.path.join(outDir, 'planDose.nii.gz')
            scanSize = planC.scan[0].getScanSize()
            mask3M = np.zeros(scanSize, dtype=bool)
            lungStrNums = []
            lungStrNums.extend(rLungInd)
            lungStrNums.extend(lLungInd)
            if len(planC.dose) != 1 or len(lungStrNums) < 2:
                return
            for strNum in lungStrNums:
                mask3M = mask3M | rs.getStrMask(strNum, planC)
            planC = pc.importStructureMask(mask3M, 0, 'Lungs_AI_generated', planC)
            lungInd = len(planC.structure) - 1
            expansionMargin = 0.1 # cm
            planC = cerrStr.getSurfaceExpand(lungInd, expansionMargin, planC)
            planC.scan[0].saveNii(planningCTfile)
            planC.structure[lungInd].saveNii(lungsFile, planC)
            planC.dose[0].saveNii(doseFile)
        except:
            pass
        return


    #Dataset 2
    baseDir = r'\\pisidsmph\deasylab1\Maria\LA-NSCLC_Durva_N230\N_65_ValidationCohort\DCM\RDRPRS_PET-CT'
    ptList = os.listdir(baseDir)
    dirToImportList = []
    lungsList = []
    for pt in ptList:
        try:
            ptDir = os.path.join(baseDir, pt)
            filesToImport = [f for f in os.scandir(ptDir) if f.is_file()]  #Exclude subdirs
            dirToImportList.append(filesToImport)
            planC = pc.loadDcmDir(filesToImport)
            strNames = [s.structureName for s in planC.structure]
            rLungInd = cerrStr.getMatchingIndex('Lung_R', strNames)
            lLungInd = cerrStr.getMatchingIndex('Lung_L', strNames)
            lungInd = cerrStr.getMatchingIndex('Lungs', strNames)
            noGTVLungInd = cerrStr.getMatchingIndex('Lungs_NOT_GTV', strNames)
            lungDict = {}
            lungDict['rightLung'] = rLungInd
            lungDict['leftLung'] = lLungInd
            lungDict['lungs'] = lungInd
            lungDict['noGTVLungs'] = noGTVLungInd
            lungDict['numDoses'] = len(planC.dose)
            lungsList.append(lungDict)
            outDir = os.path.join(outputDir, pt)
            if os.path.exists(outDir):
                savePlanningData(outDir, planC)
        except:
            continue


    #Dataset 1
    baseDir = r'\\pisidsmph\deasylab1\Maria\LA-NSCLC_Durva_N230\N_113_DerivationCohort\N_113_pCT_4DCT\DCM_RD_RP_RS_pCT_4DCT'
    ptList = os.listdir(baseDir)
    dirToImportList = []
    lungsList = []
    for pt in ptList:
        ptDir = os.path.join(baseDir, pt)
        dirList = os.listdir(ptDir)
        if 'pCT' in dirList:
            try:
                dirToImport = os.path.join(ptDir, 'pCT')
                dirToImportList.append(dirToImport)
                planC = pc.loadDcmDir(dirToImport)
                strNames = [s.structureName for s in planC.structure]
                rLungInd = cerrStr.getMatchingIndex('Lung_R', strNames)
                lLungInd = cerrStr.getMatchingIndex('Lung_L', strNames)
                lungInd = cerrStr.getMatchingIndex('Lungs', strNames)
                noGTVLungInd = cerrStr.getMatchingIndex('Lungs_NOT_GTV', strNames)
                lungDict = {}
                lungDict['rightLung'] = rLungInd
                lungDict['leftLung'] = lLungInd
                lungDict['lungs'] = lungInd
                lungDict['noGTVLungs'] = noGTVLungInd
                lungDict['numDoses'] = len(planC.dose)
                lungsList.append(lungDict)
                outDir = os.path.join(outputDir, pt)
                if os.path.exists(outDir):
                    savePlanningData(outDir, planC)
            except:
                continue
        else:
            continue

    numDosesList = [l['numDoses'] for l in lungsList if l['numDoses'] == 1]
    rLungList = [l['rightLung'] for l in lungsList if len(l['rightLung']) > 0]
    lLungList = [l['leftLung'] for l in lungsList if len(l['leftLung']) > 0]


    #Dataset 3
    baseDir = r'\\pisidsmph\deasylab1\Maria\LA-NSCLC_Durva_N230\NewN_50_TotN230\pCT\DCM'
    dirToImportList = []
    lungsList = []
    ptList = os.listdir(baseDir)
    for pt in ptList:
        try:
            dirToImport = os.path.join(baseDir, pt)
            dirToImportList.append(dirToImport)
            planC = pc.loadDcmDir(dirToImport)
            strNames = [s.structureName for s in planC.structure]
            rLungInd = cerrStr.getMatchingIndex('Lung_R', strNames)
            lLungInd = cerrStr.getMatchingIndex('Lung_L', strNames)
            lungInd = cerrStr.getMatchingIndex('Lungs', strNames)
            noGTVLungInd = cerrStr.getMatchingIndex('Lungs_NOT_GTV', strNames)
            lungDict = {}
            lungDict['rightLung'] = rLungInd
            lungDict['leftLung'] = lLungInd
            lungDict['lungs'] = lungInd
            lungDict['noGTVLungs'] = noGTVLungInd
            lungDict['numDoses'] = len(planC.dose)
            lungsList.append(lungDict)
            outDir = os.path.join(outputDir, pt)
            if os.path.exists(outDir):
                savePlanningData(outDir, planC)
        except:
            continue

    numDosesList = [l['numDoses'] for l in lungsList if l['numDoses'] == 1]
    rLungList = [l['rightLung'] for l in lungsList if len(l['rightLung']) > 0]
    lLungList = [l['leftLung'] for l in lungsList if len(l['leftLung']) > 0]



    #
    from cerr import viewer as vwr
    import ants
    ptDir = r'L:\Maria\LA-NSCLC_Durva_N230\ventilation_features\38145513'
    exhaleFile = os.path.join(ptDir,'exhaleCT.nii.gz')
    txFile = [os.path.join(ptDir,'warp_field.nii.gz')]
    txFile = [os.path.join(ptDir,'ants_regComposite.h5')]
    jacobianFile = os.path.join(ptDir,'jacobian.nii.gz')

    domainImage = ants.image_read(exhaleFile)
    jacobianImage = ants.create_jacobian_determinant_image(domain_image=domainImage,
                                                           tx=txFile, do_log=False, geom=False)

    inhaleFile = os.path.join(ptDir,'inhaleCT.nii.gz')
    inhaleLungMaskFile =  os.path.join(ptDir,'inhaleLungMask.nii.gz')
    exhaleLungMaskFile =  os.path.join(ptDir,'exhaleLungMask.nii.gz')
    planC = pc.loadNiiScan(exhaleFile, 'CT SCAN')
    planC = pc.loadNiiScan(jacobianFile, 'Jacobian','', planC)
    planC = pc.loadNiiScan(inhaleFile, 'CT SCAN', '', planC)
    planC = pc.loadNiiStructure(exhaleLungMaskFile, 0, planC, {'exhale_Lung':1})
    planC = pc.loadNiiStructure(inhaleLungMaskFile, 2, planC, {'inhale_Lung':1})
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0,1,2], [0,1], [], {}, '2d')


    # View Inhale/Exhale registration
    dirName = r'L:\Maria\LA-NSCLC_Durva_N230\N_65_ValidationCohort\DCM\test\ventilation_features\38128946'
    exhaleNii = os.path.join(dirName, 'exhaleCT.nii.gz')
    warpedInhaleNii = os.path.join(dirName, 'warped_inhaleCT.nii.gz')
    inhaleNii = os.path.join(dirName, 'inhaleCT.nii.gz')
    planC = pc.loadNiiScan(exhaleNii, 'MR SCAN')
    planC = pc.loadNiiScan(warpedInhaleNii, 'MR SCAN', '', planC)
    planC = pc.loadNiiScan(inhaleNii, 'MR SCAN', '', planC)
    len(planC.scan)
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0,1,2], [], [], {}, '2d')




    # Read NM scan
    dcmDir = r'\\pensmph\MPHShared\Public\Alexandre\forAditya\CT3'
    planC = pc.loadDcmDir(dcmDir)

    niiFile = r'\\pensmph\MPHShared\Public\Alexandre\forAditya\SPECT1\1.2.752.37.54.2572.174742083697393884255028573616428379114.nii.gz'
    planC = pc.loadNiiScan(niiFile, 'NM SCAN', '', planC)



    # SUV QA
    from cerr import plan_container as pc
    from cerr.contour import rasterseg as rs
    import numpy as np

    baseDir = r'C:\software\suv_computation\DRO'
    dirList = ['DRO_0_0', 'DRO_1_0','DRO_2_0', 'DRO_2_1_0', 'DRO_2_1_1', 'DRO_2_1_2', 'DRO_2_2_0', 'DRO_2_2_1','DRO_2_2_2',
               'DRO_2_3', 'DRO_2_4','DRO_2_5', 'DRO_2_6_0', 'DRO_2_6_1', 'DRO_2_6_2','DRO_3_0','DRO_3_1',
               'DRO_3_2_0','DRO_3_2_1','DRO_3_2_2','DRO_3_3_0','DRO_3_3_1','DRO_3_4_0','DRO_3_4_1', 'DRO_3_4_2',
               'DRO_3_5_0','DRO_3_5_1','DRO_3_5_2','DRO_4_0', 'DRO_4_1', 'DRO_4_2', 'DRO_5_0']
    suvStats = []
    for petDir in dirList:
        dcmDir = os.path.join(baseDir, petDir)
        # 'suvType': Choose from 'BW', 'BSA', 'LBM', 'LBMJANMA'
        planC = pc.loadDcmDir(dcmDir)
        mask3M = rs.getStrMask(0, planC)
        scan3M = planC.scan[0].getScanArray()
        scanV = scan3M[mask3M]
        suvStats.append([petDir, np.min(scanV), np.median(scanV), np.max(scanV)])



    dcmDir = r'L:\Data\RTOG0617\DICOM\NSCLC-Cetuximab_flat_dirs\0617-259694_09-09-2000-79111'
    dcmDir = r'L:\Sharif\maite\debug\pyCERR'
    dcmDir = r'M:\Data\HNSCC_fuller_TCIA\DICOM\HNSCC\HNSCC-01-0142\03-03-2001-PETCT HEAD  NECK CA-47428\3-CT SOFT-24171'
    planC = pc.loadDcmDir(dcmDir)

    import csv
    from cerr import dvh as cerrDvh
    from cerr.dataclasses import dose as cerrDose

    dvhFile = r'\\pisidsmph\deasylab3\Jung\REQUITE\DVH data_2025-01-13_17-07-09.tsv'
    dvhMetricsFile = r'\\pisidsmph\deasylab3\Jung\REQUITE\dvh_metrics.csv'
    stdFrxSize = 2
    abRatioDict = {'rectum': 3, 'bladder': 6}
    pctV = range(0,100,1) # percentage
    volCutPctV = range(0,100,1) # percentage
    doseCutV = range(0,85,1)
    organList = list(abRatioDict.keys())
    doseMetrics = []# Gy
    with open(dvhFile, 'r', newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader) # skip header
        for row in reader:
            rowDict = eval(row[3])
            subjectId = rowDict['subject_id']
            if subjectId != 'RQ05042-7': #'RQ14094-9','RQ14099-6','RQ14108-5','RQ14120-7','RQ14140-3','RQ14199-7','RQ14261-2','RQ14394-7','RQ14587-4'
                continue
            organ = rowDict['organ']
            if organ not in organList:
                continue
            abRatio = abRatioDict[organ]
            cumuDvhM = np.array(rowDict['data'])
            if cumuDvhM.shape[0] <= 1:
                continue
            if cumuDvhM[-1,1] > cumuDvhM[-2,1]:
                cumuDvhM[-1,1] = cumuDvhM[-2,1]
            if cumuDvhM[0,1] == 100:
                volType = 0
            else:
                volType = 1
            volDiffHistV = - np.diff(cumuDvhM[:,1], append=cumuDvhM[-1,1])
            binwidth = cumuDvhM[1,0] - cumuDvhM[0,0]
            doseDiffBinsV = cumuDvhM[:,0] + binwidth/2
            rxDose = 70 # input per subject
            numFracts = 32 # input per subject
            inputFrxSize = rxDose / numFracts
            inputFrxSize = 2
            fxCorrectDoseBinsV = cerrDose.fractionNumCorrect(doseDiffBinsV, stdFrxSize, abRatio, None, inputFrxSize)

            # Calculate metrics
            #cerrDvh.MOCx(fxCorrectDoseBinsV, volDiffHistV, 5)
            #cerrDvh.MOHx(fxCorrectDoseBinsV, volDiffHistV, 5)
            #cerrDvh.Vx(fxCorrectDoseBinsV, volDiffHistV, 0, volumeType=0)
            #dx = cerrDvh.Dx(fxCorrectDoseBinsV, volDiffHistV, 0, volumeType=volType)

            meanDose = cerrDvh.meanDose(fxCorrectDoseBinsV, volDiffHistV)
            medianDose = cerrDvh.medianDose(fxCorrectDoseBinsV, volDiffHistV)
            maxDose = cerrDvh.maxDose(fxCorrectDoseBinsV, volDiffHistV)
            minDose = cerrDvh.minDose(fxCorrectDoseBinsV, volDiffHistV)
            mohxV = [cerrDvh.MOHx(fxCorrectDoseBinsV, volDiffHistV, pct) for pct in pctV]
            mocxV = [cerrDvh.MOCx(fxCorrectDoseBinsV, volDiffHistV, pct) for pct in pctV]
            medDose = cerrDvh.medianDose(fxCorrectDoseBinsV, volDiffHistV)
            dxV = [cerrDvh.Dx(fxCorrectDoseBinsV, volDiffHistV, volCut, volumeType=volType) for volCut in volCutPctV]
            vxV = [cerrDvh.Vx(fxCorrectDoseBinsV, volDiffHistV, doseCut, volumeType=volType) for doseCut in doseCutV]

            metricsDict = {}
            metricsDict['id'] = subjectId
            metricsDict['organ'] = organ
            metricsDict['meanDose'] = meanDose
            metricsDict['medianDose'] = medianDose
            metricsDict['maxDose'] = maxDose
            metricsDict['minDose'] = minDose
            metricsDict['mohxV'] = mohxV
            metricsDict['mocxV'] = mocxV
            metricsDict['dxV'] = dxV
            metricsDict['vxV'] = vxV
            doseMetrics.append(metricsDict)

    fieldnames = list(metricsDict.keys())
    with open(dvhMetricsFile, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(doseMetrics)



    dataDir = r'N:\data\lymph_node_segmentation\dicom\35207081'
    planC = pc.loadDcmDir(dataDir)

    lungNiiScan = r'C:\Users\aptea\Downloads\sample_lung_ct_scan_3D.nii'
    lungMaskNiiOut = r'C:\Users\aptea\Downloads\model_Mhub_sample_lung_ct_scan_3D.nii'
    planC = pc.loadNiiScan(lungNiiScan, 'CT Scan')
    planC = pc.loadNiiStructure(lungMaskNiiOut, 0, planC, {1: 'GTV'})
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [0], [], {}, '2d')

    baseScanDir = r'L:\Data\RTOG0617\DICOM\NSCLC-Cetuximab_flat_dirs\0617-381906_09-09-2000-87022'
    planC = pc.loadDcmDir(baseScanDir)


    dcmDir = r'\\vpensmph\deasylab3\data\fromKaiming\nii\gdrive\209764\scans\DICOM'
    segDir = r'\\vpensmph\deasylab3\data\fromKaiming\nii\gdrive\209764\assessors'
    niiSegFile = r'\\vpensmph\deasylab3\data\fromKaiming\nii\gdrive\209764\assessors\maskXC.nii'
    planC = pc.loadDcmDir(dcmDir)
    planC = pc.loadNiiStructure(niiSegFile, 0, planC)
    #planC = pc.loadDcmDir(segDir, planC)
    triggerTimeV = np.array([s.scanInfo[0].triggerTime for s in planC.scan])
    indSortV = np.argsort(triggerTimeV)
    timeDiffV = np.diff(triggerTimeV[indSortV])



    # imageDir = r'N:\data\Breast-MRI-NACT-Pilot\manifest-RbPGRCVv7392292744865323559\Breast-MRI-NACT-Pilot\UCSF-BR-07\09-12-1991-228612-MR BREAS UNIT-23936\1.000000-Axial-T1 locator-80183'
    # segDir = r'N:\data\Breast-MRI-NACT-Pilot\manifest-RbPGRCVv7392292744865323559\Breast-MRI-NACT-Pilot\UCSF-BR-07\09-12-1991-228612-MR BREAS UNIT-23936\32001.000000-Breast Tissue Segmentation-26968'
    # planC = pc.loadDcmDir(imageDir)
    # planC = pc.loadDcmDir(segDir, {}, planC)

    niiScanFile = r'L:\Data\fromKaiming\nii\gdrive\209764\scans\scans_Perfusion_post_10_phase_inj_delay_20130423133543_11.nii.gz'
    niiMaskFile = r'L:\Data\fromKaiming\nii\gdrive\209764\assessors\maskXC.nii'
    planC = pc.loadNiiScan(niiScanFile, 'MR SCAN')
    planC = pc.loadNiiStructure(niiMaskFile, 0, planC)

    # Interpolation
    from cerr import plan_container as pc
    from cerr.radiomics.preprocess import imgResample3D, getResampledGrid

    # Define resampled scan resolution and method
    outputResV = [0.03,0.03,0.03]
    scanInterpMethod = 'sitkLanczosWindowedSinc'

    # Define dicom directory location
    dcmDir = r'N:\data\ROBIN\supplement_funding_2024\MR\dicom\38127154\pre_treatment\2020-9-11_Axial.T2.oblique'

    # Define location of nii file to write resampled scan
    resampNiiSaveName = r'N:\data\ROBIN\supplement_funding_2024\MR\dicom\38127154\pre_treatment_2020-9-11_Axial.T2.oblique_resamp.nii.gz'

    # Load dicom to planC
    planC = pc.loadDcmDir(dcmDir)

    # Define scan index to resample. 0 since only one scan is present
    scanNum = 0

    # Resample scan
    xValsV, yValsV, zValsV = planC.scan[scanNum].getScanXYZVals()
    scan3M = planC.scan[scanNum].getScanArray()
    [xResampleV,yResampleV,zResampleV] = getResampledGrid(outputResV,\
                                             xValsV, yValsV, zValsV)
    resampScan3M = imgResample3D(scan3M, xValsV, yValsV, zValsV,\
                            xResampleV,yResampleV,zResampleV, scanInterpMethod)

    # Import resampled scan to planC
    planC = pc.importScanArray(resampScan3M, xResampleV,yResampleV,zResampleV, 'MR', scanNum, planC)
    resampScanNum = len(planC.scan) - 1

    # Save Resampled scan to Nii
    planC.scan[resampScanNum].saveNii(resampNiiSaveName)

    # Visualize original and resampled scans
    from cerr import viewer as vwr
    vwr.showNapari(planC, [scanNum])
    vwr.showNapari(planC, [resampScanNum])



    from cerr.radiomics.preprocess import imgResample3D
    dcmDir = r'N:\data\ROBIN\supplement_funding_2024\planningCT\38019597\Plan_RECTUM'
    planC = pc.loadDcmDir(dcmDir)
    strNames = [s.structureName for s in planC.structure]
    scanInterpMethod = 'sitkLanczosWindowedSinc'
    extrapScanVal = 0
    inPlaneFlag = False
    scanNum = 0
    strNum = 4
    xValsV, yValsV, zValsV = planC.scan[scanNum].getScanXYZVals()
    mask3M = rs.getStrMask(strNum, planC)
    rV,cV,sV = np.where(mask3M)
    uniqSv = np.sort(np.unique(sV))
    allSlcV = np.arange(uniqSv[0], uniqSv[-1]+1)
    missingSlcV = allSlcV[[slc not in uniqSv for slc in allSlcV]]
    scanSize = planC.scan[scanNum].getScanSize()
    filledMask3M = mask3M.copy()
    for slc in missingSlcV:
        minSlc = uniqSv[max(np.where(uniqSv < slc)[0])]
        maxSlc = uniqSv[min(np.where(uniqSv > slc)[0])]
        zContourSlcValsV = np.asarray([zValsV[minSlc],zValsV[maxSlc]], dtype=float)
        contouredMask3M = mask3M[:,:,[minSlc,maxSlc]].astype(float)
        xyzVals = np.meshgrid(xValsV, yValsV, zValsV[slc])
        resampScan3M = imgResample3D(contouredMask3M, xValsV, yValsV, zContourSlcValsV,\
                                xValsV, yValsV, [zValsV[slc]-0.001,zValsV[slc]], scanInterpMethod,
                                extrapScanVal, inPlaneFlag)
        filledMask3M[:,:,slc] = resampScan3M[:,:,1] >= 0.5

    planC = pc.importStructureMask(filledMask3M, scanNum, 'filled GTV', planC)
    vwr.showNapari(planC, scan_nums=[0,1], struct_nums=[4, len(planC.structure)-1])

    # Mesh approach
    from skimage import measure
    import trimesh
    from cerr.contour import rasterseg as rs
    maskForShape3M = np.pad(mask3M, ((1,1),(1,1),(1,1)),
                            mode='constant', constant_values=((0, 0),))
    verts, faces, normals, values = measure.marching_cubes(maskForShape3M, level=0.5, spacing=[1,1,1]) # image units

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    slice_path = mesh.section(plane_origin=[0,0,missingSlcV[2]], plane_normal=[0,0,1])
    boundary_coordinates = slice_path.vertices
    rowV = boundary_coordinates[:,0]
    colV = boundary_coordinates[:,1]
    numRows, numCols, numSlcs = mask3M.shape
    maskM = rs.polyFill(rowV, colV, numRows, numCols)

    # # ROBIN DICOM
    # dcmDir = r'L:\Data\ROBIN\Cornell_data\15515009'
    # planC = pc.loadDcmDir(dcmDir)

    planC = pc.loadDcmDir(r'\\vpensmph\DeasyLab1\Maria\LA-NSCLC_Durva_N230\N_113_DerivationCohort\N_113_pCT_4DCT\DCM_RD_RP_RS_pCT_4DCT\35574178\PlanningCT')
    fname = r'L:\Aditya\forSharif\test_1.json'
    bb = planC.structure[3].saveContoursToJson(planC, False, fname)

    from cerr import viewer as vwr
    fpath = r'L:\Aditi\forAditya\napari_debug'
    fname = r'p33_v1_orig.pkl'
    strNum = 0
    planC = pc.loadPlanCFromPkl(os.path.join(fpath,fname))
    tthpIDx = [planC.scan[scnNum].scanType for scnNum in range(len(planC.scan))].index('TimeToHalfPeak')
    vwr.showNapari(planC, scan_nums=[0,tthpIDx], struct_nums=[strNum])

    ##  MAITE read scan from DB and sacnInfoList
    dirPath = r'N:\data\ROBIN\supplement_funding_2024\MR\dicom'
    dirNames = os.listdir(dirPath)
    numTimes = []
    prePostTimes = []
    for dirName in dirNames:
        subDirName = os.path.join(dirPath,dirName)
        numTimes.append(len(os.listdir(subDirName)))
        if 'pre_treatment' in os.listdir(subDirName) and 'pre_surgery' in os.listdir(subDirName):
            prePostTimes.append(True)
        else:
            prePostTimes.append(False)
    print('Number of patients with Pre and Post:', np.sum(prePostTimes))


    dcmDir = r'M:\Aditya\Cornell_AI_Imaging_course\notebooks\heartOAR\input_dcm\0617-259694_09-09-2000-79111'
    planC = pc.loadDcmDir(dcmDir)
    strList = cerrStr.getJsonList(0, planC)
    del planC.structure[0]
    planC = cerrStr.importJson(planC, strList=strList)



    dcmDir = r'M:\Aditya\Cornell_AI_Imaging_course\notebooks\heartOAR\input_dcm\0617-259694_09-09-2000-79111'
    dcmDir = r'\\pensmph6\treatplanapp\dicom_images\MAITE\001'
    planC = pc.loadDcmDir(dcmDir)

    # Save structure polygons to json file
    jsonFileDir = r'L:\Aditya\forSharif'
    for structNum in [20,42]:
        structName = planC.structure[structNum].structureName
        jsonFile = os.path.join(jsonFileDir, structName + '.json')
        planC.structure[structNum].saveContoursToJson(jsonFile, planC)


    # Save scan to nii file and scanInfo to a JSON file
    import json
    niiFile = r'L:\Aditya\forSharif\testscan.nii.gz'
    infoJsonFile = r'L:\Aditya\forSharif\scanInfo.json'
    planC.scan[0].saveNii(niiFile)
    scanInfoList = planC.scan[0].getDcmScanInfo()
    with open(infoJsonFile, 'w') as file:
        json.dump(scanInfoList, file, indent=4)

    # Load scanInfo and scan nii file into planC
    planC = pc.loadScanFromDB(scanInfoList, niiFile, planC)

    # Load scanInfo json file
    with open(infoJsonFile, 'r') as file:
        scanInfoData = json.load(file)


    with open(jsonFile, 'r') as file:
        structData = json.load(file)

    # Get structure name and color
    structName = structData['name']
    structColor = structData['color']

    # Get a list of all segments and corresponding SPOInstanceUID
    allSegs = []
    refSopInstanceUIDs = []
    for ctr in structData['contour']:
        for iSeg, seg in enumerate(ctr['segments']):
            allSegs.append(seg)
            refSopInstanceUIDs.append(ctr['referencedSopInstanceUID'])




    ### Debug for Gabor

    import os
    from cerr import plan_container as pc
    import SimpleITK as sitk
    import numpy as np
    from scipy import stats
    import cerr.dataclasses.scan as cerrScn
    from cerr.contour import rasterseg as rs
    from cerr.radiomics import ibsi1

    dataDir = r'\\vpensmph\deasylab3\data\ROBIN\PANTHER'
    gtvNames = r'\\vpensmph\deasylab3\data\ROBIN\analysis\GTV_names.csv'
    filterSettingsFile = r'\\vpensmph\deasylab3\data\ROBIN\analysis\radiomics_settings\gabor_filter_texture.json'


    import csv
    gtvNamesDict = {}
    with open(gtvNames, mode ='r') as file:
           csvFile = csv.DictReader(file)
           for lines in csvFile:
                gtvNamesDict[lines['id']] = lines['gtv']
    print(gtvNamesDict)


    dcmDirList = [f.path for f in os.scandir(dataDir) if f.is_dir()]

    def hu_to_density(hu):
        """
        Converts Hounsfield Units (HU) to density (g/cm³).

        Args:
            hu (float or numpy.ndarray): Hounsfield Units value(s).

        Returns:
            float or numpy.ndarray: Density value(s) in g/cm³.
        """
        if isinstance(hu, (int, float)):
            if hu < -1000:
                return 0  # or handle as appropriate for your application
            elif hu < 0:
                return hu / 1000 + 1
            else:
                return hu / 1955 + 1
        elif isinstance(hu, np.ndarray):
             density = np.zeros_like(hu, dtype=np.float32)
             density[hu < -1000] = 0
             density[(hu >= -1000) & (hu < 0)] = hu[(hu >= -1000) & (hu < 0)] / 1000 + 1
             density[hu >= 0] = hu[hu >= 0] / 1955 + 1
             return density
        else:
            raise TypeError("Input must be a number or a NumPy array.")


    from cerr.radiomics import texture_utils

    dirInd = 7 # Complete response
    dirInd = 17 # Complete response
    #dirInd = 15 # Disease progression
    #dirInd = 11 # Partial response
    d = dcmDirList[dirInd]
    baseDir, id = os.path.split(d)
    print("Data dir: " + id)

    planningCTdir = os.path.join(d,'planningCT')
    planC = pc.loadDcmDir(planningCTdir)

    strNames = [s.structureName for s in planC.structure]

    gtvName = gtvNamesDict[id]
    gtvInid = strNames.index(gtvName)

    scanNum = cerrScn.getScanNumFromUID(planC.structure[gtvInid].assocScanUID, planC)
    structNum = gtvInid

    # Create density scan
    hu3M = planC.scan[scanNum].getScanArray()
    density3M = hu_to_density(hu3M)

    density3M[density3M < 0.4] = 0

    # Add density scan to planC
    xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
    planC = pc.importScanArray(density3M, xV, yV, zV, 'Density', scanNum, planC)

    densityScanNum = len(planC.scan) - 1

    # Copy structNum to density scan
    planC = cerrStr.copyToScan(structNum, densityScanNum, planC)

    structNum = len(planC.structure) - 1

    planC = texture_utils.generateTextureMapFromPlanC(planC, densityScanNum, structNum, filterSettingsFile)


    from cerr import viewer as vwr

    filtScanNum = len(planC.scan) - 1
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [densityScanNum, filtScanNum], [structNum], [], {}, '2d')




    import pandas as pd
    from cerr import viewer as vwr
    from cerr.dataclasses import scan as cerrScn
    from cerr.contour import rasterseg as rs
    from skimage.io import imsave, imshow

    segDir = r'\\vpensmph\deasylab3\data\TCGA-BRCA\manifest-25vRPwyh8987165612391086998\RTSTRUCTs_subdirs'
    dataLocations = r'N:\data\TCGA-BRCA\data_locations_selected.csv'
    snapshotDir = r'N:\data\TCGA-BRCA\manifest-25vRPwyh8987165612391086998\segSnapshots'
    dataLocs = pd.read_csv(dataLocations)
    numStructV = []
    failedIDList = []
    for i in range(len(dataLocs)):
        id = dataLocs.iloc[i]['Pt']
        try:
            imageDir = dataLocs.iloc[i]['Post contrast path']
            segPtDir = os.path.join(segDir,id)
            planC = pc.loadDcmDir(imageDir)
            planC = pc.loadDcmDir(segPtDir, {}, planC)
            numStructs = len(planC.structure)
            numStructV.append(numStructs)
            if numStructs == 0:
                continue
            structNum = 0
            scanNum = cerrScn.getScanNumFromUID(planC.structure[structNum].assocScanUID, planC)
            viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                        vwr.showNapari(planC, [scanNum], [0], [], {}, '2d')
            snapFile = os.path.join(snapshotDir, id + '.png')
            mask3M = rs.getStrMask(structNum, planC)
            scan3M = planC.scan[scanNum].getScanArray()
            minScan = np.quantile(scan3M,0.05)
            maxScan = np.quantile(scan3M,0.95)
            rV, cV, sV = np.where(mask3M)
            midSliceInd = int(np.round(sV.mean()))
            # update viewer to display the central slice and capture screenshot
            xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
            viewer.dims.set_point(2, zV[midSliceInd])

            scan_layer[0].opacity = 1
            scan_layer[0].contrast_limits_range = [minScan, maxScan]
            scan_layer[0].contrast_limits = [minScan, maxScan]
            scan_layer[0].gamma = 0.7
            viewer.camera.zoom = 40
            #viewer.camera.center = (0, 100, 100)
            screenshot = viewer.screenshot(size =(600, 600))
            viewer.close()
            imsave(snapFile, screenshot)
        except:
            failedIDList.append(id)


    from cerr import viewer as vwr
    import scipy.io as sio
    from cerr.dcm_export import rtstruct_iod
    maskDir = r'N:\data\TCGA-BRCA\manifest-25vRPwyh8987165612391086998\exportedMasks\mat'
    rtstructExportDir = r'N:\data\TCGA-BRCA\manifest-25vRPwyh8987165612391086998\RTSTRUCTs'
    DICOMsegDir = r'\\vpensmph\deasylab3\data\TCGA-BRCA\manifest-25vRPwyh8987165612391086998\DICOMforSeg'

    id = 'TCGA-E2-A1LG'
    postDir = os.path.join(DICOMsegDir, id, 'Post')
    preDir = os.path.join(DICOMsegDir, id, 'Pre')
    segFile = os.path.join(maskDir, id + '.mat')
    planC = pc.loadDcmDir(postDir)
    mask3M = sio.loadmat(segFile)['mask3M']
    scanNum = len(planC.scan) - 1
    #planC = pc.importStructureMask(np.flip(mask3M,(0,2)), scanNum, 'tumor', planC)
    aa = np.flip(mask3M,(1))
    aa = mask3M
    aa = np.roll(aa,(12,10,10),axis=(0,1,2))
    planC = pc.importStructureMask(aa, scanNum, 'tumor', planC)
    planC = pc.importStructureMask(mask3M, scanNum, 'tumor', planC)
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [0], [], {}, '2d')

    # Export RTSTRUCT
    rtstructFile = os.path.join(rtstructExportDir, id + '.dcm')
    rtstruct_iod.create(structNumV = [1], filePath = rtstructFile, planC = planC, seriesOpts = {'SeriesDescription':'Exported from pyCERR'})



    segFileDir= r'\\vpensmph\deasylab3\data\TCGA-BRCA\manifest-25vRPwyh8987165612391086998\RTSTRUCTs_subdirs'
    id = 'TCGA-AR-A1AQ'
    imageDir = r'\\vpensmph\deasylab3\data\TCGA-BRCA\manifest-25vRPwyh8987165612391086998\TCGA-BRCA\TCGA-AR-A1AQ\11-21-2001-NA-MRI - BREAST-98628\8.000000-VIBRANT-56538'
    ptSegFileDir = os.path.join(segFileDir,id)
    planC = pc.loadDcmDir(imageDir)
    planC = pc.loadDcmDir(ptSegFileDir, {}, planC)


    # DCE MRI
    dataDir = r'M:\Data\DCE_OMT\dicom_WeiHuang_and_TCIA\BreastChemo1\BreastChemo1'
    scanNii = r'M:\Data\DCE_OMT\dicom_WeiHuang_and_TCIA\BreastChemo1\BreastChemo1_test.nii.gz'
    segFile = os.path.join(dataDir,'bc1v1segmented.nii.gz')
    planC = pc.loadNiiScan(scanNii, 'MR')
    planC = pc.loadNiiStructure(segFile, 0, planC)
    #planC = cerrStr.getSurfaceExpand(0,-0.25, planC, False)
    #mask3M = rs.getStrMask(0, planC)
    #mask3M[:,:,np.arange(61,82,1)] = 0
    #planC = pc.importStructureMask(mask3M, 0, 'test', planC)
    planC = cerrStr.getSurfaceExpand(0, -0.25, planC, True)

    planC = cerrStr.getSurfaceExpand(0, -0.25, planC, True)
    planC = cerrStr.getSurfaceExpand(0, 0.25, planC, True)

    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [0], [], {}, '2d')



    # PostNAC missing cases
    dcmDir = r'S:\Amita_Dave_Collaboration_2019\23-191\23_191_Post_NAC\RIA_23-191_post_084'
    planC = pc.loadDcmDir(dcmDir)


    # Gabor texture image
    dcmDir = r'\\vpensmph\deasylab3\data\ROBIN\PANTHER\11566846\planningCT'
    settingsFile = r'\\vpensmph\deasylab3\data\ROBIN\analysis\radiomics_settings\gabor_filter_texture.json'
    planC = pc.loadDcmDir(dcmDir)
    from cerr.radiomics import texture_utils
    # Path to JSON settings file with filter parameters
    paramS, __ = texture_utils.loadSettingsFromFile(settingsFile)
    # Compute filter response
    scanNum = 0
    strNum = 0
    planC = texture_utils.generateTextureMapFromPlanC(planC, scanNum, strNum, settingsFile)
    filtIdx = len(planC.scan)-1         # Index of filtered scan


    # DVH metrics
    eclipseDir = r'H:\Public\Lei\For Aditya\From Eclipse'
    mimRtstructDir = r'H:\Public\Lei\For Aditya\From MIM\2023-03__Studies\GOJ^BOGUSLAW_35320128_RTst_2023-03-08_092130_L3_TEST-.LSP.for.L3_n1__00000'

    planC = pc.loadDcmDir(eclipseDir)

    planC = pc.loadDcmDir(mimRtstructDir, {}, planC)

    from cerr import dvh
    import numpy as np

    structNum = 31
    doseNum = 0
    print(planC.structure[structNum].structureName)
    dosesV, volsV, isErr = dvh.getDVH(structNum, doseNum, planC)
    binWidth = 0.015
    doseBinsV,volHistV = dvh.doseHist(dosesV, volsV, binWidth)
    percent = 70
    dvh.Dx(doseBinsV,volHistV,10,0)


    # LAVA Post
    dceDir = r'\\vpensmph\deasylab3\data\ROBIN\PANTHER\12328465\2020-10__Studies\WCM-ROBIN.1U54CA274291-01--12328465_12328465_MR_2020-10-28_130242_._30.sec.3D.Ax.LAVA.POST_n150__00000'
    dceDir = r'\\vpensmph\deasylab3\data\ROBIN\PANTHER\12328465\2020-10__Studies\WCM-ROBIN.1U54CA274291-01--12328465_12328465_MR_2020-10-28_130242_._(10346.11.1)-(10346.10.1)_n150__00000'
    dwiDir = r'\\vpensmph\deasylab3\data\ROBIN\PANTHER\12328465\2020-10__Studies\WCM-ROBIN.1U54CA274291-01--12328465_12328465_MR_2020-10-28_130242_._Ax.DWI.B-1000.OBLIQUE_n102__00000'
    adcDir = r'\\vpensmph\deasylab3\data\ROBIN\PANTHER\12328465\2020-10__Studies\WCM-ROBIN.1U54CA274291-01--12328465_12328465_MR_2020-10-28_130242_._ADC.(10^-6.mm².s)_n51__00000'
    planC = pc.loadDcmDir(dwiDir)
    planC = pc.loadDcmDir(adcDir, {}, planC)

    dcmDir = r'\\vpensmph\deasylab3\data\ROBIN\PANTHER\15240971\planningCT'
    planC = pc.loadDcmDir(dcmDir)

    # Load Planning CT DICOM
    planCtDir = r'L:\Data\Cornell_Panther\PANTHER-ROBIN\pixnat_download\planningCT'
    planC = pc.loadDcmDir(planCtDir)

    # Load daily CBCT images
    cbctDir = r'L:\Data\Cornell_Panther\PANTHER-ROBIN\pixnat_download\CBCT'
    planC = pc.loadDcmDir(cbctDir,{},planC)

    # Register images
    transformSaveDir = r'L:\Data\Cornell_Panther\PANTHER-ROBIN\pyCERR_transforms'
    baseScanIndexV = [1] #[1,2,3,4,5]
    baseScanIndexV = [1]
    movScanIndex = 0

    for baseScanIndex in baseScanIndexV:

        # DVF file name
        baseMovUIDs = planC.scan[baseScanIndex].scanUID + '_' + planC.scan[movScanIndex].scanUID
        vfFileName = os.path.join(transformSaveDir,baseMovUIDs,'vf.nii.gz')

        # Segmentation file name
        segFileName = os.path.join(transformSaveDir,baseMovUIDs,'GTV.nii.gz')

        # Deformed scan file name
        defScanFileName = os.path.join(transformSaveDir,baseMovUIDs,'deformedPlanningCT.nii.gz')

        # Load deformed segmentation
        planC = pc.loadNiiStructure(segFileName, baseScanIndex, planC)

        # Load deformed scan
        planC = pc.loadNiiScan(defScanFileName, 'CT SCAN', '', planC)

        # Load DVF
        planC = pc.loadNiiVf(vfFileName, baseScanIndex, planC)



    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [0], [], {}, '2d')


    scanFile = r'S:\Amita Dave HN\PURE-01\T2w_n4_corrected\HSR_PURE-01_001-089D20190115_HSR_E7874_N4_scan.nii.gz'
    segFile = r'S:\Amita Dave HN\PURE-01\T2w_n4_corrected\HSR_PURE-01_001-089D20190115_HSR_E7874_seg.nii.gz'
    planC = pc.loadNiiScan(scanFile, 'MR SCAN')
    planC = pc.loadNiiStructure(segFile, 0, planC)


    from cerr.dataclasses import structure as cerrStr


    dcmDir = r'M:\Aditya\Cornell_AI_Imaging_course\notebooks\heartOAR\input_dcm\0617-259694_09-09-2000-79111'
    heartSubSegDict = {2: 'DL_AORTA', 3: 'DL_LA',
                       4: 'DL_LV', 5: 'DL_RA',
                       6: 'DL_RV', 7: 'DL_IVC',
                       8: 'DL_SVC', 9: 'DL_PA'}
    periLabelDict = {1: 'DL_Pericardium'}
    planC = pc.loadDcmDir(dcmDir)
    segFile = r'C:\Users\aptea\Downloads\0617-259694_09-09-2000-79111_scan_3D_pericardium.nii.gz'
    planC = pc.loadNiiStructure(segFile, 0, planC, periLabelDict)
    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [10], [], {}, '2d')



    scanFile = r'S:\Amita Dave HN\PURE-01\PIXNAT_HSR_ROI_2024-12-13\HSR_PURE-01_001-089\D20190115_HSR_E7874\T2\NIfTI\401-HSR_PURE-01_001-089_T2w_TSE_ax-20190115134725.nii'
    segFile = r'S:\Amita Dave HN\PURE-01\PIXNAT_HSR_ROI_2024-12-13\HSR_PURE-01_001-089\D20190115_HSR_E7874\T2\NIfTI\mask_mc-t2-added.nii.gz'
    planC = pc.loadNiiScan(scanFile, 'MR SCAN')
    planC = pc.loadNiiStructure(segFile, 0, planC)

    dcmDir = r'L:\Data\Cornell_Panther\27705804\20190406-1.2.840.113619.2.55.3.2970255078.994.1660932669.244'
    planC = pc.loadDcmDir(dcmDir)
    segSaveName = r'S:\Amita Dave HN\PURE-01\PIXNAT_HSR_ROI_2024-12-13\HSR_PURE-01_001-053\D20180608_HSR_E4434\T2\mask_1.nii.gz'
    planC.structure[0].saveNii(segSaveName, planC)



    scanFile = r'S:\Amita Dave HN\PURE-01\PIXNAT_HSR_ROI_2024-12-13\HSR_PURE-01_001-089\D20190115_HSR_E7874\T2\NIfTI\401-HSR_PURE-01_001-089_T2w_TSE_ax-20190115134725.nii'
    segFile = r'S:\Amita Dave HN\PURE-01\PIXNAT_HSR_ROI_2024-12-13\HSR_PURE-01_001-089\D20190115_HSR_E7874\T2\NIfTI\mask_mc-t2-added.nii.gz'

    scanFile = r'\\vpensmph\DeasyLab2\Data\HN_OPC_CT_PET\dicom\testPyCERR\nii\phantom\fdg_scan.nii.gz'
    segFile = r'\\vpensmph\DeasyLab2\Data\HN_OPC_CT_PET\dicom\testPyCERR\nii\phantom\gtv_seg.nii.gz'
    planC = pc.loadNiiScan(scanFile, 'MR SCAN')
    planC = pc.loadNiiStructure(segFile, 0, planC)
    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [0], [], {}, '2d')


    dcmDir = r'S:\Amita_Dave_Collaboration_2019\23-191\organizedZscoreSeg_Both_Pre_NAC_MRI\T2w\RIA_23-191_000_pre_106_N4_scan.nii.gz'
    planC = pc.loadDcmDir(dcmDir)

    ptIndex = 0
    planC = cerrStr.copyToScan(0,ptIndex,planC)

    exportDir = r'\\vpensmph\DeasyLab2\Data\HN_OPC_CT_PET\dicom\testPyCERR\nii\10077530'
    scanFile = os.path.join(exportDir, 'pt_scan.nii.gz')
    fmisoFile = os.path.join(exportDir, 'fmiso_pred.nii.gz')
    segFile = os.path.join(exportDir, 'gtv_seg.nii.gz')
    planC = pc.loadNiiScan(scanFile, 'FDG PT SCAN')
    planC = pc.loadNiiScan(fmisoFile, 'TBR PT SCAN', '', planC)
    planC = pc.loadNiiStructure(segFile, 0, planC)

    # Resample TBR scan to the original scan resolution
    from cerr.radiomics.preprocess import imgResample3D
    fdgScanNum = 0
    tbrScanNum = 1
    xFDG, yFDG, zFDG = planC.scan[fdgScanNum].getScanXYZVals()
    xTBR, yTBR, zTBR = planC.scan[tbrScanNum].getScanXYZVals()
    tbrArray = planC.scan[tbrScanNum].getScanArray()
    resampMethod = 'sitkLinear'
    extrapVal = 0
    resampTBR = imgResample3D(tbrArray,
                                     xTBR, yTBR, zTBR,
                                     xFDG, yFDG, zFDG,
                                     resampMethod, extrapVal)
    planC = pc.importScanArray(resampTBR, xFDG, yFDG, zFDG, 'TBR SCAN', fdgScanNum, planC)



    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0,1], [0], [], {}, '2d')


    # ======= registration example

    # base scan
    baseScanDir = r"\\vpensmph\deasylab2\Pierre\FDG_CT_for_Hypoxia_Nancy\35394153\9455504\D2013_07_22\CT\S0002_CT_for_Dyn"
    # CBCT
    baseScanDir = r"\\vpensmph\deasylab2\Data\CT_CBCT_tumor_eso_model\DICOM\38024458\20181022"
    baseScanDir = r'\\vpensmph\deasylab2\Data\HN_OPC_CT_PET\dicom\planningCT_TopModule\00042740\PHARYNX_CD'
    planC = pc.loadDcmDir(baseScanDir)

    # moving scan
    movScanDir = r"\\vpensmph\deasylab2\Pierre\FDG_CT_for_Hypoxia_Nancy\35394153\9438173\D2013_07_03\CT\S0002_CT_H&N"
    #CT
    movScanDir = r"\\vpensmph\deasylab2\Data\CT_CBCT_tumor_eso_model\DICOM\38024458\plan\CT_LUNG_101618"
    movScanDir = r'\\vpensmph\deasylab1\Data\HN_Hypoxia_Mid_Tx\DICOM_mid_FMISO\00042740\1.2.840.113619.2.55.3.2743897697.38.1390450946.896\CT\2014-1-23_CT.FMISO.180MIN'
    planMovC = pc.loadDcmDir(movScanDir)

    # Register images
    transformSaveDir = r'\\vpensmph\deasylab1\Aditya\pyCERR_transforms'
    baseScanIndex = 0
    movScanIndex = 0
    baseScan3M = planC.scan[baseScanIndex].getScanArray()
    movScan3M = planMovC.scan[movScanIndex].getScanArray()
    baseOutline3M = mask.getPatientOutline(baseScan3M, -400)
    movOutline3M = mask.getPatientOutline(movScan3M, -400)
    #baseOutline3M[baseScan3M>1500] = False
    #movOutline3M[movScan3M>1500] = False
    # planC = register.registerScans(planC, baseScanIndex, planMovC, movScanIndex, transformSaveDir,
    #                                deforAlgorithm='affine', registrationTool='plastimatch')

    planC = register.registerScans(planC, baseScanIndex, planMovC, movScanIndex, transformSaveDir,
                                   deforAlgorithm='bsplines', registrationTool='plastimatch',
                                    baseMask3M=baseOutline3M, movMask3M=movOutline3M, inputCmdFile=None)


    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0,2], [], [], {}, '2d')


    #dcmDir = r'L:\Data\Puneet_et_al_R01_Lung_tumor_radiomics\analysis\query_retrieve_PACS\prelim_data_export\baselineDcm\00386988'
    dcmDir = r'\\vpensmph\deasylab1\Harini\data\Lung-Radiogenomics\LUNG-Radiogenomics-Segmented\R01-001_CONTRAST\09-06-1990-CT CHEST ABD PELVIS WITH CON-98785'
    dcmDir = r'S:\Amita_Dave_Collaboration_2019\23-191\23-191_Both_pre_NAC_MRI_redo\RIA_23-191_000_pre_130\577205'
    planC = pc.loadDcmDir(dcmDir)

    dcmDir = r'L:\Maria\LA-NSCLC_Durva_N230\NewN_50_TotN230\PET\38174084\2021-06__Studies\SNYDER^EDWARD W_38174084_CT_2021-06-16_124202_FOREIGN.PET.SCAN.-.CD_PET.CT.AXIALS_n854__00000'
    planC = pc.loadDcmDir(dcmDir)


    dcmDir = r'H:\Public\UM\3DPrints_CERR\DICOMS\cerr_files_dcm_export\S0002_AsirV1\Scan_2'
    planC = pc.loadDcmDir(dcmDir)


    dcmDir = r'S:\Amita_Dave_Collaboration_2019\23-191\23-191_Pre_NAC_MRI\RIA_23-191_000_pre_008\577113\T1w\7\DICOM'
    planC = pc.loadDcmDir(dcmDir)
    scanNum = 0
    scanSaveFileName = r'S:\Amita_Dave_Collaboration_2019\23-191\23-191_Pre_NAC_MRI\RIA_23-191_000_pre_008\577113\T1w\nii_test.nii'
    planC.scan[scanNum].saveNii(scanSaveFileName)





    dcmDir = r'S:\\Amita_Dave_Collaboration_2019\\23-191\\23-191_Pre_NAC_MRI\\RIA_23-191_000_pre_046'
    planC = pc.loadDcmDir(dcmDir)


    # Register and warp dose
    baseScanDir = r'L:\Data\RTOG0617\DICOM\NSCLC-Cetuximab_flat_dirs\0617-381906_09-09-2000-87022'
    planC = pc.loadDcmDir(baseScanDir)

    # moving scan
    movScanDir = r'L:\Data\RTOG0617\DICOM\NSCLC-Cetuximab_flat_dirs\0617-820440_09-09-2000-44688'
    planMovC = pc.loadDcmDir(movScanDir)

    # Register images
    transformSaveDir = r'\\vpensmph\deasylab1\Aditya\pyCERR_transforms'
    baseScanIndex = 0
    movScanIndex = 0
    baseScan3M = planC.scan[baseScanIndex].getScanArray()
    movScan3M = planMovC.scan[movScanIndex].getScanArray()
    baseOutline3M = mask.getPatientOutline(baseScan3M, -400)
    movOutline3M = mask.getPatientOutline(movScan3M, -400)
    #baseOutline3M[baseScan3M>1500] = False
    #movOutline3M[movScan3M>1500] = False
    # planC = register.registerScans(planC, baseScanIndex, planMovC, movScanIndex, transformSaveDir,
    #                                deforAlgorithm='affine', registrationTool='plastimatch')

    planC = register.registerScans(planC, baseScanIndex, planMovC, movScanIndex, transformSaveDir,
                                   deforAlgorithm='bsplines', registrationTool='plastimatch',
                                    baseMask3M=baseOutline3M, movMask3M=movOutline3M, inputCmdFile=None)



    deformS = planC.deform[-1]

    # Warp scan
    #planC = register.warpScan(planC, baseScanIndex, planMovC, movScanIndex, deformS)

    # Warp structures
    #movStrNumV = [5, 6]
    #planC = register.warpStructures(planC, baseScanIndex, planMovC, movStrNumV, deformS)

    # Warp dose
    movDoseNumV = 0
    planC = register.warpDose(planC, baseScanIndex, planMovC, movDoseNumV, deformS)




    dcmDir = r'L:\Data\HNC-IMRT-70-33\manifest-1714488710321\HNC-IMRT-70-33\HNC_011\08-06-2003-NA-IMRT HN-34473'
    planC = pc.loadDcmDir(dcmDir)
    niiFileName = r'L:\Data\HNC-IMRT-70-33\manifest-1714488710321\sample_data\test\dose.nii'
    planC.dose[0].saveNii(niiFileName)
    planC = pc.loadNiiDose(niiFileName, 0, planC)
    planC.dose[1].saveNii(niiFileName)
    planC = pc.loadNiiDose(niiFileName, 0, planC)


    #dcmDir = r'L:\Data\ROBIN\Cornell_data\15515009\15515009'
    #planC = pc.loadDcmDir(dcmDir)

    dcmDir = r'S:\Amita_Dave_Collaboration_2019\23-191\23-191_Both_pre_NAC_MRI\RIA_23-191_000_pre_021'
    planC = pc.loadDcmDir(dcmDir)


    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [0], [], {}, '2d')


    import numpy as np
    dcmDir = r'\\vpensmph\deasylab1\Data\HNC-IMRT-70-33\manifest-1714488710321\HNC-IMRT-70-33\HNC_040\11-10-2009-NA-IMRT HN-40505\3.000000-IMRT HN-49698'
    planC = pc.loadDcmDir(dcmDir)
    scanSiz = planC.scan[0].getScanSize()
    mask3M = np.zeros(scanSiz, dtype=bool)
    mask3M[10:-10,10:-10,99] = 1
    planC = pc.importStructureMask(mask3M, 0, 'test', planC)

    from cerr.dcm_export import rtstruct_iod
    rtstructFile = r'\\vpensmph\deasylab1\Data\HNC-IMRT-70-33\manifest-1714488710321\HNC-IMRT-70-33\HNC_040\pyCERR_rtstruct_box.dcm'
    rtstruct_iod.create(structNumV = [1], filePath = rtstructFile, planC = planC, seriesOpts = {'SeriesDescription':'re-exported from pyCERR'})




    imgFile = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\organizedZscoreSeg\T2w\PURE-01_001-067_2018_08_17_N4_scan.nii.gz'
    segFile = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\organizedZscoreSeg\T2w\PURE-01_001-067_2018_08_17_seg.nii.gz'
    ibsi1Settings = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\pycerr_features\radiomics_settings\ibsi1_2d_linear.json'
    imgFile = r'S:\Amita_Dave_Collaboration_2019\23-191\23-191_Post_NAC_zScore\T1w-post\RIA_23-191_post_182_N4_scan.nii.gz'
    segFile = r'S:\Amita_Dave_Collaboration_2019\23-191\23-191_Post_NAC_zScore\T1w-post\RIA_23-191_post_182_seg.nii.gz'
    ibsi1Settings = r'S:\Amita_Dave_Collaboration_2019\23-191\analysis\radiomics_settings\ibsi1_2d_linear.json'
    planC = pc.loadNiiScan(imgFile, 'MR SCAN')
    planC = pc.loadNiiStructure(segFile,0,planC,{})

    from cerr.radiomics import ibsi1
    # Compute radiomics features
    scanNum = 0
    structNum = 0
    featDictIbsi1, _ = ibsi1.computeScalarFeatures(scanNum, structNum, ibsi1Settings, planC)

    #
    # imgFile = r'S:\\Amita Dave HN\\PURE-01\\ROI_BORA_ALL DATA_AMRESHA 2023\\organizedScanSeg\\T2w\\PURE-01_001-025_2017_11_13_N4_scan.nii.gz'
    # segFile = r'S:\\Amita Dave HN\\PURE-01\\ROI_BORA_ALL DATA_AMRESHA 2023\\organizedScanSeg\\T2w\\PURE-01_001-025_2017_11_13_seg.nii.gz'
    # planC = pc.loadNiiScan(imgFile, 'MR SCAN')
    # planC = pc.loadNiiStructure(segFile,0,planC,{})


    from cerr.dataclasses import structure as cerrStr

    dcmDir = r'\\vpensmph\deasylab1\Data\HNC-IMRT-70-33\manifest-1714488710321\HNC-IMRT-70-33\HNC_040\11-10-2009-NA-IMRT HN-40505\3.000000-IMRT HN-49698'
    planC = pc.loadDcmDir(dcmDir)

    import cerr.dataclasses.structure as cerrStruct
    structNum = 0
    resFactor = 1.25
    smoothFactor = 0.15
    planC = cerrStruct.getBsplineSmoothing(structNum, resFactor, smoothFactor, planC, \
                  replaceFlag=False, procSructName=None)


    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [0], [], {}, '2d')

    def getPolygons(zVal, structNum):
        polyList = []
        _,_,zValsV = planC.scan[0].getScanXYZVals()
        slcNum = np.argmin((zValsV - zVal) **2)
        if hasattr(planC.structure[structNum].contour[slcNum], 'segments'):
            segs = planC.structure[structNum].contour[slcNum].segments
            for seg in segs:
                polygons = seg.points.copy()
                if len(polygons[:,0]) < 3:
                    continue
                polygons = polygons[:,[1,0,2]]
                polygons[:,0] = -polygons[:,0]
                polyList.append(polygons)
        return polyList

    #polygons = cerrStruct.getContourPolygons(1, planC, True)

    _,_,zValsV = planC.scan[0].getScanXYZVals()
    zVal = zValsV[65]
    layerOrig = viewer.add_shapes(
        getPolygons(zVal,0),
        shape_type='polygon',
        edge_width=0.1,
        edge_color='coral',
        face_color=[0,0,0,0],
        name='original'
    )
    layerSmooth = viewer.add_shapes(
        getPolygons(zVal,1),
        shape_type='polygon',
        edge_width=0.1,
        edge_color='blue',
        face_color=[0,0,0,0],
        name='smooth'
    )

    # def showContours(slcNum, structNum1, structNum2):
    #     zVal = zValsV[slcNum]
    #     layerOrig.data = getPolygons(zVal, structNum1)
    #     layerOrig.edge_width = 0.1
    #     layerSmooth.data = getPolygons(zVal, structNum2)
    #     layerSmooth.edge_width = 0.1
    #     viewer.dims.set_point(2, zVal)
    # showContours(70, 0, 1)

    slcNum = 65
    zVal = zValsV[slcNum]
    layerOrig.data = getPolygons(zVal, 0)
    layerSmooth.data = getPolygons(zVal, 1)
    layerOrig.edge_width = 0.1
    layerSmooth.edge_width = 0.1
    viewer.dims.set_point(2, zVal)


    # Add random noise to structure coordinates
    for slc in range(0, len(planC.structure[0].contour)):
        if hasattr(planC.structure[0].contour[slc],'segments'):
            for segNum in range(0,len(planC.structure[0].contour[slc].segments)):
                ptsShap = planC.structure[0].contour[slc].segments[segNum].points.shape
                planC.structure[0].contour[slc].segments[segNum].points[:,[0,1]] += np.random.random((ptsShap[0],2))*1e-5



    from cerr.dcm_export import rtstruct_iod
    structFileName = r'\\vpensmph\deasylab1\Data\HNC-IMRT-70-33\manifest-1714488710321\HNC-IMRT-70-33\HNC_040\rtstruct_pycerr.dcm'
    seriesDescription = "AI Generated"
    exportOpts = {'seriesDescription': seriesDescription}
    rtstruct_iod.create([0],structFileName,planC,exportOpts)


    dcmDir = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\All file\PURE-01_Batch-1_03312023/PURE-01_001-011/D2017_08_01/T1w/S0301_9692'
    planC = pc.loadDcmDir(dcmDir)




    import cerr.dataclasses.structure as cerrStruct
    imgFile = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\organizedScanSeg\T1w\PURE-01_001-001_2017_04_19_N4_scan.nii.gz'
    segFile = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\organizedScanSeg\T1w\PURE-01_001-001_2017_04_19_seg.nii.gz'
    planC = pc.loadNiiScan(imgFile, 'MR SCAN')
    planC = pc.loadNiiStructure(segFile,0,planC)
    planC = cerrStruct.getBsplineSmoothing(0, 2, planC, \
                  replaceFlag=False, procSructName=None)
    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [0], [], {}, '2d')


    from cerr.dataclasses import structure as cerrStr
    dcmDir = r'\\vpensmph\deasylab1\Data\HNC-IMRT-70-33\manifest-1714488710321\HNC-IMRT-70-33\HNC_040'
    dcmDir = r'\\vpensmph\deasylab1\Data\HNC-IMRT-70-33\manifest-1714488710321\HNC-IMRT-70-33\HNC_040\11-10-2009-NA-IMRT HN-40505'
    planC = pc.loadDcmDir(dcmDir)

    from cerr.dcm_export import rtstruct_iod
    rtstructFile = r'\\vpensmph\deasylab1\Data\HNC-IMRT-70-33\manifest-1714488710321\HNC-IMRT-70-33\HNC_040\pyCERR_rtstruct.dcm'
    rtstruct_iod.create(structNumV = [1], filePath = rtstructFile, planC = planC, seriesOpts = {'SeriesDescription':'re-exported from pyCERR'})


    numComponents = 1
    strNumV = range(0,16)
    for strNum in strNumV:
        procMask3M = cerrStr.getLargestConnComps(strNum, numComponents, planC, saveFlag=True, replaceFlag=True)

    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [0, 1], [], {}, '2d')


    from cerr.dataclasses.structure import getGaussianBlurredMask
    sigmaVoxel = 4
    _, planC = getGaussianBlurredMask(16, sigmaVoxel, planC,
                     saveFlag=True, replaceFlag=False)
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], [16, 40], [], {}, '2d')



    from cerr.contour import rasterseg as rs
    import csv
    csvFileName = r'L:\Data\Puneet_et_al_R01_Lung_tumor_radiomics\tracking_sample_data.csv'

    # Tracking
    dcmDir = r'L:\Data\Puneet_et_al_R01_Lung_tumor_radiomics\tracking_sample_data\35412422'
    dcmDir = r'L:\Data\Puneet_et_al_R01_Lung_tumor_radiomics\anan_dicom\D2017-02-24_E0320\301\DICOM'
    dcmDir = r'L:\Data\HNC-IMRT-70-33\manifest-1714488710321\HNC-IMRT-70-33\HNC_001'
    ptDir = r'\\vpensmph\deasylab1\Data\Puneet_et_al_R01_Lung_tumor_radiomics\tracking_sample_data\35249060'
    dcmDir = r'\\vpensmph\deasylab1\Data\Puneet_et_al_R01_Lung_tumor_radiomics\tracking_sample_data'
    dcmDir = r'\\vpensmph\deasylab1\Data\Puneet_et_al_R01_Lung_tumor_radiomics\anan_dicom\MSK-IRB16568-19208348-20161118\MSK-IRB16568-19208348-20161118_CT_UNKNOWN'

    dcmDir = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\All file\PURE-01_Batch-7_08032023\PURE-01_001-137\D2019_12_03\T1w\S0301_4894'
    structFile = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\All file\PURE-01_Batch-7_08032023\PURE-01_001-137\D2019_12_03\T1w\301_T1_pre_ROI_YA.nii.gz'
    planC = pc.loadNiiStructure(structFile, 0, planC)

    scanFile = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\organizedScanSeg\T1w-Post\PURE-01_001-010_2017_07_31_N4_scan.nii.gz'
    segFile = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\organizedScanSeg\T1w-Post\PURE-01_001-010_2017_07_31_seg.nii.gz'
    planC = pc.loadNiiScan(scanFile, 'T1w')
    planC = pc.loadNiiStructure(segFile, 0, planC)



    from cerr.dataclasses import structure as cerrStruct
    planC = pc.loadDcmDir(dcmDir)
    sigmaVoxel = 2
    _, planC = cerrStruct.getGaussianBlurredMask(0, sigmaVoxel, planC,
                             saveFlag=True, replaceFlag=False)

    mask3M = rs.getStrMask(0, planC)
    x,y,z = planC.scan[0].getScanXYZVals()
    planC = pc.importStructureMask(mask3M, 0, 'smooth', planC)

    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0], range(len(planC.structure)), [], {}, '2d')


    writeHeader = True
    for d in os.scandir(dcmDir):

        ptDir = d.path
        _, id = os.path.split(ptDir)

        planC = pc.loadDcmDir(ptDir)
        assocScanNums = [s.getStructureAssociatedScan(planC) for s in planC.structure]

        featList = []
        for structNum in range(len(planC.structure)):
            mask3M = rs.getStrMask(structNum, planC)
            scanNum = planC.structure[structNum].getStructureAssociatedScan(planC)
            scanDate = planC.scan[scanNum].scanInfo[0].seriesDate
            vol = np.prod(planC.scan[scanNum].getScanSpacing()) * np.sum(mask3M)
            featList.append({'MRN':id, 'Scan Date':scanDate, 'Volume (cc)':vol})

        with open(csvFileName, 'a', newline='') as csvfile:
            flatFeatDict = featList[0]
            fieldnames = flatFeatDict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if writeHeader:
                writer.writeheader()
                writeHeader = False
            for flatFeatDict in featList:
                writer.writerow(flatFeatDict)



    from cerr.dcm_export import rtstruct_iod
    rtstructFile = r'\\vpensmph\deasylab1\Data\Puneet_et_al_R01_Lung_tumor_radiomics\anan_dicom\MSK-IRB16568-19208348-20161118\pyCERR_rtstruct.dcm'
    rtstruct_iod.create(structNumV = [0], filePath = rtstructFile, planC = planC, seriesOpts = {'SeriesDescription':'re-exported from pyCERR'})


    mask3M = rs.getStrMask(0, planC)
    connMask3M = mask.largestConnComps(mask3M,5)

    #Extract connected components
    from scipy.ndimage import label
    labeledmask3M, numFeatures = label(mask3M, structure=np.ones((3, 3, 3)))

    for comp in range(numFeatures):
        planC = pc.importStructureMask(labeledmask3M,0,'Comp_'+str(comp),planC)

    from cerr.radiomics import shape
    shape.calcShapeFeatures()


    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
            vwr.showNapari(planC, [0], range(len(planC.structure)), [], {}, '2d')


    from cerr.radiomics import ibsi1
    d = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HNTH-36'
    ibsi1Settings = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\pycerr_features\radiomics_settings/ibsi1.json'

    imgFile = os.path.join(d, 'zScoreTumor.nii.gz')
    segFile = os.path.join(d, 'Seg.nii')

    planC = pc.loadNiiScan(imgFile, 'MR SCAN')
    planC = pc.loadNiiStructure(segFile,0,planC,{1: 'GTV'})

    # Compute radiomics features
    scanNum = 0
    structNum = 0
    featDictIbsi1, _ = ibsi1.computeScalarFeatures(scanNum, structNum, ibsi1Settings, planC)


    t2ImgFile = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HN164D\zScoreTumor.nii.gz'
    segFile = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HN164D\Seg.nii'

    planC = pc.loadNiiScan(t2ImgFile, 'MR SCAN', 'LPS')
    planC = pc.loadNiiStructure(segFile, 0, planC, {1: "tumor"})

    from cerr.radiomics import texture_utils

    settingsFile = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\pycerr_features\gabor_filter.json'
    scanNum = 0
    strNum = 0
    planC = texture_utils.generateTextureMapFromPlanC(planC, scanNum, strNum, settingsFile)

    #Visualize in Napari
    filtIdx = len(planC.scan)-1
    dispScanNum = [scanNum,filtIdx]
    dispStrNum = strNum
    viewer, scan_layers, struct_layer, dose_layers, dvf_layer = \
        vwr.showNapari(planC, scan_nums=dispScanNum, struct_nums=dispStrNum, dose_nums=[], vectors_dict={}, displayMode='2d')

    # h5File = r'L:\Sharif\data\22\plan_c\planC.h5'
    # planC2 = pc.loadFromH5(h5File)


    # from cerr import plan_container as pc
    # # from cerr.dcm_export import rtstruct_iod
    # # dcmDataDir = r"\\vpensmph\deasylab2\Data\CT_CBCT_tumor_eso_model\DICOM\38024458\plan\CT_LUNG_101618"
    # #
    # #
    # dcmDataDir = r"L:\Data\RTOG0617\DICOM\NSCLC-Cetuximab_flat_dirs\0617-381906_09-09-2000-87022"
    # planC = pc.loadDcmDir(dcmDataDir)
    # viewer, scanLayer, structLayer, doseLayer, dvfLayer = vwr.showNapari(planC, scanNumV, strNumV, doseNumV)
    #
    #
    # # Write planC to h5 file
    # h5File = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\test.hdf5'
    # strNumToExportV = [5,6]
    # scanToExportV = [0]
    # doseNumToExportV = []
    # pc.saveToH5(planC, h5File, scanToExportV, strNumToExportV, doseNumToExportV)
    #
    # # Read h5 to planC
    # h5File = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\test.hdf5'
    # h5File = r'L:\Sharif\data\22\plan_c\planC.h5'
    # planC2 = pc.loadFromH5(h5File)
    # scanNum = [0]
    # doseNum = []
    # strNum = [0,1]
    # displayMode = '2d' # 'path' or 'surface'
    # vectDict = {}
    # viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
    #         vwr.showNapari(scanNum, strNum, doseNum, vectDict, planC2, displayMode)




    #
    #
    # # Read h5 file
    # import h5py
    # f = h5py.File(h5File, "r")
    #





    #
    #
    # import cerr.dataclasses.structure as cerrStr
    # outFile = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\test.json'
    # h5File = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\test.hdf5'
    # structNumList = [5, 6] # List of structures to export to json
    # cerrStr.saveJson(structNumList, outFile, planC)
    #
    # del planC.structure[5]
    # del planC.structure[5]
    # planC = cerrStr.importJson(outFile, planC)
    #
    #
    # import json
    # with open(outFile, 'r', encoding='utf-8') as f:
    #     jsonStrList = json.load(f)

    # Create dictionary of series metadata tags (seriesDescription, seriesDate, seriesTime)
    #     seriesDate = 'YYYYMMDD' e.g. '20240315'
    #     seriesTime = 'HHMMSS.MS' e.g. '171354.838973'
    #     seriesDescription = 'string describing RTSTRUCT series'
    #     seriesOpts = {'seriesDescription': seriesDescription,
    #                   'seriesDate': seriesDate,
    #                   'seriesTime': seriesTime}

    # seriesDescription = "pyCER test"
    # seriesDate = '20240315'
    # seriesTime = '171354.838973'
    # seriesOpts = {'seriesDescription': seriesDescription,
    #               'seriesDate': seriesDate,
    #               'seriesTime': seriesTime}
    # strToExportV = [5,6]
    # # Assign ROI generation algorithm and description
    # planC.structure[5].roiGenerationAlgorithm = 'AUTOMATIC' # 'AUTOMATIC', 'MANUAL' or 'SEMIAUTOMATIC'
    # planC.structure[5].roiGenerationDescription = 'SMIT+ version xxx' # string describing the algorithm
    # planC.structure[5].ROIInterpretedType = "ORGAN"
    # structFilePath = r'\\vpensmph\deasylab1\Aditya\forSharif\test.dcm'
    # rtstruct_iod.create(strToExportV,structFilePath,planC,seriesOpts)


    ## ======= registration example

    # base scan
    baseScanDir = r"\\pisidsmph\deasylab2\Pierre\FDG_CT_for_Hypoxia_Nancy\35394153\9455504\D2013_07_22\CT\S0002_CT_for_Dyn"
    # CBCT
    baseScanDir = r"\\pisidsmph\deasylab2\Data\CT_CBCT_tumor_eso_model\DICOM\38024458\20181022"
    baseScanDir = r'\\pisidsmph\deasylab2\Data\HN_OPC_CT_PET\dicom\planningCT_TopModule\00042740\PHARYNX_CD'
    planC = pc.loadDcmDir(baseScanDir)

    # moving scan
    movScanDir = r"\\pisidsmph\deasylab2\Pierre\FDG_CT_for_Hypoxia_Nancy\35394153\9438173\D2013_07_03\CT\S0002_CT_H&N"
    #CT
    movScanDir = r"\\pisidsmph\deasylab2\Data\CT_CBCT_tumor_eso_model\DICOM\38024458\plan\CT_LUNG_101618"
    movScanDir = r'\\pisidsmph\deasylab1\Data\HN_Hypoxia_Mid_Tx\DICOM_mid_FMISO\00042740\1.2.840.113619.2.55.3.2743897697.38.1390450946.896\CT\2014-1-23_CT.FMISO.180MIN'
    planMovC = pc.loadDcmDir(movScanDir)

    # Register images
    transformSaveDir = r'\\pisidsmph\deasylab1\Aditya\pyCERR_transforms'
    baseScanIndex = 0
    movScanIndex = 0
    baseScan3M = planC.scan[baseScanIndex].getScanArray()
    movScan3M = planMovC.scan[movScanIndex].getScanArray()
    baseOutline3M = mask.getPatientOutline(baseScan3M, -400)
    movOutline3M = mask.getPatientOutline(movScan3M, -400)
    #baseOutline3M[baseScan3M>1500] = False
    #movOutline3M[movScan3M>1500] = False
    # planC = register.registerScans(planC, baseScanIndex, planMovC, movScanIndex, transformSaveDir,
    #                                deforAlgorithm='affine', registrationTool='plastimatch')

    planC = register.registerScans(planC, baseScanIndex, planMovC, movScanIndex, transformSaveDir,
                                   deforAlgorithm='bsplines', registrationTool='plastimatch',
                                    baseMask3M=baseOutline3M, movMask3M=movOutline3M, inputCmdFile=None)


    deformS = planC.deform[-1]

    # Warp scan
    #planC = register.warpScan(planC, baseScanIndex, planMovC, movScanIndex, deformS)

    # Warp structures
    #movStrNumV = [5, 6]
    #planC = register.warp_structures(planC, baseScanIndex, planMovC, movStrNumV, deformS)

    warpedScanIndex = len(planC.scan) - 1

    # Get vector field
    planC = register.calcVectorField(deformS, planC, baseScanIndex, transformSaveDir)

    # Calculate Jacobian
    deformS = planC.deform[-1]
    planC = register.calcJacobian(deformS, planC)

    deformS = planC.deform[-1]
    dvfFile = deformS.deformOutFilePath
    planC = pc.loadNiiVf(dvfFile, baseScanIndex, planC)

    # Visualize CBCT, deformed CT and segmentation
    import numpy as np
    scanNum = [baseScanIndex, warpedScanIndex]
    doseNum = []
    strNum = [14, 15]
    deformNum = len(planC.deform)-1
    deformS = planC.deform[deformNum]
    deformStructNum = 14
    displayMode = '2d' # 'path' or 'surface'
    resolution = [0.5,0.5,0]
    surfFlag = False
    vectors = register.getDvfVectors(deformS, planC, baseScanIndex, resolution, deformStructNum, surfFlag)
    deformMedian = np.median(vectors, axis = 0)[1,:]
    vectorsMinusMedian = vectors.copy()
    vectorsMinusMedian[:,1,:] -= deformMedian
    lengthV = np.sum(vectorsMinusMedian **2, axis = 2)[:,1] ** 0.5 * 10
    feats = {'length (mm)': lengthV,  'dx (mm)': np.abs(vectorsMinusMedian[:,1,1]) * 10,
             'dy (mm)': np.abs(vectorsMinusMedian[:,1,0]) * 10, 'dz (mm)': np.abs(vectorsMinusMedian[:,1,2]) * 10}
    vectDict = {'vectors': vectorsMinusMedian, 'features': feats}
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
            vwr.showNapari(planC, scanNum, strNum, doseNum, vectDict, displayMode)




    # Visualize Jacobian
    jacobianIndex = len(planC.scan)-1
    scanNum = [baseScanIndex, warpedScanIndex, jacobianIndex]
    doseNum = []
    strNum = [0, 1]
    displayMode = '2d' # 'path' or 'surface'
    vectDict = {}
    viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
            vwr.showNapari(scanNum, strNum, doseNum, vectDict, planC, displayMode)

    # Visualize moving scan (planning CT)
    scanNum = [0]
    doseNum = []
    strNum = [5, 6]
    displayMode = '2d' # 'path' or 'surface'
    vectDict = {}
    viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
            vwr.showNapari(scanNum, strNum, doseNum, vectDict, planMovC, displayMode)















    deformS = planC.deform[-1]
    sampleRate = 2
    vectors = register.getDvfVectors(deformS, planC, baseScanIndex, resolution, deformStructNum)

    import numpy as np
    lengthV = np.sum(vectors **2, axis = 2)[:,1] ** 0.5
    vectors[:,1,:] = vectors[:,1,:] / lengthV[:,None] * 0.09080204
    feats = {'length': lengthV}
    vect_layr = viewer.add_vectors(vectors, edge_width=0.1, opacity=0.3,
                                   length=1, name="DVF",
                                   ndim=3, features=feats,
                                   edge_color='length',
                                   edge_colormap='husl')


    import numpy as np
    import napari
    viewer = napari.Viewer(title='vectors')
    deformS = planC.deform[-1]
    structNum = 0
    sampleRate = 2
    #vectors = register.get_dvf_vectors(deformS, structNum, planC, sampleRate)
    #lengthV = np.sum(vectors **2, axis = 2)[:,1] ** 0.5
    #vectors[:,1,:] = vectors[:,1,:] / lengthV[:,None] * 0.09080204
    vectors = np.load('vectors.npy')
    lengthV = np.load('lengths.npy')
    feats = {'length': lengthV}
    vect_layr = viewer.add_vectors(vectors, edge_width=0.1, opacity=0.3,
                                   length=1, name="DVF",
                                   ndim=3, features=feats,
                                   edge_color='length',
                                   edge_colormap='husl',
                                   metadata = {'dataclass': 'dvf',
                                           'planC': planC,
                                           'deformNum': 0}
                                   )
    viewer.dims.ndisplay = 3
    viewer.dims.order = (2, 0, 1)
    napari.run()



    # Vector points
import numpy as np
from skimage import data
import napari


blobs = np.stack(
    [
        data.binary_blobs(
            length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=f
        )
        for f in np.linspace(0.05, 0.5, 10)
    ],
    axis=0,
)
viewer = napari.view_image(blobs.astype(float))

# sample vector coord-like data
n = 200
pos = np.zeros((n, 2, 2), dtype=np.float32)
phi_space = np.linspace(0, 4 * np.pi, n)
radius_space = np.linspace(0, 20, n)

# assign x-y position
pos[:, 0, 0] = radius_space * np.cos(phi_space) + 64
pos[:, 0, 1] = radius_space * np.sin(phi_space) + 64

# assign x-y projection
pos[:, 1, 0] = 2 * radius_space * np.cos(phi_space)
pos[:, 1, 1] = 2 * radius_space * np.sin(phi_space)

planes = np.round(np.linspace(0, 128, n)).astype(int)
planes = np.concatenate(
    (planes.reshape((n, 1, 1)), np.zeros((n, 1, 1))), axis=1
)
vectors = np.concatenate((planes, pos), axis=2)

# add the sliced vectors
layer = viewer.add_vectors(
    vectors, edge_width=0.4, name='sliced vectors', edge_color='blue'
)

viewer.dims.ndisplay = 3

napari.run()
