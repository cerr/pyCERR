from cerr.dataclasses import structure as cerrStruct
import os
from cerr import plan_container as pc
import SimpleITK as sitk
import numpy as np
import cerr.dataclasses.scan as cerrScn
import cerr.viewer as vwr
from cerr.contour import rasterseg as rs
import pandas as pd

if __name__ == "__main__":

    niiDir = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2'
    outcomesFile = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Thyroid_w_wo_aggress_06062024.csv'

    zScoreList = []
    dataList = []
    zScoreAggresiveList = []
    zScoreNonAggresiveList = []
    outcomesList = []

    df = pd.read_csv(outcomesFile)

    noSegList = []
    #dList = ['S:\\Amita_Dave_Collaboration_2019\\Thyroid_ROI_AI_DL\\Organized T2\\Thyroid_MR_T2\\HN169D']
    for d in os.scandir(niiDir):
        _, id = os.path.split(d)
        t2N4Scan = os.path.join(d,'Img_N4.nii')
        segFile = os.path.join(d,'Seg.nii')
        smoreZScore = os.path.join(d,'Img_N4\Img_N4_smore4_zscore.nii')

        print("Patient ID: " + id)
        if not os.path.exists(segFile):
            noSegList.append(id)
            continue

        planC = pc.loadNiiScan(t2N4Scan, 'MR SCAN')
        planC = pc.loadNiiStructure(segFile, 0, planC, {1: 'tumor'})
        planC = pc.loadNiiScan(smoreZScore, 'z-score', '', planC)
        t2StructNum = 0
        zSocreScanNum = 1
        planC = cerrStruct.copyToScan(t2StructNum, zSocreScanNum, planC)

        zSocreStrNum = 1
        mask3M = rs.getStrMask(zSocreStrNum, planC)
        zScore3M = planC.scan[zSocreScanNum].getScanArray()

        zScoreV = zScore3M[mask3M]
        #zScoreV -= zScoreV.mean()

        zScoreList.append(zScoreV)

        ind = df.index[df['PatientFolder'] == id].tolist()
        if len(ind) == 1:
            if df.iloc[ind]['Label'].values[0] == "aggressive":
                outcomesList.append('aggressive')
            else:
                outcomesList.append('non-aggressive')


    zScoreDict = {
        'Aggressive': zScoreAggresiveList,
        'Non-aggressive': zScoreNonAggresiveList,
    }

    zScoreDict = {
        'Aggressive': zScoreNonAggresiveList
    }

    from matplotlib import pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(1,2)
    for i in range(len(outcomesList)):
        if outcomesList[i] == 'aggressive':
            color = "red"
            axis = ax[0]
        else:
            color = "blue"
            axis = ax[1]
        if len(zScoreList[i]) == 0:
            continue
        sns.kdeplot(data=zScoreList[i],ax=axis,alpha=0.05, linewidth=0.2,fill=True,
                    color=color,label=outcomesList[i],common_norm=False)
        #sns.histplot(data=zScoreList[i],ax=axis,alpha=0.5, linewidth=2,
        #            color=color,label=outcomesList[i])
    xLim = [-1,10]
    yLim = [0,4]
    ax[0].set_ylabel('Kernel Density Estimate', fontsize=14)
    ax[0].set_xlabel('z-Score (Aggressive)', fontsize=14)
    ax[1].set_ylabel('', fontsize=14)
    ax[1].set_xlabel('z-Score (Non-Aggressive)', fontsize=14)
    for i in range(len(ax)):
        ax[i].grid(True)
        ax[i].set_xlim(xLim)
        ax[i].set_ylim(yLim)

    # ax.legend(['Aggressive', 'Non-aggressive'], fontsize=14)






    dataList = []
    data = {}
    # 16
    data['t2N4Scan'] = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HNTH-16\Img_N4.nii'
    data['segFile'] = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HNTH-16\Seg.nii'
    data['smoreZScore'] = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HNTH-16\Img_N4\Img_N4_smore4_zscore.nii'
    dataList.append(data)

    # 76
    data = {}
    data['t2N4Scan'] = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HNTH-76\Img_N4.nii'
    data['segFile'] = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HNTH-76\Seg.nii'
    data['smoreZScore'] = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HNTH-76\Img_N4\Img_N4_smore4_zscore.nii'
    dataList.append(data)

    patID = []
    patID.append('HNTH-16')
    patID.append('HNTH-76')

    zScoreList = []
    t2List = []
    for data in dataList:
        t2N4Scan = data['t2N4Scan']
        segFile = data['segFile']
        smoreZScore = data['smoreZScore']

        planC = pc.loadNiiScan(t2N4Scan, 'MR SCAN', 'LPS')
        planC = pc.loadNiiStructure(segFile, 0, planC, {1: 'tumor'})
        planC = pc.loadNiiScan(smoreZScore, 'z-score', 'LPS', planC)
        t2StructNum = 0
        t2ScanNum = 0
        zSocreScanNum = 1
        planC = cerrStruct.copyToScan(t2StructNum, zSocreScanNum, planC)
        zSocreStrNum = 1
        mask3M = rs.getStrMask(zSocreStrNum, planC)
        zScore3M = planC.scan[zSocreScanNum].getScanArray()
        zScoreV = zScore3M[mask3M]
        zScoreList.append(zScoreV)

        mask3M = rs.getStrMask(t2StructNum, planC)
        t23M = planC.scan[t2ScanNum].getScanArray()
        t2V = t23M[mask3M]
        t2List.append(t2V)

    import pandas as pd
    s = pd.Series(zScoreList[1])
    ax = s.plot.kde()

    #df = pd.DataFrame()
    #ax = df.plot.kde()

    zScoreDict = {
        patID[0]: zScoreList[0],
        patID[1]: zScoreList[1],
    }

    zScoreDict = {
        patID[0]: t2List[0],
        patID[1]: t2List[1]
    }

    import seaborn as sns
    ax = sns.kdeplot(data=zScoreDict,bw_adjust=.3,common_norm=False)
    ax.grid(True)
    ax.set_ylabel('Kernel Density Estimate', fontsize=14)
    ax.set_xlabel('z-Score', fontsize=14)
    ax.legend(patID, fontsize=14)



    viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
            vwr.showNapari([0,1], [0,1], [], {}, planC, '2d')
