from cerr import plan_container as pc
from cerr import viewer as vwr
from cerr.radiomics import texture_utils


if __name__ == "__main__":

    t2ImgFile = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HN164D\Img_N4.nii'
    segFile = r'S:\Amita_Dave_Collaboration_2019\Thyroid_ROI_AI_DL\Organized T2\Thyroid_MR_T2\HN164D\Seg.nii'

    planC = pc.loadNiiScan(t2ImgFile, 'MR SCAN', 'LPS')
    planC = pc.loadNiiStructure(segFile, 0, planC, {1: "tumor"})


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
