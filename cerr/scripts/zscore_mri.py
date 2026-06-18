from cerr import plan_container as pc
from cerr import viewer as vwr
import os
from cerr.dataclasses import structure as cerrStr
import cerr.contour.rasterseg as rs
import numpy as np
from scipy import stats
from skimage.io import imsave, imshow

if __name__ == "__main__":

    # texture example
    from cerr import plan_container as pc
    from cerr import viewer as vwr
    from cerr.radiomics import texture_utils
    from cerr.radiomics import ibsi1


    # Import DICOM to planC
    dcmDir = r'\\vpensmph\deasylab1\Data\TCIA_HN_Jung_Eva\dicom\tcga-bb-7870'
    dcmDir = r'\\vpensmph\deasylab1\Data\TCIA_HN_Jung_Eva\dicom\tcga-cn-a49a'
    dcmDir = r'L:\Sharif\maite\pet_examples\20190829-CALIDONNA_STEVEN_M-38071705-43690'
    planC = pc.loadDcmDir(dcmDir)

    #ibsi1SettingsCT = r'\\vpensmph\deasylab1\Data\Puneet_et_al_R01_Lung_tumor_radiomics\analysis\radiomics_settings\ibsi1CT.json'
    #featDictIbsi1CT, _ = ibsi1.computeScalarFeatures(0, 0, ibsi1SettingsCT, planC)

    # Extract texture
    settingsFile = r'\\vpensmph\deasylab1\Data\TCIA_HN_Jung_Eva\analysis_code\settings\filter_example.json'
    planC = texture_utils.generateTextureMapFromPlanC(planC, 0, 0, settingsFile)

    # Visualize
    scanNum = [0,1]
    doseNum = []
    strNum = [0]
    displayMode = '2d'
    vectDict = {}
    viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
            vwr.showNapari(scanNum, strNum, doseNum, vectDict, planC, displayMode)



    # Load PET CT scans for Puneet et al

    dcmDir = r'\\vpensmph\deasylab1\Data\Puneet_et_al_R01_Lung_tumor_radiomics\DICOM'
    niiDir = r'\\vpensmph\deasylab1\Data\Puneet_et_al_R01_Lung_tumor_radiomics\nii'
    imageSaveDir = r'\\vpensmph\deasylab1\Data\Puneet_et_al_R01_Lung_tumor_radiomics\snapshots'

    errDict = {}
    for d in os.scandir(dcmDir):

        try:

            _, id = os.path.split(d)
            print("Data dir: " + id)

            planC = pc.loadDcmDir(d)

            numScans = len(planC.scan)
            scanTypes = [s.scanInfo[0].imageType for s in planC.scan]
            if numScans == 2 and 'CT SCAN' in scanTypes and 'PT SCAN' in scanTypes:
                ptExportDir = os.path.join(niiDir,id)
                ctNiiFile = os.path.join(ptExportDir,'CT.nii.gz')
                ptNiiFile = os.path.join(ptExportDir,'PT.nii.gz')
                if not os.path.exists(ptExportDir):
                    os.mkdir(ptExportDir)
                # Export CT and PET to nii
                ctIndex = scanTypes.index('CT SCAN')
                ptIndex = scanTypes.index('PT SCAN')
                planC.scan[ctIndex].saveNii(ctNiiFile)
                planC.scan[ptIndex].saveNii(ptNiiFile)

                # Plot overlay of PET and CT images
                petScan3M = planC.scan[ptIndex].getScanArray()
                maxSuv = petScan3M.max()
                row,col,slc = np.where(np.abs(petScan3M-maxSuv) < 1e-5)
                row = row[0]
                col = col[0]
                slc = slc[0]

                # update viewer to display the central slice and capture screenshot
                xV, yV, zV = planC.scan[ptIndex].getScanXYZVals()
                minVal = petScan3M.min()
                maxVal = petScan3M.max()

                scanNum = [ctIndex, ptIndex]
                doseNum = []
                strNum = []
                displayMode = '2d'
                vectDict = {}
                viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
                        vwr.showNapari(scanNum, strNum, doseNum, vectDict, planC, displayMode)

                scan_layer[0].opacity = 0.7
                scan_layer[1].opacity = 0.7
                scan_layer[0].contrast_limits_range = [-1250, 250]
                scan_layer[0].contrast_limits = [-1250, 250]
                scan_layer[0].gamma = 0.7
                scan_layer[1].contrast_limits_range = [0, maxSuv]
                scan_layer[1].contrast_limits = [0, maxSuv]
                scan_layer[1].gamma = 0.7

                # Transverse display
                viewer.update({'dims': {'order': (2,0,1)}})
                viewer.dims.set_point(2, zV[slc])
                #viewer.camera.zoom = 5
                #viewer.camera.center = (0, 100, 100)
                screenshotTrans = viewer.screenshot(size =(600, 600))

                # Coronal display
                viewer.update({'dims': {'order': (0,2,1)}})
                viewer.dims.set_point(0, -yV[row])
                screenshotCor = viewer.screenshot(size =(600, 600))

                # Sagittal display
                viewer.update({'dims': {'order': (1,2,0)}})
                viewer.dims.set_point(1, xV[col])
                screenshotSag = viewer.screenshot(size =(600, 600))

                screenshot = np.concatenate((screenshotTrans,screenshotCor,screenshotSag),axis=1)

                viewer.close()
                fname = os.path.join(imageSaveDir,id+'.png')
                imsave(fname, screenshot)

            else:
                errDict[id] = scanTypes

        except:
            errDict[id] = []

    print(errDict)













    niiScan = r'L:\Data\fromKaiming\nii\gdrive\222562\scans\DICOM_Perfusion_Post_10_phase_inj_delay_20130626153915_9.nii.gz'
    niiScan = r'L:\Data\fromKaiming\nii\gdrive\210833\scans\DICOM_Axial_DCE_Perfusion_post_20150730154816_12.nii.gz'
    #niiScan = r'L:\Data\fromKaiming\nii\gdrive\210833\sFlux\Brain210833_source(rho_r)_E15_60_T_17_19.nii'
    niiScan = r'L:\Data\fromKaiming\nii\gdrive\210833\speed\Brain210833_speed_E15_60_T_43_45'
    niiScan = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\n4_nii_data\wk0\HN412D\Img.nii.gz'
    niiMask = r'L:\Data\fromKaiming\nii\gdrive\210833\assessors\maskKM.nii'
    niiMask = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\n4_nii_data\wk0\HN412D\Seg.nii.gz'
    planC = pc.loadNiiScan(niiScan, 'MR SCAN', 'LPS')
    asocScanNum = 0
    planC = pc.loadNiiStructure(niiMask, asocScanNum, planC, {1: 'tumor'})

    scan1 = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\zscore_nii_data\wk0\HN416D\Img.nii.gz'
    planC = pc.loadNiiScan(scan1, 'MR SCAN', 'LPS')
    scan2 = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\hist_eq_nii_data\wk0\HN416D\Img.nii.gz'
    planC = pc.loadNiiScan(scan2, 'MR SCAN', 'LPS', planC)
    seg1 = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\zscore_nii_data\wk0\HN416D\Seg.nii.gz'
    planC = pc.loadNiiStructure(seg1, 0, planC, {1: 'GTV'})

    import cerr.viewer as vwr
    scanNum = [0]
    doseNum = []
    strNum = [0]
    displayMode = '2d' # 'path' or 'surface'
    vectDict = {}
    viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
            vwr.showNapari(scanNum, strNum, doseNum, vectDict, planC, displayMode)


    niiDir = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\n4_nii_data'
    niiDir = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\orig_nii_data'
    submandDir = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\dicom_submandibular'
    patIdList = ['HN401D', 'HN403D', 'HN407D', 'HN410D', 'HN417D', 'HN419D', 'HN424D', \
                    'HN426D', 'HN428D', 'HN430D', 'HN437D', 'HN439D', 'HN440D', 'HN449D', \
                    'HN459D']
    wks = ['wk0', 'wk1', 'wk2']
    wksOrig = ['Wk0', 'Wk1', 'Wk2']
    patID = 'HN401D'
    scanV = []
    robustScanV = []
    #patIdList = ['HN401D']
    patIdList = ['HN417D']
    for patID in patIdList:
        imgFile = os.path.join(niiDir, wks[0], patID,'Img.nii.gz')
        segFile = os.path.join(niiDir, wks[0], patID,'Seg.nii.gz')
        imgFile = os.path.join(niiDir, patID, wksOrig[0],'Img.nii.gz')
        segFile = os.path.join(niiDir, patID, wksOrig[0],'Seg.nii.gz')
        submandSegDir = os.path.join(submandDir,patID,wks[0])
        opts = {'importMRPreciseValueFlag': 'yes'}
        planC = pc.loadDcmDir(submandSegDir, opts)
        scanOriStr = planC.scan[0].getScanOrientation()
        planC = pc.loadNiiScan(imgFile, 'T2 MR', scanOriStr, planC)
        planC = pc.loadNiiStructure(segFile, 1, planC, {1: 'tumor'})

        scanArrayDcm = planC.scan[0].getScanArray()
        scanArrayNii = planC.scan[1].getScanArray()
        diff3M = (scanArrayDcm - scanArrayNii) / (scanArrayNii+0.000001) * 100
        maxDiff = np.max(np.abs(diff3M))
        print(maxDiff)
        continue


        # Get structure names
        structNames = [s.structureName for s in planC.structure]
        submandInds = cerrStr.getMatchingIndex('DL_Left Submandibular', structNames, 'exact')
        submandInds.extend(cerrStr.getMatchingIndex('DL_Right Submandibular', structNames, 'exact'))

        # N4-corrected scan index
        n4SanNum = 1
        scan3M = planC.scan[n4SanNum].getScanArray()

        # Get T2 MRI intensities in sub-mandibular glands
        for strInd in submandInds:
            mask3M = rs.getStrMask(strInd, planC)
            scanV.extend(scan3M[mask3M])

        # Calculate Robust z-score
        medianVal = np.median(scanV)
        madVal = stats.median_abs_deviation(scanV)
        robustZscore3M = 0.6745 * (scan3M - medianVal) / madVal

        # Get robust z-score values in sub-mandibular glands
        for strInd in submandInds:
            mask3M = rs.getStrMask(strInd, planC)
            robustScanV.extend(robustZscore3M[mask3M])

        # # Add robust z-score to planC
        # x,y,z = planC.scan[0].getScanXYZVals()
        # planC = pc.import_scan_array(robustZscore3M, x, y, z, 'robust z-score', 0, planC)

    np.mean(scanV)
    np.mean(robustScanV)

    import pandas as pd
    import seaborn as sns
    scanDict = {'original_T2': scanV, 'robust_z_score': robustScanV}
    df = pd.DataFrame(data=scanDict)
    sns.displot(df,x='robust_z_score',kind="kde")

    # Visualize scan and segmentation
    import cerr.viewer as vwr
    scanNum = [0,1]
    doseNum = []
    strNum = [0]
    displayMode = '2d' # 'path' or 'surface'
    vectDict = {}
    viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
            vwr.showNapari(scanNum, strNum, doseNum, vectDict, planC, displayMode)




### ========== Mirror-scope ==============
viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
        vwr.showNapari(scanNum, strNum, doseNum, vectDict, planC, displayMode)

# Initialize mirror-scope images for the base and moving scans

def getScanExtents(r, c, s,numRows, numCols, numSlcs,
                   offset, axNum):
    r = int(r)
    c = int(c)
    s = int(s)
    halfOff = int(offset / 2)
    halfOff = offset
    if axNum == 1:
        rMin = r - halfOff
        rMax = r + halfOff
        cMin = c
        cMax = c + int(offset*1.5)
        sMin = s
        sMax = s + 1
    elif axNum == 2:
        pass
    elif axNum == 0:
        pass
    if rMin < 0:
        rMin = 0
    if rMax < 0:
        rMax = 0
    if rMin >= numRows:
        rMin = numRows - 1
    if rMax >= numRows:
        rMax = numRows - 1
    if cMin < 0:
        cMin = 0
    if cMax < 0:
        cMax = 0
    if cMin >= numCols:
        cMin = numCols - 1
    if cMax >= numCols:
        cMax = numCols - 1
    if sMin < 0:
        sMin = 0
    if sMax < 0:
        sMax = 0
    if sMin >= numSlcs:
        sMin = numSlcs - 1
    if sMax >= numSlcs:
        sMax = numSlcs - 1

    return rMin, rMax, cMin, cMax, sMin, sMax


baseInd = 0
movInd = 1

mrrScp = np.zeros((31,31,1))
#mrrScp[:] = np.NAN
# scan_affine = scan_layer[].affine
xb,yb,zb = planC.scan[baseInd].getScanXYZVals()
yb = -yb
dxB,dyB,dzB = planC.scan[baseInd].getScanSpacing()
xm,ym,zm = planC.scan[movInd].getScanXYZVals()
ym = -ym
dxM,dyM,dzM = planC.scan[movInd].getScanSpacing()


#currentSlcInd = np.where(np.abs(z - viewer.dims.point[2]) < 0.0001)
currPt = viewer.dims.point
mirror_affine = np.array([[dyB, 0, 0, yb[0]], [0, dxB, 0, xb[0]], [0, 0, dzB, zb[0]], [0, 0, 0, 1]])
mrrScpLayerBase = viewer.add_image(mrrScp,name='Mirror-Scope-base',
                            opacity=1, colormap=scan_layer[0].colormap,
                            affine=mirror_affine,
                            blending="opaque",interpolation2d="linear",
                            interpolation3d="linear"
                            )
mrrScpLayerMov = viewer.add_image(mrrScp,name='Mirror-Scope-mov',
                            opacity=1, colormap=scan_layer[0].colormap,
                            affine=mirror_affine,
                            blending="opaque",interpolation2d="linear",
                            interpolation3d="linear"
                            )
mirrorLine = viewer.add_shapes([[0,0,0], [0,0,0]], name = 'Mirror-line',
                               face_color = "red", edge_color = "red", edge_width = 0.5,
                               opacity=1, blending="opaque",
                               affine=scan_layer[0].affine.affine_matrix,
                               shape_type='line')
numLayers = len(viewer.layers)
mrrScpLayerBase.interactive = True
mrrScpLayerBase.mouse_pan = False
mrrScpLayerBase.visible = False
mrrScpLayerMov.interactive = True
mrrScpLayerMov.mouse_pan = False
mrrScpLayerMov.visible = False
mirrorLine.visible = False
viewer.layers.selection.active = mrrScpLayerBase

mirrorSize = 15


def getLayerIndex(scanNum,lyrType,layers):
    if lyrType == 'scan':
        scanNums = [s.metadata['scanNum'] for s in layers]
        return scanNums.index(scanNum)
    return

def updateMirror(baseInd, movInd, mrrScpLayerBase, mrrScpLayerMov, mirrorLine):
    currPt = viewer.cursor.position
    baseLyrInd = getLayerIndex(scanNum,'scan',scan_layer)
    movLyrInd = getLayerIndex(scanNum,'scan',scan_layer)
    rb,cb,sb = np.round(scan_layer[baseLyrInd].world_to_data(currPt))
    numRows, numCols, numSlcs = planC.scan[baseInd].getScanSize()
    axNum = 1
    rMinB, rMaxB, cMinB, cMaxB, sMinB, sMaxB = getScanExtents(rb,cb,sb,numRows, numCols, numSlcs,
                                                              mirrorSize, axNum)
    rm,cm,sm = np.round(scan_layer[movLyrInd].world_to_data(viewer.cursor.position))
    numRows, numCols, numSlcs = planC.scan[movInd].getScanSize()
    axNum = 1
    rMinM, rMaxM, cMinM, cMaxM, sMinM, sMaxM = getScanExtents(rm,cm,sm,numRows, numCols, numSlcs,
                                                              mirrorSize, axNum)
    if rMinB == rMaxB == 0:
        rMaxB = 1
    elif rMinB == rMaxB:
        rMinB = rMinB - 1
    if cMinB == cMaxB == 0:
        cMaxB = 1
    elif cMinB == cMaxB:
        cMinB = cMinB-1

    if rMinM == rMaxM == 0:
        rMaxM = 1
    elif rMinM == rMaxM:
        rMinM = rMinM - 1
    if cMinM == cMaxM == 0:
        cMaxM = 1
    elif cMinM == cMaxM:
        cMinM = cMinM-1

    deltaXmov = dxM * (cMaxM - cMinM)
    deltaYmov = dyM * (rMaxM - rMinM)
    croppedScanBase = scan_layer[0].data[rMinB:rMaxB,cMinB:cMaxB,int(sb)]
    croppedScanMov = scan_layer[1].data[rMinM:rMaxM,cMinM:cMaxM,int(sm)]
    croppedScanMov = np.flip(croppedScanMov,axis=1)
    if viewer.dims.order[0] == 2:
        mirrorAffineB = np.array([[dyB, 0, 0, yb[rMinB]], [0, dxB, 0, xb[cMinB]], [0, 0, dzB, zb[int(sb)]], [0, 0, 0, 1]])
        mirrorAffineM = np.array([[dyM, 0, 0, ym[rMinM]], [0, dxM, 0, xm[cMinM]-deltaXmov], [0, 0, dzM, zm[int(sm)]], [0, 0, 0, 1]])
    elif viewer.dims.order[0] == 1:
        mirrorAffineB = np.array([[dyB, 0, 0, yb[int(rb)]], [0, dxB, 0, xb[cMinB]], [0, 0, dzB, zb[sMinB]], [0, 0, 0, 1]])
        mirrorAffineM = np.array([[dyM, 0, 0, ym[int(rm)]], [0, dxM, 0, xm[cMinM]], [0, 0, dzM, zm[sMinM]], [0, 0, 0, 1]])
    else:
        mirrorAffineB = np.array([[dyB, 0, 0, yb[rMinB]], [0, dxB, 0, xb[int(cb)]], [0, 0, dzB, zb[sMinB]], [0, 0, 0, 1]])
        mirrorAffineM = np.array([[dyM, 0, 0, ym[rMinM]], [0, dxM, 0, xm[int(cm)]], [0, 0, dzM, zm[sMinM]], [0, 0, 0, 1]])

    mrrScpLayerBase.affine.affine_matrix = mirrorAffineB
    mrrScpLayerMov.affine.affine_matrix = mirrorAffineM
    #cropNumRows, cropNumCols, cropNumSlcs = croppedScan.shape
    mrrScpLayerBase.data = croppedScanBase[:,:,None]
    mrrScpLayerMov.data = croppedScanMov[:,:,None]
    mrrScpLayerBase.refresh()
    mrrScpLayerBase.contrast_limits = scan_layer[baseLyrInd].contrast_limits
    mrrScpLayerBase.contrast_limits_range = scan_layer[baseLyrInd].contrast_limits_range
    mrrScpLayerMov.contrast_limits = scan_layer[movLyrInd].contrast_limits
    mrrScpLayerMov.contrast_limits_range = scan_layer[movLyrInd].contrast_limits_range

    data = [[[rMinB,cb,sb], [rMaxB,cb,sb]]]
    data.append([[rMinB,2*cMinB-cMaxB,sb], [rMinB,cMaxB,sb]])
    data.append([[rMinB,2*cMinB-cMaxB,sb], [rMaxB,2*cMinB-cMaxB,sb]])
    data.append([[rMaxB,2*cMinB-cMaxB,sb], [rMaxB,cMaxB,sb]])
    data.append([[rMinB,cMaxB,sb], [rMaxB,cMaxB,sb]])
    mirrorLine.data = np.asarray(data)
    mirrorLine.refresh()

#@mrrScpLayer.mouse_drag_callbacks.append
def mirror_scope(layer, event):
    # on click
    #print('mouse clicked')
    mrrScpLayerBase.visible = True
    mrrScpLayerMov.visible = True
    mirrorLine.visible = True
    dragged = False
    updateMirror(baseInd, movInd, mirrorSize, mrrScpLayerBase, mrrScpLayerMov, mirrorLine)
    #r,c,s = np.round(layer.world_to_data(viewer.cursor.position))
    #s = int(s)
    # mirror_affine = np.array([[dy, 0, 0, y[0]], [0, dx, 0, x[0]-dx*len(x)], [0, 0, dz, z[s]], [0, 0, 0, 1]])
    # mrrScpLayer = viewer.add_image(mrrScp,name='Mirror-Scope',
    #                             opacity=1, colormap='gray',
    #                             affine=mirror_affine,
    #                             blending="opaque",interpolation2d="linear",
    #                             interpolation3d="linear"
    #                             )
    # mrrScpLayer.interactive = True
    # viewer.layers.selection.active = scan_layer[0]
    # print(event.pos)
    # print(viewer.cursor.position)  # (0,0) is the center of the upper left pixel
    # print(layer.world_to_data(viewer.cursor.position))
    #print('mirror-scope layer created')
    yield

    # on move
    while event.type == 'mouse_move':
        dragged = True
        #print(event.pos)
        #print(viewer.cursor.position)  # (0,0) is the center of the upper left pixel

        updateMirror(baseInd, movInd, mirrorSize, mrrScpLayerBase, mrrScpLayerMov, mirrorLine)
        # rb,cb,sb = np.round(scan_layer[0].world_to_data(viewer.cursor.position))
        # numRows, numCols, numSlcs = planC.scan[baseInd].getScanSize()
        # axNum = 1
        # rMinB, rMaxB, cMinB, cMaxB, sMinB, sMaxB = getScanExtents(rb,cb,sb,numRows, numCols, numSlcs,
        #                                                           mirrorSize, axNum)
        # rm,cm,sm = np.round(scan_layer[1].world_to_data(viewer.cursor.position))
        # numRows, numCols, numSlcs = planC.scan[movInd].getScanSize()
        # axNum = 1
        # rMinM, rMaxM, cMinM, cMaxM, sMinM, sMaxM = getScanExtents(rm,cm,sm,numRows, numCols, numSlcs,
        #                                                           mirrorSize, axNum)
        # if rMinB == rMaxB == 0:
        #     rMaxB = 1
        # elif rMinB == rMaxB:
        #     rMinB = rMinB - 1
        # if cMinB == cMaxB == 0:
        #     cMaxB = 1
        # elif cMinB == cMaxB:
        #     cMinB = cMinB-1
        #
        # if rMinM == rMaxM == 0:
        #     rMaxM = 1
        # elif rMinM == rMaxM:
        #     rMinM = rMinM - 1
        # if cMinM == cMaxM == 0:
        #     cMaxM = 1
        # elif cMinM == cMaxM:
        #     cMinM = cMinM-1
        #
        # deltaXmov = dxM * (cMaxM - cMinM)
        # deltaYmov = dyM * (rMaxM - rMinM)
        # croppedScanBase = scan_layer[0].data[rMinB:rMaxB,cMinB:cMaxB,int(sb)]
        # croppedScanMov = scan_layer[1].data[rMinM:rMaxM,cMinM:cMaxM,int(sm)]
        # croppedScanMov = np.flip(croppedScanMov,axis=1)
        # if viewer.dims.order[0] == 2:
        #     mirrorAffineB = np.array([[dyB, 0, 0, yb[rMinB]], [0, dxB, 0, xb[cMinB]], [0, 0, dzB, zb[int(sb)]], [0, 0, 0, 1]])
        #     mirrorAffineM = np.array([[dyM, 0, 0, ym[rMinM]], [0, dxM, 0, xm[cMinM]-deltaXmov], [0, 0, dzM, zm[int(sm)]], [0, 0, 0, 1]])
        # elif viewer.dims.order[0] == 1:
        #     mirrorAffineB = np.array([[dyB, 0, 0, yb[int(rb)]], [0, dxB, 0, xb[cMinB]], [0, 0, dzB, zb[sMinB]], [0, 0, 0, 1]])
        #     mirrorAffineM = np.array([[dyM, 0, 0, ym[int(rm)]], [0, dxM, 0, xm[cMinM]], [0, 0, dzM, zm[sMinM]], [0, 0, 0, 1]])
        # else:
        #     mirrorAffineB = np.array([[dyB, 0, 0, yb[rMinB]], [0, dxB, 0, xb[int(cb)]], [0, 0, dzB, zb[sMinB]], [0, 0, 0, 1]])
        #     mirrorAffineM = np.array([[dyM, 0, 0, ym[rMinM]], [0, dxM, 0, xm[int(cm)]], [0, 0, dzM, zm[sMinM]], [0, 0, 0, 1]])
        #
        # mrrScpLayerBase.affine.affine_matrix = mirrorAffineB
        # mrrScpLayerMov.affine.affine_matrix = mirrorAffineM
        # #cropNumRows, cropNumCols, cropNumSlcs = croppedScan.shape
        # mrrScpLayerBase.data = croppedScanBase[:,:,None]
        # mrrScpLayerMov.data = croppedScanMov[:,:,None]
        # mrrScpLayerBase.refresh()
        # mrrScpLayerBase.contrast_limits = scan_layer[0].contrast_limits
        # mrrScpLayerBase.contrast_limits_range = scan_layer[0].contrast_limits_range
        # mrrScpLayerMov.contrast_limits = scan_layer[1].contrast_limits
        # mrrScpLayerMov.contrast_limits_range = scan_layer[1].contrast_limits_range
        #
        # data = [[[rMinB,cb,sb], [rMaxB,cb,sb]]]
        # data.append([[rMinB,2*cMinB-cMaxB,sb], [rMinB,cMaxB,sb]])
        # data.append([[rMinB,2*cMinB-cMaxB,sb], [rMaxB,2*cMinB-cMaxB,sb]])
        # data.append([[rMaxB,2*cMinB-cMaxB,sb], [rMaxB,cMaxB,sb]])
        # data.append([[rMinB,cMaxB,sb], [rMaxB,cMaxB,sb]])
        # mirrorLine.data = np.asarray(data)
        # mirrorLine.refresh()

        yield

    # on release
    if dragged:
        pass
        #print('drag end')
        #viewer.layers.remove(mrrScpLayer)

mrrScpLayerBase.mouse_drag_callbacks.append(mirror_scope)
mrrScpLayerMov.mouse_drag_callbacks.append(mirror_scope)

