def capturs_snaps():
    dataDir = r'\\vpensmph\deasylab3\data\ROBIN\supplement_funding_2024\planningCT'
    csvGtvNameFile = r'\\vpensmph\deasylab3\data\ROBIN\supplement_funding_2024\gtv_names.csv'
    snapSavdDir = r'\\vpensmph\deasylab3\data\ROBIN\supplement_funding_2024\planningCTSnaps'

    import os
    import csv

    os.environ["NAPARI_APPLICATION_IPY_INTERACTIVE"] = "0"

    idPlanDict = {}
    with open(csvGtvNameFile, 'r') as file:
        # Create a csv.DictReader object
        fileDict = csv.DictReader(file)
        for rowDict in fileDict:
            idPlanDict[rowDict['id']] = \
                (rowDict['plan_name'], rowDict['GTV_primary_name'])


    import os
    import numpy as np
    import csv
    from cerr import plan_container as pc
    from cerr.dataclasses import structure as cerrStr
    from cerr.dataclasses import scan as cerrScn
    from cerr import viewer as vwr
    from cerr.contour import rasterseg as rs
    from skimage.io import imsave, imshow

    allPatDirs = os.listdir(dataDir)
    for patDir in allPatDirs:
        planName = idPlanDict[patDir][0]
        planDir = os.path.join(dataDir, patDir, planName)
        planC = pc.loadDcmDir(planDir)
        # Get structure index
        strNames = [s.structureName for s in planC.structure]
        gtvName = idPlanDict[patDir][1]
        strNum = strNames.index(gtvName)
        assocScanUID = planC.structure[strNum].assocScanUID
        scanNum = cerrScn.getScanNumFromUID(assocScanUID, planC)
        if len(planC.dose) == 1:
            doseNum = 0
        else:
            doseNum = []
        doseNum = []
        vectorDict = {}
        # Display scan
        viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
            vwr.showNapari(planC, scanNum, [strNum], doseNum, vectors_dict={}, displayMode='2d')

        # Show structure central slice
        mask3M = rs.getStrMask(strNum, planC)
        rV, cV, sV = np.where(mask3M)
        midSliceInd = int(np.round(sV.mean()))
        # update viewer to display the central slice and capture screenshot
        xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
        viewer.dims.set_point(2, zV[midSliceInd])

        # Capture and save screenshot
        planGtvStr = patDir + '_' + idPlanDict[patDir][0] + '_' + idPlanDict[patDir][1]
        captureFname = os.path.join(snapSavdDir,
                                      planGtvStr + '.png')
        _ = viewer.screenshot(path=captureFname)

        viewer.close()


# --- Main runner ---
if __name__ == '__main__':
    capturs_snaps()
