from cerr import plan_container as pc
import os
import ants


if __name__ == "__main__":

    dcmDir = r'L:\Maria\HNC_NormalTissue_pCT\Trismus_PrePostSegGuideline\38125959'
    niiOutDir = r'L:\Maria\HNC_NormalTissue_pCT\38125959_reg_test'
    antsOutDir = r'L:\Maria\HNC_NormalTissue_pCT\38125959_reg_test\ants_reg'

    planC = pc.loadDcmDir(dcmDir)

    baseScanIndex = 0
    movScanIndex = 1

    baseNiiName = os.path.join(niiOutDir,'baseScan.nii.gz')
    movNiiName = os.path.join(niiOutDir,'movScan.nii.gz')
    warpedScanFileName = os.path.join(niiOutDir,'mov_warped.nii.gz')
    planC.scan[baseScanIndex].saveNii(baseNiiName)
    planC.scan[movScanIndex].saveNii(movNiiName)

    imgAntsBase = ants.image_read(baseNiiName)
    imgAntsMov = ants.image_read(movNiiName)
    result = ants.registration(imgAntsBase, imgAntsMov, type_of_transform = 'SyN',
                               write_composite_transform = True, outprefix = antsOutDir)
    warpedimage = ants.apply_transforms(fixed=imgAntsBase, moving=imgAntsMov,
                                               transformlist=result['fwdtransforms'] )
    warpedimage.image_write(warpedScanFileName)
    planC = pc.loadNiiScan(warpedScanFileName, 'CT SCAN', '', planC)

    # Write nii files for structures
    chewStrList = ['Left_masseter','Right_masseter','Left_medial_pterygoid','Right_medial_pterygoid']
    strList = [str.structureName for str in planC.structure]
    assocUIDv = [str.assocScanUID for str in planC.structure]
    strFileNameList = []
    strIndList = []
    for chewStr in chewStrList:
        ind = strList.index(chewStr)
        if not planC.scan[baseScanIndex].scanUID == assocUIDv[ind]:
            continue
        strIndList.append(ind)
        strFileName = os.path.join(niiOutDir,chewStr+'.nii.gz')
        strFileNameList.append(strFileName)
        planC.structure[ind].saveNii(strFileName, planC)
        imgAntsStr = ants.image_read(strFileName)
        warpedStrImage = ants.apply_transforms(fixed=imgAntsBase, moving=imgAntsStr,
                                               transformlist=result['fwdtransforms'],
                                               defaultvalue = 0, singleprecision = True)
        warpedStrFileName = os.path.join(niiOutDir,chewStr+'_warped.nii.gz')
        warpedStrImage.image_write(warpedStrFileName)
        planC = pc.loadNiiStructure(warpedStrFileName, baseScanIndex,planC, {1: chewStr+'_warped'})


    # Visualize
    from cerr import viewer as vwr
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0,1,2], [], [], {}, '2d')

    numStructs = len(planC.structure)
    deformedStrList = range(numStructs-len(strIndList), numStructs)
    strToView = strIndList.copy()
    strToView.extend(deformedStrList)
    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0,1,2], [84,85,86,87,187,188,189,190], [], {}, '2d')
