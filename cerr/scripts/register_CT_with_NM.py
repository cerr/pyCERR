
from cerr import plan_container as pc
from cerr import viewer as vwr
import numpy as np
import os
import ants
from datetime import datetime

if __name__ == "__main__":

    regDir = r'\\pensmph\MPHShared\Public\Alexandre\regFiles'
    dataDir = r'\\pensmph\MPHShared\Public\Alexandre\forAditya'
    planC = pc.loadDcmDir(dataDir)

    seriesDateV = np.array([datetime.strptime(s.scanInfo[0].seriesDate, '%Y%m%d') for s in planC.scan])
    modalityV = np.array([s.scanInfo[0].imageType for s in planC.scan])
    folderNameV = np.array([os.path.basename(os.path.dirname(s.scanInfo[0].scanFileName)) for s in planC.scan])
    sortIndV = np.argsort(seriesDateV)
    modalityV == 'CT SCAN'
    seriesDateV[sortIndV]
    modalityV[sortIndV]

    baseScanInd = np.where((modalityV == 'CT SCAN') & (folderNameV == 'CT1'))[0][0]
    movScanIndV = np.where((modalityV == 'CT SCAN') & (folderNameV != 'CT1'))[0]

    for scanNum in movScanIndV:

        # Find index of SPECT scan associated with moving CT scan
        spectScanInd = np.where(seriesDateV == seriesDateV[scanNum])[0]
        spectScanInd = spectScanInd[spectScanInd != scanNum][0]

        # Get scan names
        spectScanName = os.path.basename(os.path.dirname(planC.scan[spectScanInd].scanInfo[0].scanFileName))
        spectMovScanFile = os.path.join(regDir, spectScanName + '_movScan.nii.gz')
        baseScanName = os.path.basename(os.path.dirname(planC.scan[baseScanInd].scanInfo[0].scanFileName))
        baseScanFile = os.path.join(regDir, baseScanName + '_baseScan.nii.gz')
        movScanName = os.path.basename(os.path.dirname(planC.scan[scanNum].scanInfo[0].scanFileName))
        movScanFile = os.path.join(regDir, movScanName + '_movScan.nii.gz')

        # Save to nii
        planC.scan[baseScanInd].saveNii(baseScanFile)
        planC.scan[scanNum].saveNii(movScanFile)
        planC.scan[spectScanInd].saveNii(spectMovScanFile)
        #planC.structure[baseLungInd].saveNii(baseMaskFile, planC)
        #planC.structure[movLungInd].saveNii(movMaskFile, planC)

        # Define fixed and moving images + masks
        imgAntsBase = ants.image_read(baseScanFile)
        imgAntsMov = ants.image_read(movScanFile)
        imgAntsSpectMov = ants.image_read(spectMovScanFile)
        #maskAntsBase = ants.image_read(baseMaskFile)
        #maskAntsMov = ants.image_read(movMaskFile)

        # Output files
        warpedScanCTFileName = os.path.join(regDir, 'warped_' + movScanName + '.nii.gz')
        warpedScanSPECTFileName = os.path.join(regDir, 'warped_' + spectScanName + '.nii.gz')
        txPath = os.path.join(regDir,'ants_reg')
        regFile = os.path.join(regDir, 'ants_regComposite.h5')

        # Transform
        tx = 'antsRegistrationSyNQuick[b]' #'antsRegistrationSyN[b]'

        # Register images
        #baseSiz = maskAntsBase.shape
        #movSiz = maskAntsMov.shape
        #mask=np.full(baseSiz,True), moving_mask=np.full(movSiz,True), mask_all_stages=True
        regResult = ants.registration(imgAntsBase, imgAntsMov, type_of_transform=tx,
                                      write_composite_transform=True, outprefix=txPath)

        # Warp moving scan
        warpedScanImage = ants.apply_transforms(fixed=imgAntsBase, moving=imgAntsMov,
                                                transformlist=regResult['fwdtransforms'])
        ants.image_write(warpedScanImage, warpedScanCTFileName)

        warpedScanImage = ants.apply_transforms(fixed=imgAntsBase, moving=imgAntsSpectMov,
                                                transformlist=regResult['fwdtransforms'])
        ants.image_write(warpedScanImage, warpedScanSPECTFileName)


    viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
                vwr.showNapari(planC, [0,1,2,3,4,5], [], [], {}, '2d')


