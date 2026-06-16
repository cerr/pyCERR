
from cerr import plan_container as pc
from cerr.registration import register
from cerr import viewer as vwr
from cerr.utils import mask
import cerr.contour.rasterseg as rs
import numpy as np
import os
from cerr.dataclasses import structure as cerrStr
from cerr.dataclasses import scan as cerrScn
from pathlib import Path
import ants

if __name__ == "__main__":

    def findMatchingDir(parentDir, matchStr, isDir=True):
        dirsMatchList = []
        startDir = Path(parentDir)
        # Use rglob to recursively find all items ("*")
        for path_obj in startDir.rglob("*"):
            # Check if the item is a directory and contains the target string
            if isDir and path_obj.is_dir() and matchStr in path_obj.name:
                dirsMatchList.append(str(path_obj.resolve())) # Use resolve() to get the absolute path
            elif matchStr in path_obj.name:
                dirsMatchList.append(str(path_obj.resolve()))
        return dirsMatchList


    #inhaleSegNii = os.path.join(inhaleDir, 'thorax_ct_oars_EHRLICH^HARVEY_00183656_CT_2020-12-29_123254_4D_inhale.nii.gz')
    #exhaleSegNii = os.path.join(exhaleDir, 'thorax_ct_oars_EHRLICH^HARVEY_00183656_CT_2020-12-29_123254_4D_exhale.nii.gz')

    #inhaleDir = r'L:\Maria\LA-NSCLC_Durva_N230\N_65_ValidationCohort\DCM\4DCT\00183656\2020-12__Studies\EHRLICH^HARVEY_00183656_CT_2020-12-29_123254_4D.CT.Chest.3mm_0%.Linear,.iDose.(3)_n89__00000'
    #exhaleDir = r'L:\Maria\LA-NSCLC_Durva_N230\N_65_ValidationCohort\DCM\4DCT\00183656\2020-12__Studies\EHRLICH^HARVEY_00183656_CT_2020-12-29_123254_4D.CT.Chest.3mm_50%.Linear,.iDose.(3)_n89__00000'
    #inhaleSegNii = os.path.join(inhaleDir, 'thorax_ct_oars_EHRLICH^HARVEY_00183656_CT_2020-12-29_123254_4D_inhale.nii.gz')
    #exhaleSegNii = os.path.join(exhaleDir, 'thorax_ct_oars_EHRLICH^HARVEY_00183656_CT_2020-12-29_123254_4D_exhale.nii.gz')


    dataDir = r'L:\Maria\LA-NSCLC_Durva_N230\N_65_ValidationCohort\DCM\4DCT'
    regDataDir = r'L:\Maria\LA-NSCLC_Durva_N230\N_65_ValidationCohort\DCM\test\ventilation_features'

    ptDirs = [name for name in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir, name))]
    ptDirs = ['35197621']

    for ptId in ptDirs:

        #ptDir = r'L:\Maria\LA-NSCLC_Durva_N230\N_65_ValidationCohort\DCM\4DCT\35021184'
        #regDir = r'L:\Maria\LA-NSCLC_Durva_N230\N_65_ValidationCohort\DCM\test\ventilation_features\00183656'

        ptDir = os.path.join(dataDir, ptId)
        regDir = os.path.join(regDataDir, ptId)

        # Create directory to store registration
        if os.path.exists(regDir):
            continue
        os.makedirs(regDir, exist_ok=True)

        # Define the starting directory path
        start_path = Path(regDir) # '.' refers to the current directory

        # Define the folder name to search for (e.g., 'results')
        #inhaleDirName = '_0%.Linear'
        #exhaleDirName = '_50%.Linear'
        inhaleDirName = '_0%.'
        exhaleDirName = '_50%.'

        # Use rglob with a pattern for directories. The trailing '/' ensures only directories match.
        # Note: This might require specific shell settings in some environments, but in Python itself
        # we can use is_dir() for certainty. The pattern "**" matches zero or more directories.
        inhaleDir = findMatchingDir(ptDir, inhaleDirName)
        exhaleDir = findMatchingDir(ptDir, exhaleDirName)

        if len(inhaleDir) == 1:
            inhaleDir = inhaleDir[0]
        else:
            continue

        if len(exhaleDir) == 1:
            exhaleDir = exhaleDir[0]
        else:
            continue

        inhaleSegNii = findMatchingDir(inhaleDir, '.nii.gz', isDir=False)
        exhaleSegNii = findMatchingDir(exhaleDir, '.nii.gz', isDir=False)

        if len(inhaleSegNii) == 1:
            inhaleSegNii = inhaleSegNii[0]
        else:
            continue

        if len(exhaleSegNii) == 1:
            exhaleSegNii = exhaleSegNii[0]
        else:
            continue

        planC = pc.loadDcmDir(exhaleDir)
        planC = pc.loadDcmDir(inhaleDir, {}, planC)

        baseScanNum = 0
        movScanNum = 1

        planC = pc.loadNiiStructure(exhaleSegNii, baseScanNum, planC)
        planC = pc.loadNiiStructure(inhaleSegNii, movScanNum, planC)

        baseLungStrNums = [0,1]
        movLungStrNums = [4,5]

        baseScanSize = planC.scan[baseScanNum].getScanSize()
        baseMask3M = np.zeros(baseScanSize, dtype=bool)

        movScanSize = planC.scan[movScanNum].getScanSize()
        movMask3M = np.zeros(movScanSize, dtype=bool)

        for strNum in baseLungStrNums:
            baseMask3M = baseMask3M | rs.getStrMask(strNum, planC)

        for strNum in movLungStrNums:
            movMask3M = movMask3M | rs.getStrMask(strNum, planC)

        # strNames = [s.structureName for s in planC.structure]
        # strMatchV = cerrStr.getMatchingIndex('Lung_L', strNames, 'exact')
        # strMatchV.append(cerrStr.getMatchingIndex('Lung_R', strNames, 'exact'))
        #
        # for strNum in strMatchV:
        #     if cerrScn.getScanNumFromUID(planC.structure[strNum].assocScanUID, planC) == baseScanNum:
        #         baseMask3M = baseMask3M | rs.getStrMask(strNum, planC)
        #     else:
        #         movMask3M = movMask3M | rs.getStrMask(strNum, planC)

        planC = pc.importStructureMask(baseMask3M, baseScanNum, 'Lungs_AI_generated', planC)
        baseLungInd = len(planC.structure) - 1
        planC = pc.importStructureMask(movMask3M, movScanNum, 'Lungs_AI_generated', planC)
        movLungInd = len(planC.structure) - 1

        expansionMargin = 0.1 # cm
        planC = cerrStr.getSurfaceExpand(baseLungInd, expansionMargin, planC)
        baseLungInd = len(planC.structure) - 1
        planC = cerrStr.getSurfaceExpand(movLungInd, expansionMargin, planC)
        movLungInd = len(planC.structure) - 1


        baseScanFile = os.path.join(regDir, 'exhaleCT.nii.gz')
        movScanFile = os.path.join(regDir, 'inhaleCT.nii.gz')
        baseMaskFile = os.path.join(regDir, 'exhaleLungMask.nii.gz')
        movMaskFile = os.path.join(regDir, 'inhaleLungMask.nii.gz')

        planC.scan[baseScanNum].saveNii(baseScanFile)
        planC.scan[movScanNum].saveNii(movScanFile)


        planC.structure[baseLungInd].saveNii(baseMaskFile, planC)
        planC.structure[movLungInd].saveNii(movMaskFile, planC)

        # Define fixed and moving images + masks
        imgAntsBase = ants.image_read(baseScanFile)
        imgAntsMov = ants.image_read(movScanFile)
        maskAntsBase = ants.image_read(baseMaskFile)
        maskAntsMov = ants.image_read(movMaskFile)

        # Output files
        movScanName = os.path.basename(movScanFile)
        warpedScanFileName = os.path.join(regDir, 'warped_' + movScanName)
        txPath = os.path.join(regDir,'ants_reg')
        regFile = os.path.join(regDir, 'ants_regComposite.h5')

        tx = 'antsRegistrationSyN[bo]'

        # Register images
        #baseSiz = maskAntsBase.shape
        #movSiz = maskAntsMov.shape
        #mask=np.full(baseSiz,True), moving_mask=np.full(movSiz,True), mask_all_stages=True
        regResult = ants.registration(imgAntsBase, imgAntsMov, type_of_transform=tx,
                                      initial_transform = 'identity',
                                      mask=maskAntsBase, moving_mask=maskAntsMov, mask_all_stages=True,
                                      write_composite_transform=True, outprefix=txPath)

        # Warp moving scan
        warpedScanImage = ants.apply_transforms(fixed=imgAntsBase, moving=imgAntsMov,
                                                transformlist=regResult['fwdtransforms'])
        ants.image_write(warpedScanImage, warpedScanFileName)


    regResult


