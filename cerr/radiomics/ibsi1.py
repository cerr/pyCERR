"""
This module contains routines for IBSI1 compatible radiomics calculation
"""

import numpy as np
from cerr.radiomics import first_order, gray_level_cooccurence, run_length,\
    size_zone, neighbor_gray_level_dependence, neighbor_gray_tone
from cerr.utils.mask import compute_boundingbox
from cerr.radiomics import preprocess, textureUtils
import json


def getDirectionOffsets(direction):
    if direction == 1:
        offsetsM = np.asarray([[1, 0, 0],
                               [0, 1, 0],
                               [1, 1, 0],
                               [1, -1, 0],
                               [0, 0, 1],
                               [1, 0, 1],
                               [1, 1, 1],
                               [1, -1, 1],
                               [0, 1, 1],
                               [0, -1, 1],
                               [-1, -1, 1],
                               [-1, 0, 1],
                               [-1, 1, 1]
                                            ],
                              dtype=int)
    elif direction == 2:
        offsetsM = np.asarray([[1, 0, 0],
                               [0, 1, 0],
                               [1, 1, 0],
                               [1, -1, 0]
                                        ],
                              dtype=int)

    return offsetsM


def calcRadiomicsForImgType(volToEval, maskBoundingBox3M, morphMask3M, gridS, paramS):

    featDict = {}

    # Get feature extraction settings
    firstOrderOffsetEnergy = np.double(paramS['settings']['firstOrder']['offsetForEnergy'])
    firstOrderEntropyBinWidth = None
    firstOrderEntropyBinNum = None
    textureBinNum = None
    textureBinWidth = None
    if 'binWidthEntropy' in paramS['settings']['firstOrder'] \
        and isinstance(paramS['settings']['firstOrder']['binWidthEntropy'], (int, float)):
        firstOrderEntropyBinWidth = paramS['settings']['firstOrder']['binWidthEntropy']
    if 'binNumEntropy' in paramS['settings']['firstOrder'] \
        and isinstance(paramS['settings']['firstOrder']['binNumEntropy'], (int, float)):
        firstOrderEntropyBinNum = paramS['settings']['firstOrder']['binNumEntropy']
    if 'texture' in paramS['settings'] and paramS['settings']['texture'] != {}:
        if 'gtdm' in paramS['featureClass'] :
            patch_radius = paramS['settings']['texture']['patchRadiusVox']
        if'gldm' in paramS['featureClass'] :
            patch_radius = paramS['settings']['texture']['patchRadiusVox']
            difference_threshold = paramS['settings']['texture']['imgDiffThresh']
        if 'binwidth' in paramS['settings']['texture'] \
            and isinstance(paramS['settings']['texture']['binwidth'], (int, float)):
            textureBinWidth = paramS['settings']['texture']['binwidth']
        if 'binNum' in paramS['settings']['texture'] \
            and isinstance(paramS['settings']['texture']['binNum'], (int, float)):
            textureBinNum = paramS['settings']['texture']['binNum']
        if (textureBinNum is not None) and (textureBinWidth is not None):
            raise Exception("Please specify either the number of bins or bin-width for quantization")
        if any(name in ['glcm','glrlm','glszm'] for name in paramS['featureClass'].keys()):
            dirString = paramS['settings']['texture']['directionality']
            avgType = paramS['settings']['texture']['avgType']
            glcmVoxelOffset = paramS['settings']['texture']['voxelOffset']
            if avgType == 'feature':
                cooccurType = 2
                rlmType = 2
            elif avgType == 'texture':
                cooccurType = 1
                rlmType = 1
            if dirString == '3D':
                direction = 1
                szmDir = 1
            elif dirString == '2D':
                direction = 2
                szmDir = 2

        # Min/Max for image quantization
        minClipIntensity = None
        maxClipIntensity = None
        if 'minClipIntensity' in paramS['settings']['texture'] and \
               isinstance(paramS['settings']['texture']['minClipIntensity'], (int, float)):
            minClipIntensity = paramS['settings']['texture']['minClipIntensity']
        if 'maxClipIntensity' in paramS['settings']['texture'] and \
                isinstance(paramS['settings']['texture']['maxClipIntensity'], (int, float)):
            maxClipIntensity = paramS['settings']['texture']['maxClipIntensity']

    # Shape  features
    if 'shape' in paramS['featureClass'] and paramS['featureClass']['shape']["featureList"] != {}:
        from cerr.radiomics.shape import compute_shape_features
        featDict['shape'] = compute_shape_features(morphMask3M,gridS['xValsV'],gridS['yValsV'],gridS['zValsV'])

    # Assign nan values outside the mask, so that min/max within the mask are used for quantization
    volToEval[~maskBoundingBox3M] = np.nan

    # Texture-based scalar features
    if 'texture' in paramS['settings']:
        # Quantization
        quantized3M = preprocess.imquantize_cerr(volToEval, num_level=textureBinNum,\
                                                 xmin=minClipIntensity, xmax=maxClipIntensity, binwidth=textureBinWidth)
        nL = quantized3M.max()

        if any(name in ['glcm','glrlm','glszm'] for name in paramS['featureClass'].keys()):
            offsetsM = getDirectionOffsets(direction)
            offsetsM = offsetsM * glcmVoxelOffset
    else:
        quantized3M = volToEval

     # First order features
    if 'firstOrder' in paramS['featureClass'] and paramS['featureClass']['firstOrder']["featureList"] != {}:
        voxelVol = np.prod(gridS["PixelSpacingV"]) * 1000 # units of mm
        scanV = volToEval[maskBoundingBox3M]
        featDict['firstOrder'] = first_order.radiomics_first_order_stats(scanV, voxelVol,
                                        firstOrderOffsetEnergy, firstOrderEntropyBinWidth, firstOrderEntropyBinNum)

    # GLCM
    if 'glcm' in paramS['featureClass'] and paramS['featureClass']['glcm']["featureList"] != {}:
        glcmM = gray_level_cooccurence.calcCooccur(quantized3M, offsetsM, nL, cooccurType)
        featDict['glcm'] = gray_level_cooccurence.cooccurToScalarFeatures(glcmM)

    # RLM
    if 'glrlm' in paramS['featureClass'] and paramS['featureClass']['glrlm']["featureList"] != {}:
        rlmM = run_length.calcRLM(quantized3M,offsetsM,nL,rlmType)
        numVoxels = np.sum(maskBoundingBox3M.astype(int))
        if rlmType == 1: # merged RLMs for offsets
            numVoxels *= offsetsM.shape[0]
        featDict['glrlm'] = run_length.rlmToScalarFeatures(rlmM, numVoxels)

    # SZM
    if 'glszm' in paramS['featureClass'] and paramS['featureClass']['glszm']["featureList"] != {}:
        szmM = size_zone.calcSZM(quantized3M,nL,szmDir)
        numVoxels = np.sum(maskBoundingBox3M.astype(int))
        featDict['glszm'] = size_zone.szmToScalarFeatures(szmM, numVoxels)

    # NGLDM
    if 'gldm' in paramS['featureClass'] and paramS['featureClass']['gldm']["featureList"] != {}:
        s = neighbor_gray_level_dependence.calcNGLDM(quantized3M, patch_radius, nL, difference_threshold)
        featDict['gldm'] = neighbor_gray_level_dependence.ngldmToScalarFeatures(s, numVoxels)

    # NGTDM
    if 'gtdm' in paramS['featureClass'] and paramS['featureClass']['gtdm']["featureList"] != {}:
        s,p,Nvc = neighbor_gray_tone.calcNGTDM(quantized3M, patch_radius, nL)
        featDict['gtdm'] = neighbor_gray_tone.ngtdmToScalarFeatures(s,p,Nvc)

    return featDict



def computeScalarFeatures(scanNum, structNum, settingsFile, planC):

    with open(settingsFile, ) as settingsFid:
        radiomicsSettingS = json.load(settingsFid)

    # Pre-process Image
    (processedScan3M, processedMask3M, morphMask3M, gridS, radiomicsSettingS, diagS) = \
       preprocess.preProcessForRadiomics(scanNum, structNum, radiomicsSettingS, planC)
    minr,maxr,minc,maxc,mins,maxs,__ = compute_boundingbox(processedMask3M)
    voxSizeV = gridS["PixelSpacingV"]

    ############################################
    # Calculate IBSI-1 features
    ############################################
    imgTypeDict = radiomicsSettingS['imageType']
    imgTypes = imgTypeDict.keys()

    mapToIBSI = False
    if 'mapFeaturenamesToIBSI' in radiomicsSettingS['settings'] and \
        radiomicsSettingS['settings']['mapFeaturenamesToIBSI'] == "yes":
        mapToIBSI = True

    if mapToIBSI:
        mapClassDict, mapFeatDict = getIBSINameMap()
        origKeys = list(diagS.keys())
        for feat in origKeys:
            diagS[mapFeatDict[feat]] = diagS[feat]
            del diagS[feat]

    avgType = ''
    directionality = ''
    if 'texture' in radiomicsSettingS['settings'] and \
            'avgType' in radiomicsSettingS['settings']['texture']:
        avgType = radiomicsSettingS['settings']['texture']['avgType']
    if 'texture' in radiomicsSettingS['settings'] and \
            'directionality' in radiomicsSettingS['settings']['texture']:
        directionality = radiomicsSettingS['settings']['texture']['directionality']

    # Loop over image filters
    featDictAllTypes = diagS
    for imgType in imgTypes:
        if imgType.lower() == "original":
            imgFeatName = 'original'
            # Calc. radiomic features
            maskBoundingBox3M = processedMask3M[minr:maxr+1, minc:maxc+1, mins:maxs+1]
            croppedScan3M = processedScan3M[minr:maxr+1, minc:maxc+1, mins:maxs+1]
            featDict = calcRadiomicsForImgType(croppedScan3M, maskBoundingBox3M, morphMask3M, gridS, radiomicsSettingS)
            featDictAllTypes = {**featDictAllTypes, **createFlatFeatureDict(featDict, imgFeatName, avgType, directionality, mapToIBSI)}
        else:
            # Extract filter & padding parameters
            filterTypeParamS = radiomicsSettingS['imageType'][imgType]
            padSizeV = [0,0,0]
            padMethod = "none"
            if 'padding' in radiomicsSettingS["settings"] and radiomicsSettingS["settings"]["padding"]["method"].lower()!='none':
                padSizeV = radiomicsSettingS["settings"]["padding"]["size"]
                padMethod = radiomicsSettingS["settings"]["padding"]["method"]

            # Apply image filters
            if not isinstance(filterTypeParamS,list):
                filterTypeParamS = [filterTypeParamS]

            for nFilt in range(len(filterTypeParamS)):
                filterParamS = filterTypeParamS[nFilt]
                filterParamS["VoxelSize_mm"]  = voxSizeV * 10
                filterParamS["Padding"] = {"Size":padSizeV,"Method": padMethod,"Flag":False}

                paddedResponseS = textureUtils.processImage(imgType, processedScan3M, processedMask3M, filterParamS)
                filterName = list(paddedResponseS.keys())[0] # must be single output
                filteredPadScan3M = paddedResponseS[filterName]

                # Remove padding
                maskBoundingBox3M = processedMask3M[minr:maxr+1, minc:maxc+1, mins:maxs+1]
                filteredScan3M = filteredPadScan3M[minr:maxr+1, minc:maxc+1, mins:maxs+1]
                filteredScan3M[~maskBoundingBox3M] = np.nan
                # Calc. radiomic features
                featDict = calcRadiomicsForImgType(filteredScan3M, maskBoundingBox3M, morphMask3M, gridS, radiomicsSettingS)

                # Aggregate features
                imgFeatName = createFieldNameFromParameters(imgType, filterParamS)
                featDictAllTypes = {**featDictAllTypes, **createFlatFeatureDict(featDict, imgFeatName, avgType, directionality, mapToIBSI)}

    return featDictAllTypes, diagS

def getIBSINameMap():
    classDict = {'shape': 'morph',
                 'firstOrder': 'stat',
                 'glcm': 'cm',
                 'glrlm': 'rlm',
                 'glszm': 'szm',
                 'gldm': 'ngl',
                 'gtdm': 'ngt'}
    featDict = {'numVoxelsOrig':'diag_n_voxel',
                'numVoxelsInterpReseg':'diag_n_voxel_interp_reseg',
                'meanIntensityInterpReseg':'diag_mean_int_interp_reseg',
                'maxIntensityInterpReseg':'diag_max_int_interp_reseg',
                'minIntensityInterpReseg':'diag_min_int_interp_reseg',
                'majorAxis': 'pca_maj_axis',
                'minorAxis':'pca_min_axis',
                'leastAxis': 'pca_least_axis',
                'flatness': 'pca_flatness',
                'elongation': 'pca_elongation',
                'max3dDiameter': 'diam',
                'max2dDiameterAxialPlane': 'max2dDiameterAxialPlane',
                'max2dDiameterSagittalPlane': 'max2dDiameterSagittalPlane',
                'max2dDiameterCoronalPlane': 'max2dDiameterCoronalPlane',
                'surfArea': 'area_mesh',
                'volume': 'vol_approx',
                'volumeDensityAABB': 'morph_vol_dens_aabb',
                'filledVolume': 'filled_vol_approx',
                'Compactness1': 'comp_1',
                'Compactness2': 'comp_2',
                'spherDisprop': 'sph_dispr',
                'sphericity': 'sphericity',
                'surfToVolRatio': 'av',
                'min': 'min',
                'max': 'max',
                'mean': 'mean',
                'range': 'range',
                'std': 'std',
                'var': 'var',
                'median': 'median',
                'skewness': 'skew',
                'kurtosis': 'kurt',
                'entropy': 'entropy',
                'rms': 'rms',
                'energy': 'energy',
                'totalEnergy': 'total_energy',
                'meanAbsDev': 'mad',
                'medianAbsDev': 'medad',
                'P10': 'p10',
                'P90': 'p90',
                'robustMeanAbsDev': 'maad',
                'robustMedianAbsDev': 'medaad',
                'interQuartileRange': 'iqr',
                'coeffDispersion': 'qcod',
                'coeffVariation': 'cov',
                'energy': 'energy',
                'jointEntropy': 'joint_entr',
                'jointMax': 'joint_max',
                'jointAvg': 'joint_avg',
                'jointVar': 'joint_var',
                'sumAvg': 'sum_avg',
                'sumVar': 'sum_var',
                'sumEntropy': 'sum_entr',
                'contrast': 'contrast',
                'invDiffMom': 'inv_diff_mom',
                'invDiffMomNorm': 'inv_diff_mom_norm',
                'invDiff': 'inv_diff',
                'invDiffNorm': 'inv_diff_norm',
                'invVar': 'inv_var',
                'dissimilarity': 'dissimilarity',
                'diffEntropy': 'diff_entr',
                'diffVar': 'diff_var',
                'diffAvg': 'diff_avg',
                'sumAvg': 'sum_avg',
                'sumVar': 'sum_var',
                'sumEntropy': 'sum_entr',
                'corr': 'corr',
                'clustTendency': 'clust_tend',
                'clustShade': 'clust_shade',
                'clustPromin': 'clust_prom',
                'haralickCorr': 'haral_corr',
                'autoCorr': 'auto_corr',
                'firstInfCorr': 'info_corr1',
                'secondInfCorr': 'info_corr2',
                'shortRunEmphasis': 'sre',
                'longRunEmphasis': 'lre',
                'grayLevelNonUniformity': 'glnu',
                'grayLevelNonUniformityNorm': 'glnu_norm',
                'runLengthNonUniformity': 'rlnu',
                'runLengthNonUniformityNorm': 'rlnu_norm',
                'runPercentage': 'r_perc',
                'lowGrayLevelRunEmphasis': 'lgre',
                'highGrayLevelRunEmphasis': 'hgre',
                'shortRunLowGrayLevelEmphasis': 'srlge',
                'shortRunHighGrayLevelEmphasis': 'srhge',
                'longRunLowGrayLevelEmphasis': 'lrlge',
                'longRunHighGrayLevelEmphasis': 'lrhge',
                'grayLevelVariance': 'gl_var',
                'runLengthVariance': 'rl_var',
                'runEntropy': 'rl_entr',
                'smallAreaEmphasis': 'sze',
                'largeAreaEmphasis': 'lze',
                'sizeZoneNonUniformity': 'sznu',
                'sizeZoneNonUniformityNorm': 'sznu_norm',
                'zonePercentage': 'z_perc',
                'lowGrayLevelZoneEmphasis': 'lgze',
                'highGrayLevelZoneEmphasis': 'hzge',
                'smallAreaLowGrayLevelEmphasis': 'szlge',
                'largeAreaHighGrayLevelEmphasis': 'lzhge',
                'smallAreaHighGrayLevelEmphasis': 'szhge',
                'largeAreaLowGrayLevelEmphasis': 'lzlge',
                'sizeZoneVariance': 'zs_var',
                'zoneEntropy': 'zs_entr',
                'lowDependenceEmphasis': 'lde',
                'highDependenceEmphasis': 'hde',
                'lowGrayLevelCountEmphasis': 'lgce',
                'highGrayLevelCountEmphasis': 'hgce',
                'lowDependenceLowGrayLevelEmphasis': 'ldlge',
                'lowDependenceHighGrayLevelEmphasis': 'ldhge',
                'highDependenceLowGrayLevelEmphasis': 'hdlge',
                'highDependenceHighGrayLevelEmphasis': 'hdhge',
                'dependenceCountNonuniformity': 'dcnu',
                'dependenceCountNonuniformityNorm': 'dcnu_norm',
                'dependenceCountPercentage': 'dc_perc',
                'dependenceCountVariance': 'dc_var',
                'dependenceCountEntropy': 'dc_entr',
                'dependenceCountEnergy': 'dc_energy',
                'coarseness': 'coarseness',
                'busyness': 'busyness',
                'complexity': 'complexity',
                'strength': 'strength',
                }

    return classDict, featDict

def createFieldNameFromParameters(imageType, settingS):
    """
    Create unique fieldname for radiomics features returned by calcRadiomicsForImgType
    :param imageType: 'Original' or filtered image type (see processImage.m for valid options)
    :param settingS: Parameter dictionary for radiomics feature extraction
    :return: fieldName
    """
    imageType = imageType.lower()
    if imageType == 'original':
        fieldName = imageType
    elif imageType == 'sobel':
        fieldName = imageType
    elif imageType == 'mean':
        kernelSize = settingS['KernelSize']
        if len(kernelSize)==2 or kernelSize[2]==0:
            fieldName = f"{imageType}_kernel_size{settingS['KernelSize'][0]}x{settingS['KernelSize'][1]}"
        else:
            fieldName = f"{imageType}_kernel_size{settingS['KernelSize'][0]}\
                        x{settingS['KernelSize'][1]}x{settingS['KernelSize'][2]}"
    elif imageType == 'haralickcooccurance':
        dirC = ['3d', '2d']
        dirType = dirC[settingS['Directionality']]
        settingsStr = f"{settingS['Type']}_{dirType}_{settingS['NumLevels']}levels_patchsize" \
                   f"{settingS['PatchSize'][0]}{settingS['PatchSize'][1]}" \
                   f"{settingS['PatchSize'][2]}"
        fieldName = f"{imageType}_{settingsStr}"
    elif imageType == 'wavelets':
        settingsStr = f"{settingS['Wavelets']}_{settingS['Index']}_{settingS['Direction']}"
        if 'RotationInvariance' in settingS and settingS['RotationInvariance'] and settingS['RotationInvariance']:
            settingsStr += f"_rot{settingS['RotationInvariance']['Dim']}_agg{settingS['RotationInvariance']['AggregationMethod']}"
        fieldName = f"{imageType}_{settingsStr}"
    elif imageType == 'log':
        sigmaV = ' '.join(map(str, settingS['Sigma_mm']))
        cutoffV = ' '.join(map(str, settingS['CutOff_mm']))
        settingsStr = f"sigma_{sigmaV}mm_cutoff_{cutoffV}mm"
        fieldName = f"{imageType}_{settingsStr}"
    elif imageType == 'gabor':
        voxelSize_mm = ' '.join(map(str, settingS['VoxelSize_mm']))
        settingsStr = f"voxSz{voxelSize_mm}mm_Sigma{settingS['Sigma_mm']}mm_AR" \
                  f"{settingS['SpatialAspectRatio']}_wavLen{settingS['Wavlength_mm']}mm"
        thetaV = ' '.join(map(str, settingS['Orientation']))
        if len(settingS['Orientation']) == 1:
            settingsStr += f"_Orient{thetaV}"
        else:
            settingsStr += f"_OrientAvg_{thetaV}"
        fieldName = f"{imageType}_{settingsStr}"
    elif imageType in ['lawsconvolution','rotationinvariantlawsconvolution']:
        settingsStr = f"{settingS['Direction']}_type{settingS['Type']}_norm{settingS['Normalize']}"
        if 'RotationInvariance' in settingS and settingS['RotationInvariance'] and settingS['RotationInvariance']:
                settingsStr += f"_rot{settingS['RotationInvariance']['Dim']}_agg{settingS['RotationInvariance']['AggregationMethod']}"
        fieldName = f"{imageType}_{settingsStr}"
    elif imageType in['lawsenergy','rotationinvariantlawsenergy']:
        energyKernelSize = '_'.join(map(str, settingS['EnergyKernelSize']))
        energyKernelSize = energyKernelSize.replace(' ', 'x')
        settingsStr = f"{settingS['Direction']}_type{settingS['Type']}_norm{settingS['Normalize']}" \
                   f"_energyKernelSize{energyKernelSize}"
        if 'RotationInvariance' in settingS and settingS['RotationInvariance'] and settingS['RotationInvariance']:
            settingsStr += f"_rot{settingS['RotationInvariance']['Dim']}_agg{settingS['RotationInvariance']['AggregationMethod']}"
        fieldName = f"{imageType}_{settingsStr}"
    else:
        raise ValueError('Invalid image type')

    # Ensure valid fieldname
    fieldName = fieldName.replace(' ', '').replace('.', '_').replace('-', '_')

    return fieldName

def createFlatFeatureDict(featDict, imageType, avgType, directionality, mapToIBSI = False):

    featClasses = featDict.keys()
    flatFeatDict = {}
    if avgType == 'feature':
        avgString = 'avg'
    else:
        avgString = 'comb'
    if directionality.lower() == '2d':
        dirString = '2_5D'
    else:
        dirString = '3D'

    if mapToIBSI:
        mapClassDict, mapFeatDict = getIBSINameMap()

    for featClass in featClasses:
        if mapToIBSI:
            mapFeatClass = mapClassDict[featClass]
        else:
            mapFeatClass = featClass
        for item in featDict[featClass].items():
            itemName = item[0]
            if mapToIBSI:
                itemName = mapFeatDict[itemName]
            if mapFeatClass in ["cm", "rlm", "glcm", "glrlm"]:
                featStr = imageType + '_' + mapFeatClass + '_' + itemName + '_' + dirString
                flatFeatDict[featStr + '_' + avgString] = np.mean(item[1])
                flatFeatDict[featStr + '_Median'] = np.median(item[1])
                if avgString == 'avg':
                    flatFeatDict[featStr + '_StdDev'] = np.std(item[1], ddof=1)
                else:
                    if isinstance(item[1], (int,float,np.number)):
                        flatFeatDict[featStr + '_StdDev'] = item[1]
                    else:
                        flatFeatDict[featStr + '_StdDev'] = item[1][0]
                flatFeatDict[featStr + '_Min'] = np.min(item[1])
                flatFeatDict[featStr + '_Max'] = np.max(item[1])
            else:
                if mapFeatClass in ["morph", "stat", "shape", "firstOrder"]:
                    flatFeatDict[imageType + '_' + mapFeatClass + '_' + itemName] = item[1]
                else:
                    flatFeatDict[imageType + '_' + mapFeatClass + '_' + itemName + '_' + dirString] = item[1]
    return flatFeatDict

def writeFeaturesToFile(featList, csvFileName, writeHeader = True):
    import csv
    if not isinstance(featList,list):
        featList = [featList]
    with open(csvFileName, 'a', newline='') as csvfile:
        flatFeatDict = featList[0]
        fieldnames = flatFeatDict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if writeHeader:
            writer.writeheader()
        for flatFeatDict in featList:
            writer.writerow(flatFeatDict)
