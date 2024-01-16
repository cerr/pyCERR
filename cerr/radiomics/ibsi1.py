import numpy as np

from cerr.radiomics import first_order, gray_level_cooccurence, run_length,\
    size_zone, neighbor_gray_level_dependence, neighbor_gray_tone
from cerr.utils import bbox
from cerr.radiomics import preprocess, textureFilters
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


def calcRadiomicsForImgType(volToEval, maskBoundingBox3M, gridS, paramS):

    featDict = {}

    # Get feature extraction settings
    firstOrderOffsetEnergy = paramS['settings']['firstOrder']['offsetForEnergy']
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
        featDict['shape'] = compute_shape_features(maskBoundingBox3M,gridS['xValsV'],gridS['yValsV'],gridS['zValsV'])

    # First order features
    if 'firstOrder' in paramS['featureClass'] and paramS['featureClass']['firstOrder']["featureList"] != {}:
        voxelVol = np.prod(gridS["PixelSpacingV"]) * 1000 # units of mm
        scanV = volToEval[maskBoundingBox3M]
        featDict['firstOrder'] = first_order.radiomics_first_order_stats(scanV, voxelVol,
                                        firstOrderOffsetEnergy, firstOrderEntropyBinWidth, firstOrderEntropyBinNum)

    # Texture-based scalar features
    if 'texture' in paramS['settings']:
        # Crop and Pad
        (rmin,rmax,cmin,cmax,smin,smax,_) = bbox.compute_boundingbox(maskBoundingBox3M)
        croppedScan3M = volToEval[rmin:rmax+1,cmin:cmax+1,smin:smax+1]
        croppedMask3M = maskBoundingBox3M[rmin:rmax+1,cmin:cmax+1,smin:smax+1]
        croppedScan3M[~croppedMask3M] = np.NAN

        # Quantization
        quantized3M = preprocess.imquantize_cerr(croppedScan3M, num_level=textureBinNum, xmin=minClipIntensity, xmax=maxClipIntensity, binwidth=textureBinWidth)
        nL = quantized3M.max()
        offsetsM = getDirectionOffsets(direction)
        offsetsM = offsetsM * glcmVoxelOffset

    # GLCM
    if 'glcm' in paramS['featureClass'] and paramS['featureClass']['glcm']["featureList"] != {}:
        glcmM = gray_level_cooccurence.calcCooccur(quantized3M, offsetsM, nL, cooccurType)
        featDict['glcm'] = gray_level_cooccurence.cooccurToScalarFeatures(glcmM)

    # RLM
    if 'glrlm' in paramS['featureClass'] and paramS['featureClass']['glrlm']["featureList"] != {}:
        rlmM = run_length.calcRLM(quantized3M,offsetsM,nL,rlmType)
        numVoxels = croppedMask3M.sum()
        featDict['glrlm'] = run_length.rlmToScalarFeatures(rlmM, numVoxels)

    # SZM
    if 'glszm' in paramS['featureClass'] and paramS['featureClass']['glszm']["featureList"] != {}:
        szmM = size_zone.calcSZM(quantized3M,nL,szmDir)
        numVoxels = croppedMask3M.sum()
        featDict['glszm'] = size_zone.szmToScalarFeatures(szmM, numVoxels)

    # NGLDM
    if 'gldm' in paramS['featureClass'] and paramS['featureClass']['gldm']["featureList"] != {}:
        s = neighbor_gray_level_dependence.calcNGLDM(quantized3M, patch_radius, nL, difference_threshold)
        featDict['gldm'] = neighbor_gray_level_dependence.ngldmToScalarFeatures(s, numVoxels)

    # NGTDM
    if 'gtdm' in paramS['featureClass'] and paramS['featureClass']['gtdm']["featureList"] != {}:
        s,p = neighbor_gray_tone.calcNGTDM(quantized3M, patch_radius, nL)
        featDict['gtdm'] = neighbor_gray_tone.ngtdmToScalarFeatures(s,p,numVoxels)

    return featDict



def computeScalarFeatures(scanNum, structNum, settingsFile, planC):

    with open(settingsFile, ) as settingsFid:
        radiomicsSettingS = json.load(settingsFid)

    # Pre-process Image
    (processedScan3M, processedMask3M, gridS, radiomicsSettingS, diagS) = \
        preprocess.preProcessForRadiomics(scanNum, structNum, radiomicsSettingS, planC)

    # Calculate IBSI-1 features
    imgTypeDict = radiomicsSettingS['imageType']
    imgTypes = imgTypeDict.keys()
    featDictAllTypes = {}
    for imgType in imgTypes:
        if imgType.lower() == "original":
            featDict = calcRadiomicsForImgType(processedScan3M, processedMask3M, gridS, radiomicsSettingS)
        else:
            # Call filtering routines here for imgType other than Original
            filterParamS = radiomicsSettingS['imageType'][imgType]
            paddedResponseS = textureFilters.processImage(imgType, processedScan3M, processedMask3M, filterParamS)
            filterName = paddedResponseS.keys()[0] # must be single output
            filteredScan3M = paddedResponseS[filterName]
            featDict = calcRadiomicsForImgType(filteredScan3M, processedMask3M, gridS, radiomicsSettingS)
            #imgType = imgType + equivalent of createFieldNameFromParameters
        featDictAllTypes = {**featDictAllTypes, **createFlatFeatureDict(featDict, imgType)}
    return featDictAllTypes

def createFlatFeatureDict(featDict, imageType):
    featClasses = featDict.keys()
    flatFeatDict = {}
    for featClass in featClasses:
        for item in featDict[featClass].items():
            if featClass in ["glcm", "glrlm"]:
                flatFeatDict[imageType + '_' + featClass + '_Mean_' + item[0]] = np.mean(item[1])
                flatFeatDict[imageType + '_' + featClass + '_Median_' + item[0]] = np.median(item[1])
                flatFeatDict[imageType + '_' + featClass + '_StdDev_' + item[0]] = np.std(item[1], ddof=1)
                flatFeatDict[imageType + '_' + featClass + '_Min_' + item[0]] = np.min(item[1])
                flatFeatDict[imageType + '_' + featClass + '_Max_' + item[0]] = np.max(item[1])
            else:
                flatFeatDict[imageType + '_' + featClass + '_' + item[0]] = item[1]
    return flatFeatDict

def writeFeaturesToFile(featList, csvFileName):
    import csv
    if not isinstance(featList,list):
        featList = [featList]
    with open(csvFileName, 'w', newline='') as csvfile:
        flatFeatDict = featList[0]
        fieldnames = flatFeatDict.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for flatFeatDict in featList:
            writer.writerow(flatFeatDict)
