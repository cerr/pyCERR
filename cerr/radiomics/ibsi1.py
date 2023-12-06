import numpy as np

import cerr.radiomics.filters
from cerr.radiomics import first_order, gray_level_cooccurence, run_length,\
    size_zone, neighbor_gray_level_dependence, neighbor_gray_tone
from cerr.utils import bbox
from cerr.radiomics import preprocess
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
    from cerr.radiomics.shape import compute_shape_features
    shapeS = compute_shape_features(maskBoundingBox3M,gridS['xValsV'],gridS['yValsV'],gridS['zValsV'])

    # First order features
    voxelVol = np.prod(gridS["pixelSpacingV"]) * 1000 # units of mm
    scanV = volToEval[maskBoundingBox3M]
    firstOrderFeatS = first_order.radiomics_first_order_stats(scanV, voxelVol, firstOrderOffsetEnergy, firstOrderEntropyBinWidth)

    # Texture-based scalar features
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
    glcmM = gray_level_cooccurence.calcCooccur(quantized3M, offsetsM, nL, cooccurType)
    glcmFeatS = gray_level_cooccurence.cooccurToScalarFeatures(glcmM)

    rlmM = run_length.calcRLM(quantized3M,offsetsM,nL,rlmType)
    numVoxels = croppedMask3M.sum()
    rlmFeatS = run_length.rlmToScalarFeatures(rlmM, numVoxels)

    # SZM
    szmM = size_zone.calcSZM(quantized3M,nL,szmDir)
    numVoxels = croppedMask3M.sum()
    szmFeatS = size_zone.szmToScalarFeatures(szmM, numVoxels)

    # NGLDM
    s = neighbor_gray_level_dependence.calcNGLDM(quantized3M, patch_radius, nL, difference_threshold)
    ngldmFeatS = neighbor_gray_level_dependence.ngldmToScalarFeatures(s, numVoxels)

    # NGTDM
    s,p = neighbor_gray_tone.calcNGTDM(quantized3M, patch_radius, nL)
    ngtdmFeatS = neighbor_gray_tone.ngtdmToScalarFeatures(s,p,numVoxels)

    return shapeS, firstOrderFeatS, glcmFeatS, rlmFeatS, szmFeatS, ngldmFeatS, ngtdmFeatS



def computeScalarFeatures(scanNum, structNum, settingsFile, planC):

    with open(settingsFile, ) as settingsFid:
        radiomicsSettingS = json.load(settingsFid)

    # Pre-process Image
    (processedScan3M, processedMask3M, gridS, radiomicsSettingS, diagS) = \
        preprocess.preProcessForRadiomics(scanNum, structNum, radiomicsSettingS, planC)

    # Calculate IBSI-1 features
    shapeS, firstOrderFeatS, glcmFeatS, rlmFeatS, szmFeatS, ngldmFeatS, ngtdmFeatS = \
        calcRadiomicsForImgType(processedScan3M, processedMask3M, gridS, radiomicsSettingS)

    return shapeS, firstOrderFeatS, glcmFeatS, rlmFeatS, szmFeatS, ngldmFeatS, ngtdmFeatS


def writeFeaturesToFile():
    pass
