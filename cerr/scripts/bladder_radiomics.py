import sys

from cerr import plan_container as pc
from cerr.registration import register
from cerr import viewer as vwr
from cerr.utils import mask
import cerr.contour.rasterseg as rs
import numpy as np
import os

def calc_feat_diff(matS, pyS, featType):
    featDiffV = []
    for key in matS[featType].keys():
        featDiffV.append(np.mean((matS[featType][key] - pyS[key]) / (matS[featType][key] + np.finfo(float).eps) * 100))
    return featDiffV, np.asarray(list(matS[featType].keys()))


if __name__ == "__main__":

    projectDir = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\All file'
    dataBatches = ['PURE-01_Batch-1_03312023','PURE-01_Batch-2_04052023','PURE-01_Batch-3-04182023',
                   'PURE-01_Batch-4_06152023','PURE-01_Batch-5_06222023',
                   'PURE-01_Batch-6_07272023','PURE-01_Batch-7_08032023']
    scanSegPaths = []
    for dataBatch in dataBatches:
        dirBatch = os.path.join(projectDir, dataBatch)
        dirs = os.listdir(dirBatch)
        patDirs = [f.path for f in os.scandir(dirBatch) if f.is_dir()]
        for patDir in patDirs:
            dateDirs = [f.path for f in os.scandir(patDir) if f.is_dir()]
            for dateDir in dateDirs:
                sequences = os.listdir(dateDir)
                seqsToUse = ['T1w', 'T2w', 'T1w-Post']
                for seq in seqsToUse:
                    seqDir = os.path.join(dateDir, seq)
                    if not os.path.exists(seqDir):
                        continue
                    seriesDirs = [f.path for f in os.scandir(seqDir) if f.is_dir()]
                    segFiles = []
                    for f in os.scandir(seqDir):
                        dirName = f.path
                        if not f.is_dir():
                            if 'ROI' in dirName and 'nii.gz' in dirName:
                                segFiles.append(dirName)
                    segScanPath = {}
                    if len(seriesDirs) == 1:
                        seriesDir = seriesDirs[0]
                        segScanPath['scan'] = seriesDir
                    if len(segFiles) == 1:
                        segFile = segFiles[0]
                        segScanPath['seg'] = segFile
                    scanSegPaths.append(segScanPath)



    dcmDir = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\All file\PURE-01_Batch-7_08032023\PURE-01_001-137\D2019_12_03\T1w\S0301_4894'
    structFile = r'S:\Amita Dave HN\PURE-01\ROI_BORA_ALL DATA_AMRESHA 2023\All file\PURE-01_Batch-7_08032023\PURE-01_001-137\D2019_12_03\T1w\301_T1_pre_ROI_YA.nii.gz'
    planC = pc.loadDcmDir(dcmDir)
    planC = pc.loadNiiStructure(structFile, 0, planC)
