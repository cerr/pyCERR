from IPython.display import Image, display
from cerr import datasets
import os
from cerr import plan_container as pc
from cerr import viewer as vwr
from skimage.io import imsave, imshow
import cerr.contour.rasterseg as rs
import numpy as np


# Animate structures
from cerr import plan_container as pc
from napari_animation import Animation
import numpy as np
from cerr import viewer as vwr

from cerr import datasets
import os
datasetsDir = os.path.dirname(datasets.__file__)
phantom_dir = os.path.join(datasetsDir,'radiomics_phantom_dicom')
dicomPath = os.path.join(phantom_dir, 'pat_1')
#dicomPath = r'\\vpensmph\deasylab1\Aditya\AI_workshop\test_data\0617_test\0617-292370_09-09-2000-35932' # Add Path
animationPath = r'C:\software\pycerr_scripts' #Outputh Path
num_angles = 10 # Number of angles (more angles will take longer to render and generate but you can see a better animation)
name_list = ['carina', 'ESOPHAGUS'] # Name of the organ/dicom structure names that you want to create animations for
name_list = [0] # Name or index of the organ/dicom structure names that you want to create animations for


planC = pc.loadDcmDir(dicomPath)
scanNumV = [0] # list of scan indices from planC.scan
doseNumV = [] # list of dose indices from planC.dose
strNumV = []
for idx, name in enumerate(name_list):
    if isinstance(name,str):
        for i in range(len(planC.structure)):
            if planC.structure[i].structureName == name:
                print(planC.structure[i].structureName)
                strNumV.append(i) # list of structure indices from planC.structure
    else:
        strNumV.append(name) # assume index

#planC.structure[2].structureColor = [0, 0, 255]

# Open up viewer to be able to create the animation
viewer, scan_layer, struct_layer, dose_layer, dvf_layer = \
    vwr.showNapari(planC, scan_nums=scanNumV, struct_nums=strNumV, dose_nums=doseNumV, vectors_dict={}, displayMode='3d')
viewer.dims.ndisplay = 3

# ---------
# 1. Make the struct layers translucent_no_depth so you can through them
# Only useful if you are interesting in seeing something inside the structure
# 2. Change the opacity and colormap to orange
# 3. Normalize the contrast
# You can add more to this and personalize what you want to appear and how
# ---------

for strLyr in  struct_layer:
    strLyr.blending = 'translucent_no_depth'
    strLyr.opacity = 0.5
#struct_layer[0].colormap = 'bop orange'
#dose_layer[0].opacity = 0.5
min_val, max_val = -500, 1500 #scan_layer[0].data.min(), scan_layer[0].data.max()
scan_layer[0].contrast_limits = (min_val, max_val)


# Generate the animation by moving the camera and then recording a keyframe
animation = Animation(viewer)
for angle in np.linspace(0, 360, num=num_angles):
    viewer.camera.angles = (180, angle, 0)  # Rotate around z-axis
    animation.capture_keyframe()
for angle in np.linspace(0, 360, num=num_angles):
    viewer.camera.angles = (180, 0, angle)  # Rotate around x-axis
    animation.capture_keyframe()


# Render and save the animation
animation.animate(os.path.join(animationPath, 'anim_tumor.gif'), fps=60)
viewer.close()


# zScoreSaveDir = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\zscore_nii_data'
# imageSaveDir = r'\\vpensmph\deasylab1\Aditya\HN_T2_radiomics\snapshots\wk0'
#

# Xinan Vector field

# Load DICOM
dcmDir = r'\\pensmphDeasylab\deasyLab2\Data\DCE_OMT\dicom_WeiHuang_and_TCIA\BreastChemo36\BreastChemo36\BC36V1_concatenated'
planC = pc.loadDcmDir(dcmDir)

# Load vector field
vectorFile = r'C:\Users\aptea\Downloads\test_vector_v2.csv'
import csv
ptsList = []
with open(vectorFile) as csvfile:
    pts = csv.reader(csvfile)
    for pt in pts:
        ptsList.append([float(p) for p in pt])

# Swap z-origin slice
numPts = len(ptsList)
scanNum = 0
numRows, numCols, numSlcs = planC.scan[scanNum].getScanSize()
for i in range(numPts):
    ptsList[i][2] = numSlcs - ptsList[i][2]
    ptsList[i][5] = numSlcs - ptsList[i][5]


imgToPhysTransM = planC.scan[scanNum].Image2VirtualPhysicalTransM
vectors = np.empty((numPts,2,3), dtype=np.float32)
for i,pt in enumerate(ptsList):
    xyzStart = np.matmul(imgToPhysTransM, np.asarray(pt[:3]+[1]).T)
    xyzEnd = np.matmul(imgToPhysTransM, np.asarray(pt[3:6]+[1]).T)
    deltaXYZ = (xyzEnd - xyzStart) * 10 # cm to mm conversion
    vectors[i,0,:] = pt[:3] # -y,x,z
    vectors[i,1,:] = [-deltaXYZ[1], deltaXYZ[0], deltaXYZ[2]] # -dy,dx,dz

lengthV = np.sum(vectors **2, axis = 2)[:,1] ** 0.5
feats = {'length (mm)': lengthV,  'dx (mm)': np.abs(vectors[:,1,1]),
         'dy (mm)': np.abs(vectors[:,1,0]), 'dz (mm)': np.abs(vectors[:,1,2])}
vectDict = {'vectors': vectors, 'features': feats}

scanNum = [0]
doseNum = []
strNum = []
displayMode = '2d'
viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
        vwr.showNapari(scanNum, strNum, doseNum, vectDict, planC, displayMode)



# 07617 test
dcmDir =  r'\\vpensmph\deasylab1\Aditya\AI_workshop\test_data\0617_test\0617-292370_09-09-2000-35932'
structDir = r'\\vpensmph\deasylab1\Aditya\AI_workshop\cardiac_substruct_deeplab\test_struct'
planC = pc.loadDcmDir(dcmDir)
opts = {}
planC = pc.loadDcmDir(structDir, opts, planC)

import cerr.viewer as vwr
scanNumV = [0]
strNumV = range(17,30) #[3,4,7,8,9]
doseNumV = [0]
displayMode = '2d'
vectDict = {}
viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
        vwr.showNapari(scanNumV, strNumV, doseNumV, vectDict, planC, displayMode)


#Import label map to CERR
import glob
from cerr import plan_container as pc
import time
from cerr.dataclasses import structure as cerrStr
from cerr.contour import rasterseg as rs
#from cerr.utils import identifyScan, imageProc
from cerr.utils import mask
import numpy as np

# h5FileName = r'\\vpensmph\deasylab1\Aditya\AI_workshop\cardiac_substruct_deeplab\test_struct\test.h5'
# planC = pc.loadFromH5(h5FileName)
# maskFileName = r'\\vpensmph\deasylab1\Aditya\AI_workshop\cardiac_substruct_deeplab\test_struct\mask.mat'
# import scipy.io as sio
# maskDict = sio.loadmat(maskFileName)
# planC = pc.import_structure_mask(maskDict['scan0_peri'], 0, 'Pericardium_0', [], planC)
# planC = pc.import_structure_mask(maskDict['scan1_peri'], 1, 'Pericardium_1', [], planC)
# from cerr.dataclasses import structure as cerrStr
# planC = cerrStr.copyToScan(1, 0, planC)



#from cerr.utils.aiPipeline import identifyScan, imageProc
origStrNum = 25
mask3M = rs.getStrMask(origStrNum,planC)
print(mask3M.sum())
numComponents = 1
# Post-process segmentation to retain the largest connected component
procMask3M, _ = mask.largestConnComps(mask3M,numComponents)
strName = planC.structure[origStrNum].structureName #strToLabelMap[label+1]
planC = pc.importStructureMask(procMask3M, 0, strName, planC, origStrNum)
mask2 = rs.getStrMask(origStrNum,planC)
print(mask2.sum())


from cerr.dataclasses import scan as cerrScn
scanNum = [0]
doseNum = []
strNum = range(18,len(planC.structure),1)
#strNum = [1,3,5,7,9,11,13,15,17,19,21,23]
structNumV = np.arange(18,len(planC.structure),1)
indOrigV = np.array([cerrScn.getScanNumFromUID(planC.structure[structNum].assocScanUID, planC) for structNum in structNumV], dtype=int)
strNumV = structNumV[indOrigV == 0]

displayMode = '2d'
vectDict = {}
viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
        vwr.showNapari(scanNum, strNumV, doseNum, vectDict, planC, displayMode)

h5FileName = r'\\vpensmph\deasylab1\Aditya\AI_workshop\cardiac_substruct_deeplab\test_struct\test.h5'
scanNumV = [0]
structNumV = [] #[18,19,20,21,22,23,24] # np.arange(numOrigStructs, numStructs)
structNumExportV = range(len(planC.structure)) #structNumV[indOrigV == 1]
pc.saveToH5(planC,h5FileName,scanNumV, structNumExportV)


#from cerr.utils import identifyScan, imageProc

origStrNum = 25
mask3M = rs.getStrMask(origStrNum,planC)
numComponents = 1
# Post-process segmentation to retain the largest connected component
procMask3M, _ = mask.largestConnComps(mask3M,numComponents)
strName = planC.structure[origStrNum].structureName #strToLabelMap[label+1]
planC = pc.importStructureMask(procMask3M, 0, strName, planC, origStrNum)


# Directories for the batch visualization loop below (set before running)
zScoreSaveDir = r''  # input: dir containing wk0/wk1/... subfolders with Img.nii.gz + Seg.nii.gz
imageSaveDir = r''   # output: dir to write the screenshot PNGs

for wk in ['wk0']: #['wk0', 'wk1', 'wk2']:

    wkDir = os.path.join(zScoreSaveDir, wk)

    for d in os.scandir(wkDir):

        _, id = os.path.split(d)
        print("Data dir: " + id)

        imgFile = os.path.join(d, 'Img.nii.gz')
        segFile = os.path.join(d, 'Seg.nii.gz')
        planC = pc.loadNiiScan(imgFile, 'MR SCAN', 'LPS')
        planC = pc.loadNiiStructure(segFile, 0, planC, {1: 'GTV'})

        # Get index of central tumor slice
        scanNum = 0 # index of scan in planC.scan
        strNum = 0 # index of structure in planC.structure
        mask3M = rs.getStrMask(strNum, planC)
        rV, cV, sV = np.where(mask3M)
        midSliceInd = int(np.round(sV.mean()))
        midRowInd = int(np.round(rV.mean()))
        midColInd = int(np.round(cV.mean()))
        # update viewer to display the central slice and capture screenshot
        xV, yV, zV = planC.scan[scanNum].getScanXYZVals()
        scan3M = planC.scan[0].getScanArray()
        scanV = scan3M[mask3M]
        minVal = scanV.min()
        maxVal = scanV.max()

        scanNum = [0]
        doseNum = []
        strNum = 0
        displayMode = '2d'
        vectDict = {}
        viewer, scan_layer, dose_layer, struct_layer, dvf_layer = \
                vwr.showNapari(scanNum, strNum, doseNum, vectDict, planC, displayMode)

        scan_layer[0].opacity = 1
        scan_layer[0].contrast_limits_range = [minVal, maxVal]
        scan_layer[0].contrast_limits = [minVal, maxVal]
        scan_layer[0].gamma = 0.7

        # Transverse display
        viewer.update({'dims': {'order': (2,0,1)}})
        viewer.dims.set_point(2, zV[midSliceInd])
        #viewer.camera.zoom = 5
        #viewer.camera.center = (0, 100, 100)
        screenshotTrans = viewer.screenshot(size =(600, 600))

        # Coronal display
        viewer.update({'dims': {'order': (0,2,1)}})
        viewer.dims.set_point(0, -yV[midRowInd])
        screenshotCor = viewer.screenshot(size =(600, 600))

        # Sagittal display
        viewer.update({'dims': {'order': (1,2,0)}})
        viewer.dims.set_point(1, xV[midColInd])
        screenshotSag = viewer.screenshot(size =(600, 600))

        screenshot = np.concatenate((screenshotTrans,screenshotCor,screenshotSag),axis=1)

        viewer.close()
        fname = os.path.join(imageSaveDir,id+'.png')
        imsave(fname, screenshot)
        #display(Image(filename=fname, retina=False, width=200, embed=True))

