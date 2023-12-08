from cerr import plan_container as pc
import cerr.contour.rasterseg as rs
import cerr.dataclasses.structure as strct
from cerr.utils.bbox import compute_boundingbox
from cerr.radiomics.preprocess import imgResample3D, getResampledGrid

if __name__ == "__main__":

    # Patienit from RTOG-0617 dataset
    dcm_dir = r"L:\Data\RTOG0617\DICOM\NSCLC-Cetuximab_flat_dirs\0617-381906_09-09-2000-87022"
    lungNames = ['Lung_IPSI','Lung_Contra','Lung_total']

    # Import DICOM metadata to planC
    planC = pc.load_dcm_dir(dcm_dir)

    # Find Lung segmentation to extract ROI for inference
    strNames = [s.structureName.lower() for s in planC.structure]
    minr = float('inf')
    maxr = -1
    minc = float('inf')
    maxc = -1
    mins = float('inf')
    maxs = -1
    for lung in lungNames:
        if lung.lower() in strNames:
            structNum = strNames.index(lung.lower())
            mask3M = rs.getStrMask(structNum, planC)
            strMinr, strMaxr, strMinc, strMaxc, strMins, strMaxs, __ = compute_boundingbox(mask3M)
            minr = min(minr, strMinr)
            maxr = max(maxr, strMaxr)
            minc = min(minc, strMinc)
            maxc = max(maxc, strMaxc)
            mins = min(mins, strMins)
            maxs = max(maxs, strMaxs)

    xV, yV, zV = planC.scan[0].getScanXYZVals()
    xMin = xV[minc]
    xMax = xV[maxc]
    yMin = yV[minr]
    yMax = yV[maxr]
    zMin = zV[mins]
    zMax = zV[maxs]

    # Crop grid and scan
    xCroppedV = xV[minc:maxc+1]
    yCroppedV = yV[minr:maxr+1]
    zCroppedV = zV[mins:maxs+1]
    ctScan3M = planC.scan[0].getScanArray()
    croppedCtScan3M = ctScan3M[minc:maxc+1,minr:maxr+1,mins:maxs+1]

    # Resize

    # Resample
    gridResampleMethod = 'center'
    newResV = [0.1, 0.1, 0.1]
    originV = [xV[0], yV[0], zV[-1]]
    # Fixed output resolution (dx, dy, dz)
    [xResampleV, yResampleV, zResampleV] = getResampledGrid(newResV, xV, yV, zV, gridResampleMethod)
    # Fixed output size
    #xResampleV = np.arange(xmin,xmax,numPts)
    interpMethod = 'sitkLinear'
    resamp3M = imgResample3D(croppedCtScan3M, xCroppedV, yCroppedV, zCroppedV,\
                            xResampleV, yResampleV, zResampleV, interpMethod)


    # Call inference for the cropped scan
    #segLung3M = inferLungSeg(ctScan3M, xV, yV, zV)
    #segTumor3M = inferTumor(ctScan3M, segLung3M, xV, yV, zV)

    # seg3M = resamp3M > 500

    # Resample back to original scan resolution
    seg3M = ctScan3M > 1000
    assocScanNum = 0
    structName = 'test'
    # planC = strct.import_mask(seg3M, assocScanNum, structName, planC)


    # CMEDL, MRRN Lung tumor segmentation
    # Inputs - ctScan3M numpy array, x,y,z grid
    # Output - tumor segmentation numpy array

    # CBCT pipeline
    # Inputs - planning CT nii, CBCT nii, planning CT segmentations .nii
    # Output - nii segmentation on CBCT


    # Store resulting segmentation in planC


    # Export RTSTRUCT DICOM




    # Show scan, structure and dose overlay in 3D
    from cerr import viewer as vwr
    import numpy as np
    scanNum = [0]
    doseNum = [0]
    numStructs = len(planC.structure)
    strNumList = np.arange(numStructs)
    strNumList = [numStructs-1]
    strNumList = 7
    displayMode = '2d' # 'path' or 'surface'
    viewer, scan_layer, dose_layer, struct_layer = \
        vwr.show_scan_struct_dose(scanNum, strNumList, doseNum, planC, displayMode)

