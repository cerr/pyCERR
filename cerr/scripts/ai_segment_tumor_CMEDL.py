from cerr import plan_container as pc
import cerr.contour.rasterseg as rs
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
    seg3M = resamp3M > 500


    # Store resulting segmentation in planC


    # Export RTSTRUCT DICOM




    # Show scan, structure and dose overlay in 3D
    from cerr import viewer as vwr
    import numpy as np
    scan_num = [0]
    dose_num = []
    num_structs = len(planC.structure)
    str_num_list = np.arange(num_structs)
    str_num_list = [num_structs]
    viewer, scan_layer, dose_layer, labels_layer = vwr.show_scan_struct_dose(scan_num, str_num_list, dose_num, planC)
