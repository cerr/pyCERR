"""Viewer module.

The "viewer" module defines routines for visualizing scan,
structure, dose and vector field.

"""

import typing
import warnings
import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from skimage import measure
import cerr.contour.rasterseg as rs
import cerr.dataclasses.scan as scn
import cerr.dataclasses.structure as cerrStr
import cerr.plan_container as pc
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets
from ipywidgets import interact, interactive_output
from IPython.display import display
from typing import Annotated, Literal
import importlib
if importlib.util.find_spec('napari') is not None:
    import napari
    from napari.layers import Labels, Image
    from napari.types import LayerDataTuple
    from qtpy.QtWidgets import QTabBar
    from magicgui import magicgui
    from magicgui.widgets import FunctionGui
    import vispy.color
    from napari.utils import DirectLabelColormap


warnings.filterwarnings("ignore", category=FutureWarning)

window_dict = {
        '--- Select ---': (0, 300),
        'Abd/Med': (-10, 330),
        'Head': (45, 125),
        'Liver': (80, 305),
        'Lung': (-500, 1500),
        'Spine': (30, 300),
        'Vrt/Bone': (400, 1500),
        'PET SUV': (5, 10)
                }

def initialize_image_window_widget() -> FunctionGui:
    @magicgui(CT_Window={"choices": window_dict.keys()}, call_button="Set")
    def image_window(image: Image, CT_Window='--- Select ---', Center="", Width="") -> LayerDataTuple:
        # do something with whatever layer the user has selected
        # note: it *may* be None! so your function should handle the null case
        if image is None:
            return
        ctr = float(Center)
        wdth = float(Width)
        minVal = ctr - wdth/2
        maxVal = ctr + wdth/2
        contrast_limits_range = [minVal, maxVal]
        contrast_limits = [minVal, maxVal]
        windowDict = {"name": CT_Window,
                      "center": ctr,
                      "width": wdth}
        metaDict = image.metadata
        if image.metadata['dataclass'] == 'scan':
             metaDict =  {'dataclass': metaDict['dataclass'],
                                   'planC': metaDict['planC'],
                                   'scanNum': metaDict['scanNum'],
                                   'window': windowDict}
        elif image.metadata['dataclass'] == 'dose':
            metaDict = {'dataclass': metaDict['dataclass'],
                                   'planC': metaDict['planC'],
                                   'doseNum': metaDict['doseNum'],
                                   'window': windowDict}
        else:
            return

        # scanDict = {'name': image.name,
        #              'contrast_limits_range': contrast_limits_range,
        #              'contrast_limits': contrast_limits,
        #               'metadata': metaDict }
        scanDict = {'name': image.name,
                    'metadata': metaDict }
        image.contrast_limits_range = contrast_limits_range
        image.contrast_limits = contrast_limits
        return (image.data, scanDict, "image")
    return image_window

def initialize_struct_save_widget() -> FunctionGui:
    @magicgui(label={'label': 'Select Structure', 'nullable': True}, call_button = 'Save')
    def struct_save(label: Labels, overwrite_existing_structure = True) -> LayerDataTuple:
        # do something with whatever layer the user has selected
        # note: it *may* be None! so your function should handle the null case
        if label is None:
            return
        planC = label.metadata['planC']
        structNum = label.metadata['structNum']
        assocScanNum = label.metadata['assocScanNum']
        structName = label.name
        mask3M = label.data
        if overwrite_existing_structure and structNum:
            origMask3M = rs.getStrMask(structNum, planC)
            siz = mask3M.shape
            slcToUpdateV = []
            for slc in range(siz[2]):
                if not np.all(mask3M[:,:,slc] == origMask3M[:,:,slc]):
                    slcToUpdateV.append(slc)
        strNames = [s.structureName for s in planC.structure]
        if not overwrite_existing_structure and structName in strNames:
            structName = structName + '_new'
            structNum = None
            # Set color of the layer
            colr = np.array(planC.structure[-1].structureColor) / 255
        else:
            #colr = label.color[1]
            colr = label.colormap.color_dict[1]
        planC = pc.importStructureMask(mask3M, assocScanNum, structName, planC, structNum)
        if structNum is None:
            # Assign the index of added structure
            structNum = len(planC.structure) - 1
        scanNum = label.metadata['assocScanNum']
        scan_affine = label.affine
        isocenter = cerrStr.calcIsocenter(structNum, planC)
        cmap = DirectLabelColormap(color_dict={None: None, int(1): colr, int(0): np.array([0,0,0,0])})
        labelDict = {'name': structName, 'affine': scan_affine,
                     'blending': 'translucent',
                     'opacity': 1,
                     'colormap': cmap,
                      'metadata': {'planC': planC,
                                   'structNum': structNum,
                                   'assocScanNum': scanNum,
                                   'isocenter': isocenter} }
        label.contour = 2
        return (mask3M, labelDict, "labels")
    return struct_save

def initialize_struct_add_widget() -> FunctionGui:
    @magicgui(image={'label': 'Pick a Scan'}, call_button='Create')
    def struct_add(image: Image, structure_name = "") -> Labels:
        # do something with whatever layer the user has selected
        # note: it *may* be None! so your function should handle the null case
        if image is None:
            return
        mask3M = np.zeros(image.data.shape, dtype=bool)
        scan_affine = image.affine.affine_matrix
        planC = image.metadata['planC']
        strNum = len(planC.structure)
        colr = np.array(cerrStr.getColorForStructNum(strNum)) / 255
        scanNum = image.metadata['scanNum']
        cmap = DirectLabelColormap(color_dict={None: None, int(1): colr, int(0): np.array([0,0,0,0])})
        shp = Labels(mask3M, name=structure_name, affine=scan_affine,
                                blending='translucent',
                                opacity = 1,
                                colormap = cmap,
                                metadata = {'planC': planC,
                                            'structNum': None,
                                            'assocScanNum': scanNum,
                                            'isocenter': [None, None, None]})
        shp.contour = 0
        return shp
    return struct_add


def checkerboard_indices(shape, tile_size, evenTiles=True):
    """
    Generates row and column indices for a checkerboard pattern.

    Args:
        shape (tuple): Shape of the 2D array (rows, cols).
        tile_size (int): Size of each square tile.

    Returns:
        tuple: Two lists, one for row indices and one for column indices.
               Each list contains tuples of slice objects representing the
               indices for the checkerboard pattern.
    """
    rows, cols = shape
    row_indices = []
    col_indices = []

    for i in range(rows // tile_size + 1):
        for j in range(cols // tile_size + 1):
            start_row = i * tile_size
            end_row = min(start_row + tile_size, rows)
            start_col = j * tile_size
            end_col = min(start_col + tile_size, cols)

            if (i + j) % 2 == 0:  # Even tiles
                row_indices.append(slice(start_row, end_row))
                col_indices.append(slice(start_col, end_col))

    return row_indices, col_indices


def getRCSwithinScanExtents(r, c, s,numRows, numCols, numSlcs,
                   offset, axNum):
    r = int(r)
    c = int(c)
    s = int(s)
    halfOff = int(offset / 2)
    halfOff = offset
    if axNum == 2:
        rMin = r - 2*offset
        rMax = r
        cMin = c - int(offset*1.5)
        cMax = c
        sMin = s
        sMax = s + 1
    elif axNum == 1:
        rMin = r - 2*offset
        rMax = r
        cMin = c
        cMax = c + 1
        sMin = s - int(offset*1.5)
        sMax = s
    elif axNum == 0:
        rMin = r
        rMax = r + 1
        cMin = c - 2*offset
        cMax = c
        sMin = s - int(offset*1.5)
        sMax = s
    if rMin < 0:
        rMin = 0
    if rMax < 0:
        rMax = 0
    if rMin >= numRows:
        rMin = numRows - 1
    if rMax >= numRows:
        rMax = numRows - 1
    if cMin < 0:
        cMin = 0
    if cMax < 0:
        cMax = 0
    if cMin >= numCols:
        cMin = numCols - 1
    if cMax >= numCols:
        cMax = numCols - 1
    if sMin < 0:
        sMin = 0
    if sMax < 0:
        sMax = 0
    if sMin >= numSlcs:
        sMin = numSlcs - 1
    if sMax >= numSlcs:
        sMax = numSlcs - 1

    return rMin, rMax, cMin, cMax, sMin, sMax

def updateMirror(viewer, baseLayer, movLayer, mrrScpLayerBase,
                 mrrScpLayerMov, mirrorLine, mirrorSize, displayType):
    #currPt = viewer.cursor.position
    currPt = mrrScpLayerBase.metadata['currentPos']
    #baseLyrInd = getLayerIndex(scanNum,'scan',scan_layer)
    #movLyrInd = getLayerIndex(scanNum,'scan',scan_layer)
    rb,cb,sb = np.round(baseLayer.world_to_data(currPt))
    numRows, numCols, numSlcs = baseLayer.data.shape #planC.scan[baseInd].getScanSize()
    rb = max(min(rb,numRows-1),1)
    cb = max(min(cb,numCols-1),1)
    sb = max(min(sb,numSlcs-1),1)

    planC = baseLayer.metadata['planC']
    baseScanNum = baseLayer.metadata['scanNum']
    movScanNum = movLayer.metadata['scanNum']

    # Convert mirror size from physical units to number of voxels
    baseSpacing = planC.scan[baseScanNum].getScanSpacing()
    movSpacing = planC.scan[movScanNum].getScanSpacing()

    axNum = viewer.dims.order[0]
    colsOffset = int(mirrorSize / (baseSpacing[0]*10))
    rowsOffset = int(mirrorSize / (baseSpacing[1]*10))
    slcsOffset = int(mirrorSize / (baseSpacing[2]*10))
    mirrorSizeBase = colsOffset
    rMinB, rMaxB, cMinB, cMaxB, sMinB, sMaxB = getRCSwithinScanExtents(rb,cb,sb,numRows, numCols, numSlcs,
                                                              mirrorSizeBase, axNum)
    rm,cm,sm = np.round(movLayer.world_to_data(currPt))
    numRows, numCols, numSlcs = movLayer.data.shape #planC.scan[movInd].getScanSize()
    rm = max(min(rm,numRows-1),1)
    cm = max(min(cm,numCols-1),1)
    sm = max(min(sm,numSlcs-1),1)

    axNum = viewer.dims.order[0]
    colsOffset = int(mirrorSize / (movSpacing[0]*10))
    rowsOffset = int(mirrorSize / (movSpacing[1]*10))
    slcsOffset = int(mirrorSize / (movSpacing[2]*10))
    mirrorSizeMov = colsOffset
    rMinM, rMaxM, cMinM, cMaxM, sMinM, sMaxM = getRCSwithinScanExtents(rm,cm,sm,numRows, numCols, numSlcs,
                                                              mirrorSizeMov, axNum)
    if rMinB == rMaxB == 0:
        rMaxB = 1
    elif rMinB == rMaxB:
        rMinB = rMinB - 1
    if cMinB == cMaxB == 0:
        cMaxB = 1
    elif cMinB == cMaxB:
        cMinB = cMinB-1

    if rMinM == rMaxM == 0:
        rMaxM = 1
    elif rMinM == rMaxM:
        rMinM = rMinM - 1
    if cMinM == cMaxM == 0:
        cMaxM = 1
    elif cMinM == cMaxM:
        cMinM = cMinM-1
    # planC = baseLayer.metadata['planC']
    # baseScanNum = baseLayer.metadata['scanNum']
    # movScanNum = movLayer.metadata['scanNum']
    xb,yb,zb = planC.scan[baseScanNum].getScanXYZVals()
    yb = -yb
    dxB,dyB,dzB = planC.scan[baseScanNum].getScanSpacing()
    xm,ym,zm = planC.scan[movScanNum].getScanXYZVals()
    ym = -ym
    dxM,dyM,dzM = planC.scan[movScanNum].getScanSpacing()

    deltaXmov = dxM * (cMaxM - cMinM)
    deltaXbase = dxB * (cMaxB - cMinB)
    deltaYmov = dyM * (rMaxM - rMinM)
    #croppedScanBase = baseLayer.data[rMinB:rMaxB,cMinB:cMaxB,int(sb)]
    #croppedScanMov = movLayer.data[rMinM:rMaxM,cMinM:cMaxM,int(sm)]
    if displayType == 'Mirrorscope':
        if viewer.dims.order[0] == 2:
            croppedScanBase = baseLayer.data[rMinB:rMaxB,cMinB:cMaxB,int(sb)]
            croppedScanMov = movLayer.data[rMinM:rMaxM,cMinM:cMaxM,int(sm)]
            croppedScanMov = np.flip(croppedScanMov,axis=1)
            mirrorAffineM = np.array([[dyM, 0, 0, ym[rMinM]], [0, dxM, 0, xm[cMinM]+deltaXmov], [0, 0, dzM, zm[int(sm)]], [0, 0, 0, 1]])
            mirrorAffineB = np.array([[dyB, 0, 0, yb[rMinB]], [0, dxB, 0, xb[cMinB]], [0, 0, dzB, zb[int(sb)]], [0, 0, 0, 1]])
        elif viewer.dims.order[0] == 1:
            croppedScanBase = baseLayer.data[rMinB:rMaxB,int(cb),sMinB:sMaxB]
            croppedScanMov = movLayer.data[rMinM:rMaxM,int(cm),sMinM:sMaxM]
            croppedScanMov = np.flip(croppedScanMov,axis=0)
            mirrorAffineM = np.array([[dyM, 0, 0, ym[rMinM]+deltaYmov], [0, dxM, 0, xm[int(cm)]], [0, 0, dzM, zm[sMinM]], [0, 0, 0, 1]])
            mirrorAffineB = np.array([[dyB, 0, 0, yb[rMinB]], [0, dxB, 0, xb[int(cb)]], [0, 0, dzB, zb[sMinB]], [0, 0, 0, 1]])
        else:
            croppedScanBase = baseLayer.data[int(rb),cMinB:cMaxB,sMinB:sMaxB]
            croppedScanMov = movLayer.data[int(rm),cMinM:cMaxM,sMinM:sMaxM]
            croppedScanMov = np.flip(croppedScanMov,axis=0)
            mirrorAffineM = np.array([[dyB, 0, 0, yb[int(rm)]], [0, dxM, 0, xm[cMinM]+deltaXmov], [0, 0, dzM, zm[sMinM]], [0, 0, 0, 1]])
            mirrorAffineB = np.array([[dyM, 0, 0, ym[int(rb)]], [0, dxB, 0, xb[cMinB]], [0, 0, dzB, zb[sMinB]], [0, 0, 0, 1]])
    elif displayType == 'Sidebyside':
        if viewer.dims.order[0] == 2:
            croppedScanBase = baseLayer.data[:,:int(cb),int(sb)]
            croppedScanMov = movLayer.data[:,int(cm):,int(sm)]
            mirrorAffineB = np.array([[dyB, 0, 0, yb[0]], [0, dxB, 0, xb[0]], [0, 0, dzB, zb[int(sb)]], [0, 0, 0, 1]])
            mirrorAffineM = np.array([[dyM, 0, 0, ym[0]], [0, dxM, 0, xm[int(cm)]], [0, 0, dzM, zm[int(sm)]], [0, 0, 0, 1]])
        elif viewer.dims.order[0] == 1:
            croppedScanBase = baseLayer.data[:int(rb),int(cb),:]
            croppedScanMov = movLayer.data[int(rm):,int(cm),:]
            mirrorAffineB = np.array([[dyB, 0, 0, yb[0]], [0, dxB, 0, xb[int(cb)]], [0, 0, dzB, zb[0]], [0, 0, 0, 1]])
            mirrorAffineM = np.array([[dyM, 0, 0, ym[int(rm)]], [0, dxM, 0, xm[int(cm)]], [0, 0, dzM, zm[0]], [0, 0, 0, 1]])
        else:
            croppedScanBase = baseLayer.data[int(rb),:int(cb),:]
            croppedScanMov = movLayer.data[int(rm),int(cm):,:]
            mirrorAffineB = np.array([[dyB, 0, 0, yb[int(rb)]], [0, dxB, 0, xb[0]], [0, 0, dzB, zb[0]], [0, 0, 0, 1]])
            mirrorAffineM = np.array([[dyM, 0, 0, ym[int(rm)]], [0, dxM, 0, xm[int(cm)]], [0, 0, dzM, zm[0]], [0, 0, 0, 1]])
    else: # displayType == 'AlternateGrid':
        if viewer.dims.order[0] == 2:
            baseShape = baseLayer.data[:,:,int(sb)].shape
            movShape = movLayer.data[:,:,int(sm)].shape
            croppedScanBase = np.ones(baseShape) * np.nan
            croppedScanMov = movLayer.data[:,:,int(sm)].copy()
            croppedScanBaseOrig = baseLayer.data[:,:,int(sb)].copy()
            mirrorAffineB = np.array([[dyB, 0, 0, yb[0]], [0, dxB, 0, xb[0]], [0, 0, dzB, zb[int(sb)]], [0, 0, 0, 1]])
            mirrorAffineM = np.array([[dyM, 0, 0, ym[0]], [0, dxM, 0, xm[0]], [0, 0, dzM, zm[int(sm)]], [0, 0, 0, 1]])
        elif viewer.dims.order[0] == 1:
            baseShape = baseLayer.data[:,int(cb),:].shape
            movShape = movLayer.data[:,int(cm),:].shape
            croppedScanBase = np.ones(baseShape) * np.nan
            croppedScanMov = movLayer.data[:,int(cm),:].copy()
            croppedScanBaseOrig = baseLayer.data[:,int(cb),:].copy()
            mirrorAffineB = np.array([[dyB, 0, 0, yb[0]], [0, dxB, 0, xb[int(cb)]], [0, 0, dzB, zb[0]], [0, 0, 0, 1]])
            mirrorAffineM = np.array([[dyM, 0, 0, ym[0]], [0, dxM, 0, xm[int(cm)]], [0, 0, dzM, zm[0]], [0, 0, 0, 1]])
        else:
            baseShape = baseLayer.data[int(rb),:,:].shape
            movShape = movLayer.data[int(rm),:,:].shape
            croppedScanBase = np.ones(baseShape) * np.nan
            croppedScanMov = movLayer.data[int(rm),:,:].copy()
            croppedScanBaseOrig = baseLayer.data[int(rb),:,:].copy()
            mirrorAffineB = np.array([[dyB, 0, 0, yb[int(rb)]], [0, dxB, 0, xb[0]], [0, 0, dzB, zb[0]], [0, 0, 0, 1]])
            mirrorAffineM = np.array([[dyM, 0, 0, ym[int(rm)]], [0, dxM, 0, xm[0]], [0, 0, dzM, zm[0]], [0, 0, 0, 1]])

        block_size = mirrorSize
        checkBaseRows, checkBaseCols = checkerboard_indices(baseShape, block_size)
        checkMovRows, checkMovCols = checkerboard_indices(movShape, block_size)
        for i in range(len(checkBaseRows)):
            croppedScanBase[checkBaseRows[i],checkBaseCols[i]] = croppedScanBaseOrig[checkBaseRows[i],checkBaseCols[i]]
        for i in range(len(checkMovRows)):
            croppedScanMov[checkBaseRows[i],checkBaseCols[i]] = np.nan
        #croppedScanMov[row_indices2, col_indices2] = np.nan
        #croppedScanBase = baseLayer.data[:,:,int(sb)]
        #croppedScanMov = movLayer.data[:,:,int(sm)]
        # mirrorAffineB = np.array([[dyB, 0, 0, yb[0]], [0, dxB, 0, xb[0]], [0, 0, dzB, zb[int(sb)]], [0, 0, 0, 1]])
        # mirrorAffineM = np.array([[dyM, 0, 0, ym[0]], [0, dxM, 0, xm[0]], [0, 0, dzM, zm[int(sm)]], [0, 0, 0, 1]])


    mrrScpLayerBase.affine.affine_matrix = mirrorAffineB
    mrrScpLayerMov.affine.affine_matrix = mirrorAffineM
    #cropNumRows, cropNumCols, cropNumSlcs = croppedScan.shape
    if viewer.dims.order[0] == 2:
        mrrScpLayerBase.data = croppedScanBase[:,:,None]
        mrrScpLayerMov.data = croppedScanMov[:,:,None]
    elif viewer.dims.order[0] == 0:
        mrrScpLayerBase.data = croppedScanBase[None,:,:]
        mrrScpLayerMov.data = croppedScanMov[None,:,:]
    else:
        mrrScpLayerBase.data = croppedScanBase[:,None,:]
        mrrScpLayerMov.data = croppedScanMov[:,None,:]

    #mrrScpLayerBase.refresh()
    mrrScpLayerBase.contrast_limits = baseLayer.contrast_limits
    mrrScpLayerBase.contrast_limits_range = baseLayer.contrast_limits_range
    mrrScpLayerBase.colormap = baseLayer.colormap
    mrrScpLayerMov.contrast_limits = movLayer.contrast_limits
    mrrScpLayerMov.contrast_limits_range = movLayer.contrast_limits_range
    mrrScpLayerMov.colormap = movLayer.colormap
    mrrScpLayerBase.refresh()
    mrrScpLayerMov.refresh()

    if displayType == 'Mirrorscope':
        if viewer.dims.order[0] == 2:
            data = [[[rMinB-(rMaxB-rMinB)*0.1,cb,sb], [rMinB,cb,sb]]]
            data.append([[rMaxB,cb,sb], [rMaxB+(rMaxB-rMinB)*0.1,cb,sb]])
            data.append([[rMinB,cMinB,sb], [rMinB,2*cMaxB-cMinB,sb]])
            data.append([[rMinB,cMinB,sb], [rMaxB,cMinB,sb]])
            data.append([[rMaxB,cMinB,sb], [rMaxB,2*cMaxB-cMinB,sb]])
            data.append([[rMinB,2*cMaxB-cMinB,sb], [rMaxB,2*cMaxB-cMinB,sb]])
        elif viewer.dims.order[0] == 0:
            data = [[[rb,cb,sMinB-(sMaxB-sMinB)*0.1], [rb,cb,sMinB]]]
            data.append([[rb,cb,sMaxB], [rb,cb,sMaxB+(sMaxB-sMinB)*0.1]])
            data.append([[rb,cMinB,sMinB], [rb,cMinB,sMaxB]])
            data.append([[rb,2*cMaxB-cMinB,sMinB], [rb,2*cMaxB-cMinB,sMaxB]])
            data.append([[rb,cMinB,sMinB], [rb,2*cMaxB-cMinB,sMinB]])
            data.append([[rb,cMinB,sMaxB], [rb,2*cMaxB-cMinB,sMaxB]])
        else:
            data = [[[rb,cb,sMinB-(sMaxB-sMinB)*0.1], [rb,cb,sMinB]]]
            data.append([[rb,cb,sMaxB], [rb,cb,sMaxB+(sMaxB-sMinB)*0.1]])
            data.append([[rMinB,cb,sMinB], [rMinB,cb,sMaxB]])
            data.append([[2*rMaxB-rMinB,cb,sMinB], [2*rMaxB-rMinB,cb,sMaxB]])
            data.append([[rMinB,cb,sMinB], [2*rMaxB-rMinB,cb,sMinB]])
            data.append([[rMinB,cb,sMaxB], [2*rMaxB-rMinB,cb,sMaxB]])
        mirrorLine.data = np.asarray(data)
        mirrorLine.refresh()
    elif displayType == 'Sidebyside':
        if viewer.dims.order[0] == 2:
            data = [[[0,cb,sb], [5,cb,sb]]]
            data.append([[numRows-5,cb,sb], [numRows,cb,sb]])
        elif viewer.dims.order[0] == 0:
            data = [[[rb,cb,-5], [rb,cb,0]]]
            data.append([[rb,cb,numSlcs], [rb,cb,numSlcs+5]])
        else:
            data = [[[rb,cb,-5], [rb,cb,0]]]
            data.append([[rb,cb,numSlcs], [rb,cb,numSlcs+5]])
        mirrorLine.data = np.asarray(data)
        mirrorLine.refresh()
    #elif displayType == 'AlternateGrid':
        #mirrorLine.data = np.asarray([])
        #mirrorLine.refresh()

    return

def mirror_scope_callback(layer, event):
    # on click
    #print('mouse clicked')
    # mrrScpLayerBase.visible = True
    # mrrScpLayerMov.visible = True
    # mirrorLine.visible = True
    dragged = False
    clicked = True
    if layer is None:
        return
    mirrorLine = layer.metadata['mirrorline']
    displayType = layer.metadata['displayType']
    mirrorSize = layer.metadata['mirrorSize']
    viewer = layer.metadata['viewer']
    baseLayer = layer.metadata['baseLayer']
    movLayer = layer.metadata['movLayer']
    mrrScpLayerBase = layer.metadata['mrrScpLayerBase']
    mrrScpLayerMov = layer.metadata['mrrScpLayerMov']
    #mrrScpLayerBase.visible = True
    #mrrScpLayerMov.visible = True
    #mirrorLine.visible = True
    #mrrScpLayerBase.mouse_pan = False
    #mrrScpLayerMov.mouse_pan = False
    #mrrScpLayerBase.interactive = True
    #mrrScpLayerMov.interactive = True
    # Get the center of grid
    # planC = baseLayer.metadata['planC']
    # scanNum = baseLayer.metadata['scanNum']
    # x,y,z = planC.scan[scanNum].getScanXYZVals()
    # y = -y
    currPt = viewer.cursor.position
    #if 'currentPos' not in mrrScpLayerBase.metadata:
    mrrScpLayerBase.metadata['currentPos'] = currPt #event.pos
    mrrScpLayerMov.metadata['currentPos'] = currPt #event.pos
    mrrScpLayerBase.metadata['currentAxis'] = viewer.dims.order[0] #event.pos
    mrrScpLayerMov.metadata['currentAxis'] = viewer.dims.order[0] #event.pos
    updateMirror(viewer, baseLayer, movLayer, mrrScpLayerBase, mrrScpLayerMov,
                 mirrorLine, mirrorSize, displayType)
    yield

    # on move
    while event.type == 'mouse_move':
        dragged = True
        mrrScpLayerBase.metadata['currentPos'] = viewer.cursor.position #event.pos
        mrrScpLayerMov.metadata['currentPos'] = viewer.cursor.position #event.pos
        updateMirror(viewer, baseLayer, movLayer, mrrScpLayerBase, mrrScpLayerMov,
                     mirrorLine, mirrorSize, displayType)
        yield

    # on release
    if dragged:
        pass

    return

def initialize_reg_qa_widget() -> FunctionGui:
    @magicgui(call_button=False, auto_call=True)
    def mirror_scope(viewer: 'napari.viewer.Viewer',
                     baseImage: Image,
                     movImage: Image,
                     display_type: Annotated[str, {'widget_type': "ComboBox", 'choices': ['--- OFF ---','Toggle','Mirrorscope','Sidebyside','AlternateGrid']}] = '--- OFF ---',
                     mirror_size: Annotated[float, {'widget_type': "Slider", 'min': 2, 'max': 100, 'visible': False}] = 40,
                     toggle_images: Annotated[float, {'widget_type': "Slider", 'min': 0, 'max': 100, 'visible': False}] = 50) -> typing.List[LayerDataTuple]:
        # do something with whatever layer the user has selected
        # note: it *may* be None! so your function should handle the null case
        #currPt = viewer.cursor.position
        if baseImage is None:
            return
        if movImage is None:
            return
        # Check whether mirror-scope exists
        layerNames = [lyr.name for lyr in viewer.layers]
        if display_type == '--- OFF ---' and 'Mirror-Scope-base' in layerNames:
            # # Delete mirror layers
            # baseMirrInd = layerNames.index('Mirror-Scope-base')
            # del viewer.layers[baseMirrInd]
            # layerNames = [lyr.name for lyr in viewer.layers]
            # movMirrInd = layerNames.index('Mirror-Scope-mov')
            # del viewer.layers[movMirrInd]
            # layerNames = [lyr.name for lyr in viewer.layers]
            # mirrLineInd = layerNames.index('Mirror-line')
            # del viewer.layers[mirrLineInd]
            # # Set scan layer as active
            # for lyr in viewer.layers:
            #     if lyr.visible and 'dataclass' in lyr.metadata and lyr.metadata['dataclass'] == 'scan':
            #         viewer.layers.selection.active = lyr
            #         break
            # return
            pass

        #mrrSiz = float(mirror_size)
        # Create mirror layers and shape layer
        #baseMrr, movMrr, mrrLines = mirrScp.initializeMirror()

        if 'Mirror-Scope-base' not in layerNames:
            baseInd = baseImage.metadata['scanNum']
            planC = baseImage.metadata['planC']
            xb,yb,zb = planC.scan[baseInd].getScanXYZVals()
            dxB,dyB,dzB = planC.scan[baseInd].getScanSpacing()
            yb = -yb
            mirror_affine = np.array([[dyB, 0, 0, yb[0]], [0, dxB, 0, xb[0]], [0, 0, dzB, zb[0]], [0, 0, 0, 1]])
            mrrScp = np.zeros((31,31,1))
            mrrScpLayerBase = viewer.add_image(mrrScp,name='Mirror-Scope-base',
                                        opacity=1, colormap=baseImage.colormap,
                                        affine=mirror_affine,
                                        blending="opaque",interpolation2d="linear",
                                        interpolation3d="linear",
                                        visible=False
                                        )
            mrrScpLayerMov = viewer.add_image(mrrScp,name='Mirror-Scope-mov',
                                        opacity=1, colormap=baseImage.colormap,
                                        affine=mirror_affine,
                                        blending="opaque",interpolation2d="linear",
                                        interpolation3d="linear",
                                        visible=False
                                        )
            mirrorLine = viewer.add_shapes([[0,0,0], [0,0,0]], name = 'Mirror-line',
                                           face_color = "red", edge_color = "red", edge_width = 0.5,
                                           opacity=1, blending="opaque",
                                           affine=baseImage.affine.affine_matrix,
                                           shape_type='line')
            mrrScpLayerBase.mouse_drag_callbacks.append(mirror_scope_callback)
            mrrScpLayerMov.mouse_drag_callbacks.append(mirror_scope_callback)

        lyrNames = [lyr.name for lyr in viewer.layers]
        mrrScpLayerBaseInd = lyrNames.index('Mirror-Scope-base')
        mrrScpLayerMovInd = lyrNames.index('Mirror-Scope-mov')
        mirrorLineInd = lyrNames.index('Mirror-line')
        mrrScpLayerBase = viewer.layers[mrrScpLayerBaseInd]
        mrrScpLayerMov = viewer.layers[mrrScpLayerMovInd]
        mirrorLine = viewer.layers[mirrorLineInd]
        mrrScpBase = mrrScpLayerBase.data
        mrrScpMov = mrrScpLayerMov.data

        if 'currentPos' in mrrScpLayerBase.metadata:
            currentPos = mrrScpLayerBase.metadata['currentPos']
        else:
            currentPos = [(rng[0] + rng[1])/2 for rng in viewer.dims.range]
            currentPos[2] = viewer.dims.point[2]

        mrrMeta = {'mirrorline': mirrorLine, 'mirrorSize': mirror_size,
                   'displayType': display_type, 'baseLayer': baseImage,
                   'movLayer': movImage, 'viewer': viewer,
                   'mrrScpLayerBase': mrrScpLayerBase,
                   'mrrScpLayerMov': mrrScpLayerMov,
                   'currentPos': currentPos,
                   'currentAxis': viewer.dims.order[0]}
        baseMirrorAffine = mrrScpLayerBase.affine.affine_matrix
        movMirrorAffine = mrrScpLayerMov.affine.affine_matrix

        baseMrrDict = {'name':'Mirror-Scope-base',
                        'opacity':1, 'colormap':baseImage.colormap,
                        'affine':baseMirrorAffine,
                        'blending':"opaque",'interpolation2d':"linear",
                        'interpolation3d':"linear",
                        'interactive': True,
                        'mouse_pan': False,
                        'mouse_zoom': True,
                        'visible': False,
                       'metadata': mrrMeta}
        movMrrDict = {'name':'Mirror-Scope-base',
                        'opacity':1, 'colormap':baseImage.colormap,
                        'affine':movMirrorAffine,
                        'blending':"opaque",'interpolation2d':"linear",
                        'interpolation3d':"linear",
                        'interactive': True,
                        'mouse_pan': False,
                        'mouse_zoom': True,
                        'visible': False,
                        'metadata': mrrMeta}
        mirrLinesDict = {'name': 'Mirror-line',
                         'face_color': "red", 'edge_color': "red", 'edge_width': 0.5,
                         'opacity':1, 'blending':"opaque",
                         'affine':baseImage.affine.affine_matrix,
                         'shape_type':'line'}
        mrrScpLayerBase.metadata = mrrMeta
        mrrScpLayerMov.metadata = mrrMeta
        viewer.layers.selection.active = mrrScpLayerBase
        return [(mrrScpBase, baseMrrDict, "image"),
                (mrrScpMov, movMrrDict, "image"),
                ([[0,0,0], [0,0,0]], mirrLinesDict, "shape"),
                (baseImage.data, {'name': baseImage.name, 'opacity': 1-toggle_images/100}, "image"),
                (movImage.data, {'name': movImage.name, 'opacity': toggle_images/100}, "image")]
    return mirror_scope


def initialize_dose_select_widget() -> FunctionGui:
    @magicgui(image={'label': 'Pick a Dose'}, call_button=False)
    def dose_select(image:Image) -> LayerDataTuple:
        # do something with whatever layer the user has selected
        # note: it *may* be None! so your function should handle the null case
        if image is None:
            return
        doseDict = {'name': image.name,
                    'contrast_limits_range': [],
                    'contrast_limits': [],
                    'metadata': {}}
        return (image.data, doseDict, "image")
    return dose_select

def initialize_dose_colorbar_widget() -> FunctionGui:
    with plt.style.context('dark_background'):
        mz_canvas = FigureCanvasQTAgg(Figure(figsize=(1, 10)))
    return mz_canvas

def initialize_dvf_colorbar_widget() -> FunctionGui:
    with plt.style.context('dark_background'):
        mz_canvas = FigureCanvasQTAgg(Figure(figsize=(1, 10)))
    return mz_canvas


def showNapari(planC, scan_nums=0, struct_nums=[], dose_nums=[], vectors_dict={}, displayMode = '2d'):
    """Routine to display images in the Napari viewer. This routine requires a display (physical or virtual).

    Args:
        planC (cerr.plan_container.PlanC): pyCERR's plan container object
        scan_nums (list or int): scan indices to display from planC.scan
        struct_nums (list or int): structure indices to display from planC.structure
        dose_nums (list or int): dose indices to display from planC.dose
        vectors_dict: A dictionary whose fields are "vectors" and "features".
            vectors must be an array of size nx2x3, where n is the number of vectors.
            The 1st element along the 2nd dimension contains (row,col,slc) representing the start co-ordinate
            The 2nd element along the 2nd dimension contains (yDeform,xDeform,zDeform) representing the lengths of
            vectors along y, x and z axis in CERR virtual coordinates.
            i.e. vectors[i,0,:] = [rStartV[i], cStartV[i], sStartV[i]]
                 vectors[i,1,:] = [yDeformV[i], xDeformV[i], zDeformV[i]]
        displayMode: '2d': contours are displayed by labels layer
                     '3d' contours are displayed by surface layer.

    Returns:
        napari.Viewer: Napari Viewer object
        List[napari.layers.Image]: List of scan layers corresponding to input scan_nums
        List[napari.layers.Labels]: List of structure layers corresponding to input struct_nums
        List[napari.layers.Image]: List of dose layers corresponding to input dose_nums
        List[napari.layers.Vectors]: List containing DVF layer corresponding to input vector_dict

    """

    if isinstance(scan_nums, (np.number, int, float)):
        scan_nums = [scan_nums]
    if isinstance(struct_nums, (np.number, int, float)):
        struct_nums = [struct_nums]
    if isinstance(dose_nums, (np.number, int, float)):
        dose_nums = [dose_nums]

    # Get Scan affines
    assocScanV = []
    for str_num in struct_nums:
        assocScanV.append(scn.getScanNumFromUID(planC.structure[str_num].assocScanUID, planC))
    allScanNums = [s for s in scan_nums]
    allScanNums.extend(assocScanV)
    allScanNums = np.unique(allScanNums)
    scanAffineDict = {}
    for scan_num in allScanNums:
        x,y,z = planC.scan[scan_num].getScanXYZVals()
        y = -y # negative since napari viewer y increases from top to bottom
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        #if tiled:
        #    if scan_num > 0 and np.mod(scan_num,2) == 0:
        #        x += x[-1] + 2
        scan_affine = np.array([[dy, 0, 0, y[0]], [0, dx, 0, x[0]], [0, 0, dz, z[0]], [0, 0, 0, 1]])
        scanAffineDict[scan_num] = scan_affine


    #from qtpy import QtWidgets, QtCore
    #QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    # above two lines are needed to allow to undock the widget with
    # additional viewers - for Multiview

    viewer = napari.Viewer(title='pyCERR')

    ## ======= Multiview - TBD  =====
    #from cerr import multiViewHelper
    #dock_widget = multiViewHelper.MultipleViewerWidget(viewer)
    #cross = multiViewHelper.CrossWidget(viewer)
    #viewer.window.add_dock_widget(dock_widget, name="pyCERR")
    #viewer.window.add_dock_widget(cross, name="Cross", area="left")


    scan_colormaps = ["gray","bop orange","bop purple", "cyan", "green", "blue"] * 5
    scan_layers = []
    for i, scan_num in enumerate(scan_nums):
        sa = planC.scan[scan_num].getScanArray()
        scan_affine = scanAffineDict[scan_num]
        opacity = 0.5
        scan_name = planC.scan[scan_num].scanInfo[0].imageType
        if scan_name == 'CT SCAN':
            center = 0
            width = 300
        else:
            minScan = np.percentile(sa, 5)
            scanNoBkgdV = sa[sa > minScan]
            center = np.median(scanNoBkgdV)
            lowerVal = np.percentile(scanNoBkgdV, 5)
            upperVal = np.percentile(scanNoBkgdV, 95)
            width = 2 * np.max([center - lowerVal, upperVal - center])
        scanWindow = {'name': "--- Select ---",
                      'center': center,
                      'width': width}
        scan_layers.append(viewer.add_image(sa,name=scan_name,affine=scan_affine,
                                           opacity=opacity, colormap=scan_colormaps[i],
                                            blending="additive",interpolation2d="linear",
                                            interpolation3d="linear",
                                            metadata = {'dataclass': 'scan',
                                                     'planC': planC,
                                                     'scanNum': scan_num,
                                                     'window': scanWindow},
                                            ))
        scan_layers[-1].contrast_limits = [center-width/2, center+width/2]
        scan_layers[-1].contrast_limits_range = [center-width/2, center+width/2]

    dose_layers = []
    for dose_num in dose_nums:
        doseArray = planC.dose[dose_num].doseArray
        dose_name = planC.dose[dose_num].fractionGroupID
        xd,yd,zd = planC.dose[dose_num].getDoseXYZVals()
        yd = -yd # negative since napari viewer y increases from top to bottom
        dx = xd[1] - xd[0]
        dy = yd[1] - yd[0]
        dz = zd[1] - zd[0]
        dose_affine = np.array([[dy, 0, 0, yd[0]], [0, dx, 0, xd[0]], [0, 0, dz, zd[0]], [0, 0, 0, 1]])
        minDose = doseArray.min()
        maxDose = doseArray.max()
        centerDose = (minDose + maxDose) / 2
        widthDose = (maxDose - minDose)
        doseWindow = {"name": "--- Select ---",
                      "center": centerDose,
                      "width": widthDose}
        assocScanNum = scn.getScanNumFromUID(planC.dose[dose_num].assocScanUID, planC)
        dose_lyr = viewer.add_image(doseArray,name=dose_name, affine=dose_affine,
                                  opacity=0.5,colormap="turbo",
                                  blending="additive",interpolation2d="linear",
                                  interpolation3d="linear",
                                  metadata = {'dataclass': 'dose',
                                           'planC': planC,
                                           'doseNum': dose_num,
                                           'assocScanNum': assocScanNum,
                                           'window': doseWindow}
                                   )
        dose_layers.append(dose_lyr)
        dose_layers[-1].contrast_limits = [centerDose-widthDose/2, centerDose+widthDose/2]
        dose_layers[-1].contrast_limits_range = [centerDose-widthDose/2, centerDose+widthDose/2]

    # reference: https://gist.github.com/AndiH/c957b4d769e628f506bd
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] * 4

    # Get x,y,z ranges for scaling results of marching cubes mesh
    # mins = np.array([min(y), min(x), min(z)])
    # maxes = np.array([max(y), max(x), max(z)])
    # ranges = maxes - mins
    struct_layer = []
    for i,str_num in enumerate(struct_nums):
        # Get scan affine
        scan_num = scn.getScanNumFromUID(planC.structure[str_num].assocScanUID, planC)
        scan_affine = scanAffineDict[scan_num]
        #colr = np.asarray(tableau20[i])/255
        colr = np.array(planC.structure[str_num].structureColor) / 255
        colr = np.append(colr,1)
        str_name = planC.structure[str_num].structureName

        if displayMode.lower() == '3d':
            #mins = np.array([min(y), min(x), min(z)])
            #maxes = np.array([max(y), max(x), max(z)])
            #ranges = maxes - mins
            mask3M = rs.getStrMask(str_num,planC)
            verts, faces, _, _ = measure.marching_cubes(volume=mask3M, level=0.5)
            #verts_scaled = verts * ranges / np.array(mask3M.shape) - mins
            cmap = vispy.color.Colormap([colr,colr])
            isocenter = cerrStr.calcIsocenter(str_num, planC)
            labl = viewer.add_surface((verts, faces),opacity=0.5,shading="flat",
                                              affine=scan_affine, name=str_name,
                                              colormap=cmap,
                                              metadata = {'dataclass': 'structure',
                                                'planC': planC,
                                                'structNum': str_num,
                                                'assocScanNum': scan_num,
                                                'isocenter': isocenter})
            struct_layer.append(labl)
        elif displayMode.lower() == '2d':
            cmap = DirectLabelColormap(color_dict={None: None, int(1): colr, int(0): np.array([0,0,0,0])})
            #polygons = getContourPolygons(str_num, scan_num, planC)
            # polygons =cerrStr.getContourPolygons(str_num, planC, rcsFlag=True)
            # shp = viewer.add_shapes(polygons, shape_type='polygon', edge_width=2,
            #                   edge_color=colr, face_color=[0]*4,
            #                   affine=scan_affine, name=str_name)

            # show as labels
            mask3M = rs.getStrMask(str_num,planC)
            isocenter = cerrStr.calcIsocenter(str_num, planC)
            mask3M[mask3M] = 1 #int(str_num + 1)
            # From napari 0.4.19 onwards
            # from napari.utils import DirectLabelColormap
            shp = viewer.add_labels(mask3M, name=str_name, affine=scan_affine,
                                    blending='translucent',
                                    colormap = cmap,
                                    opacity = 1,
                                    metadata = {'dataclass': 'structure',
                                                'planC': planC,
                                                'structNum': str_num,
                                                'assocScanNum': scan_num,
                                                'isocenter': isocenter})
            shp.contour = 2

            struct_layer.append(shp)

    dvf_layer = []
    if vectors_dict and 'vectors' in vectors_dict:
        vectors = vectors_dict['vectors'].copy()
        vectors[:,1,0] = -vectors[:,1,0]
        feats = vectors_dict['features']
        dvfScanNum = 0 # default
        if 'scanNum' in vectors_dict:
            dvfScanNum = vectors_dict['scanNum']
        scan_affine = scanAffineDict[dvfScanNum]# {'length': lengthV,  'dx': vectors[:,1,1], 'dy': vectors[:,1,0], 'dz': vectors[:,1,2]}
        vect_layr = viewer.add_vectors(vectors, edge_width=0.3, opacity=0.8,
                                       length=1, name="Deformation Vector Field",
                                       vector_style="arrow",
                                       ndim=3, features=feats,
                                       edge_colormap='husl',
                                       affine = scan_affine,
                                       metadata = {'dataclass': 'dvf',
                                                   'assocScanNum': dvfScanNum,
                                                    'planC': planC
                                                  }
                                       )
        dvf_layer.append(vect_layr)


    viewer.dims.ndisplay = 2
    if displayMode == '3d':
        viewer.dims.ndisplay = 3
    if len(scan_nums) > 0:
        scan_num = 0
        orientPos = ['L', 'P', 'S']
        orientNeg = ['R', 'A', 'I']
        flipDict = {}
        for i in range(len(orientPos)):
            flipDict[orientPos[i]] = orientNeg[i]
            flipDict[orientNeg[i]] = orientPos[i]
        ori = planC.scan[scan_num].getScanOrientation()
        labels_list = [flipDict[dir] + ' --> ' + dir for dir in ori]
        labels_list = [labels_list[1], labels_list[0], labels_list[2]]
        viewer.dims.axis_labels = labels_list
    viewer.dims.order = (2, 0, 1)
    #viewer.dims.displayed_order = (2,0,1)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "cm"
    #viewer.axes.visible = True
    #if len(struct_layer)> 0:
    image_window_widget = initialize_image_window_widget()
    struct_add_widget = initialize_struct_add_widget()
    struct_save_widget = initialize_struct_save_widget()
    dose_select_widget = initialize_dose_select_widget()
    dose_colorbar_widget = initialize_dose_colorbar_widget()
    dvf_colorbar_widget = initialize_dvf_colorbar_widget()
    reg_qa_widget = initialize_reg_qa_widget()


    def set_center_slice(label):
        # update viewer to display the central slice and capture screenshot
        strNum = label.metadata['structNum']
        scanNum = label.metadata['assocScanNum']
        isocenter = label.metadata['isocenter']
        viewer.layers.selection.active = label
        viewer.dims.set_point(2, isocenter[2])
        viewer.dims.set_point(1, isocenter[0])
        viewer.dims.set_point(0, - isocenter[1])
        return

    def image_changed(image):
        if image is None:
            return
        imgType = image.metadata['dataclass'] if 'dataclass' in image.metadata else ''
        if not imgType in ['scan', 'dose']:
            return
        #if 'structNum' in image.metadata:
        #    return
        if 'window' not in image.metadata:
            return

        #image = widgt[0].value
        # Set active layer to the one selected
        viewer.layers.selection.active = image
        #window_option = widgt.CT_Window.value
        #center = window_dict[window_option][0]
        #width = window_dict[window_option][1]
        #
        windows_name = image.metadata['window']['name']
        center = image.metadata['window']['center']
        width = image.metadata['window']['width']
        minVal = center - width/2
        maxVal = center + width/2
        rangeVal = [minVal, maxVal]
        image_window_widget.Center.value = center
        image_window_widget.Width.value = width
        image_window_widget.CT_Window.value = windows_name
        image.contrast_limits_range = rangeVal
        image.contrast_limits = rangeVal
        update_colorbar(image)
        return

    def window_changed(CT_Window):
        if CT_Window is None:
            return
        ctrWidth = window_dict[CT_Window]
        if CT_Window != '--- Select ---':
            image_window_widget.Center.value = ctrWidth[0]
            image_window_widget.Width.value = ctrWidth[1]
            image_window_widget.CT_Window.value = CT_Window
        return

    def center_width_changed(ctrWidth):
        if ctrWidth is None:
            return
        image_window_widget.CT_Window.value = '--- Select ---'
        return

    def label_changed(widgt):
        label = widgt[0].value
        if label is None:
            # Set active layer to scan
            viewer.layers.selection.active = scan_layers[0]
            return
        if viewer.dims.ndisplay == 2 and isinstance(label.metadata['isocenter'][0], (np.number, int, float)):
            set_center_slice(label)
        return

    def layer_active(event):
        if not hasattr(event, 'value'):
            return
        layer = event.value
        if not hasattr(layer, 'metadata'):
            return
        imgType = layer.metadata['dataclass'] if 'dataclass' in layer.metadata else ''
        if 'structNum' in layer.metadata and viewer.dims.ndisplay == 2:
            struct_save_widget[0].value = layer
            if isinstance(layer.metadata['isocenter'][0], (np.number, int, float)):
                set_center_slice(layer)
        elif imgType in ['scan', 'dose']:
            #update_colorbar(layer)
            image_changed(layer)


    def dose_changed(widgt):
        if widgt.image is None:
            return
        #mz_canvas = dose_colorbar_widget
        dose = widgt[0].value
        update_colorbar(dose)

    def cmap_changed(event):
        #print(event.value())
        update_colorbar(viewer.layers.selection.active)
        return

    def contrast_changed(event):
        image = event.source
        contrast_limits = image.contrast_limits
        center = (contrast_limits[0] + contrast_limits[1]) / 2
        width = contrast_limits[1] - contrast_limits[0]
        scanWindow = {'name': "--- Select ---",
                      'center': center,
                      'width': width}
        image.metadata['window'] = scanWindow
        image_window_widget.Center.value = scanWindow['center']
        image_window_widget.Width.value = scanWindow['width']
        image_window_widget.CT_Window.value = scanWindow['name']

        return

    def reset_mirror_position():
        lyrNames = [lyr.name for lyr in viewer.layers]
        if 'Mirror-Scope-base' in lyrNames:
            mrrScpLayerBaseInd = lyrNames.index('Mirror-Scope-base')
            mrrScpLayerMovInd = lyrNames.index('Mirror-Scope-mov')
            mrrScpLayerBase = viewer.layers[mrrScpLayerBaseInd]
            mrrScpLayerMov = viewer.layers[mrrScpLayerMovInd]
            currentAxis = mrrScpLayerBase.metadata['currentAxis']
            if currentAxis != viewer.dims.order[0]:
                currentPos = [(rng[0] + rng[1])/2 for rng in viewer.dims.range]
                currentAxis = viewer.dims.order[0]
            else:
                currentPos = list(mrrScpLayerBase.metadata['currentPos'])
                currentAxis = mrrScpLayerBase.metadata['currentAxis']
            currentPos[currentAxis] = viewer.dims.point[currentAxis]
            mrrScpLayerBase.metadata['currentAxis'] = currentAxis
            mrrScpLayerMov.metadata['currentAxis'] = currentAxis
            mrrScpLayerBase.metadata['currentPos'] = currentPos
            mrrScpLayerMov.metadata['currentPos'] = currentPos
            mirror_scope_changed(reg_qa_widget)
            return

    def dims_order_changed(event):
        dims = event.source
        if dims.order == (1,0,2): # Axial
            viewer.dims.order = (2,0,1)
        elif dims.order == (0,1,2):
            viewer.dims.order = (0,2,1)
        #mirror_scope_changed(reg_qa_widget)
        reset_mirror_position( )
        update_colorbar(viewer.layers.selection.active)
        return

    def dims_point_changed(event):
        # Update Mirrorscope
        reset_mirror_position()
        return


    def update_colorbar(image):

        if image is None:
            for lyr in viewer.layers:
                if lyr.metadata['dataclass'] in ['scan','structure']:
                    image = lyr
                    break
        if image is None:
            return

        # get Image units
        imgType = image.metadata['dataclass'] if 'dataclass' in image.metadata else ''
        units = ''
        if imgType == 'scan':
            planC = image.metadata['planC']
            scanNum = image.metadata['scanNum']
            units = planC.scan[scanNum].scanInfo[0].imageUnits
        elif imgType == 'dose':
            planC = image.metadata['planC']
            doseNum = image.metadata['doseNum']
            scanNum = image.metadata['assocScanNum']
            units = planC.dose[doseNum].doseUnits
        elif imgType == 'dvf':
            planC = image.metadata['planC']
            scanNum = image.metadata['assocScanNum']
            featureName = image._edge.color_properties.name
            units = featureName
        elif imgType == 'structure':
            planC = image.metadata['planC']
            scanNum = image.metadata['assocScanNum']
            featureName = ''
            units = ''
        else:
            return

        with plt.style.context('dark_background'):
            #mz_canvas = FigureCanvasQTAgg(Figure(figsize=(1, 0.1)))
            if imgType in  ['scan', 'dose']:
                minVal = image.contrast_limits_range[0]
                maxVal = image.contrast_limits_range[1]
                mz_canvas = dose_colorbar_widget
                norm = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
            elif imgType in ['dvf']:
                minVal = image.properties[featureName].min() #image.edge_contrast_limits[0]
                maxVal = image.properties[featureName].max() #image.edge_contrast_limits[1]
                mz_canvas = dvf_colorbar_widget
                norm = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
            else:
                mz_canvas = dose_colorbar_widget

            mz_axes = mz_canvas.figure.axes

            # Delete axes children
            for axNum in range(len(mz_axes)):
                #if imgType == 'structure' and axNum == 0:
                #    continue
                children = mz_axes[axNum].get_children()
                text_objects = [child for child in children if isinstance(child, plt.Text)]
                for text in text_objects:
                    text.set_text('')
                for child in children:
                    del child

            if len(mz_axes) == 0 and imgType in  ['scan', 'dose', 'structure']:
                mz_canvas.figure.add_axes([0.1, 0.3, 0.2, 0.4]) #mz_canvas.figure.subplots()
                mz_canvas.figure.add_axes([0.2, 0.1, 0.4, 0.1]) # orientation display axis
                mz_axes = mz_canvas.figure.axes
            elif len(mz_axes) == 0 and imgType in  ['dvf']: # imgType = 'dvf'
                mz_canvas.figure.add_axes([0.1, 0.3, 0.2, 0.4])
                mz_axes = mz_canvas.figure.axes

            if imgType in  ['scan', 'dose']:
                colors = ListedColormap(image.colormap.colors)
            elif imgType == 'dvf':
                colors = ListedColormap(image.edge_colormap.colors)
            if imgType in  ['scan', 'dose', 'dvf']:
                cb1 = mpl.colorbar.ColorbarBase(mz_axes[0], cmap=colors,
                            norm=norm,
                            orientation='vertical')
                mz_axes[0].get_xaxis().set_visible(False)
                cb1.set_label(units)

            # Draw arrows for patient direction
            orientPos = ['L', 'P', 'S']
            orientNeg = ['R', 'A', 'I']
            flipDict = {}
            for i in range(len(orientPos)):
                flipDict[orientPos[i]] = orientNeg[i]
                flipDict[orientNeg[i]] = orientPos[i]
            oriStr = planC.scan[scanNum].getScanOrientation()
            viewOrder = viewer.dims.order
            mz_axes[1].arrow(0.2, 0.2, 0.6, 0, width=0.02, head_width=0.2, head_length=0.2)
            mz_axes[1].arrow(0.2, 0.2, 0, 0.5, width=0.02, head_width=0.2, head_length=0.2)
            mz_axes[1].set_xlim(0,1)
            mz_axes[1].set_ylim(0,1)
            mz_axes[1].axis('off')
            if viewOrder[0] == 2:
                xOri =  oriStr[viewOrder[1]]
                yOri = flipDict[oriStr[viewOrder[2]]]
            elif viewOrder[0] == 1:
                xOri =  oriStr[viewOrder[0]]
                yOri = flipDict[oriStr[viewOrder[1]]]
            else:
                xOri =  oriStr[viewOrder[0]]
                yOri = flipDict[oriStr[viewOrder[1]]]
            mz_axes[1].text(0, 1, yOri, fontsize=12, color='cyan')
            mz_axes[1].text(1.1, 0.1, xOri, fontsize=12, color='cyan')

            mz_canvas.draw()
            mz_canvas.flush_events()
            #mz_axes.axis('image')
            #mz_axes.imshow(dose_layers[-1].colormap.colors)
            #mz_canvas.figure.tight_layout()
        return


    def mirror_scope_changed(widgt):
        viewer = widgt[0].value
        baseLayer = widgt[1].value
        movLayer = widgt[2].value
        displayType = widgt[3].value
        mirrorSize = widgt[4].value
        movOpacity = widgt[5].value/100
        baseOpacity = 1 - movOpacity
        baseLayer.opacity = baseOpacity
        movLayer.opacity = movOpacity
        layerNames = [lyr.name for lyr in viewer.layers]

        if 'Mirror-Scope-base' not in layerNames:
            return
        mrrBaseInd = layerNames.index('Mirror-Scope-base')
        mrrMovInd = layerNames.index('Mirror-Scope-mov')
        mrrLineInd = layerNames.index('Mirror-line')
        mrrScpLayerBase = viewer.layers[mrrBaseInd]
        mrrScpLayerMov = viewer.layers[mrrMovInd]
        mirrorLine = viewer.layers[mrrLineInd]
        mrrScpLayerBase.metadata['mirrorSize'] = mirrorSize
        mrrScpLayerMov.metadata['mirrorSize'] = mirrorSize
        #baseLayer.refresh()
        #movLayer.refresh()

        #reg_qa_widget.call_button.text = displayType
        if displayType == '--- OFF ---':
            widgt[4].visible = False
            widgt[5].visible = False
            # Delete mirror layers
            if isinstance(mrrLineInd, (int, np.integer)):
                del viewer.layers[mrrLineInd]
            if isinstance(mrrMovInd, (int, np.integer)):
                del viewer.layers[mrrMovInd]
            if isinstance(mrrBaseInd, (int, np.integer)):
                del viewer.layers[mrrBaseInd]

            # Set scan layer as active
            viewer.layers.selection.active = baseLayer

            return

        if displayType == 'Mirrorscope':
            widgt[4].visible = True
            widgt[5].visible = False
            mrrScpLayerBase.visible = True
            mrrScpLayerMov.visible = True
            mirrorLine.visible = True
            mrrScpLayerBase.mouse_pan = False
            mrrScpLayerMov.mouse_pan = False
        elif displayType == 'Sidebyside':
            widgt[4].visible = False
            widgt[5].visible = False
            mrrScpLayerBase.visible = True
            mrrScpLayerMov.visible = True
            mirrorLine.visible = True
            mrrScpLayerBase.mouse_pan = False
            mrrScpLayerMov.mouse_pan = False
        elif displayType == 'Toggle':
            widgt[4].visible = False
            widgt[5].visible = True
            mrrScpLayerBase.visible = False
            mrrScpLayerMov.visible = False
            mirrorLine.visible = False
            mrrScpLayerBase.mouse_pan = True
            mrrScpLayerMov.mouse_pan = True
        elif displayType == 'AlternateGrid':
            widgt[4].visible = True
            widgt[5].visible = False
            mrrScpLayerBase.visible = True
            mrrScpLayerMov.visible = True
            mirrorLine.visible = False
            mrrScpLayerBase.mouse_pan = True
            mrrScpLayerMov.mouse_pan = True

        if 'currentPos' in mrrScpLayerBase.metadata:
            updateMirror(viewer, baseLayer, movLayer, mrrScpLayerBase,
                         mrrScpLayerMov, mirrorLine, mirrorSize, displayType)

        return

    # Change slice to center of that structure
    struct_save_widget.changed.connect(label_changed)
    image_window_widget.image.changed.connect(image_changed)
    image_window_widget.CT_Window.changed.connect(window_changed)
    image_window_widget.Center.changed.connect(center_width_changed)
    image_window_widget.Width.changed.connect(center_width_changed)
    scanWidget = viewer.window.add_dock_widget([image_window_widget], area='left', name="Window", tabify=True)
    structWidget = viewer.window.add_dock_widget([struct_add_widget, struct_save_widget], area='left', name="Segmentation", tabify=True)
    colorbars_dock = viewer.window.add_dock_widget([dose_colorbar_widget], area='right', name="Image Colorbar", tabify=True)
    dvf_dock = viewer.window.add_dock_widget([dvf_colorbar_widget], area='right', name="DVF Colorbar", tabify=True)
    reg_qa_dock = viewer.window.add_dock_widget(reg_qa_widget, area='left', name="Reg QA", tabify=True)
    reg_qa_widget.changed.connect(mirror_scope_changed)
    #colorbars_dock.resize(5, 20)

    # This line sets the index of the active DockWidget
    scanWidget.parent().findChildren(QTabBar)[0].setCurrentIndex(0)

    viewer.layers.events.inserted.connect(struct_add_widget.reset_choices)
    viewer.layers.events.inserted.connect(struct_save_widget.reset_choices)
    viewer.layers.events.removed.connect(struct_add_widget.reset_choices)
    viewer.layers.events.removed.connect(struct_save_widget.reset_choices)
    viewer.layers.selection.events.active.connect(layer_active)
    viewer.layers.selection.events.changed.connect(layer_active)
    viewer.layers.events.changed.connect(layer_active)
    viewer.dims.events.order.connect(dims_order_changed)
    viewer.dims.events.point.connect(dims_point_changed)


    for dose_lyr in dose_layers:
        #dose_lyr.events.contrast_limits_range.connect(layer_active)
        dose_lyr.events.colormap.connect(cmap_changed)
        dose_lyr.events.contrast_limits_range.connect(cmap_changed)
        dose_lyr.events.contrast_limits.connect(contrast_changed)
    for scan_lyr in scan_layers:
        #scan_lyr.events.contrast_limits_range.connect(layer_active)
        scan_lyr.events.colormap.connect(cmap_changed)
        scan_lyr.events.contrast_limits_range.connect(cmap_changed)
        scan_lyr.events.contrast_limits.connect(contrast_changed)
    for vect_layr in dvf_layer:
        vect_layr.events.edge_color.connect(cmap_changed)

    #dose_select_widget.changed.connect(dose_changed)

    viewer.layers.selection.active = scan_layers[0]
    update_colorbar(scan_layers[0])
    if len(dvf_layer) > 0:
        update_colorbar(dvf_layer[0])

    viewer.show(block=False)
    #napari.run()

    # Set Image colorbar active
    colorbars_dock.parent().findChildren(QTabBar)[2].setCurrentIndex(0)

    return viewer, scan_layers, struct_layer, dose_layers, dvf_layer


def showMplNb(planC, scanNum=0, structNums=[], doseNum=None,
              windowPreset=None, windowCenter=0, windowWidth=300,
              doseColorMap=plt.cm.jet, doseCenter=None, doseWidth=None):
    """Routine to display interactive plot using matplotlib in a jupyter notebook

    Args:
        planC (cerr.plan_container.PlanC):
        scanNum (int): scan index to display from planC.scan
        structNums (list or int): structure indices to display from planC.structure
        doseNum (int): dose index to display from planC.dose (default:None)
        windowPreset (str): optional, string representing preset window.
            'Abd/Med': (-10, 330),
            'Head': (45, 125),
            'Liver': (80, 305),
            'Lung': (-500, 1500),
            'Spine': (30, 300),
            'Vrt/Bone': (400, 1500),
            'PET SUV': (5, 10)
        windowCenter (float): optional, defaults to 0 when windowPreset is not specified.
        windowWidth (float): optional, defaults to 300 when windowPreset is not specified.
        doseColorMap (cmap): Dose colormap. Default: plt.cm.jet

    Returns:
        None
    """

    windowPresetList = [w for w in list(window_dict.keys())]
    lowerWindowPresetList = [w.lower() for w in windowPresetList]

    # Find whether the preset window exists
    if windowPreset is not None and windowPreset.lower() in lowerWindowPresetList:
        presetIndex = lowerWindowPresetList.index(windowPreset.lower())
        windowCenter = window_dict[windowPresetList[presetIndex]][0]
        windowWidth = window_dict[windowPresetList[presetIndex]][1]

    def windowImage(image, windowCenter, windowWidth):
        imgMin = windowCenter - windowWidth // 2
        imgMax = windowCenter + windowWidth // 2
        windowedImage = image.copy()
        windowedImage[windowedImage < imgMin] = imgMin
        windowedImage[windowedImage > imgMax] = imgMax
        return windowedImage

    def rotateImage(img):
        return(list(zip(*img)))

    # def updateView(change):
    #     # outputViewSelect = widgets.Output()
    #     # with outputViewSelect:
    #     #     showSlice(change['new'], 10)
    #     showSlice(change['new'], 10)
    #
    # def updateSliceAxial(change):
    #     # outputSlcAxial = widgets.Output()
    #     # with outputSlcAxial:
    #     #     showSlice('axial', change['new'])
    #     showSlice('axial', change['new'])
    #
    # def updateSliceSagittal(change):
    #     outputSlcSagittal = widgets.Output()
    #     with outputSlcSagittal:
    #         showSlice(change['new'], 'sagittal')
    #
    # def updateSliceCoronal(change):
    #     outputSlcCoronal = widgets.Output()
    #     with outputSlcCoronal:
    #         showSlice(change['new'], 'coronal')

    def createWidgets(imgSize, scanNum, doseNum=None):

        viewSelect = widgets.Dropdown(
            options=['Axial', 'Sagittal', 'Coronal'],
            value='Axial',
            description='view',
            disabled=False
        )

        if doseNum is not None:
            doseVisFlag = True
        else:
            doseVisFlag = False

        doseAlphaSlider = widgets.FloatSlider(
            min=0,max=1,value=0.5,
            step=.02, description="doseAlpha",
            visible= doseVisFlag)

        sliceSliderAxial = widgets.IntSlider(
            min=1,max=imgSize[2],value=int(imgSize[2]/2),
            step=1, description="slcNum")

        sliders = widgets.HBox([viewSelect,sliceSliderAxial, doseAlphaSlider])

        sliceAlphaList = []
        if not isinstance(scanNum, (np.number, int, float)):
            for scanNum in scanNum:
                sliceAlphaList.append(widgets.FloatSlider(
                min=0,max=1,value=0.5,
                step=.02, description=f"scan_{scanNum}_Alpha",
                visible= doseVisFlag))
            sliders = widgets.VBox([sliders,widgets.HBox(sliceAlphaList)])

        # viewSelect.observe(updateView, names='value')
        # sliceSliderAxial.observe(updateSliceAxial, names='value')


        #outputSlcAxial = widgets.Output()


        # sliceSliderSagittal = widgets.IntSlider(min=1,max=imgSize[1],value=int(imgSize[1]/2),
        #                                         step=1, description="Sagittal")
        # outputSlcSagittal = widgets.Output()
        #
        # sliceSliderCoronal = widgets.IntSlider(min=1,max=imgSize[0],value=int(imgSize[0]/2),
        #                                        step=1, description="Coronal")
        # outputSlcCoronal = widgets.Output()
        #
        # sliceSliderSagittal.observe(updateSliceSagittal, names='value')
        # sliceSliderCoronal.observe(updateSliceCoronal, names='value')
        #
        # return sliceSliderAxial, sliceSliderSagittal, sliceSliderCoronal

        return sliceSliderAxial, viewSelect, doseAlphaSlider, sliders


    # Extract scan and mask
    scan3M = planC.scan[scanNum].getScanArray()
    xVals, yVals, zVals = planC.scan[scanNum].getScanXYZVals()
    extentTrans = xVals[0], xVals[-1], yVals[-1], yVals[0]
    extentSag = yVals[0], yVals[-1], zVals[-1], zVals[0]
    extentCor = xVals[0], xVals[-1], zVals[-1], zVals[0]
    imgSiz = np.shape(scan3M)

    if isinstance(doseNum,(np.number,int,float)):
        dose3M = planC.dose[doseNum].doseArray
        if doseCenter is not None and doseWidth is not None:
            minDose = doseCenter - doseWidth // 2
            maxDose = doseCenter + doseWidth // 2
        else:
            maxDose = dose3M.max()
            minDose = dose3M.min()
        xDoseVals, yDoseVals, zDoseVals = planC.dose[doseNum].getDoseXYZVals()
        extentDoseTrans = xDoseVals[0], xDoseVals[-1], yDoseVals[-1], yDoseVals[0]
        extentDoseSag = yDoseVals[0], yDoseVals[-1], zDoseVals[-1], zDoseVals[0]
        extentDoseCor = xDoseVals[0], xDoseVals[-1], zDoseVals[-1], zDoseVals[0]
    else:
        doseNum = None
        maxDose = None
        minDose = None

    masks = list()
    strNameList = list()
    strColorList = list()
    for nStr in range (len(structNums)):
        mask3M = rs.getStrMask(structNums[nStr],planC)
        masks.append(mask3M)
        strNameList.append(planC.structure[structNums[nStr]].structureName)
        strColorList.append(np.array(planC.structure[structNums[nStr]].structureColor)/255)

    # Create slider widgets
    imgSize = np.shape(scan3M)
    #sliceSliderAxial, sliceSliderSagittal, sliceSliderCoronal = createWidgets(imgSize)
    sliceSliderAxial, viewSelect, doseAlphaSlider, sliders = createWidgets(imgSize, scanNum, doseNum)

    def update_numSlcs(*args):
        if viewSelect.value == 'Axial':
            numSlcs = imgSize[2]
        elif viewSelect.value == 'Sagittal':
            numSlcs = imgSize[1]
        else:
            numSlcs = imgSize[0]
        sliceSliderAxial.max = numSlcs
        sliceSliderAxial.value = int(numSlcs / 2)

    viewSelect.observe(update_numSlcs, 'value')

    def showSlice(view, slcNum, doseAlpha):

        clear_output(wait=True)
        #print(view + ' view slice ' + str(slcNum))

        ax = showSlice.ax
        if ax is None:
            fig, ax = plt.subplots(1,1)
            # # #ax_legend.set_visible(False)

        # colors = ['Oranges', 'Blues', 'Greens','Purples',
        #           'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
        #           'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
        # cmaps = [plt.colormaps[color].copy() for color in colors] * 10
        #
        # cm = plt.colormaps['tab20'].copy()
        # colors = cm.colors * 5

        doseImage = None
        if view.lower() == 'axial':
            windowedImage = windowImage(scan3M[: ,: ,slcNum - 1], windowCenter, windowWidth)
            extent = extentTrans
            if doseNum is not None and (zDoseVals[0] <= zVals[slcNum-1] <= zDoseVals[-1]):
                doseSlcNum = np.argmin((zVals[slcNum-1] - zDoseVals)**2)
                doseImage = dose3M[:,:,doseSlcNum]
                extentDose = extentDoseTrans
        elif view.lower() == 'sagittal':
            windowedImage = rotateImage(windowImage(scan3M[:, slcNum - 1, :], windowCenter, windowWidth))
            extent = extentSag
            if doseNum is not None and (xDoseVals[0] <= xVals[slcNum-1] <= xDoseVals[-1]):
                doseSlcNum = np.argmin((xVals[slcNum-1] - xDoseVals)**2)
                doseImage = rotateImage(dose3M[:,doseSlcNum,:])
                extentDose = extentDoseSag
        elif view.lower() == 'coronal':
            windowedImage = rotateImage(windowImage(scan3M[slcNum - 1, :, :], windowCenter, windowWidth))
            extent = extentCor
            if doseNum is not None and (yDoseVals[-1] <= yVals[slcNum-1] <= yDoseVals[0]):
                doseSlcNum = np.argmin((yVals[slcNum-1] - yDoseVals)**2)
                doseImage = rotateImage(dose3M[doseSlcNum,:,:])
                extentDose = extentDoseCor
        else:
            raise ValueError('Invalid view type: ' + view)

        # Display scan
        im1 = ax.imshow(windowedImage, cmap=plt.cm.gray, alpha=1,
                    interpolation='nearest', extent=extent)
        if doseNum is not None and doseImage is not None:
            imDose = ax.imshow(doseImage, cmap=doseColorMap, alpha=doseAlpha,
                        interpolation='nearest', extent=extentDose,
                        vmin=minDose, vmax=maxDose)

        #Display mask
        numLabel = len(masks)
        if view.lower() == 'axial':
            for maskNum in range(0,numLabel,1):
                #maskCmap = cmaps[maskNum]
                #maskCmap.set_under('k', alpha=0)
                mask3M = masks[maskNum]
                #col = colors[maskNum]
                col = strColorList[maskNum]
                if mask3M.any():
                    im2 = ax.contour(np.flip(np.squeeze(mask3M[:,:,slcNum-1]), axis=0),
                            levels = [0.5], colors = [col],
                            extent=extent, linewidths = 2)
                    # im2 = ax.imshow(mask3M[:,:,slcNum-1],
                    #             cmap=maskCmap, alpha=1, extent=extent,
                    #             interpolation='none', clim=[0.5, 1])

        elif view.lower() == 'sagittal':
            for maskNum in range(0,numLabel,1):
                #maskCmap = cmaps[maskNum]
                #maskCmap.set_under('k', alpha=0)
                mask3M = masks[maskNum]
                #col = colors[maskNum]
                col = strColorList[maskNum]
                if mask3M.any():
                    im2 = ax.contour(np.flip(rotateImage(mask3M[:,slcNum-1,:]),axis=0),
                            levels = [0.5], colors = [col],
                            extent=extent, linewidths = 2)
                    # im2 = ax.imshow(rotateImage(mask3M[:, slcNum - 1, :]),
                    #             cmap=maskCmap, alpha=.8, extent=extent,
                    #             interpolation='none', clim=[0.5, 1])

        elif view.lower() == 'coronal':
            for maskNum in range(0,numLabel,1):
                #maskCmap = cmaps[maskNum]
                #maskCmap.set_under('k', alpha=0)
                mask3M = masks[maskNum]
                #col = colors[maskNum]
                col = strColorList[maskNum]
                if mask3M.any():
                    im2 = ax.contour(np.flip(rotateImage(mask3M[slcNum-1,:,:]), axis=0),
                            levels = [0.5], colors = [col],
                            extent=extent, linewidths = 2)
                    # im2 = ax.imshow(rotateImage(mask3M[slcNum - 1, :, :]),
                    #             cmap=maskCmap, alpha=.8, extent=extent,
                    #             interpolation='none', clim=[0.5, 1])

        proxy = [plt.Rectangle((0,0),1,1,fc = col) for col in strColorList]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.legend(strNameList, fontsize=12)
        if doseImage is not None:
            cbar = plt.colorbar(imDose, location='left', shrink=0.6)
            doseName = planC.dose[doseNum].fractionGroupID
            cbar.set_label(doseName)
        plt.rcParams["figure.figsize"] = (6, 6)
        plt.legend(proxy, strNameList, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    showSlice.ax = None

    if doseAlphaSlider == None:
        out = interactive_output(showSlice, {'view':viewSelect, 'slcNum':sliceSliderAxial})
    else:
        out = interactive_output(showSlice, {'view':viewSelect, 'slcNum':sliceSliderAxial, 'doseAlpha':doseAlphaSlider})

    display(sliders, out)

    return
