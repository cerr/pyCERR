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
            colr = label.color[1]
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


def getRCSwithinScanExtents(r, c, s,numRows, numCols, numSlcs,
                   offset, axNum):
    r = int(r)
    c = int(c)
    s = int(s)
    halfOff = int(offset / 2)
    halfOff = offset
    if axNum == 1:
        rMin = r - 2*offset
        rMax = r
        cMin = c
        cMax = c + int(offset*1.5)
        sMin = s
        sMax = s + 1
    elif axNum == 2:
        pass
    elif axNum == 0:
        pass
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

def updateMirror(viewer, baseLayer, movLayer, mrrScpLayerBase, mrrScpLayerMov, mirrorLine, mirrorSize):
    #currPt = viewer.cursor.position
    currPt = mrrScpLayerBase.metadata['currentPos']
    #baseLyrInd = getLayerIndex(scanNum,'scan',scan_layer)
    #movLyrInd = getLayerIndex(scanNum,'scan',scan_layer)
    rb,cb,sb = np.round(baseLayer.world_to_data(currPt))
    numRows, numCols, numSlcs = baseLayer.data.shape #planC.scan[baseInd].getScanSize()
    axNum = 1
    rMinB, rMaxB, cMinB, cMaxB, sMinB, sMaxB = getRCSwithinScanExtents(rb,cb,sb,numRows, numCols, numSlcs,
                                                              mirrorSize, axNum)
    rm,cm,sm = np.round(movLayer.world_to_data(currPt))
    numRows, numCols, numSlcs = movLayer.data.shape #planC.scan[movInd].getScanSize()
    axNum = 1
    rMinM, rMaxM, cMinM, cMaxM, sMinM, sMaxM = getRCSwithinScanExtents(rm,cm,sm,numRows, numCols, numSlcs,
                                                              mirrorSize, axNum)
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
    planC = baseLayer.metadata['planC']
    baseScanNum = baseLayer.metadata['scanNum']
    movScanNum = movLayer.metadata['scanNum']
    xb,yb,zb = planC.scan[baseScanNum].getScanXYZVals()
    yb = -yb
    dxB,dyB,dzB = planC.scan[baseScanNum].getScanSpacing()
    xm,ym,zm = planC.scan[movScanNum].getScanXYZVals()
    ym = -ym
    dxM,dyM,dzM = planC.scan[movScanNum].getScanSpacing()

    deltaXmov = dxM * (cMaxM - cMinM)
    deltaYmov = dyM * (rMaxM - rMinM)
    croppedScanBase = baseLayer.data[rMinB:rMaxB,cMinB:cMaxB,int(sb)]
    croppedScanMov = movLayer.data[rMinM:rMaxM,cMinM:cMaxM,int(sm)]
    croppedScanMov = np.flip(croppedScanMov,axis=1)
    if viewer.dims.order[0] == 2:
        mirrorAffineB = np.array([[dyB, 0, 0, yb[rMinB]], [0, dxB, 0, xb[cMinB]], [0, 0, dzB, zb[int(sb)]], [0, 0, 0, 1]])
        mirrorAffineM = np.array([[dyM, 0, 0, ym[rMinM]], [0, dxM, 0, xm[cMinM]-deltaXmov], [0, 0, dzM, zm[int(sm)]], [0, 0, 0, 1]])
    elif viewer.dims.order[0] == 1:
        mirrorAffineB = np.array([[dyB, 0, 0, yb[int(rb)]], [0, dxB, 0, xb[cMinB]], [0, 0, dzB, zb[sMinB]], [0, 0, 0, 1]])
        mirrorAffineM = np.array([[dyM, 0, 0, ym[int(rm)]], [0, dxM, 0, xm[cMinM]], [0, 0, dzM, zm[sMinM]], [0, 0, 0, 1]])
    else:
        mirrorAffineB = np.array([[dyB, 0, 0, yb[rMinB]], [0, dxB, 0, xb[int(cb)]], [0, 0, dzB, zb[sMinB]], [0, 0, 0, 1]])
        mirrorAffineM = np.array([[dyM, 0, 0, ym[rMinM]], [0, dxM, 0, xm[int(cm)]], [0, 0, dzM, zm[sMinM]], [0, 0, 0, 1]])

    mrrScpLayerBase.affine.affine_matrix = mirrorAffineB
    mrrScpLayerMov.affine.affine_matrix = mirrorAffineM
    #cropNumRows, cropNumCols, cropNumSlcs = croppedScan.shape
    mrrScpLayerBase.data = croppedScanBase[:,:,None]
    mrrScpLayerMov.data = croppedScanMov[:,:,None]
    mrrScpLayerBase.refresh()
    mrrScpLayerBase.contrast_limits = baseLayer.contrast_limits
    mrrScpLayerBase.contrast_limits_range = baseLayer.contrast_limits_range
    mrrScpLayerMov.contrast_limits = movLayer.contrast_limits
    mrrScpLayerMov.contrast_limits_range = movLayer.contrast_limits_range

    data = [[[rMinB,cb,sb], [rMaxB,cb,sb]]]
    data.append([[rMinB,2*cMinB-cMaxB,sb], [rMinB,cMaxB,sb]])
    data.append([[rMinB,2*cMinB-cMaxB,sb], [rMaxB,2*cMinB-cMaxB,sb]])
    data.append([[rMaxB,2*cMinB-cMaxB,sb], [rMaxB,cMaxB,sb]])
    data.append([[rMinB,cMaxB,sb], [rMaxB,cMaxB,sb]])
    mirrorLine.data = np.asarray(data)
    mirrorLine.refresh()
    return

def mirror_scope_callback(layer, event):
    # on click
    #print('mouse clicked')
    # mrrScpLayerBase.visible = True
    # mrrScpLayerMov.visible = True
    # mirrorLine.visible = True
    dragged = False
    clicked = True
    mirrorLine = layer.metadata['mirrorline']
    mirrorSize = layer.metadata['mirrorSize']
    viewer = layer.metadata['viewer']
    baseLayer = layer.metadata['baseLayer']
    movLayer = layer.metadata['movLayer']
    mrrScpLayerBase = layer.metadata['mrrScpLayerBase']
    mrrScpLayerMov = layer.metadata['mrrScpLayerMov']
    mrrScpLayerBase.visible = True
    mrrScpLayerMov.visible = True
    mirrorLine.visible = True
    # Get the center of grid
    # planC = baseLayer.metadata['planC']
    # scanNum = baseLayer.metadata['scanNum']
    # x,y,z = planC.scan[scanNum].getScanXYZVals()
    # y = -y
    currPt = viewer.cursor.position
    if 'currentPos' not in mrrScpLayerBase.metadata:
        mrrScpLayerBase.metadata['currentPos'] = currPt #event.pos
        mrrScpLayerMov.metadata['currentPos'] = currPt #event.pos
    updateMirror(viewer, baseLayer, movLayer, mrrScpLayerBase, mrrScpLayerMov, mirrorLine, mirrorSize)
    yield

    # on move
    while event.type == 'mouse_move':
        dragged = True
        mrrScpLayerBase.metadata['currentPos'] = viewer.cursor.position #event.pos
        mrrScpLayerMov.metadata['currentPos'] = viewer.cursor.position #event.pos
        updateMirror(viewer, baseLayer, movLayer, mrrScpLayerBase, mrrScpLayerMov, mirrorLine, mirrorSize)
        #updateMirror(viewer, baseLayer, movLayer, mrrScpLayerBase, mrrScpLayerMov, mirrorLine, mirrorSize)
        yield

    # on release
    if dragged:
        pass

def initialize_reg_qa_widget() -> FunctionGui:
    @magicgui(call_button="Mirror-Scope")
    def mirror_scope(viewer: 'napari.viewer.Viewer',
                     baseImage: Image,
                     movImage: Image,
                     mirror_size: Annotated[float, {'widget_type': "Slider", 'min': 0, 'max': 100}] = 10,
                     toggle_images: Annotated[float, {'widget_type': "Slider", 'min': 0, 'max': 100}] = 50) -> typing.List[LayerDataTuple]:
        # do something with whatever layer the user has selected
        # note: it *may* be None! so your function should handle the null case
        #currPt = viewer.cursor.position
        if baseImage is None:
            return
        if movImage is None:
            return
        # Check whether mirror-scope exists
        layerNames = [lyr.name for lyr in viewer.layers]
        if 'Mirror-Scope-base' in layerNames:
            return
        #mrrSiz = float(mirror_size)
        # Create mirror layers and shape layer
        #baseMrr, movMrr, mrrLines = mirrScp.initializeMirror()
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
                                    interpolation3d="linear"
                                    )
        mrrScpLayerMov = viewer.add_image(mrrScp,name='Mirror-Scope-mov',
                                    opacity=1, colormap=baseImage.colormap,
                                    affine=mirror_affine,
                                    blending="opaque",interpolation2d="linear",
                                    interpolation3d="linear"
                                    )
        mirrorLine = viewer.add_shapes([[0,0,0], [0,0,0]], name = 'Mirror-line',
                                       face_color = "red", edge_color = "red", edge_width = 0.5,
                                       opacity=1, blending="opaque",
                                       affine=baseImage.affine.affine_matrix,
                                       shape_type='line')
        mrrMeta = {'mirrorline': mirrorLine, 'mirrorSize': mirror_size,
                   'layerType': 'base', 'baseLayer': baseImage,
                   'movLayer': movImage, 'viewer': viewer,
                   'mrrScpLayerBase': mrrScpLayerBase,
                   'mrrScpLayerMov': mrrScpLayerMov}

        baseMrrDict = {'name':'Mirror-Scope-base',
                        'opacity':1, 'colormap':baseImage.colormap,
                        'affine':mirror_affine,
                        'blending':"opaque",'interpolation2d':"linear",
                        'interpolation3d':"linear",
                        'interactive': True,
                        'mouse_pan': False,
                        'visible': False,
                       'metadata': mrrMeta}
        movMrrDict = {'name':'Mirror-Scope-base',
                        'opacity':1, 'colormap':baseImage.colormap,
                        'affine':mirror_affine,
                        'blending':"opaque",'interpolation2d':"linear",
                        'interpolation3d':"linear",
                        'interactive': True,
                        'mouse_pan': False,
                        'visible': False,
                        'metadata': mrrMeta}
        mirrLinesDict = {'name': 'Mirror-line',
                         'face_color': "red", 'edge_color': "red", 'edge_width': 0.5,
                         'opacity':1, 'blending':"opaque",
                         'affine':baseImage.affine.affine_matrix,
                         'shape_type':'line'}
        viewer.layers.selection.active = mrrScpLayerBase
        mrrScpLayerBase.mouse_drag_callbacks.append(mirror_scope_callback)
        mrrScpLayerMov.mouse_drag_callbacks.append(mirror_scope_callback)
        return [(mrrScp, baseMrrDict, "image"),
                (mrrScp, movMrrDict, "image"),
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

    viewer = napari.Viewer(title='pyCERR')

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
        dose_lyr = viewer.add_image(doseArray,name='dose',affine=dose_affine,
                                  opacity=0.5,colormap="turbo",
                                  blending="additive",interpolation2d="linear",
                                  interpolation3d="linear",
                                  metadata = {'dataclass': 'dose',
                                           'planC': planC,
                                           'doseNum': dose_num,
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
            labl = viewer.add_surface((verts, faces),opacity=0.5,shading="flat",
                                              affine=scan_affine, name=str_name,
                                              colormap=cmap)
            struct_layer.append(labl)
        elif displayMode.lower() == '2d':
            # polygons = getContourPolygons(str_num, scan_num, planC)
            # polygons =cerrStr.getContourPolygons(str_num, planC, rcsFlag=True)
            #
            # shp = viewer.add_shapes(polygons, shape_type='polygon', edge_width=2,
            #                   edge_color=colr, face_color=[0]*4,
            #                   affine=scan_affine, name=str_name)
            mask3M = rs.getStrMask(str_num,planC)
            isocenter = cerrStr.calcIsocenter(str_num, planC)
            mask3M[mask3M] = 1 #int(str_num + 1)
            # From napari 0.4.19 onwards
            # from napari.utils import DirectLabelColormap
            cmap = DirectLabelColormap(color_dict={None: None, int(1): colr, int(0): np.array([0,0,0,0])})
            shp = viewer.add_labels(mask3M, name=str_name, affine=scan_affine,
                                    blending='translucent',
                                    colormap = cmap,
                                    opacity = 1,
                                    metadata = {'planC': planC,
                                                'structNum': str_num,
                                                'assocScanNum': scan_num,
                                                'isocenter': isocenter})
            # shp.colormap = cmap
            shp.contour = 2
            struct_layer.append(shp)

    dvf_layer = []
    if vectors_dict and 'vectors' in vectors_dict:
        vectors = vectors_dict['vectors'].copy()
        vectors[:,1,0] = -vectors[:,1,0]
        feats = vectors_dict['features']
        scan_affine = scanAffineDict[0]# {'length': lengthV,  'dx': vectors[:,1,1], 'dy': vectors[:,1,0], 'dz': vectors[:,1,2]}
        vect_layr = viewer.add_vectors(vectors, edge_width=0.3, opacity=0.8,
                                       length=1, name="Deformation Vector Field",
                                       vector_style="arrow",
                                       ndim=3, features=feats,
                                       edge_colormap='husl',
                                       affine = scan_affine,
                                       metadata = {'dataclass': 'dvf',
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

    def update_colorbar(image):

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
            units = planC.dose[doseNum].doseUnits
        elif imgType == 'dvf':
            #planC = image.metadata['planC']
            featureName = image._edge.color_properties.name
            units = featureName
        else:
            return

        with plt.style.context('dark_background'):
            #mz_canvas = FigureCanvasQTAgg(Figure(figsize=(1, 0.1)))
            if imgType in  ['scan', 'dose']:
                minVal = image.contrast_limits_range[0]
                maxVal = image.contrast_limits_range[1]
                mz_canvas = dose_colorbar_widget
            else:
                minVal = image.properties[featureName].min() #image.edge_contrast_limits[0]
                maxVal = image.properties[featureName].max() #image.edge_contrast_limits[1]
                mz_canvas = dvf_colorbar_widget

            norm = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
            mz_axes = mz_canvas.figure.axes
            if len(mz_axes) == 0:
                mz_canvas.figure.add_axes([0.1, 0.3, 0.2, 0.4]) #mz_canvas.figure.subplots()
                mz_axes = mz_canvas.figure.axes
            for ax in mz_axes:
                colorbar_plt = ax.get_children()
                for chld in colorbar_plt:
                    del chld

            if imgType in  ['scan', 'dose']:
                colors = ListedColormap(image.colormap.colors)
            else:
                colors = ListedColormap(image.edge_colormap.colors)

            cb1 = mpl.colorbar.ColorbarBase(mz_axes[0], cmap=colors,
                            norm=norm,
                            orientation='vertical')

            cb1.set_label(units)
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
        mirrorSize = widgt[3].value
        movOpacity = widgt[4].value/100
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
        if 'currentPos' in mrrScpLayerBase.metadata:
            updateMirror(viewer, baseLayer, movLayer, mrrScpLayerBase, mrrScpLayerMov, mirrorLine, mirrorSize)
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

    napari.run()

    # Set Image colorbar active
    dvf_dock.parent().findChildren(QTabBar)[2].setCurrentIndex(0)

    return viewer, scan_layers, struct_layer, dose_layers, dvf_layer


def showMplNb(planC, scan_nums=0, struct_nums=[], dose_nums=None, windowPreset=None, windowCenter=0, windowWidth=300):
    """Routine to display interactive plot using matplotlib in a jupyter notebook

    Args:
        planC (cerr.plan_container.PlanC):
        scan_nums (list or int): scan indices to display from planC.scan
        struct_nums (list or int): structure indices to display from planC.structure
        dose_nums (list or int): dose indices to display from planC.dose
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

    def createWidgets(imgSize, scanNumV, doseNum=None):

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
        if not isinstance(scanNumV, (np.number, int, float)):
            for scanNum in scanNumV:
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
    scan3M = planC.scan[scan_nums].getScanArray()
    xVals, yVals, zVals = planC.scan[scan_nums].getScanXYZVals()
    extentTrans = xVals[0], xVals[-1], yVals[-1], yVals[0]
    extentSag = yVals[0], yVals[-1], zVals[-1], zVals[0]
    extentCor = xVals[0], xVals[-1], zVals[-1], zVals[0]
    imgSiz = np.shape(scan3M)

    if isinstance(dose_nums,(np.number,int,float)):
        dose3M = planC.dose[dose_nums].doseArray
        maxDose = dose3M.max()
        minDose = dose3M.min()
        xDoseVals, yDoseVals, zDoseVals = planC.dose[dose_nums].getDoseXYZVals()
        extentDoseTrans = xDoseVals[0], xDoseVals[-1], yDoseVals[-1], yDoseVals[0]
        extentDoseSag = yDoseVals[0], yDoseVals[-1], zDoseVals[-1], zDoseVals[0]
        extentDoseCor = xDoseVals[0], xDoseVals[-1], zDoseVals[-1], zDoseVals[0]
    else:
        dose_nums = None

    masks = list()
    strNameList = list()
    strColorList = list()
    for nStr in range (len(struct_nums)):
        mask3M = rs.getStrMask(struct_nums[nStr],planC)
        masks.append(mask3M)
        strNameList.append(planC.structure[struct_nums[nStr]].structureName)
        strColorList.append(np.array(planC.structure[struct_nums[nStr]].structureColor)/255)

    # Create slider widgets
    imgSize = np.shape(scan3M)
    #sliceSliderAxial, sliceSliderSagittal, sliceSliderCoronal = createWidgets(imgSize)
    sliceSliderAxial, viewSelect, doseAlphaSlider, sliders = createWidgets(imgSize, scan_nums, dose_nums)

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
            if dose_nums is not None and (zDoseVals[0] <= zVals[slcNum-1] <= zDoseVals[-1]):
                doseSlcNum = np.argmin((zVals[slcNum-1] - zDoseVals)**2)
                doseImage = dose3M[:,:,doseSlcNum]
                extentDose = extentDoseTrans
        elif view.lower() == 'sagittal':
            windowedImage = rotateImage(windowImage(scan3M[:, slcNum - 1, :], windowCenter, windowWidth))
            extent = extentSag
            if dose_nums is not None and (xDoseVals[0] <= xVals[slcNum-1] <= xDoseVals[-1]):
                doseSlcNum = np.argmin((xVals[slcNum-1] - xDoseVals)**2)
                doseImage = rotateImage(dose3M[:,doseSlcNum,:])
                extentDose = extentDoseSag
        elif view.lower() == 'coronal':
            windowedImage = rotateImage(windowImage(scan3M[slcNum - 1, :, :], windowCenter, windowWidth))
            extent = extentCor
            if dose_nums is not None and (yDoseVals[-1] <= yVals[slcNum-1] <= yDoseVals[0]):
                doseSlcNum = np.argmin((yVals[slcNum-1] - yDoseVals)**2)
                doseImage = rotateImage(dose3M[doseSlcNum,:,:])
                extentDose = extentDoseCor
        else:
            raise ValueError('Invalid view type: ' + view)

        # Display scan
        im1 = ax.imshow(windowedImage, cmap=plt.cm.gray, alpha=1,
                    interpolation='nearest', extent=extent)
        if dose_nums is not None and doseImage is not None:
            imDose = ax.imshow(doseImage, cmap=plt.cm.jet, alpha=doseAlpha,
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
            plt.colorbar(imDose, location='left', shrink=0.6)
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
