import matplotlib.pyplot as plt
import cerr.contour.rasterseg as rs
import napari
import numpy as np
from skimage import measure
import vispy.color
import cerr.dataclasses.scan as scn
import cerr.dataclasses.structure as cerrStr
from napari.types import LabelsData, ImageData, LayerDataTuple
from napari.layers import Labels, Image
from magicgui import magicgui
import cerr.plan_container as pc
from magicgui.widgets import FunctionGui, Select
from qtpy.QtWidgets import QTabBar
from enum import Enum
from magicgui import magic_factory

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

def initialize_scan_window_widget() -> FunctionGui:
    @magicgui(CT_Window={"choices": window_dict.keys()}, call_button=False)
    def scan_window(Scan: Image, CT_Window='--- Select ---', Center="", Width="") -> LayerDataTuple:
        # do something with whatever layer the user has selected
        # note: it *may* be None! so your function should handle the null case
        if Scan is None:
            return
        ctr = float(Center)
        wdth = float(Width)
        minVal = ctr - wdth/2
        maxVal = ctr + wdth/2
        contrast_limits_range = [minVal, maxVal]
        contrast_limits = [minVal, maxVal]
        scanDict = {'name': Scan.name,
                     'contrast_limits_range': contrast_limits_range,
                     'contrast_limits': contrast_limits,
                      'metadata': {'planC': Scan.metadata['planC'],
                                   'scanNum': Scan.metadata['scanNum']} }
        return (Scan.data, scanDict, "image")
    return scan_window

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
        planC = pc.import_structure_mask(mask3M, assocScanNum, structName, structNum, planC)
        if not structNum:
            structNum = len(planC.structure) - 1
        scanNum = label.metadata['assocScanNum']
        scan_affine = label.affine
        colr = label.color[1]
        isocenter = cerrStr.calcIsocenter(structNum, planC)
        labelDict = {'name': structName, 'affine': scan_affine,
                     'num_colors': 1, 'blending': 'translucent',
                     'contour': 2, 'opacity': 1,
                     'color': {1: colr, 0: np.array([0,0,0,0])},
                      'metadata': {'planC': planC,
                                   'structNum': structNum,
                                   'assocScanNum': scanNum,
                                   'isocenter': isocenter} }
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
        shp = Labels(mask3M, name=structure_name, affine=scan_affine,
                                num_colors=1, blending='translucent',
                                opacity = 1,
                                color = {1: colr, 0: np.array([0,0,0,0])},
                                metadata = {'planC': planC,
                                            'structNum': None,
                                            'assocScanNum': scanNum,
                                            'isocenter': [None, None, None]})
        shp.contour = 0
        return shp
    return struct_add


def getContourPolygons(strNum, assocScanNum, planC):
    numSlcs = len(planC.structure[strNum].contour)
    polygons = []
    for slc in range(numSlcs):
        if planC.structure[strNum].contour[slc]:
            for seg in planC.structure[strNum].contour[slc].segments:
                rowV, colV = rs.xytom(seg.points[:,0], seg.points[:,1],slc,planC, assocScanNum)
                pts = np.array((rowV, colV, slc*np.ones_like(rowV)), dtype=np.float64).T
                polygons.append(pts)
    return polygons


def show_scan_struct_dose(scan_nums, str_nums, dose_nums, planC, displayMode = '2d'):

    if not isinstance(scan_nums, list):
        scan_nums = [scan_nums]
    if not isinstance(str_nums, list):
        str_nums = [str_nums]
    if not isinstance(dose_nums, list):
        dose_nums = [dose_nums]

    # Get Scan affines
    assocScanV = []
    for str_num in str_nums:
        assocScanV.append(scn.getScanNumFromUID(planC.structure[str_num].assocScanUID, planC))
    allScanNums = scan_nums.copy()
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

    viewer = napari.Viewer()

    scan_colormaps = ["gray","bop orange","bop purple", "cyan", "green", "blue"] * 5
    scan_layers = []
    for i, scan_num in enumerate(scan_nums):
        sa = planC.scan[scan_num].getScanArray()
        scan_affine = scanAffineDict[scan_num]
        opacity = 0.5
        scan_name = planC.scan[scan_num].scanInfo[0].imageType
        scan_layers.append(viewer.add_image(sa,name=scan_name,affine=scan_affine,
                                           opacity=opacity, colormap=scan_colormaps[i],
                                            blending="additive",interpolation2d="lanczos",
                                            interpolation3d="lanczos",
                                            metadata = {'planC': planC,
                                                     'scanNum': scan_num}
                                            ))

    dose_layers = []
    for dose_num in dose_nums:
        doseArray = planC.dose[dose_num].doseArray
        xd,yd,zd = planC.dose[dose_num].getDoseXYZVals()
        yd = -yd # negative since napari viewer y increases from top to bottom
        dx = xd[1] - xd[0]
        dy = yd[1] - yd[0]
        dz = zd[1] - zd[0]
        dose_affine = np.array([[dy, 0, 0, yd[0]], [0, dx, 0, xd[0]], [0, 0, dz, zd[0]], [0, 0, 0, 1]])
        dose_layers.append(viewer.add_image(doseArray,name='dose',affine=dose_affine,
                                  opacity=0.5,colormap="gist_earth",
                                  blending="additive",interpolation2d="lanczos",
                                  interpolation3d="lanczos"
                                   ))

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
    for i,str_num in enumerate(str_nums):
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
            #
            # shp = viewer.add_shapes(polygons, shape_type='polygon', edge_width=2,
            #                   edge_color=colr, face_color=[0]*4,
            #                   affine=scan_affine, name=str_name)
            mask3M = rs.getStrMask(str_num,planC)
            isocenter = cerrStr.calcIsocenter(str_num, planC)
            mask3M[mask3M] = 1 #int(str_num + 1)
            shp = viewer.add_labels(mask3M, name=str_name, affine=scan_affine,
                                    num_colors=1, blending='translucent',
                                    color = {1: colr, 0: np.array([0,0,0,0])},
                                    opacity = 1, metadata = {'planC': planC,
                                                               'structNum': str_num,
                                                               'assocScanNum': scan_num,
                                                               'isocenter': isocenter})
            shp.contour = 2
            struct_layer.append(shp)

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
    scan_window_widget = initialize_scan_window_widget()
    struct_add_widget = initialize_struct_add_widget()
    struct_save_widget = initialize_struct_save_widget()

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

    def image_changed(widgt):
        #image = widgt[0].value
        window_option = widgt.CT_Window.value
        center = window_dict[window_option][0]
        width = window_dict[window_option][1]
        widgt.Center.value = center
        widgt.Width.value = width
        minVal = center - width/2
        maxVal = center + width/2
        rangeVal = [minVal, maxVal]
        widgt.Scan.value.contrast_limits_range = rangeVal
        widgt.Scan.value.contrast_limits = rangeVal
        viewer.layers.selection.active = widgt.Scan.value
        return

    def label_changed(widgt):
        label = widgt[0].value
        if label is None:
            # Set active layer to scan
            viewer.layers.selection.active = scan_layers[0]
            return
        if viewer.dims.ndisplay == 2 and isinstance(label.metadata['isocenter'][0], (int, float)):
            set_center_slice(label)
        return

    def layer_active(event):
        if viewer.dims.ndisplay == 3 or not hasattr(event, 'value'):
            return
        layer = event.value
        if not hasattr(layer, 'metadata'):
            return
        if 'structNum' in layer.metadata:
            struct_save_widget[0].value = layer
            if isinstance(layer.metadata['isocenter'][0], (int, float)):
                set_center_slice(layer)

        # Change slice to center of that structure
    struct_save_widget.changed.connect(label_changed)
    scan_window_widget.changed.connect(image_changed)
    scanWidget = viewer.window.add_dock_widget([scan_window_widget], area='left', name="Scan", tabify=True)
    structWidget = viewer.window.add_dock_widget([struct_add_widget, struct_save_widget], area='left', name="Segmentation", tabify=True)
    # This line sets the index of the active DockWidget
    structWidget.parent().findChildren(QTabBar)[0].setCurrentIndex(0)

    viewer.layers.events.inserted.connect(struct_add_widget.reset_choices)
    viewer.layers.events.inserted.connect(struct_save_widget.reset_choices)
    viewer.layers.events.removed.connect(struct_add_widget.reset_choices)
    viewer.layers.events.removed.connect(struct_save_widget.reset_choices)
    viewer.layers.selection.events.active.connect(layer_active)
    viewer.layers.selection.events.changed.connect(layer_active)

    viewer.layers.selection.active = scan_layers[0]

    napari.run()

    return viewer, scan_layers, dose_layers, struct_layer

    # mask3M = rs.getStrMask(struct_num,planC)
    # sa = planC.scan[scan_num].scanArray - planC.scan[scan_num].scanInfo[0].CTOffset
    # fig,ax = plt.subplots(1,2)
    # h_scan = ax[0].imshow(sa[:,:,scan_slc_num])
    # h_struct = ax[0].imshow(mask3M[:,:,scan_slc_num],alpha=alpha)
    # plt.show(block=True)
    # return h_scan,h_struct

def show_scan_dose(scan_num,dose_num,slc_num,planC):
    sa = planC.scan[scan_num].scanArray - planC.scan[scan_num].scanInfo[0].CTOffset
    da = planC.dose[scan_num].doseArray
    c1 = plt.cm.ScalarMappable(cmap='gray')
    c2 = plt.cm.ScalarMappable(cmap='jet')
    fig,ax = plt.subplots(1,2)
    h_scan = ax[0].imshow(sa[:,:,slc_num])
    h_dose = ax[1].imshow(da[:,:,slc_num])
    #ax[0].colorbar(c1)
    #ax[1].colorbar(c1)
    plt.show(block=True)
    return h_scan, h_dose
