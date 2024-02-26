import cerr.contour.rasterseg as rs
import warnings

import matplotlib as mpl
import napari
import numpy as np
import vispy.color
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from cerr.utils.mask import getSurfacePoints
from matplotlib.figure import Figure
from napari.layers import Labels, Image
from napari.types import LayerDataTuple
from qtpy.QtWidgets import QTabBar
from skimage import measure
import cerr.contour.rasterseg as rs
import cerr.dataclasses.scan as scn
import cerr.dataclasses.structure as cerrStr
import cerr.plan_container as pc
import matplotlib.pyplot as plt
from IPython.display import clear_output
import ipywidgets as widgets
from ipywidgets import interact

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

        scanDict = {'name': image.name,
                     'contrast_limits_range': contrast_limits_range,
                     'contrast_limits': contrast_limits,
                      'metadata': metaDict }
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
        planC = pc.import_structure_mask(mask3M, assocScanNum, structName, structNum, planC)
        if structNum is None:
            # Assign the index of added structure
            structNum = len(planC.structure) - 1
        scanNum = label.metadata['assocScanNum']
        scan_affine = label.affine
        isocenter = cerrStr.calcIsocenter(structNum, planC)
        labelDict = {'name': structName, 'affine': scan_affine,
                     'blending': 'translucent',
                     'opacity': 1,
                     'color': {1: colr, 0: np.array([0,0,0,0]), None: np.array([0,0,0,0])},
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
        shp = Labels(mask3M, name=structure_name, affine=scan_affine,
                                blending='translucent',
                                opacity = 1,
                                color = {1: colr, 0: np.array([0,0,0,0])},
                                metadata = {'planC': planC,
                                            'structNum': None,
                                            'assocScanNum': scanNum,
                                            'isocenter': [None, None, None]})
        shp.contour = 0
        return shp
    return struct_add

def initialize_dose_select_widget() -> FunctionGui:
    @magicgui(image={'label': 'Pick a Dose'}, call_button=False)
    def dose_select(image:Image) -> LayerDataTuple:
        # do something with whatever layer the user has selected
        # note: it *may* be None! so your function should handle the null case
        if image is None:
            return
        doseDict = {'name': image.name}
        return (image.data, doseDict, "image")
    return dose_select

def initialize_dose_colorbar_widget() -> FunctionGui:
    with plt.style.context('dark_background'):
        mz_canvas = FigureCanvasQTAgg(Figure(figsize=(1, 10)))
    return mz_canvas


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


def showNapari(scan_nums, str_nums, dose_nums, planC, displayMode = '2d'):

    if not isinstance(scan_nums, list):
        scan_nums = [scan_nums]
    if not isinstance(str_nums, list):
        str_nums = [str_nums]
    if not isinstance(dose_nums, list):
        dose_nums = [dose_nums]

    # Default scan window
    scanWindow = {'name': "--- Select ---",
                  'center': 0,
                  'width': 300}

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

    viewer = napari.Viewer(title='pyCERR')

    scan_colormaps = ["gray","bop orange","bop purple", "cyan", "green", "blue"] * 5
    scan_layers = []
    for i, scan_num in enumerate(scan_nums):
        sa = planC.scan[scan_num].getScanArray()
        scan_affine = scanAffineDict[scan_num]
        opacity = 0.5
        scan_name = planC.scan[scan_num].scanInfo[0].imageType
        scan_layers.append(viewer.add_image(sa,name=scan_name,affine=scan_affine,
                                           opacity=opacity, colormap=scan_colormaps[i],
                                            blending="additive",interpolation2d="linear",
                                            interpolation3d="linear",
                                            metadata = {'dataclass': 'scan',
                                                     'planC': planC,
                                                     'scanNum': scan_num,
                                                     'window': scanWindow},
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
        minDose = doseArray.min()
        maxDose = doseArray.max()
        centerDose = (minDose + maxDose) / 2
        widthDose = (maxDose - minDose)
        doseWindow = {"name": "--- Select ---",
                      "center": centerDose,
                      "width": widthDose}
        dose_lyr = viewer.add_image(doseArray,name='dose',affine=dose_affine,
                                  opacity=0.5,colormap="gist_earth",
                                  blending="additive",interpolation2d="linear",
                                  interpolation3d="linear",
                                  metadata = {'dataclass': 'dose',
                                           'planC': planC,
                                           'doseNum': dose_num,
                                           'window': doseWindow}
                                   )
        dose_layers.append(dose_lyr)

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
                                    blending='translucent',
                                    color = {1: colr, 0: np.array([0,0,0,0])},
                                    opacity = 1,
                                    metadata = {'planC': planC,
                                                'structNum': str_num,
                                                'assocScanNum': scan_num,
                                                'isocenter': isocenter})
            # From napari 0.4.19 onwards
            # from napari.utils import DirectLabelColormap
            # cmap = DirectLabelColormap(color_dict={None: None, int(1): colr, int(0): np.array([0,0,0,0])})
            # shp.colormap = cmap
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
    image_window_widget = initialize_image_window_widget()
    struct_add_widget = initialize_struct_add_widget()
    struct_save_widget = initialize_struct_save_widget()
    dose_select_widget = initialize_dose_select_widget()
    dose_colorbar_widget = initialize_dose_colorbar_widget()


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
        if 'structNum' in image.metadata:
            return
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
        uptate_colorbar(image)
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
        if viewer.dims.ndisplay == 2 and isinstance(label.metadata['isocenter'][0], (int, float)):
            set_center_slice(label)
        return

    def layer_active(event):
        if not hasattr(event, 'value'):
            return
        layer = event.value
        if not hasattr(layer, 'metadata'):
            return
        if 'structNum' in layer.metadata and viewer.dims.ndisplay == 2:
            struct_save_widget[0].value = layer
            if isinstance(layer.metadata['isocenter'][0], (int, float)):
                set_center_slice(layer)
        else:
            #uptate_colorbar(layer)
            image_changed(layer)


    def dose_changed(widgt):
        if widgt.image is None:
            return
        #mz_canvas = dose_colorbar_widget
        dose = widgt[0].value
        uptate_colorbar(dose)

    def cmap_changed(event):
        #print(event.value())
        uptate_colorbar(viewer.layers.selection.active)
        return

    def uptate_colorbar(image):
        # get Image units
        imgType = image.metadata['dataclass']
        planC = image.metadata['planC']
        units = ''
        if imgType == 'scan':
            scanNum = image.metadata['scanNum']
            units = planC.scan[scanNum].scanInfo[0].imageUnits
        elif imgType == 'dose':
            doseNum = image.metadata['doseNum']
            units = planC.dose[doseNum].doseUnits

        with plt.style.context('dark_background'):
            #mz_canvas = FigureCanvasQTAgg(Figure(figsize=(1, 0.1)))
            mz_canvas = dose_colorbar_widget
            mz_axes = mz_canvas.figure.axes
            if len(mz_axes) == 0:
                mz_canvas.figure.add_axes([0.1, 0.3, 0.2, 0.4]) #mz_canvas.figure.subplots()
                mz_axes = mz_canvas.figure.axes
            for ax in mz_axes:
                colorbar_plt = ax.get_children()
                for chld in colorbar_plt:
                    del chld
            minVal = image.contrast_limits_range[0]
            maxVal = image.contrast_limits_range[1]
            norm = mpl.colors.Normalize(vmin=minVal, vmax=maxVal)
            cb1 = mpl.colorbar.ColorbarBase(mz_axes[0], cmap=ListedColormap(image.colormap.colors),
                                norm=norm,
                                orientation='vertical')
            cb1.set_label(units)
            mz_canvas.draw()
            mz_canvas.flush_events()
            #mz_axes.axis('image')
            #mz_axes.imshow(dose_layers[-1].colormap.colors)
            #mz_canvas.figure.tight_layout()
        return


        # Change slice to center of that structure
    struct_save_widget.changed.connect(label_changed)
    image_window_widget.image.changed.connect(image_changed)
    image_window_widget.CT_Window.changed.connect(window_changed)
    image_window_widget.Center.changed.connect(center_width_changed)
    image_window_widget.Width.changed.connect(center_width_changed)
    scanWidget = viewer.window.add_dock_widget([image_window_widget], area='left', name="Window", tabify=True)
    structWidget = viewer.window.add_dock_widget([struct_add_widget, struct_save_widget], area='left', name="Segmentation", tabify=True)
    colorbars_dock = viewer.window.add_dock_widget([dose_colorbar_widget], area='right', name="Colorbar", tabify=False)
    #colorbars_dock.resize(5, 20)

    # This line sets the index of the active DockWidget
    structWidget.parent().findChildren(QTabBar)[0].setCurrentIndex(0)


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
    for scan_lyr in scan_layers:
        #scan_lyr.events.contrast_limits_range.connect(layer_active)
        scan_lyr.events.colormap.connect(cmap_changed)
        scan_lyr.events.contrast_limits_range.connect(cmap_changed)

    #dose_select_widget.changed.connect(dose_changed)

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


def show_dvf(baseScanNum, structNum, planC, viewer):
    xValsV, yValsV, zValsV = planC.scan[baseScanNum].getScanXYZVals()
    mask3M = rs.getStrMask(structNum, planC)
    # Get the surface points for the structure mask
    surf_points = getSurfacePoints(mask3M)
    sample_rate = 1
    dx = abs(np.median(np.diff(xValsV)))
    dz = abs(np.median(np.diff(zValsV)))
    while surf_points.shape[0] > 20000:
        sample_rate += 1
        if dz / dx < 2:
            surf_points = getSurfacePoints(mask3M, sample_rate, sample_rate)
        else:
            surf_points = getSurfacePoints(mask3M, sample_rate, 1)
    xSurfV = xValsV[surf_points[:, 1]]
    ySurfV = yValsV[surf_points[:, 0]]
    zSurfV = zValsV[surf_points[:, 2]]

    # Get x,y,z deformations at surface points
    xDeformV = []
    yDeformV = []
    zDeformV = []
    numPts = len(xSurfV)
    vectors = np.zeros((numPts, 2, 3), dtype=np.float32)
    vectors[:,1,0] = -yDeformV
    vectors[:,1,1] = xDeformV
    vectors[:,1,2] = zDeformV
    vectors[:,0] = [-ySurfV, xSurfV, zSurfV]
    vect_layr = viewer.add_vectors(vectors, edge_width=0.5, opacity=0.3,
                                   length=1, name="DVF",
                                   ndim=3)



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

def windowImage(image, windowCenter, windowWidth):
    imgMin = windowCenter - windowWidth // 2
    imgMax = windowCenter + windowWidth // 2
    windowedImage = image.copy()
    windowedImage[windowedImage < imgMin] = imgMin
    windowedImage[windowedImage > imgMax] = imgMax
    return windowedImage

def rotateImage(img):
    return(list(zip(*img)))

def showMplNb(scanNum, structNumV, planC, windowCenter=0, windowWidth=300):
    """
    Interactive plot using matplotlib for jupyter notebooks
    """

    # Extract scan and mask
    scan3M = planC.scan[scanNum].getScanArray()
    xVals, yVals, zVals = planC.scan[scanNum].getScanXYZVals()
    extentTrans = np.min(xVals), np.max(xVals), np.min(yVals), np.max(yVals)
    extentSag = np.min(yVals), np.max(yVals), np.min(zVals), np.max(zVals)
    extentCor = np.min(xVals), np.max(xVals), np.min(zVals), np.max(zVals)
    imgSiz = np.shape(scan3M)

    masks = list()
    for nStr in range (len(structNumV)):
        mask3M = rs.getStrMask(structNumV[nStr],planC)
        masks.append(mask3M)

    # Create slider widgets
    clear_output(wait=True)
    imgSize = np.shape(scan3M)
    sliceSliderAxial, sliceSliderSagittal, sliceSliderCoronal = createWidgets(imgSize)

    def showSlice(slcNum, view):

        clear_output(wait=True)
        print(view + ' view slice ' + str(slcNum))

        if 'fig' in locals():
            fig.remove()
        fig, (ax,ax_legend) = plt.subplots(1,2)
        ax_legend.set_visible(False)

        cmaps = [plt.colormaps["Oranges"].copy(),plt.colormaps["Oranges"].copy(), \
        plt.colormaps["Blues"].copy(),plt.colormaps["Blues"].copy(), \
        plt.colormaps["Purples"].copy(),plt.colormaps["Greens"].copy()]

        if view.lower() == 'axial':
            windowedImage = windowImage(scan3M[: ,: ,slcNum - 1], windowCenter, windowWidth)
            extent = extentTrans
        elif view.lower() == 'sagittal':
            windowedImage = rotateImage(windowImage(scan3M[:, slcNum - 1, :], windowCenter, windowWidth))
            extent = extentSag
        elif view.lower() == 'coronal':
            windowedImage = rotateImage(windowImage(scan3M[slcNum - 1, :, :], windowCenter, windowWidth))
            extent = extentCor
        else:
            raise Exeception('Invalid view type: ' + view)

        # Display scan
        im1 = ax.imshow(windowedImage, cmap=plt.cm.gray, alpha=1,
                    interpolation='nearest', extent=extent)

        #Display mask
        numLabel = len(masks)
        if view.lower() == 'axial':
            for maskNum in range(0,numLabel,1):
                maskCmap = cmaps[maskNum]
                maskCmap.set_under('k', alpha=0)
                mask3M = masks[maskNum]
                im2 = ax.imshow(mask3M[:,:,slcNum-1],
                            cmap=maskCmap, alpha=1, extent=extent,
                            interpolation='none', clim=[0.5, 1])

        elif view.lower() == 'sagittal':
            for maskNum in range(0,numLabel,1):
                maskCmap = cmaps[maskNum]
                maskCmap.set_under('k', alpha=0)
                mask3M = masks[maskNum]
                im2 = ax.imshow(rotateImage(mask3M[:, slcNum - 1, :]),
                            cmap=maskCmap, alpha=.8, extent=extent,
                            interpolation='none', clim=[0.5, 1])

        elif view.lower() == 'coronal':
            for maskNum in range(0,numLabel,1):
                maskCmap = cmaps[maskNum]
                maskCmap.set_under('k', alpha=0)
                mask3M = masks[maskNum]
                im2 = ax.imshow(rotateImage(mask3M[slcNum - 1, :, :]),
                            cmap=maskCmap, alpha=.8, extent=extent,
                            interpolation='none', clim=[0.5, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.rcParams["figure.figsize"] = (10, 10)
        plt.show()

    sliceSliderAxial.value = round(imgSiz[2]/2)
    sliceSliderSagittal.value = round(imgSiz[1]/2)
    sliceSliderCoronal.value = round(imgSiz[0]/2)

    interact(showSlice, slcNum=sliceSliderAxial.value, view='axial')
    interact(showSlice, slcNum=sliceSliderSagittal.value, view='sagittal')
    interact(showSlice, slcNum=sliceSliderCoronal.value, view='coronal')

def updateSliceAxial(change):
    outputSlcAxial = widgets.Output()
    with outputSlcAxial:
        showSlice(change['new'], 'axial')

def updateSliceSagittal(change):
    outputSlcSagittal = widgets.Output()
    with outputSlcSagittal:
        showSlice(change['new'], 'sagittal')

def updateSliceCoronal(change):
    outputSlcCoronal = widgets.Output()
    with outputSlcCoronal:
        showSlice(change['new'], 'coronal')

def createWidgets(imgSize):

    sliceSliderAxial = widgets.IntSlider(min=1,max=imgSize[2],step=1)
    outputSlcAxial = widgets.Output()

    sliceSliderSagittal = widgets.IntSlider(min=1,max=imgSize[1],step=1)
    outputSlcSagittal = widgets.Output()

    sliceSliderCoronal = widgets.IntSlider(min=1,max=imgSize[0],step=1)
    outputSlcCoronal = widgets.Output()

    sliceSliderAxial.observe(updateSliceAxial, names='value')
    sliceSliderSagittal.observe(updateSliceSagittal, names='value')
    sliceSliderCoronal.observe(updateSliceCoronal, names='value')

    return sliceSliderAxial, sliceSliderSagittal, sliceSliderCoronal
