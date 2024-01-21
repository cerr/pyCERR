import matplotlib.pyplot as plt
import cerr.contour.rasterseg as rs
import napari
#from napari.utils import transforms
import numpy as np
#import matplotlib.colors as mcolors
from skimage import measure
import vispy.color
import cerr.dataclasses.scan as scn
import cerr.dataclasses.structure as cerrStr
from napari.layers import Layer
from napari.types import LabelsData, ImageData
from napari.layers import Labels, Image
from magicgui import magicgui
import cerr.plan_container as pc
from napari.utils.events import Event
from napari import Viewer

@magicgui(label={'label': 'Select Structure'}, call_button = 'Save updates')
def struct_save(label: Labels, structure_name=None):
    # do something with whatever layer the user has selected
    # note: it *may* be None! so your function should handle the null case
    if label is None:
        return
    planC = label.metadata['planC']
    structNum = label.metadata['structNum']
    assocScanNum = label.metadata['assocScanNum']
    structName = label.name
    planC = pc.import_structure_mask(label.data, assocScanNum, structName, planC)
    return planC

@magicgui(image={'label': 'Pick a Scan'}, call_button='Create')
def struct_add(image: Image, structure_name = "") -> Labels:
    # do something with whatever layer the user has selected
    # note: it *may* be None! so your function should handle the null case
    # planC = label.metadata['planC']
    # structNum = label.metadata['structNum']
    # assocScanNum = label.metadata['assocScanNum']
    # structName = label.name
    #print(label.name)
    #print(planC.structure[structNum].structureName)
    #planC = pc.import_structure_mask(label.data, assocScanNum, structName, planC)
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
                            color = {1: colr, 0: np.array([0,0,0,0])},
                            opacity = 1, metadata = {'planC': planC,
                                                       'structNum': strNum,
                                                       'assocScanNum': scanNum})
    shp.contour = 2
    return shp

@struct_add.call_button.clicked.connect
def update_structure_names(event):
    print(type(event))
    #len(viewer.layers)


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
        #doseArray= np.flip(doseArray,axis=0)
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
            mask3M[mask3M] = int(str_num + 1)
            shp = viewer.add_labels(mask3M, name=str_name, affine=scan_affine,
                                    num_colors=1, blending='translucent',
                                    color = {1: colr, 0: np.array([0,0,0,0])},
                                    opacity = 1, metadata = {'planC': planC,
                                                               'structNum': str_num,
                                                               'assocScanNum': scan_num})
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
    if len(struct_layer)> 0:
        viewer.window.add_dock_widget([struct_add, struct_save], area='left', name="Structure", tabify=True)
        #viewer.layers.selection.events.changed.connect(struct_show)
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
