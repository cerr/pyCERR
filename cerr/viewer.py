import matplotlib.pyplot as plt
import cerr.contour.rasterseg as rs
import napari
#from napari.utils import transforms
import numpy as np
#import matplotlib.colors as mcolors
from skimage import measure
import vispy.color
import cerr.dataclasses.scan as scn


def getContours(strNum, assocScanNum, planC):
    numSlcs = len(planC.structure[strNum].contour)
    polygons = []
    for slc in range(numSlcs):
        if planC.structure[strNum].contour[slc]:
            for seg in planC.structure[strNum].contour[slc].segments:
                rowV, colV = rs.xytom(seg.points[:,0], seg.points[:,1],slc,planC, assocScanNum)
                pts = np.array((rowV, colV, slc*rowV**0), dtype=float).T
                polygons.append(pts)
    return polygons

def show_scan_struct_dose(scan_nums, str_nums, dose_nums, planC, displayMode = '2d'):

    if not isinstance(scan_nums, list):
        scan_nums = [scan_nums]
    if not isinstance(str_nums, list):
        str_nums = [str_nums]
    if not isinstance(dose_nums, list):
        dose_nums = [dose_nums]

    # scan_num = 0
    # sa = planC.scan[scan_num].scanArray - planC.scan[scan_num].scanInfo[0].CTOffset
    # #sa = np.flip(sa,axis=0)
    # x,y,z = planC.scan[0].getScanXYZVals()
    # y = -y
    # dx = x[1] - x[0]
    # dy = y[1] - y[0]
    # dz = z[1] - z[0]
    # scan_affine = np.array([[dy, 0, 0, y[0]], [0, dx, 0, x[0]], [0, 0, dz, z[0]], [0, 0, 0, 1]])

    #rotate = np.eye(3)
    #shear = np.eye(3)
    #scale = [dy,dx,dz]
    #translate = y[0],x[0],z[0]
    #scan_affine = transforms.Affine(scale, translate, rotate=rotate, shear=shear, name='scan2world')

    # dose_num = 0
    # doseArray = planC.dose[dose_num].doseArray
    # #doseArray= np.flip(doseArray,axis=0)
    # xd,yd,zd = planC.dose[0].getDoseXYZVals()
    # yd = -yd
    # dx = xd[1] - xd[0]
    # dy = yd[1] - yd[0]
    # dz = zd[1] - zd[0]
    # dose_affine = np.array([[dy, 0, 0, yd[0]], [0, dx, 0, xd[0]], [0, 0, dz, zd[0]], [0, 0, 0, 1]])

    #scale = [dy,dx,dz]
    #translate = y[0],x[0],z[0]
    #dose_affine = transforms.Affine(scale, translate, rotate=rotate, shear=shear, name='dose2world')
    #viewer, image_layer = napari.imshow(sa.astype(float), name='scan', affine=scan_affine)

    viewer = napari.Viewer()

    scan_colormaps = ["gray","bop orange","bop purple", "cyan", "green", "blue"] * 5
    scan_layers = []
    for i, scan_num in enumerate(scan_nums):
        sa = planC.scan[scan_num].scanArray - planC.scan[scan_num].scanInfo[0].CTOffset
        #sa = np.flip(sa,axis=0)
        x,y,z = planC.scan[scan_num].getScanXYZVals()
        y = -y # negative since napari viewer y increases from top to bottom
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        scan_affine = np.array([[dy, 0, 0, y[0]], [0, dx, 0, x[0]], [0, 0, dz, z[0]], [0, 0, 0, 1]])
        opacity = 0.5
        scan_name = planC.scan[scan_num].scanInfo[0].imageType
        scan_layers.append(viewer.add_image(sa,name=scan_name,affine=scan_affine,
                                           opacity=opacity, colormap=scan_colormaps[i],
                                            blending="additive",interpolation2d="linear",
                                            interpolation3d="linear"
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
                                  blending="additive",interpolation2d="linear",
                                  interpolation3d="linear"
                                   ))

    # reference: https://gist.github.com/AndiH/c957b4d769e628f506bd
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)] * 4

    # Get x,y,z ranges for scaling results of marching cubes mesh
    mins = np.array([min(y), min(x), min(z)])
    maxes = np.array([max(y), max(x), max(z)])
    ranges = maxes - mins
    struct_layer = np.empty(len(str_nums))
    for i,str_num in enumerate(str_nums):
        mask3M = rs.getStrMask(str_num,planC)
        str_name = planC.structure[str_num].structureName
        verts, faces, _, _ = measure.marching_cubes(volume=mask3M, level=0.5)
        verts_scaled = verts * ranges / np.array(mask3M.shape) - mins
        colr = np.asarray(tableau20[i])/255
        cmap = vispy.color.Colormap([colr,colr])
        # Get scan affine
        scan_num = scn.getScanNumFromUID(planC.structure[str_num].assocScanUID, planC)
        x,y,z = planC.scan[scan_num].getScanXYZVals()
        y = -y # negative since napari viewer y increases from top to bottom
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        scan_affine = np.array([[dy, 0, 0, y[0]], [0, dx, 0, x[0]], [0, 0, dz, z[0]], [0, 0, 0, 1]])
        if displayMode.lower() == '3d':
            labl = viewer.add_surface((verts, faces),opacity=0.5,shading="flat",
                                              affine=scan_affine, name=str_name,
                                              colormap=cmap)
            struct_layer = np.append(struct_layer, labl)
            # #labels_layer = viewer.add_labels(mask3M, name=str_name, affine=scan_affine,
            # #                                 num_colors=1,opacity=0.5,visible=False,
            # #                                 color={1:np.asarray(tableau20[i])/255})
        elif displayMode.lower() == '2d':
            polygons = getContours(str_num, scan_num, planC)

            shp = viewer.add_shapes(polygons, shape_type='path', edge_width=2,
                              edge_color=np.array(tableau20[i])/255, face_color=[0]*4,
                              affine=scan_affine, name=str_name)
            struct_layer = np.append(struct_layer, shp)

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
    viewer.axes.visible = True
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
