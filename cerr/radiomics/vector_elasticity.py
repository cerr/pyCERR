import sys
import os
import numpy as np
import SimpleITK as sitk
import itk


def readVfToItk(vectorFieldPath, dims=3):
    """
    Read vector field to an ITK image object to be used downstream for strain calculation.

    Args:
        vectorFieldPath (str): Path to vector field file.
        dims (int): number of dimensions for the inout vector field. Default is 3 for 3 dimensional deformation vector field.
    Returns:
        itkImg (itk.Image): ITK Image object
    """

    #disp_img = itk.imread(input_field_path, itk.F)
    # Define types for 3D vector field (Vector dimension=3, Image dimension=3, Component type=float)
    PixelType = itk.Vector[itk.F, dims]
    ImageType = itk.Image[PixelType, 3]

    # Read the vector image
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(vectorFieldPath)
    reader.Update()
    itkImg = reader.GetOutput()

    return itkImg


def computeStrainTensor(vectorField, strain_form='linear', dims=3):
    """
    Computes strain tensors from the input vector field.

    Args:
        vectorField (str or np.array): Path to vector field file or mxnxlxk size array where the last dimension stores x,y,z deformations.
        strain_form (str): string specifying the type of strain. Valid options are "linear", "Lagrangian" and "eulerian".
        dims (int): number of dimensions for the inout vector field. Default is 3 for 3 dimensional deformation vector field.
    Returns:
        tensorArray (np.array): mxnxlx6 tensor where the last dimension stores 6 elements of the
        symmetric strain matrix. [0]=Txx, [1]=Txy, [2]=Txz, [3]=Tyy, [4]=Tyz, [5]=Tzz

    Ref.: Jan Ehrhardt, Cristian Lorenz, 2013, "4D Modeling and Estimation of Respiratory Motion for Radiation Therapy",
    ISSN 1618-7210, ISBN 978-3-642-36440-2, ISBN 978-3-642-36441-9 (eBook), DOI 10.1007/978-3-642-36441-9
    """

    if isinstance(vectorField, str) and os.path.exists(vectorField):
        vectorFieldItkImage = readVfToItk(vectorField)
    else:
        vectorFieldItkImage = vectorField

    # In Python, ITK determines types automatically, but you can be explicit
    PixelType = itk.F  # Float
    strain_filter = itk.StrainImageFilter[itk.Image[itk.Vector[PixelType, dims], dims], PixelType, PixelType].New(vectorFieldItkImage)

    # Set the Strain Form
    if strain_form == 'linear': # 0 corresponds to INFINITESIMAL (Linear)
        strain_form = 0
    elif strain_form == 'Lagrangian': # 1 corresponds to GREEN (Lagrangian)
        strain_form = 1
    elif strain_form == 'eulerian': # 2 corresponds to EULERIAN (Almansi)
        strain_form = 2

    strain_filter.SetStrainForm(strain_form)

    # Verify the value (similar to TEST_SET_GET_VALUE)
    if strain_filter.GetStrainForm() != strain_form:
        print("Warning: Strain form value mismatch!")

    # Update the filter
    try:
        strain_filter.Update()
    except Exception as e:
        print(f"Filter update failed: {e}")
        return sys.exit(1)

    tensorImage = strain_filter.GetOutput()

    tensorArray = itk.array_from_image(tensorImage)

    return tensorArray


def getScalarMapsFromTensor(tensorArray):
    """
    Computes scalar features max eigen, DI and ADI from the input strain tensor.

    Args:
        tensorArray (np.array): mxnxlx6 tensor where the last dimension stores 6 elements of the
        symmetric strain matrix. [0]=Txx, [1]=Txy, [2]=Txz, [3]=Tyy, [4]=Tyz, [5]=Tzz

    Returns:
        Tuple of maxEig, di, adi maps

    Ref.: Jan Ehrhardt, Cristian Lorenz, 2013, "4D Modeling and Estimation of Respiratory Motion for Radiation Therapy",
    ISSN 1618-7210, ISBN 978-3-642-36440-2, ISBN 978-3-642-36441-9 (eBook), DOI 10.1007/978-3-642-36441-9
    """

    # Map 6-vector to 3x3 matrix
    #
    new_shape = list(tensorArray.shape[:-1])
    new_shape.extend([3,3])
    tensorArraywMatrx = np.zeros(new_shape, dtype=float)
    tensorArraywMatrx[...,0,0] = tensorArray[...,0]
    tensorArraywMatrx[...,0,1] = tensorArraywMatrx[...,1,0] = tensorArray[...,1]
    tensorArraywMatrx[...,0,2] = tensorArraywMatrx[...,2,0] = tensorArray[...,2]
    tensorArraywMatrx[...,1,1] = tensorArray[...,3]
    tensorArraywMatrx[...,1,2] = tensorArraywMatrx[...,2,1] = tensorArray[...,4]
    tensorArraywMatrx[...,2,2] = tensorArray[...,5]

    evals, evecs = np.linalg.eigh(tensorArraywMatrx)

    # eigh returns eigenvalues in ascending order. We want Descending (L1 > L2 > L3)
    l1 = evals[..., 2] # Largest
    l2 = evals[..., 1]
    l3 = evals[..., 0] # Smallest

    # Calculate Scalars
    maxEig = l1 # Eq. 7.23

    # DI as Anisotropy Ratio
    safe_l3 = np.where(l3 == 0, 1e-9, l3)
    di = l1 / safe_l3 # Eq. 7.24

    # ADI as Fractional Anisotropy-like measure or Deviatoric strain magnitude
    #mean_l = (l1 + l2 + l3) / 3.0
    #adi = np.sqrt((l1 - mean_l)**2 + (l2 - mean_l)**2 + (l3 - mean_l)**2)
    adi = np.sqrt(((l1-l2)/np.where(l2 == 0, 1e-9, l2))**2 + ((l2-l3)/np.where(l3 == 0, 1e-9, l3))**2) # 7.25

    return maxEig, di, adi
