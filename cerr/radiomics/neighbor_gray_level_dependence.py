"""
This module contains routines for claculation of Gray Level Dependence texture features
"""

import numpy as np

def calcNGLDM(scan_array, patch_size, num_grayscale_levels, a):
    """

    This function calculates the Neighborhood Gray Level Dependence Matrix for the passed quantized image based on
    IBSI definitions https://ibsi.readthedocs.io/en/latest/03_Image_features.html#neighbouring-grey-level-dependence-based-features

    Args:
        scan_array (np.ndarray(dtype=int)): quantized 3d matrix obtained, for example, from radiomics.preprocess.imquantize_cerr
        patch_size (list): list of length 3 defining patch radius for row, col, slc dimensions.
        num_grayscale_levels (int): Number of gray levels.
        a: coarseness parameter

    Returns:
        np.ndarray: NGLDM matrix os size (num_grayscale_levels, max_nbhood_size)

        The output can be passed to ngldmToScalarFeatures to get NGLDM texture features.

    """

    # Get indices of non-NaN voxels
    calc_ind = scan_array > 0

    # Grid resolution
    slc_window = 2 * patch_size[2] + 1
    row_window = 2 * patch_size[0] + 1
    col_window = 2 * patch_size[1] + 1

    # Build distance matrices
    num_cols_pad = col_window // 2
    num_rows_pad = row_window // 2
    num_slcs_pad = slc_window // 2

    # Get number of voxels per slice
    num_rows, num_cols, num_slices = scan_array.shape
    num_voxels = num_rows * num_cols

    # Pad q for sliding window
    q = np.pad(scan_array, ((num_rows_pad, num_rows_pad), (num_cols_pad, num_cols_pad), (num_slcs_pad, num_slcs_pad)),
               mode='constant', constant_values=(0,))

    calc_ind = np.pad(calc_ind, ((0, 0), (0, 0), (num_slcs_pad, num_slcs_pad)), mode='constant', constant_values=0)

     # Index calculation adapted from
     # http://stackoverflow.com/questions/25449279/efficient-implementation-of-im2col-and-col2im
    (m,n,_) = q.shape
    start_ind = np.arange(1,m-row_window+2)[None,:] + (np.arange(0,n-col_window+1)*m)[:,None]
    start_ind = start_ind.T.ravel(order="F")[:,None]

    # Row indices
    lin_row = start_ind + np.arange(0,row_window).T
    lin_row = np.moveaxis(lin_row[:,:,None],[0,1,2],[2,0,1])

    # Get linear indices based on row and col indices and get desired output
    indM = lin_row + np.arange(0,col_window)[None,:,None]*m
    indM = np.reshape(indM,(row_window*col_window,indM.shape[2]),order="F") - 1
    # Initialize the s (NGLDM) matrix
    max_nbhood_size = (2 * patch_size[0] + 1) * (2 * patch_size[1] + 1) * (2 * patch_size[2] + 1) - 1

    s = np.zeros((num_grayscale_levels, max_nbhood_size+1), dtype=np.uint32)
    #p = np.zeros((num_grayscale_levels, 1), dtype=np.uint32)

    for slc_num in range(num_slcs_pad, num_slices + num_slcs_pad):
        calc_slc_ind = calc_ind[:, :, slc_num].ravel(order="F")
        num_calc_voxels = np.sum(calc_slc_ind)
        indSlcM = indM[:,calc_slc_ind]
        #ind_slc = np.where(calc_slc_ind)[0]
        slc_v = np.arange(slc_num - patch_size[2], slc_num + patch_size[2] + 1)
        nbhood_size = indSlcM.shape[0]

        slcNeighborSiz = slc_v.size * nbhood_size

        q_m = np.zeros((slcNeighborSiz, num_calc_voxels), dtype=np.float32)
        m_m = np.zeros((slcNeighborSiz, num_calc_voxels), dtype=np.float32)

        count = 0
        for i_slc in slc_v:
            q_slc = q[:, :, i_slc]
            mask_slc_m = np.pad(calc_ind[:, :, i_slc], ((num_rows_pad, num_rows_pad), (num_cols_pad, num_cols_pad)),
                               mode='constant', constant_values=0)
            q_m[count * nbhood_size:(count + 1) * nbhood_size, :] = q_slc.ravel(order="F")[indSlcM]
            m_m[count * nbhood_size:(count + 1) * nbhood_size, :] = mask_slc_m.ravel(order="F")[indSlcM]
            count += 1

        current_voxel_index = nbhood_size * slc_v.size // 2
        vox_val_v = q_m[current_voxel_index, :].copy()
        vox_mask_v = m_m[current_voxel_index, :].copy()
        q_m = np.delete(q_m, current_voxel_index, axis=0)
        m_m = np.delete(m_m, current_voxel_index, axis=0)
        #q_m[np.isnan(q_m)] = 0
        q_m = np.abs(q_m - vox_val_v) <= a
        q_m[m_m==0] = 0
        q_m = np.sum(q_m, axis=0)

        for lev in range(1, num_grayscale_levels + 1):
            ind_lev_v = (vox_val_v == lev) & vox_mask_v.astype(bool)
            vals_v = q_m[ind_lev_v]
            s[lev - 1, :] += np.bincount(vals_v + 1, minlength=slcNeighborSiz + 1)[1:].astype(s.dtype)

    return s


def ngldmToScalarFeatures(s, numVoxels):
    """

    This function calculates scalar texture features from Neighborhood Gray Level Dependence matrix as per
    IBSI definitions https://ibsi.readthedocs.io/en/latest/03_Image_features.html#neighbouring-grey-level-dependence-based-features

    Args:
        s (np.ndarray: NGLDM matrix of size (num_grayscale_levels, max_nbhood_size)
        numVoxels (int): number of voxels in the region of interest used for generating s.

    Returns:
        dict: dictionary with scalar NGLDM texture features as its
             fields.
    """

    featuresS = {}

    Ns = np.sum(s)
    Nn = s.shape[1]
    Ng = s.shape[0]
    lenV = np.arange(1, Nn + 1, dtype = np.uint64)
    levV = np.arange(1, Ng + 1, dtype = np.uint64)
    levV = levV[None,:].astype(np.int64)
    lenV = lenV[None,:].astype(np.int64)

    s = s.astype(float)

    # Low dependence emphasis
    sLdeM = s / lenV**2
    featuresS['lowDependenceEmphasis'] = sLdeM.sum() / Ns

    # High dependence emphasis
    sHdeM = s * lenV**2
    featuresS['highDependenceEmphasis'] = sHdeM.sum() / Ns

    # Low gray level count emphasis
    sLgceM = s.T / levV**2
    featuresS['lowGrayLevelCountEmphasis'] = sLgceM.sum() / Ns

    # High gray level count emphasis
    sHgceM = s.T * levV**2
    featuresS['highGrayLevelCountEmphasis'] = sHgceM.sum() / Ns

    # Low dependence low gray level emphasis
    sLdlgeM = sLdeM.T / levV**2
    featuresS['lowDependenceLowGrayLevelEmphasis'] = sLdlgeM.sum() / Ns

    # Low dependence high gray level emphasis
    sLdhgeM = sLdeM.T * levV**2
    featuresS['lowDependenceHighGrayLevelEmphasis'] = sLdhgeM.sum() / Ns

    # High dependence low gray level emphasis
    sHdlgeM = sHdeM.T / levV**2
    featuresS['highDependenceLowGrayLevelEmphasis'] = sHdlgeM.sum() / Ns

    # High dependence high gray level emphasis
    sHdhgeM =  sHdeM.T * levV**2
    featuresS['highDependenceHighGrayLevelEmphasis'] = sHdhgeM.sum() / Ns

    # Gray level non-uniformity
    featuresS['grayLevelNonUniformity'] = np.sum(np.sum(s, axis=1)**2) / Ns

    # Gray level non-uniformity normalized
    featuresS['grayLevelNonUniformityNorm'] = np.sum(np.sum(s, axis=1)**2) / (Ns**2)

    # Dependence count non-uniformity
    featuresS['dependenceCountNonuniformity'] = np.sum(np.sum(s, axis=0)**2) / Ns

    # Dependence count non-uniformity normalized
    featuresS['dependenceCountNonuniformityNorm'] = np.sum(np.sum(s, axis=0)**2) / (Ns**2)

    # Dependence count percentage
    featuresS['dependenceCountPercentage'] = Ns / numVoxels

    # Gray level variance
    iPij = s.T / np.sum(s) * levV
    mu = np.sum(iPij)
    iMinusMuPij = s.T / np.sum(s) * (levV - mu)**2
    featuresS['grayLevelVariance'] = np.sum(iMinusMuPij)

    # Dependence count variance
    jPij = s / np.sum(s) * lenV
    mu = np.sum(jPij)
    jMinusMuPij = s / np.sum(s) * (lenV - mu)**2
    featuresS['dependenceCountVariance'] = np.sum(jMinusMuPij)

    # Dependence count entropy
    p = s / np.sum(s)
    featuresS['dependenceCountEntropy'] = -np.sum(p * np.log2(p + np.finfo(float).eps))

    # Dependence count energy
    featuresS['dependenceCountEnergy'] = np.sum(p**2)

    return featuresS
