import numpy as np


def calcNGTDM(scan_array, patch_size, numGrLevels):

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
    s = np.zeros((numGrLevels, 1), dtype=float)
    p = np.zeros((numGrLevels, 1), dtype=float)

    for slc_num in range(num_slcs_pad, num_slices + num_slcs_pad):
        calc_slc_ind = calc_ind[:, :, slc_num].ravel(order="F")
        num_calc_voxels = np.sum(calc_slc_ind)
        indSlcM = indM[:,calc_slc_ind]
        #ind_slc = np.where(calc_slc_ind)[0]
        slc_v = np.arange(slc_num - patch_size[2], slc_num + patch_size[2] + 1)
        nbhood_size = indSlcM.shape[0]

        q_m = np.empty((slc_v.size * nbhood_size, num_calc_voxels), dtype=np.float32)
        m_m = np.empty((slc_v.size * nbhood_size, num_calc_voxels), dtype=np.float32)

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
        numNeighborsV = m_m.sum(axis=0) - 1
        numNeighborsV[numNeighborsV == 0] = np.finfo(float).eps # avoid divide by 0 warning
        q_m = q_m / numNeighborsV
        q_m[np.isnan(q_m)] = 0
        q_m = np.abs(vox_val_v - q_m.sum(axis=0))
        #q_m[m_m==0] = 0
        #q_m = np.sum(q_m, axis=0)

        for lev in range(1, numGrLevels + 1):
            ind_lev_v = (vox_val_v == lev) & vox_mask_v.astype(bool)
            vals_v = q_m[ind_lev_v]
            s[lev - 1, :] += vals_v.sum().astype(s.dtype)

    # Calculate level probabilities (p)
    numVoxels = np.sum(scan_array > 0)
    for iLev in range(1, numGrLevels + 1):
        scanLev3M = scan_array == iLev
        p[iLev - 1] = np.sum(scanLev3M) / numVoxels

    return s, p

# Example usage:
# s, p = calcNGTDM(scanArray3M, patchSizeV, numGrLevels, hWait)


def ngtdmToScalarFeatures(s, p, numVoxels):
    featuresS = {}

    # Coarseness
    featuresS['coarseness'] = 1 / (np.sum(s * p) + 1e-6)

    # Contrast
    Ng = np.sum(p > 0)
    numLevels = len(p)
    indV = np.arange(1, numLevels + 1)
    indV = indV[:,None]
    term1 = 0
    term2 = 0
    for lev in range(1, numLevels + 1):
        term1 += np.sum(p * np.roll(p, lev) * (indV - np.roll(indV, lev))**2)
        term2 += s[lev - 1]
    featuresS['contrast'] = 1 / (Ng * (Ng - 1)) * term1 * term2 / numVoxels

    # Busyness
    denom = 0
    for lev in range(1, numLevels + 1):
        pShiftV = np.roll(p, lev)
        indShiftV = np.roll(indV, lev)
        usePv = p > 0
        usePshiftV = pShiftV > 0
        denom += np.sum(usePv * usePshiftV * np.abs(p * indV - pShiftV * indShiftV))
    featuresS['busyness'] = np.sum(p * s) / denom

    # Complexity
    complxty = 0
    for lev in range(1, numLevels + 1):
        pShiftV = np.roll(p, lev)
        sShiftV = np.roll(s, lev)
        indShiftV = np.roll(indV, lev)
        usePv = p > 0
        usePshiftV = pShiftV > 0
        term1 = np.abs(indV - indShiftV)
        term2 = usePv * usePshiftV * (p * s + pShiftV * sShiftV) / (p + pShiftV + np.finfo(float).eps)
        complxty += np.sum(term1 * term2)
    featuresS['complexity'] = complxty / numVoxels

    # Texture strength
    strength = 0
    for lev in range(1, numLevels + 1):
        pShiftV = np.roll(p, -lev)
        indShiftV = np.roll(indV, -lev)
        usePv = p > 0
        usePshiftV = pShiftV > 0
        term = np.sum(usePv * usePshiftV * (p + pShiftV) * (indV - indShiftV)**2)
        strength += term
    featuresS['strength'] = strength / np.sum(s)

    return featuresS
