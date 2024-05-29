"""

This module contains routine for calculation of 1st order statistical features.

"""

import numpy as np
from scipy.stats import skew, kurtosis, entropy
import cerr.contour.rasterseg as rs
import cerr.dataclasses.scan as scn
import cerr.plan_container as pc
from cerr.utils.statistics_utils import quantile, prctile

def radiomics_first_order_stats(planC, structNum, offsetForEnergy=0, binWidth=None, binNum=None):
    """

    This routine calculates 1st order statistical features from the input image and segmentation based on
    IBSI definitions https://ibsi.readthedocs.io/en/latest/03_Image_features.html#intensity-based-statistical-features.

    Args:
        planC (plan_container.planC or np.ndarray(dtype=flat)): planC containing the image and segmentation
                                or scan np.ndarray.
        structNum (int or np.ndarray(dtype=int)): index of structure number in planC or binary mask for segmentation
                                that matches the scan dimensions
        offsetForEnergy(float): optional, value to add to scan for computing the Energy, TotalEnergy and RMS features
        binWidth(float): optional, bin width for discretizing the input scan.  Required when binNum is None.
                                The default value is None.
        binNum(int): optional, number of bins to discretize the input scan. Required when binWidth is None.
                                The default value is None.
    Returns:
        dict: dictionary of features

    """

    if isinstance(planC, pc.PlanC):
        # Get structure Mask
        maskStruct3M = rs.getStrMask(structNum,planC)

        # Get uniformized scan in HU
        assocScanUID = planC.structure[structNum].assocScanUID
        scanNum = scn.getScanNumFromUID(assocScanUID,planC)
        maskScan3M = np.double(planC.scan[scanNum].getScanArray())

        # Offset for energy calculation
        if offsetForEnergy is None:
            offsetForEnergy = planC.scan[scanNum].scanInfo[0].CTOffset

        # Get Pixel-size
        xUnifV, yUnifV, zUnifV = planC.scan[scanNum].getScanXYZVals()
        PixelSpacingXi = abs(xUnifV[1] - xUnifV[0])
        PixelSpacingYi = abs(yUnifV[1] - yUnifV[0])
        PixelSpacingZi = abs(zUnifV[1] - zUnifV[0])
        VoxelVol = PixelSpacingXi * PixelSpacingYi * PixelSpacingZi * 1000 # units of mm

        # Iarray = Data.Image(~isnan(Data.Image));     # Array of Image values
        indStructV = maskStruct3M == 1
        Iarray = maskScan3M[indStructV]
    else:
        if offsetForEnergy is None:
            offsetForEnergy = 0

        Iarray = planC
        VoxelVol = structNum

    # Calculate standard PET parameters
    RadiomicsFirstOrderS = dict()
    RadiomicsFirstOrderS['min'] = np.nanmin(Iarray)
    RadiomicsFirstOrderS['max'] = np.nanmax(Iarray)
    RadiomicsFirstOrderS['mean'] = np.nanmean(Iarray)
    RadiomicsFirstOrderS['range'] = np.ptp(Iarray)
    RadiomicsFirstOrderS['std'] = np.nanstd(Iarray, ddof=0)
    RadiomicsFirstOrderS['var'] = np.nanvar(Iarray, ddof=0)
    RadiomicsFirstOrderS['median'] = np.nanmedian(Iarray)

    # Skewness is a measure of the asymmetry of the data around the sample mean.
    RadiomicsFirstOrderS['skewness'] = skew(Iarray, nan_policy='omit')

    # Kurtosis is a measure of how outlier-prone a distribution is.
    RadiomicsFirstOrderS['kurtosis'] = kurtosis(Iarray, fisher=False, nan_policy='omit') - 3

    # Entropy is a statistical measure of randomness that can be used to characterize the texture of the input image
    xmin = np.min(Iarray)
    xmax = np.max(Iarray)
    edgeMin = 0  # to match the MATLAB definition
    if binWidth is None and binNum is not None:
        binWidth = (xmax - xmin) / binNum
    elif binWidth is None:
        binWidth = 25
    if binWidth == 0:
        RadiomicsFirstOrderS['entropy'] = 0
    else:
        N = np.ceil((xmax - edgeMin) / binWidth).astype(int)
        offsetForEntropy = -np.min(Iarray.min(),0)
        xmin = 0
        xmax = np.max(Iarray) + offsetForEntropy + binWidth/2
        counts, _ = np.histogram(Iarray + offsetForEntropy, bins= np.arange(xmin,xmax,binWidth))
        RadiomicsFirstOrderS['entropy'] = entropy(counts, base=2)

    # Root mean square (RMS)
    RadiomicsFirstOrderS['rms'] = np.sqrt(np.nansum((Iarray + offsetForEnergy) ** 2) / Iarray.size)

    # Energy (integraal(a^2))
    RadiomicsFirstOrderS['energy'] = np.nansum((Iarray + offsetForEnergy) ** 2)

    # Total Energy (voxelVolume * integraal(a^2))
    RadiomicsFirstOrderS['totalEnergy'] = np.nansum((Iarray + offsetForEnergy) ** 2) * VoxelVol

    # Mean deviation (also called mean absolute deviation)
    RadiomicsFirstOrderS['meanAbsDev'] = np.mean(np.abs(Iarray - np.nanmean(Iarray)))

    # Median absolute deviation
    RadiomicsFirstOrderS['medianAbsDev'] = np.nansum(np.abs(Iarray - RadiomicsFirstOrderS['median'])) / Iarray.size

    # P10
    RadiomicsFirstOrderS['P10'] = prctile(Iarray, 10)

    # P90
    RadiomicsFirstOrderS['P90'] = prctile(Iarray, 90)

    Iarray10_90 = Iarray.copy()
    idx10_90 = (Iarray >= RadiomicsFirstOrderS['P10']) & (Iarray <= RadiomicsFirstOrderS['P90'])
    Iarray10_90[~idx10_90] = np.nan
    idx10_90[np.isnan(idx10_90)] = 0

    # Robust Mean Absolute Deviation
    devM = np.abs(Iarray10_90 - np.nanmean(Iarray10_90))
    RadiomicsFirstOrderS['robustMeanAbsDev'] = np.nanmean(devM)

    # Robust Median Absolute Deviation
    RadiomicsFirstOrderS['robustMedianAbsDev'] = np.nansum(np.abs(Iarray10_90 - np.nanmedian(Iarray10_90))) / np.sum(idx10_90)

    # Inter-Quartile Range (IQR)
    # P75 - P25
    p75 = prctile(Iarray, 75)
    p25 = prctile(Iarray, 25)
    RadiomicsFirstOrderS['interQuartileRange'] = p75 - p25

    # Quartile coefficient of Dispersion
    RadiomicsFirstOrderS['coeffDispersion'] = (p75 - p25) / (p75 + p25)

    # Coefficient of variation
    RadiomicsFirstOrderS['coeffVariation'] = RadiomicsFirstOrderS['std'] / (RadiomicsFirstOrderS['mean'] + np.finfo(float).eps)

    return RadiomicsFirstOrderS
