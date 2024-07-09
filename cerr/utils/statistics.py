"""
Functions to compute statistics
"""
import numpy as np

def quantile(x,q):
    """
        Function to compute specified quantile from input array.

        Returns:
            qth quantile of values in input x.
        """
    n = len(x)
    y = np.sort(x)
    return(np.interp(q, np.linspace(1/(2*n), (2*n-1)/(2*n), n), y))

def prctile(x,p):
    """MATLAB prctile.m equivalent function to compute percentile"""
    y = x[~np.isnan(x)]
    return(quantile(y,np.array(p)/100))


def round(x):
    """
    Substitute for numpy.round(), which uses a fast but inexact algorithm.
    This function avoids documented issues of numpy.round() e.g. rounding of inputs
    exactly halfway between rounded decimal values to the nearest even value.
    See: https://numpy.org/doc/stable/reference/generated/numpy.round.html
    """
    y = np.array(x+0.5, dtype=int)
    return y