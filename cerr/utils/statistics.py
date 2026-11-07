"""
Functions to compute statistics
"""
import numpy as np

def quantile(x,q):
    """Quantile of the values in ``x`` (MATLAB ``quantile`` definition).

    Args:
        x (np.ndarray): 1D values (no NaN; see :func:`prctile`).
        q (float or np.ndarray): Quantile(s) in [0, 1].

    Returns:
        float or np.ndarray: The q-th quantile(s) of ``x``.
    """
    n = len(x)
    y = np.sort(x)
    return(np.interp(q, np.linspace(1/(2*n), (2*n-1)/(2*n), n), y))

def prctile(x,p):
    """Percentile of the non-NaN values in ``x`` (MATLAB ``prctile`` equivalent).

    Args:
        x (np.ndarray): 1D values; NaN values are ignored.
        p (float or list): Percentile(s) in [0, 100].

    Returns:
        float or np.ndarray: The p-th percentile(s).
    """
    y = x[~np.isnan(x)]
    return(quantile(y,np.array(p)/100))


def round(x):
    """
    Substitute for numpy.round(), which uses a fast but inexact algorithm.
    This function avoids documented issues of numpy.round() e.g. rounding of inputs
    exactly halfway between rounded decimal values to the nearest even value.
    See: https://numpy.org/doc/stable/reference/generated/numpy.round.html

    Args:
        x (np.ndarray or float): Values to round.

    Returns:
        np.ndarray(dtype=int): ``int(x + 0.5)``, i.e. halves round up. The conversion truncates
        toward zero, so this is only correct for x >= -0.5 (e.g. -1.6 gives -1).
    """
    y = np.array(x+0.5, dtype=int)
    return y