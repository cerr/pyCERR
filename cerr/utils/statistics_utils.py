"""
Funcitons to compute statistics
"""
import numpy as np

def quantile(x,q):
    n = len(x)
    y = np.sort(x)
    return(np.interp(q, np.linspace(1/(2*n), (2*n-1)/(2*n), n), y))

def prctile(x,p):
    """Equivalent to Matlab's prctile"""
    y = x[~np.isnan(x)]
    return(quantile(y,np.array(p)/100))
