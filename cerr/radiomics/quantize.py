import numpy as np

def imquantize_cerr(x, nL=None, xmin=None, xmax=None, binwidth=None):
    """
    Function to quantize an image based on the number of bins (nL) or the bin width (binwidth).

    Args:
        x: Input image matrix.
        nL: Number of quantization levels (optional).
        xmin: Minimum value for quantization (optional).
        xmax: Maximum value for quantization (optional).
        binwidth: Bin width for quantization (optional).

    Returns:
        q: Quantized image.

    Note: If xmin and xmax are not provided, they are computed from the input image x.
    """
    if xmin is not None:
        x[x < xmin] = xmin
    else:
        xmin = np.nanmin(x)

    if xmax is not None:
        x[x > xmax] = xmax
    else:
        xmax = np.nanmax(x)

    if nL is not None:
        slope = (nL - 1) / (xmax - xmin)
        intercept = 1 - (slope * xmin)
        q = np.round(slope * x + intercept)
        q[np.isnan(q)] = 0
        q = q.astype(np.int)
    elif binwidth is not None:
        q = (x - xmin) / binwidth
        q[np.isnan(q)] = -1
        q = q.astype(int) + 1
    else:
        # No quantization
        print('Returning input image. Specify the number of bins or the binwidth to quantize.')
        q = x

    return q
