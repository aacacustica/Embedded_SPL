import numpy as np

def lfilter_np(b, a, x, zi=None):
    """
    scipy.signal.lfilter

    Parameters
    ----------
    b, a : array_like
        Numerator/denominator coefficients.
    x : array_like
        Input signal (1D).
    zi : array_like or None
        Initial filter state of length max(len(a), len(b)) - 1.

    Returns
    -------
    y : ndarray
        Filtered output.
    zf : ndarray
        Final filter state (use it for streaming).
    """
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError("lfilter_np only supports 1D arrays")
    if a.size == 0 or b.size == 0:
        raise ValueError("Empty filter coefficients")



    # normalize a[0] == 1
    a0 = a[0]
    if a0 == 0:
        raise ValueError("a[0] must be non-zero")
    if a0 != 1.0:
        b = b/a0
        a = a /a0
    n = max(a.size, b.size)-1  # number of state variables
    if zi is None:
        z = np.zeros(n, dtype=np.float64)
    else:
        z = np.asarray(zi, dtype=np.float64).copy()
        if z.size != n:
            raise ValueError(f"zi must have length {n}, got {z.size}")
    y = np.empty_like(x, dtype=np.float64)




    # filtering
    for i in range(x.size):
        xi = x[i]
        if n == 0:
            y[i] = b[0] * xi
            continue

        y0 = b[0] * xi + z[0]
        y[i] = y0

        # update states
        for k in range(1, n):
            bk = b[k] if k < b.size else 0.0
            ak = a[k] if k < a.size else 0.0
            z[k - 1] = z[k] + bk * xi - ak * y0

        bn = b[n] if n < b.size else 0.0
        an = a[n] if n < a.size else 0.0
        # last state
        z[n - 1] = bn * xi - an * y0



    return y, z
