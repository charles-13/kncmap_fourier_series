"""Evaluation of the Fourier partial sum."""
import numpy as np


def fourier_series(x, a0, a, b, L):
    """
    Evaluate the partial Fourier sum at x.

    S_N(x) = a0/2 + sum_{n=1}^{N} [ a_n cos(n*pi*x/L) + b_n sin(n*pi*x/L) ]

    Parameters
    ----------
    x : scalar or array-like
    a0 : float
    a, b : array-like of length N
    L : float

    Returns
    -------
    ndarray of the same shape as x
    """
    x = np.asarray(x, dtype=float)
    s = np.full_like(x, a0 / 2.0)
    for n in range(1, len(a) + 1):
        s += a[n - 1] * np.cos(n * np.pi * x / L) + b[n - 1] * np.sin(n * np.pi * x / L)
    return s