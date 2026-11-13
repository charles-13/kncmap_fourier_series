"""Numeric Fourier coefficients on [-L, L]."""
import numpy as np
from scipy.integrate import quad


def fourier_coefficients(f, L, N, eps=1e-12):
    """
    Numeric Fourier coefficients of f on [-L, L].

    Parameters
    ----------
    f : callable
        Function to expand, must be integrable on [-L, L].
    L : float
        Half-period.
    N : int
        Number of harmonics.
    eps : float, default 1e-12
        Values below ``eps`` in magnitude are snapped to 0 (kills numerical noise).

    Returns
    -------
    a0 : float
    a  : ndarray, shape (N,)
    b  : ndarray, shape (N,)
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if N < 1:
        raise ValueError("N must be a positive integer")

    def clean(v):
        return 0.0 if abs(v) < eps else float(v)

    a0 = clean((1 / L) * quad(f, -L, L, limit=200)[0])
    a = np.zeros(N)
    b = np.zeros(N)
    for n in range(1, N + 1):
        a[n - 1] = clean((1 / L) * quad(lambda x: f(x) * np.cos(n * np.pi * x / L), -L, L, limit=200)[0])
        b[n - 1] = clean((1 / L) * quad(lambda x: f(x) * np.sin(n * np.pi * x / L), -L, L, limit=200)[0])
    return a0, a, b