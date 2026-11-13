"""
fourierseries
=============

Fourier series on [-L, L] (period 2L):

    f(x) ~ a0/2 + sum_{n>=1} [ a_n cos(n*pi*x/L) + b_n sin(n*pi*x/L) ]

    a0 = (1/L) * ∫_{-L}^{L} f(x) dx
    a_n = (1/L) * ∫_{-L}^{L} f(x) cos(n*pi*x/L) dx
    b_n = (1/L) * ∫_{-L}^{L} f(x) sin(n*pi*x/L) dx
"""
from .coefficients import fourier_coefficients
from .series import fourier_series
from .symbolic import symbolic_coefficients, closed_form_square_wave
from .plotting import plot_fourier

__version__ = "0.1.0"
__all__ = [
    "fourier_coefficients",
    "fourier_series",
    "symbolic_coefficients",
    "closed_form_square_wave",
    "plot_fourier",
]