"""Run:  python examples/square_wave.py"""
import numpy as np
import sympy as sp

from fourierseries import (
    fourier_coefficients,
    fourier_series,
    symbolic_coefficients,
    closed_form_square_wave,
    plot_fourier,
)

L = np.pi
N = 9
f = lambda x: np.sign(np.sin(x))

a0, a, b = fourier_coefficients(f, L, N)
print("a0 =", a0)
print("b_n =", np.round(b, 6))

x = sp.symbols("x", real=True)
a0s, a_s, b_s = symbolic_coefficients(sp.sign(sp.sin(x)), x, sp.pi, N)
print("Exact b_1 =", b_s[0], " (numeric", b[0], ")")

fig, ax = plot_fourier(f, a0, a, b, L, N=N, save_path="square_wave.png")
print("Saved square_wave.png")