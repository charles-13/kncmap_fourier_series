import numpy as np
import sympy as sp
import pytest

from fourierseries import (
    fourier_coefficients,
    fourier_series,
    symbolic_coefficients,
    closed_form_square_wave,
)


def test_square_wave_numeric_odd_only():
    f = lambda x: np.sign(np.sin(x))
    L = np.pi
    a0, a, b = fourier_coefficients(f, L, 9)
    assert a0 == 0.0
    assert np.allclose(a, 0.0)
    assert np.isclose(b[0], 4 / np.pi)
    assert b[1] == 0.0
    assert np.isclose(b[2], 4 / (3 * np.pi))


def test_square_wave_exact():
    x = sp.symbols("x", real=True)
    a0, a, b = symbolic_coefficients(sp.sign(sp.sin(x)), x, sp.pi, 5)
    assert sp.simplify(a0) == 0
    assert b[0] == 4 / sp.pi
    assert b[1] == 0
    assert b[2] == 4 / (3 * sp.pi)


def test_fourier_series_reconstruction_x2():
    L = np.pi
    N = 200
    b = np.array([2 * (-1) ** (n + 1) / n for n in range(1, N + 1)])
    x = 0.5
    S = fourier_series(x, 0.0, np.zeros(N), b, L)
    assert abs(S - x) < 5e-3


def test_closed_form():
    assert closed_form_square_wave(1) == 4 / sp.pi
    assert closed_form_square_wave(2) == 0
    assert closed_form_square_wave(3) == 4 / (3 * sp.pi)