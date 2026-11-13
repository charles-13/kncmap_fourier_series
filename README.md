# fourierseries

[![tests](https://github.com/charles-13/kncmap_fourier_series/actions/workflows/test.yml/badge.svg)](https://github.com/charles-13/kncmap_fourier_series/actions/workflows/test.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Fourier series on the interval $[-L, L]$ (period $2L$), with **numeric**, **exact symbolic**, and **plotting** tools.

$$
f(x) \sim \frac{a_0}{2} + \sum_{n=1}^{\infty}\left(a_n\cos\frac{n\pi x}{L} + b_n\sin\frac{n\pi x}{L}\right)
$$

with

$$
a_0 = \frac{1}{L}\int_{-L}^{L} f(x)\,dx,\qquad
a_n = \frac{1}{L}\int_{-L}^{L} f(x)\cos\frac{n\pi x}{L}\,dx,\qquad
b_n = \frac{1}{L}\int_{-L}^{L} f(x)\sin\frac{n\pi x}{L}\,dx.
$$

## Install

```bash
pip install -e ".[dev]"       # from a clone, for development
```

(PyPI release coming soon.)

## Quick start

```python
import numpy as np
import sympy as sp
from fourierseries import (
    fourier_coefficients,
    symbolic_coefficients,
    fourier_series,
    plot_fourier,
)

# ----- numeric coefficients -----
f = lambda x: np.sign(np.sin(x))
a0, a, b = fourier_coefficients(f, L=np.pi, N=9)

# ----- exact symbolic coefficients -----
x = sp.symbols("x", real=True)
a0, a, b = symbolic_coefficients(sp.sign(sp.sin(x)), x, sp.pi, 5)
# a0 = 0,  b = [4/pi, 0, 4/(3*pi), 0, 4/(5*pi)]

# ----- evaluate -----
S = fourier_series(np.linspace(-np.pi, np.pi, 100), a0, a, b, L=np.pi)

# ----- plot -----
plot_fourier(f, a0, a, b, L=np.pi, N=9, save_path="square_wave.png")
```

## What's in the box

| Module | Purpose |
|---|---|
| `coefficients.py` | Numeric coefficients via `scipy.integrate.quad` — works for any integrable function |
| `symbolic.py` | Exact symbolic coefficients via `sympy`, including piecewise functions like `sign(sin x)` and `Abs(x)` |
| `series.py` | Evaluate the partial Fourier sum at arbitrary points |
| `plotting.py` | Plot the partial sum against the true function |

## Why a symbolic backend matters

Sympy's `integrate` gives wrong answers for `sign(sin(x))` on `[-π, π]` (it returns `π` instead of `0` for the total integral). This package detects the breakpoints of `sign`/`Abs` internally, substitutes the constant sign on each sub-interval, and hands sympy only sign-free integrands — which it handles correctly.

## Development

```bash
git clone https://github.com/charles-13/kncmap_fourier_series.git
cd kncmap_fourier_series
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

Tests run against Python 3.10–3.13 on every push via GitHub Actions.

## License

MIT — see [LICENSE](LICENSE).