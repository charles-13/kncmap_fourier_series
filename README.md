# fourierseries

Fourier series on `[-L, L]` (period `2L`) — numeric, symbolic, and plotting.

## Install (from a local clone)
    pip install -e .

## Quick start
    import numpy as np
    from fourierseries import fourier_coefficients, fourier_series

    f = lambda x: np.sign(np.sin(x))
    a0, a, b = fourier_coefficients(f, L=np.pi, N=9)
    S = fourier_series(np.linspace(-np.pi, np.pi, 100), a0, a, b, L=np.pi)

See `examples/square_wave.py` for a full demo.