"""Plot Fourier approximation vs true function."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .series import fourier_series


def plot_fourier(f, a0, a, b, L, N=None, n_points=2000, save_path=None, show=False):
    """
    Plot f and its Fourier partial sum on [-L, L].

    Parameters
    ----------
    f : callable
    a0, a, b : coefficients (as returned by fourier_coefficients)
    L : float
    N : int, optional
        Only used for the legend label.
    save_path : str or Path, optional
        If given, saves the figure there (works headless).
    show : bool, default False
        If True, calls plt.show().

    Returns
    -------
    (fig, ax)
    """
    x_vals = np.linspace(-L, L, n_points)
    f_vals = np.array([f(x) for x in x_vals])
    S_vals = fourier_series(x_vals, a0, a, b, L)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_vals, f_vals, "k--", lw=1.2, label="f(x)")
    label = f"S_{N}(x)" if N is not None else "S(x)"
    ax.plot(x_vals, S_vals, "r", lw=1.5, label=label)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title("Fourier series on [-L, L]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax