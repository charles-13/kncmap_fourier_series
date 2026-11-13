"""
Fourier series on [-L, L]  (period 2L)

    f(x) ~ a0/2 + sum_{n>=1} [ a_n cos(n*pi*x/L) + b_n sin(n*pi*x/L) ]

    a0 = (1/L) * ∫_{-L}^{L} f(x) dx
    a_n = (1/L) * ∫_{-L}^{L} f(x) cos(n*pi*x/L) dx
    b_n = (1/L) * ∫_{-L}^{L} f(x) sin(n*pi*x/L) dx
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; saves file instead of popping a window
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp


# ---------- numeric engine (works for ANY integrable f) ----------
def fourier_coefficients(f, L, N, eps=1e-12):
    """Numerical a0, a_n, b_n with denoising."""
    def clean(v):
        return 0.0 if abs(v) < eps else v

    a0 = clean((1 / L) * quad(f, -L, L, limit=200)[0])
    a = np.zeros(N); b = np.zeros(N)
    for n in range(1, N + 1):
        a[n - 1] = clean((1 / L) * quad(lambda x: f(x) * np.cos(n*np.pi*x/L), -L, L, limit=200)[0])
        b[n - 1] = clean((1 / L) * quad(lambda x: f(x) * np.sin(n*np.pi*x/L), -L, L, limit=200)[0])
    return a0, a, b


def fourier_series(x, a0, a, b, L):
    x = np.asarray(x, dtype=float)
    s = np.full_like(x, a0 / 2.0)
    for n in range(1, len(a) + 1):
        s += a[n-1]*np.cos(n*np.pi*x/L) + b[n-1]*np.sin(n*np.pi*x/L)
    return s


# ---------- symbolic engine (exact coefficients when f is smooth/piecewise) ----------
def symbolic_coefficients(f_sym, x, L, N):
    """
    Exact a0, a_n, b_n via sympy.
    f_sym : sympy expression in x
    """
    a0 = sp.simplify((1/L) * sp.integrate(f_sym, (x, -L, L)))
    a, b = [], []
    for n in range(1, N + 1):
        a_n = sp.simplify((1/L) * sp.integrate(f_sym*sp.cos(n*sp.pi*x/L), (x, -L, L)))
        b_n = sp.simplify((1/L) * sp.integrate(f_sym*sp.sin(n*sp.pi*x/L), (x, -L, L)))
        a.append(a_n); b.append(b_n)
    return a0, a, b


# ---------- pretty printing ----------
def pretty(a0, a, b, N, L_str="L"):
    print(f"a0 = {a0}")
    print("\n n |        a_n        |        b_n")
    print("-" * 40)
    for n in range(1, N + 1):
        print(f"{n:2d} | {str(a[n-1]):>17} | {str(b[n-1]):>17}")


# ==========================================================
# EXAMPLE: square wave  f(x) = sign(sin x),  L = pi,  N = 9
# ==========================================================
if __name__ == "__main__":
    L = np.pi
    N = 9                          # small N -> human-readable table
    L_str = r"\pi"

    # --- numeric ---
    f = lambda x: np.sign(np.sin(x))
    a0, a, b = fourier_coefficients(f, L, N)

    print("=" * 50)
    print("NUMERIC coefficients (square wave on [-pi, pi])")
    print("=" * 50)
    pretty(a0, a, b, N)

    # --- symbolic (exact) ---
    x = sp.symbols("x", real=True)
    f_sym = sp.sign(sp.sin(x))
    a0s, a_s, b_s = symbolic_coefficients(f_sym, x, sp.pi, N)

    print("\n" + "=" * 50)
    print("EXACT (sympy) coefficients")
    print("=" * 50)
    pretty(a0s, a_s, b_s, N)

    # A mathematician's closed form, computed & displayed:
    print("\nClosed form for odd n:  b_n = 4/(n*pi)")
    for n in range(1, N + 1):
        exact = 0 if n % 2 == 0 else sp.Rational(4, n * sp.pi)
        print(f"  n={n}:  b_n = {exact}   (numeric {b[n-1]: .6f})")

    # --- plot & save ---
    x_vals = np.linspace(-L, L, 2000)
    f_vals = f(x_vals)
    S_vals = fourier_series(x_vals, a0, a, b, L)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_vals, f_vals, "k--", lw=1.2, label=r"$f(x)=\mathrm{sgn}(\sin x)$")
    ax.plot(x_vals, S_vals, "r", lw=1.5, label=rf"$S_{{{N}}}(x)$")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(rf"Fourier series on $[-{L_str},{L_str}]$, $N={N}$")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fourier_plot.png", dpi=150)
    print("\nSaved figure -> fourier_plot.png")