"""Exact symbolic Fourier coefficients via sympy."""
import sympy as sp
import numpy as np


# ---------- helpers ----------

def _find_sign_atoms(expr):
    """Return list of sign(...) / Abs(...) atoms in expr."""
    found = []
    for atom in sp.preorder_traversal(expr):
        if getattr(atom, "func", None) in (sp.sign, sp.Abs):
            found.append(atom)
    return found


def _breakpoints(expr, x, L):
    """
    Return sorted points in (-L, L) where any sign(g)/Abs(g) inside expr
    can flip — i.e. roots of every g.
    """
    Lf = float(L)
    pts = set()

    for atom in _find_sign_atoms(expr):
        g = atom.args[0]
        try:
            roots = sp.solve(sp.Eq(g, 0), x)
        except Exception:
            roots = []
        for r in roots:
            try:
                rv = complex(r.evalf())
            except Exception:
                continue
            if abs(rv.imag) < 1e-12 and abs(rv.real) < Lf:
                pts.add(round(rv.real, 12))

    # also include sympy's own singularities (for non-sign discontinuities)
    try:
        for p in sp.calculus.util.singularities(expr, x):
            try:
                pv = complex(p.evalf())
            except Exception:
                continue
            if abs(pv.imag) < 1e-12 and abs(pv.real) < Lf:
                pts.add(round(pv.real, 12))
    except Exception:
        pass

    return sorted(p for p in pts if -Lf < p < Lf)


def _substitute_signs_on_interval(expr, x, lo, hi):
    """
    Replace every sign(g)/Abs(g) in expr by a literal +1/-1/(g)/(-g)
    according to the sign of g at the midpoint of (lo, hi).

    This produces a sign-free expression on the open interval (lo, hi).
    """
    mid = sp.Rational(1, 2) * (lo + hi)

    def val(e):
        try:
            return float(e.subs(x, mid).evalf())
        except Exception:
            return 0.0

    out = expr
    for atom in _find_sign_atoms(expr):
        g = atom.args[0]
        v = val(g)
        if atom.func is sp.sign:
            repl = sp.Integer(1) if v > 0 else (sp.Integer(-1) if v < 0 else sp.Integer(0))
        else:  # Abs
            repl = g if v >= 0 else -g
        out = out.replace(atom, repl)
    return out


def _integrate_piecewise(f_sym, x, L):
    """
    Integrate f_sym over [-L, L] by splitting at every breakpoint of
    any sign(...)/Abs(...) inside it.  On each sub-interval the sign is
    constant, so we substitute it explicitly and integrate symbolically.
    """
    breaks = _breakpoints(f_sym, x, L)
    edges = [sp.sympify(-L)] + [sp.sympify(b) for b in breaks] + [sp.sympify(L)]

    total = sp.Integer(0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        piece = _substitute_signs_on_interval(f_sym, x, lo, hi)
        total += sp.integrate(piece, (x, lo, hi))
    return sp.simplify(total)


# ---------- public API ----------

def symbolic_coefficients(f_sym, x, L, N):
    """
    Exact a0, a_n, b_n for a sympy expression f_sym in x, on [-L, L].

    Works for piecewise functions containing sign(...) and Abs(...) by
    splitting the integration interval at their breakpoints and
    substituting the (constant) sign value on each sub-interval.
    """
    L_sym = sp.sympify(L)

    a0 = sp.simplify((sp.Integer(1) / L_sym) * _integrate_piecewise(f_sym, x, L_sym))
    a, b = [], []
    for n in range(1, N + 1):
        a.append(sp.simplify(
            (sp.Integer(1) / L_sym)
            * _integrate_piecewise(f_sym * sp.cos(n * sp.pi * x / L_sym), x, L_sym)
        ))
        b.append(sp.simplify(
            (sp.Integer(1) / L_sym)
            * _integrate_piecewise(f_sym * sp.sin(n * sp.pi * x / L_sym), x, L_sym)
        ))
    return a0, a, b


def closed_form_square_wave(n):
    """
    Closed form for the sign(sin x) square wave on [-pi, pi]:
        b_n = 4/(n*pi) for odd n, 0 for even n;  a_n = 0;  a0 = 0.
    """
    n = int(n)
    if n < 1:
        raise ValueError("n >= 1")
    if n % 2 == 0:
        return sp.Integer(0)
    return sp.Integer(4) / (n * sp.pi)