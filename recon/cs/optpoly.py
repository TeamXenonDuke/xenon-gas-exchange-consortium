"""Polynomial optimization for compressed sensing."""
import logging

import numpy as np
import sympy

from recon.cs import polynomial as chebpoly
from recon.cs import sigpy as sp


def l_inf_opt(degree: int, l: float = 0.0, L: float = 1.0, verbose: bool = True):
    """Calculate polynomial p(x) that minimizes the supremum of |1 - x p(x)| over(l, L).

    Example:
      (coeffs, polyexpr) = l_inf_opt(degree, l=0, L=1, verbose=True)

    Based on Equation 50 of:
       Shewchuk, J. R.
       An introduction to the conjugate gradient method without the agonizing
       pain, Edition 1¼.

    Uses the following package:
      https://github.com/mlazaric/Chebyshev/
      DOI: 10.5281/zenodo.5831845

    Inputs:
      degree (Int): Degree of polynomial to calculate.
      l (Float): Lower bound of interval.
      L (Float): Upper bound of interval.
      verbose (Bool): log information.

    Returns:
      coeffs (Array): Coefficients of optimized polynomial.
      polyexpr (SymPy): Resulting polynomial as a SymPy expression.
    """
    assert degree >= 0

    if verbose:
        logging.info("L-infinity optimized polynomial.")
        logging.info("> Degree:   %d" % degree)
        logging.info("> Spectrum: [%0.2f, %0.2f]" % (l, L))

    T = chebpoly.get_nth_chebyshev_polynomial(degree + 1)

    y = sympy.symbols("y")
    P = T((L + l - 2 * y) / (L - l))
    P = P / P.subs(y, 0)
    P = sympy.simplify((1 - P) / y)

    if verbose:
        logging.info("> Resulting polynomial: %s" % repr(P))

    if degree > 0:
        points = sympy.stationary_points(P, y, sympy.Interval(l, L))
        vals = np.array(
            [P.subs(y, point) for point in points] + [P.subs(y, l)] + [P.subs(y, L)]  # type: ignore
        )
        assert np.abs(vals).min() > 1e-8, "Polynomial not injective."

    c = sympy.Poly(P).all_coeffs()[::-1] if degree > 0 else (sympy.Float(P),)
    return (np.array(c, dtype=np.float32), P)


def l_2_opt(degree: int, l: float = 0.0, L: float = 1, weight=1, verbose: bool = True):
    r"""Calculate polynomial p(x).

    Calculate polynomial p(x) that minimizes the following    ..math:
      \int_l^l w(x) (1 - x p(x))^2 dx

    Example:
      (coeffs, polyexpr) = l_2_opt(degree, l=0, L=1, verbose=True)

    To incorporate priors, w(x) can be used to weight regions of the
    interval (l, L) of the expression above.

    Based on:
      Polynomial Preconditioners for Conjugate Gradient Calculations
      Olin G. Johnson, Charles A. Micchelli, and George Paul
      DOI: 10.1137/0720025

    Inputs:
      degree (Int): Degree of polynomial to calculate.
      l (Float): Lower bound of interval.
      L (Float): Upper bound of interval.
      weight (SymPy): Sympy expression to include prior weight.
      verbose (Bool): logging.info information.

    Returns:
      coeffs (Array): Coefficients of optimized polynomial.
      polyexpr (SymPy): Resulting polynomial as a SymPy expression.
    """
    if verbose:
        logging.info("L-2 optimized polynomial.")
        logging.info("> Degree:   %d" % degree)
        logging.info("> Spectrum: [%0.2f, %0.2f]" % (l, L))

    c = sympy.symbols("c0:%d" % (degree + 1))
    x = sympy.symbols("x")

    p = sum([(c[k] * x**k) for k in range(degree + 1)])
    f = weight * (1 - x * p) ** 2
    J = sympy.integrate(f, (x, l, L))

    mat = [[0.0] * (degree + 1) for _ in range(degree + 1)]
    vec = [0.0] * (degree + 1)

    for edx in range(degree + 1):
        edx = int(edx)
        eqn = sympy.diff(J, c[edx])
        tmp = eqn.copy()
        # Coefficient index
        for cdx in range(degree + 1):
            mat[edx][cdx] = float(sympy.Poly(eqn, c[cdx]).coeffs()[0])
            tmp = tmp.subs(c[cdx], 0)
        vec[edx] = float(-tmp)

    mat = np.array(mat, dtype=np.double)
    vec = np.array(vec, dtype=np.double)
    res = np.array(np.linalg.pinv(mat) @ vec, dtype=np.float32)

    poly = sum([(res[k] * x**k) for k in range(degree + 1)])
    if verbose:
        logging.info("> Resulting polynomial: %s" % repr(poly))

    if degree > 0:
        points = sympy.stationary_points(poly, x, sympy.Interval(l, L))
        vals = np.array(
            [poly.subs(x, point) for point in points]
            + [poly.subs(x, l)]
            + [poly.subs(x, L)]
        )
        assert vals.min() > 1e-8, "Polynomial is not positive."

    return (res, poly)


def create_polynomial_preconditioner(
    degree: int,
    T: sp.linop.Linop,
    l: float = 0,
    L: float = 1,
    norm: str = "l_2",
    verbose: bool = False,
):
    r"""
    P = create_polynomial_preconditioner(degree, T, l, L, verbose=False)

    Inputs:
      degree (Int): Degree of polynomial to use.
      T (Linop): Normal linear operator.
      l (Float): Smallest eigenvalue of T. If not known, assumed to be zero.
      L (Float): Largest eigenvalue of T. If not known, assumed to be one.
      norm (String): Norm to optimize. Currently only supports "l_2" and
                     "l_inf".
      verbose (Bool): logging.info information.

    Returns:
      P (Linop): Polynomial preconditioner.
    """
    assert degree >= 0

    if norm == "l_2":
        (c, _) = l_2_opt(degree, l, L, verbose=verbose)
    elif norm == "l_inf":
        (c, _) = l_inf_opt(degree, l, L, verbose=verbose)
    else:
        raise Exception("Unknown norm option.")

    I = sp.linop.Identity(T.ishape)

    def phelper(c):
        if c.size == 1:
            return c[0] * I
        return c[0] * I + T * phelper(c[1:])

    P = phelper(c)
    return P
