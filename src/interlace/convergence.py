"""Convergence and boundary diagnostics for fitted LME models.

Provides :func:`isSingular` which detects whether a fitted model is at the
boundary of the parameter space — i.e., any variance component has collapsed
to zero.  This mirrors ``lme4::isSingular()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from interlace.profiled_reml import n_theta_for_spec

if TYPE_CHECKING:
    from interlace.result import CrossedLMEResult


def _diagonal_positions(p_j: int) -> list[int]:
    """Return theta-vector indices corresponding to diagonal entries of L_j.

    For a p_j×p_j lower-triangular Cholesky factor stored in row-major order,
    the diagonal entry L_j[k, k] sits at index ``k*(k+1)//2 + k``.
    """
    return [k * (k + 1) // 2 + k for k in range(p_j)]


def _spec_is_singular(
    theta_j: np.ndarray, n_terms: int, correlated: bool, tol: float
) -> bool:
    """Return True if this spec's theta slice is at the boundary."""
    if n_terms == 1:
        return bool(abs(theta_j[0]) < tol)
    if correlated:
        diag_pos = _diagonal_positions(n_terms)
        return any(abs(theta_j[pos]) < tol for pos in diag_pos)
    # independent (||): each theta_j[k] is a standard deviation
    return bool(np.any(np.abs(theta_j) < tol))


def isSingular(result: CrossedLMEResult, tol: float = 1e-4) -> bool:
    """Return True if the model is at or near the boundary of the parameter space.

    A model is singular when one or more variance components have collapsed to
    zero — the corresponding diagonal entry of the relative covariance factor
    Lambda_theta is less than *tol*.  This matches the behaviour of
    ``lme4::isSingular()``.

    Parameters
    ----------
    result:
        A fitted :class:`~interlace.result.CrossedLMEResult`.
    tol:
        Tolerance for declaring a diagonal entry "effectively zero".
        Defaults to ``1e-4``, matching lme4.

    Returns
    -------
    bool
        ``True`` if any variance component is at the boundary.
    """
    theta = result.theta
    theta_idx = 0
    for spec in result._random_specs:
        n_terms: int = spec.n_terms
        correlated: bool = spec.correlated
        n_theta_j = n_theta_for_spec(n_terms, correlated)
        theta_j = theta[theta_idx : theta_idx + n_theta_j]
        theta_idx += n_theta_j
        if _spec_is_singular(theta_j, n_terms, correlated, tol):
            return True
    return False
