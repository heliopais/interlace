"""Sparse Z matrix construction for crossed random intercepts.

Builds per-factor indicator matrices as scipy.sparse.csc_matrix and
horizontally stacks them into the joint random-effects design matrix Z.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_indicator_matrix(codes: np.ndarray, n_levels: int) -> sp.csc_matrix:
    """Build a sparse (n_obs x n_levels) indicator matrix for one grouping factor.

    Each row has exactly one 1.0 in the column corresponding to its group code.

    Parameters
    ----------
    codes:
        Integer array of group codes, shape (n_obs,). Values in [0, n_levels).
    n_levels:
        Number of unique levels (number of columns in the result).

    Returns
    -------
    scipy.sparse.csc_matrix of shape (n_obs, n_levels).
    """
    n_obs = len(codes)
    rows = np.arange(n_obs)
    data = np.ones(n_obs)
    return sp.csc_matrix((data, (rows, codes)), shape=(n_obs, n_levels))


def build_joint_z(
    factors: list[tuple[str, np.ndarray, int]],
) -> sp.csc_matrix:
    """Horizontally stack per-factor indicator matrices into the joint Z.

    Parameters
    ----------
    factors:
        List of ``(name, codes, n_levels)`` tuples as returned by
        :func:`interlace.formula.extract_group_factors`.

    Returns
    -------
    scipy.sparse.csc_matrix of shape (n_obs, sum(n_levels_j)).
    """
    blocks = [build_indicator_matrix(codes, n_levels) for _, codes, n_levels in factors]
    return sp.hstack(blocks, format="csc")
