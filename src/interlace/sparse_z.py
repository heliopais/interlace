"""Sparse Z matrix construction for crossed random effects.

Builds per-factor indicator (and slope) matrices as scipy.sparse.csc_matrix
and horizontally stacks them into the joint random-effects design matrix Z.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp

if TYPE_CHECKING:
    from interlace.formula import RandomEffectSpec


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


def build_z_block(
    spec: RandomEffectSpec,
    data: pd.DataFrame,
    codes: np.ndarray,
    n_levels: int,
) -> sp.csc_matrix:
    """Build the Z block for a single RandomEffectSpec.

    Column ordering follows the Bates lme4 convention: all ``n_levels``
    intercept columns first (if the intercept term is present), then all
    ``n_levels`` columns for each slope predictor in order.

    Parameters
    ----------
    spec:
        Random effect specification (group, predictors, intercept, correlated).
    data:
        DataFrame containing the slope predictor columns.
    codes:
        Integer group codes (0-indexed), shape (n_obs,).
    n_levels:
        Number of unique levels for this grouping factor.

    Returns
    -------
    scipy.sparse.csc_matrix of shape (n_obs, spec.n_terms * n_levels).
    """
    n_obs = len(codes)
    rows = np.arange(n_obs)
    col_blocks: list[sp.csc_matrix] = []

    if spec.intercept:
        col_blocks.append(build_indicator_matrix(codes, n_levels))

    for pred in spec.predictors:
        x = np.asarray(data[pred], dtype=float)
        Z_slope = sp.csc_matrix((x, (rows, codes)), shape=(n_obs, n_levels))
        col_blocks.append(Z_slope)

    return sp.hstack(col_blocks, format="csc")


def build_joint_z_from_specs(
    specs: list[RandomEffectSpec],
    data: pd.DataFrame,
) -> sp.csc_matrix:
    """Build the joint Z matrix from a list of RandomEffectSpec objects.

    For each spec, factorizes the group column, builds a Z block via
    :func:`build_z_block`, and horizontally stacks all blocks.

    Parameters
    ----------
    specs:
        List of random effect specifications.
    data:
        DataFrame containing all group and predictor columns.

    Returns
    -------
    scipy.sparse.csc_matrix of shape (n_obs, sum(spec.n_terms * n_levels_j)).
    """
    col_blocks: list[sp.csc_matrix] = []
    for spec in specs:
        codes, uniques = pd.factorize(data[spec.group], sort=True)
        codes = codes.astype(np.intp)
        n_levels = len(uniques)
        col_blocks.append(build_z_block(spec, data, codes, n_levels))
    return sp.hstack(col_blocks, format="csc")
