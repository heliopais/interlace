"""Formula parsing for interlace — statsmodels-compatible API.

Uses statsmodels FormulaManager (which wraps formulaic) to build the
fixed-effects design matrix from a standard patsy-syntax formula string.
Groups are a separate parameter, mirroring MixedLM.from_formula().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy


@dataclass
class ParsedFormula:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    term_names: list[str]


def extract_group_factors(
    data: pd.DataFrame,
    group_cols: list[str],
) -> list[tuple[str, np.ndarray, int]]:
    """Extract grouping factor integer codes from a DataFrame.

    Uses ``pd.factorize`` (the same mechanism as statsmodels internally) to
    convert each group column into sorted integer codes.

    Parameters
    ----------
    data:
        DataFrame containing the grouping columns.
    group_cols:
        Names of columns to extract as grouping factors.

    Returns
    -------
    list of (name, codes, n_levels) tuples
        ``codes`` is a 1-D integer array of length ``n_obs``;
        ``n_levels`` is the number of unique levels.
    """
    result = []
    for col in group_cols:
        codes, uniques = pd.factorize(data[col], sort=True)
        result.append((col, codes.astype(np.intp), len(uniques)))
    return result


def parse_formula(
    formula: str,
    data: pd.DataFrame,
    groups: str | np.ndarray,
) -> ParsedFormula:
    """Parse a fixed-effects formula and extract response + groups.

    Parameters
    ----------
    formula:
        Patsy-syntax formula string, e.g. ``"y ~ x1 + x2"``.
    data:
        DataFrame containing all variables referenced in *formula* and *groups*.
    groups:
        Column name (str) or array of group labels for the random intercept.

    Returns
    -------
    ParsedFormula
        Dataclass with ``X``, ``y``, ``groups``, and ``term_names``.
    """
    # Validate groups
    if isinstance(groups, str):
        if groups not in data.columns:
            raise ValueError(
                f"groups column '{groups}' not found in data "
                f"(available: {list(data.columns)})"
            )
        groups_arr: np.ndarray = np.asarray(data[groups])
    else:
        groups_arr = np.asarray(groups)

    # Use patsy.dmatrices (the same call statsmodels MixedLM.from_formula uses)
    endog_dm, exog_dm = patsy.dmatrices(formula, data, return_type="dataframe")

    term_names = list(exog_dm.columns)

    return ParsedFormula(
        X=np.asarray(exog_dm),
        y=np.asarray(endog_dm).squeeze(),
        groups=groups_arr,
        term_names=term_names,
    )
