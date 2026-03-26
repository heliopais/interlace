"""Formula parsing for interlace — statsmodels-compatible API.

Uses statsmodels FormulaManager (which wraps formulaic) to build the
fixed-effects design matrix from a standard patsy-syntax formula string.
Groups are a separate parameter, mirroring MixedLM.from_formula().
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import patsy


@dataclass
class RandomEffectSpec:
    """Specification for a single random effect term.

    Parameters
    ----------
    group:
        Column name of the grouping factor.
    predictors:
        Names of slope predictor columns (empty for intercept-only).
    intercept:
        Whether a random intercept is included.
    correlated:
        True = full Cholesky (correlated intercept/slopes);
        False = diagonal (independent, ``||`` syntax).
    """

    group: str
    predictors: list[str] = field(default_factory=list)
    intercept: bool = True
    correlated: bool = True

    @property
    def n_terms(self) -> int:
        """Total number of random effect columns per group level."""
        return int(self.intercept) + len(self.predictors)


# Regex: optional outer parens, effects side, pipe (single or double), group name
_RE_SPEC = re.compile(
    r"^\(\s*(?P<effects>.+?)\s*(?P<pipe>\|\|?)\s*(?P<group>\w+)\s*\)$"
)


def parse_random_effects(random: list[str]) -> list[RandomEffectSpec]:
    """Parse a list of lme4-style random effect strings into RandomEffectSpec objects.

    Supported syntax examples::

        "(1 | g)"           # random intercept
        "(1 + x | g)"       # correlated random intercept + slope
        "(1 + x || g)"      # independent random intercept + slope
        "(0 + x | g)"       # random slope only (no intercept)
        "(x | g)"           # random slope only (implicit no intercept)

    Parameters
    ----------
    random:
        List of lme4-style random effect specification strings.

    Returns
    -------
    list[RandomEffectSpec]
    """
    specs = []
    for s in random:
        m = _RE_SPEC.match(s.strip())
        if m is None:
            raise ValueError(
                f"Invalid random effect specification {s!r}. "
                "Expected lme4-style syntax like '(1 + x | g)' or '(1 + x || g)'."
            )
        effects_str = m.group("effects")
        group = m.group("group")
        correlated = m.group("pipe") == "|"

        terms = [t.strip() for t in effects_str.split("+")]

        intercept = False
        predictors: list[str] = []
        for term in terms:
            if term == "1":
                intercept = True
            elif term == "0":
                pass  # explicit suppression of intercept — leave intercept=False
            else:
                predictors.append(term)

        # If no explicit 1 or 0, intercept defaults to False (slope-only)
        specs.append(
            RandomEffectSpec(
                group=group,
                predictors=predictors,
                intercept=intercept,
                correlated=correlated,
            )
        )
    return specs


def groups_to_random_effects(groups: str | list[str]) -> list[RandomEffectSpec]:
    """Convert the legacy ``groups`` parameter to a list of RandomEffectSpec.

    Produces intercept-only, correlated specs — equivalent to ``(1 | g)`` for
    each group name.

    Parameters
    ----------
    groups:
        Single group column name or list of group column names.
    """
    if isinstance(groups, str):
        groups = [groups]
    return [
        RandomEffectSpec(group=g, predictors=[], intercept=True, correlated=True)
        for g in groups
    ]


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
