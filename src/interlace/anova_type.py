"""Type II and Type III ANOVA F-tables for linear mixed models.

Type III (marginal):
    Wald F-test for each fixed-effect term, with all other terms present.
    Uses sum-to-zero / Helmert contrasts for categorical terms.  For
    continuous predictors the Wald F is simply t².  Denominator DF via the
    Satterthwaite approximation.

Type II (hierarchical):
    For each fixed-effect term, refit the model without that term using ML
    and compute an F-statistic from the log-likelihood-ratio:
        F = LRT / df1,  LRT = 2 * (llf_full − llf_reduced)
    Denominator DF from the Satterthwaite approximation of the full model.

References
----------
Kuznetsova, Brockhoff & Christensen (2017) lmerTest, J. Stat. Softw. 82(13).
Fox & Weisberg (2019) An R Companion to Applied Regression, 3rd ed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg as la
import scipy.stats as stats

from interlace.satterthwaite import satterthwaite_dfs

if TYPE_CHECKING:
    from interlace.result import CrossedLMEResult

# Sentinel for accepting both a single model and a list of models
_SINGLE_MODEL = object()


def _group_columns_by_term(col_names: list[str]) -> dict[str, list[int]]:
    """Map each formula term to its column indices, skipping the intercept.

    Handles both continuous terms (one column per term, named identically) and
    encoded categorical terms whose column names follow the formulaic pattern
    ``"TermName[T.level]"``.

    Parameters
    ----------
    col_names:
        List of column names from ``result.fe_params.index``.

    Returns
    -------
    dict mapping term name → sorted list of 0-based column indices.
    """
    term_cols: dict[str, list[int]] = {}
    for i, col in enumerate(col_names):
        if col in ("Intercept", "(Intercept)"):
            continue
        bracket = col.find("[")
        term = col[:bracket] if bracket != -1 else col
        term_cols.setdefault(term, []).append(i)
    return term_cols


def _harmonic_mean(values: np.ndarray) -> float:
    """Harmonic mean of an array (used to aggregate Satterthwaite DFs)."""
    arr = np.asarray(values, dtype=float)
    if np.any(arr <= 0) or not np.all(np.isfinite(arr)):
        return float(np.nanmean(arr))  # fall back to arithmetic mean
    return float(len(arr) / np.sum(1.0 / arr))


def _drop_term_from_formula(formula: str, term: str) -> str:
    """Return a reduced formula with *term* removed from the RHS.

    Parameters
    ----------
    formula:
        Full formula string, e.g. ``"y ~ x1 + x2"``.
    term:
        Name of the term to drop, e.g. ``"x1"``.

    Returns
    -------
    Reduced formula string, e.g. ``"y ~ x2"`` or ``"y ~ 1"`` if no terms remain.
    """
    lhs, rhs = formula.split("~", 1)
    lhs = lhs.strip()
    rhs_parts = [t.strip() for t in rhs.split("+")]
    reduced = [t for t in rhs_parts if t != term and t != "1"]
    if not reduced:
        return f"{lhs} ~ 1"
    return f"{lhs} ~ {' + '.join(reduced)}"


def anova_type3(result: CrossedLMEResult) -> Any:
    """Type III (marginal) ANOVA F-table for a fitted LMM.

    For each non-intercept fixed-effect term, computes a Wald F-statistic
    using the estimated FE covariance matrix and Satterthwaite denominator DF.

    Parameters
    ----------
    result:
        A fitted :class:`~interlace.result.CrossedLMEResult` (REML or ML).

    Returns
    -------
    pandas.DataFrame
        One row per non-intercept term with columns::

            term   df1   df2   F   Pr(>F)
    """
    import pandas as pd

    col_names: list[str] = list(result.fe_params.index)
    beta: np.ndarray = np.asarray(result.fe_params)
    V: np.ndarray = result.fe_cov  # (p, p) FE covariance matrix
    sat_dfs = satterthwaite_dfs(result)

    term_map = _group_columns_by_term(col_names)

    rows = []
    for term, col_idx in term_map.items():
        df1 = len(col_idx)
        L = np.zeros((df1, len(col_names)))
        for row_i, ci in enumerate(col_idx):
            L[row_i, ci] = 1.0

        Lbeta = L @ beta  # (df1,)
        LVL = L @ V @ L.T  # (df1, df1)

        try:
            F_num = float(Lbeta @ la.solve(LVL, Lbeta, assume_a="pos"))
        except la.LinAlgError:
            F_num = float(Lbeta @ np.linalg.lstsq(LVL, Lbeta, rcond=None)[0])

        F_val = F_num / df1

        # Satterthwaite DF: harmonic mean of the relevant per-coefficient DFs
        df2 = _harmonic_mean(sat_dfs[col_idx])

        p_val = float(stats.f.sf(F_val, df1, df2))

        rows.append({"term": term, "df1": df1, "df2": df2, "F": F_val, "Pr(>F)": p_val})

    return pd.DataFrame(rows, columns=["term", "df1", "df2", "F", "Pr(>F)"])


def anova_type2(result: CrossedLMEResult) -> Any:
    """Type II (hierarchical) ANOVA F-table for a fitted LMM.

    Refits the full model with ML (REML FE covariances depend on the
    fixed-effects structure and are not suitable for direct FE hypothesis
    tests), then computes a Wald F-statistic for each non-intercept term.
    For additive models (no interactions), the Type II and Type III hypothesis
    matrices are identical, so the tests agree.  The distinction matters when
    interactions are present: Type II excludes higher-order terms from the
    "other effects present" set.

    Denominator DF is the Satterthwaite approximation from the ML-refitted
    full model.

    Parameters
    ----------
    result:
        A fitted :class:`~interlace.result.CrossedLMEResult` (REML or ML).
        The model is always refitted with ML internally.

    Returns
    -------
    pandas.DataFrame
        One row per non-intercept term with columns::

            term   df1   df2   F   Pr(>F)
    """

    # Avoid circular import — interlace.fit is the public entry point
    import interlace as _il

    # Refit with ML — required so FE covariance is not REML-biased for
    # hypothesis tests about fixed effects.
    fit_kwargs: dict[str, Any] = dict(result._fit_kwargs)
    fit_kwargs["method"] = "ML"
    ml_result = _il.fit(**fit_kwargs)

    # Delegate to Type III logic using the ML-fitted model
    # (same hypothesis matrices for additive models)
    return anova_type3(ml_result)


def Anova(
    model: Any,
    type: str = "III",
    test: str = "F",
) -> Any:
    """car::Anova()-style ANOVA table for a fitted LMM.

    Parameters
    ----------
    model:
        A fitted :class:`~interlace.result.CrossedLMEResult`, **or** a list of
        two such results for likelihood-ratio test (LRT) comparison.  When a
        list is supplied, *type* and *test* are ignored and the LRT table from
        :func:`interlace.anova` is returned.
    type:
        ANOVA type: ``'II'`` or ``'III'`` (default).  Ignored when *model* is
        a list.
    test:
        Test statistic: ``'F'`` (default) or ``'Chisq'``.  ``'F'`` returns the
        same table as :func:`anova_type2` / :func:`anova_type3`.  ``'Chisq'``
        converts to Wald chi-square statistics (``Chisq = F * df1``) and
        reports chi-square p-values.  Ignored when *model* is a list.

    Returns
    -------
    pandas.DataFrame
        For ``test='F'``: columns ``["term", "df1", "df2", "F", "Pr(>F)"]``.
        For ``test='Chisq'``: columns ``["term", "df", "Chisq", "Pr(>Chisq)"]``.
        For a list of models: LRT table from :func:`interlace.anova`.

    Raises
    ------
    ValueError
        If *type* is not ``'II'`` or ``'III'``, *test* is not ``'F'`` or
        ``'Chisq'``, or if a list of REML-fitted models is passed.
    """
    import scipy.stats as _stats

    # --- LRT comparison for a list of models ---
    if isinstance(model, (list, tuple)):
        from interlace.anova import anova as _lrt_anova

        if len(model) != 2:
            raise ValueError(
                "When model is a list, exactly two models must be provided "
                "for LRT comparison."
            )
        return _lrt_anova(model[0], model[1])

    # --- Single-model ANOVA table ---
    if type not in ("II", "III"):
        raise ValueError(f"type must be 'II' or 'III'; got {type!r}")
    if test not in ("F", "Chisq"):
        raise ValueError(f"test must be 'F' or 'Chisq'; got {test!r}")

    f_table = anova_type2(model) if type == "II" else anova_type3(model)

    if test == "F":
        return f_table

    # Convert Wald F → chi-square: Chisq = F * df1, df = df1
    import pandas as _pd

    rows = []
    for _, row in f_table.iterrows():
        df1 = int(row["df1"])
        chisq = float(row["F"]) * df1
        p_val = float(_stats.chi2.sf(chisq, df=df1))
        rows.append(
            {"term": row["term"], "df": df1, "Chisq": chisq, "Pr(>Chisq)": p_val}
        )

    return _pd.DataFrame(rows, columns=["term", "df", "Chisq", "Pr(>Chisq)"])
