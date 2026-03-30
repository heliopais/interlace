"""emmeans(): estimated marginal means for linear mixed models.

Mirrors R's emmeans package: for each level combination of the *specs*
factor(s), compute the marginal mean of the fixed-effects predictor after
averaging continuous covariates at their observed means.  Standard errors and
degrees of freedom use the Satterthwaite approximation for the linear
combination c'β.

Reference: Lenth (2016) "Least-Squares Means: The R Package lsmeans",
J. Stat. Softw. 69(1).
"""

from __future__ import annotations

from itertools import combinations, product
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import scipy.stats as stats

if TYPE_CHECKING:
    from interlace.result import CrossedLMEResult


class EmmResult(pd.DataFrame):
    """DataFrame subclass returned by :func:`emmeans`.

    Fully backwards compatible with pandas DataFrame.  Carries extra
    attributes used by :func:`contrast`:

    Attributes
    ----------
    _emm_model : CrossedLMEResult
        The fitted model used to build the EMMs.
    _emm_L : np.ndarray
        Contrast matrix (n_cells × p) — each row is the linear combination
        of fixed-effect coefficients that gives one EMM.
    _emm_specs : list[str]
        The specs factor column names.
    """

    # Required by pandas to preserve subclass through operations
    _metadata = ["_emm_model", "_emm_L", "_emm_specs"]

    _emm_model: Any
    _emm_L: np.ndarray
    _emm_specs: list[str]

    @property
    def _constructor(self) -> type[EmmResult]:
        return EmmResult


def emmeans(
    model: CrossedLMEResult,
    specs: str | list[str],
    at: dict[str, Any] | None = None,
    level: float = 0.95,
) -> Any:
    """Compute estimated marginal means (EMMs) for a fitted LMM.

    Parameters
    ----------
    model:
        A fitted :class:`~interlace.result.CrossedLMEResult`.
    specs:
        Name of the factor (str) or list of factor names for which to compute
        marginal means.  Each named column must appear in the fixed-effects
        formula.
    at:
        Optional dict of covariate values to use in the reference grid instead
        of their training-data means.  For example ``at={"x": 0.0}`` centres
        the estimates at ``x = 0``.
    level:
        Nominal CI coverage (default 0.95).

    Returns
    -------
    pandas.DataFrame
        One row per level combination of *specs*, with columns:

        ``<specs columns>``, ``estimate``, ``SE``, ``df``,
        ``lower``, ``upper``, ``t.ratio``, ``p.value``.
    """
    import pandas as _pd

    specs_list = [specs] if isinstance(specs, str) else list(specs)

    # Training data (pandas)
    pd_frame: _pd.DataFrame = model.model.data._pandas_frame

    # Fixed-effects formula RHS
    fe_formula = model.model.formula.split("~", 1)[1].strip()

    # Build the reference grid
    ref_grid = _build_reference_grid(pd_frame, specs_list, at)

    # Build the contrast matrix L (n_cells × p)
    L = _build_contrast_matrix(fe_formula, ref_grid, model.fe_params)

    # Point estimates
    beta = np.asarray(model.fe_params)
    estimates = L @ beta

    # Variances and standard errors
    fe_cov = model.fe_cov
    var_estimates = np.einsum("ij,jk,ik->i", L, fe_cov, L)
    se = np.sqrt(np.maximum(var_estimates, 0.0))

    # Satterthwaite degrees of freedom for each linear combination
    dfs = _satterthwaite_df_contrasts(model, L)

    # t-ratios, p-values, and CIs
    t_ratios = np.where(se > 0, estimates / se, np.nan)
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_ratios), df=dfs))
    z_alpha = stats.t.ppf((1.0 + level) / 2.0, df=dfs)
    lower = estimates - z_alpha * se
    upper = estimates + z_alpha * se

    # Assemble output
    rows: list[dict[str, Any]] = []
    for i, grid_row in enumerate(ref_grid.itertuples(index=False)):
        row: dict[str, Any] = {col: getattr(grid_row, col) for col in specs_list}
        row["estimate"] = float(estimates[i])
        row["SE"] = float(se[i])
        row["df"] = float(dfs[i])
        row["lower"] = float(lower[i])
        row["upper"] = float(upper[i])
        row["t.ratio"] = float(t_ratios[i])
        row["p.value"] = float(p_values[i])
        rows.append(row)

    result = EmmResult(rows)
    result._emm_model = model
    result._emm_L = L
    result._emm_specs = specs_list
    return result


# ---------------------------------------------------------------------------
# Reference grid construction
# ---------------------------------------------------------------------------


def _build_reference_grid(
    pd_frame: Any,
    specs_list: list[str],
    at: dict[str, Any] | None,
) -> Any:
    """Build a reference grid: cross all specs factor levels, others at mean.

    Non-specs numeric columns are set to their training-data mean.
    Non-specs non-numeric columns are set to their most common value (mode).
    *at* overrides any column's default value.
    """
    import pandas as _pd

    # Unique levels for each specs factor (sorted for determinism)
    level_lists = [sorted(pd_frame[s].unique().tolist()) for s in specs_list]

    # Default values for all non-specs columns
    other_defaults: dict[str, Any] = {}
    for col in pd_frame.columns:
        if col in specs_list:
            continue
        try:
            other_defaults[col] = float(pd_frame[col].mean())
        except (TypeError, ValueError):
            other_defaults[col] = pd_frame[col].mode().iloc[0]

    if at is not None:
        other_defaults.update(at)

    # Cross specs levels
    rows: list[dict[str, Any]] = []
    for combo in product(*level_lists):
        row = dict(zip(specs_list, combo, strict=True))
        row.update(other_defaults)
        rows.append(row)

    return _pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Contrast matrix construction
# ---------------------------------------------------------------------------


def _build_contrast_matrix(
    fe_formula: str,
    reference_grid: Any,
    fe_params: Any,
) -> np.ndarray:
    """Evaluate the fixed-effects design matrix on the reference grid.

    Handles column reordering / padding to match the fitting-time fe_params
    order (same approach as :func:`interlace.predict.predict`).
    """
    import formulaic

    mm = formulaic.model_matrix(fe_formula, reference_grid)
    mm_cols = list(mm.columns)
    mm_arr = np.asarray(mm)

    if hasattr(fe_params, "index"):
        fe_cols = list(fe_params.index)
        if mm_cols != fe_cols:
            n_rows = mm_arr.shape[0]
            col_lookup = {c: mm_arr[:, i] for i, c in enumerate(mm_cols)}
            mm_arr = np.column_stack(
                [col_lookup.get(c, np.zeros(n_rows)) for c in fe_cols]
            )

    return mm_arr


# ---------------------------------------------------------------------------
# Satterthwaite DF for arbitrary linear combinations
# ---------------------------------------------------------------------------


def _satterthwaite_df_contrasts(
    model: CrossedLMEResult,
    L: np.ndarray,
) -> np.ndarray:
    """Satterthwaite denominator DFs for a set of linear combinations L @ β.

    For the i-th row c = L[i]:

        V(c) = c' · fe_cov(θ) · c
        ν(c) = V(c)² / [(∂V(c)/∂θ)' · H_D⁻¹ · (∂V(c)/∂θ)]

    where H_D is the Hessian of the REML deviance.  Both the gradient and the
    Hessian are computed by central finite differences (same as
    :func:`interlace.satterthwaite.satterthwaite_dfs`).

    Parameters
    ----------
    model:
        Fitted :class:`~interlace.result.CrossedLMEResult`.
    L:
        Array of shape (n_contrasts, p) — rows are contrast vectors.

    Returns
    -------
    np.ndarray of shape (n_contrasts,)
        Satterthwaite DFs, clipped to a minimum of 1.
    """
    from interlace.profiled_reml import (
        _build_A11,
        _precompute,
        _sparse_solve,
        make_lambda,
        reml_objective,
    )

    y: np.ndarray = model.model.endog
    X: np.ndarray = model.model.exog
    Z: Any = model._Z
    theta_hat: np.ndarray = model.theta
    specs = model._random_specs
    n_levels: list[int] = model._n_levels

    n, p = X.shape
    k = len(theta_hat)
    n_contrasts = L.shape[0]

    cache = _precompute(y, X, Z)
    ZtX = np.asarray(cache["ZtX"])
    Zty = np.asarray(cache["Zty"])
    XtX = np.asarray(cache["XtX"])
    Xty = np.asarray(cache["Xty"])
    yty = float(cache["yty"])

    def _fe_cov_at(theta: np.ndarray) -> np.ndarray:
        """Return fe_cov = σ²(θ) · (X'Ω⁻¹X)⁻¹ as a dense (p×p) array."""
        Lambda = make_lambda(theta, specs, n_levels)
        A11 = _build_A11(sp.csc_matrix(cache["ZtZ"]), Lambda)
        lZtX = np.asarray(Lambda.T @ ZtX)
        lZty = np.asarray(Lambda.T @ Zty).squeeze()
        C_X = _sparse_solve(A11, lZtX)
        c1 = _sparse_solve(A11, lZty)
        MX = XtX - lZtX.T @ C_X
        rhs = Xty - lZtX.T @ c1
        try:
            beta_at = la.solve(MX, rhs, assume_a="pos")
        except la.LinAlgError:
            return np.full((p, p), np.nan)
        yPy = yty - lZty @ c1 - rhs @ beta_at
        if yPy <= 0:
            return np.full((p, p), np.nan)
        sigma2 = yPy / (n - p)
        try:
            MX_inv = np.linalg.inv(MX)
        except np.linalg.LinAlgError:
            return np.full((p, p), np.nan)
        return np.asarray(sigma2 * MX_inv)  # (p, p)

    def _v_contrast(theta: np.ndarray, c: np.ndarray) -> float:
        """V(c) = c' · fe_cov(θ) · c."""
        fc = _fe_cov_at(theta)
        if np.any(np.isnan(fc)):
            return np.nan
        return float(c @ fc @ c)

    # Hessian of REML deviance (shared across all contrasts)
    def _deviance(theta: np.ndarray) -> float:
        return reml_objective(
            theta, y, X, Z, [], _cache=cache, specs=specs, n_levels=n_levels
        )

    h_hess = 1e-3
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = h_hess
            ej[j] = h_hess
            H[i, j] = (
                _deviance(theta_hat + ei + ej)
                - _deviance(theta_hat + ei - ej)
                - _deviance(theta_hat - ei + ej)
                + _deviance(theta_hat - ei - ej)
            ) / (4.0 * h_hess**2)
            H[j, i] = H[i, j]

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)

    # Gradient of V(c) w.r.t. theta for each contrast row
    h_grad = 1e-4
    dfs = np.zeros(n_contrasts)
    for row_idx in range(n_contrasts):
        c = L[row_idx]
        V_c = _v_contrast(theta_hat, c)
        grad = np.zeros(k)
        for i in range(k):
            theta_p = theta_hat.copy()
            theta_m = theta_hat.copy()
            theta_p[i] += h_grad
            theta_m[i] -= h_grad
            vp = _v_contrast(theta_p, c)
            vm = _v_contrast(theta_m, c)
            grad[i] = (vp - vm) / (2.0 * h_grad)

        denom = float(grad @ H_inv @ grad)
        if denom <= 0 or not np.isfinite(denom) or not np.isfinite(V_c):
            dfs[row_idx] = np.inf
        else:
            dfs[row_idx] = V_c**2 / denom

    return np.maximum(dfs, 1.0)


# ---------------------------------------------------------------------------
# contrast(): linear contrasts on emmeans results
# ---------------------------------------------------------------------------


def _adjust_pvalues(
    p_raw: np.ndarray,
    adjust: str,
    t_ratios: np.ndarray,
    dfs: np.ndarray,
    n_levels: int,
) -> np.ndarray:
    """Apply multiple-comparison p-value adjustment.

    Parameters
    ----------
    p_raw:
        Unadjusted p-values, shape (n,).
    adjust:
        Adjustment method: ``'none'``, ``'bonferroni'``, ``'holm'``,
        ``'fdr'``, or ``'tukey'``.
    t_ratios:
        t-statistics (used only for ``'tukey'``).
    dfs:
        Denominator DFs (used only for ``'tukey'``).
    n_levels:
        Number of EMM levels (used only for ``'tukey'``).

    Returns
    -------
    np.ndarray of shape (n,) with adjusted p-values in [0, 1].
    """
    n = len(p_raw)

    if adjust == "none":
        return p_raw.copy()

    if adjust == "bonferroni":
        return np.asarray(np.minimum(p_raw * n, 1.0), dtype=float)

    if adjust == "holm":
        # Step-down Bonferroni: sort ascending, multiply by (n - rank), cummax
        order = np.argsort(p_raw)
        p_adj = np.empty(n)
        cummax = 0.0
        for k, idx in enumerate(order):
            val = min(p_raw[idx] * (n - k), 1.0)
            cummax = max(cummax, val)
            p_adj[idx] = cummax
        return p_adj

    if adjust == "fdr":
        # Benjamini-Hochberg: sort ascending, multiply by n/rank, cummin from right
        order = np.argsort(p_raw)
        p_adj = np.empty(n)
        cummin = 1.0
        for k in range(n - 1, -1, -1):
            idx = order[k]
            val = min(p_raw[idx] * n / (k + 1), 1.0)
            cummin = min(cummin, val)
            p_adj[idx] = cummin
        return p_adj

    if adjust == "tukey":
        # Tukey HSD: use studentized range distribution q ~ Tukey(n_levels, df)
        # p = P(q > |t| * sqrt(2)) where q is studentized range
        # scipy: studentized_range(k=n_levels, df=df).sf(|t|*sqrt(2))
        p_adj = np.empty(n)
        for i in range(n):
            df_i = float(dfs[i])
            q_stat = float(np.abs(t_ratios[i])) * np.sqrt(2.0)
            try:
                p_adj[i] = float(
                    stats.studentized_range.sf(q_stat, k=n_levels, df=df_i)
                )
            except Exception:
                p_adj[i] = float(p_raw[i])
        return np.minimum(p_adj, 1.0)

    raise ValueError(
        f"adjust must be 'none', 'bonferroni', 'holm', 'fdr', or 'tukey'; "
        f"got {adjust!r}"
    )


def contrast(
    emm: EmmResult,
    method: str | list[np.ndarray] | dict[str, np.ndarray] = "pairwise",
    adjust: str = "none",
) -> Any:
    """Apply linear contrasts to estimated marginal means.

    Parameters
    ----------
    emm:
        An :class:`EmmResult` returned by :func:`emmeans`.
    method:
        How to form the contrasts:

        ``'pairwise'``
            All pairwise differences ``i - j`` for ``i < j`` (alphabetical
            level order).  Produces ``n*(n-1)//2`` rows.
        ``'trt.vs.ctrl'``
            Each non-first level minus the first (control) level.  Produces
            ``n-1`` rows.
        list of np.ndarray
            Each array is a contrast vector of length ``n`` (number of EMM
            rows).  Generic names ``contrast1``, ``contrast2``, … are used.
        dict mapping str → np.ndarray
            Same as a list but uses the dict keys as contrast names.
    adjust:
        Multiple-comparison p-value adjustment: ``'none'`` (default),
        ``'bonferroni'``, ``'holm'``, ``'fdr'``, or ``'tukey'``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``["contrast", "estimate", "SE", "df", "t.ratio", "p.value"]``.

    Raises
    ------
    ValueError
        If *method* is a string other than ``'pairwise'`` or ``'trt.vs.ctrl'``,
        or if *adjust* is not a recognised method.
    """
    if not isinstance(emm, EmmResult) or not hasattr(emm, "_emm_model"):
        raise TypeError(
            "emm must be an EmmResult returned by emmeans(). "
            "Plain DataFrames are not supported."
        )

    model: Any = emm._emm_model
    L_emm: np.ndarray = np.asarray(emm._emm_L)  # (n_cells, p)
    n_cells = L_emm.shape[0]

    # Build named contrast matrix C (n_contrasts, n_cells) in EMM space,
    # then project to FE space: L_c = C @ L_emm  (n_contrasts, p)
    specs_col: str = emm._emm_specs[0] if emm._emm_specs else "level"
    levels = list(emm[specs_col]) if specs_col in emm.columns else list(range(n_cells))

    if method == "pairwise":
        pairs = list(combinations(range(n_cells), 2))
        names = [f"{levels[i]} - {levels[j]}" for i, j in pairs]
        C = np.zeros((len(pairs), n_cells))
        for k, (i, j) in enumerate(pairs):
            C[k, i] = 1.0
            C[k, j] = -1.0

    elif method == "trt.vs.ctrl":
        ctrl = 0  # first level (sorted order) is control
        names = [f"{levels[i]} - {levels[ctrl]}" for i in range(1, n_cells)]
        C = np.zeros((n_cells - 1, n_cells))
        for k, i in enumerate(range(1, n_cells)):
            C[k, i] = 1.0
            C[k, ctrl] = -1.0

    elif isinstance(method, dict):
        names = list(method.keys())
        vectors = list(method.values())
        C = np.array(vectors, dtype=float)

    elif isinstance(method, (list, tuple)):
        C = np.array(list(method), dtype=float)
        names = [f"contrast{i + 1}" for i in range(len(C))]

    else:
        raise ValueError(
            f"method must be 'pairwise', 'trt.vs.ctrl', a list, or a dict; "
            f"got {method!r}"
        )

    # Project: (n_contrasts, p)
    L_c = C @ L_emm

    # Point estimates
    beta = np.asarray(model.fe_params)
    estimates = L_c @ beta

    # Variances and standard errors
    fe_cov = model.fe_cov
    var_c = np.einsum("ij,jk,ik->i", L_c, fe_cov, L_c)
    se = np.sqrt(np.maximum(var_c, 0.0))

    # Satterthwaite DFs for the projected contrasts
    dfs = _satterthwaite_df_contrasts(model, L_c)

    # t-ratios and (unadjusted) p-values
    t_ratios = np.where(se > 0, estimates / se, np.nan)
    p_raw = 2.0 * (1.0 - stats.t.cdf(np.abs(t_ratios), df=dfs))

    # p-value adjustment
    p_values = _adjust_pvalues(p_raw, adjust, t_ratios, dfs, n_cells)

    rows = [
        {
            "contrast": names[i],
            "estimate": float(estimates[i]),
            "SE": float(se[i]),
            "df": float(dfs[i]),
            "t.ratio": float(t_ratios[i]),
            "p.value": float(p_values[i]),
        }
        for i in range(len(names))
    ]

    return pd.DataFrame(
        rows, columns=["contrast", "estimate", "SE", "df", "t.ratio", "p.value"]
    )


def pairs(emm: EmmResult, adjust: str = "tukey") -> Any:
    """All pairwise comparisons of estimated marginal means.

    Convenience wrapper for ``contrast(emm, method='pairwise', adjust=adjust)``.
    Matches R's ``pairs()`` ergonomics with Tukey HSD adjustment by default.

    Parameters
    ----------
    emm:
        An :class:`EmmResult` returned by :func:`emmeans`.
    adjust:
        Multiple-comparison adjustment (default ``'tukey'``).  Any value
        accepted by :func:`contrast` is valid.

    Returns
    -------
    pandas.DataFrame
        Columns: ``["contrast", "estimate", "SE", "df", "t.ratio", "p.value"]``.
    """
    return contrast(emm, method="pairwise", adjust=adjust)
