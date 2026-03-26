"""interlace: REML estimation for linear mixed models with crossed random intercepts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.sparse as sp
import scipy.stats as stats

from interlace.augment import hlm_augment
from interlace.formula import extract_group_factors, parse_formula
from interlace.influence import (
    cooks_distance,
    hlm_influence,
    mdffits,
    n_influential,
    tau_gap,
)
from interlace.leverage import leverage
from interlace.plotting import dotplot_diag, plot_influence, plot_resid
from interlace.profiled_reml import (
    _build_A11,
    _precompute,
    _sparse_solve,
    fit_reml,
    make_lambda_diag,
)
from interlace.residuals import hlm_resid
from interlace.result import CrossedLMEResult, ModelInfo, _DataWrapper
from interlace.sparse_z import build_joint_z

__all__ = [
    "fit",
    "CrossedLMEResult",
    # Residuals
    "hlm_resid",
    # Leverage
    "leverage",
    # Influence diagnostics
    "hlm_influence",
    "cooks_distance",
    "mdffits",
    "n_influential",
    "tau_gap",
    # Combined
    "hlm_augment",
    # Plotting
    "plot_resid",
    "plot_influence",
    "dotplot_diag",
]


def fit(
    formula: str,
    data: pd.DataFrame,
    groups: str | list[str],
    method: str = "REML",
) -> CrossedLMEResult:
    """Fit a linear mixed model with crossed random intercepts via profiled REML.

    Parameters
    ----------
    formula:
        Patsy-syntax fixed-effects formula, e.g. ``"y ~ x1 + x2"``.
    data:
        DataFrame containing all variables.
    groups:
        Column name (str) or list of column names for crossed random intercepts.
        The first entry is the primary grouping factor (used for ``_gpgap_group_col``).
    method:
        Estimation method; only ``"REML"`` is currently supported.

    Returns
    -------
    CrossedLMEResult
        Drop-in replacement for statsmodels ``MixedLMResults``.
    """
    if method != "REML":
        raise ValueError(f"Only method='REML' is supported; got '{method}'")

    group_cols: list[str] = [groups] if isinstance(groups, str) else list(groups)

    # --- 1. Parse fixed-effects formula ---
    parsed = parse_formula(formula, data, groups=group_cols[0])
    y = parsed.y
    X = parsed.X
    term_names = parsed.term_names
    n, p = X.shape

    # --- 2. Extract grouping factor integer codes ---
    factors = extract_group_factors(data, group_cols)

    # --- 3. Build joint sparse Z ---
    Z = build_joint_z(factors)
    q_sizes = [n_levels for _, _, n_levels in factors]

    # --- 4. Fit REML ---
    reml = fit_reml(y, X, Z, q_sizes)

    # --- 5. Recover quantities at optimum ---
    lambda_diag = make_lambda_diag(reml.theta, q_sizes)
    cache = _precompute(y, X, Z)

    ZtZ = sp.csc_matrix(cache["ZtZ"])
    ZtX = np.asarray(cache["ZtX"])
    Zty = np.asarray(cache["Zty"])
    XtX = np.asarray(cache["XtX"])
    Xty = np.asarray(cache["Xty"])

    A11 = _build_A11(ZtZ, lambda_diag)
    lZty = lambda_diag * Zty
    lZtX = lambda_diag[:, None] * ZtX
    c1 = _sparse_solve(A11, lZty)
    C_X = _sparse_solve(A11, lZtX)
    MX = XtX - lZtX.T @ C_X  # X'Ω⁻¹X  (p×p)
    rhs = Xty - lZtX.T @ c1  # X'Ω⁻¹y  (p,)
    beta = la.solve(MX, rhs, assume_a="pos")

    # --- 6. BLUPs: b̂ = λ ⊙ A11⁻¹(λ ⊙ Z'ε) ---
    eps = y - X @ beta  # marginal residual
    Zte = np.asarray(Z.T @ eps).squeeze()
    blups = lambda_diag * _sparse_solve(A11, lambda_diag * Zte)

    # --- 7. Fitted values and conditional residuals ---
    fittedvalues = X @ beta + Z @ blups
    resid = y - fittedvalues

    # --- 8. Standard errors (Wald) ---
    sigma2 = reml.sigma2
    MX_inv = np.linalg.inv(MX)
    fe_cov = sigma2 * MX_inv  # p×p FE covariance: scale * (X'Ω⁻¹X)⁻¹
    fe_bse = np.sqrt(sigma2 * np.diag(MX_inv))
    z_scores = beta / fe_bse
    fe_pvalues = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_scores)))
    fe_conf_int = pd.DataFrame(
        {
            "lower": beta - 1.96 * fe_bse,
            "upper": beta + 1.96 * fe_bse,
        },
        index=term_names,
    )

    # --- 9. Package random effects per factor ---
    random_effects: dict[str, pd.Series] = {}
    variance_components: dict[str, float] = {}
    ngroups: dict[str, int] = {}
    offset = 0
    for (col, _codes, n_levels), theta_j in zip(factors, reml.theta, strict=True):
        uniques = sorted(data[col].unique())
        re_series = pd.Series(
            blups[offset : offset + n_levels], index=uniques, name=col
        )
        random_effects[col] = re_series
        variance_components[col] = float(sigma2 * theta_j**2)
        ngroups[col] = n_levels
        offset += n_levels

    # --- 10. Build ModelInfo ---
    model_info = ModelInfo(
        exog=X,
        endog=y,
        groups=np.asarray(data[group_cols[0]]),
        endog_names=formula.split("~")[0].strip(),
        formula=formula,
        data=_DataWrapper(frame=data),
    )

    return CrossedLMEResult(
        fe_params=pd.Series(beta, index=term_names),
        fe_bse=pd.Series(fe_bse, index=term_names),
        fe_pvalues=pd.Series(fe_pvalues, index=term_names),
        fe_conf_int=fe_conf_int,
        random_effects=random_effects,
        variance_components=variance_components,
        theta=reml.theta,
        resid=np.asarray(resid),
        fittedvalues=np.asarray(fittedvalues),
        scale=sigma2,
        fe_cov=fe_cov,
        model=model_info,
        converged=reml.converged,
        nobs=n,
        ngroups=ngroups,
        method=method,
        llf=reml.llf,
        aic=reml.aic,
        bic=reml.bic,
        _gpgap_group_col=group_cols[0],
        _gpgap_vc_cols=group_cols[1:],
    )
