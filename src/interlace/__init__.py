"""interlace: REML estimation for linear mixed models with crossed random intercepts."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.stats as stats

from interlace.augment import hlm_augment
from interlace.formula import (
    extract_group_factors,
    groups_to_random_effects,
    parse_formula,
    parse_random_effects,
)
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
    make_lambda,
)
from interlace.residuals import hlm_resid
from interlace.result import CrossedLMEResult, ModelInfo, _DataWrapper, _SimpleRE
from interlace.sparse_z import build_joint_z_from_specs

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
    data: Any,
    groups: str | list[str] | None = None,
    method: str = "REML",
    random: list[str] | None = None,
    optimizer: str = "lbfgsb",
    theta0: np.ndarray | None = None,
) -> CrossedLMEResult:
    """Fit a linear mixed model with crossed random effects via profiled REML.

    Parameters
    ----------
    formula:
        Patsy-syntax fixed-effects formula, e.g. ``"y ~ x1 + x2"``.
    data:
        DataFrame containing all variables.  Any narwhals-compatible frame
        (pandas, polars, …) is accepted.
    groups:
        Column name (str) or list of column names for crossed random
        intercepts. Shorthand for ``random=["(1|g1)", "(1|g2)", ...]``.
        The first entry is the primary grouping factor.
    random:
        List of lme4-style random effect specifications, e.g.
        ``["(1 + x | g1)", "(1 | g2)"]``.  Supports correlated (``|``) and
        independent (``||``) parameterisations.  Takes precedence over
        *groups* when both are provided.
    method:
        Estimation method; only ``"REML"`` is currently supported.
    optimizer:
        Optimizer for the REML criterion.  ``"lbfgsb"`` (default) uses
        ``scipy.optimize.minimize`` with L-BFGS-B.  ``"bobyqa"`` uses
        ``pybobyqa`` (requires the ``bobyqa`` optional extra), a
        gradient-free trust-region method that is more robust near
        variance-parameter boundaries and matches lme4's default.
    theta0:
        Initial theta for the optimizer.  Defaults to ``np.ones(n_theta)``.
        Pass the ``theta`` attribute of a previously fitted model to
        warm-start the optimizer (e.g. for case-deletion refits).

    Returns
    -------
    CrossedLMEResult
        Drop-in replacement for statsmodels ``MixedLMResults``.
    """
    if method != "REML":
        raise ValueError(f"Only method='REML' is supported; got '{method}'")
    if random is None and groups is None:
        raise ValueError("Either 'groups' or 'random' must be provided.")

    # Wrap with narwhals for uniform column access (pandas, polars, …).
    nw_data = nw.from_native(data, eager_only=True)

    # --- Build RandomEffectSpec list ---
    if random is not None:
        specs = parse_random_effects(random)
    else:
        specs = groups_to_random_effects(groups)  # type: ignore[arg-type]

    group_cols = [s.group for s in specs]

    # --- 1. Parse fixed-effects formula ---
    parsed = parse_formula(formula, data, groups=group_cols[0])
    y = parsed.y
    X = parsed.X
    term_names = parsed.term_names
    n, p = X.shape

    # --- 2. Build joint sparse Z and collect n_levels per spec ---
    Z = build_joint_z_from_specs(specs, data)
    group_factors = extract_group_factors(data, group_cols)
    n_levels_list: list[int] = [gf[2] for gf in group_factors]

    # --- 3. Fit REML ---
    reml = fit_reml(
        y,
        X,
        Z,
        q_sizes=[],
        specs=specs,
        n_levels=n_levels_list,
        optimizer=optimizer,
        theta0=theta0,
    )

    # --- 4. Recover quantities at optimum ---
    Lambda = make_lambda(reml.theta, specs, n_levels_list)
    cache = _precompute(y, X, Z)

    ZtZ = sp.csc_matrix(cache["ZtZ"])
    ZtX = np.asarray(cache["ZtX"])
    Zty = np.asarray(cache["Zty"])
    XtX = np.asarray(cache["XtX"])
    Xty = np.asarray(cache["Xty"])

    A11 = _build_A11(ZtZ, Lambda)
    lZty = np.asarray(Lambda.T @ Zty).squeeze()
    lZtX = np.asarray(Lambda.T @ ZtX)
    c1 = _sparse_solve(A11, lZty)
    C_X = _sparse_solve(A11, lZtX)
    MX = XtX - lZtX.T @ C_X  # X'Ω⁻¹X  (p×p)
    rhs = Xty - lZtX.T @ c1  # X'Ω⁻¹y  (p,)
    beta = la.solve(MX, rhs, assume_a="pos")

    # --- 5. BLUPs: b̂ = Lambda A11⁻¹ Lambda' Z'ε ---
    eps = y - X @ beta
    Zte = np.asarray(Z.T @ eps).squeeze()
    blups = np.asarray(
        Lambda @ _sparse_solve(A11, np.asarray(Lambda.T @ Zte).squeeze())
    ).squeeze()

    # --- 6. Fitted values and conditional residuals ---
    fittedvalues = X @ beta + Z @ blups
    resid = y - fittedvalues

    # --- 7. Standard errors (Wald) ---
    sigma2 = reml.sigma2
    MX_inv = np.linalg.inv(MX)
    fe_cov = sigma2 * MX_inv
    fe_bse_arr = np.sqrt(sigma2 * np.diag(MX_inv))
    z_scores = beta / fe_bse_arr
    fe_pvalues_arr = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_scores)))

    # Build FE result objects — pd.Series/pd.DataFrame when pandas is available,
    # plain numpy arrays otherwise.
    try:
        import pandas as _pd

        fe_params: Any = _pd.Series(beta, index=term_names)
        fe_bse: Any = _pd.Series(fe_bse_arr, index=term_names)
        fe_pvalues: Any = _pd.Series(fe_pvalues_arr, index=term_names)
        fe_conf_int: Any = _pd.DataFrame(
            {"lower": beta - 1.96 * fe_bse_arr, "upper": beta + 1.96 * fe_bse_arr},
            index=term_names,
        )
        _pandas_available = True
    except ImportError:
        fe_params = beta
        fe_bse = fe_bse_arr
        fe_pvalues = fe_pvalues_arr
        fe_conf_int = np.column_stack(
            [beta - 1.96 * fe_bse_arr, beta + 1.96 * fe_bse_arr]
        )
        _pandas_available = False

    # --- 8. Package random effects per spec ---
    random_effects: dict[str, Any] = {}
    variance_components: dict[str, Any] = {}
    ngroups: dict[str, int] = {}
    theta_idx = 0
    blup_offset = 0
    for spec, q_j in zip(specs, n_levels_list, strict=True):
        from interlace.profiled_reml import n_theta_for_spec

        n_theta_j = n_theta_for_spec(spec.n_terms, spec.correlated)
        n_blups_j = spec.n_terms * q_j
        blup_block = blups[blup_offset : blup_offset + n_blups_j]
        uniques: list[Any] = sorted(np.unique(nw_data[spec.group].to_numpy()).tolist())

        if spec.n_terms == 1:
            # Intercept-only: backward-compatible Series/SimpleRE + scalar variance
            if _pandas_available:
                random_effects[spec.group] = _pd.Series(
                    blup_block, index=uniques, name=spec.group
                )
            else:
                random_effects[spec.group] = _SimpleRE(
                    values=blup_block, index=uniques, name=spec.group
                )
            theta_j0 = reml.theta[theta_idx]
            variance_components[spec.group] = float(sigma2 * theta_j0**2)
        else:
            # Multi-term: DataFrame(index=levels, columns=term_names)
            term_names_j = (["(Intercept)"] if spec.intercept else []) + list(
                spec.predictors
            )
            theta_j = reml.theta[theta_idx : theta_idx + n_theta_j]

            # blup_block is term-first: [q_j intercept BLUPs, q_j slope BLUPs, ...]
            # reshape to (n_terms, q_j) then transpose → (q_j, n_terms)
            re_mat = blup_block.reshape(spec.n_terms, q_j).T

            if _pandas_available:
                random_effects[spec.group] = _pd.DataFrame(
                    re_mat, index=uniques, columns=term_names_j
                )
            else:
                random_effects[spec.group] = re_mat  # (q_j, n_terms) numpy array

            # Covariance matrix: sigma2 * L_j @ L_j.T
            p_j = spec.n_terms
            if spec.correlated:
                L_j = np.zeros((p_j, p_j))
                idx = 0
                for row in range(p_j):
                    for col in range(row + 1):
                        L_j[row, col] = theta_j[idx]
                        idx += 1
                cov_mat = sigma2 * L_j @ L_j.T
            else:
                # Independent: diagonal covariance
                cov_mat = np.diag(sigma2 * theta_j**2)

            if _pandas_available:
                variance_components[spec.group] = _pd.DataFrame(
                    cov_mat, index=term_names_j, columns=term_names_j
                )
            else:
                variance_components[spec.group] = cov_mat  # numpy ndarray

        ngroups[spec.group] = q_j
        theta_idx += n_theta_j
        blup_offset += n_blups_j

    # --- 9. Build ModelInfo ---
    # Cache a pandas copy of the data if pandas is installed (used by diagnostics
    # that rely on pandas-specific operations like the statsmodels compat path).
    if _pandas_available:
        pd_frame: Any = _pd.DataFrame(
            {col: nw_data[col].to_numpy() for col in nw_data.columns}
        )
    else:
        pd_frame = None

    model_info = ModelInfo(
        exog=X,
        endog=y,
        groups=nw_data[group_cols[0]].to_numpy(),
        endog_names=formula.split("~")[0].strip(),
        formula=formula,
        data=_DataWrapper(frame=data, _pandas_frame=pd_frame),
    )

    return CrossedLMEResult(
        fe_params=fe_params,
        fe_bse=fe_bse,
        fe_pvalues=fe_pvalues,
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
        _random_specs=list(specs),
    )
