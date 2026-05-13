"""Heteroscedastic LMM with dispformula via Block-Coordinate Ascent (BCA).

Model
-----
    y_i | b, d ~ N(X_i β + Z_i b, σ_i²)
    log σ_i = W_i γ + V_i d
    b ~ N(0, G(θ_mean)),  d ~ N(0, H(θ_disp))

The mean and dispersion blocks are updated alternately. The mean block is
a weighted LMM (existing :func:`interlace.fit`); the dispersion block is a
Gamma GLMM with log link on squared residuals, fit via
:func:`interlace.glmm_laplace.fit_glmm`. Internal coefficients are on the
log-variance scale (since E[e²] = σ²) and divided by 2 at the boundary so
that reported ``disp_params`` and ``disp_variance_components`` match glmmTMB
(log-σ scale).

This BCA path supports both fixed-effect-only and random-effects-on-disp
dispformulae. For FE-only dispformulae it can be cross-checked against
the joint-Laplace path (:func:`interlace.glmer` with ``dispformula=``).
"""

from __future__ import annotations

import re
from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd

from interlace.formula import RandomEffectSpec, parse_random_effects

_BCA_TOL = 1e-5
_BCA_MAXITER = 60
_E_SQ_FLOOR = 1e-12  # squared-residual floor to keep log(e²) finite

# Pattern matching a single "(...|...)" or "(...||...)" RE term inside a formula RHS.
_RE_TOKEN = re.compile(r"\([^()]*\|\|?[^()]*\)")


def _parse_dispformula_raw(dispformula: str) -> tuple[str, list[str]]:
    """Split a dispformula into (FE-RHS, list of raw lme4-style RE tokens).

    Returns the raw "(...|g/h)" strings so they can be re-parsed downstream
    (lme4's nested-shorthand cannot be round-tripped through
    :func:`spec_to_str` because the derived `g:h` interaction group is not a
    valid identifier).
    """
    rhs = dispformula.lstrip("~").strip()
    re_tokens = _RE_TOKEN.findall(rhs)
    fe_part = _RE_TOKEN.sub("", rhs)
    fe_part = re.sub(r"\+\s*\+", "+", fe_part)
    fe_part = fe_part.strip().strip("+").strip()
    if not fe_part:
        fe_part = "1"
    return fe_part, re_tokens


def _parse_dispformula(dispformula: str) -> tuple[str, list[RandomEffectSpec]]:
    """Split a dispformula into (FE-RHS, list of RandomEffectSpec).

    Examples
    --------
    >>> _parse_dispformula("~1")
    ('1', [])
    >>> _parse_dispformula("~ z")
    ('z', [])
    >>> rhs, specs = _parse_dispformula("~ (1|g)")
    >>> rhs, [s.group for s in specs]
    ('1', ['g'])
    >>> rhs, specs = _parse_dispformula("~ z + (1|g)")
    >>> rhs, [s.group for s in specs]
    ('z', ['g'])
    >>> rhs, specs = _parse_dispformula("~ (1|g1/g2)")
    >>> rhs, [s.group for s in specs]
    ('1', ['g1', 'g1:g2'])
    """
    rhs = dispformula.lstrip("~").strip()
    re_tokens = _RE_TOKEN.findall(rhs)
    fe_part = _RE_TOKEN.sub("", rhs)
    # Tidy stray '+' separators left by token removal.
    fe_part = re.sub(r"\+\s*\+", "+", fe_part)
    fe_part = fe_part.strip().strip("+").strip()
    if not fe_part:
        fe_part = "1"
    re_specs = parse_random_effects(re_tokens) if re_tokens else []
    return fe_part, re_specs


def _to_pandas(data: Any) -> pd.DataFrame:
    """Return a pandas copy of ``data``, regardless of input backend."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    out: pd.DataFrame = nw.from_native(data, eager_only=True).to_pandas()
    return out


def _build_disp_design_matrices(
    df: pd.DataFrame,
    disp_fe_rhs: str,
    disp_re_specs: list[RandomEffectSpec],
) -> tuple[np.ndarray, list[str], np.ndarray | None, list[str]]:
    """Build W (FE) and stacked V (RE) matrices for the dispersion model.

    Returns
    -------
    W : (n, p_d) array of dispersion fixed-effect design.
    disp_term_names : list of column names for W.
    V : (n, q_d) array (dense) of dispersion random-effect design, or None.
        For the BCA dispersion fit we only need to build V at the end for
        predicted sigma; the inner glmm_laplace call constructs its own Z
        from the disp_re_specs.
    disp_re_group_cols : group column names per spec (for sigma reconstruction).
    """
    import formulaic

    W_mm = formulaic.model_matrix("~ " + disp_fe_rhs, df)
    disp_term_names = list(W_mm.columns)
    W = np.asarray(W_mm, dtype=np.float64)

    # V is reconstructed from the disp-side glmm BLUPs after the fit; here
    # we return the list of group columns so the caller can build sigma.
    group_cols = [s.group for s in disp_re_specs]
    return W, disp_term_names, None, group_cols


def _materialize_interaction_columns(
    df: pd.DataFrame,
    disp_re_specs: list[RandomEffectSpec],
) -> pd.DataFrame:
    """Add derived interaction columns (e.g. 'g1:g2') to df for nested REs."""
    df = df.copy()
    for spec in disp_re_specs:
        if spec.interaction_cols and spec.group not in df.columns:
            df[spec.group] = df[spec.interaction_cols[0]].astype(str)
            for col in spec.interaction_cols[1:]:
                df[spec.group] = df[spec.group] + ":" + df[col].astype(str)
    return df


def fit_dispformula_bca(
    formula: str,
    data: Any,
    dispformula: str,
    groups: str | list[str] | None = None,
    random: list[str] | None = None,
    method: str = "REML",
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    df_method: str = "satterthwaite",
    max_iter: int = _BCA_MAXITER,
    tol: float = _BCA_TOL,
) -> Any:
    """Fit Gaussian heteroscedastic LMM with dispformula via BCA.

    Returns a :class:`CrossedLMEResult` with ``disp_params``,
    ``disp_random_effects``, ``disp_variance_components``, and
    ``dispersion`` populated (and ``scale`` fixed at 1, since the
    dispformula absorbs all variance).
    """
    from interlace import fit as fit_lmm
    from interlace.glmm_family import GammaFamily
    from interlace.glmm_laplace import fit_glmm

    disp_fe_rhs, disp_re_specs = _parse_dispformula(dispformula)
    _, disp_re_raw = _parse_dispformula_raw(dispformula)
    has_disp_re = len(disp_re_specs) > 0

    # Coerce data to pandas for stable column manipulation in inner fits.
    df_pd = _to_pandas(data)
    df_pd = _materialize_interaction_columns(df_pd, disp_re_specs)
    n = df_pd.shape[0]

    base_weights = (
        np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    )

    W, disp_term_names, _, disp_group_cols = _build_disp_design_matrices(
        df_pd, disp_fe_rhs, disp_re_specs
    )

    # State: per-obs sigma. Init homoscedastic.
    sigma_i = np.ones(n)
    delta_hat: np.ndarray = np.zeros(W.shape[1])
    disp_blups: dict[str, np.ndarray] = {}
    disp_varcomps: dict[str, float] = {}
    mean_fit = None
    converged = False

    for _ in range(max_iter):
        # ---- Mean step: weighted LMM ----
        # Effective weights: base_w / sigma_i^2 rescales residual variance to 1.
        weights_eff = base_weights / (sigma_i**2)
        mean_fit = fit_lmm(
            formula=formula,
            data=df_pd,
            groups=groups,
            random=random,
            method=method,
            weights=weights_eff,
            offset=offset,
            df_method=df_method,
        )

        # Residuals on original scale (interlace.fit subtracts X@beta + Z@b
        # from the unweighted y, so this is correct regardless of weights).
        e = np.asarray(mean_fit.resid, dtype=np.float64)
        e_sq = np.maximum(e**2, _E_SQ_FLOOR)

        # ---- Disp step: Gamma GLM(M) on e² with log link ----
        # E[e²|d] = σ_i² = exp(2*(W γ + V d)).
        # We fit log E[e²] = (W α + V c), with α = 2 γ, c = 2 d.
        # The glmm_laplace path always profiles over all variance components,
        # so even with no random effects on disp side, we call a pure Gamma
        # GLM (not GLMM) via the BCA Gamma helper below.
        df_pd_d = df_pd.copy()
        df_pd_d["_e_sq"] = e_sq

        if has_disp_re:
            # Gamma GLMM on e² with log link.  Use shape=0.5 (chi-squared on
            # 1 df), since e_i² ~ σ_i² * χ²_1 = Gamma(1/2, 2σ_i²).  Point
            # estimates of fixed effects are invariant to shape (the score
            # rescales uniformly), but disp variance components rely on the
            # marginal Laplace log-lik whose curvature DOES depend on shape.
            disp_glmm = fit_glmm(
                formula=f"_e_sq ~ {disp_fe_rhs}",
                data=df_pd_d,
                family=GammaFamily(link="log", shape=0.5),
                random=disp_re_raw,
                weights=base_weights,
            )
            alpha_fe = np.asarray(disp_glmm.fe_params.values, dtype=np.float64)
            new_delta = alpha_fe / 2.0
            disp_term_names = list(disp_glmm.fe_params.index)

            # BLUPs and variance components on the alpha (log-variance) scale.
            new_blups: dict[str, np.ndarray] = {}
            new_varcomps: dict[str, float] = {}
            for spec in disp_re_specs:
                gname = spec.group
                if gname in disp_glmm.random_effects:
                    raw_blup = np.asarray(
                        disp_glmm.random_effects[gname], dtype=np.float64
                    )
                else:
                    raw_blup = np.zeros(0, dtype=np.float64)
                # Convert from log-variance to log-sigma scale: divide by 2.
                new_blups[gname] = raw_blup / 2.0
                # Var(c) was on log-variance scale; Var(d) = Var(c)/4.
                if gname in disp_glmm.variance_components:
                    new_varcomps[gname] = (
                        float(disp_glmm.variance_components[gname]) / 4.0
                    )
                else:
                    new_varcomps[gname] = 0.0

            # Reconstruct log_sigma per observation.
            log_sigma = W @ new_delta
            for spec in disp_re_specs:
                gname = spec.group
                blup = new_blups[gname]
                group_arr = df_pd[gname].to_numpy()
                _, codes = np.unique(group_arr, return_inverse=True)
                if blup.size == codes.max() + 1:
                    log_sigma = log_sigma + blup[codes]
            disp_blups = new_blups
            disp_varcomps = new_varcomps
            delta_hat = new_delta
        else:
            # FE-only dispformula: closed-form-ish IRLS over alpha = 2*delta.
            # Iterate the Gamma GLM (log link, shape=1) to fit log E[e²]=W α.
            alpha_fe = _gamma_log_link_irls(W, e_sq, weights=base_weights)
            new_delta = alpha_fe / 2.0
            log_sigma = W @ new_delta
            disp_blups = {}
            disp_varcomps = {}
            delta_hat = new_delta

        sigma_i_new = np.exp(log_sigma)
        # Convergence on log-sigma scale.
        max_change = float(np.max(np.abs(np.log(sigma_i_new) - np.log(sigma_i))))
        sigma_i = sigma_i_new
        if max_change < tol:
            converged = True
            break

    assert mean_fit is not None  # loop runs at least once

    # ---- Package as CrossedLMEResult ----
    # Most attributes come from mean_fit; we overlay the dispformula state
    # and fix scale to 1 (the dispformula absorbs all variance).
    result = mean_fit
    result.disp_params = pd.Series(delta_hat, index=disp_term_names)
    result.disp_random_effects = {g: pd.Series(b) for g, b in disp_blups.items()}
    result.disp_variance_components = disp_varcomps
    result.dispersion = sigma_i**2  # phi = sigma^2 per glmmTMB convention
    result.disp_method = "bca"
    # The BCA mean step absorbed all variance into weights = 1/σ_i², so the
    # LMM's profiled sigma² is meaningless as a "residual variance". Fix it
    # to 1.0 and rely on the dispformula for per-observation variance.
    result.scale = 1.0
    # Tally extra parameters (FE coefs on disp side + RE varcomps on disp side).
    n_disp_fe = int(delta_hat.size)
    n_disp_vc = sum(1 for v in disp_varcomps.values())
    result.nparams = result.nparams + n_disp_fe + n_disp_vc
    if not converged:
        result.converged = False
    return result


def fit_dispformula_joint_laplace(
    formula: str,
    data: Any,
    dispformula: str,
    groups: str | list[str] | None = None,
    random: list[str] | None = None,
    method: str = "REML",
    weights: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    df_method: str = "satterthwaite",
) -> Any:
    """Fit Gaussian heteroscedastic LMM (FE-only dispformula) via joint Laplace.

    Wraps :func:`interlace.glmm_laplace.fit_glmm` (family="gaussian",
    ``dispformula=...``) and adapts the GLMMResult into a
    :class:`CrossedLMEResult` so callers using :func:`interlace.fit` get a
    drop-in LMM result.
    """
    from interlace.glmm_laplace import fit_glmm

    glmm_res = fit_glmm(
        formula=formula,
        data=data,
        family="gaussian",
        groups=groups,
        random=random,
        weights=weights,
        offset=offset,
        dispformula=dispformula,
    )
    return _glmm_to_lme(glmm_res, formula, data, groups, random, method, df_method)


def _glmm_to_lme(
    glmm_res: Any,
    formula: str,
    data: Any,
    groups: Any,
    random: Any,
    method: str,
    df_method: str,
) -> Any:
    """Convert a Gaussian GLMMResult (with dispformula) to a CrossedLMEResult."""
    import scipy.stats as stats

    from interlace.formula import groups_to_random_effects, parse_random_effects
    from interlace.result import CrossedLMEResult, ModelInfo, _DataWrapper

    nw_data = nw.from_native(data, eager_only=True)
    if random is not None:
        specs = parse_random_effects(random)
    else:
        specs = groups_to_random_effects(groups)
    primary_group_col = specs[0].group

    fe_params = glmm_res.fe_params
    fe_bse = glmm_res.fe_bse
    p = int(fe_params.size)
    n = int(glmm_res.nobs)

    # Approximate FE inference: large-sample normal on |z|, residual DFs ≈ n - p.
    # Joint Laplace SEs already reflect the full marginal covariance, so we
    # do not run Satterthwaite/KR here (which assume Gaussian LMM with a single
    # residual variance). Downstream consumers needing those should use
    # ``interlace.fit`` without ``dispformula``.
    z_scores = np.asarray(fe_params.values) / np.asarray(fe_bse.values)
    fe_df_arr = np.full(p, max(n - p, 1.0))
    fe_pvalues_arr = 2.0 * (1.0 - stats.t.cdf(np.abs(z_scores), df=fe_df_arr))
    fe_pvalues = pd.Series(fe_pvalues_arr, index=fe_params.index)
    fe_df = pd.Series(fe_df_arr, index=fe_params.index)
    fe_conf_int = pd.DataFrame(
        {
            "lower": fe_params.values - 1.96 * fe_bse.values,
            "upper": fe_params.values + 1.96 * fe_bse.values,
        },
        index=fe_params.index,
    )
    fe_cov = np.diag(np.asarray(fe_bse.values) ** 2)

    endog_name = formula.split("~", 1)[0].strip()
    endog = nw_data[endog_name].to_numpy().astype(np.float64)
    exog = np.zeros((n, p))  # placeholder; downstream LMM diagnostics needing
    # exog can refit without dispformula.
    model = ModelInfo(
        exog=exog,
        endog=endog,
        groups=nw_data[primary_group_col].to_numpy(),
        endog_names=endog_name,
        formula=formula,
        data=_DataWrapper(frame=data),
    )

    return CrossedLMEResult(
        fe_params=fe_params,
        fe_bse=fe_bse,
        fe_pvalues=fe_pvalues,
        fe_conf_int=fe_conf_int,
        fe_df=fe_df,
        random_effects=glmm_res.random_effects,
        variance_components=glmm_res.variance_components,
        theta=np.asarray(glmm_res.theta),
        resid=np.asarray(endog - glmm_res.fittedvalues),
        fittedvalues=np.asarray(glmm_res.fittedvalues),
        scale=1.0,  # dispformula absorbs variance per-obs
        fe_cov=fe_cov,
        model=model,
        converged=bool(glmm_res.converged),
        nobs=n,
        ngroups=glmm_res.ngroups,
        method=method,
        llf=float(glmm_res.llf),
        aic=float(glmm_res.aic),
        bic=float(glmm_res.bic),
        nparams=p
        + int(np.asarray(glmm_res.theta).size)
        + int(glmm_res.disp_params.size if glmm_res.disp_params is not None else 0),
        _primary_group_col=primary_group_col,
        _secondary_group_cols=[s.group for s in specs[1:]],
        _random_specs=list(specs),
        df_method=df_method,
        disp_params=glmm_res.disp_params,
        disp_random_effects={},
        disp_variance_components={},
        dispersion=np.asarray(glmm_res.dispersion)
        if glmm_res.dispersion is not None
        else None,
        disp_method="joint_laplace",
    )


def _gamma_log_link_irls(
    W: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> np.ndarray:
    """Fit a Gamma GLM with log link via IRLS.

    Models log E[y] = W α. Returns α. Standard scoring iteration, which for
    Gamma GLM with log link reduces to weighted least squares with working
    response z = η + (y - μ)/μ and weights = w_i (μ-independent for log link).
    """
    n, p = W.shape
    alpha = np.zeros(p)
    # Initialise intercept to log(mean(y)) if W has an intercept column of 1s.
    if W.shape[1] >= 1 and np.allclose(W[:, 0], 1.0):
        alpha[0] = np.log(float(np.average(y, weights=weights)))
    for _ in range(max_iter):
        eta = W @ alpha
        mu = np.exp(eta)
        # Working response z = η + (y - μ)/μ;  weights = w_i (since for log
        # link Gamma, mu_eta = μ and Var(y) = μ²/φ; w_i = w_obs * μ²/(φμ²) =
        # w_obs/φ — φ uniform → ignored).
        z = eta + (y - mu) / mu
        # WLS: minimise Σ w_i (z_i - W_i α)^2  →  α = (W'WW)^{-1} W'Wz
        Wd = weights[:, None] * W
        A = W.T @ Wd
        b = W.T @ (weights * z)
        try:
            alpha_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            alpha_new = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.max(np.abs(alpha_new - alpha)) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new
    return alpha
