"""Student-t residual likelihood composed with dispformula (interlace-hpzy).

EM over the latent-scale representation of the Student-t

    tau_i ~ Gamma(nu/2, nu/2)
    y_i | tau_i, sigma_i ~ N(x_i' beta + z_i' b, sigma_i^2 / tau_i)
    log sigma_i = W_i delta + V_i d

interleaved with the existing BCA (block-coordinate ascent) heteroscedastic
LMM scheme. After substituting in the latent-scale conditional, the M-step
reduces to two weighted GLMs:

* Mean step: weighted Gaussian LMM with effective weights
  ``base_weights * tau / sigma_i^2``.
* Disp step: Gamma GL(MM) with log link on ``tau_i * e_i^2`` and shape
  ``0.5`` (the t-likelihood sufficient statistic for ``log sigma^2``),
  with prior weights ``base_weights``.

``nu`` is updated by a 1-D bounded search on the marginal Student-t
profile log-likelihood applied to the standardised residuals
``e_i / sigma_i`` (ECM).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from interlace.student_t import StudentTResult, _log_t_density, _profile_nu


@dataclass
class StudentTDispResult(StudentTResult):
    """Student-t LMM with dispformula. Extends :class:`StudentTResult`."""

    disp_params: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    disp_random_effects: dict[str, pd.Series] = field(default_factory=dict)
    disp_variance_components: dict[str, float] = field(default_factory=dict)
    dispersion: np.ndarray = field(default_factory=lambda: np.zeros(0))
    disp_method: str = "bca"


_OUTER_TOL = 1e-5
_OUTER_MAXITER = 80
_E_SQ_FLOOR = 1e-12


def fit_student_t_dispformula_bca(
    formula: str,
    data: Any,
    dispformula: str,
    groups: str | list[str] | None = None,
    random: list[str] | None = None,
    nu: float | None = None,
    weights: np.ndarray | None = None,
    method: str = "REML",
    df_method: str = "satterthwaite",
    nu_init: float = 4.0,
    nu_min: float = 2.001,
    nu_max: float = 200.0,
    max_iter: int = _OUTER_MAXITER,
    tol: float = _OUTER_TOL,
) -> StudentTDispResult:
    """Fit a Student-t LMM with a log-linear sub-model for sigma_i.

    The dispformula path supports both FE-only (``~ z``) and RE-on-disp
    (``~ (1|g)``) sub-models, mirroring :func:`fit_dispformula_bca`.
    """
    if nu is not None and nu <= 2.0:
        raise ValueError(
            f"nu must be > 2 (variance is undefined for nu <= 2); got {nu}."
        )
    if nu_min <= 2.0:
        raise ValueError(f"nu_min must be > 2; got {nu_min}.")

    from interlace import fit as lmm_fit
    from interlace.dispformula_bca import (
        _build_disp_design_matrices,
        _gamma_log_link_irls,
        _materialize_interaction_columns,
        _parse_dispformula,
        _parse_dispformula_raw,
        _to_pandas,
    )
    from interlace.glmm_family import GammaFamily
    from interlace.glmm_laplace import fit_glmm

    nu_estimated = nu is None
    nu_current = float(nu_init) if nu_estimated else float(nu)  # type: ignore[arg-type]

    user_weights = None if weights is None else np.asarray(weights, dtype=np.float64)

    disp_fe_rhs, disp_re_specs = _parse_dispformula(dispformula)
    _, disp_re_raw = _parse_dispformula_raw(dispformula)
    has_disp_re = len(disp_re_specs) > 0

    df_pd = _to_pandas(data)
    df_pd = _materialize_interaction_columns(df_pd, disp_re_specs)
    n = df_pd.shape[0]

    base_weights = (
        np.ones(n) if user_weights is None else user_weights.astype(np.float64)
    )

    W, disp_term_names, _, _disp_group_cols = _build_disp_design_matrices(
        df_pd, disp_fe_rhs, disp_re_specs
    )

    # State.
    sigma_i = np.ones(n)
    delta_hat: np.ndarray = np.zeros(W.shape[1])
    disp_blups: dict[str, np.ndarray] = {}
    disp_varcomps: dict[str, float] = {}
    mean_fit = None

    # Initial mean fit: unweighted (homoscedastic) Gaussian LMM to get residuals.
    mean_fit = lmm_fit(
        formula=formula,
        data=df_pd,
        groups=groups,
        random=random,
        method=method,
        weights=base_weights,
        df_method=df_method,
    )
    e = np.asarray(mean_fit.resid, dtype=np.float64)
    # Initialise nu from homoscedastic residuals if estimable.
    if nu_estimated:
        nu_current = _profile_nu(e, float(mean_fit.scale), user_weights, nu_min, nu_max)

    prev_ll = -np.inf
    converged = False

    for _ in range(max_iter):
        # --- E-step: posterior expectations of latent precisions ---
        # E[tau_i | y] = (nu + 1) / (nu + (e_i / sigma_i)^2).
        r2 = (e * e) / np.maximum(sigma_i * sigma_i, _E_SQ_FLOOR)
        tau = (nu_current + 1.0) / (nu_current + r2)

        # --- Mean step: weighted LMM with weights base_w * tau / sigma_i^2 ---
        weights_eff = (base_weights * tau) / np.maximum(sigma_i**2, _E_SQ_FLOOR)
        mean_fit = lmm_fit(
            formula=formula,
            data=df_pd,
            groups=groups,
            random=random,
            method=method,
            weights=weights_eff,
            df_method=df_method,
        )
        e = np.asarray(mean_fit.resid, dtype=np.float64)
        e_sq = np.maximum(e * e, _E_SQ_FLOOR)

        # --- Disp step: Gamma GL(MM) on tau * e_sq with shape 0.5 ---
        # E[tau * e^2 | y] = sigma_i^2 under the latent-scale t model.
        # Internal coefficients alpha are on log-variance scale; final
        # delta = alpha / 2 (log-sigma scale, glmmTMB convention).
        disp_target = tau * e_sq

        if has_disp_re:
            df_pd_d = df_pd.copy()
            df_pd_d["_disp_target"] = disp_target
            disp_glmm = fit_glmm(
                formula=f"_disp_target ~ {disp_fe_rhs}",
                data=df_pd_d,
                family=GammaFamily(link="log", shape=0.5),
                random=disp_re_raw,
                weights=base_weights,
            )
            alpha_fe = np.asarray(disp_glmm.fe_params.values, dtype=np.float64)
            new_delta = alpha_fe / 2.0
            disp_term_names = list(disp_glmm.fe_params.index)

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
                new_blups[gname] = raw_blup / 2.0
                if gname in disp_glmm.variance_components:
                    new_varcomps[gname] = (
                        float(disp_glmm.variance_components[gname]) / 4.0
                    )
                else:
                    new_varcomps[gname] = 0.0

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
            alpha_fe = _gamma_log_link_irls(W, disp_target, weights=base_weights)
            new_delta = alpha_fe / 2.0
            log_sigma = W @ new_delta
            disp_blups = {}
            disp_varcomps = {}
            delta_hat = new_delta

        sigma_i = np.exp(log_sigma)

        # --- nu CM-step ---
        # Profile over nu using marginal Student-t density on standardised
        # residuals r_i = e_i / sigma_i.
        r = e / np.maximum(sigma_i, np.sqrt(_E_SQ_FLOOR))
        if nu_estimated:
            nu_current = _profile_nu(r, 1.0, user_weights, nu_min, nu_max)

        # --- Convergence: change in marginal Student-t log-lik on
        # standardised residuals, minus log-Jacobian sum(log sigma_i). ---
        ll = _log_t_density(r, 1.0, nu_current)
        if user_weights is not None:
            ll = ll * user_weights
            log_jac = user_weights * np.log(np.maximum(sigma_i, np.sqrt(_E_SQ_FLOOR)))
        else:
            log_jac = np.log(np.maximum(sigma_i, np.sqrt(_E_SQ_FLOOR)))
        new_ll = float(np.sum(ll) - np.sum(log_jac))

        if abs(new_ll - prev_ll) < tol * (abs(prev_ll) + tol):
            converged = True
            prev_ll = new_ll
            break
        prev_ll = new_ll

    assert mean_fit is not None

    # Overlay dispformula state onto the inner CrossedLMEResult, matching
    # the BCA result surface so downstream consumers see a uniform shape.
    result = mean_fit
    result.disp_params = pd.Series(delta_hat, index=disp_term_names)
    result.disp_random_effects = {g: pd.Series(b) for g, b in disp_blups.items()}
    result.disp_variance_components = disp_varcomps
    result.dispersion = sigma_i**2
    result.disp_method = "bca"
    result.scale = 1.0
    result.nparams = (
        result.nparams + int(delta_hat.size) + sum(1 for _ in disp_varcomps.values())
    )

    return StudentTDispResult(
        lmm=result,
        nu=float(nu_current),
        sigma=float(np.exp(float(np.mean(np.log(sigma_i))))),
        n_iter=max_iter,
        converged=converged,
        marginal_loglik=float(prev_ll),
        nu_estimated=nu_estimated,
        disp_params=result.disp_params,
        disp_random_effects=result.disp_random_effects,
        disp_variance_components=result.disp_variance_components,
        dispersion=result.dispersion,
        disp_method="bca",
    )
