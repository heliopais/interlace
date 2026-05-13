"""Student-t × dispformula via joint Laplace (interlace-1t0v).

Outer EM over the latent-scale Student-t representation

    tau_i ~ Gamma(nu/2, nu/2)
    y_i | tau_i ~ N(mu_i, sigma_i^2 / tau_i)

interleaved with the **Gaussian joint-Laplace** dispformula solver as
the M-step. Each outer iteration calls the existing
:func:`interlace.dispformula_joint.fit_dispformula_joint` (RE-on-disp)
or :func:`interlace.dispformula_bca.fit_dispformula_joint_laplace`
(FE-only) with prior weights ``user_w * tau``. This delegates the
heavy lifting — joint optimisation over (theta_m, theta_d, delta) under
proper marginalisation — to a path that does not exhibit the BCA
under-estimation of disp variance components (see the standing
``dispformula-bca-vs-joint-laplace`` memory).

``nu`` is updated each outer iteration via an ECM 1-D bounded search on
the marginal Student-t profile log-likelihood applied to the
standardised residuals ``e_i / sigma_i``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from interlace.student_t import _log_t_density, _profile_nu
from interlace.student_t_dispformula import StudentTDispResult

_OUTER_TOL = 1e-4
_OUTER_MAXITER = 15
_E_SQ_FLOOR = 1e-12


def fit_student_t_dispformula_joint_laplace(
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
    """Fit a Student-t LMM with dispformula via outer EM + Gaussian
    joint Laplace M-step."""
    if nu is not None and nu <= 2.0:
        raise ValueError(
            f"nu must be > 2 (variance is undefined for nu <= 2); got {nu}."
        )
    if nu_min <= 2.0:
        raise ValueError(f"nu_min must be > 2; got {nu_min}.")

    from interlace.dispformula_bca import (
        _parse_dispformula,
        _to_pandas,
        fit_dispformula_joint_laplace,
    )
    from interlace.dispformula_joint import fit_dispformula_joint

    _, disp_re_specs = _parse_dispformula(dispformula)
    has_disp_re = len(disp_re_specs) > 0

    df_pd = _to_pandas(data)
    n = df_pd.shape[0]

    user_weights = None if weights is None else np.asarray(weights, dtype=np.float64)
    base_weights = (
        np.ones(n) if user_weights is None else user_weights.astype(np.float64)
    )

    nu_estimated = nu is None
    nu_current = float(nu_init) if nu_estimated else float(nu)  # type: ignore[arg-type]

    def _inner(eff_w: np.ndarray) -> Any:
        if has_disp_re:
            return fit_dispformula_joint(
                formula=formula,
                data=df_pd,
                dispformula=dispformula,
                groups=groups,
                random=random,
                method=method,
                weights=eff_w,
                df_method=df_method,
            )
        return fit_dispformula_joint_laplace(
            formula=formula,
            data=df_pd,
            dispformula=dispformula,
            groups=groups,
            random=random,
            method=method,
            weights=eff_w,
            df_method=df_method,
        )

    # --- Initial fit: tau = 1 (Gaussian) ---
    result = _inner(base_weights)
    e = np.asarray(result.resid, dtype=np.float64)
    disp = np.asarray(result.dispersion, dtype=np.float64)
    sigma_i = np.sqrt(np.maximum(disp, _E_SQ_FLOOR))

    if nu_estimated:
        r = e / sigma_i
        nu_current = _profile_nu(r, 1.0, user_weights, nu_min, nu_max)

    prev_ll = -np.inf
    converged = False

    for _ in range(max_iter):
        # --- E-step ---
        r2 = (e * e) / np.maximum(sigma_i * sigma_i, _E_SQ_FLOOR)
        tau = (nu_current + 1.0) / (nu_current + r2)

        # --- M-step: Gaussian joint Laplace with weights = base_w * tau ---
        eff_w = base_weights * tau
        result = _inner(eff_w)
        e = np.asarray(result.resid, dtype=np.float64)
        disp = np.asarray(result.dispersion, dtype=np.float64)
        sigma_i = np.sqrt(np.maximum(disp, _E_SQ_FLOOR))

        # --- nu CM-step ---
        r = e / sigma_i
        if nu_estimated:
            nu_current = _profile_nu(r, 1.0, user_weights, nu_min, nu_max)

        # --- Convergence: marginal Student-t profile log-lik on
        # standardised residuals, minus log-Jacobian sum(log sigma_i). ---
        ll = _log_t_density(r, 1.0, nu_current)
        if user_weights is not None:
            ll = ll * user_weights
            log_jac = user_weights * np.log(sigma_i)
        else:
            log_jac = np.log(sigma_i)
        new_ll = float(np.sum(ll) - np.sum(log_jac))

        if abs(new_ll - prev_ll) < tol * (abs(prev_ll) + tol):
            converged = True
            prev_ll = new_ll
            break
        prev_ll = new_ll

    # Overlay Student-t-specific metadata on the inner result.
    result.disp_method = "joint_laplace"

    return StudentTDispResult(
        lmm=result,
        nu=float(nu_current),
        sigma=float(np.exp(float(np.mean(np.log(sigma_i))))),
        n_iter=max_iter,
        converged=converged and bool(result.converged),
        marginal_loglik=float(prev_ll),
        nu_estimated=nu_estimated,
        disp_params=result.disp_params
        if isinstance(result.disp_params, pd.Series)
        else pd.Series(result.disp_params),
        disp_random_effects=dict(result.disp_random_effects)
        if result.disp_random_effects is not None
        else {},
        disp_variance_components=dict(result.disp_variance_components)
        if result.disp_variance_components is not None
        else {},
        dispersion=np.asarray(result.dispersion, dtype=np.float64),
        disp_method="joint_laplace",
    )
