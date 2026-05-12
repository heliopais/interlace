"""Student-t (robust) response family for linear mixed models.

EM with latent scales (Lange-Little-Taylor 1989, Pinheiro-Bates 2001):
    tau_i ~ Gamma(nu/2, nu/2)
    y_i | tau_i ~ N(x_i' beta + z_i' b, sigma^2 / tau_i)
Marginally y_i ~ t(x_i' beta + z_i' b, sigma^2, nu).

Each EM iteration is a weighted LMM (closed-form M-step), reusing the
existing profiled REML machinery via ``interlace.fit(..., weights=...)``.
``nu`` is either fixed by the caller or estimated via a 1-D Brent search
on the profile log-likelihood interleaved with the M-step (ECM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize, special

from interlace.result import CrossedLMEResult


@dataclass
class StudentTResult:
    """Result of a Student-t LMM fit.

    Wraps the final weighted-LMM ``CrossedLMEResult`` and exposes the
    Student-t-specific scale (``sigma``) and degrees-of-freedom (``nu``).
    """

    lmm: CrossedLMEResult
    nu: float
    sigma: float
    n_iter: int
    converged: bool
    marginal_loglik: float
    nu_estimated: bool

    @property
    def fe_params(self) -> Any:
        return self.lmm.fe_params

    @property
    def variance_components(self) -> dict[str, Any]:
        return self.lmm.variance_components

    @property
    def resid(self) -> Any:
        return self.lmm.resid

    @property
    def fittedvalues(self) -> Any:
        return self.lmm.fittedvalues

    @property
    def random_effects(self) -> Any:
        return self.lmm.random_effects

    @property
    def model(self) -> Any:
        return self.lmm.model

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self.lmm.predict(*args, **kwargs)


def _log_t_density(resid: np.ndarray, sigma2: float, nu: float) -> np.ndarray:
    """Log-density of location-scale Student-t at residual values."""
    ll: np.ndarray = (
        special.gammaln((nu + 1.0) / 2.0)
        - special.gammaln(nu / 2.0)
        - 0.5 * np.log(nu * np.pi * sigma2)
        - 0.5 * (nu + 1.0) * np.log1p(resid * resid / (nu * sigma2))
    )
    return ll


def _profile_nu(
    resid: np.ndarray,
    sigma2: float,
    weights: np.ndarray | None,
    nu_min: float,
    nu_max: float,
) -> float:
    """1-D maximisation of profile log-lik in nu given conditional residuals."""

    def neg_ll(nu: float) -> float:
        ll = _log_t_density(resid, sigma2, nu)
        if weights is not None:
            ll = ll * weights
        return -float(np.sum(ll))

    res = optimize.minimize_scalar(
        neg_ll,
        bounds=(nu_min, nu_max),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(res.x)


def student_t_fit(
    formula: str,
    data: Any,
    groups: str | list[str] | None = None,
    random: list[str] | None = None,
    nu: float | None = None,
    weights: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    nu_init: float = 4.0,
    nu_min: float = 2.001,
    nu_max: float = 200.0,
    method: str = "REML",
    df_method: str = "satterthwaite",
) -> StudentTResult:
    """Fit a linear mixed model with Student-t residuals via EM.

    Parameters
    ----------
    formula, data, groups, random, method, df_method
        See :func:`interlace.fit`.
    nu
        Degrees of freedom of the Student-t residuals. ``None`` (default) →
        estimate via interleaved 1-D Brent search on the profile log-lik
        (ECM). Pass a positive scalar ``> 2`` to fix it.
    weights
        Optional observation-level prior weights (multiplicative on the
        log-likelihood, as in :func:`interlace.fit`).
    max_iter, tol
        EM stopping criteria. Convergence is declared when the relative
        change in marginal log-lik is below ``tol``.
    nu_init, nu_min, nu_max
        Initial value and bounds for ``nu`` when estimable. ``nu_min`` must
        exceed 2 (variance is undefined otherwise).
    """
    if nu is not None and nu <= 2.0:
        raise ValueError(
            f"nu must be > 2 (variance is undefined for nu <= 2); got {nu}."
        )
    if nu_min <= 2.0:
        raise ValueError(f"nu_min must be > 2; got {nu_min}.")

    # Local import to avoid circulars at module load.
    from interlace import fit as lmm_fit

    nu_estimated = nu is None
    nu_current = float(nu_init) if nu_estimated else float(nu)  # type: ignore[arg-type]

    user_weights = None if weights is None else np.asarray(weights, dtype=np.float64)

    # --- Initialisation: unweighted Gaussian LMM ---
    res = lmm_fit(
        formula=formula,
        data=data,
        groups=groups,
        random=random,
        method=method,
        weights=user_weights,
        df_method=df_method,
    )
    sigma2 = float(res.scale)
    resid = np.asarray(res.resid, dtype=np.float64)

    if nu_estimated:
        nu_current = _profile_nu(resid, sigma2, user_weights, nu_min, nu_max)

    prev_ll = float(np.sum(_log_t_density(resid, sigma2, nu_current)))
    if user_weights is not None:
        prev_ll = float(
            np.sum(_log_t_density(resid, sigma2, nu_current) * user_weights)
        )

    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        # E-step: posterior expectations of latent precisions.
        tau = (nu_current + 1.0) / (nu_current + resid * resid / sigma2)

        # M-step: weighted LMM with effective weights = user_weights * tau.
        eff_w = tau if user_weights is None else user_weights * tau
        res = lmm_fit(
            formula=formula,
            data=data,
            groups=groups,
            random=random,
            method=method,
            weights=eff_w,
            df_method=df_method,
        )
        sigma2 = float(res.scale)
        resid = np.asarray(res.resid, dtype=np.float64)

        # CM-step for nu (if estimable).
        if nu_estimated:
            nu_current = _profile_nu(resid, sigma2, user_weights, nu_min, nu_max)

        # Monitor marginal log-lik (conditional on BLUPs — standard EM
        # observed-data approximation).
        ll = _log_t_density(resid, sigma2, nu_current)
        if user_weights is not None:
            ll = ll * user_weights
        new_ll = float(np.sum(ll))

        if abs(new_ll - prev_ll) < tol * (abs(prev_ll) + tol):
            converged = True
            prev_ll = new_ll
            break
        prev_ll = new_ll

    return StudentTResult(
        lmm=res,
        nu=float(nu_current),
        sigma=float(np.sqrt(sigma2)),
        n_iter=n_iter,
        converged=converged,
        marginal_loglik=float(prev_ll),
        nu_estimated=nu_estimated,
    )
