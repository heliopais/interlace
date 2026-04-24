"""Cox PH with Gaussian frailty (shared frailty model).

Implements penalized partial likelihood estimation for the Cox proportional
hazards model with normally-distributed random effects (frailties), matching
R's ``coxme::coxme()``.

The model is:

    h(t | x_i, b_j) = h_0(t) exp(x_i' beta + z_i' b_j)

where b_j ~ N(0, sigma^2 I) for shared frailty.

Estimation uses a two-level optimization:

1. **Inner loop** (Newton-Raphson): for fixed theta (variance parameter),
   find (beta, b) maximizing the penalized partial log-likelihood.
2. **Outer loop** (L-BFGS-B): optimize theta via the Laplace-approximated
   integrated partial likelihood.

References
----------
Therneau, T.M. (2003). Penalized Cox models and frailty.
    Technical Report #66, Mayo Foundation.
Ripatti, S. & Palmgren, J. (2000). Estimation of multivariate
    frailty models using penalized partial likelihood.
    Biometrics, 56(4), 1016-1022.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.sparse as sp

from interlace.formula import (
    groups_to_random_effects,
    parse_random_effects,
)
from interlace.profiled_reml import (
    make_lambda,
    n_theta_for_spec,
    sparse_chol_logdet,
)
from interlace.sparse_z import build_joint_z_from_specs, group_array

if TYPE_CHECKING:
    from interlace.formula import RandomEffectSpec

# ---------------------------------------------------------------------------
# Surv() formula parsing
# ---------------------------------------------------------------------------

_SURV_RE = re.compile(
    r"^\s*Surv\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*~\s*(.+)$", re.IGNORECASE
)


def parse_surv_formula(formula: str) -> tuple[str, str, str]:
    """Parse ``Surv(time, event) ~ rhs`` formula.

    Returns (time_col, event_col, rhs_formula).
    """
    m = _SURV_RE.match(formula)
    if m is None:
        raise ValueError(
            f"Expected 'Surv(time, event) ~ ...' formula, got: {formula!r}"
        )
    return m.group(1), m.group(2), m.group(3).strip()


# ---------------------------------------------------------------------------
# Breslow partial likelihood and derivatives
# ---------------------------------------------------------------------------


def _order_by_time(
    time: np.ndarray, event: np.ndarray, eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort by descending time.

    Returns (time_sorted, event_sorted, eta_sorted, order).
    """
    order = np.argsort(-time)
    return time[order], event[order], eta[order], order


def breslow_loglik(eta: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Breslow partial log-likelihood.

    Parameters
    ----------
    eta : array of shape (n,)
        Linear predictor values.
    time : array of shape (n,)
        Observed times.
    event : array of shape (n,)
        Event indicator (1 = event, 0 = censored).

    Returns
    -------
    float
        Partial log-likelihood value.
    """
    n = len(time)
    order = np.argsort(time)
    t_sorted = time[order]
    e_sorted = event[order]
    eta_sorted = eta[order]

    # For numerical stability, subtract max
    eta_max = eta_sorted.max()
    exp_eta = np.exp(eta_sorted - eta_max)

    # Cumulative risk set sum (from last to first)
    cum_exp = np.zeros(n)
    cum_exp[n - 1] = exp_eta[n - 1]
    for i in range(n - 2, -1, -1):
        cum_exp[i] = cum_exp[i + 1] + exp_eta[i]

    # Handle ties: all subjects at the same time share the same risk set sum
    # Walk backwards to propagate the risk-set sum for tied times
    for i in range(1, n):
        if t_sorted[i] == t_sorted[i - 1]:
            cum_exp[i] = cum_exp[i - 1]

    loglik = 0.0
    for i in range(n):
        if e_sorted[i]:
            loglik += (eta_sorted[i] - eta_max) - np.log(cum_exp[i])

    return float(loglik)


def breslow_score(eta: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Score (gradient of Breslow log-likelihood w.r.t. eta).

    Returns array of shape (n,) in original (unsorted) order.
    """
    n = len(time)
    order = np.argsort(time)
    inv_order = np.empty(n, dtype=int)
    inv_order[order] = np.arange(n)

    t_sorted = time[order]
    e_sorted = event[order]
    eta_sorted = eta[order]

    eta_max = eta_sorted.max()
    exp_eta = np.exp(eta_sorted - eta_max)

    # Risk set sums
    cum_exp = np.zeros(n)
    cum_exp[n - 1] = exp_eta[n - 1]
    for i in range(n - 2, -1, -1):
        cum_exp[i] = cum_exp[i + 1] + exp_eta[i]
    for i in range(1, n):
        if t_sorted[i] == t_sorted[i - 1]:
            cum_exp[i] = cum_exp[i - 1]

    # score_i = delta_i - exp(eta_i) * sum_{j<=i, d_j=1} 1/cum_exp[j]
    d_over_risk = np.where(e_sorted, 1.0 / cum_exp, 0.0)
    cum_d_over_risk = np.cumsum(d_over_risk)
    score_sorted = e_sorted.astype(np.float64) - exp_eta * cum_d_over_risk

    return np.asarray(score_sorted[inv_order])


def _breslow_hessian_diag(
    eta: np.ndarray, time: np.ndarray, event: np.ndarray
) -> np.ndarray:
    """Diagonal of the negative Hessian of the Breslow log-likelihood.

    Returns array of shape (n,) in original order. This is the "working
    weight" for each observation, analogous to IRLS weights in GLMs.

    This is an approximation (diagonal only) used for the working weights
    in the penalized Newton step.
    """
    n = len(time)
    order = np.argsort(time)
    inv_order = np.empty(n, dtype=int)
    inv_order[order] = np.arange(n)

    t_sorted = time[order]
    e_sorted = event[order]
    eta_sorted = eta[order]

    eta_max = eta_sorted.max()
    exp_eta = np.exp(eta_sorted - eta_max)

    # Risk set sums
    cum_exp = np.zeros(n)
    cum_exp[n - 1] = exp_eta[n - 1]
    for i in range(n - 2, -1, -1):
        cum_exp[i] = cum_exp[i + 1] + exp_eta[i]
    for i in range(1, n):
        if t_sorted[i] == t_sorted[i - 1]:
            cum_exp[i] = cum_exp[i - 1]

    # -d^2 l / d eta_i^2 = exp(eta_i) * sum_{j: t_j<=t_i, d_j=1} [
    #     1/cum_exp[j] - exp(eta_i)/cum_exp[j]^2
    # ]
    d_over_risk = np.where(e_sorted, 1.0 / cum_exp, 0.0)
    d_over_risk2 = np.where(e_sorted, 1.0 / cum_exp**2, 0.0)
    cum_d_over_risk = np.cumsum(d_over_risk)
    cum_d_over_risk2 = np.cumsum(d_over_risk2)

    w_sorted = exp_eta * cum_d_over_risk - exp_eta**2 * cum_d_over_risk2

    # Floor at small positive value for numerical stability
    w_sorted = np.maximum(w_sorted, 1e-10)

    return np.asarray(w_sorted[inv_order])


# ---------------------------------------------------------------------------
# Concordance index
# ---------------------------------------------------------------------------


def _concordance(eta: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's concordance index (C-statistic).

    Only considers comparable pairs (at least one event).
    """
    concordant = 0
    discordant = 0
    tied = 0
    n = len(time)
    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[j] < time[i]:
                continue
            # Pair (i, j): i had event at time[i], j survived at least until time[i]
            if eta[i] > eta[j]:
                concordant += 1
            elif eta[i] < eta[j]:
                discordant += 1
            else:
                tied += 1

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied) / total


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CoxmeResult:
    """Result container for a fitted Cox frailty model.

    Attributes
    ----------
    fe_params : pd.Series
        Fixed-effect (log hazard ratio) estimates.
    fe_bse : pd.Series
        Standard errors of the fixed effects.
    fe_pvalues : pd.Series
        Two-sided Wald p-values (z-test).
    fe_conf_int : pd.DataFrame
        95% confidence intervals for the log hazard ratios.
    random_effects : dict[str, pd.Series]
        BLUPs per grouping factor.
    variance_components : dict[str, float]
        Estimated frailty variances.
    theta : np.ndarray
        Raw variance-component parameter vector.
    converged : bool
        Whether the optimizer converged.
    nobs : int
        Number of observations.
    n_events : int
        Number of events (non-censored).
    ngroups : dict[str, int]
        Number of levels per grouping factor.
    llf : float
        Integrated partial log-likelihood at the optimum.
    aic : float
        Akaike information criterion.
    bic : float
        Bayesian information criterion.
    concordance : float
        Harrell's concordance index (C-statistic).
    baseline_hazard : pd.DataFrame
        Breslow baseline cumulative hazard estimate.
    """

    fe_params: pd.Series
    fe_bse: pd.Series
    fe_pvalues: pd.Series
    fe_conf_int: pd.DataFrame
    random_effects: dict[str, Any]
    variance_components: dict[str, float]
    theta: np.ndarray
    converged: bool
    nobs: int
    n_events: int
    ngroups: dict[str, int]
    llf: float
    aic: float
    bic: float
    concordance: float
    baseline_hazard: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    _eta_hat: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    _rhs_formula: str = field(default="", repr=False)
    _group_cols: list[str] = field(default_factory=list, repr=False)
    _time: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    _event: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    _X: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    def predict(
        self,
        newdata: Any | None = None,
        include_re: bool = True,
        type: str = "lp",
    ) -> np.ndarray:
        """Predict from the fitted Cox frailty model.

        Parameters
        ----------
        newdata :
            DataFrame with the same covariates (and group columns) as
            the training data. If ``None``, returns in-sample predictions.
        include_re :
            If ``True`` (default), add BLUP contributions for known group
            levels. If ``False``, return fixed-effects-only prediction.
        type :
            ``"lp"`` — linear predictor x'beta + z'b (default).
            ``"risk"`` — exp(linear predictor), i.e. hazard ratio.

        Returns
        -------
        np.ndarray of shape (n_obs,)
        """
        valid_types = ("lp", "risk")
        if type not in valid_types:
            raise ValueError(f"Unknown type '{type}'. Choose from: {valid_types}")

        if newdata is None:
            lp = self._eta_hat
        else:
            import formulaic

            beta = np.asarray(self.fe_params)
            if len(beta) == 0:
                lp = np.zeros(len(newdata))
            else:
                mm = formulaic.model_matrix(f"~ 0 + {self._rhs_formula}", newdata)
                X = np.asarray(mm, dtype=np.float64)
                lp = X @ beta

            if include_re:
                for grp in self._group_cols:
                    if grp not in self.random_effects:
                        continue
                    re = self.random_effects[grp]
                    col_vals = np.asarray(newdata[grp])
                    lookup = re.to_dict() if isinstance(re, pd.Series) else {}
                    contrib = np.array(
                        [lookup.get(v, 0.0) for v in col_vals], dtype=np.float64
                    )
                    lp = lp + contrib

        if type == "risk":
            return np.asarray(np.exp(lp))
        return np.asarray(lp)

    def summary(self) -> str:
        """Format a summary table similar to R's print.coxme."""
        from scipy.stats import norm

        lines: list[str] = []
        lines.append("Cox Frailty Model (coxme)")
        lines.append(f"  n={self.nobs}, events={self.n_events}")
        lines.append("")

        # Fixed effects table
        if len(self.fe_params) > 0:
            lines.append("Fixed Effects:")
            lines.append(
                f"  {'':12s} {'coef':>9s} {'exp(coef)':>9s} "
                f"{'se(coef)':>9s} {'z':>8s} {'p':>10s}"
            )
            for name in self.fe_params.index:
                coef = self.fe_params[name]
                se = self.fe_bse[name]
                z = coef / se if se > 0 else np.nan
                p = 2.0 * (1.0 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
                lines.append(
                    f"  {name:12s} {coef:9.4f} {np.exp(coef):9.4f} "
                    f"{se:9.4f} {z:8.3f} {p:10.4g}"
                )
            lines.append("")

        # Variance components
        lines.append("Random Effects (Variance):")
        for grp, var in self.variance_components.items():
            n_lev = self.ngroups.get(grp, 0)
            lines.append(f"  {grp:12s}  {var:.4f}  (sd: {np.sqrt(var):.4f}, n={n_lev})")
        lines.append("")

        # Fit statistics
        lines.append(f"Concordance: {self.concordance:.4f}")
        lines.append(f"Log-lik (IPL): {self.llf:.2f}  AIC: {self.aic:.2f}")
        lines.append(f"Converged: {self.converged}")

        return "\n".join(lines)

    def resid(self, type: str = "martingale") -> np.ndarray | pd.DataFrame:
        """Compute residuals for the fitted Cox frailty model.

        Parameters
        ----------
        type :
            ``"martingale"`` — delta_i - Lambda_0(t_i)*exp(eta_i).
            ``"deviance"`` — signed sqrt of deviance contribution.
            ``"schoenfeld"`` — covariate residuals at each event time
            (one row per event, columns = covariates). Returned as DataFrame.

        Returns
        -------
        np.ndarray of shape (n,) for martingale/deviance, or
        pd.DataFrame of shape (n_events, p) for schoenfeld.
        """
        valid = ("martingale", "deviance", "schoenfeld")
        if type not in valid:
            raise ValueError(f"Unknown type '{type}'. Choose from: {valid}")

        time = self._time
        event = self._event
        eta = self._eta_hat

        if type in ("martingale", "deviance"):
            return self._martingale_or_deviance(time, event, eta, type)
        return self._schoenfeld(time, event, eta, self._X)

    def _martingale_or_deviance(
        self,
        time: np.ndarray,
        event: np.ndarray,
        eta: np.ndarray,
        rtype: str,
    ) -> np.ndarray:
        """Martingale: M_i = delta_i - Lambda_0(t_i)*exp(eta_i)."""
        n = len(time)
        order = np.argsort(time)
        t_sorted = time[order]
        e_sorted = event[order]
        exp_eta_sorted = np.exp(eta[order])

        # Risk set sums (same pattern as breslow_loglik)
        cum_exp = np.zeros(n)
        cum_exp[n - 1] = exp_eta_sorted[n - 1]
        for i in range(n - 2, -1, -1):
            cum_exp[i] = cum_exp[i + 1] + exp_eta_sorted[i]
        for i in range(1, n):
            if t_sorted[i] == t_sorted[i - 1]:
                cum_exp[i] = cum_exp[i - 1]

        # Cumulative baseline hazard at each observation time
        # Lambda_0(t_i) = sum_{j: t_j <= t_i, d_j=1} d_j / S_0(t_j)
        d_over_risk = np.where(e_sorted, 1.0 / cum_exp, 0.0)
        cum_baseline = np.cumsum(d_over_risk)

        # Martingale: M_i = delta_i - Lambda_0(t_i)*exp(eta_i)
        mart_sorted = e_sorted.astype(np.float64) - cum_baseline * exp_eta_sorted

        # Undo sort
        inv_order = np.empty(n, dtype=int)
        inv_order[order] = np.arange(n)
        mart = mart_sorted[inv_order]

        if rtype == "martingale":
            return np.asarray(mart)

        # Deviance: sign(M_i) * sqrt(-2[M_i + delta_i*log(delta_i - M_i)])
        d = event.astype(np.float64)
        term = d * np.log(np.maximum(d - mart, 1e-20))
        dev_sq = -2.0 * (mart + term)
        dev_sq = np.maximum(dev_sq, 0.0)
        return np.asarray(np.sign(mart) * np.sqrt(dev_sq))

    def _schoenfeld(
        self,
        time: np.ndarray,
        event: np.ndarray,
        eta: np.ndarray,
        X: np.ndarray,
    ) -> pd.DataFrame:
        """Schoenfeld residuals: x_i - E[x | R(t_i)] at each event time."""
        n = len(time)
        p = X.shape[1]
        order = np.argsort(time)
        t_sorted = time[order]
        e_sorted = event[order]
        eta_sorted = eta[order]
        X_sorted = X[order]

        exp_eta = np.exp(eta_sorted - eta_sorted.max())

        # Risk set weighted sums (backward cumsum)
        cum_exp = np.zeros(n)
        cum_exp[n - 1] = exp_eta[n - 1]
        cum_Xexp = np.zeros((n, p))
        cum_Xexp[n - 1] = X_sorted[n - 1] * exp_eta[n - 1]
        for i in range(n - 2, -1, -1):
            cum_exp[i] = cum_exp[i + 1] + exp_eta[i]
            cum_Xexp[i] = cum_Xexp[i + 1] + X_sorted[i] * exp_eta[i]
        # Propagate for ties
        for i in range(1, n):
            if t_sorted[i] == t_sorted[i - 1]:
                cum_exp[i] = cum_exp[i - 1]
                cum_Xexp[i] = cum_Xexp[i - 1]

        # Collect Schoenfeld residuals at event times
        rows = []
        for i in range(n):
            if e_sorted[i]:
                x_bar = cum_Xexp[i] / cum_exp[i]
                rows.append(X_sorted[i] - x_bar)

        return pd.DataFrame(rows, columns=list(self.fe_params.index))


# ---------------------------------------------------------------------------
# Exact Breslow information
# ---------------------------------------------------------------------------


def _breslow_info_products(
    X: np.ndarray,
    Z: sp.csc_matrix,
    eta: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Breslow information products for SE computation.

    The negative Hessian of the Cox partial log-likelihood is:
        -H = Σ_{k:event} [diag(a_k) - a_k a_k']
    where a_k[j] = exp(η_j)*1(j∈R_k) / S_0(t_k).

    Using diag(w) (our inner-loop approximation) drops the rank-1 outer
    products, overestimating information by ~5-15% and underestimating SEs.

    This function computes X'(-H)X, X'(-H)Z, Z'(-H)Z exactly without
    forming the full n×n Hessian, in O(n * d * (p+q)) time where d is
    the number of events.

    Returns (XtHX, XtHZ, ZtHZ) as dense arrays.
    """
    n = len(time)
    p = X.shape[1]
    q = Z.shape[1]
    Z_d = Z.toarray()

    order = np.argsort(time)
    t_sorted = time[order]
    e_sorted = event[order]
    eta_sorted = eta[order]

    eta_max = eta_sorted.max()
    exp_eta = np.exp(eta_sorted - eta_max)

    # Risk set sums (backward cumsum)
    cum_exp = np.zeros(n)
    cum_exp[n - 1] = exp_eta[n - 1]
    for i in range(n - 2, -1, -1):
        cum_exp[i] = cum_exp[i + 1] + exp_eta[i]
    for i in range(1, n):
        if t_sorted[i] == t_sorted[i - 1]:
            cum_exp[i] = cum_exp[i - 1]

    X_s = X[order]
    Z_s = Z_d[order]

    XtHX = np.zeros((p, p))
    XtHZ = np.zeros((p, q))
    ZtHZ = np.zeros((q, q))

    # Backward cumulative weighted sums for O(n*d*(p+q)) → O(n*(p+q))
    # Pre-compute cumulative sums to avoid re-summing the risk set each time
    cum_wX = np.zeros((n + 1, p))  # cum_wX[i] = Σ_{j=i..n-1} w_j x_j
    cum_wZ = np.zeros((n + 1, q))
    cum_wXX = np.zeros((n + 1, p, p))  # cum_wXX[i] = Σ_{j=i..n-1} w_j x_j x_j'
    cum_wXZ = np.zeros((n + 1, p, q))
    cum_wZZ = np.zeros((n + 1, q, q))

    for j in range(n - 1, -1, -1):
        w_j = exp_eta[j]
        x_j = X_s[j]
        z_j = Z_s[j]
        cum_wX[j] = cum_wX[j + 1] + w_j * x_j
        cum_wZ[j] = cum_wZ[j + 1] + w_j * z_j
        cum_wXX[j] = cum_wXX[j + 1] + w_j * np.outer(x_j, x_j)
        cum_wXZ[j] = cum_wXZ[j + 1] + w_j * np.outer(x_j, z_j)
        cum_wZZ[j] = cum_wZZ[j + 1] + w_j * np.outer(z_j, z_j)

    for i in range(n):
        if not e_sorted[i]:
            continue
        S0 = cum_exp[i]

        # For ties: use the same risk set as the first of the tied group
        x_bar = cum_wX[i] / S0
        z_bar = cum_wZ[i] / S0

        XtHX += cum_wXX[i] / S0 - np.outer(x_bar, x_bar)
        XtHZ += cum_wXZ[i] / S0 - np.outer(x_bar, z_bar)
        ZtHZ += cum_wZZ[i] / S0 - np.outer(z_bar, z_bar)

    return XtHX, XtHZ, ZtHZ


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_sigma_inv(
    theta: np.ndarray,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
) -> sp.csc_matrix:
    """Build Sigma^{-1} as a sparse matrix from theta and specs."""
    blocks: list[sp.spmatrix] = []
    theta_idx = 0
    for spec, n_lev in zip(specs, n_levels, strict=True):
        n_th = n_theta_for_spec(spec.n_terms, spec.correlated)
        if spec.n_terms == 1:
            th = theta[theta_idx]
            val = 1.0 / (th**2 + 1e-20)
            blocks.append(sp.eye(n_lev, format="csc") * val)
        else:
            p_j = spec.n_terms
            th_j = theta[theta_idx : theta_idx + n_th]
            if spec.correlated:
                L_j = np.zeros((p_j, p_j))
                idx = 0
                for row in range(p_j):
                    for col in range(row + 1):
                        L_j[row, col] = th_j[idx]
                        idx += 1
            else:
                L_j = np.diag(th_j)
            Sig_inv_j = np.linalg.inv(L_j @ L_j.T + 1e-20 * np.eye(p_j))
            blocks.append(sp.kron(Sig_inv_j, sp.eye(n_lev), format="csc"))
        theta_idx += n_th
    return sp.block_diag(blocks, format="csc")


def _build_sigma_inv_diag(
    theta: np.ndarray,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    q: int,
) -> np.ndarray:
    """Build the diagonal of Sigma^{-1} (fast path for inner solve)."""
    diag = np.zeros(q)
    offset = 0
    theta_idx = 0
    for spec, n_lev in zip(specs, n_levels, strict=True):
        n_th = n_theta_for_spec(spec.n_terms, spec.correlated)
        block_size = spec.n_terms * n_lev
        if spec.n_terms == 1:
            th = theta[theta_idx]
            diag[offset : offset + block_size] = 1.0 / (th**2 + 1e-20)
        else:
            p_j = spec.n_terms
            th_j = theta[theta_idx : theta_idx + n_th]
            if spec.correlated:
                L_j = np.zeros((p_j, p_j))
                idx = 0
                for row in range(p_j):
                    for col in range(row + 1):
                        L_j[row, col] = th_j[idx]
                        idx += 1
            else:
                L_j = np.diag(th_j)
            Sig_inv_j = np.linalg.inv(L_j @ L_j.T + 1e-20 * np.eye(p_j))
            for k in range(n_lev):
                for t in range(p_j):
                    diag[offset + t * n_lev + k] = Sig_inv_j[t, t]
        theta_idx += n_th
        offset += block_size
    return diag


# ---------------------------------------------------------------------------
# Inner Newton-Raphson solver for (beta, b) given theta
# ---------------------------------------------------------------------------


def _inner_solve(
    X: np.ndarray,
    Z: sp.csc_matrix,
    time: np.ndarray,
    event: np.ndarray,
    theta: np.ndarray,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    beta0: np.ndarray | None = None,
    b0: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Newton-Raphson for penalized partial likelihood.

    Uses a block-diagonal inner step: full dense solve for beta (via
    X'WX), diagonal step for b (via diag(Z'WZ + Sigma_inv)).  This is
    stable and converges reliably for shared-frailty models.

    Returns (beta, b, penalized_loglik, working_weights).
    """
    from scipy.sparse.linalg import spsolve_triangular

    p = X.shape[1]
    q = Z.shape[1]

    beta = beta0 if beta0 is not None else np.zeros(p)
    b = b0 if b0 is not None else np.zeros(q)

    Lambda = make_lambda(theta, specs, n_levels)
    Lambda_csr = Lambda.tocsr()
    sigma_inv_diag = _build_sigma_inv_diag(theta, specs, n_levels, q)

    Z_csc = Z.tocsc()

    for _iteration in range(max_iter):
        eta = X @ beta + Z_csc @ b

        score_eta = breslow_score(eta, time, event)
        w = _breslow_hessian_diag(eta, time, event)

        # Penalty score via triangular solves
        Linv_b = spsolve_triangular(Lambda_csr, b, lower=True)
        penalty_score_b = -spsolve_triangular(Lambda.T.tocsr(), Linv_b, lower=False)

        score_beta = X.T @ score_eta
        score_b = np.asarray(Z_csc.T @ score_eta).ravel() + penalty_score_b

        W_sp = sp.diags(w)
        XtWX = X.T @ W_sp @ X
        ZtWZ_diag = np.asarray((Z_csc.T @ W_sp @ Z_csc).diagonal()).ravel()
        H_b_diag = ZtWZ_diag + sigma_inv_diag

        # Beta: full dense solve
        try:
            delta_beta = np.linalg.solve(XtWX, score_beta)
        except np.linalg.LinAlgError:
            delta_beta = np.linalg.lstsq(XtWX, score_beta, rcond=None)[0]

        # b: diagonal step
        delta_b = score_b / np.maximum(H_b_diag, 1e-10)

        # Step halving
        old_pll = breslow_loglik(eta, time, event) - 0.5 * np.dot(Linv_b, Linv_b)

        step = 1.0
        for _ in range(10):
            b_new = b + step * delta_b
            beta_new = beta + step * delta_beta
            eta_new = X @ beta_new + Z_csc @ b_new
            Linv_b_new = spsolve_triangular(Lambda_csr, b_new, lower=True)
            new_pll = breslow_loglik(eta_new, time, event) - 0.5 * np.dot(
                Linv_b_new, Linv_b_new
            )
            if new_pll >= old_pll - 1e-8:
                break
            step *= 0.5

        beta = beta + step * delta_beta
        b = b + step * delta_b

        if (
            np.max(np.abs(step * delta_beta)) < tol
            and np.max(np.abs(step * delta_b)) < tol
        ):
            break

    # Final PLL
    eta = X @ beta + Z_csc @ b
    Linv_b = spsolve_triangular(Lambda_csr, b, lower=True)
    pll = breslow_loglik(eta, time, event) - 0.5 * np.dot(Linv_b, Linv_b)
    w_final = _breslow_hessian_diag(eta, time, event)

    return beta, b, float(pll), w_final


# ---------------------------------------------------------------------------
# Outer objective: Laplace-approximated integrated partial likelihood
# ---------------------------------------------------------------------------


def _laplace_ipl(
    theta: np.ndarray,
    X: np.ndarray,
    Z: sp.csc_matrix,
    time: np.ndarray,
    event: np.ndarray,
    specs: list[RandomEffectSpec],
    n_levels: list[int],
    beta_warm: np.ndarray | None = None,
    b_warm: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Negative Laplace-approximated integrated partial log-likelihood.

    Returns (neg_ipl, beta_hat, b_hat) for warm-starting next call.
    """
    theta = np.maximum(theta, 1e-6)

    beta, b, pll, w = _inner_solve(
        X,
        Z,
        time,
        event,
        theta,
        specs,
        n_levels,
        beta0=beta_warm,
        b0=b_warm,
    )

    # Build full sparse H_bb + Sigma_inv for log-determinant
    W_sp = sp.diags(w)
    ZtWZ = (Z.T @ W_sp @ Z).tocsc()
    Sigma_inv = _build_sigma_inv(theta, specs, n_levels)
    H_plus_Sinv = (ZtWZ + Sigma_inv).tocsc()

    log_det = sparse_chol_logdet(H_plus_Sinv)
    log_det_sigma_inv = sparse_chol_logdet(Sigma_inv)

    # IPL = PLL + 0.5*log|Sigma_inv| - 0.5*log|H_bb + Sigma_inv|
    ipl = pll + 0.5 * log_det_sigma_inv - 0.5 * log_det

    return -ipl, beta, b  # Return negative for minimization


# ---------------------------------------------------------------------------
# Breslow baseline hazard
# ---------------------------------------------------------------------------


def _breslow_baseline_hazard(
    time: np.ndarray,
    event: np.ndarray,
    eta: np.ndarray,
) -> pd.DataFrame:
    """Breslow estimator of the cumulative baseline hazard."""
    order = np.argsort(time)
    t_sorted = time[order]
    e_sorted = event[order]
    exp_eta = np.exp(eta[order])

    # Risk set sums
    n = len(time)
    cum_exp = np.zeros(n)
    cum_exp[n - 1] = exp_eta[n - 1]
    for i in range(n - 2, -1, -1):
        cum_exp[i] = cum_exp[i + 1] + exp_eta[i]
    for i in range(1, n):
        if t_sorted[i] == t_sorted[i - 1]:
            cum_exp[i] = cum_exp[i - 1]

    # Unique event times
    event_mask = e_sorted.astype(bool)
    event_times = t_sorted[event_mask]
    event_risk = cum_exp[event_mask]

    unique_times, indices = np.unique(event_times, return_inverse=True)
    d_k = np.bincount(indices).astype(float)
    risk_k = np.array([event_risk[indices == k][0] for k in range(len(unique_times))])

    cum_haz = np.cumsum(d_k / risk_k)

    return pd.DataFrame(
        {"time": unique_times, "cumhaz": cum_haz, "n_events": d_k.astype(int)}
    )


# ---------------------------------------------------------------------------
# Public fit function
# ---------------------------------------------------------------------------


def fit_coxme(
    formula: str,
    data: Any,
    groups: str | list[str] | None = None,
    random: list[str] | None = None,
    optimizer: str = "lbfgsb",
    theta0: np.ndarray | None = None,
) -> CoxmeResult:
    """Fit a Cox PH model with Gaussian frailty.

    Parameters
    ----------
    formula :
        Survival formula ``"Surv(time, event) ~ x1 + x2"``.
    data :
        DataFrame (pandas, polars, or narwhals-compatible).
    groups :
        Column name(s) for shared frailty grouping.
    random :
        lme4-style random effect specs (takes precedence over ``groups``).
    optimizer :
        ``"lbfgsb"`` (default).
    theta0 :
        Initial theta. Defaults to ones.

    Returns
    -------
    CoxmeResult
    """
    import formulaic
    import narwhals as nw

    nw_data = nw.from_native(data, eager_only=True)

    # --- Parse Surv() formula ---
    time_col, event_col, rhs = parse_surv_formula(formula)
    time_arr = nw_data[time_col].to_numpy().astype(np.float64)
    event_arr = nw_data[event_col].to_numpy().astype(np.int32)

    # --- Build random effect specs ---
    if random is not None:
        specs = parse_random_effects(random)
    elif groups is not None:
        specs = groups_to_random_effects(groups)
    else:
        raise ValueError("Either 'groups' or 'random' must be provided.")

    # --- Build fixed-effects design matrix (RHS only, no intercept for Cox) ---
    # Cox PH has no intercept (absorbed into baseline hazard).
    pd_data = pd.DataFrame({col: nw_data[col].to_numpy() for col in nw_data.columns})

    if rhs.strip() == "1":
        # No covariates (frailty-only model)
        X = np.empty((len(time_arr), 0))
        term_names: list[str] = []
    else:
        mm = formulaic.model_matrix(f"~ 0 + {rhs}", pd_data)
        X = np.asarray(mm, dtype=np.float64)
        term_names = list(mm.columns)

    p = X.shape[1]
    n = len(time_arr)

    # --- Build sparse Z ---
    Z = build_joint_z_from_specs(specs, data)
    n_levels_list: list[int] = [
        int(np.unique(group_array(spec, nw_data)).shape[0]) for spec in specs
    ]
    # --- Initial theta ---
    n_theta_total = sum(n_theta_for_spec(s.n_terms, s.correlated) for s in specs)
    theta_init = np.ones(n_theta_total) if theta0 is None else theta0.copy()

    # --- Optimize theta ---
    beta_warm: np.ndarray | None = None
    b_warm: np.ndarray | None = None

    def objective(th: np.ndarray) -> float:
        nonlocal beta_warm, b_warm
        neg_ipl, beta_warm, b_warm = _laplace_ipl(
            th,
            X,
            Z,
            time_arr,
            event_arr,
            specs,
            n_levels_list,
            beta_warm=beta_warm,
            b_warm=b_warm,
        )
        return neg_ipl

    if n_theta_total == 1:
        # 1D: Brent's method is more robust than L-BFGS-B
        def obj_scalar(th_val: float) -> float:
            return objective(np.array([th_val]))

        res = opt.minimize_scalar(
            obj_scalar,
            bounds=(1e-4, 5.0),
            method="bounded",
            options={"xatol": 1e-6, "maxiter": 200},
        )
        theta_opt = np.array([res.x])
        converged = res.success if hasattr(res, "success") else True
    else:
        # Multi-dimensional: L-BFGS-B
        bounds = [(1e-4, None)] * n_theta_total
        res = opt.minimize(
            objective,
            theta_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        theta_opt = res.x
        converged = res.success

    # --- Final solve at optimum ---
    beta_hat, b_hat, pll, w_final = _inner_solve(
        X,
        Z,
        time_arr,
        event_arr,
        theta_opt,
        specs,
        n_levels_list,
        beta0=beta_warm,
        b0=b_warm,
    )

    # --- Standard errors from exact observed information ---
    # Use exact Breslow Hessian products (not diagonal approximation)
    # to get SEs matching R's coxme.
    eta_hat = X @ beta_hat + Z @ b_hat if p > 0 else (Z @ b_hat)
    Sigma_inv = _build_sigma_inv(theta_opt, specs, n_levels_list)

    if p > 0:
        XtHX, XtHZ, ZtHZ = _breslow_info_products(X, Z, eta_hat, time_arr, event_arr)
        H_bb = ZtHZ + Sigma_inv.toarray()
        H_bb_inv = np.linalg.inv(H_bb)
        schur = XtHX - XtHZ @ H_bb_inv @ XtHZ.T
        try:
            schur_inv = np.linalg.inv(schur)
            fe_bse_arr = np.sqrt(np.diag(schur_inv))
        except np.linalg.LinAlgError:
            fe_bse_arr = np.full(p, np.nan)
    else:
        fe_bse_arr = np.array([])

    # --- Build result objects ---
    fe_params = pd.Series(beta_hat, index=term_names)
    fe_bse = pd.Series(fe_bse_arr, index=term_names)

    # Wald z-test p-values
    if p > 0:
        from scipy.stats import norm

        z_scores = beta_hat / fe_bse_arr
        fe_pvalues_arr = 2.0 * (1.0 - norm.cdf(np.abs(z_scores)))
    else:
        fe_pvalues_arr = np.array([])
    fe_pvalues = pd.Series(fe_pvalues_arr, index=term_names)

    # Confidence intervals
    fe_conf_int = pd.DataFrame(
        {
            "lower": beta_hat - 1.96 * fe_bse_arr,
            "upper": beta_hat + 1.96 * fe_bse_arr,
        },
        index=term_names,
    )

    # --- Package random effects ---
    random_effects: dict[str, Any] = {}
    variance_components: dict[str, float] = {}
    ngroups: dict[str, int] = {}

    theta_idx = 0
    blup_offset = 0
    for spec, q_j in zip(specs, n_levels_list, strict=True):
        n_th = n_theta_for_spec(spec.n_terms, spec.correlated)
        n_blups_j = spec.n_terms * q_j
        blup_block = b_hat[blup_offset : blup_offset + n_blups_j]
        uniques = sorted(np.unique(group_array(spec, nw_data)).tolist())

        if spec.n_terms == 1:
            random_effects[spec.group] = pd.Series(
                blup_block, index=uniques, name=spec.group
            )
            th = theta_opt[theta_idx]
            variance_components[spec.group] = float(th**2)
        else:
            # Multi-term (future extension)
            term_names_j = (["(Intercept)"] if spec.intercept else []) + list(
                spec.predictors
            )
            re_mat = blup_block.reshape(spec.n_terms, q_j).T
            random_effects[spec.group] = pd.DataFrame(
                re_mat, index=uniques, columns=term_names_j
            )
            th_j = theta_opt[theta_idx : theta_idx + n_th]
            if spec.correlated:
                p_j = spec.n_terms
                L_j = np.zeros((p_j, p_j))
                idx = 0
                for row in range(p_j):
                    for col in range(row + 1):
                        L_j[row, col] = th_j[idx]
                        idx += 1
                variance_components[spec.group] = float(np.trace(L_j @ L_j.T) / p_j)
            else:
                variance_components[spec.group] = float(np.mean(th_j**2))

        ngroups[spec.group] = q_j
        theta_idx += n_th
        blup_offset += n_blups_j

    # --- Information criteria ---
    # Number of parameters: p fixed + n_theta variance
    k = p + n_theta_total
    llf = -res.fun  # IPL at optimum
    aic = -2 * llf + 2 * k
    bic = -2 * llf + k * np.log(n)

    # --- Concordance ---
    conc = _concordance(eta_hat, time_arr, event_arr)

    # --- Baseline hazard ---
    bh = _breslow_baseline_hazard(time_arr, event_arr, eta_hat)

    return CoxmeResult(
        fe_params=fe_params,
        fe_bse=fe_bse,
        fe_pvalues=fe_pvalues,
        fe_conf_int=fe_conf_int,
        random_effects=random_effects,
        variance_components=variance_components,
        theta=theta_opt,
        converged=converged,
        nobs=n,
        n_events=int(event_arr.sum()),
        ngroups=ngroups,
        llf=llf,
        aic=aic,
        bic=bic,
        concordance=conc,
        baseline_hazard=bh,
        _eta_hat=eta_hat,
        _rhs_formula=rhs,
        _group_cols=[spec.group for spec in specs],
        _time=time_arr,
        _event=event_arr,
        _X=X,
    )
