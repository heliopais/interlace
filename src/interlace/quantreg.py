"""Quantile regression: LP-based coefficient fitting + kernel SE.

Coefficient fitting
-------------------
``quantreg(formula, data, tau)`` solves the standard quantile regression LP::

    min_{beta, u, v}  tau * 1'u + (1-tau) * 1'v
    s.t.  X beta + u - v = y,  u >= 0,  v >= 0

using ``scipy.optimize.linprog`` with the HiGHS solver.

Kernel SE
---------
R reference:
    Koenker, R. (2005). Quantile Regression. Cambridge University Press. §3.4.
    quantreg::bandwidth.rq() and quantreg::summary.rq(se="ker")

The Hall-Sheather bandwidth (hs=True):
    h = n^(-1/3) * z_{α/2}^(2/3) * [(1.5 * φ(Φ⁻¹(τ))²) / (2*Φ⁻¹(τ)² + 1)]^(1/3)

Sparsity estimate from residuals:
    f̂ = 2h / (Q(τ+h; r) − Q(τ−h; r))

Covariance:
    Cov(β̂) = τ(1−τ) / f̂² * (X'X)⁻¹
"""

from __future__ import annotations

from typing import Any

import formulaic
import narwhals as nw
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats


def _hall_sheather_bandwidth(n: int, tau: float, alpha: float = 0.05) -> float:
    """Hall-Sheather bandwidth for kernel SE estimation.

    Replicates R's ``bandwidth.rq(tau, n, hs=TRUE, alpha=alpha)``.

    Parameters
    ----------
    n : sample size
    tau : quantile level (0 < tau < 1)
    alpha : significance level for the normal critical value (default 0.05)
    """
    x = stats.norm.ppf(tau)
    f = stats.norm.pdf(x)
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return float(
        n ** (-1.0 / 3.0)
        * z ** (2.0 / 3.0)
        * ((1.5 * f**2) / (2.0 * x**2 + 1.0)) ** (1.0 / 3.0)
    )


def _bofinger_bandwidth(n: int, tau: float) -> float:
    """Bofinger bandwidth for kernel SE estimation.

    Replicates R's ``bandwidth.rq(tau, n, hs=FALSE)``.

    Parameters
    ----------
    n : sample size
    tau : quantile level (0 < tau < 1)
    """
    x = stats.norm.ppf(tau)
    f = stats.norm.pdf(x)
    return float(((4.5 * f**4) / (2.0 * x**2 + 1.0) ** 2) ** 0.2 * n ** (-0.2))


def quantreg_ker_se(
    residuals: np.ndarray,
    X: np.ndarray,
    tau: float = 0.5,
    hs: bool = True,
) -> np.ndarray:
    """Quantile regression kernel SE matching R's ``summary.rq(se="ker")``.

    Exact port of R quantreg's Hendricks-Koenker sandwich kernel estimator:
    uses a Gaussian kernel density evaluated at each residual, with bandwidth
    derived from the Hall-Sheather (or Bofinger) formula scaled to data units.

    Parameters
    ----------
    residuals : array-like, shape (n,)
        QR residuals ``y − X @ beta_hat``.
    X : array-like, shape (n, p)
        Design matrix (including intercept column if present).
    tau : quantile level (default 0.5)
    hs : use Hall-Sheather bandwidth (True, default) or Bofinger (False).

    Returns
    -------
    se : ndarray, shape (p,)
        Standard errors for each coefficient, matching R's kernel SE.

    Raises
    ------
    ValueError
        If the bandwidth is too large for the given sample size and tau.
    """
    residuals = np.asarray(residuals, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(residuals)

    h = _hall_sheather_bandwidth(n, tau) if hs else _bofinger_bandwidth(n, tau)

    if tau + h > 1.0 or tau - h < 0.0:
        raise ValueError(
            f"bandwidth h={h:.4f} is too large for tau={tau} and n={n}; "
            "decrease tau distance from boundaries or increase sample size."
        )

    bhi = float(np.quantile(residuals, tau + h))
    blo = float(np.quantile(residuals, tau - h))

    if bhi == blo:
        raise ValueError(
            "Residual quantiles Q(tau+h) and Q(tau-h) are equal; "
            "sparsity is undefined. Try a larger sample or different tau."
        )

    f_hat = 2.0 * h / (bhi - blo)

    XtX_inv = np.linalg.inv(X.T @ X)
    cov = (tau * (1.0 - tau) / f_hat**2) * XtX_inv
    return np.sqrt(np.diag(cov))


# ---------------------------------------------------------------------------
# LP-based quantile regression fitting
# ---------------------------------------------------------------------------


class QuantRegResult:
    """Result of a quantile regression fit via LP solve.

    Attributes
    ----------
    params : pd.Series
        Named coefficients; supports ``.get(name)`` and ``.values``.
    resid : np.ndarray, shape (n,)
        ``y - X @ beta``.
    fittedvalues : np.ndarray, shape (n,)
        ``X @ beta``.
    tau : float
        Quantile level used for fitting.
    """

    def __init__(
        self,
        params: pd.Series,
        resid: np.ndarray,
        fittedvalues: np.ndarray,
        tau: float,
        X: np.ndarray,
        rhs_model_spec: Any,
    ) -> None:
        self.params = params
        self.resid = resid
        self.fittedvalues = fittedvalues
        self.tau = tau
        # Store design matrix for ker_se; prefixed to avoid name clash with params
        self._X = X
        self._rhs_model_spec = rhs_model_spec

    def ker_se(self, hs: bool = True) -> np.ndarray:
        """Kernel standard errors for the coefficients.

        Delegates directly to :func:`quantreg_ker_se`.

        Parameters
        ----------
        hs : use Hall-Sheather bandwidth (True, default) or Bofinger (False).

        Returns
        -------
        np.ndarray, shape (p,)
        """
        return quantreg_ker_se(self.resid, self._X, tau=self.tau, hs=hs)

    def predict(self, data: Any) -> np.ndarray:
        """Predict on new data by re-evaluating the RHS formula.

        Parameters
        ----------
        data:
            DataFrame containing predictor columns. Any narwhals-compatible
            frame is accepted.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        nw_data = nw.from_native(data, eager_only=True)
        X_new = np.asarray(
            formulaic.model_matrix(self._rhs_model_spec, nw_data), dtype=float
        )
        return np.asarray(X_new @ self.params.values, dtype=float)


def quantreg(formula: str, data: Any, tau: float = 0.5) -> QuantRegResult:
    """Fit a quantile regression model using a formulaic formula string.

    Solves the standard LP::

        min  tau * 1'u + (1-tau) * 1'v
        s.t. X beta + u - v = y,  u >= 0, v >= 0

    using ``scipy.optimize.linprog`` with the HiGHS interior-point solver.

    Parameters
    ----------
    formula:
        Formula string, e.g. ``"y ~ x1 + x2"``.
    data:
        DataFrame containing all variables. Any narwhals-compatible frame
        (pandas, polars, …) is accepted.
    tau:
        Quantile level in (0, 1). Default 0.5 (median regression).

    Returns
    -------
    QuantRegResult
    """
    nw_data = nw.from_native(data, eager_only=True)

    matrices = formulaic.model_matrix(formula, nw_data)
    X = np.asarray(matrices.rhs, dtype=float)
    y = np.asarray(matrices.lhs, dtype=float).squeeze()
    term_names: list[str] = list(matrices.rhs.columns)
    rhs_model_spec = matrices.rhs.model_spec

    n, p = X.shape

    # LP formulation: variables = [beta (p), u (n), v (n)]
    # objective: min 0*beta + tau*u + (1-tau)*v
    c = np.concatenate([np.zeros(p), np.full(n, tau), np.full(n, 1.0 - tau)])

    # Equality constraints: X @ beta + u - v = y
    # A_eq @ x = b_eq  where x = [beta, u, v]
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    b_eq = y

    # bounds: beta unbounded, u >= 0, v >= 0
    bounds = (
        [(None, None)] * p  # beta: unconstrained
        + [(0.0, None)] * n  # u: non-negative
        + [(0.0, None)] * n  # v: non-negative
    )

    result = opt.linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs-ipm",
    )

    beta = result.x[:p]
    fittedvalues = X @ beta
    resid = y - fittedvalues

    params = pd.Series(beta, index=term_names)

    return QuantRegResult(
        params=params,
        resid=resid,
        fittedvalues=fittedvalues,
        tau=tau,
        X=X,
        rhs_model_spec=rhs_model_spec,
    )
