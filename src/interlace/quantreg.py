"""Quantile regression utilities: kernel SE matching R's quantreg::summary.rq(se="ker").

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

import numpy as np
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

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.regression.quantile_regression as sqr
    >>> from interlace.quantreg import quantreg_ker_se
    >>> rng = np.random.default_rng(0)
    >>> n = 500; male = np.concatenate([np.ones(n//2), np.zeros(n//2)])
    >>> y = 10_000 + 3_000 * male + rng.normal(0, 2_000, n)
    >>> X = np.column_stack([np.ones(n), male])
    >>> fit = sqr.QuantReg(y, X).fit(q=0.5, disp=False)
    >>> se = quantreg_ker_se(fit.resid, X, tau=0.5)
    >>> se.shape
    (2,)
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
